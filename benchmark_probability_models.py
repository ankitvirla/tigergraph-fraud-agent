#!/usr/bin/env python3
"""
benchmark_probability_models.py

Benchmark candidate statistical fraud-probability models using the
leakage-safe historical dataset produced by build_probability_training_dataset.py.

Design rules:
- Uses ONLY the 5,565 labeled historical cases.
- Does NOT use the 20 HHG benchmark cases.
- Uses a chronological train/validation/test split.
- Keeps risk_score separate from fraud_probability conceptually:
  risk_score is benchmarked as a baseline/input, never treated as the
  final calibrated probability.
- Uses the deterministic evidence features already validated in the
  historical training dataset.
- Does NOT use case pattern, analyst notes, actions_taken, report_filed,
  exposure_usd, or other outcome-derived fields.
- Produces raw model scores plus validation-fitted calibration results.
- The final test set remains untouched during model fitting/calibration.

Inputs:
    analysis/probability_training_dataset.csv

Outputs:
    analysis/probability_model_benchmark_metrics.csv
    analysis/probability_model_benchmark_predictions.csv
    analysis/probability_model_benchmark_summary.json
    analysis/probability_model_benchmark_notes.txt

Run:
    python benchmark_probability_models.py

Optional:
    python benchmark_probability_models.py --analysis-dir analysis
    python benchmark_probability_models.py --train-fraction 0.60 --validation-fraction 0.20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_ANALYSIS_DIR = "analysis"
INPUT_FILENAME = "probability_training_dataset.csv"

METRICS_FILENAME = "probability_model_benchmark_metrics.csv"
PREDICTIONS_FILENAME = "probability_model_benchmark_predictions.csv"
SUMMARY_FILENAME = "probability_model_benchmark_summary.json"
NOTES_FILENAME = "probability_model_benchmark_notes.txt"

RANDOM_STATE = 42

# These are the exact model features established by the validated
# probability-training dataset.
MODEL_FEATURES = [
    "risk_score",
    "strong_support_count",
    "medium_support_count",
    "weak_support_count",
    "strong_contradiction_count",
    "medium_contradiction_count",
    "weak_contradiction_count",
    "context_evidence_count",
    "independent_evidence_count",
    "card_testing",
    "cnp",
    "cnp_new_device",
    "account_takeover",
    "out_of_region",
    "new_device",
    "proxy_network",
    "channel_novelty",
    "product_novelty",
    "out_of_region_behavior",
    "repeated_historical_abuse",
    "shared_device_profile",
]

TARGET_COLUMN = "label"
CASE_ID_COLUMN = "case_id"
TIME_COLUMN = "opened_at"

FORBIDDEN_FEATURES = {
    "outcome",
    "pattern",
    "analyst_notes",
    "actions_taken",
    "report_filed",
    "exposure_usd",
    "closed_at",
    "first_fraud_txn_id",
    "txn_ids",
    "n_txns",
    "target_txn_id",
    "target_selection",
}

EPS = 1e-15


# ============================================================================
# LOGGING / HELPERS
# ============================================================================

def log(message: str) -> None:
    from time import strftime
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return default


def clipped_probability(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, EPS, 1.0 - EPS)


# ============================================================================
# DATA VALIDATION
# ============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    log(f"Loading probability training dataset: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {path}\n"
            "Run build_probability_training_dataset.py first."
        )

    df = pd.read_csv(path, low_memory=False)

    required = {
        CASE_ID_COLUMN,
        TARGET_COLUMN,
        TIME_COLUMN,
        *MODEL_FEATURES,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[CASE_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate case_id values found.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Null target labels found.")

    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    if labels.isna().any():
        raise ValueError("Non-numeric target labels found.")

    labels = labels.astype(int)
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected target labels: {sorted(labels.unique())}")

    df[TARGET_COLUMN] = labels
    df[TIME_COLUMN] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")

    if df[TIME_COLUMN].isna().any():
        raise ValueError(
            f"{int(df[TIME_COLUMN].isna().sum())} rows have invalid opened_at."
        )

    # Explicitly reject leakage-prone columns if they are ever accidentally
    # added to MODEL_FEATURES.
    leakage = set(MODEL_FEATURES) & FORBIDDEN_FEATURES
    if leakage:
        raise ValueError(
            f"Forbidden target/outcome-derived features configured: {sorted(leakage)}"
        )

    for feature in MODEL_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    missing_values = df[MODEL_FEATURES].isna().sum()
    bad_missing = missing_values[missing_values > 0]
    if not bad_missing.empty:
        raise ValueError(
            "Model features contain missing values:\n"
            + bad_missing.to_string()
        )

    df = df.sort_values([TIME_COLUMN, CASE_ID_COLUMN], kind="mergesort").reset_index(
        drop=True
    )

    log(f"Rows: {len(df):,}")
    log(f"Class distribution: {df[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    log(
        f"Date range: {df[TIME_COLUMN].min().isoformat()} -> "
        f"{df[TIME_COLUMN].max().isoformat()}"
    )

    return df


# ============================================================================
# CHRONOLOGICAL SPLIT
# ============================================================================

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.50 <= train_fraction < 0.90):
        raise ValueError("train_fraction must be between 0.50 and 0.90.")

    if not (0.10 <= validation_fraction < 0.40):
        raise ValueError("validation_fraction must be between 0.10 and 0.40.")

    if train_fraction + validation_fraction >= 0.95:
        raise ValueError("train_fraction + validation_fraction must be < 0.95.")

    n = len(df)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One chronological split is empty.")

    return train, validation, test


def split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> Dict[str, Any]:
    output = {}

    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        output[name] = {
            "rows": int(len(frame)),
            "fraud": int((frame[TARGET_COLUMN] == 1).sum()),
            "cleared": int((frame[TARGET_COLUMN] == 0).sum()),
            "fraud_rate": float(frame[TARGET_COLUMN].mean()),
            "min_opened_at": frame[TIME_COLUMN].min().isoformat(),
            "max_opened_at": frame[TIME_COLUMN].max().isoformat(),
        }

    return output


# ============================================================================
# BASELINES / MODELS
# ============================================================================

def build_models() -> Dict[str, Any]:
    return {
        # Simple, interpretable statistical model.
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        # Nonlinear interaction model.
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=10,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        # Gradient-boosted nonlinear model. No class weighting is used so its
        # raw probability scale is not deliberately shifted by reweighting.
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def fit_predict(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_eval: pd.DataFrame,
) -> np.ndarray:
    fitted = clone(model)
    fitted.fit(x_train, y_train)

    if not hasattr(fitted, "predict_proba"):
        raise TypeError("Model does not expose predict_proba().")

    return clipped_probability(fitted.predict_proba(x_eval)[:, 1])


# ============================================================================
# METRICS
# ============================================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = clipped_probability(probabilities)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (probabilities >= edges[i]) & (probabilities <= edges[i + 1])
        else:
            mask = (probabilities >= edges[i]) & (probabilities < edges[i + 1])

        if not mask.any():
            continue

        confidence = probabilities[mask].mean()
        observed_rate = y_true[mask].mean()
        ece += (mask.sum() / total) * abs(confidence - observed_rate)

    return float(ece)


def metric_row(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    probabilities = clipped_probability(probabilities)
    y_true = np.asarray(y_true, dtype=int)

    # Average precision is PR-AUC in sklearn terminology.
    pr_auc = average_precision_score(y_true, probabilities)

    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc_auc = float("nan")

    return {
        "model": model_name,
        "split": split_name,
        "rows": int(len(y_true)),
        "fraud": int(y_true.sum()),
        "cleared": int((y_true == 0).sum()),
        "fraud_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "ece_10bin": expected_calibration_error(y_true, probabilities, bins=10),
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(np.median(probabilities)),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
    }


def threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: List[float],
) -> List[Dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = clipped_probability(probabilities)

    rows = []

    for threshold in thresholds:
        predicted = probabilities >= threshold

        tp = int(((predicted == 1) & (y_true == 1)).sum())
        fp = int(((predicted == 1) & (y_true == 0)).sum())
        tn = int(((predicted == 0) & (y_true == 0)).sum())
        fn = int(((predicted == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        rows.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "predicted_fraud_rate": float(predicted.mean()),
            }
        )

    return rows


# ============================================================================
# CALIBRATION
# ============================================================================

def sigmoid_calibration(
    validation_raw: np.ndarray,
    y_validation: np.ndarray,
    test_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Platt-style calibration implemented with logistic regression on the
    validation raw scores.

    The calibration model sees ONLY validation predictions and validation
    labels. Test labels are never used to fit it.
    """
    calibrator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)

    x_val = np.asarray(validation_raw, dtype=float).reshape(-1, 1)
    x_test = np.asarray(test_raw, dtype=float).reshape(-1, 1)

    calibrator.fit(x_val, y_validation)

    calibrated_validation = clipped_probability(
        calibrator.predict_proba(x_val)[:, 1]
    )
    calibrated_test = clipped_probability(
        calibrator.predict_proba(x_test)[:, 1]
    )

    coef = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])

    return calibrated_validation, calibrated_test, {
        "coef": coef,
        "intercept": intercept,
    }


# ============================================================================
# FEATURE IMPORTANCE / COEFFICIENTS
# ============================================================================

def logistic_coefficients(
    fitted_pipeline: Pipeline,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    model = fitted_pipeline.named_steps["model"]
    coefficients = model.coef_[0]

    rows = []
    for feature, coefficient in zip(feature_names, coefficients):
        rows.append(
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "odds_ratio_per_unit": float(np.exp(np.clip(coefficient, -50, 50))),
            }
        )

    return sorted(rows, key=lambda x: abs(x["coefficient"]), reverse=True)


def tree_importances(
    fitted_model: Any,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    if not hasattr(fitted_model, "feature_importances_"):
        return []

    return sorted(
        [
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(
                feature_names,
                fitted_model.feature_importances_,
            )
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark fraud probability models on historical cases."
    )
    parser.add_argument(
        "--analysis-dir",
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory containing probability_training_dataset.csv.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
        help="Chronological training fraction. Default: 0.60",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.20,
        help="Chronological validation fraction. Default: 0.20",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    input_path = analysis_dir / INPUT_FILENAME

    metrics_path = analysis_dir / METRICS_FILENAME
    predictions_path = analysis_dir / PREDICTIONS_FILENAME
    summary_path = analysis_dir / SUMMARY_FILENAME
    notes_path = analysis_dir / NOTES_FILENAME

    log("=" * 72)
    log("FRAUD PROBABILITY MODEL BENCHMARK")
    log("=" * 72)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df = load_dataset(input_path)

        train, validation, test = chronological_split(
            df,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )

        log("")
        log("CHRONOLOGICAL SPLIT")
        log("-" * 72)

        splits = split_summary(train, validation, test)
        for split_name, info in splits.items():
            log(
                f"{split_name}: rows={info['rows']:,} "
                f"fraud={info['fraud']:,} "
                f"cleared={info['cleared']:,} "
                f"fraud_rate={info['fraud_rate']:.4f} "
                f"range={info['min_opened_at']} -> {info['max_opened_at']}"
            )

        # Hard temporal assertions.
        if train[TIME_COLUMN].max() >= validation[TIME_COLUMN].min():
            raise RuntimeError("Training/validation temporal overlap detected.")

        if validation[TIME_COLUMN].max() >= test[TIME_COLUMN].min():
            raise RuntimeError("Validation/test temporal overlap detected.")

        x_train = train[MODEL_FEATURES].astype(float)
        y_train = train[TARGET_COLUMN].astype(int)

        x_validation = validation[MODEL_FEATURES].astype(float)
        y_validation = validation[TARGET_COLUMN].astype(int)

        x_test = test[MODEL_FEATURES].astype(float)
        y_test = test[TARGET_COLUMN].astype(int)

        models = build_models()

        all_metric_rows: List[Dict[str, Any]] = []
        prediction_frames: List[pd.DataFrame] = []
        diagnostics: Dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Risk-score baseline
        # ------------------------------------------------------------------
        # This is deliberately labelled as a baseline, not a calibrated
        # fraud probability. It tells us how much information is already
        # present in the upstream risk score.
        log("")
        log("Evaluating risk_score baseline...")

        risk_validation = clipped_probability(validation["risk_score"].to_numpy())
        risk_test = clipped_probability(test["risk_score"].to_numpy())

        all_metric_rows.append(
            metric_row(
                "risk_score_baseline",
                "validation",
                y_validation.to_numpy(),
                risk_validation,
            )
        )
        all_metric_rows.append(
            metric_row(
                "risk_score_baseline",
                "test",
                y_test.to_numpy(),
                risk_test,
            )
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "case_id": validation[CASE_ID_COLUMN].astype(str),
                    "opened_at": validation[TIME_COLUMN].astype(str),
                    "split": "validation",
                    "model": "risk_score_baseline",
                    "raw_probability": risk_validation,
                    "calibrated_probability": np.nan,
                    "label": y_validation.to_numpy(),
                }
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "case_id": test[CASE_ID_COLUMN].astype(str),
                    "opened_at": test[TIME_COLUMN].astype(str),
                    "split": "test",
                    "model": "risk_score_baseline",
                    "raw_probability": risk_test,
                    "calibrated_probability": np.nan,
                    "label": y_test.to_numpy(),
                }
            )
        )

        # ------------------------------------------------------------------
        # Candidate statistical models
        # ------------------------------------------------------------------
        for model_name, model in models.items():
            log("")
            log(f"Training {model_name}...")

            fitted = clone(model)
            fitted.fit(x_train, y_train)

            validation_raw = clipped_probability(
                fitted.predict_proba(x_validation)[:, 1]
            )
            test_raw = clipped_probability(
                fitted.predict_proba(x_test)[:, 1]
            )

            all_metric_rows.append(
                metric_row(
                    model_name,
                    "validation",
                    y_validation.to_numpy(),
                    validation_raw,
                )
            )
            all_metric_rows.append(
                metric_row(
                    model_name,
                    "test",
                    y_test.to_numpy(),
                    test_raw,
                )
            )

            # Fit calibration ONLY on validation.
            calibrated_validation, calibrated_test, calibration_params = (
                sigmoid_calibration(
                    validation_raw,
                    y_validation.to_numpy(),
                    test_raw,
                )
            )

            calibrated_name = f"{model_name}_sigmoid_calibrated"

            all_metric_rows.append(
                metric_row(
                    calibrated_name,
                    "validation",
                    y_validation.to_numpy(),
                    calibrated_validation,
                )
            )
            all_metric_rows.append(
                metric_row(
                    calibrated_name,
                    "test",
                    y_test.to_numpy(),
                    calibrated_test,
                )
            )

            diagnostics[model_name] = {
                "calibration": calibration_params,
            }

            if model_name == "logistic_regression":
                diagnostics[model_name]["coefficients"] = logistic_coefficients(
                    fitted,
                    MODEL_FEATURES,
                )
            elif model_name in {"random_forest", "hist_gradient_boosting"}:
                diagnostics[model_name]["feature_importances"] = tree_importances(
                    fitted,
                    MODEL_FEATURES,
                )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "case_id": validation[CASE_ID_COLUMN].astype(str),
                        "opened_at": validation[TIME_COLUMN].astype(str),
                        "split": "validation",
                        "model": model_name,
                        "raw_probability": validation_raw,
                        "calibrated_probability": calibrated_validation,
                        "label": y_validation.to_numpy(),
                    }
                )
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "case_id": test[CASE_ID_COLUMN].astype(str),
                        "opened_at": test[TIME_COLUMN].astype(str),
                        "split": "test",
                        "model": model_name,
                        "raw_probability": test_raw,
                        "calibrated_probability": calibrated_test,
                        "label": y_test.to_numpy(),
                    }
                )
            )

            # Calibrated model predictions are stored as a separate model row
            # so downstream analysis can compare raw vs calibrated behavior.
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "case_id": validation[CASE_ID_COLUMN].astype(str),
                        "opened_at": validation[TIME_COLUMN].astype(str),
                        "split": "validation",
                        "model": calibrated_name,
                        "raw_probability": validation_raw,
                        "calibrated_probability": calibrated_validation,
                        "label": y_validation.to_numpy(),
                    }
                )
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "case_id": test[CASE_ID_COLUMN].astype(str),
                        "opened_at": test[TIME_COLUMN].astype(str),
                        "split": "test",
                        "model": calibrated_name,
                        "raw_probability": test_raw,
                        "calibrated_probability": calibrated_test,
                        "label": y_test.to_numpy(),
                    }
                )
            )

        metrics_df = pd.DataFrame(all_metric_rows)

        # ------------------------------------------------------------------
        # Threshold diagnostics on calibrated test probabilities
        # ------------------------------------------------------------------
        threshold_rows: List[Dict[str, Any]] = []
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        test_prediction_df = pd.concat(prediction_frames, ignore_index=True)
        test_prediction_df = test_prediction_df[
            test_prediction_df["split"] == "test"
        ].copy()

        calibrated_models = sorted(
            name
            for name in test_prediction_df["model"].unique()
            if name.endswith("_sigmoid_calibrated")
        )

        for model_name in calibrated_models:
            subset = test_prediction_df[
                test_prediction_df["model"] == model_name
            ]

            for row in threshold_metrics(
                subset["label"].to_numpy(),
                subset["calibrated_probability"].to_numpy(),
                thresholds,
            ):
                row["model"] = model_name
                row["split"] = "test"
                threshold_rows.append(row)

        thresholds_df = pd.DataFrame(threshold_rows)

        # ------------------------------------------------------------------
        # Compact benchmark table
        # ------------------------------------------------------------------
        benchmark_table = metrics_df[
            metrics_df["split"] == "test"
        ].copy()

        benchmark_table = benchmark_table.sort_values(
            ["pr_auc", "brier_score"],
            ascending=[False, True],
            kind="mergesort",
        )

        # Do not call this a winner/ranking. It is simply the raw metric table.
        metrics_df.to_csv(metrics_path, index=False)

        prediction_output = pd.concat(
            prediction_frames,
            ignore_index=True,
        )

        prediction_output.to_csv(
            predictions_path,
            index=False,
        )

        summary = {
            "purpose": (
                "Benchmark candidate statistical fraud-probability models "
                "using the validated historical training dataset."
            ),
            "dataset": {
                "input": str(input_path),
                "rows": int(len(df)),
                "fraud": int((df[TARGET_COLUMN] == 1).sum()),
                "cleared": int((df[TARGET_COLUMN] == 0).sum()),
                "date_min": df[TIME_COLUMN].min().isoformat(),
                "date_max": df[TIME_COLUMN].max().isoformat(),
            },
            "features": MODEL_FEATURES,
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
            "split": splits,
            "models": [
                "risk_score_baseline",
                "logistic_regression",
                "logistic_regression_sigmoid_calibrated",
                "random_forest",
                "random_forest_sigmoid_calibrated",
                "hist_gradient_boosting",
                "hist_gradient_boosting_sigmoid_calibrated",
            ],
            "metrics_definition": {
                "roc_auc": "ROC AUC; higher means better ranking separation.",
                "pr_auc": "Average precision / PR AUC; higher means better precision-recall performance.",
                "brier_score": "Mean squared probability error; lower is better.",
                "log_loss": "Probabilistic log loss; lower is better.",
                "ece_10bin": "10-bin expected calibration error; lower indicates closer empirical calibration.",
            },
            "test_metrics": benchmark_table.to_dict(orient="records"),
            "threshold_metrics_test": threshold_rows,
            "diagnostics": diagnostics,
            "calibration_method": (
                "Sigmoid/Platt-style logistic calibration fitted only on the "
                "chronological validation split. Test labels are not used for "
                "calibration."
            ),
            "important_distinction": (
                "risk_score is an upstream input/baseline and is not assumed "
                "to be a calibrated fraud probability."
            ),
            "benchmark_scope": (
                "HHG-001 through HHG-020 are excluded. They remain unseen "
                "evaluation cases without authoritative current-case labels."
            ),
        }

        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                summary,
                handle,
                indent=2,
                ensure_ascii=False,
                default=json_default,
            )

        # ------------------------------------------------------------------
        # Human-readable notes
        # ------------------------------------------------------------------
        lines = [
            "=" * 72,
            "FRAUD PROBABILITY MODEL BENCHMARK",
            "=" * 72,
            "",
            "DATASET",
            "-" * 72,
            f"Rows: {len(df):,}",
            f"Confirmed fraud: {(df[TARGET_COLUMN] == 1).sum():,}",
            f"Cleared: {(df[TARGET_COLUMN] == 0).sum():,}",
            f"Date range: {df[TIME_COLUMN].min()} -> {df[TIME_COLUMN].max()}",
            "",
            "TEMPORAL SPLIT",
            "-" * 72,
        ]

        for name, info in splits.items():
            lines.append(
                f"{name}: {info['rows']:,} rows | "
                f"fraud={info['fraud']:,} | "
                f"cleared={info['cleared']:,} | "
                f"fraud_rate={info['fraud_rate']:.4f} | "
                f"{info['min_opened_at']} -> {info['max_opened_at']}"
            )

        lines.extend(
            [
                "",
                "MODEL FEATURES",
                "-" * 72,
                *[f"- {feature}" for feature in MODEL_FEATURES],
                "",
                "TEST METRICS",
                "-" * 72,
            ]
        )

        display_columns = [
            "model",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
            "ece_10bin",
        ]

        for _, row in benchmark_table[display_columns].iterrows():
            lines.append(
                f"{row['model']}: "
                f"ROC-AUC={row['roc_auc']:.4f} | "
                f"PR-AUC={row['pr_auc']:.4f} | "
                f"Brier={row['brier_score']:.4f} | "
                f"LogLoss={row['log_loss']:.4f} | "
                f"ECE={row['ece_10bin']:.4f}"
            )

        lines.extend(
            [
                "",
                "INTERPRETATION RULES",
                "-" * 72,
                "1. No metric is treated as a final model-selection verdict by this script.",
                "2. PR-AUC and ROC-AUC describe discrimination.",
                "3. Brier, log loss, and ECE describe probability quality/calibration.",
                "4. The sigmoid calibration layer is fitted on validation only.",
                "5. The chronological test set is not used for fitting or calibration.",
                "6. risk_score remains distinct from fraud_probability.",
                "7. HHG-001 through HHG-020 are not used for fitting.",
                "",
                "NEXT STEP",
                "-" * 72,
                "Use this benchmark output to decide which model families warrant",
                "deeper calibration and error analysis. Do not copy a test metric",
                "directly into a production probability formula.",
                "",
                f"Metrics: {metrics_path}",
                f"Predictions: {predictions_path}",
                f"Summary: {summary_path}",
                f"Notes: {notes_path}",
            ]
        )

        notes_path.write_text("\n".join(lines), encoding="utf-8")

    log("")
    log("=" * 72)
    log("BENCHMARK COMPLETE")
    log("=" * 72)
    log(f"Metrics:      {metrics_path}")
    log(f"Predictions:  {predictions_path}")
    log(f"Summary:      {summary_path}")
    log(f"Notes:        {notes_path}")
    log("")
    log("IMPORTANT: This is a model benchmark, not a production probability policy.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
