#!/usr/bin/env python3
"""
Controlled ablation benchmark for fraud probability features.

Input:
    analysis/probability_training_dataset.csv

Variants:
    1. baseline
       Current benchmark feature set.
    2. cleaned
       Remove all-zero features and exact duplicate out_of_region_behavior.
    3. inverted_risk
       Cleaned + replace risk_score with (1 - risk_score).
    4. no_risk_score
       Cleaned + remove risk_score.
    5. minimal_evidence
       Cleaned + remove risk_score and overlapping weak_support_count/product_novelty
       redundancy, retaining the stronger evidence/count signals.

Models:
    - logistic_regression
    - random_forest
    - hist_gradient_boosting

Evaluation:
    Chronological 60/20/20 split.
    Validation-only sigmoid calibration.
    Test set remains untouched until final evaluation.

Outputs:
    analysis/probability_ablation_metrics.csv
    analysis/probability_ablation_predictions.csv
    analysis/probability_ablation_summary.json
    analysis/probability_ablation_notes.txt

This is a focused experiment designed to minimize iteration time. It does not
modify the training dataset or claim that any variant is production-ready.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT = Path("analysis/probability_training_dataset.csv")
OUTDIR = Path("analysis")

TARGET = "label"
DATE_COL = "opened_at"

BASE_FEATURES = [
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

# Cleaned according to the audit:
# - strong_contradiction_count: all zero
# - weak_contradiction_count: all zero
# - out_of_region_behavior: exact duplicate of out_of_region
CLEANED_REMOVE = {
    "strong_contradiction_count",
    "weak_contradiction_count",
    "out_of_region_behavior",
}

VARIANTS = {
    "baseline": {
        "features": BASE_FEATURES,
        "risk_transform": "none",
    },
    "cleaned": {
        "features": [f for f in BASE_FEATURES if f not in CLEANED_REMOVE],
        "risk_transform": "none",
    },
    "inverted_risk": {
        "features": [f for f in BASE_FEATURES if f not in CLEANED_REMOVE],
        "risk_transform": "invert",
    },
    "no_risk_score": {
        "features": [
            f for f in BASE_FEATURES
            if f not in CLEANED_REMOVE and f != "risk_score"
        ],
        "risk_transform": "none",
    },
    "minimal_evidence": {
        "features": [
            f for f in BASE_FEATURES
            if f not in CLEANED_REMOVE
            and f not in {"risk_score", "weak_support_count", "product_novelty"}
        ],
        "risk_transform": "none",
    },
}


def log(msg: str) -> None:
    print(msg, flush=True)


def ece_10(y_true: np.ndarray, p: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    total = len(y_true)
    if total == 0:
        return float("nan")

    value = 0.0
    for i in range(10):
        if i == 9:
            mask = (p >= bins[i]) & (p <= bins[i + 1])
        else:
            mask = (p >= bins[i]) & (p < bins[i + 1])

        n = int(mask.sum())
        if n == 0:
            continue

        value += (n / total) * abs(
            float(y_true[mask].mean()) - float(p[mask].mean())
        )

    return float(value)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10": ece_10(y, p),
        "mean_probability": float(np.mean(p)),
        "median_probability": float(np.median(p)),
        "min_probability": float(np.min(p)),
        "max_probability": float(np.max(p)),
    }


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(work)
    a = int(n * 0.60)
    b = int(n * 0.80)
    return work.iloc[:a].copy(), work.iloc[a:b].copy(), work.iloc[b:].copy()


def make_models() -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def get_raw_probability(model, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def fit_sigmoid_calibrator(
    validation_raw: np.ndarray,
    y_validation: np.ndarray,
) -> LogisticRegression:
    # Platt calibration: logistic regression on the model logit.
    eps = 1e-7
    clipped = np.clip(validation_raw, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    calibrator.fit(logits, y_validation)
    return calibrator


def apply_sigmoid_calibrator(
    calibrator: LogisticRegression,
    raw: np.ndarray,
) -> np.ndarray:
    eps = 1e-7
    clipped = np.clip(raw, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def prepare_features(
    df: pd.DataFrame,
    feature_names: List[str],
    risk_transform: str,
) -> pd.DataFrame:
    x = df[feature_names].copy()

    for feature in feature_names:
        x[feature] = pd.to_numeric(x[feature], errors="coerce")

    if risk_transform == "invert":
        x["risk_score"] = 1.0 - x["risk_score"]

    return x


def threshold_summary(y: np.ndarray, p: np.ndarray) -> Dict[str, Dict[str, float]]:
    out = {}
    for threshold in (0.3, 0.5, 0.7, 0.9):
        pred = p >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        out[str(threshold)] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "predicted_fraud_rate": float(pred.mean()),
        }
    return out


def main() -> None:
    start = time.time()

    log("=" * 72)
    log("FRAUD PROBABILITY ABLATION BENCHMARK")
    log("=" * 72)
    log(f"Input: {INPUT}")

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

    missing = sorted(set(BASE_FEATURES + [TARGET, DATE_COL]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train, validation, test = chronological_split(df)

    log(f"Rows: {len(df):,}")
    log(
        f"Split: train={len(train):,}, validation={len(validation):,}, "
        f"test={len(test):,}"
    )
    log(
        f"Fraud rates: train={train[TARGET].mean():.4f}, "
        f"validation={validation[TARGET].mean():.4f}, "
        f"test={test[TARGET].mean():.4f}"
    )
    log("")

    all_metrics = []
    all_predictions = []
    summary = {
        "dataset": {
            "rows": int(len(df)),
            "fraud": int(df[TARGET].sum()),
            "cleared": int((1 - df[TARGET]).sum()),
            "fraud_rate": float(df[TARGET].mean()),
            "date_min": str(df[DATE_COL].min()),
            "date_max": str(df[DATE_COL].max()),
        },
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "variants": {},
        "methodology": {
            "calibration": "sigmoid/Platt fitted only on chronological validation",
            "test_used_for_model_selection": False,
            "thresholds": [0.3, 0.5, 0.7, 0.9],
        },
    }

    for variant_name, config in VARIANTS.items():
        log(f"[VARIANT] {variant_name}")
        features = config["features"]
        transform = config["risk_transform"]

        summary["variants"][variant_name] = {
            "features": features,
            "feature_count": len(features),
            "risk_transform": transform,
            "models": {},
        }

        x_train = prepare_features(train, features, transform)
        x_val = prepare_features(validation, features, transform)
        x_test = prepare_features(test, features, transform)

        y_train = train[TARGET].to_numpy()
        y_val = validation[TARGET].to_numpy()
        y_test = test[TARGET].to_numpy()

        models = make_models()

        for model_name, model in models.items():
            model_start = time.time()
            log(f"  Training {model_name}...")

            model.fit(x_train, y_train)

            raw_val = get_raw_probability(model, x_val)
            raw_test = get_raw_probability(model, x_test)

            calibrator = fit_sigmoid_calibrator(raw_val, y_val)
            cal_val = apply_sigmoid_calibrator(calibrator, raw_val)
            cal_test = apply_sigmoid_calibrator(calibrator, raw_test)

            raw_metrics = calibration_metrics(y_test, raw_test)
            calibrated_metrics = calibration_metrics(y_test, cal_test)

            for calibration_type, probs, metrics in [
                ("raw", raw_test, raw_metrics),
                ("sigmoid_calibrated", cal_test, calibrated_metrics),
            ]:
                row = {
                    "variant": variant_name,
                    "model": model_name,
                    "calibration": calibration_type,
                    "feature_count": len(features),
                    **metrics,
                }
                all_metrics.append(row)

                for i, probability in enumerate(probs):
                    all_predictions.append(
                        {
                            "variant": variant_name,
                            "model": model_name,
                            "calibration": calibration_type,
                            "row_index_in_test": i,
                            "target": int(y_test[i]),
                            "probability": float(probability),
                        }
                    )

            summary["variants"][variant_name]["models"][model_name] = {
                "raw_test": raw_metrics,
                "sigmoid_calibrated_test": calibrated_metrics,
                "calibrated_thresholds": threshold_summary(y_test, cal_test),
                "runtime_seconds": time.time() - model_start,
            }

            log(
                f"    calibrated ROC-AUC={calibrated_metrics['roc_auc']:.4f} "
                f"PR-AUC={calibrated_metrics['pr_auc']:.4f} "
                f"Brier={calibrated_metrics['brier']:.4f} "
                f"LogLoss={calibrated_metrics['log_loss']:.4f} "
                f"ECE={calibrated_metrics['ece_10']:.4f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    # Compact ranking by each metric, but no single "winner" is selected.
    calibrated = metrics_df[metrics_df["calibration"] == "sigmoid_calibrated"].copy()

    metric_directions = {
        "roc_auc": "max",
        "pr_auc": "max",
        "brier": "min",
        "log_loss": "min",
        "ece_10": "min",
    }

    summary["metric_extremes"] = {}
    for metric, direction in metric_directions.items():
        if direction == "max":
            idx = calibrated[metric].idxmax()
        else:
            idx = calibrated[metric].idxmin()

        row = calibrated.loc[idx]
        summary["metric_extremes"][metric] = {
            "direction": direction,
            "variant": row["variant"],
            "model": row["model"],
            "value": float(row[metric]),
        }

    # Risk-only diagnostic: actual score and inverted score.
    risk = pd.to_numeric(df["risk_score"], errors="coerce")
    y = df[TARGET].to_numpy()
    summary["risk_score_diagnostic"] = {
        "raw_auc": float(roc_auc_score(y, risk)),
        "inverted_auc": float(roc_auc_score(y, 1.0 - risk)),
        "mean_fraud": float(risk[y == 1].mean()),
        "mean_cleared": float(risk[y == 0].mean()),
    }

    metrics_path = OUTDIR / "probability_ablation_metrics.csv"
    predictions_path = OUTDIR / "probability_ablation_predictions.csv"
    summary_path = OUTDIR / "probability_ablation_summary.json"
    notes_path = OUTDIR / "probability_ablation_notes.txt"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    notes = []
    notes.append("FRAUD PROBABILITY ABLATION BENCHMARK")
    notes.append("=" * 72)
    notes.append("")
    notes.append("Purpose: controlled feature ablation and risk_score direction test.")
    notes.append("This experiment does not select a production model.")
    notes.append("")
    notes.append("VARIANTS")
    notes.append("-" * 72)
    for name, cfg in VARIANTS.items():
        notes.append(
            f"{name}: {len(cfg['features'])} features; "
            f"risk_transform={cfg['risk_transform']}"
        )
    notes.append("")
    notes.append("CALIBRATION")
    notes.append("- Sigmoid/Platt calibration fitted only on chronological validation.")
    notes.append("- Test set is used only for final evaluation.")
    notes.append("")
    notes.append("METRIC EXTREMES")
    notes.append("-" * 72)
    for metric, info in summary["metric_extremes"].items():
        notes.append(
            f"{metric} ({info['direction']}): "
            f"{info['variant']} / {info['model']} = {info['value']:.6f}"
        )
    notes.append("")
    notes.append("INTERPRETATION")
    notes.append("-" * 72)
    notes.append(
        "Compare metrics across variants rather than relying on one metric alone."
    )
    notes.append(
        "The inverted risk_score experiment is a diagnostic test of direction, "
        "not permission to redefine the upstream score without verifying its semantics."
    )
    notes.append(
        "The high historical fraud prevalence remains a population/calibration limitation."
    )
    notes.append(
        "The benchmark is case-level: label=1 means historical case outcome was confirmed fraud."
    )
    notes.append("")

    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    elapsed = time.time() - start
    log("")
    log("ABLATION BENCHMARK COMPLETE")
    log(f"Metrics: {metrics_path}")
    log(f"Predictions: {predictions_path}")
    log(f"Summary: {summary_path}")
    log(f"Notes: {notes_path}")
    log(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
