#!/usr/bin/env python3
"""
Build and evaluate the final fraud-probability model.

Design:
- Uses the cleaned feature set selected after the probability-model ablation.
- Keeps risk_score as an upstream input feature; it is NOT redefined here.
- Uses a chronological 60/20/20 train/validation/test split.
- Fits the HistGradientBoostingClassifier on train only.
- Fits sigmoid/Platt calibration on validation only.
- Evaluates once on the untouched test set.
- Produces probability-band calibration diagnostics for the final model.

Input:
    analysis/probability_training_dataset.csv

Outputs:
    analysis/final_probability_model_metrics.csv
    analysis/final_probability_model_predictions.csv
    analysis/final_probability_model_summary.json
    analysis/final_probability_model_notes.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


INPUT_PATH = Path("analysis/probability_training_dataset.csv")
OUTPUT_DIR = Path("analysis")

# Cleaned feature set selected after the ablation study.
FEATURES = [
    "risk_score",
    "strong_support_count",
    "medium_support_count",
    "weak_support_count",
    "medium_contradiction_count",
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
    "repeated_historical_abuse",
    "shared_device_profile",
]

TARGET = "target"
TARGET_ALIASES = ("target", "is_fraud", "label", "fraud_label")

# Chronological split: 60% train, 20% validation, 20% test.
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20

RANDOM_STATE = 42


def safe_float(value):
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error using equal-width probability bins."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (probabilities >= edges[i]) & (probabilities <= edges[i + 1])
        else:
            mask = (probabilities >= edges[i]) & (probabilities < edges[i + 1])

        count = int(mask.sum())
        if count == 0:
            continue

        observed = float(y_true[mask].mean())
        predicted = float(probabilities[mask].mean())
        ece += (count / len(y_true)) * abs(observed - predicted)

    return float(ece)


def probability_bands(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict]:
    """
    Produce production-facing probability-band diagnostics.

    Bands are deliberately descriptive:
    predicted_mean = average model probability in the band
    observed_fraud_rate = empirical fraud rate in the held-out test set
    calibration_error = absolute difference between the two
    """
    edges = np.array(
        [0.0, 0.10, 0.20, 0.30, 0.40, 0.50,
         0.60, 0.70, 0.80, 0.90, 1.0000001]
    )

    rows = []

    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high)
        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "band": f"{low:.2f}-{min(high, 1.0):.2f}",
                    "count": 0,
                    "predicted_probability_mean": None,
                    "observed_fraud_rate": None,
                    "calibration_error": None,
                }
            )
            continue

        predicted_mean = float(probabilities[mask].mean())
        observed_rate = float(y_true[mask].mean())

        rows.append(
            {
                "band": f"{low:.2f}-{min(high, 1.0):.2f}",
                "count": count,
                "predicted_probability_mean": predicted_mean,
                "observed_fraud_rate": observed_rate,
                "calibration_error": abs(predicted_mean - observed_rate),
            }
        )

    return rows


def chronological_split(df: pd.DataFrame):
    if "opened_at" not in df.columns:
        raise ValueError("Required chronological column 'opened_at' is missing.")

    df = df.copy()
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")

    if df["opened_at"].isna().any():
        raise ValueError("Some opened_at values could not be parsed.")

    df = df.sort_values("opened_at").reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    return train, validation, test


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run build_probability_training_dataset.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    # The training-dataset builder used the historical case outcome as the
    # label. Resolve the actual label column instead of assuming a literal
    # `target` column name.
    target_candidates = [column for column in TARGET_ALIASES if column in df.columns]

    if not target_candidates:
        # Safe fallback: derive the binary target from the documented historical
        # outcome field. `outcome` is used only to construct y and is never used
        # as a model feature.
        if "outcome" in df.columns:
            normalized_outcome = (
                df["outcome"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
            valid_outcomes = {"confirmed_fraud", "cleared"}
            unexpected = sorted(set(normalized_outcome.dropna()) - valid_outcomes)
            if unexpected:
                raise ValueError(
                    "Could not derive target from outcome. Unexpected outcome values: "
                    f"{unexpected}"
                )
            df[TARGET] = (normalized_outcome == "confirmed_fraud").astype(int)
            target_candidates = [TARGET]
        else:
            raise ValueError(
                "Could not find a target/label column. Expected one of "
                f"{TARGET_ALIASES}, or an `outcome` column."
            )

    actual_target_column = target_candidates[0]
    if actual_target_column != TARGET:
        df[TARGET] = df[actual_target_column]

    required = FEATURES + [TARGET, "opened_at"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Force model inputs to numeric and fail loudly on malformed values.
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        bad_columns = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Missing/non-numeric values found in model features: {bad_columns}"
        )

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    if df[TARGET].isna().any():
        raise ValueError("Missing/non-numeric target values found.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0/1.")

    # Replace the original feature columns with the validated numeric versions.
    df[FEATURES] = X

    train, validation, test = chronological_split(df)

    X_train = train[FEATURES].to_numpy(dtype=float)
    y_train = train[TARGET].to_numpy(dtype=int)

    X_val = validation[FEATURES].to_numpy(dtype=float)
    y_val = validation[TARGET].to_numpy(dtype=int)

    X_test = test[FEATURES].to_numpy(dtype=float)
    y_test = test[TARGET].to_numpy(dtype=int)

    print("=" * 72)
    print("FINAL FRAUD PROBABILITY MODEL")
    print("=" * 72)
    print(f"Input rows: {len(df):,}")
    print(f"Features: {len(FEATURES)}")
    print(f"Train: {len(train):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Test: {len(test):,}")
    print(
        "Fraud rates: "
        f"{y_train.mean():.4f} / "
        f"{y_val.mean():.4f} / "
        f"{y_test.mean():.4f}"
    )

    # Base model. No test data is used during fitting or calibration.
    base_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    base_model.fit(X_train, y_train)

    # Platt/sigmoid calibration is fitted ONLY on the chronological
    # validation set. We implement it explicitly because newer scikit-learn
    # versions removed the old `cv="prefit"` API.
    #
    # Step 1: obtain the already-fitted HGB probabilities on validation.
    validation_raw_probabilities = base_model.predict_proba(X_val)[:, 1]

    # Step 2: convert probabilities to logits. Clipping avoids +/- infinity.
    eps = 1e-7
    validation_clipped = np.clip(
        validation_raw_probabilities,
        eps,
        1.0 - eps,
    )
    validation_logits = np.log(
        validation_clipped / (1.0 - validation_clipped)
    ).reshape(-1, 1)

    # Step 3: fit the Platt sigmoid on validation only.
    platt_model = LogisticRegression(
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    platt_model.fit(validation_logits, y_val)

    # Step 4: apply the frozen HGB + frozen Platt calibration to test.
    test_raw_probabilities = base_model.predict_proba(X_test)[:, 1]
    test_clipped = np.clip(
        test_raw_probabilities,
        eps,
        1.0 - eps,
    )
    test_logits = np.log(
        test_clipped / (1.0 - test_clipped)
    ).reshape(-1, 1)

    probabilities = platt_model.predict_proba(test_logits)[:, 1]

    metrics = {
        "model": "hist_gradient_boosting_sigmoid_calibrated",
        "n_features": len(FEATURES),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_fraud_rate": float(y_train.mean()),
        "validation_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "roc_auc": safe_float(roc_auc_score(y_test, probabilities)),
        "pr_auc": safe_float(average_precision_score(y_test, probabilities)),
        "brier_score": safe_float(brier_score_loss(y_test, probabilities)),
        "log_loss": safe_float(log_loss(y_test, probabilities, labels=[0, 1])),
        "ece_10_bin": safe_float(
            expected_calibration_error(y_test, probabilities, n_bins=10)
        ),
        "mean_probability": float(probabilities.mean()),
        "median_probability": float(np.median(probabilities)),
        "min_probability": float(probabilities.min()),
        "max_probability": float(probabilities.max()),
    }

    bands = probability_bands(y_test, probabilities)

    # Save test predictions with useful identifiers and timestamps.
    prediction_columns = []

    for column in [
        "case_id",
        "customer_id",
        "opened_at",
        "outcome",
        "target_txn_id",
        "target_selection",
    ]:
        if column in test.columns:
            prediction_columns.append(column)

    predictions = test[prediction_columns].copy()
    predictions["actual_fraud"] = y_test
    predictions["fraud_probability"] = probabilities

    # Keep probabilities bounded for clean presentation.
    predictions["fraud_probability"] = predictions["fraud_probability"].clip(0.0, 1.0)

    predictions_path = OUTPUT_DIR / "final_probability_model_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    metrics_path = OUTPUT_DIR / "final_probability_model_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    summary = {
        "model_name": "hist_gradient_boosting_sigmoid_calibrated",
        "purpose": "case-level probability of historical fraud outcome given investigation evidence",
        "input": str(INPUT_PATH),
        "target": TARGET,
        "features": FEATURES,
        "removed_from_previous_model": [
            "strong_contradiction_count",
            "weak_contradiction_count",
            "out_of_region_behavior",
        ],
        "split": {
            "method": "chronological",
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_date_min": train["opened_at"].min().isoformat(),
            "train_date_max": train["opened_at"].max().isoformat(),
            "validation_date_min": validation["opened_at"].min().isoformat(),
            "validation_date_max": validation["opened_at"].max().isoformat(),
            "test_date_min": test["opened_at"].min().isoformat(),
            "test_date_max": test["opened_at"].max().isoformat(),
        },
        "calibration": {
            "method": "sigmoid_platt_explicit_logistic_regression",
            "fit_on": "validation_only",
            "test_used_for_fitting": False,
        },
        "test_metrics": metrics,
        "probability_bands": bands,
        "interpretation_notes": [
            "risk_score is retained as an upstream model input and is not redefined here.",
            "The historical case dataset has a high fraud prevalence, so these probabilities should not be assumed to be calibrated for a different production population without external prevalence validation.",
            "This model estimates case-level historical outcome probability, not an independently established transaction-level fraud probability.",
            "Probability bands are descriptive held-out-test diagnostics; they are not policy thresholds.",
        ],
    }

    summary_path = OUTPUT_DIR / "final_probability_model_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    notes = f"""FINAL FRAUD PROBABILITY MODEL
==============================

Model:
  HistGradientBoostingClassifier + sigmoid/Platt calibration

Feature count:
  {len(FEATURES)}

Features:
  {", ".join(FEATURES)}

Removed after ablation:
  - strong_contradiction_count
  - weak_contradiction_count
  - out_of_region_behavior

Chronological split:
  Train      : {len(train):,} rows, fraud rate {y_train.mean():.4f}
  Validation : {len(validation):,} rows, fraud rate {y_val.mean():.4f}
  Test       : {len(test):,} rows, fraud rate {y_test.mean():.4f}

Calibration:
  Sigmoid/Platt calibration fitted explicitly on validation only.
  Test set remained untouched until final evaluation.

TEST METRICS
------------
ROC-AUC   : {metrics["roc_auc"]:.6f}
PR-AUC    : {metrics["pr_auc"]:.6f}
Brier     : {metrics["brier_score"]:.6f}
Log Loss  : {metrics["log_loss"]:.6f}
ECE (10)  : {metrics["ece_10_bin"]:.6f}

Probability bands
-----------------
band       count    predicted_mean    observed_fraud_rate    abs_error
"""

    for row in bands:
        if row["count"] == 0:
            notes += (
                f'{row["band"]:>9} '
                f'{0:>8} '
                f'{"N/A":>17} '
                f'{"N/A":>21} '
                f'{"N/A":>10}\\n'
            )
        else:
            notes += (
                f'{row["band"]:>9} '
                f'{row["count"]:>8} '
                f'{row["predicted_probability_mean"]:>17.6f} '
                f'{row["observed_fraud_rate"]:>21.6f} '
                f'{row["calibration_error"]:>10.6f}\\n'
            )

    notes += """
Important interpretation:
- risk_score remains an input feature. The audit found its direction differs
  from the historical fraud label, but this script does not redefine its
  upstream semantics.
- Duplicate/all-zero features removed in the cleaned feature set had negligible
  impact in the ablation benchmark.
- The historical case population is approximately 84% fraud overall and about
  90% fraud in the held-out test split. Therefore, high predicted probabilities
  should not be interpreted as universally calibrated transaction-level
  probabilities in a lower-prevalence production population.
- HHG-001 through HHG-020 are not used here because they do not have authoritative
  labels.
"""

    notes_path = OUTPUT_DIR / "final_probability_model_notes.txt"
    notes_path.write_text(notes, encoding="utf-8")

    print()
    print("FINAL TEST METRICS")
    print("-" * 72)
    for key in [
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "ece_10_bin",
    ]:
        print(f"{key:>12}: {metrics[key]:.6f}")

    print()
    print("PROBABILITY BANDS")
    print("-" * 72)
    print(
        f'{"band":>9} {"count":>8} {"predicted":>12} '
        f'{"observed":>12} {"abs_error":>12}'
    )

    for row in bands:
        print(
            f'{row["band"]:>9} '
            f'{row["count"]:>8} '
            f'{str(round(row["predicted_probability_mean"], 6)) if row["predicted_probability_mean"] is not None else "N/A":>12} '
            f'{str(round(row["observed_fraud_rate"], 6)) if row["observed_fraud_rate"] is not None else "N/A":>12} '
            f'{str(round(row["calibration_error"], 6)) if row["calibration_error"] is not None else "N/A":>12}'
        )

    print()
    print("Created:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {notes_path}")


if __name__ == "__main__":
    main()
