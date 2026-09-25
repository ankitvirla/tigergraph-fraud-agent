"""
fraud_probability_calibration.py

First-stage analysis for designing the fraud-probability calibration layer.

IMPORTANT:
    This script does NOT claim to produce statistically calibrated fraud
    probabilities. The current benchmark evidence output does not contain a
    verified current-case fraud outcome label.

    It therefore:
      1. Loads calibrated evidence.
      2. Reconstructs evidence configuration features.
      3. Summarizes evidence patterns across the 20 benchmark cases.
      4. Separates risk_score from evidence-derived features.
      5. Identifies repeated evidence configurations.
      6. Produces candidate probability bands for design discussion only.
      7. Does NOT write fraud_probability into the benchmark cases.

Inputs:
    analysis/calibrated_evidence.csv

Outputs:
    analysis/probability_calibration_cases.csv
    analysis/probability_calibration_summary.json
    analysis/probability_calibration_notes.txt

Run:
    python fraud_probability_calibration.py
    python fraud_probability_calibration.py --analysis-dir analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORT_SIGNAL_COLUMNS = [
    "card_testing",
    "cnp",
    "cnp_new_device",
    "out_of_region",
    "account_takeover",
    "new_device",
    "proxy_network",
    "channel_novelty",
    "product_novelty",
    "out_of_region_behavior",
    "repeated_historical_abuse",
    "shared_device_profile",
]

CONTEXT_SIGNAL_COLUMNS = [
    "identity_match_status",
    "email_novelty",
    "post_flagged_activity",
]

VALID_STRENGTHS = {"strong", "medium", "weak", "not_material"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean(value: Any) -> Optional[str]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()
    return value if value else None


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def strength(row: pd.Series, signal: str) -> Optional[str]:
    return clean(row.get(f"{signal}_strength"))


def direction(row: pd.Series, signal: str) -> Optional[str]:
    return clean(row.get(f"{signal}_direction"))


def count_strength(row: pd.Series, target: str) -> int:
    count = 0

    for signal in SUPPORT_SIGNAL_COLUMNS + CONTEXT_SIGNAL_COLUMNS:
        if strength(row, signal) == target:
            count += 1

    # historical_cleared_signal is flattened separately in the CSV.
    if clean(row.get("historical_cleared_signal")) == target:
        count += 1

    return count


def supporting_signal_count(row: pd.Series) -> int:
    count = 0

    for signal in SUPPORT_SIGNAL_COLUMNS:
        if direction(row, signal) == "supports_fraud":
            count += 1

    return count


def contradiction_signal_count(row: pd.Series) -> int:
    count = 0

    for signal in SUPPORT_SIGNAL_COLUMNS + CONTEXT_SIGNAL_COLUMNS:
        if direction(row, signal) == "contradicts_fraud":
            count += 1

    if clean(row.get("historical_cleared_signal")):
        count += 1

    return count


def context_signal_count(row: pd.Series) -> int:
    count = 0

    for signal in SUPPORT_SIGNAL_COLUMNS + CONTEXT_SIGNAL_COLUMNS:
        if direction(row, signal) == "context":
            count += 1

    return count


def supporting_strength_count(row: pd.Series, target: str) -> int:
    count = 0

    for signal in SUPPORT_SIGNAL_COLUMNS:
        if (
            direction(row, signal) == "supports_fraud"
            and strength(row, signal) == target
        ):
            count += 1

    return count


def contradiction_strength_count(row: pd.Series, target: str) -> int:
    count = 0

    for signal in SUPPORT_SIGNAL_COLUMNS + CONTEXT_SIGNAL_COLUMNS:
        if (
            direction(row, signal) == "contradicts_fraud"
            and strength(row, signal) == target
        ):
            count += 1

    cleared_strength = clean(row.get("historical_cleared_signal"))
    if cleared_strength == target:
        count += 1

    return count


def evidence_signature(row: pd.Series) -> str:
    """
    Compact, deterministic signature of the evidence configuration.

    The signature intentionally excludes risk_score because risk_score is
    not fraud_probability and should not define evidence configuration.
    """

    parts = [
        f"strong_support={supporting_strength_count(row, 'strong')}",
        f"medium_support={supporting_strength_count(row, 'medium')}",
        f"weak_support={supporting_strength_count(row, 'weak')}",
        f"strong_contradiction={contradiction_strength_count(row, 'strong')}",
        f"medium_contradiction={contradiction_strength_count(row, 'medium')}",
        f"weak_contradiction={contradiction_strength_count(row, 'weak')}",
        f"context={context_signal_count(row)}",
        f"independent={as_int(row.get('independent_evidence_count'))}",
    ]

    groups = clean(row.get("independent_evidence_groups"))
    parts.append(f"groups={groups or ''}")

    return "|".join(parts)


# ---------------------------------------------------------------------------
# Candidate band
# ---------------------------------------------------------------------------

def candidate_band(row: pd.Series) -> str:
    """
    Design-only evidence sufficiency band.

    This is NOT a calibrated probability and must not be interpreted as one.

    The bands intentionally describe evidence configuration rather than
    claiming a probability of fraud.
    """

    independent = as_int(row.get("independent_evidence_count"))
    supporting = supporting_signal_count(row)
    contradicting = contradiction_signal_count(row)

    strong_support = supporting_strength_count(row, "strong")
    medium_support = supporting_strength_count(row, "medium")

    if independent == 0 and supporting == 0:
        return "insufficient_supporting_evidence"

    if contradicting >= 2 and independent <= 1:
        return "mixed_or_conflicted_evidence"

    if independent >= 4 and strong_support >= 2:
        return "high_evidence_convergence"

    if independent >= 3 and (strong_support >= 1 or medium_support >= 2):
        return "substantial_evidence_convergence"

    if independent >= 2:
        return "moderate_evidence_convergence"

    if independent == 1:
        return "limited_independent_evidence"

    return "insufficient_supporting_evidence"


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_case_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        case_id = clean(row.get("case_id"))

        record: Dict[str, Any] = {
            "case_id": case_id,
            "risk_score": as_float(row.get("risk_score")),

            "strong_evidence_count": count_strength(row, "strong"),
            "medium_evidence_count": count_strength(row, "medium"),
            "weak_evidence_count": count_strength(row, "weak"),
            "not_material_evidence_count": count_strength(
                row, "not_material"
            ),

            "supporting_evidence_count": supporting_signal_count(row),
            "contradicting_evidence_count": contradiction_signal_count(row),
            "context_evidence_count": context_signal_count(row),

            "strong_support_count": supporting_strength_count(
                row, "strong"
            ),
            "medium_support_count": supporting_strength_count(
                row, "medium"
            ),
            "weak_support_count": supporting_strength_count(
                row, "weak"
            ),

            "strong_contradiction_count": contradiction_strength_count(
                row, "strong"
            ),
            "medium_contradiction_count": contradiction_strength_count(
                row, "medium"
            ),
            "weak_contradiction_count": contradiction_strength_count(
                row, "weak"
            ),

            "independent_evidence_count": as_int(
                row.get("independent_evidence_count")
            ),
            "independent_evidence_groups": clean(
                row.get("independent_evidence_groups")
            ),

            "historical_cleared_signal": clean(
                row.get("historical_cleared_signal")
            ),

            "evidence_signature": evidence_signature(row),
        }

        record["candidate_evidence_band"] = candidate_band(row)

        rows.append(record)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def build_summary(features: pd.DataFrame) -> Dict[str, Any]:
    band_counts = (
        features["candidate_evidence_band"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    signature_counts = (
        features["evidence_signature"]
        .value_counts()
        .to_dict()
    )

    repeated_signatures = {
        signature: int(count)
        for signature, count in signature_counts.items()
        if count > 1
    }

    independent_distribution = (
        features["independent_evidence_count"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    risk_stats = {
        "min": float(features["risk_score"].min()),
        "max": float(features["risk_score"].max()),
        "mean": float(features["risk_score"].mean()),
        "median": float(features["risk_score"].median()),
    }

    return {
        "purpose": (
            "First-stage probability calibration analysis. "
            "This output is not statistically calibrated fraud probability."
        ),
        "benchmark_case_count": int(len(features)),
        "risk_score_is_fraud_probability": False,
        "verified_current_case_ground_truth_available": False,
        "candidate_evidence_band_counts": band_counts,
        "independent_evidence_distribution": {
            str(int(k)): int(v)
            for k, v in independent_distribution.items()
        },
        "risk_score_summary": risk_stats,
        "repeated_evidence_configurations": repeated_signatures,
        "next_required_step": (
            "Provide or derive a verified current-case outcome label "
            "for benchmark calibration before fitting statistical "
            "fraud probabilities."
        ),
    }


def build_notes(features: pd.DataFrame, summary: Dict[str, Any]) -> str:
    lines = [
        "=" * 72,
        "FRAUD PROBABILITY CALIBRATION - FIRST STAGE ANALYSIS",
        "=" * 72,
        "",
        "STATUS",
        "------",
        "The evidence calibration layer has been consumed successfully.",
        "This script does NOT produce statistically calibrated fraud",
        "probabilities and does NOT make fraud/policy decisions.",
        "",
        "IMPORTANT DISTINCTION",
        "---------------------",
        "risk_score is treated as an observed upstream score/signal.",
        "risk_score is NOT used as fraud_probability.",
        "",
        "CURRENT LIMITATION",
        "------------------",
        "The calibrated benchmark output does not contain a verified",
        "current-case fraud outcome label. Therefore the 20 benchmark cases",
        "cannot by themselves justify a statistically calibrated mapping",
        "from evidence configuration to probability.",
        "",
        "EVIDENCE BANDS",
        "--------------",
    ]

    for band, count in summary["candidate_evidence_band_counts"].items():
        lines.append(f"{band}: {count}")

    lines.extend(
        [
            "",
            "These bands are descriptive evidence configurations only.",
            "They must NOT be interpreted as probabilities.",
            "",
            "INDEPENDENT EVIDENCE DISTRIBUTION",
            "---------------------------------",
        ]
    )

    for count, cases in summary["independent_evidence_distribution"].items():
        lines.append(f"{count} independent groups: {cases} cases")

    lines.extend(
        [
            "",
            "RISK SCORE SUMMARY",
            "------------------",
        ]
    )

    risk = summary["risk_score_summary"]
    lines.extend(
        [
            f"min    = {risk['min']:.6f}",
            f"max    = {risk['max']:.6f}",
            f"mean   = {risk['mean']:.6f}",
            f"median = {risk['median']:.6f}",
            "",
            "NEXT STEP",
            "---------",
            "Before implementing a probability formula, establish a verified",
            "outcome target for the benchmark cases, or obtain an appropriate",
            "historical labeled dataset.",
            "",
            "Once an outcome target exists, evaluate candidate methods such as:",
            "1. logistic regression over calibrated evidence features,",
            "2. monotonic evidence-score calibration,",
            "3. isotonic/Platt calibration when enough labeled data exists,",
            "4. benchmark holdout/cross-validation to measure calibration.",
            "",
            "Do not assign arbitrary probabilities such as 0.85 merely because",
            "a case has many evidence signals. Evidence strength and independence",
            "must first be related to observed outcomes.",
            "",
            "=" * 72,
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="First-stage fraud probability calibration analysis."
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis",
        help="Directory containing calibrated_evidence.csv.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    input_path = analysis_dir / "calibrated_evidence.csv"

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}")
        return 2

    print("=" * 72)
    print("FRAUD PROBABILITY CALIBRATION")
    print("=" * 72)
    print(f"Input: {input_path}")

    df = pd.read_csv(input_path)

    if "case_id" not in df.columns:
        print("ERROR: case_id column is missing.")
        return 2

    if len(df) != 20:
        print(
            f"WARNING: expected 20 benchmark rows, found {len(df)}."
        )

    print(f"Benchmark cases: {len(df)}")

    features = build_case_features(df)

    output_csv = analysis_dir / "probability_calibration_cases.csv"
    output_json = analysis_dir / "probability_calibration_summary.json"
    output_notes = analysis_dir / "probability_calibration_notes.txt"

    features.to_csv(output_csv, index=False)

    summary = build_summary(features)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    notes = build_notes(features, summary)
    output_notes.write_text(notes, encoding="utf-8")

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Cases:  {output_csv}")
    print(f"JSON:   {output_json}")
    print(f"Notes:  {output_notes}")
    print()
    print("IMPORTANT:")
    print(
        "No statistically calibrated fraud probability was calculated."
    )
    print(
        "risk_score was kept separate from fraud_probability."
    )
    print(
        "Verified outcome labels are required before fitting a true "
        "probability model."
    )
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
