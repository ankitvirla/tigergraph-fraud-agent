"""
validate_calibration.py

Validates the semantic output of benchmark_evidence_calibration.py.

This script intentionally does NOT calculate fraud probability or make
fraud/policy decisions. It validates the evidence-calibration layer only.

Expected inputs:
    analysis/calibrated_evidence.csv
    analysis/calibrated_evidence.json   (optional; CSV validation still runs)

Run:
    python validate_calibration.py

Optional:
    python validate_calibration.py --analysis-dir analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPECTED_CASES = {f"HHG-{i:03d}" for i in range(1, 21)}

VALID_STRENGTHS = {
    "strong",
    "medium",
    "weak",
    "not_material",
}

VALID_DIRECTIONS = {
    "supports_fraud",
    "contradicts_fraud",
    "context",
}

# Signal -> expected direction.
# shared_device_profile is special: its direction depends on rarity, so it is
# validated separately.
EXPECTED_DIRECTIONS = {
    "card_testing": "supports_fraud",
    "cnp": "supports_fraud",
    "cnp_new_device": "supports_fraud",
    "out_of_region": "supports_fraud",
    "account_takeover": "supports_fraud",
    "new_device": "supports_fraud",
    "proxy_network": "supports_fraud",
    "channel_novelty": "supports_fraud",
    "product_novelty": "supports_fraud",
    "out_of_region_behavior": "supports_fraud",
    "repeated_historical_abuse": "supports_fraud",

    "identity_match_status": "context",
    "email_novelty": "context",
    "post_flagged_activity": "context",

    # Historical direction is conditional:
    # confirmed history -> supports_fraud
    # cleared history   -> contradicts_fraud
    "historical_case": "conditional",
}

# Signals which are allowed to share an independence family.
EXPECTED_GROUPS = {
    "card_testing": "behavioral_pattern",
    "cnp": "behavioral_pattern",
    "cnp_new_device": "behavioral_pattern",
    "out_of_region": "behavioral_pattern",
    "channel_novelty": "behavioral_pattern",
    "product_novelty": "behavioral_pattern",
    "out_of_region_behavior": "behavioral_pattern",

    "account_takeover": "behavioral_identity",

    "new_device": "identity_device",
    "proxy_network": "identity_network",

    "historical_case": "historical",
    "repeated_historical_abuse": "historical",

    "shared_device_profile": "network",
}

# These are explicitly NOT allowed to create a supporting independent group.
NON_SUPPORTING_SIGNALS = {
    "identity_match_status",
    "email_novelty",
    "post_flagged_activity",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Validator:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passes: List[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def passed(self, message: str) -> None:
        self.passes.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    text = str(value).strip()
    return text if text else None


def as_int(value: Any) -> Optional[int]:
    if is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_number(value: Any) -> bool:
    if is_missing(value):
        return True
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def signal_columns(df: pd.DataFrame) -> List[str]:
    """Return base signal names that have *_strength and *_direction columns."""
    result = []
    for col in df.columns:
        if not col.endswith("_strength"):
            continue
        base = col[:-len("_strength")]
        if f"{base}_direction" in df.columns:
            result.append(base)
    return sorted(result)


def get_signal_strength(row: pd.Series, signal: str) -> Optional[str]:
    return clean(row.get(f"{signal}_strength"))


def get_signal_direction(row: pd.Series, signal: str) -> Optional[str]:
    return clean(row.get(f"{signal}_direction"))


def parse_groups(value: Any) -> Set[str]:
    text = clean(value)
    if not text:
        return set()
    return {part.strip() for part in text.split("|") if part.strip()}


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_structure(df: pd.DataFrame, v: Validator) -> None:
    required = {
        "case_id",
        "risk_score",
        "strong_evidence_count",
        "medium_evidence_count",
        "weak_evidence_count",
        "not_material_evidence_count",
        "supporting_evidence_count",
        "contradicting_evidence_count",
        "context_evidence_count",
        "independent_evidence_count",
        "independent_evidence_groups",
        "contradiction_count",
        "fraud_probability",
        "decision",
        "stop_reason",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        v.error(f"Missing required CSV columns: {missing}")
    else:
        v.passed("Required CSV columns are present.")

    if len(df) != 20:
        v.error(f"Expected 20 benchmark rows, found {len(df)}.")
    else:
        v.passed("Exactly 20 benchmark rows are present.")

    actual_cases = set(df["case_id"].dropna().astype(str))
    missing_cases = sorted(EXPECTED_CASES - actual_cases)
    extra_cases = sorted(actual_cases - EXPECTED_CASES)

    if missing_cases:
        v.error(f"Missing benchmark cases: {missing_cases}")
    if extra_cases:
        v.error(f"Unexpected benchmark cases: {extra_cases}")
    if not missing_cases and not extra_cases:
        v.passed("HHG-001 through HHG-020 are all present exactly once.")

    duplicates = df["case_id"].astype(str).duplicated()
    if duplicates.any():
        dup_ids = sorted(df.loc[duplicates, "case_id"].astype(str).unique())
        v.error(f"Duplicate case IDs found: {dup_ids}")
    else:
        v.passed("Case IDs are unique.")

    if not df["risk_score"].apply(is_number).all():
        v.error("One or more risk_score values are non-numeric.")
    else:
        v.passed("risk_score values are numeric.")

    # Calibration layer must not populate final inference fields.
    for column in ("fraud_probability", "decision", "stop_reason"):
        populated = df[column].notna() & (df[column].astype(str).str.strip() != "")
        if populated.any():
            cases = df.loc[populated, "case_id"].astype(str).tolist()
            v.error(
                f"{column} is populated for calibration cases: {cases}. "
                "This layer must not calculate final inference/decision."
            )
        else:
            v.passed(f"{column} remains empty as expected.")


# ---------------------------------------------------------------------------
# Signal-level validation
# ---------------------------------------------------------------------------

def validate_signal_semantics(df: pd.DataFrame, v: Validator) -> None:
    signals = signal_columns(df)

    if not signals:
        v.error("No *_strength / *_direction signal pairs were found.")
        return

    v.passed(f"Found {len(signals)} calibrated signal types.")

    for signal in signals:
        strength_col = f"{signal}_strength"
        direction_col = f"{signal}_direction"

        for idx, row in df.iterrows():
            strength = get_signal_strength(row, signal)
            direction = get_signal_direction(row, signal)
            case_id = str(row["case_id"])

            # If a strength exists, direction should exist.
            if strength and not direction:
                v.error(
                    f"{case_id}: {signal} has strength={strength!r} "
                    "but no direction."
                )
                continue

            # If direction exists, strength should exist.
            if direction and not strength:
                v.error(
                    f"{case_id}: {signal} has direction={direction!r} "
                    "but no strength."
                )
                continue

            if strength and strength not in VALID_STRENGTHS:
                v.error(
                    f"{case_id}: {signal} has invalid strength={strength!r}."
                )

            if direction and direction not in VALID_DIRECTIONS:
                v.error(
                    f"{case_id}: {signal} has invalid direction={direction!r}."
                )

            if not direction:
                continue

            expected = EXPECTED_DIRECTIONS.get(signal)

            if expected and expected != "conditional" and direction != expected:
                v.error(
                    f"{case_id}: {signal} direction={direction!r}; "
                    f"expected {expected!r}."
                )

            if signal in NON_SUPPORTING_SIGNALS and direction == "supports_fraud":
                v.error(
                    f"{case_id}: {signal} is contextual but is marked "
                    "supports_fraud."
                )

            # not_material evidence should not become supporting evidence.
            if strength == "not_material" and direction == "supports_fraud":
                v.error(
                    f"{case_id}: {signal} is not_material but marked "
                    "supports_fraud."
                )

    # Shared-device direction is intentionally conditional.
    v.passed(
        "Signal direction/strength combinations were checked against the "
        "calibration rules."
    )


# ---------------------------------------------------------------------------
# Historical validation
# ---------------------------------------------------------------------------

def validate_historical(df: pd.DataFrame, v: Validator) -> None:
    if "historical_case_direction" not in df.columns:
        v.error("historical_case_direction column is missing.")
        return

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        historical_direction = clean(row.get("historical_case_direction"))
        cleared_strength = clean(row.get("historical_cleared_signal"))

        if historical_direction and historical_direction not in {
            "supports_fraud",
            "context",
            "contradicts_fraud",
        }:
            v.error(
                f"{case_id}: invalid historical_case_direction="
                f"{historical_direction!r}."
            )

        # Cleared historical evidence is a separate signal. It is valid
        # for a case to contain both:
        #   confirmed historical cases -> supports_fraud
        #   cleared historical cases  -> contradicts_fraud
        # The two must not overwrite each other.
        if cleared_strength:
            if cleared_strength not in VALID_STRENGTHS:
                v.error(
                    f"{case_id}: invalid historical_cleared_signal="
                    f"{cleared_strength!r}."
                )

            contradiction_count = as_int(row.get("contradiction_count"))
            if contradiction_count is not None and contradiction_count < 1:
                v.error(
                    f"{case_id}: cleared historical evidence exists but "
                    f"contradiction_count={contradiction_count}."
                )

    v.passed(
        "Historical evidence was checked so cleared history cannot silently "
        "become supporting fraud evidence."
    )


# ---------------------------------------------------------------------------
# Shared-device validation
# ---------------------------------------------------------------------------

def validate_shared_device(df: pd.DataFrame, v: Validator) -> None:
    if "shared_device_profile_strength" not in df.columns:
        v.warning("shared_device_profile columns are absent; device checks skipped.")
        return

    for _, row in df.iterrows():
        case_id = str(row["case_id"])
        strength = clean(row.get("shared_device_profile_strength"))
        direction = clean(row.get("shared_device_profile_direction"))

        if not strength:
            continue

        if strength == "not_material" and direction == "supports_fraud":
            v.error(
                f"{case_id}: shared device is not_material but supports fraud."
            )

        if direction not in VALID_DIRECTIONS:
            v.error(
                f"{case_id}: shared device has invalid direction={direction!r}."
            )

    # Known benchmark expectations from the evidence calibration analysis.
    # These are semantic guardrails, not probability labels.
    for case_id in ("HHG-006", "HHG-010", "HHG-017"):
        row = df[df["case_id"].astype(str) == case_id]
        if row.empty:
            continue
        strength = clean(row.iloc[0].get("shared_device_profile_strength"))
        if strength == "strong":
            v.error(
                f"{case_id}: known common shared-device benchmark became "
                "strong; inspect device calibration."
            )

    v.passed(
        "Shared-device calibration passed common-device guardrails."
    )


# ---------------------------------------------------------------------------
# Evidence-count validation
# ---------------------------------------------------------------------------

def validate_counts(df: pd.DataFrame, v: Validator) -> None:
    signals = signal_columns(df)

    for _, row in df.iterrows():
        case_id = str(row["case_id"])

        counts = {
            "strong": 0,
            "medium": 0,
            "weak": 0,
            "not_material": 0,
        }

        supporting = 0
        contradicting = 0
        context = 0

        for signal in signals:
            strength = get_signal_strength(row, signal)
            direction = get_signal_direction(row, signal)

            if strength in counts:
                counts[strength] += 1

            if direction == "supports_fraud":
                supporting += 1
            elif direction == "contradicts_fraud":
                contradicting += 1
            elif direction == "context":
                context += 1

        # historical_cleared_signal is a derived contradiction column in the
        # flattened CSV rather than a *_strength/*_direction signal pair.
        # It still represents real calibrated evidence and therefore must be
        # included in the strength and contradiction counts.
        cleared_strength = clean(row.get("historical_cleared_signal"))
        if cleared_strength in counts:
            counts[cleared_strength] += 1
            contradicting += 1

        for strength, expected in counts.items():
            actual = as_int(row.get(f"{strength}_evidence_count"))
            if actual is None:
                v.error(
                    f"{case_id}: missing/non-numeric {strength}_evidence_count."
                )
            elif actual != expected:
                v.error(
                    f"{case_id}: {strength}_evidence_count={actual}, "
                    f"but signal rows contain {expected}."
                )

        actual_supporting = as_int(row.get("supporting_evidence_count"))
        actual_contradicting = as_int(row.get("contradicting_evidence_count"))
        actual_context = as_int(row.get("context_evidence_count"))

        if actual_supporting != supporting:
            v.error(
                f"{case_id}: supporting_evidence_count={actual_supporting}, "
                f"but directions contain {supporting} supporting signals."
            )

        if actual_contradicting != contradicting:
            v.error(
                f"{case_id}: contradicting_evidence_count="
                f"{actual_contradicting}, but directions contain "
                f"{contradicting} contradiction signals."
            )

        if actual_context != context:
            v.error(
                f"{case_id}: context_evidence_count={actual_context}, "
                f"but directions contain {context} context signals."
            )

    if not v.errors:
        v.passed(
            "Evidence strength/direction counts match signal columns, "
            "including historical_cleared_signal."
        )


# ---------------------------------------------------------------------------
# Independence validation
# ---------------------------------------------------------------------------

def validate_independence(df: pd.DataFrame, v: Validator) -> None:
    signals = signal_columns(df)

    for _, row in df.iterrows():
        case_id = str(row["case_id"])

        actual_groups = parse_groups(row.get("independent_evidence_groups"))
        actual_count = as_int(row.get("independent_evidence_count"))

        if actual_count is None:
            v.error(f"{case_id}: independent_evidence_count is missing/non-numeric.")
            continue

        if actual_count != len(actual_groups):
            v.error(
                f"{case_id}: independent_evidence_count={actual_count}, "
                f"but independent_evidence_groups contains "
                f"{len(actual_groups)} groups."
            )

        # Reconstruct expected groups from calibrated signal columns.
        expected_groups: Set[str] = set()

        for signal in signals:
            strength = get_signal_strength(row, signal)
            direction = get_signal_direction(row, signal)

            if (
                direction != "supports_fraud"
                or strength not in {"strong", "medium"}
            ):
                continue

            group = EXPECTED_GROUPS.get(signal)

            if group:
                expected_groups.add(group)

        if expected_groups != actual_groups:
            v.error(
                f"{case_id}: independent groups mismatch. "
                f"expected={sorted(expected_groups)}, "
                f"actual={sorted(actual_groups)}."
            )

        # Explicitly prevent context/contradiction from creating groups.
        for signal in NON_SUPPORTING_SIGNALS:
            direction = get_signal_direction(row, signal)
            if direction in {"context", "contradicts_fraud"}:
                expected = EXPECTED_GROUPS.get(signal)
                if expected and expected in actual_groups:
                    v.error(
                        f"{case_id}: {signal} created independent group "
                        f"{expected!r} despite being non-supporting."
                    )

    if not v.errors:
        v.passed(
            "Independent evidence groups match supporting strong/medium "
            "signals and exclude context/contradictions."
        )


# ---------------------------------------------------------------------------
# Benchmark sanity cases
# ---------------------------------------------------------------------------

def validate_sanity_cases(df: pd.DataFrame, v: Validator) -> None:
    def row_for(case_id: str) -> Optional[pd.Series]:
        rows = df[df["case_id"].astype(str) == case_id]
        return None if rows.empty else rows.iloc[0]

    # HHG-009: intentionally insufficient.
    row = row_for("HHG-009")
    if row is not None:
        independent = as_int(row.get("independent_evidence_count"))
        supporting = as_int(row.get("supporting_evidence_count"))
        if independent != 0 or supporting != 0:
            v.error(
                f"HHG-009 sanity failure: expected 0 independent and 0 "
                f"supporting evidence, got independent={independent}, "
                f"supporting={supporting}."
            )
        else:
            v.passed("HHG-009 sanity check passed: no supporting evidence.")

    # HHG-011: convergence across multiple independent families.
    row = row_for("HHG-011")
    if row is not None:
        independent = as_int(row.get("independent_evidence_count"))
        if independent is None or independent < 3:
            v.error(
                f"HHG-011 sanity failure: expected multi-family convergence; "
                f"independent_evidence_count={independent}."
            )
        else:
            v.passed(
                f"HHG-011 sanity check passed: "
                f"{independent} independent evidence groups."
            )

    # HHG-019: rare-device/network evidence should survive calibration.
    row = row_for("HHG-019")
    if row is not None:
        independent = as_int(row.get("independent_evidence_count"))
        device_strength = clean(row.get("shared_device_profile_strength"))
        if independent is None or independent < 3:
            v.error(
                f"HHG-019 sanity failure: expected several independent "
                f"evidence families; got {independent}."
            )
        elif device_strength not in {"strong", "medium"}:
            v.warning(
                "HHG-019 shared-device signal is not strong/medium. "
                "This may be valid, but inspect the device calibration."
            )
        else:
            v.passed(
                "HHG-019 sanity check passed: multi-family evidence and "
                "device signal retained."
            )


# ---------------------------------------------------------------------------
# Optional JSON validation
# ---------------------------------------------------------------------------

def validate_json(json_path: Path, v: Validator) -> None:
    if not json_path.exists():
        v.warning(
            f"JSON not found at {json_path}; CSV validation completed without it."
        )
        return

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        v.error(f"Could not parse calibrated_evidence.json: {exc}")
        return

    if isinstance(data, dict):
        # Common possible wrappers.
        for key in ("cases", "calibrated_cases", "results", "benchmark_cases"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        v.warning(
            "calibrated_evidence.json was parsed but its top-level structure "
            "is not a list/wrapper containing a list; skipped deep JSON checks."
        )
        return

    json_cases = set()
    missing_direction = []
    malformed_evidence = []

    for item in data:
        if not isinstance(item, dict):
            malformed_evidence.append("non-object case")
            continue

        case_id = clean(item.get("case_id"))
        if case_id:
            json_cases.add(case_id)

        evidence = item.get("evidence", [])

        if evidence is None:
            evidence = []

        if not isinstance(evidence, list):
            malformed_evidence.append(
                f"{case_id or '<unknown>'}: evidence is not a list"
            )
            continue

        for evidence_item in evidence:
            if not isinstance(evidence_item, dict):
                malformed_evidence.append(
                    f"{case_id or '<unknown>'}: evidence item is not an object"
                )
                continue

            required = {
                "signal_type",
                "strength",
                "direction",
                "independence_group",
                "source",
                "raw",
                "explanation",
            }

            missing = required - set(evidence_item)
            if missing:
                malformed_evidence.append(
                    f"{case_id or '<unknown>'}: missing evidence fields "
                    f"{sorted(missing)}"
                )

            if not clean(evidence_item.get("direction")):
                missing_direction.append(case_id or "<unknown>")

    if missing_direction:
        v.error(
            "JSON evidence items without direction: "
            f"{sorted(set(missing_direction))}"
        )
    else:
        v.passed("JSON evidence items contain direction fields.")

    if malformed_evidence:
        for issue in malformed_evidence[:20]:
            v.error(f"JSON evidence schema: {issue}")
        if len(malformed_evidence) > 20:
            v.error(
                f"...and {len(malformed_evidence) - 20} additional JSON schema issues."
            )
    else:
        v.passed("JSON evidence objects contain the expected evidence schema.")

    if json_cases and json_cases != EXPECTED_CASES:
        v.error(
            f"JSON case IDs differ from benchmark set: "
            f"missing={sorted(EXPECTED_CASES - json_cases)}, "
            f"extra={sorted(json_cases - EXPECTED_CASES)}"
        )
    elif json_cases:
        v.passed("JSON contains HHG-001 through HHG-020.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(v: Validator, csv_path: Path, json_path: Path) -> None:
    print()
    print("=" * 72)
    print("CALIBRATION VALIDATION")
    print("=" * 72)
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print()

    print(f"PASS : {len(v.passes)}")
    print(f"WARN : {len(v.warnings)}")
    print(f"FAIL : {len(v.errors)}")
    print()

    if v.passes:
        print("PASSED CHECKS")
        print("-" * 72)
        for item in v.passes:
            print(f"[PASS] {item}")
        print()

    if v.warnings:
        print("WARNINGS")
        print("-" * 72)
        for item in v.warnings:
            print(f"[WARN] {item}")
        print()

    if v.errors:
        print("FAILURES")
        print("-" * 72)
        for item in v.errors:
            print(f"[FAIL] {item}")
        print()

    print("=" * 72)
    if v.ok:
        if v.warnings:
            print("OVERALL: PASS WITH WARNINGS")
        else:
            print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")
    print("=" * 72)
    print()

    if v.ok:
        print(
            "Calibration layer is structurally/semantically ready for the "
            "next review step."
        )
        print(
            "No fraud probability or final decision was calculated by this script."
        )
    else:
        print(
            "Do NOT build/freeze the probability engine yet. "
            "Fix the calibration failures first."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate benchmark evidence calibration output."
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis",
        help="Directory containing calibrated_evidence.csv/json.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    csv_path = analysis_dir / "calibrated_evidence.csv"
    json_path = analysis_dir / "calibrated_evidence.json"

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 2

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"ERROR: Could not read {csv_path}: {exc}", file=sys.stderr)
        return 2

    v = Validator()

    validate_structure(df, v)
    validate_signal_semantics(df, v)
    validate_historical(df, v)
    validate_shared_device(df, v)
    validate_counts(df, v)
    validate_independence(df, v)
    validate_sanity_cases(df, v)
    validate_json(json_path, v)

    print_report(v, csv_path, json_path)

    return 0 if v.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())