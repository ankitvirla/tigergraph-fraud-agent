#!/usr/bin/env python3

"""
benchmark_evidence_calibration.py

Evidence calibration layer for the TigerGraph Agentic Fraud Investigation
project.

Purpose
-------
Convert raw observations produced by benchmark_deep_analysis.py into
calibrated, auditable evidence for the future investigation agent.

This script DOES:
    - classify evidence strength as strong / medium / weak / not_material
    - account for shared-device profile rarity
    - group correlated signals into independent evidence groups
    - assign evidence direction (supports / contradicts / context)
    - identify contradictions
    - preserve raw observations
    - generate human-readable explanations
    - retain historical evidence separately from current behavior

This script DOES NOT:
    - calculate fraud_probability
    - declare fraud
    - make a final policy decision
    - convert risk_score into fraud_probability
    - invent card K1/K2 mappings
    - treat a common device profile as proof of a shared physical device

Inputs
------
analysis/benchmark_summary.csv
analysis/shared_device_network.csv
analysis/historical_case_summary.csv

Optional:
analysis/region_analysis.csv
analysis/email_analysis.csv
analysis/transaction_windows.csv

Outputs
-------
analysis/calibrated_evidence.json
analysis/calibrated_evidence.csv
analysis/evidence_calibration_notes.txt

Run
---
python benchmark_evidence_calibration.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "analysis"

BENCHMARK_SUMMARY_FILE = ANALYSIS_DIR / "benchmark_summary.csv"
SHARED_DEVICE_FILE = ANALYSIS_DIR / "shared_device_network.csv"
HISTORICAL_FILE = ANALYSIS_DIR / "historical_case_summary.csv"
REGION_FILE = ANALYSIS_DIR / "region_analysis.csv"
EMAIL_FILE = ANALYSIS_DIR / "email_analysis.csv"
WINDOW_FILE = ANALYSIS_DIR / "transaction_windows.csv"

OUTPUT_JSON = ANALYSIS_DIR / "calibrated_evidence.json"
OUTPUT_CSV = ANALYSIS_DIR / "calibrated_evidence.csv"
OUTPUT_NOTES = ANALYSIS_DIR / "evidence_calibration_notes.txt"


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True,
    )


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def is_missing(value: Any) -> bool:
    """
    Robust missing-value check.
    """
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def clean_value(value: Any) -> Any:
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """
    if is_missing(value):
        return None

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (np.ndarray,)):
        return value.tolist()

    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]

    if isinstance(value, dict):
        return {
            str(k): clean_value(v)
            for k, v in value.items()
        }

    return value


def safe_bool(value: Any) -> bool:
    """
    Convert common CSV boolean representations to bool.
    """
    if is_missing(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if is_missing(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        if is_missing(value):
            return default
        return float(value)
    except Exception:
        return default


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""

    return str(value).strip().lower()


def first_existing(
    row: pd.Series,
    candidates: Iterable[str],
    default: Any = None,
) -> Any:
    """
    Return the first available non-null column value.
    """
    for column in candidates:
        if column in row.index:
            value = row[column]

            if not is_missing(value):
                return value

    return default


def column_exists(
    df: pd.DataFrame,
    *columns: str,
) -> bool:
    return all(column in df.columns for column in columns)


def read_csv_if_exists(
    path: Path,
) -> Optional[pd.DataFrame]:

    if not path.exists():
        log(f"Optional file not found: {path}")
        return None

    log(f"Loading {path.name}...")

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    log(
        f"Loaded {path.name}: "
        f"{len(df):,} rows, "
        f"{len(df.columns):,} columns"
    )

    return df


# ============================================================================
# FILE VALIDATION
# ============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Run benchmark_deep_analysis.py first."
        )


def validate_inputs() -> None:

    require_file(BENCHMARK_SUMMARY_FILE)

    if not SHARED_DEVICE_FILE.exists():
        log(
            "WARNING: shared_device_network.csv not found. "
            "Device evidence will be limited."
        )

    if not HISTORICAL_FILE.exists():
        log(
            "WARNING: historical_case_summary.csv not found. "
            "Historical evidence will be limited."
        )


# ============================================================================
# COLUMN DISCOVERY
# ============================================================================

def find_case_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "case_id",
        "benchmark_case_id",
        "case",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def find_customer_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "customer_id",
        "customer",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


def find_card_id_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "card_id",
        "card",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================================
# DEVICE PROFILE CALIBRATION
# ============================================================================

def device_strength(
    customer_count: int,
    transaction_count: int,
    specificity: Optional[int],
) -> Tuple[str, str]:

    """
    Calibrate shared-device profile evidence.

    IMPORTANT:
    Exact profile equality does not prove a physical shared device.

    A profile such as:
        Windows + Windows 10 + Chrome + 1920x1080

    can naturally occur across many customers.

    Therefore common profiles are deliberately down-weighted.
    """

    customer_count = max(customer_count, 0)
    transaction_count = max(transaction_count, 0)

    specificity_value = (
        safe_int(specificity)
        if specificity is not None
        else 0
    )

    # No meaningful exact profile.
    if specificity_value <= 1:
        return (
            "not_material",
            "Device profile is too incomplete to support a meaningful "
            "cross-customer shared-profile signal.",
        )

    # Very common profile.
    if customer_count > 50:
        return (
            "weak",
            "The exact device profile is used by many customers, so "
            "profile matching is treated as a weak signal rather than "
            "evidence of a shared physical device.",
        )

    # Moderately common profile.
    if customer_count > 10:
        return (
            "weak",
            "The exact device profile is shared across multiple customers. "
            "It is potentially useful as network context but is not strong "
            "evidence of a shared physical device.",
        )

    # Rare profile with complete components.
    if specificity_value == 4 and customer_count <= 3:
        return (
            "strong",
            "The complete device profile is rare across customers and "
            "matches activity involving other customers before case opening. "
            "This remains a profile-level signal, not proof of physical "
            "device ownership.",
        )

    # Rare/infrequent profile but incomplete.
    if customer_count <= 3:
        return (
            "medium",
            "The device profile is relatively rare across customers, but "
            "profile completeness does not justify treating it as definitive "
            "shared-device evidence.",
        )

    return (
        "weak",
        "The device profile has some cross-customer reuse and is retained "
        "as contextual network evidence.",
    )


def extract_device_evidence(
    row: pd.Series,
    shared_device_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    candidate = safe_bool(
        first_existing(
            row,
            [
                "shared_device_candidate",
                "shared_device",
                "device_shared",
            ],
        )
    )

    if not candidate and shared_device_df is None:
        return evidence

    customer_count = safe_int(
        first_existing(
            row,
            [
                "shared_device_customer_count",
                "device_customer_count",
            ],
        )
    )

    transaction_count = safe_int(
        first_existing(
            row,
            [
                "shared_device_transaction_count",
                "device_transaction_count",
            ],
        )
    )

    other_customer_count = safe_int(
        first_existing(
            row,
            [
                "other_customers_on_device",
                "other_customer_count",
            ],
        )
    )

    specificity = safe_int(
        first_existing(
            row,
            [
                "device_profile_specificity",
                "device_specificity",
                "profile_specificity",
            ],
            default=0,
        )
    )

    profile = first_existing(
        row,
        [
            "device_profile",
            "device_profile_fingerprint",
        ],
    )

    # If summary doesn't contain counts, attempt lookup from network file.
    if (
        shared_device_df is not None
        and profile is not None
        and "device_profile" in shared_device_df.columns
    ):

        matches = shared_device_df[
            shared_device_df["device_profile"].astype(str)
            == str(profile)
        ]

        if not matches.empty:

            network_row = matches.iloc[0]

            customer_count = max(
                customer_count,
                safe_int(
                    first_existing(
                        network_row,
                        [
                            "customer_count",
                            "shared_device_customer_count",
                        ],
                    )
                ),
            )

            transaction_count = max(
                transaction_count,
                safe_int(
                    first_existing(
                        network_row,
                        [
                            "transaction_count",
                            "shared_device_transaction_count",
                        ],
                    )
                ),
            )

            other_customer_count = max(
                other_customer_count,
                safe_int(
                    first_existing(
                        network_row,
                        [
                            "other_customer_count",
                            "other_customers_on_device",
                        ],
                    )
                ),
            )

            specificity = max(
                specificity,
                safe_int(
                    first_existing(
                        network_row,
                        [
                            "device_profile_specificity",
                            "profile_specificity",
                        ],
                        default=0,
                    )
                ),
            )

    if not candidate and other_customer_count <= 0:
        return evidence

    strength, explanation = device_strength(
        customer_count=customer_count,
        transaction_count=transaction_count,
        specificity=specificity,
    )

    evidence.append(
        {
            "signal_type": "shared_device_profile",
            "strength": strength,
            "direction": (
                "supports_fraud"
                if strength in {"strong", "medium"}
                else "context"
            ),
            "independence_group": "network",
            "source": "shared_device_network",
            "raw": {
                "candidate": candidate,
                "device_profile": clean_value(profile),
                "device_profile_specificity": specificity,
                "customer_count": customer_count,
                "transaction_count": transaction_count,
                "other_customer_count": other_customer_count,
            },
            "explanation": explanation,
        }
    )

    return evidence


# ============================================================================
# HISTORICAL EVIDENCE
# ============================================================================

def historical_strength(
    confirmed_count: int,
    cleared_count: int,
    total_count: int,
) -> Tuple[str, str]:

    if total_count <= 0:
        return (
            "not_material",
            "No qualifying historical cases were found before the "
            "benchmark case opened.",
        )

    if confirmed_count >= 2 and cleared_count == 0:
        return (
            "strong",
            f"{confirmed_count} prior cases were confirmed fraud with "
            "no prior cleared cases in the qualifying history.",
        )

    if confirmed_count >= 2 and cleared_count > 0:
        return (
            "strong",
            f"{confirmed_count} prior cases were confirmed fraud, although "
            f"{cleared_count} prior cases were also cleared. Historical "
            "evidence therefore supports elevated concern but is not proof "
            "of the current event.",
        )

    if confirmed_count == 1 and cleared_count == 0:
        return (
            "medium",
            "One prior case was confirmed fraud with no qualifying cleared "
            "case in the available history.",
        )

    if confirmed_count >= 1 and cleared_count >= 1:
        return (
            "medium",
            f"Historical activity contains {confirmed_count} confirmed "
            f"case(s) and {cleared_count} cleared case(s). This provides "
            "context but also introduces contradictory historical evidence.",
        )

    if confirmed_count == 0 and cleared_count > 0:
        return (
            "weak",
            "Prior alerts were cleared rather than confirmed fraud. "
            "This is historical context and should not be interpreted as "
            "support for current fraud.",
        )

    return (
        "weak",
        "Historical cases exist, but their outcomes do not provide strong "
        "support for the current investigation.",
    )


def extract_historical_evidence(
    row: pd.Series,
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    total = safe_int(
        first_existing(
            row,
            [
                "historical_case_count",
                "total_historical_cases",
            ],
        )
    )

    confirmed = safe_int(
        first_existing(
            row,
            [
                "historical_confirmed_count",
                "confirmed_historical_cases",
            ],
        )
    )

    cleared = safe_int(
        first_existing(
            row,
            [
                "historical_cleared_count",
                "cleared_historical_cases",
            ],
        )
    )

    if total <= 0:
        return evidence

    strength, explanation = historical_strength(
        confirmed_count=confirmed,
        cleared_count=cleared,
        total_count=total,
    )

    patterns = first_existing(
        row,
        [
            "historical_patterns",
            "patterns",
        ],
    )

    exposure = safe_float(
        first_existing(
            row,
            [
                "historical_exposure",
                "total_historical_exposure",
            ],
        )
    )

    report_count = safe_int(
        first_existing(
            row,
            [
                "historical_report_count",
                "report_count",
            ],
        )
    )

    evidence.append(
        {
            "signal_type": "historical_case_history",
            "strength": strength,
            "direction": (
                "supports_fraud"
                if confirmed > 0
                else "context"
            ),
            "independence_group": "historical",
            "source": "historical_case_summary",
            "raw": {
                "historical_case_count": total,
                "historical_confirmed_count": confirmed,
                "historical_cleared_count": cleared,
                "historical_patterns": clean_value(patterns),
                "historical_exposure": exposure,
                "historical_report_count": report_count,
            },
            "explanation": explanation
            + " Historical evidence is support/context only and is "
              "restricted to cases available before the current case opening.",
        }
    )

    if cleared > 0:
        evidence.append(
            {
                "signal_type": "historical_cleared_cases",
                "strength": "medium",
                "direction": "contradicts_fraud",
                "independence_group": "historical",
                "source": "historical_case_summary",
                "raw": {
                    "historical_cleared_count": cleared,
                },
                "explanation": (
                    f"{cleared} prior case(s) were cleared. This is a "
                    "contradictory historical signal and should be considered "
                    "when assessing current evidence."
                ),
            }
        )

    return evidence


# ============================================================================
# PATTERN EVIDENCE
# ============================================================================

def extract_pattern_evidence(
    row: pd.Series,
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    card_testing = safe_bool(
        first_existing(
            row,
            ["card_testing_candidate"],
        )
    )

    cnp = safe_bool(
        first_existing(
            row,
            ["cnp_candidate"],
        )
    )

    cnp_new_device = safe_bool(
        first_existing(
            row,
            ["cnp_new_device_candidate"],
        )
    )

    out_of_region = safe_bool(
        first_existing(
            row,
            ["out_of_region_candidate"],
        )
    )

    ato = safe_bool(
        first_existing(
            row,
            ["account_takeover_candidate"],
        )
    )

    repeated_abuse = safe_bool(
        first_existing(
            row,
            ["repeated_abuse_candidate"],
        )
    )

    small_auth_48h = safe_int(
        first_existing(
            row,
            ["small_auth_count_48h"],
        )
    )

    # ------------------------------------------------------------------
    # Card testing
    # ------------------------------------------------------------------

    if card_testing:

        if small_auth_48h >= 5:
            strength = "strong"
        elif small_auth_48h >= 3:
            strength = "medium"
        else:
            strength = "weak"

        evidence.append(
            {
                "signal_type": "card_testing_pattern",
                "strength": strength,
                "direction": "supports_fraud",
                "independence_group": "behavioral_pattern",
                "source": "benchmark_summary",
                "raw": {
                    "card_testing_candidate": True,
                    "small_auth_count_48h": small_auth_48h,
                },
                "explanation": (
                    f"The transaction sequence contains {small_auth_48h} "
                    "small authorizations within the previous 48 hours. "
                    "This is consistent with the documented card-testing "
                    "candidate pattern but does not independently establish "
                    "fraud."
                ),
            }
        )

    # ------------------------------------------------------------------
    # CNP
    # ------------------------------------------------------------------

    if cnp:

        evidence.append(
            {
                "signal_type": "card_not_present_pattern",
                "strength": "medium",
                "direction": "supports_fraud",
                "independence_group": "behavioral_pattern",
                "source": "benchmark_summary",
                "raw": {
                    "cnp_candidate": True,
                },
                "explanation": (
                    "The flagged transaction and surrounding activity match "
                    "the documented online card-not-present candidate pattern."
                ),
            }
        )

    # ------------------------------------------------------------------
    # CNP + new device
    #
    # This belongs to the SAME behavioral/pattern family as CNP.
    # It must not be counted as a second independent evidence group.
    # ------------------------------------------------------------------

    if cnp_new_device:

        evidence.append(
            {
                "signal_type": "cnp_new_device_pattern",
                "strength": "medium",
                "direction": "supports_fraud",
                "independence_group": "behavioral_pattern",
                "source": "benchmark_summary",
                "raw": {
                    "cnp_new_device_candidate": True,
                },
                "explanation": (
                    "The flagged transaction is an online transaction with "
                    "the documented new-device indicator. This strengthens "
                    "the CNP pattern but is intentionally grouped with the "
                    "same behavioral-pattern evidence family."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Out of region
    # ------------------------------------------------------------------

    if out_of_region:

        evidence.append(
            {
                "signal_type": "out_of_region_pattern",
                "strength": "medium",
                "direction": "supports_fraud",
                "independence_group": "geographic_behavior",
                "source": "benchmark_summary",
                "raw": {
                    "out_of_region_candidate": True,
                },
                "explanation": (
                    "The transaction matches the documented out-of-region "
                    "candidate pattern. The customer's historical modal "
                    "region is treated only as a proxy, not ground truth."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Account takeover
    # ------------------------------------------------------------------

    if ato:

        ato_signal_count = 0

        for column in [
            "channel_novel",
            "product_novel",
            "new_device",
            "id_23_proxy",
            "proxy_indicator",
            "device_match_anomaly",
        ]:
            if column in row.index and safe_bool(row[column]):
                ato_signal_count += 1

        if ato_signal_count >= 2:
            strength = "strong"
        else:
            strength = "medium"

        evidence.append(
            {
                "signal_type": "account_takeover_pattern",
                "strength": strength,
                "direction": "supports_fraud",
                "independence_group": "behavioral_identity",
                "source": "benchmark_summary",
                "raw": {
                    "account_takeover_candidate": True,
                    "supporting_signal_count": ato_signal_count,
                },
                "explanation": (
                    "The transaction behavior contains signals compatible "
                    "with the documented account-takeover candidate pattern. "
                    "The candidate remains an inference and is not itself "
                    "a fraud verdict."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Repeated abuse
    # ------------------------------------------------------------------

    if repeated_abuse:

        confirmed = safe_int(
            first_existing(
                row,
                ["historical_confirmed_count"],
            )
        )

        if confirmed >= 2:
            strength = "strong"
        elif confirmed == 1:
            strength = "medium"
        else:
            strength = "weak"

        evidence.append(
            {
                "signal_type": "repeated_historical_abuse",
                "strength": strength,
                "direction": "supports_fraud",
                "independence_group": "historical",
                "source": "benchmark_summary",
                "raw": {
                    "repeated_abuse_candidate": True,
                    "historical_confirmed_count": confirmed,
                },
                "explanation": (
                    "The customer's/card's prior history contains repeated "
                    "abuse-related cases. This is historical support and is "
                    "not independent current-transaction evidence."
                ),
            }
        )

    return evidence


# ============================================================================
# IDENTITY / DEVICE SIGNALS
# ============================================================================

def extract_identity_evidence(
    row: pd.Series,
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    id_15 = normalize_text(
        first_existing(
            row,
            ["id_15"],
        )
    )

    id_23 = normalize_text(
        first_existing(
            row,
            ["id_23"],
        )
    )

    id_34 = normalize_text(
        first_existing(
            row,
            ["id_34"],
        )
    )

    # New device.
    if id_15 == "new":

        evidence.append(
            {
                "signal_type": "new_device_indicator",
                "strength": "medium",
                "direction": "supports_fraud",
                "independence_group": "identity_device",
                "source": "identity",
                "raw": {
                    "id_15": first_existing(
                        row,
                        ["id_15"],
                    ),
                },
                "explanation": (
                    "The identity record contains id_15=New. This is treated "
                    "as an observed identity/device indicator and not as proof "
                    "of unauthorized activity."
                ),
            }
        )

    # Proxy / network indicator.
    if id_23:

        proxy_like = (
            "proxy" in id_23
            or "anonymous" in id_23
        )

        if proxy_like:

            evidence.append(
                {
                    "signal_type": "proxy_network_indicator",
                    "strength": "medium",
                    "direction": "supports_fraud",
                    "independence_group": "identity_network",
                    "source": "identity",
                    "raw": {
                        "id_23": first_existing(
                            row,
                            ["id_23"],
                        ),
                    },
                    "explanation": (
                        "The identity record contains a proxy/anonymous "
                        "network-related value. The field is preserved as an "
                        "observed signal; its undocumented semantics are not "
                        "expanded beyond the recorded value."
                    ),
                }
            )

    # Match-status field.
    if id_34:

        evidence.append(
            {
                "signal_type": "identity_match_status",
                "strength": "weak",
                "direction": "context",
                "independence_group": "identity_device",
                "source": "identity",
                "raw": {
                    "id_34": first_existing(
                        row,
                        ["id_34"],
                    ),
                },
                "explanation": (
                    "An id_34 match-status value is present. Because the "
                    "field's semantics are not documented, it is retained "
                    "as an opaque observed signal rather than assigned a "
                    "specific fraud meaning."
                ),
            }
        )

    return evidence


# ============================================================================
# BEHAVIORAL EVIDENCE
# ============================================================================

def extract_behavior_evidence(
    row: pd.Series,
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    # Prefer explicit novelty fields. If they are unavailable, derive
    # novelty from *_seen_before using the correct inverse semantics.
    channel_novel_raw = first_existing(
        row,
        ["channel_novel"],
        default=None,
    )
    if channel_novel_raw is not None:
        channel_novel = safe_bool(channel_novel_raw)
        channel_novel_source = "channel_novel"
    else:
        channel_seen_before = first_existing(
            row,
            ["channel_seen_before"],
            default=None,
        )
        channel_novel = (
            not safe_bool(channel_seen_before)
            if channel_seen_before is not None
            else False
        )
        channel_novel_source = "channel_seen_before_inverse"

    product_novel_raw = first_existing(
        row,
        ["product_novel"],
        default=None,
    )
    if product_novel_raw is not None:
        product_novel = safe_bool(product_novel_raw)
        product_novel_source = "product_novel"
    else:
        product_seen_before = first_existing(
            row,
            ["product_seen_before"],
            default=None,
        )
        product_novel = (
            not safe_bool(product_seen_before)
            if product_seen_before is not None
            else False
        )
        product_novel_source = "product_seen_before_inverse"

    post_flagged_activity = safe_int(
        first_existing(
            row,
            [
                "post_flagged_activity_until_case_open",
                "post_flagged_transaction_count",
            ],
        )
    )

    if channel_novel:

        evidence.append(
            {
                "signal_type": "channel_novelty",
                "strength": "weak",
                "direction": "supports_fraud",
                "independence_group": "behavior",
                "source": "benchmark_summary",
                "raw": {
                    "channel_novel": True,
                    "source_field": channel_novel_source,
                },
                "explanation": (
                    "The flagged transaction uses a channel that was not "
                    "previously observed in the analyzed customer history."
                ),
            }
        )

    if product_novel:

        evidence.append(
            {
                "signal_type": "product_novelty",
                "strength": "weak",
                "direction": "supports_fraud",
                "independence_group": "behavior",
                "source": "benchmark_summary",
                "raw": {
                    "product_novel": True,
                    "source_field": product_novel_source,
                },
                "explanation": (
                    "The ProductCD was not previously observed in the "
                    "analyzed customer history."
                ),
            }
        )

    if post_flagged_activity > 0:

        evidence.append(
            {
                "signal_type": "post_flagged_activity",
                "strength": "weak",
                "direction": "context",
                "independence_group": "temporal_behavior",
                "source": "transaction_windows",
                "raw": {
                    "post_flagged_activity_until_case_open": (
                        post_flagged_activity
                    ),
                },
                "explanation": (
                    f"{post_flagged_activity} transaction(s) occurred after "
                    "the flagged transaction but before case opening. This "
                    "signal respects the case-opening cutoff and does not use "
                    "future transactions."
                ),
            }
        )

    return evidence


# ============================================================================
# REGION EVIDENCE
# ============================================================================

def extract_region_evidence(
    row: pd.Series,
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    candidate = safe_bool(
        first_existing(
            row,
            ["out_of_region_candidate"],
        )
    )

    if not candidate:
        return evidence

    is_new_region = safe_bool(
        first_existing(
            row,
            [
                "is_new_region",
                "new_region",
            ],
        )
    )

    home_continues = safe_bool(
        first_existing(
            row,
            [
                "home_activity_continues",
            ],
        )
    )

    current_region = first_existing(
        row,
        [
            "current_region",
        ],
    )

    home_region = first_existing(
        row,
        [
            "modal_home_region_proxy",
        ],
    )

    if is_new_region and home_continues:
        strength = "medium"
    elif is_new_region:
        strength = "weak"
    else:
        strength = "weak"

    evidence.append(
        {
            "signal_type": "out_of_region_behavior",
            "strength": strength,
            "direction": "supports_fraud",
            "independence_group": "geographic_behavior",
            "source": "region_analysis",
            "raw": {
                "current_region": clean_value(current_region),
                "modal_home_region_proxy": clean_value(home_region),
                "is_new_region": is_new_region,
                "home_activity_continues": home_continues,
            },
            "explanation": (
                "The current region differs from the customer's historical "
                "region proxy. The modal historical in-person region is "
                "explicitly treated as a proxy rather than ground truth."
            ),
        }
    )

    return evidence


# ============================================================================
# EMAIL EVIDENCE
# ============================================================================

def extract_email_evidence(
    row: pd.Series,
) -> List[Dict[str, Any]]:

    evidence: List[Dict[str, Any]] = []

    email_novel = safe_bool(
        first_existing(
            row,
            [
                "email_novelty",
                "email_novel",
                "email_new",
            ],
        )
    )

    purchaser_seen = first_existing(
        row,
        [
            "purchaser_email_seen_before",
            "purchaser_email_seen",
        ],
    )

    recipient_seen = first_existing(
        row,
        [
            "recipient_email_seen_before",
            "recipient_email_seen",
        ],
    )

    if email_novel:

        evidence.append(
            {
                "signal_type": "email_novelty",
                "strength": "weak",
                "direction": "context",
                "independence_group": "email",
                "source": "email_analysis",
                "raw": {
                    "email_novelty": True,
                    "purchaser_email_seen_before": clean_value(
                        purchaser_seen
                    ),
                    "recipient_email_seen_before": clean_value(
                        recipient_seen
                    ),
                },
                "explanation": (
                    "Email-related activity appears novel relative to the "
                    "available customer history. This is contextual evidence "
                    "rather than proof of fraud."
                ),
            }
        )

    return evidence


# ============================================================================
# CONTRADICTIONS
# ============================================================================

def build_contradictions(
    row: pd.Series,
    evidence: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    contradictions: List[Dict[str, Any]] = []

    historical_cleared = safe_int(
        first_existing(
            row,
            [
                "historical_cleared_count",
                "cleared_historical_cases",
            ],
        )
    )

    if historical_cleared > 0:

        contradictions.append(
            {
                "type": "historical_cleared_cases",
                "severity": "medium",
                "direction": "contradicts_fraud",
                "independence_group": "historical",
                "explanation": (
                    f"{historical_cleared} prior case(s) involving the "
                    "customer/card were cleared. This does not negate current "
                    "signals but should prevent historical evidence from being "
                    "treated as conclusive."
                ),
            }
        )

    # High-risk score with weak behavioral evidence is NOT a contradiction.
    # It is simply evidence that risk_score is not the same thing as
    # fraud_probability.
    risk_score = safe_float(
        first_existing(
            row,
            ["risk_score"],
        )
    )

    strong_evidence = [
        item
        for item in evidence
        if item.get("strength") == "strong"
    ]

    if (
        risk_score is not None
        and risk_score <= 0.15
        and strong_evidence
    ):
        contradictions.append(
            {
                "type": "risk_score_evidence_mismatch",
                "severity": "low",
                "direction": "contradicts_fraud",
                "independence_group": "risk_score",
                "explanation": (
                    f"risk_score={risk_score:.2f} is low despite the presence "
                    "of strong calibrated evidence. This reinforces that "
                    "risk_score must not be used directly as fraud_probability."
                ),
            }
        )

    if (
        risk_score is not None
        and risk_score >= 0.85
        and not strong_evidence
    ):
        contradictions.append(
            {
                "type": "risk_score_evidence_mismatch",
                "severity": "low",
                "direction": "contradicts_fraud",
                "independence_group": "risk_score",
                "explanation": (
                    f"risk_score={risk_score:.2f} is high while no strong "
                    "calibrated evidence is currently present. The agent "
                    "should investigate rather than equating risk_score "
                    "with fraud_probability."
                ),
            }
        )

    return contradictions


# ============================================================================
# INDEPENDENCE GROUPS
# ============================================================================

def calculate_independence(
    evidence: List[Dict[str, Any]],
) -> Tuple[List[str], int]:

    """
    Count independent evidence families rather than individual signals.

    This prevents correlated observations such as:
        CNP
        CNP + new device

    from being counted as two independent pieces of evidence.
    """

    groups = set()

    for item in evidence:

        strength = item.get("strength")
        direction = item.get("direction")

        if direction != "supports_fraud":
            continue

        if strength not in {
            "strong",
            "medium",
        }:
            continue

        group = item.get("independence_group")

        if group:
            groups.add(group)

    groups_sorted = sorted(groups)

    return groups_sorted, len(groups_sorted)


# ============================================================================
# CASE CALIBRATION
# ============================================================================

def calibrate_case(
    row: pd.Series,
    shared_device_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:

    case_id = first_existing(
        row,
        [
            "case_id",
            "benchmark_case_id",
        ],
    )

    evidence: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # Current transaction / pattern evidence
    # ------------------------------------------------------------

    evidence.extend(
        extract_pattern_evidence(row)
    )

    # ------------------------------------------------------------
    # Identity/device evidence
    # ------------------------------------------------------------

    evidence.extend(
        extract_identity_evidence(row)
    )

    evidence.extend(
        extract_device_evidence(
            row,
            shared_device_df,
        )
    )

    # ------------------------------------------------------------
    # Behavioral evidence
    # ------------------------------------------------------------

    evidence.extend(
        extract_behavior_evidence(row)
    )

    # ------------------------------------------------------------
    # Region
    # ------------------------------------------------------------

    evidence.extend(
        extract_region_evidence(row)
    )

    # ------------------------------------------------------------
    # Email
    # ------------------------------------------------------------

    evidence.extend(
        extract_email_evidence(row)
    )

    # ------------------------------------------------------------
    # Historical evidence
    # ------------------------------------------------------------

    evidence.extend(
        extract_historical_evidence(row)
    )

    # ------------------------------------------------------------
    # Independent evidence groups
    # ------------------------------------------------------------

    independent_groups, independent_count = (
        calculate_independence(evidence)
    )

    # ------------------------------------------------------------
    # Contradictions
    # ------------------------------------------------------------

    contradictions = build_contradictions(
        row,
        evidence,
    )

    # ------------------------------------------------------------
    # Summary counts
    # ------------------------------------------------------------

    strength_counts = {
        "strong": 0,
        "medium": 0,
        "weak": 0,
        "not_material": 0,
    }

    for item in evidence:

        strength = item.get("strength")

        if strength in strength_counts:
            strength_counts[strength] += 1

    # ------------------------------------------------------------
    # Raw benchmark context
    # ------------------------------------------------------------

    raw_context = {
        "case_id": clean_value(case_id),
        "trigger_type": clean_value(
            first_existing(
                row,
                ["trigger_type"],
            )
        ),
        "flagged_txn_id": clean_value(
            first_existing(
                row,
                [
                    "flagged_txn_id",
                    "txn_id",
                    "transaction_id",
                ],
            )
        ),
        "customer_id": clean_value(
            first_existing(
                row,
                ["customer_id"],
            )
        ),
        "card_id": clean_value(
            first_existing(
                row,
                ["card_id"],
            )
        ),
        "opened_at": clean_value(
            first_existing(
                row,
                [
                    "opened_at",
                    "case_opened_at",
                ],
            )
        ),
        "risk_score": clean_value(
            first_existing(
                row,
                ["risk_score"],
            )
        ),
        "amount": clean_value(
            first_existing(
                row,
                [
                    "TransactionAmt",
                    "amount",
                    "transaction_amount",
                ],
            )
        ),
        "channel": clean_value(
            first_existing(
                row,
                ["channel"],
            )
        ),
        "ProductCD": clean_value(
            first_existing(
                row,
                ["ProductCD"],
            )
        ),
    }

    # ------------------------------------------------------------
    # Directional evidence buckets
    # ------------------------------------------------------------

    supporting_evidence = [
        e for e in evidence
        if e.get("direction") == "supports_fraud"
    ]

    contradicting_evidence = [
        e for e in evidence
        if e.get("direction") == "contradicts_fraud"
    ]

    context_evidence = [
        e for e in evidence
        if e.get("direction") == "context"
    ]

    # ------------------------------------------------------------
    # Final calibrated case object
    # ------------------------------------------------------------

    return {
        "case_id": clean_value(case_id),

        "raw_context": raw_context,

        "evidence": evidence,

        "evidence_summary": {
            "strong_count": strength_counts["strong"],
            "medium_count": strength_counts["medium"],
            "weak_count": strength_counts["weak"],
            "not_material_count": strength_counts["not_material"],

            "supporting_count": len(supporting_evidence),
            "contradicting_count": len(contradicting_evidence),
            "context_count": len(context_evidence),

            "independent_evidence_groups": independent_groups,
            "independent_evidence_count": independent_count,

            # Independent evidence is deliberately derived only from
            # fraud-supporting strong/medium evidence.
            "independent_evidence_rule": (
                "direction=supports_fraud AND "
                "strength in {strong,medium} AND "
                "unique independence_group"
            ),
        },

        "contradictions": contradictions,

        # Intentionally null.
        #
        # This script must NOT calculate the final fraud probability.
        "fraud_probability": None,

        "decision": None,

        "stop_reason": None,
    }


# ============================================================================
# CSV FLATTENING
# ============================================================================

def flatten_case_for_csv(
    case: Dict[str, Any],
) -> Dict[str, Any]:

    raw = case["raw_context"]
    summary = case["evidence_summary"]

    evidence = case["evidence"]

    by_type = {
        item["signal_type"]: item
        for item in evidence
    }

    def strength(signal_type: str) -> Optional[str]:

        item = by_type.get(signal_type)

        if item is None:
            return None

        return item.get("strength")

    def direction(signal_type: str) -> Optional[str]:

        item = by_type.get(signal_type)

        if item is None:
            return None

        return item.get("direction")

    return {
        "case_id": case["case_id"],
        "trigger_type": raw.get("trigger_type"),
        "flagged_txn_id": raw.get("flagged_txn_id"),
        "customer_id": raw.get("customer_id"),
        "card_id": raw.get("card_id"),
        "opened_at": raw.get("opened_at"),
        "risk_score": raw.get("risk_score"),
        "amount": raw.get("amount"),
        "channel": raw.get("channel"),
        "ProductCD": raw.get("ProductCD"),

        "card_testing_strength": strength(
            "card_testing_pattern"
        ),
        "card_testing_direction": direction(
            "card_testing_pattern"
        ),
        "cnp_strength": strength(
            "card_not_present_pattern"
        ),
        "cnp_direction": direction(
            "card_not_present_pattern"
        ),
        "cnp_new_device_strength": strength(
            "cnp_new_device_pattern"
        ),
        "cnp_new_device_direction": direction(
            "cnp_new_device_pattern"
        ),
        "out_of_region_strength": strength(
            "out_of_region_pattern"
        ),
        "out_of_region_direction": direction(
            "out_of_region_pattern"
        ),
        "account_takeover_strength": strength(
            "account_takeover_pattern"
        ),
        "account_takeover_direction": direction(
            "account_takeover_pattern"
        ),

        "new_device_strength": strength(
            "new_device_indicator"
        ),
        "new_device_direction": direction(
            "new_device_indicator"
        ),
        "proxy_network_strength": strength(
            "proxy_network_indicator"
        ),
        "proxy_network_direction": direction(
            "proxy_network_indicator"
        ),
        "identity_match_status_strength": strength(
            "identity_match_status"
        ),
        "identity_match_status_direction": direction(
            "identity_match_status"
        ),
        "shared_device_profile_strength": strength(
            "shared_device_profile"
        ),
        "shared_device_profile_direction": direction(
            "shared_device_profile"
        ),

        "channel_novelty_strength": strength(
            "channel_novelty"
        ),
        "channel_novelty_direction": direction(
            "channel_novelty"
        ),
        "product_novelty_strength": strength(
            "product_novelty"
        ),
        "product_novelty_direction": direction(
            "product_novelty"
        ),
        "post_flagged_activity_strength": strength(
            "post_flagged_activity"
        ),
        "post_flagged_activity_direction": direction(
            "post_flagged_activity"
        ),
        "out_of_region_behavior_strength": strength(
            "out_of_region_behavior"
        ),
        "out_of_region_behavior_direction": direction(
            "out_of_region_behavior"
        ),
        "email_novelty_strength": strength(
            "email_novelty"
        ),
        "email_novelty_direction": direction(
            "email_novelty"
        ),

        "historical_case_strength": strength(
            "historical_case_history"
        ),
        "historical_case_direction": direction(
            "historical_case_history"
        ),
        "repeated_historical_abuse_strength": strength(
            "repeated_historical_abuse"
        ),
        "repeated_historical_abuse_direction": direction(
            "repeated_historical_abuse"
        ),
        "historical_cleared_signal": strength(
            "historical_cleared_cases"
        ),

        "strong_evidence_count": summary[
            "strong_count"
        ],
        "medium_evidence_count": summary[
            "medium_count"
        ],
        "weak_evidence_count": summary[
            "weak_count"
        ],
        "not_material_evidence_count": summary[
            "not_material_count"
        ],

        "supporting_evidence_count": summary[
            "supporting_count"
        ],

        "contradicting_evidence_count": summary[
            "contradicting_count"
        ],

        "context_evidence_count": summary[
            "context_count"
        ],
        "independent_evidence_count": summary[
            "independent_evidence_count"
        ],
        "independent_evidence_groups": "|".join(
            summary["independent_evidence_groups"]
        ),
        "contradiction_count": len(
            case["contradictions"]
        ),

        # Explicitly retained for downstream validation.
        "fraud_probability": None,
        "decision": None,
        "stop_reason": None,
    }


# ============================================================================
# NOTES
# ============================================================================

def write_notes(
    cases: List[Dict[str, Any]],
) -> None:

    total = len(cases)

    strength_totals = {
        "strong": 0,
        "medium": 0,
        "weak": 0,
        "not_material": 0,
    }

    independent_distribution: Dict[int, int] = {}

    contradiction_cases = 0
    supporting_total = 0
    contradicting_total = 0
    context_total = 0

    for case in cases:

        summary = case["evidence_summary"]

        for key in strength_totals:
            strength_totals[key] += summary[
                f"{key}_count"
            ]

        supporting_total += summary["supporting_count"]
        contradicting_total += summary["contradicting_count"]
        context_total += summary["context_count"]

        independent_count = summary[
            "independent_evidence_count"
        ]

        independent_distribution[
            independent_count
        ] = independent_distribution.get(
            independent_count,
            0,
        ) + 1

        if case["contradictions"]:
            contradiction_cases += 1

    lines = []

    lines.append(
        "EVIDENCE CALIBRATION NOTES"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        f"Benchmark cases calibrated: {total}"
    )

    lines.append("")

    lines.append(
        "IMPORTANT DESIGN RULES"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "1. This script does not calculate fraud_probability."
    )

    lines.append(
        "2. risk_score is retained only as an input/context field."
    )

    lines.append(
        "3. No final fraud decision is produced."
    )

    lines.append(
        "4. Correlated signals are grouped into evidence families."
    )

    lines.append(
        "5. CNP and CNP+new-device are not counted as two independent "
        "evidence groups."
    )

    lines.append(
        "6. Historical cases are support/context, not proof of current fraud."
    )

    lines.append(
        "7. Cleared historical cases are retained as contradictions/context."
    )

    lines.append(
        "8. Common device profiles are deliberately down-weighted."
    )

    lines.append(
        "9. Exact device-profile matching does not prove physical device "
        "sharing."
    )

    lines.append(
        "10. The case-opening timestamp remains the temporal evidence cutoff."
    )

    lines.append("")

    lines.append(
        "CALIBRATED EVIDENCE COUNTS"
    )

    lines.append(
        "-" * 72
    )

    for key, value in strength_totals.items():

        lines.append(
            f"{key:16s}: {value}"
        )

    lines.append("")

    lines.append(
        "DIRECTIONAL EVIDENCE COUNTS"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"supports_fraud     : {supporting_total}"
    )
    lines.append(
        f"contradicts_fraud  : {contradicting_total}"
    )
    lines.append(
        f"context            : {context_total}"
    )

    lines.append("")

    lines.append(
        "INDEPENDENT EVIDENCE GROUP DISTRIBUTION"
    )

    lines.append(
        "-" * 72
    )

    for count in sorted(independent_distribution):

        lines.append(
            f"{count} independent groups: "
            f"{independent_distribution[count]} case(s)"
        )

    lines.append("")

    lines.append(
        f"Cases with contradictions/context conflicts: "
        f"{contradiction_cases}"
    )

    lines.append("")

    lines.append(
        "NEXT STAGE"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "The downstream investigation agent should combine these calibrated "
        "signals with TigerGraph graph evidence, additional deterministic "
        "investigation, and explicit evidence requests before calculating "
        "fraud_probability and applying stopping conditions."
    )

    OUTPUT_NOTES.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:

    log("=" * 72)
    log("BENCHMARK EVIDENCE CALIBRATION")
    log("=" * 72)

    try:

        validate_inputs()

        # ------------------------------------------------------------
        # Load benchmark summary
        # ------------------------------------------------------------

        benchmark_df = pd.read_csv(
            BENCHMARK_SUMMARY_FILE,
            low_memory=False,
        )

        log(
            f"Benchmark summary: "
            f"{len(benchmark_df):,} rows"
        )

        if benchmark_df.empty:
            raise ValueError(
                "benchmark_summary.csv is empty."
            )

        case_column = find_case_id_column(
            benchmark_df
        )

        if case_column is None:
            raise ValueError(
                "Could not find case_id column in "
                "benchmark_summary.csv.\n"
                f"Available columns: "
                f"{benchmark_df.columns.tolist()}"
            )

        # ------------------------------------------------------------
        # Load optional evidence files
        # ------------------------------------------------------------

        shared_device_df = read_csv_if_exists(
            SHARED_DEVICE_FILE
        )

        historical_df = read_csv_if_exists(
            HISTORICAL_FILE
        )

        # These are currently loaded for visibility/forward compatibility.
        # Most case-level fields should already have been merged into
        # benchmark_summary.csv by benchmark_deep_analysis.py.
        region_df = read_csv_if_exists(
            REGION_FILE
        )

        email_df = read_csv_if_exists(
            EMAIL_FILE
        )

        window_df = read_csv_if_exists(
            WINDOW_FILE
        )

        # Avoid unused-variable confusion while keeping these files
        # available for future calibration expansion.
        _ = (
            historical_df,
            region_df,
            email_df,
            window_df,
        )

        # ------------------------------------------------------------
        # Calibrate cases
        # ------------------------------------------------------------

        calibrated_cases: List[Dict[str, Any]] = []

        for index, row in benchmark_df.iterrows():

            case_id = first_existing(
                row,
                [
                    "case_id",
                    "benchmark_case_id",
                ],
                default=f"UNKNOWN-{index}",
            )

            log(
                f"Calibrating {case_id} "
                f"({index + 1}/{len(benchmark_df)})"
            )

            calibrated = calibrate_case(
                row=row,
                shared_device_df=shared_device_df,
            )

            calibrated_cases.append(
                calibrated
            )

        # ------------------------------------------------------------
        # JSON output
        # ------------------------------------------------------------

        json_payload = {
            "metadata": {
                "script": (
                    "benchmark_evidence_calibration.py"
                ),
                "generated_at": pd.Timestamp.now("UTC").isoformat(),
                "case_count": len(
                    calibrated_cases
                ),
                "fraud_probability_calculated": False,
                "final_decision_calculated": False,
            },
            "calibration_rules": {
                "device_profile": (
                    "DeviceInfo + id_30 + id_31 + id_33"
                ),
                "common_device_profiles_are_downweighted": True,
                "risk_score_is_not_fraud_probability": True,
                "historical_cases_are_support_only": True,
                "historical_cleared_cases_contradict_fraud": True,
                "directional_evidence_schema": True,
                "independence_requires_supporting_strong_or_medium": True,
                "correlated_signals_share_evidence_groups": True,
                "future_transactions_after_case_opening_are_not_added": True,
            },
            "cases": calibrated_cases,
        }

        OUTPUT_JSON.write_text(
            json.dumps(
                json_payload,
                indent=2,
                ensure_ascii=False,
                default=clean_value,
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------------
        # CSV output
        # ------------------------------------------------------------

        csv_rows = [
            flatten_case_for_csv(case)
            for case in calibrated_cases
        ]

        calibrated_df = pd.DataFrame(
            csv_rows
        )

        calibrated_df.to_csv(
            OUTPUT_CSV,
            index=False,
        )

        # ------------------------------------------------------------
        # Notes
        # ------------------------------------------------------------

        write_notes(
            calibrated_cases
        )

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------

        log("")
        log("=" * 72)
        log("CALIBRATION COMPLETE")
        log("=" * 72)

        log(
            f"Cases calibrated: "
            f"{len(calibrated_cases)}"
        )

        log(
            f"JSON: {OUTPUT_JSON}"
        )

        log(
            f"CSV:  {OUTPUT_CSV}"
        )

        log(
            f"Notes: {OUTPUT_NOTES}"
        )

        log("")
        log(
            "IMPORTANT: No fraud probability or final decision "
            "was calculated."
        )

        return 0

    except Exception as exc:

        log("")
        log("=" * 72)
        log("ERROR")
        log("=" * 72)

        log(
            f"{type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())