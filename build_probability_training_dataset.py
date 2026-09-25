#!/usr/bin/env python3
"""
build_probability_training_dataset.py

Build a leakage-safe historical training dataset for the fraud-probability
model used by the TigerGraph Agentic Fraud Investigation project.

This script intentionally:
- streams transactions.csv in chunks; the ~708 MB file stays local.
- uses closed_cases_history.csv as labels only.
- reconstructs the same evidence families/rules used by the benchmark pipeline.
- applies the current historical case's opened_at as the evidence cutoff.
- treats prior cleared cases as contradiction/context, never supporting fraud.
- keeps risk_score separate from fraud_probability.
- does NOT train a model.
- does NOT use the 20 HHG benchmark cases.

Inputs:
    data/transactions.csv
    data/identity.csv
    data/closed_cases_history.csv

Outputs:
    analysis/probability_training_dataset.csv
    analysis/probability_training_summary.json
    analysis/probability_training_validation.txt

Run:
    python build_probability_training_dataset.py

Optional:
    python build_probability_training_dataset.py --chunk-size 100000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "analysis"

TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
IDENTITY_FILE = DATA_DIR / "identity.csv"
CLOSED_CASES_FILE = DATA_DIR / "closed_cases_history.csv"

DEFAULT_CHUNK_SIZE = 100_000

TRANSACTION_COLUMNS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "customer_id",
    "ts",
    "channel",
    "risk_score",
]

IDENTITY_COLUMNS = [
    "TransactionID",
    "id_15",
    "id_23",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "DeviceType",
    "DeviceInfo",
]

DEVICE_PROFILE_COMPONENTS = [
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]

LABEL_MAP = {
    "confirmed_fraud": 1,
    "cleared": 0,
}


# ============================================================================
# LOGGING / BASIC HELPERS
# ============================================================================

def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True,
    )


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass

    return False


def clean_str(value: Any) -> Optional[str]:
    if is_missing(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.lower() in {
        "nan",
        "none",
        "nat",
        "<na>",
    }:
        return None

    return text


def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        if is_missing(value):
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    value = safe_float(value)

    if value is None:
        return default

    return int(value)


def safe_bool(value: Any) -> bool:
    if is_missing(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def parse_pipe_list(value: Any) -> List[str]:
    text = clean_str(value)

    if not text:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def normalize_transaction_ids(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).astype("Int64")


# ============================================================================
# DEVICE PROFILE
# ============================================================================

def build_device_profile(
    row: pd.Series,
) -> Tuple[Optional[str], int]:
    """
    Same documented device profile concept used by the benchmark analysis:

        DeviceInfo + id_30 + id_31 + id_33

    id_15 and id_23 are deliberately excluded.
    """

    values: List[str] = []
    specificity = 0

    for field in DEVICE_PROFILE_COMPONENTS:
        value = clean_str(row.get(field))

        if value is None:
            values.append("")
        else:
            values.append(value)
            specificity += 1

    if specificity == 0:
        return None, 0

    return "|".join(values), specificity


def add_device_profile_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    profiles: List[Optional[str]] = []
    specificities: List[int] = []

    for _, row in df.iterrows():
        profile, specificity = build_device_profile(row)

        profiles.append(profile)
        specificities.append(specificity)

    df["device_profile"] = profiles
    df["device_profile_specificity"] = specificities

    return df


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_historical_cases(
    historical: pd.DataFrame,
) -> List[str]:
    errors: List[str] = []

    required = {
        "case_id",
        "customer_id",
        "opened_at",
        "closed_at",
        "outcome",
        "pattern",
        "first_fraud_txn_id",
        "txn_ids",
        "n_txns",
    }

    missing = sorted(
        required - set(historical.columns)
    )

    if missing:
        errors.append(
            f"Missing historical columns: {missing}"
        )
        return errors

    if historical["case_id"].isna().any():
        errors.append(
            "Historical cases contain null case_id."
        )

    if historical["case_id"].astype(str).duplicated().any():
        errors.append(
            "Duplicate historical case_id values found."
        )

    actual_outcomes = set(
        historical["outcome"]
        .dropna()
        .astype(str)
    )

    unexpected_outcomes = sorted(
        actual_outcomes - set(LABEL_MAP)
    )

    if unexpected_outcomes:
        errors.append(
            f"Unexpected outcome values: "
            f"{unexpected_outcomes}"
        )

    opened = pd.to_datetime(
        historical["opened_at"],
        errors="coerce",
    )

    closed = pd.to_datetime(
        historical["closed_at"],
        errors="coerce",
    )

    if opened.isna().any():
        errors.append(
            f"{int(opened.isna().sum())} cases have "
            "invalid opened_at."
        )

    if closed.isna().any():
        errors.append(
            f"{int(closed.isna().sum())} cases have "
            "invalid closed_at."
        )

    invalid_order = (
        (closed < opened)
        & opened.notna()
        & closed.notna()
    )

    if invalid_order.any():
        errors.append(
            "At least one historical case has "
            "closed_at < opened_at."
        )

    declared_n = pd.to_numeric(
        historical["n_txns"],
        errors="coerce",
    )

    actual_n = historical["txn_ids"].apply(
        lambda value: len(parse_pipe_list(value))
    )

    mismatch = (
        declared_n.notna()
        & (
            declared_n.astype("Int64")
            != actual_n.astype("Int64")
        )
    )

    if mismatch.any():
        errors.append(
            f"{int(mismatch.sum())} cases have "
            "n_txns != number of txn_ids."
        )

    return errors


def load_historical_cases(
    path: Path,
) -> pd.DataFrame:
    log(
        f"Loading historical cases: {path}"
    )

    historical = pd.read_csv(
        path,
        low_memory=False,
    )

    historical["case_id"] = (
        historical["case_id"]
        .astype("string")
    )

    historical["customer_id"] = (
        historical["customer_id"]
        .astype("string")
    )

    historical["opened_at"] = pd.to_datetime(
        historical["opened_at"],
        errors="coerce",
    )

    historical["closed_at"] = pd.to_datetime(
        historical["closed_at"],
        errors="coerce",
    )

    historical["outcome"] = (
        historical["outcome"]
        .astype("string")
    )

    historical["pattern"] = (
        historical["pattern"]
        .astype("string")
    )

    errors = validate_historical_cases(
        historical
    )

    if errors:
        raise ValueError(
            "Historical case validation failed:\n"
            + "\n".join(
                f"  - {error}"
                for error in errors
            )
        )

    log(
        f"Historical cases: "
        f"{len(historical):,}"
    )

    class_distribution = historical["outcome"].value_counts(dropna=False).to_dict()

    log(
        "Historical class distribution: "
        f"{class_distribution}"
    )

    return historical


def load_identity(
    path: Path,
) -> pd.DataFrame:
    log(
        f"Loading identity: {path}"
    )

    identity = pd.read_csv(
        path,
        usecols=lambda column: (
            column in IDENTITY_COLUMNS
        ),
        low_memory=False,
    )

    identity["TransactionID"] = (
        normalize_transaction_ids(
            identity["TransactionID"]
        )
    )

    identity = (
        identity
        .dropna(subset=["TransactionID"])
        .drop_duplicates(
            subset=["TransactionID"],
            keep="first",
        )
    )

    log(
        f"Identity rows after deduplication: "
        f"{len(identity):,}"
    )

    return identity


# ============================================================================
# HISTORICAL CASE ANCHOR TRANSACTION
# ============================================================================

def build_case_metadata(
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Choose the transaction around which current-case evidence is reconstructed.

    Preference:
      1. first_fraud_txn_id when supplied.
      2. first txn_ids entry otherwise.

    The current case outcome is NOT used as a feature.

    For cleared cases, the dataset does not provide a separate flagged
    transaction field, so the first txn_ids entry is used as the deterministic
    anchor and the selection method is recorded.
    """

    rows: List[Dict[str, Any]] = []

    for _, case in historical.iterrows():
        case_id = clean_str(
            case["case_id"]
        )

        outcome = clean_str(
            case["outcome"]
        )

        if (
            case_id is None
            or outcome not in LABEL_MAP
        ):
            continue

        first_fraud_txn = clean_str(
            case.get(
                "first_fraud_txn_id"
            )
        )

        txn_ids = parse_pipe_list(
            case.get("txn_ids")
        )

        if first_fraud_txn:
            anchor_raw = first_fraud_txn
            selection = (
                "first_fraud_txn_id"
            )
        elif txn_ids:
            anchor_raw = txn_ids[0]
            selection = "first_txn_ids"
        else:
            anchor_raw = None
            selection = "missing"

        anchor = safe_int(
            anchor_raw,
            default=-1,
        )

        if anchor <= 0:
            anchor_value = None
        else:
            anchor_value = anchor

        rows.append(
            {
                "case_id": case_id,
                "label": LABEL_MAP[outcome],
                "outcome": outcome,
                "customer_id": clean_str(
                    case["customer_id"]
                ),
                "opened_at": case["opened_at"],
                "closed_at": case["closed_at"],
                "target_txn_id": anchor_value,
                "target_selection": selection,
                "declared_n_txns": safe_int(
                    case.get("n_txns"),
                    default=0,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# TRANSACTION STREAMING
# ============================================================================

def scan_customer_history(
    path: Path,
    customer_ids: set[str],
    target_txn_ids: set[int],
    max_opened_at: pd.Timestamp,
    chunk_size: int,
) -> pd.DataFrame:
    """
    Stream transactions.csv.

    Retain:
      - transactions for customers with historical cases, through the latest
        historical case opening time;
      - target transactions even when their timestamp is after case opening,
        so temporal leakage can be detected explicitly.

    Per-case feature functions later apply the exact case-level cutoff.
    """

    log(
        f"Streaming transactions for "
        f"{len(customer_ids):,} historical customers..."
    )

    chunks: List[pd.DataFrame] = []

    rows_scanned = 0
    rows_retained = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=lambda column: (
                column in TRANSACTION_COLUMNS
            ),
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        rows_scanned += len(chunk)

        chunk["TransactionID"] = (
            normalize_transaction_ids(
                chunk["TransactionID"]
            )
        )

        chunk["ts"] = pd.to_datetime(
            chunk["ts"],
            errors="coerce",
        )

        chunk["customer_id"] = (
            chunk["customer_id"]
            .astype("string")
        )

        customer_mask = (
            chunk["customer_id"]
            .astype(str)
            .isin(customer_ids)
        )

        target_mask = (
            chunk["TransactionID"]
            .isin(target_txn_ids)
        )

        within_global_cutoff = (
            chunk["ts"].isna()
            | (
                chunk["ts"]
                <= max_opened_at
            )
        )

        keep = (
            (
                customer_mask
                & within_global_cutoff
            )
            | target_mask
        )

        retained = chunk.loc[keep].copy()

        if not retained.empty:
            chunks.append(retained)
            rows_retained += len(retained)

        if chunk_no % 5 == 0:
            log(
                f"Transaction scan: "
                f"chunks={chunk_no:,} "
                f"rows_scanned={rows_scanned:,} "
                f"rows_retained={rows_retained:,}"
            )

    if not chunks:
        return pd.DataFrame(
            columns=TRANSACTION_COLUMNS
        )

    history = pd.concat(
        chunks,
        ignore_index=True,
    )

    history = (
        history
        .drop_duplicates(
            subset=["TransactionID"],
            keep="first",
        )
        .sort_values(
            [
                "customer_id",
                "ts",
                "TransactionID",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    log(
        f"Retained transaction history rows: "
        f"{len(history):,}"
    )

    return history


def join_identity(
    transactions: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:
    joined = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        suffixes=("", "_identity"),
    )

    return add_device_profile_columns(
        joined
    )


# ============================================================================
# TEMPORAL HELPERS
# ============================================================================

def prior_window(
    customer_history: pd.DataFrame,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    if pd.isna(flagged_ts):
        return customer_history.iloc[0:0].copy()

    upper = (
        case_opened_at
        if not pd.isna(case_opened_at)
        else flagged_ts
    )

    lower = (
        flagged_ts
        - pd.Timedelta(days=days)
    )

    return customer_history.loc[
        (
            customer_history["ts"]
            >= lower
        )
        & (
            customer_history["ts"]
            <= upper
        )
    ].copy()


def prior_before_flag(
    customer_history: pd.DataFrame,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
) -> pd.DataFrame:
    if pd.isna(flagged_ts):
        return customer_history.iloc[0:0].copy()

    upper = (
        case_opened_at
        if not pd.isna(case_opened_at)
        else flagged_ts
    )

    return customer_history.loc[
        (
            customer_history["ts"]
            < flagged_ts
        )
        & (
            customer_history["ts"]
            <= upper
        )
    ].copy()


def activity_after_flag(
    customer_history: pd.DataFrame,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
) -> pd.DataFrame:
    if pd.isna(flagged_ts):
        return customer_history.iloc[0:0].copy()

    upper = (
        case_opened_at
        if not pd.isna(case_opened_at)
        else flagged_ts
    )

    return customer_history.loc[
        (
            customer_history["ts"]
            > flagged_ts
        )
        & (
            customer_history["ts"]
            <= upper
        )
    ].copy()


# ============================================================================
# DEVICE NETWORK
# ============================================================================

def build_device_customer_index(
    identity: pd.DataFrame,
    target_profiles: set[str],
    transactions_file: Path,
    chunk_size: int,
) -> Dict[
    str,
    Dict[str, List[pd.Timestamp]],
]:
    """
    Build:

        device_profile
            -> customer_id
                -> timestamps

    for all customers, but only for profiles that occur on historical-case
    anchor transactions.

    This second streaming pass is necessary to detect cross-customer device
    profile reuse outside the historical-case customer population.
    """

    if not target_profiles:
        return {}

    log(
        "Scanning all transactions for "
        f"{len(target_profiles):,} relevant device profiles..."
    )

    output: Dict[
        str,
        Dict[str, List[pd.Timestamp]],
    ] = {
        profile: {}
        for profile in target_profiles
    }

    identity_subset = identity[
        [
            "TransactionID",
            "id_30",
            "id_31",
            "id_33",
            "DeviceInfo",
        ]
    ].copy()

    rows_scanned = 0

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            transactions_file,
            usecols=lambda column: (
                column
                in {
                    "TransactionID",
                    "customer_id",
                    "ts",
                }
            ),
            chunksize=chunk_size,
            low_memory=False,
        ),
        start=1,
    ):
        rows_scanned += len(chunk)

        chunk["TransactionID"] = (
            normalize_transaction_ids(
                chunk["TransactionID"]
            )
        )

        chunk["ts"] = pd.to_datetime(
            chunk["ts"],
            errors="coerce",
        )

        chunk["customer_id"] = (
            chunk["customer_id"]
            .astype("string")
        )

        joined = chunk.merge(
            identity_subset,
            on="TransactionID",
            how="left",
        )

        joined = add_device_profile_columns(
            joined
        )

        joined = joined[
            joined["device_profile"].isin(
                target_profiles
            )
        ]

        if not joined.empty:
            for profile, profile_df in joined.groupby(
                "device_profile",
                sort=False,
            ):
                for customer_id, customer_df in profile_df.groupby(
                    profile_df["customer_id"].astype(str),
                    sort=False,
                ):
                    times = (
                        pd.to_datetime(
                            customer_df["ts"],
                            errors="coerce",
                        )
                        .dropna()
                        .tolist()
                    )

                    if times:
                        output.setdefault(
                            profile,
                            {},
                        ).setdefault(
                            str(customer_id),
                            [],
                        ).extend(times)

        if chunk_no % 5 == 0:
            log(
                f"Device scan: "
                f"chunks={chunk_no:,} "
                f"rows_scanned={rows_scanned:,}"
            )

    for profile in output:
        for customer_id in output[profile]:
            output[profile][customer_id] = sorted(
                set(
                    output[profile][customer_id]
                )
            )

    return output


def count_profile_activity_before_cutoff(
    profile_index: Dict[str, List[pd.Timestamp]],
    cutoff: pd.Timestamp,
) -> int:
    """Count profile activity timestamps at or before the cutoff.

    profile_index is shaped as:
        customer_id -> [timestamps]

    The previous implementation iterated over profile_index directly, which
    yields customer_id strings rather than timestamps and causes comparisons
    such as ``customer_id <= Timestamp``.
    """
    if not profile_index:
        return 0

    if pd.isna(cutoff):
        return sum(
            len(timestamps)
            for timestamps in profile_index.values()
        )

    return sum(
        1
        for timestamps in profile_index.values()
        for timestamp in timestamps
        if not pd.isna(timestamp)
        and pd.Timestamp(timestamp) <= cutoff
    )


# ============================================================================
# HISTORICAL EVIDENCE STRENGTH
# ============================================================================

def historical_strength(
    confirmed_count: int,
    cleared_count: int,
    total_count: int,
) -> Optional[str]:
    """
    Mirrors historical_strength() in benchmark_evidence_calibration.py.

    The direction is handled separately because cleared history is
    contradictory evidence.
    """

    if total_count <= 0:
        return None

    if (
        confirmed_count >= 2
        and cleared_count == 0
    ):
        return "strong"

    if (
        confirmed_count >= 2
        and cleared_count > 0
    ):
        return "strong"

    if (
        confirmed_count == 1
        and cleared_count == 0
    ):
        return "medium"

    if (
        confirmed_count >= 1
        and cleared_count >= 1
    ):
        return "medium"

    if (
        confirmed_count == 0
        and cleared_count > 0
    ):
        return "weak"

    return "weak"


# ============================================================================
# CASE FEATURE CONSTRUCTION
# ============================================================================

def calculate_case_features(
    case: pd.Series,
    transaction_history: pd.DataFrame,
    historical_cases: pd.DataFrame,
    device_customer_index: Dict[
        str,
        Dict[str, List[pd.Timestamp]],
    ],
) -> Dict[str, Any]:
    case_id = clean_str(
        case["case_id"]
    )

    customer_id = clean_str(
        case["customer_id"]
    )

    opened_at = pd.to_datetime(
        case["opened_at"],
        errors="coerce",
    )

    closed_at = pd.to_datetime(
        case["closed_at"],
        errors="coerce",
    )

    target_txn_id = safe_int(
        case["target_txn_id"],
        default=-1,
    )

    if not case_id:
        raise ValueError(
            "Historical case has no case_id."
        )

    if not customer_id:
        raise ValueError(
            f"{case_id}: missing customer_id."
        )

    if target_txn_id <= 0:
        raise ValueError(
            f"{case_id}: no usable target "
            "transaction."
        )

    customer_history = (
        transaction_history[
            transaction_history["customer_id"]
            .astype(str)
            == customer_id
        ]
        .copy()
    )

    target = customer_history[
        customer_history["TransactionID"]
        == target_txn_id
    ]

    if target.empty:
        raise ValueError(
            f"{case_id}: target transaction "
            f"{target_txn_id} not found."
        )

    target_row = target.iloc[0]

    flagged_ts = pd.to_datetime(
        target_row["ts"],
        errors="coerce",
    )

    target_after_case_open = bool(
        (
            not pd.isna(flagged_ts)
            and not pd.isna(opened_at)
            and flagged_ts > opened_at
        )
    )

    # ------------------------------------------------------------------
    # Temporal cutoff
    # ------------------------------------------------------------------

    prior = prior_before_flag(
        customer_history,
        flagged_ts,
        opened_at,
    )

    h24 = prior_window(
        customer_history,
        flagged_ts,
        opened_at,
        1,
    )

    h48 = prior_window(
        customer_history,
        flagged_ts,
        opened_at,
        2,
    )

    h7 = prior_window(
        customer_history,
        flagged_ts,
        opened_at,
        7,
    )

    h30 = prior_window(
        customer_history,
        flagged_ts,
        opened_at,
        30,
    )

    post_flag = activity_after_flag(
        customer_history,
        flagged_ts,
        opened_at,
    )

    # ------------------------------------------------------------------
    # Current transaction context
    # ------------------------------------------------------------------

    amount = safe_float(
        target_row.get("TransactionAmt")
    )

    risk_score = safe_float(
        target_row.get("risk_score")
    )

    channel = clean_str(
        target_row.get("channel")
    )

    product = clean_str(
        target_row.get("ProductCD")
    )

    addr1 = clean_str(
        target_row.get("addr1")
    )

    purchaser_email = clean_str(
        target_row.get("P_emaildomain")
    )

    recipient_email = clean_str(
        target_row.get("R_emaildomain")
    )

    # ------------------------------------------------------------------
    # Small authorizations / card testing
    # Mirrors benchmark_deep_analysis.py.
    # ------------------------------------------------------------------

    small_24 = h24.loc[
        (
            h24["channel"]
            .astype(str)
            == "online"
        )
        & (
            pd.to_numeric(
                h24["TransactionAmt"],
                errors="coerce",
            )
            < 5
        )
        & (
            h24["ts"]
            < flagged_ts
        )
    ]

    small_48 = h48.loc[
        (
            h48["channel"]
            .astype(str)
            == "online"
        )
        & (
            pd.to_numeric(
                h48["TransactionAmt"],
                errors="coerce",
            )
            < 5
        )
        & (
            h48["ts"]
            < flagged_ts
        )
    ]

    small_7 = h7.loc[
        (
            h7["channel"]
            .astype(str)
            == "online"
        )
        & (
            pd.to_numeric(
                h7["TransactionAmt"],
                errors="coerce",
            )
            < 5
        )
        & (
            h7["ts"]
            < flagged_ts
        )
    ]

    card_testing = bool(
        len(small_48) >= 3
        and amount is not None
        and amount > 5
    )

    if card_testing:
        if len(small_48) >= 5:
            card_testing_strength = "strong"
        elif len(small_48) >= 3:
            card_testing_strength = "medium"
        else:
            card_testing_strength = "weak"
    else:
        card_testing_strength = None

    # ------------------------------------------------------------------
    # Channel / product novelty
    # ------------------------------------------------------------------

    channel_novel = bool(
        channel
        and not (
            prior["channel"]
            .astype(str)
            == channel
        ).any()
    )

    product_novel = bool(
        product
        and not (
            prior["ProductCD"]
            .astype(str)
            == product
        ).any()
    )

    channel_novelty_strength = (
        "weak"
        if channel_novel
        else None
    )

    product_novelty_strength = (
        "weak"
        if product_novel
        else None
    )

    # ------------------------------------------------------------------
    # Email novelty - context only.
    # ------------------------------------------------------------------

    purchaser_seen = bool(
        purchaser_email
        and (
            prior["P_emaildomain"]
            .astype(str)
            == purchaser_email
        ).any()
    )

    recipient_seen = bool(
        recipient_email
        and (
            prior["R_emaildomain"]
            .astype(str)
            == recipient_email
        ).any()
    )

    email_novelty = bool(
        (
            purchaser_email
            or recipient_email
        )
        and not (
            purchaser_seen
            and recipient_seen
        )
    )

    email_novelty_strength = (
        "weak"
        if email_novelty
        else None
    )

    # ------------------------------------------------------------------
    # Region - exact benchmark logic.
    # ------------------------------------------------------------------

    out_of_region = False
    is_new_region = False
    home_activity_continues = False
    home_region = None

    if (
        channel == "in_person"
        and addr1
    ):
        in_person_prior = prior.loc[
            (
                prior["channel"]
                .astype(str)
                == "in_person"
            )
            & prior["addr1"].notna()
        ].copy()

        if in_person_prior.empty:
            is_new_region = True

        else:
            region_counts = (
                in_person_prior["addr1"]
                .astype(str)
                .value_counts()
            )

            home_region = str(
                region_counts.index[0]
            )

            is_new_region = (
                addr1
                not in set(
                    region_counts.index
                    .astype(str)
                )
            )

            after_flag = post_flag.loc[
                (
                    post_flag["channel"]
                    .astype(str)
                    == "in_person"
                )
                & post_flag["addr1"].notna()
            ]

            home_activity_continues = bool(
                (
                    after_flag["addr1"]
                    .astype(str)
                    == home_region
                ).any()
            )

            out_of_region = bool(
                is_new_region
                and home_activity_continues
            )

    out_region_strength = (
        "medium"
        if out_of_region
        else None
    )

    out_region_behavior_strength = (
        "medium"
        if out_of_region
        else None
    )

    # ------------------------------------------------------------------
    # Identity/device indicators
    # ------------------------------------------------------------------

    id15 = clean_str(
        target_row.get("id_15")
    )

    id23 = clean_str(
        target_row.get("id_23")
    )

    id34 = clean_str(
        target_row.get("id_34")
    )

    new_device = bool(
        id15
        and id15.lower() == "new"
    )

    proxy_indicator = bool(
        id23
        and (
            "proxy" in id23.lower()
            or "anonymous"
            in id23.lower()
        )
    )

    device_match_anomaly = bool(
        id34
    )

    new_device_strength = (
        "medium"
        if new_device
        else None
    )

    proxy_network_strength = (
        "medium"
        if proxy_indicator
        else None
    )

    # ------------------------------------------------------------------
    # CNP / CNP + new device / ATO
    # Mirrors benchmark_deep_analysis.py.
    # ------------------------------------------------------------------

    online_count_48h = int(
        (
            h48["channel"]
            .astype(str)
            == "online"
        ).sum()
    )

    cnp = bool(
        channel == "online"
        and 2 <= online_count_48h <= 4
    )

    cnp_new_device = bool(
        cnp
        and new_device
    )

    mixed_channel = bool(
        (
            h48["channel"]
            .astype(str)
            == "online"
        ).any()
        and (
            h48["channel"]
            .astype(str)
            == "in_person"
        ).any()
    )

    account_takeover = bool(
        mixed_channel
        and (
            new_device
            or proxy_indicator
            or device_match_anomaly
        )
    )

    cnp_strength = (
        "medium"
        if cnp
        else None
    )

    cnp_new_device_strength = (
        "medium"
        if cnp_new_device
        else None
    )

    # Same supporting-signal logic used by evidence calibration.
    ato_signal_count = (
        int(channel_novel)
        + int(product_novel)
        + int(new_device)
        + int(proxy_indicator)
        + int(device_match_anomaly)
    )

    ato_strength = None

    if account_takeover:
        ato_strength = (
            "strong"
            if ato_signal_count >= 2
            else "medium"
        )

    # ------------------------------------------------------------------
    # Exact device-profile network signal
    # ------------------------------------------------------------------

    device_profile = clean_str(
        target_row.get(
            "device_profile"
        )
    )

    device_specificity = safe_int(
        target_row.get(
            "device_profile_specificity"
        ),
        default=0,
    )

    other_device_customers: set[str] = set()

    device_customer_count = 0
    device_transaction_count = 0

    if (
        device_profile
        and device_profile in device_customer_index
    ):
        profile_index = (
            device_customer_index[
                device_profile
            ]
        )

        for cid, timestamps in (
            profile_index.items()
        ):
            valid_count = (
                count_profile_activity_before_cutoff(
                    {str(i): [t] for i, t in enumerate(timestamps)},
                    opened_at,
                )
            )

            if valid_count > 0:
                device_customer_count += 1
                device_transaction_count += (
                    valid_count
                )

                if cid != customer_id:
                    other_device_customers.add(
                        cid
                    )

    shared_device_candidate = bool(
        device_profile
        and device_specificity >= 2
        and len(other_device_customers) > 0
    )

    shared_device_strength = None
    shared_device_direction = None

    if shared_device_candidate:
        if device_specificity <= 1:
            shared_device_strength = (
                "not_material"
            )
        elif device_customer_count > 50:
            shared_device_strength = "weak"
        elif device_customer_count > 10:
            shared_device_strength = "weak"
        elif (
            device_specificity == 4
            and device_customer_count <= 3
        ):
            shared_device_strength = (
                "strong"
            )
        elif device_customer_count <= 3:
            shared_device_strength = (
                "medium"
            )
        else:
            shared_device_strength = "weak"

        shared_device_direction = (
            "supports_fraud"
            if shared_device_strength
            in {"strong", "medium"}
            else "context"
        )

    # ------------------------------------------------------------------
    # Historical evidence.
    #
    # Current case is excluded. Only prior cases with
    # closed_at <= current opened_at are eligible.
    # ------------------------------------------------------------------

    qualifying_history = (
        historical_cases.loc[
            (
                historical_cases[
                    "customer_id"
                ].astype(str)
                == customer_id
            )
            & (
                historical_cases[
                    "case_id"
                ].astype(str)
                != case_id
            )
            & (
                historical_cases[
                    "closed_at"
                ]
                <= opened_at
            )
        ]
        .copy()
    )

    historical_case_count = len(
        qualifying_history
    )

    historical_confirmed_count = int(
        (
            qualifying_history[
                "outcome"
            ].astype(str)
            == "confirmed_fraud"
        ).sum()
    )

    historical_cleared_count = int(
        (
            qualifying_history[
                "outcome"
            ].astype(str)
            == "cleared"
        ).sum()
    )

    historical_case_strength = (
        historical_strength(
            confirmed_count=(
                historical_confirmed_count
            ),
            cleared_count=(
                historical_cleared_count
            ),
            total_count=(
                historical_case_count
            ),
        )
    )

    historical_case_direction = None

    if historical_case_strength:
        historical_case_direction = (
            "supports_fraud"
            if historical_confirmed_count > 0
            else "context"
        )

    historical_cleared_strength = (
        "medium"
        if historical_cleared_count > 0
        else None
    )

    repeated_historical_abuse = (
        historical_confirmed_count >= 2
    )

    repeated_historical_strength = (
        "strong"
        if repeated_historical_abuse
        else None
    )

    # ------------------------------------------------------------------
    # Evidence object-like records.
    #
    # Each tuple is:
    #     signal, strength, direction, independence_group
    #
    # This is deliberately kept equivalent to the calibrated evidence
    # semantics rather than creating a new scoring interpretation.
    # ------------------------------------------------------------------

    evidence: List[
        Tuple[str, str, str, str]
    ] = []

    def add_evidence(
        signal: str,
        strength: Optional[str],
        direction: Optional[str],
        independence_group: str,
    ) -> None:
        if (
            strength
            and direction
        ):
            evidence.append(
                (
                    signal,
                    strength,
                    direction,
                    independence_group,
                )
            )

    # Pattern evidence.
    add_evidence(
        "card_testing",
        card_testing_strength,
        (
            "supports_fraud"
            if card_testing
            else None
        ),
        "behavioral_pattern",
    )

    add_evidence(
        "cnp",
        cnp_strength,
        (
            "supports_fraud"
            if cnp
            else None
        ),
        "behavioral_pattern",
    )

    add_evidence(
        "cnp_new_device",
        cnp_new_device_strength,
        (
            "supports_fraud"
            if cnp_new_device
            else None
        ),
        "behavioral_pattern",
    )

    add_evidence(
        "out_of_region",
        out_region_strength,
        (
            "supports_fraud"
            if out_of_region
            else None
        ),
        "geographic_behavior",
    )

    add_evidence(
        "account_takeover",
        ato_strength,
        (
            "supports_fraud"
            if account_takeover
            else None
        ),
        "behavioral_identity",
    )

    # Identity/device evidence.
    add_evidence(
        "new_device",
        new_device_strength,
        (
            "supports_fraud"
            if new_device
            else None
        ),
        "identity_device",
    )

    add_evidence(
        "proxy_network",
        proxy_network_strength,
        (
            "supports_fraud"
            if proxy_indicator
            else None
        ),
        "identity_network",
    )

    # Behavioral novelty is weak and therefore cannot create an
    # independent supporting group.
    add_evidence(
        "channel_novelty",
        channel_novelty_strength,
        (
            "supports_fraud"
            if channel_novel
            else None
        ),
        "behavioral_pattern",
    )

    add_evidence(
        "product_novelty",
        product_novelty_strength,
        (
            "supports_fraud"
            if product_novel
            else None
        ),
        "behavioral_pattern",
    )

    # Out-of-region behavior is kept in the behavioral-pattern family
    # for compatibility with the validation schema, but it is weak/medium
    # only when the candidate exists.
    add_evidence(
        "out_of_region_behavior",
        out_region_behavior_strength,
        (
            "supports_fraud"
            if out_of_region
            else None
        ),
        "behavioral_pattern",
    )

    # Context-only evidence.
    add_evidence(
        "email_novelty",
        email_novelty_strength,
        (
            "context"
            if email_novelty
            else None
        ),
        "email",
    )

    add_evidence(
        "post_flagged_activity",
        (
            "weak"
            if len(post_flag) > 0
            else None
        ),
        (
            "context"
            if len(post_flag) > 0
            else None
        ),
        "temporal_behavior",
    )

    # Shared device.
    add_evidence(
        "shared_device_profile",
        shared_device_strength,
        shared_device_direction,
        "network",
    )

    # Historical evidence.
    add_evidence(
        "historical_case",
        historical_case_strength,
        historical_case_direction,
        "historical",
    )

    # Cleared history is always contradictory, never supporting.
    if historical_cleared_count > 0:
        add_evidence(
            "historical_cleared_cases",
            historical_cleared_strength,
            "contradicts_fraud",
            "historical",
        )

    # Repeated abuse uses the same historical independence group.
    if repeated_historical_abuse:
        add_evidence(
            "repeated_historical_abuse",
            repeated_historical_strength,
            "supports_fraud",
            "historical",
        )

    # ------------------------------------------------------------------
    # Evidence counts.
    # ------------------------------------------------------------------

    strong_evidence_count = sum(
        strength == "strong"
        for _, strength, _, _ in evidence
    )

    medium_evidence_count = sum(
        strength == "medium"
        for _, strength, _, _ in evidence
    )

    weak_evidence_count = sum(
        strength == "weak"
        for _, strength, _, _ in evidence
    )

    not_material_evidence_count = sum(
        strength == "not_material"
        for _, strength, _, _ in evidence
    )

    supporting_evidence_count = sum(
        direction == "supports_fraud"
        for _, _, direction, _ in evidence
    )

    contradicting_evidence_count = sum(
        direction == "contradicts_fraud"
        for _, _, direction, _ in evidence
    )

    context_evidence_count = sum(
        direction == "context"
        for _, _, direction, _ in evidence
    )

    strong_support_count = sum(
        strength == "strong"
        and direction == "supports_fraud"
        for _, strength, direction, _ in evidence
    )

    medium_support_count = sum(
        strength == "medium"
        and direction == "supports_fraud"
        for _, strength, direction, _ in evidence
    )

    weak_support_count = sum(
        strength == "weak"
        and direction == "supports_fraud"
        for _, strength, direction, _ in evidence
    )

    strong_contradiction_count = sum(
        strength == "strong"
        and direction == "contradicts_fraud"
        for _, strength, direction, _ in evidence
    )

    medium_contradiction_count = sum(
        strength == "medium"
        and direction == "contradicts_fraud"
        for _, strength, direction, _ in evidence
    )

    weak_contradiction_count = sum(
        strength == "weak"
        and direction == "contradicts_fraud"
        for _, strength, direction, _ in evidence
    )

    # ------------------------------------------------------------------
    # Independent evidence.
    #
    # ONLY:
    #   direction=supports_fraud
    #   strength in {strong, medium}
    #   unique independence group
    #
    # Therefore CNP + CNP/new-device = one behavioral_pattern group.
    # Cleared history and context never count.
    # ------------------------------------------------------------------

    independent_groups = sorted(
        {
            group
            for _, strength, direction, group
            in evidence
            if (
                direction
                == "supports_fraud"
                and strength
                in {"strong", "medium"}
            )
        }
    )

    independent_evidence_count = len(
        independent_groups
    )

    # ------------------------------------------------------------------
    # Output row.
    # ------------------------------------------------------------------

    return {
        "case_id": case_id,
        "label": int(case["label"]),
        "outcome": clean_str(
            case["outcome"]
        ),
        "customer_id": customer_id,
        "target_txn_id": target_txn_id,
        "target_selection": clean_str(
            case.get(
                "target_selection"
            )
        ),

        "opened_at": opened_at,
        "closed_at": closed_at,
        "target_ts": flagged_ts,
        "target_after_case_open": (
            target_after_case_open
        ),

        # Risk score is an input feature only.
        "risk_score": risk_score,

        "amount": amount,
        "channel": channel,
        "ProductCD": product,

        # Evidence summary features.
        "strong_evidence_count": (
            strong_evidence_count
        ),
        "medium_evidence_count": (
            medium_evidence_count
        ),
        "weak_evidence_count": (
            weak_evidence_count
        ),
        "not_material_evidence_count": (
            not_material_evidence_count
        ),

        "supporting_evidence_count": (
            supporting_evidence_count
        ),
        "contradicting_evidence_count": (
            contradicting_evidence_count
        ),
        "context_evidence_count": (
            context_evidence_count
        ),

        "strong_support_count": (
            strong_support_count
        ),
        "medium_support_count": (
            medium_support_count
        ),
        "weak_support_count": (
            weak_support_count
        ),

        "strong_contradiction_count": (
            strong_contradiction_count
        ),
        "medium_contradiction_count": (
            medium_contradiction_count
        ),
        "weak_contradiction_count": (
            weak_contradiction_count
        ),

        "independent_evidence_count": (
            independent_evidence_count
        ),
        "independent_evidence_groups": (
            "|".join(
                independent_groups
            )
        ),
        "contradiction_count": (
            contradicting_evidence_count
        ),

        # Candidate flags.
        "card_testing": int(
            card_testing
        ),
        "cnp": int(cnp),
        "cnp_new_device": int(
            cnp_new_device
        ),
        "account_takeover": int(
            account_takeover
        ),
        "out_of_region": int(
            out_of_region
        ),
        "new_device": int(
            new_device
        ),
        "proxy_network": int(
            proxy_indicator
        ),
        "channel_novelty": int(
            channel_novel
        ),
        "product_novelty": int(
            product_novel
        ),
        "out_of_region_behavior": int(
            out_of_region
        ),
        "repeated_historical_abuse": int(
            repeated_historical_abuse
        ),
        "shared_device_profile": int(
            shared_device_candidate
        ),

        # Evidence strengths.
        "card_testing_strength": (
            card_testing_strength
        ),
        "cnp_strength": cnp_strength,
        "cnp_new_device_strength": (
            cnp_new_device_strength
        ),
        "account_takeover_strength": (
            ato_strength
        ),
        "out_of_region_strength": (
            out_region_strength
        ),
        "new_device_strength": (
            new_device_strength
        ),
        "proxy_network_strength": (
            proxy_network_strength
        ),
        "channel_novelty_strength": (
            channel_novelty_strength
        ),
        "product_novelty_strength": (
            product_novelty_strength
        ),
        "out_of_region_behavior_strength": (
            out_region_behavior_strength
        ),
        "email_novelty_strength": (
            email_novelty_strength
        ),
        "post_flagged_activity_strength": (
            "weak"
            if len(post_flag) > 0
            else None
        ),
        "shared_device_profile_strength": (
            shared_device_strength
        ),
        "historical_case_strength": (
            historical_case_strength
        ),
        "historical_cleared_signal": (
            historical_cleared_strength
        ),
        "repeated_historical_abuse_strength": (
            repeated_historical_strength
        ),

        # Historical context features.
        "historical_case_count": (
            historical_case_count
        ),
        "historical_confirmed_count": (
            historical_confirmed_count
        ),
        "historical_cleared_count": (
            historical_cleared_count
        ),

        # Device-network features.
        "shared_device_customer_count": (
            device_customer_count
            if shared_device_candidate
            else 0
        ),
        "shared_device_other_customer_count": (
            len(other_device_customers)
            if shared_device_candidate
            else 0
        ),
        "shared_device_transaction_count": (
            device_transaction_count
            if shared_device_candidate
            else 0
        ),
        "device_profile_specificity": (
            device_specificity
        ),

        # Temporal features.
        "txn_count_24h": len(h24),
        "txn_count_48h": len(h48),
        "txn_count_7d": len(h7),
        "txn_count_30d": len(h30),
        "online_count_24h": int(
            (
                h24["channel"]
                .astype(str)
                == "online"
            ).sum()
        ),
        "online_count_48h": online_count_48h,
        "online_count_7d": int(
            (
                h7["channel"]
                .astype(str)
                == "online"
            ).sum()
        ),
        "in_person_count_7d": int(
            (
                h7["channel"]
                .astype(str)
                == "in_person"
            ).sum()
        ),
        "small_auth_count_24h": len(
            small_24
        ),
        "small_auth_count_48h": len(
            small_48
        ),
        "small_auth_count_7d": len(
            small_7
        ),
        "post_flagged_activity_until_case_open": (
            len(post_flag)
        ),

        # Region features.
        "is_new_region": int(
            is_new_region
        ),
        "home_activity_continues": int(
            home_activity_continues
        ),
        "modal_home_region_proxy": (
            home_region
        ),

        # Email context.
        "purchaser_email_seen_before": int(
            purchaser_seen
        ),
        "recipient_email_seen_before": int(
            recipient_seen
        ),
        "email_novelty": int(
            email_novelty
        ),

        # Explicit cutoff marker.
        "feature_cutoff": opened_at,
    }


# ============================================================================
# TRAINING DATASET VALIDATION
# ============================================================================

MODEL_FEATURE_COLUMNS = [
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


EXPECTED_INDEPENDENCE_GROUPS = {
    "card_testing": "behavioral_pattern",
    "cnp": "behavioral_pattern",
    "cnp_new_device": "behavioral_pattern",
    "out_of_region": "geographic_behavior",
    "account_takeover": "behavioral_identity",
    "new_device": "identity_device",
    "proxy_network": "identity_network",
    "channel_novelty": "behavioral_pattern",
    "product_novelty": "behavioral_pattern",
    "out_of_region_behavior": "behavioral_pattern",
    "historical_case": "historical",
    "repeated_historical_abuse": "historical",
    "shared_device_profile": "network",
}


def validate_training_dataset(
    dataset: pd.DataFrame,
    historical: pd.DataFrame,
) -> Dict[str, Any]:
    validation: Dict[str, Any] = {}

    validation["row_count"] = int(
        len(dataset)
    )

    validation["expected_historical_cases"] = int(
        len(historical)
    )

    validation["row_count_matches_cases"] = bool(
        len(dataset)
        == len(historical)
    )

    validation["duplicate_case_ids"] = int(
        dataset["case_id"]
        .duplicated()
        .sum()
    )

    validation["duplicate_target_txn_ids"] = int(
        dataset["target_txn_id"]
        .duplicated()
        .sum()
    )

    label_counts = (
        dataset["label"]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    validation["label_distribution"] = {
        str(key): int(value)
        for key, value in label_counts.items()
    }

    expected_labels = {
        "0": int(
            (
                historical["outcome"]
                == "cleared"
            ).sum()
        ),
        "1": int(
            (
                historical["outcome"]
                == "confirmed_fraud"
            ).sum()
        ),
    }

    validation["expected_label_distribution"] = (
        expected_labels
    )

    validation["label_distribution_matches"] = bool(
        validation[
            "label_distribution"
        ].get("0", 0)
        == expected_labels["0"]
        and
        validation[
            "label_distribution"
        ].get("1", 0)
        == expected_labels["1"]
    )

    validation["null_case_ids"] = int(
        dataset["case_id"].isna().sum()
    )

    validation["null_labels"] = int(
        dataset["label"].isna().sum()
    )

    validation["null_customers"] = int(
        dataset["customer_id"].isna().sum()
    )

    validation["null_target_txn"] = int(
        dataset["target_txn_id"].isna().sum()
    )

    validation["target_after_case_open_count"] = int(
        dataset["target_after_case_open"]
        .sum()
    )

    validation[
        "target_after_case_open_is_zero"
    ] = bool(
        validation[
            "target_after_case_open_count"
        ]
        == 0
    )

    # ------------------------------------------------------------------
    # Feature availability.
    # ------------------------------------------------------------------

    missing_model_features = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in dataset.columns
    ]

    validation[
        "missing_model_features"
    ] = missing_model_features

    if not missing_model_features:
        numeric = dataset[
            MODEL_FEATURE_COLUMNS
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        validation[
            "model_feature_null_counts"
        ] = {
            column: int(
                numeric[column]
                .isna()
                .sum()
            )
            for column in MODEL_FEATURE_COLUMNS
        }

        validation[
            "model_feature_nonfinite_counts"
        ] = {
            column: int(
                (
                    ~np.isfinite(
                        numeric[column]
                        .fillna(0)
                        .to_numpy()
                    )
                ).sum()
            )
            for column in MODEL_FEATURE_COLUMNS
        }

    # ------------------------------------------------------------------
    # Evidence consistency.
    # ------------------------------------------------------------------

    independent_rule_violations = 0

    for _, row in dataset.iterrows():
        groups = set()

        for signal, group in (
            EXPECTED_INDEPENDENCE_GROUPS.items()
        ):
            strength = clean_str(
                row.get(
                    f"{signal}_strength"
                )
            )

            if signal == "shared_device_profile":
                # shared_device_profile has no standalone boolean feature;
                # its supporting status is represented by its strength.
                supports = (
                    strength
                    in {"strong", "medium"}
                )

            elif signal == "historical_case":
                # historical_case is also not a boolean model feature.
                # A prior confirmed case supports fraud only when the
                # historical evidence itself is strong/medium. Prior
                # cleared cases are contradiction/context and must NOT
                # create an independent supporting group.
                confirmed_count = safe_int(
                    row.get("historical_confirmed_count"),
                    default=0,
                )
                supports = (
                    confirmed_count > 0
                    and strength
                    in {"strong", "medium"}
                )

            else:
                supports = safe_bool(
                    row.get(signal)
                )

            if (
                supports
                and strength
                in {"strong", "medium"}
            ):
                groups.add(group)

        expected_count = len(groups)

        actual_count = safe_int(
            row.get(
                "independent_evidence_count"
            ),
            default=0,
        )

        if expected_count != actual_count:
            independent_rule_violations += 1

    validation[
        "independent_rule_violations"
    ] = independent_rule_violations

    # ------------------------------------------------------------------
    # Cleared-history semantics.
    # ------------------------------------------------------------------

    cleared_semantic_violations = 0

    for _, row in dataset.iterrows():
        cleared_count = safe_int(
            row.get(
                "historical_cleared_count"
            ),
            default=0,
        )

        cleared_signal = clean_str(
            row.get(
                "historical_cleared_signal"
            )
        )

        if (
            cleared_count > 0
            and cleared_signal != "medium"
        ):
            cleared_semantic_violations += 1

    validation[
        "cleared_history_semantic_violations"
    ] = cleared_semantic_violations

    # ------------------------------------------------------------------
    # Historical temporal cutoff re-check.
    #
    # Reconstruct against source data rather than trusting generated fields.
    # ------------------------------------------------------------------

    historical_cutoff_violations = 0

    for _, row in dataset.iterrows():
        case_id = str(
            row["case_id"]
        )

        opened_at = pd.to_datetime(
            row["opened_at"],
            errors="coerce",
        )

        if pd.isna(opened_at):
            historical_cutoff_violations += 1
            continue

        customer_id = str(
            row["customer_id"]
        )

        prior_cases = historical.loc[
            (
                historical[
                    "customer_id"
                ].astype(str)
                == customer_id
            )
            & (
                historical[
                    "case_id"
                ].astype(str)
                != case_id
            )
            & (
                historical[
                    "closed_at"
                ]
                <= opened_at
            )
        ]

        # The generated count must exactly equal the source-derived count.
        generated_count = safe_int(
            row.get(
                "historical_case_count"
            ),
            default=-1,
        )

        if generated_count != len(
            prior_cases
        ):
            historical_cutoff_violations += 1

    validation[
        "historical_cutoff_violations"
    ] = historical_cutoff_violations

    # ------------------------------------------------------------------
    # Leakage: current outcome must not appear as a feature.
    # ------------------------------------------------------------------

    forbidden_feature_columns = {
        "outcome",
        "pattern",
        "actions_taken",
        "report_filed",
        "analyst_notes",
        "exposure_usd",
        "connected_card_ids",
        "first_fraud_txn_id",
    }

    model_forbidden_present = sorted(
        forbidden_feature_columns
        & set(
            MODEL_FEATURE_COLUMNS
        )
    )

    validation[
        "forbidden_target_leakage_features"
    ] = model_forbidden_present

    validation[
        "forbidden_target_leakage_ok"
    ] = not bool(
        model_forbidden_present
    )

    # ------------------------------------------------------------------
    # Final status.
    # ------------------------------------------------------------------

    validation[
        "overall_pass"
    ] = all([
        validation[
            "row_count_matches_cases"
        ],
        validation[
            "duplicate_case_ids"
        ] == 0,
        validation[
            "label_distribution_matches"
        ],
        validation[
            "null_case_ids"
        ] == 0,
        validation[
            "null_labels"
        ] == 0,
        validation[
            "null_customers"
        ] == 0,
        validation[
            "null_target_txn"
        ] == 0,
        validation[
            "target_after_case_open_is_zero"
        ],
        not missing_model_features,
        validation[
            "independent_rule_violations"
        ] == 0,
        validation[
            "cleared_history_semantic_violations"
        ] == 0,
        validation[
            "historical_cutoff_violations"
        ] == 0,
        validation[
            "forbidden_target_leakage_ok"
        ],
    ])

    return validation


# ============================================================================
# OUTPUT
# ============================================================================

def json_default(value: Any) -> Any:
    if isinstance(
        value,
        (
            pd.Timestamp,
            np.datetime64,
        ),
    ):
        return pd.Timestamp(
            value
        ).isoformat()

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    raise TypeError(
        f"Not JSON serializable: "
        f"{type(value).__name__}"
    )


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )


def write_validation_notes(
    path: Path,
    summary: Dict[str, Any],
) -> None:
    validation = summary[
        "validation"
    ]

    lines = [
        "PROBABILITY TRAINING DATASET VALIDATION",
        "=" * 72,
        "",
        f"OVERALL PASS: "
        f"{validation['overall_pass']}",
        "",
        "DATASET",
        "-" * 72,
        f"Rows: "
        f"{validation['row_count']:,}",
        f"Expected historical cases: "
        f"{validation['expected_historical_cases']:,}",
        f"Row count matches cases: "
        f"{validation['row_count_matches_cases']}",
        "",
        "CLASS DISTRIBUTION",
        "-" * 72,
        f"Observed: "
        f"{validation['label_distribution']}",
        f"Expected: "
        f"{validation['expected_label_distribution']}",
        f"Matches: "
        f"{validation['label_distribution_matches']}",
        "",
        "LEAKAGE / TEMPORAL CHECKS",
        "-" * 72,
        f"Target transaction after case opening: "
        f"{validation['target_after_case_open_count']}",
        f"Target cutoff check: "
        f"{validation['target_after_case_open_is_zero']}",
        f"Historical cutoff violations: "
        f"{validation['historical_cutoff_violations']}",
        f"Forbidden target leakage features: "
        f"{validation['forbidden_target_leakage_features']}",
        "",
        "DUPLICATES",
        "-" * 72,
        f"Duplicate case IDs: "
        f"{validation['duplicate_case_ids']}",
        f"Duplicate target transaction IDs: "
        f"{validation['duplicate_target_txn_ids']}",
        "",
        "FEATURE VALIDATION",
        "-" * 72,
        f"Missing model features: "
        f"{validation['missing_model_features']}",
        f"Independent evidence rule violations: "
        f"{validation['independent_rule_violations']}",
        f"Cleared-history semantic violations: "
        f"{validation['cleared_history_semantic_violations']}",
        "",
        "DESIGN RULES",
        "-" * 72,
        "1. risk_score is an input feature, not fraud_probability.",
        "2. Current historical-case outcome is the label only.",
        "3. Prior historical cases require closed_at <= current opened_at.",
        "4. Cleared historical cases are contradiction/context, never support.",
        "5. Independent evidence requires supporting direction + strong/medium strength + unique group.",
        "6. CNP and CNP+new-device share behavioral_pattern.",
        "7. Weak/context evidence cannot create an independent supporting group.",
        "8. The 20 HHG benchmark cases are not included.",
        "",
        "IMPORTANT",
        "-" * 72,
        "Do not fit the probability model if OVERALL PASS is False.",
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "transactions.csv streaming chunk size "
            "(default: 100000)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory.",
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_files = [
        TRANSACTIONS_FILE,
        IDENTITY_FILE,
        CLOSED_CASES_FILE,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        print(
            "ERROR: required files missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            ),
            file=sys.stderr,
        )
        return 1

    if args.chunk_size <= 0:
        print(
            "ERROR: --chunk-size must be > 0.",
            file=sys.stderr,
        )
        return 1

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. Historical cases
    # ------------------------------------------------------------------

    historical = load_historical_cases(
        CLOSED_CASES_FILE
    )

    case_metadata = build_case_metadata(
        historical
    )

    if case_metadata.empty:
        print(
            "ERROR: no historical case metadata "
            "could be constructed.",
            file=sys.stderr,
        )
        return 2

    missing_anchor_count = int(
        case_metadata[
            "target_txn_id"
        ].isna().sum()
    )

    if missing_anchor_count:
        log(
            "WARNING: "
            f"{missing_anchor_count:,} historical cases "
            "have no usable anchor transaction."
        )

    # ------------------------------------------------------------------
    # 2. Customer / cutoff sets
    # ------------------------------------------------------------------

    customer_ids = set(
        historical[
            "customer_id"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    target_txn_ids = set(
        case_metadata[
            "target_txn_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    max_opened_at = pd.to_datetime(
        historical["opened_at"],
        errors="coerce",
    ).max()

    # ------------------------------------------------------------------
    # 3. Identity
    # ------------------------------------------------------------------

    identity = load_identity(
        IDENTITY_FILE
    )

    # ------------------------------------------------------------------
    # 4. Stream transactions.csv
    # ------------------------------------------------------------------

    transaction_history = (
        scan_customer_history(
            path=TRANSACTIONS_FILE,
            customer_ids=customer_ids,
            target_txn_ids=target_txn_ids,
            max_opened_at=max_opened_at,
            chunk_size=args.chunk_size,
        )
    )

    if transaction_history.empty:
        print(
            "ERROR: no relevant transactions "
            "were found.",
            file=sys.stderr,
        )
        return 2

    transaction_history = join_identity(
        transaction_history,
        identity,
    )

    # ------------------------------------------------------------------
    # 5. Verify anchor transaction availability
    # ------------------------------------------------------------------

    found_target_ids = set(
        transaction_history[
            "TransactionID"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    case_metadata[
        "target_txn_found"
    ] = case_metadata[
        "target_txn_id"
    ].apply(
        lambda value: (
            False
            if pd.isna(value)
            else int(value)
            in found_target_ids
        )
    )

    missing_targets = case_metadata.loc[
        (
            ~case_metadata[
                "target_txn_found"
            ]
        )
        & case_metadata[
            "target_txn_id"
        ].notna()
    ]

    if not missing_targets.empty:
        log(
            "ERROR: "
            f"{len(missing_targets):,} "
            "target transactions were not found."
        )

        log(
            "First missing cases: "
            + ", ".join(
                missing_targets[
                    "case_id"
                ]
                .astype(str)
                .head(20)
                .tolist()
            )
        )

        return 2

    # ------------------------------------------------------------------
    # 6. Identify target device profiles
    # ------------------------------------------------------------------

    target_transaction_ids = set(
        case_metadata[
            "target_txn_id"
        ]
        .dropna()
        .astype(int)
        .tolist()
    )

    target_transactions = (
        transaction_history.loc[
            transaction_history[
                "TransactionID"
            ].isin(
                target_transaction_ids
            ),
            [
                "TransactionID",
                "device_profile",
                "device_profile_specificity",
            ],
        ]
        .drop_duplicates(
            subset=["TransactionID"]
        )
    )

    target_profiles = set(
        target_transactions[
            "device_profile"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    # ------------------------------------------------------------------
    # 7. Full cross-customer device-profile scan
    # ------------------------------------------------------------------

    device_customer_index = (
        build_device_customer_index(
            identity=identity,
            target_profiles=target_profiles,
            transactions_file=(
                TRANSACTIONS_FILE
            ),
            chunk_size=args.chunk_size,
        )
    )

    # ------------------------------------------------------------------
    # 8. Build one feature row per historical case
    # ------------------------------------------------------------------

    # Keep case_id as a column as well as the lookup index.
    # calculate_case_features() expects case["case_id"], while set_index()
    # drops that column by default.
    metadata_by_case = (
        case_metadata
        .set_index("case_id", drop=False)
    )

    feature_rows: List[
        Dict[str, Any]
    ] = []

    log(
        "Constructing historical "
        "evidence features..."
    )

    for index, historical_case in (
        historical.iterrows()
    ):
        case_id = str(
            historical_case["case_id"]
        )

        case = metadata_by_case.loc[
            case_id
        ]

        row = calculate_case_features(
            case=case,
            transaction_history=(
                transaction_history
            ),
            historical_cases=historical,
            device_customer_index=(
                device_customer_index
            ),
        )

        feature_rows.append(row)

        if (
            (index + 1) % 250
            == 0
        ):
            log(
                "Cases processed: "
                f"{index + 1:,}/"
                f"{len(historical):,}"
            )

    dataset = pd.DataFrame(
        feature_rows
    )

    if dataset.empty:
        print(
            "ERROR: generated training "
            "dataset is empty.",
            file=sys.stderr,
        )
        return 2

    # Deterministic ordering.
    dataset = (
        dataset
        .sort_values(
            [
                "opened_at",
                "case_id",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 9. Validate
    # ------------------------------------------------------------------

    validation = (
        validate_training_dataset(
            dataset=dataset,
            historical=historical,
        )
    )

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------

    summary = {
        "purpose": (
            "Leakage-safe historical training dataset "
            "for fraud probability modeling."
        ),
        "risk_score_is_fraud_probability": False,
        "benchmark_cases_included": False,
        "label_source": (
            "closed_cases_history.csv"
        ),
        "source_files": {
            "closed_cases_history": str(
                CLOSED_CASES_FILE
            ),
            "transactions": str(
                TRANSACTIONS_FILE
            ),
            "identity": str(
                IDENTITY_FILE
            ),
        },
        "historical_cases": int(
            len(historical)
        ),
        "labels": {
            "confirmed_fraud": int(
                (
                    historical[
                        "outcome"
                    ]
                    == "confirmed_fraud"
                ).sum()
            ),
            "cleared": int(
                (
                    historical[
                        "outcome"
                    ]
                    == "cleared"
                ).sum()
            ),
        },
        "transactions": {
            "retained_history_rows": int(
                len(transaction_history)
            ),
            "unique_transaction_ids": int(
                transaction_history[
                    "TransactionID"
                ].nunique()
            ),
            "unique_customers": int(
                transaction_history[
                    "customer_id"
                ].nunique()
            ),
            "target_transactions_expected": int(
                case_metadata[
                    "target_txn_id"
                ].notna().sum()
            ),
            "target_transactions_found": int(
                case_metadata[
                    "target_txn_found"
                ].sum()
            ),
        },
        "identity": {
            "rows": int(
                len(identity)
            ),
            "target_profiles": int(
                len(target_profiles)
            ),
        },
        "model_feature_columns": (
            MODEL_FEATURE_COLUMNS
        ),
        "feature_columns": (
            dataset.columns.tolist()
        ),
        "validation": validation,
        "runtime_seconds": round(
            time.time() - start_time,
            2,
        ),
    }

    # ------------------------------------------------------------------
    # 11. Write outputs
    # ------------------------------------------------------------------

    dataset_path = (
        output_dir
        / "probability_training_dataset.csv"
    )

    summary_path = (
        output_dir
        / "probability_training_summary.json"
    )

    validation_path = (
        output_dir
        / "probability_training_validation.txt"
    )

    dataset.to_csv(
        dataset_path,
        index=False,
    )

    write_json(
        summary_path,
        summary,
    )

    write_validation_notes(
        validation_path,
        summary,
    )

    # ------------------------------------------------------------------
    # 12. Console summary
    # ------------------------------------------------------------------

    log("")
    log("=" * 72)
    log(
        "PROBABILITY TRAINING DATASET COMPLETE"
    )
    log("=" * 72)

    log(
        f"Rows: {len(dataset):,}"
    )

    log(
        "Confirmed fraud: "
        f"{int((dataset['label'] == 1).sum()):,}"
    )

    log(
        "Cleared: "
        f"{int((dataset['label'] == 0).sum()):,}"
    )

    log(
        "Target transactions found: "
        f"{int(case_metadata['target_txn_found'].sum()):,}"
    )

    log(
        "Target transactions after case opening: "
        f"{int(dataset['target_after_case_open'].sum()):,}"
    )

    log(
        "Historical cutoff violations: "
        f"{validation['historical_cutoff_violations']:,}"
    )

    log(
        "Independent evidence rule violations: "
        f"{validation['independent_rule_violations']:,}"
    )

    log(
        "Validation PASS: "
        f"{validation['overall_pass']}"
    )

    log(
        f"Dataset: {dataset_path}"
    )

    log(
        f"Summary: {summary_path}"
    )

    log(
        f"Validation: {validation_path}"
    )

    if not validation["overall_pass"]:
        log("")
        log(
            "WARNING: validation did not pass."
        )
        log(
            "Do NOT fit the probability model yet."
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
