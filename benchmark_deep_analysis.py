#!/usr/bin/env python3

"""
benchmark_deep_analysis.py

Deep benchmark analysis for the TigerGraph Agentic Fraud Investigation project.

Project structure expected:

telegraph_fraud_agent/
├── benchmark_deep_analysis.py
├── data/
│   ├── transactions.csv
│   ├── identity.csv
│   ├── closed_cases_history.csv
│   └── case_pack.csv
└── analysis/

Run:
    python benchmark_deep_analysis.py

The script intentionally does NOT produce a final fraud probability or final
fraud decision. It produces structured evidence and deterministic pattern
candidates for the investigation agent.

Important design choices:
- Transactions are processed in chunks.
- Future transactions beyond the benchmark case's opened_at are excluded.
- risk_score is treated as an input signal, not as a fraud verdict.
- Opaque identity fields are reported as observed signals only.
- card_id from case_pack is preserved as supplied; no K1/K2 mapping is invented.
- Historical cases are restricted to cases closed before/equal to the benchmark
  case opening time to avoid temporal leakage.
- Device sharing uses an exact device-profile fingerprint based on:
      DeviceInfo + id_30 + id_31 + id_33
  matching the documented DeviceProfile concept.
- addr1 is treated as a region identifier only. A "home region" is derived
  from the customer's modal historical in-person addr1 and explicitly treated
  as a proxy, not ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "analysis"

TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
IDENTITY_FILE = DATA_DIR / "identity.csv"
CLOSED_CASES_FILE = DATA_DIR / "closed_cases_history.csv"
CASE_PACK_FILE = DATA_DIR / "case_pack.csv"

CHUNK_SIZE = 100_000

# Only these transaction columns are required.
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

# The documented device profile concept is:
# DeviceInfo + OS + browser + screen.
DEVICE_PROFILE_COMPONENTS = [
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]

WINDOWS = {
    "24h": pd.Timedelta(hours=24),
    "48h": pd.Timedelta(hours=48),
    "7d": pd.Timedelta(days=7),
    "30d": pd.Timedelta(days=30),
}

SMALL_AUTH_AMOUNT = 5.0


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def log(message: str) -> None:
    """Simple timestamped console logging."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def fail(message: str, code: int = 1) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(code)


def ensure_files_exist() -> None:
    required = [
        TRANSACTIONS_FILE,
        IDENTITY_FILE,
        CLOSED_CASES_FILE,
        CASE_PACK_FILE,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        fail(
            "The following required files were not found:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nExpected data directory:\n{DATA_DIR}"
        )


def safe_str(value: Any) -> Optional[str]:
    """Convert a value into a clean string, treating NaN/NA as None."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()

    if text == "" or text.lower() in {"nan", "none", "nat", "<na>"}:
        return None

    return text


def normalize_value(value: Any) -> Optional[str]:
    """
    Normalize a categorical value for deterministic fingerprints.

    Keeps the original semantic value but removes surrounding whitespace.
    """
    value = safe_str(value)
    return value


def json_safe(value: Any) -> Any:
    """
    Recursively convert pandas/numpy objects into JSON-serializable values.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return pd.Timestamp(value).isoformat()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(payload),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # Preserve useful column information even for an empty dataframe.
        df.to_csv(path, index=False)
        return

    df.to_csv(path, index=False)


def clean_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps and NaNs into CSV-friendly values."""
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return out


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def make_pipe_key(values: Sequence[Any]) -> Optional[str]:
    """
    Build a stable pipe-delimited key.

    None values are represented as '<NULL>'.
    """
    normalized = []

    for value in values:
        value = safe_str(value)
        normalized.append(value if value is not None else "<NULL>")

    if all(v == "<NULL>" for v in normalized):
        return None

    return "|".join(normalized)


def parse_pipe_list(value: Any) -> List[str]:
    value = safe_str(value)

    if value is None:
        return []

    return [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]


def unique_preserve_order(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []

    for value in values:
        value = safe_str(value)

        if value is None or value in seen:
            continue

        seen.add(value)
        output.append(value)

    return output


# ============================================================================
# IDENTITY / DEVICE HELPERS
# ============================================================================

def build_device_profile_from_row(row: pd.Series) -> Optional[str]:
    """
    Exact device profile:

        DeviceInfo + id_30 + id_31 + id_33

    This follows the documented DeviceProfile concept:
        DeviceInfo + OS + browser + screen
    """
    components = [
        normalize_value(row.get("DeviceInfo")),
        normalize_value(row.get("id_30")),
        normalize_value(row.get("id_31")),
        normalize_value(row.get("id_33")),
    ]

    if all(value is None for value in components):
        return None

    return make_pipe_key(components)


def build_device_profile_from_dict(record: Dict[str, Any]) -> Optional[str]:
    components = [
        normalize_value(record.get("DeviceInfo")),
        normalize_value(record.get("id_30")),
        normalize_value(record.get("id_31")),
        normalize_value(record.get("id_33")),
    ]

    if all(value is None for value in components):
        return None

    return make_pipe_key(components)


def device_profile_specificity(record: Dict[str, Any]) -> int:
    """
    Number of non-null core device-profile fields.
    """
    return sum(
        1
        for field in DEVICE_PROFILE_COMPONENTS
        if normalize_value(record.get(field)) is not None
    )


def build_card_fingerprint(row: pd.Series) -> Optional[str]:
    """
    Raw card attribute fingerprint.

    IMPORTANT:
    This is deliberately NOT treated as the synthetic case_pack card_id.
    """
    values = [
        row.get("card1"),
        row.get("card2"),
        row.get("card3"),
        row.get("card4"),
        row.get("card5"),
        row.get("card6"),
    ]

    if all(pd.isna(v) for v in values):
        return None

    return make_pipe_key(values)


# ============================================================================
# LOAD CASE PACK
# ============================================================================

def load_case_pack() -> pd.DataFrame:
    log("Loading case_pack.csv...")

    cases = pd.read_csv(CASE_PACK_FILE)

    required = [
        "case_id",
        "trigger_type",
        "flagged_txn_id",
        "card_id",
        "customer_id",
    ]

    missing = [c for c in required if c not in cases.columns]

    if missing:
        fail(
            "case_pack.csv is missing required columns: "
            + ", ".join(missing)
        )

    cases["flagged_txn_id"] = pd.to_numeric(
        cases["flagged_txn_id"],
        errors="coerce",
    )

    cases["flagged_txn_id"] = cases["flagged_txn_id"].astype("Int64")

    cases["customer_id"] = cases["customer_id"].astype(str)
    cases["card_id"] = cases["card_id"].astype(str)

    if "opened_at" in cases.columns:
        cases["opened_at"] = pd.to_datetime(
            cases["opened_at"],
            errors="coerce",
        )

    if "created_at" in cases.columns:
        cases["created_at"] = pd.to_datetime(
            cases["created_at"],
            errors="coerce",
        )

    # Some versions of case_pack may use a trigger column with slightly
    # different naming. Preserve whatever exists.
    if "trigger" not in cases.columns:
        cases["trigger"] = cases.get("trigger_type", "")

    log(f"Loaded {len(cases)} benchmark cases.")

    return cases


# ============================================================================
# LOAD IDENTITY
# ============================================================================

def load_identity() -> pd.DataFrame:
    log("Loading identity.csv...")

    identity = pd.read_csv(
        IDENTITY_FILE,
        usecols=lambda col: col in IDENTITY_COLUMNS,
        low_memory=False,
    )

    identity["TransactionID"] = pd.to_numeric(
        identity["TransactionID"], errors="coerce"
    )

    for col in IDENTITY_COLUMNS:
        if col != "TransactionID" and col in identity.columns:
            identity[col] = identity[col].replace(
                ["", "nan", "NaN", "None", "null"], pd.NA
            )

    log(f"Loaded identity rows: {len(identity):,}")

    return identity


# ------------------------------------------------------------
# TRANSACTION HELPERS
# ------------------------------------------------------------

def normalize_transaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize only fields needed by the deep analysis."""

    required = [
        "TransactionID",
        "customer_id",
        "ts",
        "channel",
        "risk_score",
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
    ]

    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df["TransactionID"] = pd.to_numeric(
        df["TransactionID"], errors="coerce"
    )

    df["customer_id"] = df["customer_id"].astype("string")

    df["ts"] = pd.to_datetime(
        df["ts"], errors="coerce"
    )

    numeric_cols = [
        "risk_score",
        "TransactionAmt",
        "card1",
        "card2",
        "card3",
        "card5",
        "addr1",
        "addr2",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    string_cols = [
        "channel",
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
    ]

    for col in string_cols:
        df[col] = df[col].astype("string")

    return df


def make_card_fingerprint(row: pd.Series) -> str:
    """
    Construct a raw-data card fingerprint.

    IMPORTANT:
    This is NOT the case_pack Cxxx-K1/K2 identifier.
    """

    parts = []

    for col in [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
    ]:
        value = row.get(col)

        if pd.isna(value):
            parts.append("")
        else:
            parts.append(str(value))

    if not any(parts):
        return ""

    return "|".join(parts)


def safe_scalar(value):
    """Convert pandas/numpy values to JSON-safe Python values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        return value.item()

    return value


def safe_list(values):
    return [safe_scalar(v) for v in values]


def json_default(value):
    value = safe_scalar(value)

    if value is not None:
        return value

    return str(value)


# ------------------------------------------------------------
# CASE / TIMESTAMP HELPERS
# ------------------------------------------------------------

def parse_case_opened_at(case_row: pd.Series) -> pd.Timestamp:
    """
    Resolve the benchmark case opening timestamp.

    case_pack may contain opened_at directly. If absent, fall back
    to the transaction timestamp because the transaction is the
    investigation trigger.

    The fallback is explicitly recorded by the caller.
    """

    opened = case_row.get("opened_at")

    if opened is not None and not pd.isna(opened):
        parsed = pd.to_datetime(opened, errors="coerce")

        if not pd.isna(parsed):
            return parsed

    return pd.NaT


def build_case_lookup(case_pack: pd.DataFrame) -> dict:
    lookup = {}

    for _, row in case_pack.iterrows():
        case_id = str(row["case_id"])
        lookup[case_id] = row.to_dict()

    return lookup


# ------------------------------------------------------------
# STREAM TRANSACTIONS
# ------------------------------------------------------------

def get_transaction_columns() -> list:
    """
    Columns required for the deep benchmark analysis.

    Vesta contains hundreds of columns, so only the required subset
    is loaded from the 675 MB transaction file.
    """

    return [
        "TransactionID",
        "customer_id",
        "ts",
        "channel",
        "risk_score",
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
    ]


def collect_benchmark_transactions(
    case_pack: pd.DataFrame,
) -> pd.DataFrame:
    """
    First pass over transactions.csv.

    Finds all benchmark transaction IDs and keeps their complete
    raw transaction records.
    """

    if "flagged_txn_id" not in case_pack.columns:
        raise KeyError(
            "Expected 'flagged_txn_id' in case_pack.csv. "
            f"Available columns: {case_pack.columns.tolist()}"
        )

    target_ids = set(
        pd.to_numeric(
            case_pack["flagged_txn_id"],
            errors="coerce",
        )
        .dropna()
        .astype("int64")
        .tolist()
    )

    log(
        f"Scanning transactions.csv for "
        f"{len(target_ids)} benchmark transactions..."
    )

    found = []

    usecols = get_transaction_columns()

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            TRANSACTIONS_FILE,
            usecols=lambda c: c in usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        chunk = normalize_transaction_columns(chunk)

        mask = chunk["TransactionID"].isin(target_ids)

        if mask.any():
            found.append(chunk.loc[mask].copy())

        if chunk_no % 10 == 0:
            log(
                f"Transaction scan: processed chunk {chunk_no}"
            )

    if not found:
        raise RuntimeError(
            "None of the benchmark transactions were found "
            "in transactions.csv"
        )

    benchmark_txns = pd.concat(
        found,
        ignore_index=True,
    )

    benchmark_txns = benchmark_txns.drop_duplicates(
        subset=["TransactionID"]
    )

    log(
        f"Benchmark transactions found: "
        f"{len(benchmark_txns):,}/{len(target_ids):,}"
    )

    missing = target_ids - set(
        benchmark_txns["TransactionID"]
        .dropna()
        .astype("int64")
        .tolist()
    )

    if missing:
        log(
            f"WARNING: missing benchmark transaction IDs: "
            f"{sorted(missing)}"
        )

    return benchmark_txns


# ------------------------------------------------------------
# IDENTITY JOIN
# ------------------------------------------------------------

def join_benchmark_identity(
    benchmark_txns: pd.DataFrame,
    identity: pd.DataFrame,
) -> pd.DataFrame:

    identity = identity.copy()

    identity["TransactionID"] = pd.to_numeric(
        identity["TransactionID"], errors="coerce"
    )

    identity = identity.drop_duplicates(
        subset=["TransactionID"]
    )

    joined = benchmark_txns.merge(
        identity,
        on="TransactionID",
        how="left",
        suffixes=("", "_identity"),
    )

    return joined


# ------------------------------------------------------------
# CASE TRANSACTION MATCHING
# ------------------------------------------------------------

def attach_case_metadata(
    benchmark_txns: pd.DataFrame,
    case_pack: pd.DataFrame,
) -> pd.DataFrame:

    case_df = case_pack.copy()

    if "flagged_txn_id" not in case_df.columns:
        raise KeyError(
            "Expected 'flagged_txn_id' in case_pack.csv. "
            f"Available columns: {case_df.columns.tolist()}"
        )

    case_df["flagged_txn_id"] = pd.to_numeric(
        case_df["flagged_txn_id"],
        errors="coerce",
    )

    merged = case_df.merge(
        benchmark_txns,
        left_on="flagged_txn_id",
        right_on="TransactionID",
        how="left",
        suffixes=("_case", ""),
    )

    return merged


# ------------------------------------------------------------
# DEVICE PROFILE
# ------------------------------------------------------------

CORE_DEVICE_FIELDS = [
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]


def build_device_profile(row: pd.Series):
    """
    Exact device profile:

        DeviceInfo + id_30 + id_31 + id_33

    id_15 and id_23 are deliberately excluded.
    """

    values = []

    non_null_count = 0

    for field in CORE_DEVICE_FIELDS:
        value = row.get(field)

        if value is None or pd.isna(value):
            values.append("")
        else:
            value = str(value).strip()
            values.append(value)

            if value:
                non_null_count += 1

    if non_null_count == 0:
        return "", 0

    return " | ".join(values), non_null_count


def add_device_profile_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    profiles = []
    specificity = []

    for _, row in df.iterrows():
        profile, count = build_device_profile(row)

        profiles.append(profile)
        specificity.append(count)

    df["device_profile"] = profiles
    df["device_profile_specificity"] = specificity

    return df


# ------------------------------------------------------------
# BENCHMARK OPENING TIME
# ------------------------------------------------------------

def resolve_case_open_time(
    case_row: pd.Series,
    flagged_ts,
) -> tuple:

    opened = parse_case_opened_at(case_row)

    if not pd.isna(opened):
        return opened, False

    flagged = pd.to_datetime(
        flagged_ts,
        errors="coerce",
    )

    return flagged, True


# ------------------------------------------------------------
# STREAM ALL TRANSACTIONS FOR TEMPORAL ANALYSIS
# ------------------------------------------------------------

def scan_transaction_history(
    benchmark_cases: pd.DataFrame,
) -> dict:

    """
    Second pass over transactions.csv.

    Builds only the information required by the 20 benchmark cases.

    No transaction after the individual case opening time is retained
    for evidence calculations.
    """

    customers = set(
        benchmark_cases["customer_id_case"]
        .dropna()
        .astype(str)
    )

    if "customer_id" in benchmark_cases.columns:
        customers.update(
            benchmark_cases["customer_id"]
            .dropna()
            .astype(str)
        )

    customer_targets = customers

    log(
        f"Scanning historical transaction activity for "
        f"{len(customer_targets):,} customers..."
    )

    relevant_chunks = []

    usecols = get_transaction_columns()

    case_cutoffs = {}

    for _, row in benchmark_cases.iterrows():
        case_id = str(row["case_id"])

        cutoff = pd.to_datetime(
            row["_case_opened_at"],
            errors="coerce",
        )

        if not pd.isna(cutoff):
            case_cutoffs[case_id] = cutoff

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            TRANSACTIONS_FILE,
            usecols=lambda c: c in usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        chunk = normalize_transaction_columns(chunk)

        chunk = chunk[
            chunk["customer_id"].astype(str).isin(customer_targets)
        ]

        if chunk.empty:
            continue

        relevant_chunks.append(chunk)

        if chunk_no % 10 == 0:
            log(
                f"History scan: processed chunk {chunk_no}"
            )

    if not relevant_chunks:
        return pd.DataFrame(columns=usecols)

    history = pd.concat(
        relevant_chunks,
        ignore_index=True,
    )

    history = history.drop_duplicates(
        subset=["TransactionID"]
    )

    history = history.sort_values(
        ["customer_id", "ts"],
        kind="mergesort",
    )

    log(
        f"Relevant transaction history rows: "
        f"{len(history):,}"
    )

    return history


# ------------------------------------------------------------
# CASE-SPECIFIC TEMPORAL FILTER
# ------------------------------------------------------------

def case_history_window(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    days: int,
) -> pd.DataFrame:

    if pd.isna(flagged_ts):
        return history.iloc[0:0].copy()

    lower = flagged_ts - pd.Timedelta(days=days)

    # Evidence cutoff is the case opening time.
    upper = case_opened_at

    if pd.isna(upper):
        upper = flagged_ts

    mask = (
        (history["customer_id"].astype(str) == str(customer_id))
        & (history["ts"] >= lower)
        & (history["ts"] <= upper)
    )

    return history.loc[mask].copy()


def activity_until_case_open(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
) -> pd.DataFrame:

    if pd.isna(flagged_ts):
        return history.iloc[0:0].copy()

    upper = case_opened_at

    if pd.isna(upper):
        upper = flagged_ts

    mask = (
        (history["customer_id"].astype(str) == str(customer_id))
        & (history["ts"] >= flagged_ts)
        & (history["ts"] <= upper)
    )

    return history.loc[mask].copy()


# ------------------------------------------------------------
# GENERIC TEMPORAL METRICS
# ------------------------------------------------------------

def calculate_temporal_metrics(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
) -> dict:

    result = {
        "txn_count_24h": 0,
        "txn_count_48h": 0,
        "txn_count_7d": 0,
        "txn_count_30d": 0,
        "online_count_24h": 0,
        "online_count_48h": 0,
        "online_count_7d": 0,
        "in_person_count_7d": 0,
        "amount_sum_48h": 0.0,
        "avg_amount_48h": None,
        "median_amount_48h": None,
        "max_amount_48h": None,
        "post_flagged_activity_until_case_open": 0,
        "channel_seen_before": False,
        "channel_novel": False,
        "product_seen_before": False,
        "product_novel": False,
        "amount_vs_prior_median": None,
    }

    if pd.isna(flagged_ts):
        return result

    h24 = case_history_window(
        history,
        customer_id,
        flagged_ts,
        case_opened_at,
        1,
    )

    h48 = case_history_window(
        history,
        customer_id,
        flagged_ts,
        case_opened_at,
        2,
    )

    h7 = case_history_window(
        history,
        customer_id,
        flagged_ts,
        case_opened_at,
        7,
    )

    h30 = case_history_window(
        history,
        customer_id,
        flagged_ts,
        case_opened_at,
        30,
    )

    result["txn_count_24h"] = len(h24)
    result["txn_count_48h"] = len(h48)
    result["txn_count_7d"] = len(h7)
    result["txn_count_30d"] = len(h30)

    result["online_count_24h"] = int(
        (h24["channel"] == "online").sum()
    )

    result["online_count_48h"] = int(
        (h48["channel"] == "online").sum()
    )

    result["online_count_7d"] = int(
        (h7["channel"] == "online").sum()
    )

    result["in_person_count_7d"] = int(
        (h7["channel"] == "in_person").sum()
    )

    amounts = pd.to_numeric(
        h48["TransactionAmt"],
        errors="coerce",
    ).dropna()

    if len(amounts):
        result["amount_sum_48h"] = float(amounts.sum())
        result["avg_amount_48h"] = float(amounts.mean())
        result["median_amount_48h"] = float(amounts.median())
        result["max_amount_48h"] = float(amounts.max())

    post_activity = activity_until_case_open(
        history,
        customer_id,
        flagged_ts,
        case_opened_at,
    )

    # Exclude the flagged transaction itself.
    post_activity = post_activity[
        post_activity["ts"] > flagged_ts
    ]

    result["post_flagged_activity_until_case_open"] = len(
        post_activity
    )

    return result


# ------------------------------------------------------------
# SMALL AUTHORIZATION ANALYSIS
# ------------------------------------------------------------

def calculate_small_auth_metrics(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    flagged_amount,
) -> dict:

    result = {
        "small_auth_count_24h": 0,
        "small_auth_count_48h": 0,
        "small_auth_count_7d": 0,
        "card_testing_candidate": False,
    }

    if pd.isna(flagged_ts):
        return result

    try:
        flagged_amount = float(flagged_amount)
    except Exception:
        flagged_amount = None

    for days, key in [
        (1, "small_auth_count_24h"),
        (2, "small_auth_count_48h"),
        (7, "small_auth_count_7d"),
    ]:

        window = case_history_window(
            history,
            customer_id,
            flagged_ts,
            case_opened_at,
            days,
        )

        window = window[
            window["channel"] == "online"
        ]

        window = window[
            pd.to_numeric(
                window["TransactionAmt"],
                errors="coerce",
            ) < 5
        ]

        # Do not count the flagged transaction itself.
        window = window[
            window["ts"] < flagged_ts
        ]

        result[key] = len(window)

    if (
        result["small_auth_count_48h"] >= 3
        and flagged_amount is not None
        and flagged_amount > 5
    ):
        result["card_testing_candidate"] = True

    return result


# ------------------------------------------------------------
# CHANNEL / PRODUCT NOVELTY
# ------------------------------------------------------------

def calculate_behavior_metrics(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    flagged_channel,
    flagged_product,
    flagged_amount,
) -> dict:

    result = {
        "channel_seen_before": False,
        "channel_novel": False,
        "product_seen_before": False,
        "product_novel": False,
        "amount_vs_prior_median": None,
    }

    if pd.isna(flagged_ts):
        return result

    prior = history[
        (history["customer_id"].astype(str) == str(customer_id))
        & (history["ts"] < flagged_ts)
        & (history["ts"] <= case_opened_at)
    ].copy()

    if flagged_channel is not None and not pd.isna(flagged_channel):
        result["channel_seen_before"] = bool(
            (prior["channel"] == flagged_channel).any()
        )

        result["channel_novel"] = not result[
            "channel_seen_before"
        ]

    if flagged_product is not None and not pd.isna(flagged_product):
        result["product_seen_before"] = bool(
            (prior["ProductCD"] == flagged_product).any()
        )

        result["product_novel"] = not result[
            "product_seen_before"
        ]

    prior_amounts = pd.to_numeric(
        prior["TransactionAmt"],
        errors="coerce",
    ).dropna()

    if len(prior_amounts):
        try:
            amount = float(flagged_amount)
            median = float(prior_amounts.median())

            if median > 0:
                result["amount_vs_prior_median"] = (
                    amount / median
                )
        except Exception:
            pass

    return result


# ------------------------------------------------------------
# REGION ANALYSIS
# ------------------------------------------------------------

def calculate_region_metrics(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    flagged_channel,
    current_region,
) -> dict:

    result = {
        "current_region": safe_scalar(current_region),
        "prior_region_seen": False,
        "prior_in_person_regions": [],
        "modal_home_region_proxy": None,
        "is_new_region": False,
        "home_activity_continues": False,
        "out_of_region_candidate": False,
    }

    if (
        pd.isna(flagged_ts)
        or flagged_channel != "in_person"
        or pd.isna(current_region)
    ):
        return result

    prior = history[
        (history["customer_id"].astype(str) == str(customer_id))
        & (history["ts"] < flagged_ts)
        & (history["ts"] <= case_opened_at)
        & (history["channel"] == "in_person")
    ].copy()

    prior = prior.dropna(subset=["addr1"])

    if prior.empty:
        result["is_new_region"] = True
        return result

    prior_regions = (
        prior["addr1"]
        .dropna()
        .astype(str)
        .value_counts()
    )

    result["prior_in_person_regions"] = [
        safe_scalar(x)
        for x in prior_regions.index.tolist()
    ]

    current_region_str = str(current_region)

    result["prior_region_seen"] = (
        current_region_str in prior_regions.index.astype(str)
    )

    if len(prior_regions):
        home_region = str(prior_regions.index[0])

        result["modal_home_region_proxy"] = safe_scalar(
            prior_regions.index[0]
        )

        result["is_new_region"] = not result[
            "prior_region_seen"
        ]

        # Look for previous activity in the modal region after the
        # current flagged transaction but before case opening.
        after_flag = history[
            (history["customer_id"].astype(str) == str(customer_id))
            & (history["ts"] > flagged_ts)
            & (history["ts"] <= case_opened_at)
            & (history["channel"] == "in_person")
        ].copy()

        after_flag = after_flag.dropna(
            subset=["addr1"]
        )

        if not after_flag.empty:
            result["home_activity_continues"] = bool(
                (
                    after_flag["addr1"]
                    .astype(str)
                    == home_region
                ).any()
            )

        if (
            result["is_new_region"]
            and result["home_activity_continues"]
        ):
            result["out_of_region_candidate"] = True

    return result


# ------------------------------------------------------------
# EMAIL ANALYSIS
# ------------------------------------------------------------

def calculate_email_metrics(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    purchaser_email,
    recipient_email,
) -> dict:

    result = {
        "purchaser_email_seen_before": False,
        "recipient_email_seen_before": False,
        "email_novelty": False,
    }

    if pd.isna(flagged_ts):
        return result

    prior = history[
        (history["customer_id"].astype(str) == str(customer_id))
        & (history["ts"] < flagged_ts)
        & (history["ts"] <= case_opened_at)
    ].copy()

    purchaser_seen = False
    recipient_seen = False

    if (
        purchaser_email is not None
        and not pd.isna(purchaser_email)
    ):
        purchaser_seen = bool(
            (
                prior["P_emaildomain"].astype(str)
                == str(purchaser_email)
            ).any()
        )

    if (
        recipient_email is not None
        and not pd.isna(recipient_email)
    ):
        recipient_seen = bool(
            (
                prior["R_emaildomain"].astype(str)
                == str(recipient_email)
            ).any()
        )

    result["purchaser_email_seen_before"] = purchaser_seen
    result["recipient_email_seen_before"] = recipient_seen

    provided = [
        x for x in [purchaser_email, recipient_email]
        if x is not None and not pd.isna(x)
    ]

    if provided:
        result["email_novelty"] = not (
            purchaser_seen and recipient_seen
        )

    return result


# ------------------------------------------------------------
# CNP / NEW DEVICE / ATO PATTERNS
# ------------------------------------------------------------

def calculate_pattern_candidates(
    history: pd.DataFrame,
    customer_id: str,
    flagged_ts: pd.Timestamp,
    case_opened_at: pd.Timestamp,
    flagged_channel,
    flagged_device_new,
    proxy_signal,
    device_anomaly,
    flagged_product,
    flagged_profile,
    shared_device_candidate=False,
    repeated_abuse_candidate=False,
) -> dict:

    result = {
        "cnp_candidate": False,
        "cnp_new_device_candidate": False,
        "account_takeover_candidate": False,
        "shared_device_candidate": bool(
            shared_device_candidate
        ),
        "repeated_abuse_candidate": bool(
            repeated_abuse_candidate
        ),
    }

    if pd.isna(flagged_ts):
        return result

    h48 = case_history_window(
        history,
        customer_id,
        flagged_ts,
        case_opened_at,
        2,
    )

    if flagged_channel == "online":

        online_count = int(
            (h48["channel"] == "online").sum()
        )

        # The documented pattern describes approximately 2–4
        # transactions within 48h. The flagged transaction is
        # already included in this contextual count.
        result["cnp_candidate"] = (
            2 <= online_count <= 4
        )

        if flagged_device_new:
            result["cnp_new_device_candidate"] = True

    mixed_channel = (
        (h48["channel"] == "online").any()
        and (h48["channel"] == "in_person").any()
    )

    if (
        mixed_channel
        and (
            flagged_device_new
            or bool(proxy_signal)
            or bool(device_anomaly)
        )
    ):
        result["account_takeover_candidate"] = True

    return result


# ------------------------------------------------------------
# DEVICE HISTORY
# ------------------------------------------------------------

def scan_shared_device_usage(
    benchmark_df: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:

    """
    Find exact core device-profile matches among all relevant
    transaction history.

    Because only benchmark customers are retained by the history
    scan, this function is complemented by a full device scan below
    when shared-device discovery across unrelated customers is needed.
    """

    benchmark_profiles = benchmark_df[
        [
            "case_id",
            "customer_id_case",
            "device_profile",
            "device_profile_specificity",
            "_case_opened_at",
        ]
    ].copy()

    benchmark_profiles = benchmark_profiles[
        benchmark_profiles["device_profile"].astype(str).str.len() > 0
    ]

    if benchmark_profiles.empty:
        return pd.DataFrame()

    rows = []

    for _, case in benchmark_profiles.iterrows():

        profile = case["device_profile"]

        matched = history[
            history["device_profile"] == profile
        ].copy()

        cutoff = case["_case_opened_at"]

        if not pd.isna(cutoff):
            matched = matched[
                matched["ts"] <= cutoff
            ]

        if matched.empty:
            continue

        customers = (
            matched["customer_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        card_fingerprints = []

        for _, tx in matched.iterrows():
            fp = make_card_fingerprint(tx)

            if fp:
                card_fingerprints.append(fp)

        rows.append(
            {
                "case_id": case["case_id"],
                "benchmark_customer_id": case[
                    "customer_id_case"
                ],
                "device_profile": profile,
                "device_profile_specificity": int(
                    case["device_profile_specificity"]
                ),
                "customer_count": len(customers),
                "transaction_count": len(matched),
                "first_use": (
                    matched["ts"].min()
                    if len(matched)
                    else None
                ),
                "last_use": (
                    matched["ts"].max()
                    if len(matched)
                    else None
                ),
                "customers": json.dumps(
                    customers,
                    default=json_default,
                ),
                "card_fingerprints": json.dumps(
                    sorted(set(card_fingerprints)),
                    default=json_default,
                ),
                "other_customers": json.dumps(
                    [
                        c
                        for c in customers
                        if c != str(
                            case["customer_id_case"]
                        )
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# FULL SHARED DEVICE SCAN
# ------------------------------------------------------------

def collect_benchmark_device_profiles(
    benchmark_df: pd.DataFrame,
) -> set:

    return set(
        benchmark_df.loc[
            benchmark_df["device_profile"].astype(str).str.len()
            > 0,
            "device_profile",
        ].astype(str)
    )


def scan_full_device_network(
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Scan the transaction file again, but only retain transactions
    whose exact device profile matches one of the benchmark profiles.

    This deliberately does NOT treat partial profiles as exact matches.
    """

    target_profiles = collect_benchmark_device_profiles(
        benchmark_df
    )

    if not target_profiles:
        return pd.DataFrame()

    log(
        f"Scanning for shared exact device profiles: "
        f"{len(target_profiles)}"
    )

    usecols = get_transaction_columns()

    matched_chunks = []

    identity_lookup = load_identity()

    # Build identity lookup in memory once.
    identity_lookup = identity_lookup.drop_duplicates(
        subset=["TransactionID"]
    )

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            TRANSACTIONS_FILE,
            usecols=lambda c: c in usecols,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        chunk = normalize_transaction_columns(chunk)

        joined = chunk.merge(
            identity_lookup,
            on="TransactionID",
            how="left",
        )

        joined = add_device_profile_columns(joined)

        matched = joined[
            joined["device_profile"].isin(
                target_profiles
            )
        ].copy()

        if not matched.empty:
            matched_chunks.append(matched)

        if chunk_no % 10 == 0:
            log(
                f"Device network scan: chunk {chunk_no}"
            )

    if not matched_chunks:
        return pd.DataFrame()

    network = pd.concat(
        matched_chunks,
        ignore_index=True,
    )

    return network


def build_device_network_outputs(
    benchmark_df: pd.DataFrame,
    device_network: pd.DataFrame,
) -> tuple:

    if device_network.empty:
        empty = pd.DataFrame()

        return empty, empty

    links = []
    networks = []

    for _, case in benchmark_df.iterrows():

        case_id = case["case_id"]
        profile = case["device_profile"]

        if not profile:
            continue

        cutoff = case["_case_opened_at"]

        matched = device_network[
            device_network["device_profile"] == profile
        ].copy()

        if not pd.isna(cutoff):
            matched = matched[
                matched["ts"] <= cutoff
            ]

        if matched.empty:
            continue

        benchmark_customer = str(
            case["customer_id_case"]
        )

        customer_counts = (
            matched.groupby(
                matched["customer_id"].astype(str)
            )
            .size()
            .sort_values(ascending=False)
        )

        customers = customer_counts.index.tolist()

        others = [
            c
            for c in customers
            if c != benchmark_customer
        ]

        for customer in customers:
            customer_rows = matched[
                matched["customer_id"].astype(str)
                == customer
            ]

            fps = sorted(
                set(
                    fp
                    for fp in customer_rows.apply(
                        make_card_fingerprint,
                        axis=1,
                    )
                    if fp
                )
            )

            links.append(
                {
                    "case_id": case_id,
                    "benchmark_customer_id": benchmark_customer,
                    "matched_customer_id": customer,
                    "is_benchmark_customer": (
                        customer == benchmark_customer
                    ),
                    "device_profile": profile,
                    "device_profile_specificity": int(
                        case[
                            "device_profile_specificity"
                        ]
                    ),
                    "transaction_count": len(
                        customer_rows
                    ),
                    "first_use": customer_rows["ts"].min(),
                    "last_use": customer_rows["ts"].max(),
                    "card_fingerprints": json.dumps(
                        fps
                    ),
                }
            )

        networks.append(
            {
                "case_id": case_id,
                "benchmark_customer_id": benchmark_customer,
                "device_profile": profile,
                "device_profile_specificity": int(
                    case["device_profile_specificity"]
                ),
                "customer_count": len(customers),
                "transaction_count": len(matched),
                "other_customer_count": len(others),
                "other_customers": json.dumps(others),
                "first_use": matched["ts"].min(),
                "last_use": matched["ts"].max(),
            }
        )

    return (
        pd.DataFrame(links),
        pd.DataFrame(networks),
    )


# ------------------------------------------------------------
# HISTORICAL CASE ANALYSIS
# ------------------------------------------------------------

def load_historical_cases() -> pd.DataFrame:

    log("Loading closed_cases_history.csv...")

    historical = pd.read_csv(
        CLOSED_CASES_FILE,
        low_memory=False,
    )

    for col in [
        "opened_at",
        "closed_at",
    ]:
        if col in historical.columns:
            historical[col] = pd.to_datetime(
                historical[col],
                errors="coerce",
            )

    historical["customer_id"] = (
        historical["customer_id"]
        .astype("string")
    )

    historical["card_id"] = (
        historical["card_id"]
        .astype("string")
    )

    historical["outcome"] = (
        historical["outcome"]
        .astype("string")
    )

    historical["pattern"] = (
        historical["pattern"]
        .astype("string")
    )

    return historical


def analyze_historical_cases_for_benchmark(
    benchmark_df: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple:

    summaries = []
    links = []

    for _, case in benchmark_df.iterrows():

        case_id = str(case["case_id"])
        customer_id = str(
            case["customer_id_case"]
        )

        cutoff = pd.to_datetime(
            case["_case_opened_at"],
            errors="coerce",
        )

        candidates = historical[
            historical["customer_id"].astype(str)
            == customer_id
        ].copy()

        if not pd.isna(cutoff):
            candidates = candidates[
                candidates["closed_at"] <= cutoff
            ]

        candidates = candidates.sort_values(
            ["opened_at", "closed_at"],
            kind="mergesort",
        )

        confirmed = candidates[
            candidates["outcome"] == "confirmed_fraud"
        ]

        cleared = candidates[
            candidates["outcome"] == "cleared"
        ]

        patterns = sorted(
            set(
                p
                for p in candidates["pattern"]
                .dropna()
                .astype(str)
                if p
            )
        )

        exposure = pd.to_numeric(
            candidates["exposure_usd"],
            errors="coerce",
        ).dropna()

        reports = int(
            (
                candidates["report_filed"]
                .astype(str)
                .str.lower()
                == "yes"
            ).sum()
        )

        latest_case = None

        if not candidates.empty:
            latest = candidates.iloc[-1]
            latest_case = {
                "case_id": safe_scalar(
                    latest.get("case_id")
                ),
                "opened_at": safe_scalar(
                    latest.get("opened_at")
                ),
                "closed_at": safe_scalar(
                    latest.get("closed_at")
                ),
                "outcome": safe_scalar(
                    latest.get("outcome")
                ),
                "pattern": safe_scalar(
                    latest.get("pattern")
                ),
            }

        summaries.append(
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "historical_case_count": len(
                    candidates
                ),
                "historical_confirmed_count": len(
                    confirmed
                ),
                "historical_cleared_count": len(
                    cleared
                ),
                "historical_patterns": json.dumps(
                    patterns
                ),
                "historical_exposure": (
                    float(exposure.sum())
                    if len(exposure)
                    else 0.0
                ),
                "historical_report_count": reports,
                "latest_historical_case": json.dumps(
                    latest_case,
                    default=json_default,
                ),
            }
        )

        for _, hist in candidates.iterrows():

            links.append(
                {
                    "case_id": case_id,
                    "benchmark_customer_id": customer_id,
                    "historical_case_id": safe_scalar(
                        hist.get("case_id")
                    ),
                    "historical_card_id": safe_scalar(
                        hist.get("card_id")
                    ),
                    "opened_at": safe_scalar(
                        hist.get("opened_at")
                    ),
                    "closed_at": safe_scalar(
                        hist.get("closed_at")
                    ),
                    "outcome": safe_scalar(
                        hist.get("outcome")
                    ),
                    "pattern": safe_scalar(
                        hist.get("pattern")
                    ),
                    "first_fraud_txn_id": safe_scalar(
                        hist.get("first_fraud_txn_id")
                    ),
                    "n_txns": safe_scalar(
                        hist.get("n_txns")
                    ),
                    "exposure_usd": safe_scalar(
                        hist.get("exposure_usd")
                    ),
                    "connected_card_ids": safe_scalar(
                        hist.get("connected_card_ids")
                    ),
                    "report_filed": safe_scalar(
                        hist.get("report_filed")
                    ),
                }
            )

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(links),
    )


# ------------------------------------------------------------
# REPEATED ABUSE
# ------------------------------------------------------------

def calculate_repeated_abuse(
    historical_summary: dict,
) -> bool:

    confirmed = int(
        historical_summary.get(
            "historical_confirmed_count",
            0,
        )
        or 0
    )

    return confirmed >= 2


# ------------------------------------------------------------
# FULL BENCHMARK ANALYSIS
# ------------------------------------------------------------

def analyze_benchmark_cases(
    benchmark_df: pd.DataFrame,
    history: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple:

    summary_rows = []
    window_rows = []
    region_rows = []
    email_rows = []

    # Identity fields that are part of the benchmark output.
    identity_fields = [
        "id_15",
        "id_23",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
        "DeviceType",
        "DeviceInfo",
    ]

    for _, case in benchmark_df.iterrows():

        case_id = str(case["case_id"])

        customer_id = str(
            case["customer_id_case"]
        )

        flagged_ts = pd.to_datetime(
            case["ts"],
            errors="coerce",
        )

        case_opened_at = pd.to_datetime(
            case["_case_opened_at"],
            errors="coerce",
        )

        amount = safe_scalar(
            case.get("TransactionAmt")
        )

        channel = safe_scalar(
            case.get("channel")
        )

        product = safe_scalar(
            case.get("ProductCD")
        )

        risk_score = safe_scalar(
            case.get("risk_score")
        )

        # -------------------------
        # Temporal
        # -------------------------

        temporal = calculate_temporal_metrics(
            history,
            customer_id,
            flagged_ts,
            case_opened_at,
        )

        # -------------------------
        # Small authorizations
        # -------------------------

        small_auth = calculate_small_auth_metrics(
            history,
            customer_id,
            flagged_ts,
            case_opened_at,
            amount,
        )

        # -------------------------
        # Behavior
        # -------------------------

        behavior = calculate_behavior_metrics(
            history,
            customer_id,
            flagged_ts,
            case_opened_at,
            channel,
            product,
            amount,
        )

        # -------------------------
        # Region
        # -------------------------

        region = calculate_region_metrics(
            history,
            customer_id,
            flagged_ts,
            case_opened_at,
            channel,
            case.get("addr1"),
        )

        # -------------------------
        # Email
        # -------------------------

        email = calculate_email_metrics(
            history,
            customer_id,
            flagged_ts,
            case_opened_at,
            case.get("P_emaildomain"),
            case.get("R_emaildomain"),
        )

        # -------------------------
        # Historical cases
        # -------------------------

        historical_candidates = historical[
            historical["customer_id"].astype(str)
            == customer_id
        ].copy()

        if not pd.isna(case_opened_at):
            historical_candidates = historical_candidates[
                historical_candidates["closed_at"]
                <= case_opened_at
            ]

        historical_confirmed = int(
            (
                historical_candidates["outcome"]
                == "confirmed_fraud"
            ).sum()
        )

        historical_cleared = int(
            (
                historical_candidates["outcome"]
                == "cleared"
            ).sum()
        )

        repeated_abuse = (
            historical_confirmed >= 2
        )

        # -------------------------
        # Device
        # -------------------------

        profile = case.get(
            "device_profile",
            "",
        )

        device_specificity = int(
            case.get(
                "device_profile_specificity",
                0,
            )
            or 0
        )

        device_matches = history[
            history["device_profile"] == profile
        ].copy()

        if not pd.isna(case_opened_at):
            device_matches = device_matches[
                device_matches["ts"]
                <= case_opened_at
            ]

        device_customers = sorted(
            set(
                device_matches["customer_id"]
                .dropna()
                .astype(str)
            )
        )

        other_device_customers = [
            x
            for x in device_customers
            if x != customer_id
        ]

        shared_device_candidate = (
            bool(profile)
            and device_specificity >= 2
            and len(other_device_customers) > 0
        )

        # -------------------------
        # Device / identity signals
        # -------------------------

        id15 = case.get("id_15")

        new_device = (
            str(id15).strip().lower() == "new"
            if not pd.isna(id15)
            else False
        )

        id23 = case.get("id_23")

        proxy_signal = (
            "proxy" in str(id23).lower()
            if not pd.isna(id23)
            else False
        )

        id34 = case.get("id_34")

        device_anomaly = False

        if not pd.isna(id34):
            # Preserve the field as opaque. We only record that an
            # explicitly supplied match_status value exists.
            device_anomaly = bool(
                str(id34).strip()
            )

        # -------------------------
        # Pattern candidates
        # -------------------------

        patterns = calculate_pattern_candidates(
            history=history,
            customer_id=customer_id,
            flagged_ts=flagged_ts,
            case_opened_at=case_opened_at,
            flagged_channel=channel,
            flagged_device_new=new_device,
            proxy_signal=proxy_signal,
            device_anomaly=device_anomaly,
            flagged_product=product,
            flagged_profile=profile,
            shared_device_candidate=(
                shared_device_candidate
            ),
            repeated_abuse_candidate=repeated_abuse,
        )

        # -------------------------
        # Pattern output rows
        # -------------------------

        pattern_candidates = [
            name
            for name, value in [
                (
                    "card_testing",
                    small_auth[
                        "card_testing_candidate"
                    ],
                ),
                (
                    "card_not_present",
                    patterns["cnp_candidate"],
                ),
                (
                    "cnp_new_device",
                    patterns[
                        "cnp_new_device_candidate"
                    ],
                ),
                (
                    "out_of_region",
                    region[
                        "out_of_region_candidate"
                    ],
                ),
                (
                    "account_takeover",
                    patterns[
                        "account_takeover_candidate"
                    ],
                ),
                (
                    "shared_device",
                    shared_device_candidate,
                ),
                (
                    "repeated_abuse",
                    repeated_abuse,
                ),
            ]
            if value
        ]

        # -------------------------
        # Main summary
        # -------------------------

        row = {
            "case_id": case_id,
            "trigger_type": safe_scalar(
                case.get("trigger_type")
            ),
            "flagged_txn_id": safe_scalar(
                case.get("flagged_txn_id")
            ),
            "customer_id": customer_id,
            "card_id": safe_scalar(
                case.get("card_id")
            ),
            "case_opened_at": safe_scalar(
                case_opened_at
            ),
            "flagged_ts": safe_scalar(
                flagged_ts
            ),
            "amount": amount,
            "channel": channel,
            "ProductCD": product,
            "risk_score": risk_score,
            "addr1": safe_scalar(
                case.get("addr1")
            ),
            "addr2": safe_scalar(
                case.get("addr2")
            ),
            "P_emaildomain": safe_scalar(
                case.get("P_emaildomain")
            ),
            "R_emaildomain": safe_scalar(
                case.get("R_emaildomain")
            ),
        }

        for field in identity_fields:
            row[field] = safe_scalar(
                case.get(field)
            )

        row.update(
            {
                "device_profile": profile,
                "device_profile_specificity": (
                    device_specificity
                ),
            }
        )

        row.update(temporal)
        row.update(small_auth)
        row.update(region)
        row.update(email)
        row.update(behavior)

        row.update(
            {
                "shared_device_customer_count": len(
                    device_customers
                ),
                "shared_device_transaction_count": len(
                    device_matches
                ),
                "other_customers_on_device": json.dumps(
                    other_device_customers
                ),
                "device_activity_before_case_open": len(
                    device_matches
                ),
                "historical_case_count": len(
                    historical_candidates
                ),
                "historical_confirmed_count": (
                    historical_confirmed
                ),
                "historical_cleared_count": (
                    historical_cleared
                ),
                "historical_patterns": json.dumps(
                    sorted(
                        set(
                            historical_candidates[
                                "pattern"
                            ]
                            .dropna()
                            .astype(str)
                        )
                    )
                ),
                "historical_exposure": float(
                    pd.to_numeric(
                        historical_candidates[
                            "exposure_usd"
                        ],
                        errors="coerce",
                    )
                    .dropna()
                    .sum()
                ),
                "historical_report_count": int(
                    (
                        historical_candidates[
                            "report_filed"
                        ]
                        .astype(str)
                        .str.lower()
                        == "yes"
                    ).sum()
                ),
                "latest_historical_case": (
                    json.dumps(
                        {
                            "case_id": safe_scalar(
                                historical_candidates.iloc[-1].get(
                                    "case_id"
                                )
                            ),
                            "opened_at": safe_scalar(
                                historical_candidates.iloc[-1].get(
                                    "opened_at"
                                )
                            ),
                            "closed_at": safe_scalar(
                                historical_candidates.iloc[-1].get(
                                    "closed_at"
                                )
                            ),
                            "outcome": safe_scalar(
                                historical_candidates.iloc[-1].get(
                                    "outcome"
                                )
                            ),
                            "pattern": safe_scalar(
                                historical_candidates.iloc[-1].get(
                                    "pattern"
                                )
                            ),
                        },
                        default=json_default,
                    )
                    if not historical_candidates.empty
                    else None
                ),
                "card_testing_candidate": (
                    small_auth[
                        "card_testing_candidate"
                    ]
                ),
                "cnp_candidate": patterns[
                    "cnp_candidate"
                ],
                "cnp_new_device_candidate": patterns[
                    "cnp_new_device_candidate"
                ],
                "out_of_region_candidate": region[
                    "out_of_region_candidate"
                ],
                "account_takeover_candidate": patterns[
                    "account_takeover_candidate"
                ],
                "shared_device_candidate": (
                    shared_device_candidate
                ),
                "repeated_abuse_candidate": (
                    repeated_abuse
                ),
                "pattern_candidates": json.dumps(
                    pattern_candidates
                ),
            }
        )

        # IMPORTANT:
        # No fraud_probability is calculated here.
        # No final fraud verdict is calculated here.

        summary_rows.append(row)

        # -------------------------
        # Temporal window output
        # -------------------------

        for days, label in [
            (1, "24h"),
            (2, "48h"),
            (7, "7d"),
            (30, "30d"),
        ]:

            window = case_history_window(
                history,
                customer_id,
                flagged_ts,
                case_opened_at,
                days,
            )

            window_rows.append(
                {
                    "case_id": case_id,
                    "customer_id": customer_id,
                    "flagged_txn_id": safe_scalar(
                        case.get("flagged_txn_id")
                    ),
                    "window": label,
                    "window_start": (
                        flagged_ts
                        - pd.Timedelta(days=days)
                    ),
                    "window_end": case_opened_at,
                    "transaction_count": len(window),
                    "online_count": int(
                        (
                            window["channel"]
                            == "online"
                        ).sum()
                    ),
                    "in_person_count": int(
                        (
                            window["channel"]
                            == "in_person"
                        ).sum()
                    ),
                    "amount_sum": float(
                        pd.to_numeric(
                            window["TransactionAmt"],
                            errors="coerce",
                        )
                        .dropna()
                        .sum()
                    ),
                    "median_amount": (
                        float(
                            pd.to_numeric(
                                window[
                                    "TransactionAmt"
                                ],
                                errors="coerce",
                            )
                            .dropna()
                            .median()
                        )
                        if len(
                            pd.to_numeric(
                                window[
                                    "TransactionAmt"
                                ],
                                errors="coerce",
                            )
                            .dropna()
                        )
                        else None
                    ),
                }
            )

        # -------------------------
        # Region output
        # -------------------------

        region_rows.append(
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "flagged_txn_id": safe_scalar(
                    case.get("flagged_txn_id")
                ),
                **region,
            }
        )

        # -------------------------
        # Email output
        # -------------------------

        email_rows.append(
            {
                "case_id": case_id,
                "customer_id": customer_id,
                "flagged_txn_id": safe_scalar(
                    case.get("flagged_txn_id")
                ),
                **email,
            }
        )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(window_rows),
        pd.DataFrame(region_rows),
        pd.DataFrame(email_rows),
    )


# ------------------------------------------------------------
# NOTES
# ------------------------------------------------------------

def build_analysis_notes(
    summary: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    historical: pd.DataFrame,
) -> str:

    lines = []

    lines.append(
        "TIGERGRAPH AGENTIC FRAUD INVESTIGATION "
        "— BENCHMARK DEEP ANALYSIS"
    )
    lines.append("=" * 72)
    lines.append("")

    lines.append(
        "This analysis produces observed signals, derived signals, "
        "pattern candidates and historical support."
    )

    lines.append(
        "It is NOT a final fraud decision engine."
    )

    lines.append("")

    lines.append(
        f"Benchmark cases analyzed: {len(summary)}"
    )

    lines.append(
        f"Historical cases loaded: {len(historical):,}"
    )

    lines.append("")

    lines.append("Temporal leakage controls:")
    lines.append(
        "- Evidence is restricted to transaction activity "
        "at or before case opening."
    )
    lines.append(
        "- Historical closed cases are restricted to "
        "closed_at <= case opening."
    )
    lines.append(
        "- Future transactions after case opening are not "
        "used as evidence."
    )

    lines.append("")

    lines.append("Device profile:")
    lines.append(
        "- Exact profile = DeviceInfo + id_30 + id_31 + id_33."
    )
    lines.append(
        "- id_15 and id_23 are not part of the core profile."
    )
    lines.append(
        "- Partial/common profiles are not treated as definitive "
        "shared-device proof."
    )

    lines.append("")

    lines.append("Card identity:")
    lines.append(
        "- case_pack card_id values such as C12382-K1/K2 are "
        "preserved exactly."
    )
    lines.append(
        "- Raw transaction card fingerprints are generated only "
        "from card1..card6."
    )
    lines.append(
        "- No K1/K2 mapping is inferred."
    )

    lines.append("")

    lines.append("Fraud probability:")
    lines.append(
        "- No fraud probability is calculated by this script."
    )
    lines.append(
        "- risk_score is retained as an input signal only."
    )
    lines.append(
        "- The eventual investigation agent must combine graph "
        "evidence, historical evidence, pattern detection, "
        "policy and simulated verification."
    )

    lines.append("")

    if not summary.empty:

        pattern_cols = [
            "card_testing_candidate",
            "cnp_candidate",
            "cnp_new_device_candidate",
            "out_of_region_candidate",
            "account_takeover_candidate",
            "shared_device_candidate",
            "repeated_abuse_candidate",
        ]

        lines.append("Pattern candidate counts:")

        for col in pattern_cols:
            if col in summary.columns:
                count = int(
                    summary[col]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                )

                lines.append(
                    f"- {col}: {count}/{len(summary)}"
                )

        lines.append("")

        lines.append(
            "Historical support:"
        )

        if "historical_confirmed_count" in summary.columns:
            lines.append(
                "- Cases with >=1 prior confirmed fraud case: "
                f"{int((summary['historical_confirmed_count'] > 0).sum())}"
            )

        if "historical_cleared_count" in summary.columns:
            lines.append(
                "- Cases with >=1 prior cleared case: "
                f"{int((summary['historical_cleared_count'] > 0).sum())}"
            )

    lines.append("")

    lines.append(
        "Interpretation rule:"
    )
    lines.append(
        "Historical cases and pattern candidates are supporting "
        "evidence, not automatic proof of current fraud."
    )

    lines.append(
        "Undocumented coordinated or repeated abuse should remain "
        "discoverable instead of being forced into one of the "
        "documented patterns."
    )

    return "\n".join(lines)


# ------------------------------------------------------------
# OUTPUT HELPERS
# ------------------------------------------------------------

def write_dataframe(
    df: pd.DataFrame,
    filename: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / filename

    if df is None:
        df = pd.DataFrame()

    df.to_csv(
        path,
        index=False,
    )

    log(
        f"Wrote {filename}: "
        f"{len(df):,} rows"
    )


def write_json_records(
    records,
    filename: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = OUTPUT_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )

    log(
        f"Wrote {filename}"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    start_time = time.time()

    log("=" * 72)
    log("TIGERGRAPH FRAUD — BENCHMARK DEEP ANALYSIS")
    log("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load benchmark cases
    # --------------------------------------------------------

    log("Loading benchmark case pack...")

    case_pack = load_case_pack()

    log(
        f"Benchmark cases loaded: {len(case_pack):,}"
    )

    # --------------------------------------------------------
    # 2. Find benchmark transactions
    # --------------------------------------------------------

    benchmark_transactions = (
        collect_benchmark_transactions(
            case_pack
        )
    )

    # --------------------------------------------------------
    # 3. Load identity
    # --------------------------------------------------------

    identity = load_identity()

    benchmark_transactions = (
        join_benchmark_identity(
            benchmark_transactions,
            identity,
        )
    )

    benchmark_transactions = (
        add_device_profile_columns(
            benchmark_transactions
        )
    )

    # --------------------------------------------------------
    # 4. Attach benchmark metadata
    # --------------------------------------------------------

    benchmark_df = attach_case_metadata(
        benchmark_transactions,
        case_pack,
    )

    # --------------------------------------------------------
    # 5. Resolve case opening timestamps
    # --------------------------------------------------------

    opened_values = []
    opened_fallbacks = []

    for _, row in benchmark_df.iterrows():

        opened, fallback = resolve_case_open_time(
            row,
            row.get("ts"),
        )

        opened_values.append(opened)
        opened_fallbacks.append(fallback)

    benchmark_df["_case_opened_at"] = (
        opened_values
    )

    benchmark_df["_case_opened_at_fallback"] = (
        opened_fallbacks
    )

    # If case_pack contains opened_at, it is authoritative.
    # If it does not, the flagged transaction timestamp is used
    # as a conservative evidence cutoff.
    fallback_count = sum(
        opened_fallbacks
    )

    if fallback_count:
        log(
            "WARNING: "
            f"{fallback_count} benchmark cases did not have "
            "a usable opened_at and used flagged transaction "
            "timestamp as the cutoff."
        )

    # --------------------------------------------------------
    # 6. Historical cases
    # --------------------------------------------------------

    historical = load_historical_cases()

    # --------------------------------------------------------
    # 7. Stream relevant transaction history
    # --------------------------------------------------------

    history = scan_transaction_history(
        benchmark_df
    )

    # --------------------------------------------------------
    # 8. Identity join on relevant history
    #
    # This is needed for exact device-profile matching in the
    # benchmark customer's historical activity.
    # --------------------------------------------------------

    history = history.merge(
        identity,
        on="TransactionID",
        how="left",
        suffixes=("", "_identity"),
    )

    history = add_device_profile_columns(
        history
    )

    # --------------------------------------------------------
    # 9. Main benchmark analysis
    # --------------------------------------------------------

    (
        summary,
        transaction_windows,
        region_analysis,
        email_analysis,
    ) = analyze_benchmark_cases(
        benchmark_df,
        history,
        historical,
    )

    # --------------------------------------------------------
    # 10. Historical case tables
    # --------------------------------------------------------

    (
        historical_case_summary,
        historical_case_links,
    ) = analyze_historical_cases_for_benchmark(
        benchmark_df,
        historical,
    )

    # --------------------------------------------------------
    # 11. Shared device network
    # --------------------------------------------------------

    log(
        "Running full exact-device network scan..."
    )

    device_network = scan_full_device_network(
        benchmark_df
    )

    (
        device_links,
        shared_device_network,
    ) = build_device_network_outputs(
        benchmark_df,
        device_network,
    )

    # --------------------------------------------------------
    # 12. Enrich summary with full shared-device results
    # --------------------------------------------------------

    if (
        not summary.empty
        and not shared_device_network.empty
    ):

        network_counts = (
            shared_device_network[
                [
                    "case_id",
                    "customer_count",
                    "transaction_count",
                    "other_customer_count",
                    "other_customers",
                ]
            ]
            .drop_duplicates(
                subset=["case_id"]
            )
            .rename(
                columns={
                    "customer_count":
                        "_network_customer_count",
                    "transaction_count":
                        "_network_transaction_count",
                    "other_customer_count":
                        "_network_other_customer_count",
                    "other_customers":
                        "_network_other_customers",
                }
            )
        )

        summary = summary.merge(
            network_counts,
            on="case_id",
            how="left",
        )

        summary[
            "shared_device_customer_count"
        ] = summary[
            "_network_customer_count"
        ].fillna(
            summary[
                "shared_device_customer_count"
            ]
        )

        summary[
            "shared_device_transaction_count"
        ] = summary[
            "_network_transaction_count"
        ].fillna(
            summary[
                "shared_device_transaction_count"
            ]
        )

        summary[
            "other_customers_on_device"
        ] = summary[
            "_network_other_customers"
        ].fillna(
            summary[
                "other_customers_on_device"
            ]
        )

        summary = summary.drop(
            columns=[
                "_network_customer_count",
                "_network_transaction_count",
                "_network_other_customer_count",
                "_network_other_customers",
            ],
            errors="ignore",
        )

    # --------------------------------------------------------
    # 13. Add analysis classification columns
    # --------------------------------------------------------

    if not summary.empty:

        summary["observed_signal_count"] = 0

        observed_cols = [
            "channel_novel",
            "product_novel",
            "is_new_region",
            "email_novelty",
            "shared_device_candidate",
            "repeated_abuse_candidate",
        ]

        for col in observed_cols:
            if col in summary.columns:
                summary["observed_signal_count"] += (
                    summary[col]
                    .fillna(False)
                    .astype(bool)
                    .astype(int)
                )

        summary["historical_support_present"] = (
            summary[
                "historical_confirmed_count"
            ].fillna(0)
            > 0
        )

    # --------------------------------------------------------
    # 14. Write benchmark summary CSV
    # --------------------------------------------------------

    write_dataframe(
        summary,
        "benchmark_summary.csv",
    )

    # --------------------------------------------------------
    # 15. Write benchmark summary JSON
    # --------------------------------------------------------

    summary_records = (
        summary
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    write_json_records(
        summary_records,
        "benchmark_summary.json",
    )

    # --------------------------------------------------------
    # 16. Write all requested analysis files
    # --------------------------------------------------------

    write_dataframe(
        transaction_windows,
        "transaction_windows.csv",
    )

    write_dataframe(
        device_links,
        "device_links.csv",
    )

    write_dataframe(
        shared_device_network,
        "shared_device_network.csv",
    )

    write_dataframe(
        region_analysis,
        "region_analysis.csv",
    )

    write_dataframe(
        email_analysis,
        "email_analysis.csv",
    )

    write_dataframe(
        historical_case_summary,
        "historical_case_summary.csv",
    )

    write_dataframe(
        historical_case_links,
        "historical_case_links.csv",
    )

    # --------------------------------------------------------
    # 17. Notes
    # --------------------------------------------------------

    notes = build_analysis_notes(
        summary,
        benchmark_df,
        historical,
    )

    notes_path = OUTPUT_DIR / "analysis_notes.txt"

    with open(
        notes_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(notes)

    log(
        f"Wrote analysis_notes.txt"
    )

    # --------------------------------------------------------
    # 18. Final console summary
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    log("")
    log("=" * 72)
    log("DEEP ANALYSIS COMPLETE")
    log("=" * 72)

    log(
        f"Benchmark cases: {len(summary):,}"
    )

    log(
        f"Relevant transaction history: "
        f"{len(history):,}"
    )

    log(
        f"Historical closed cases: "
        f"{len(historical):,}"
    )

    log(
        f"Exact-device network rows: "
        f"{len(shared_device_network):,}"
    )

    log(
        f"Elapsed time: {elapsed:.2f} seconds"
    )

    log("")
    log(
        f"Output directory: {OUTPUT_DIR}"
    )

    log("")
    log(
        "IMPORTANT: This script produces investigation evidence "
        "and pattern candidates only."
    )

    log(
        "It does not make the final fraud decision and does not "
        "equate fraud_probability with risk_score."
    )


if __name__ == "__main__":
    main()