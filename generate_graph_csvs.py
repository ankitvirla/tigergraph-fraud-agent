#!/usr/bin/env python3

"""
Generate TigerGraph/Savanna-ready CSV files for the
HHGOA Fraud Investigation project.

Current graph schema:

Vertices:
    Customer
    Transaction
    Device
    Card
    FraudCase

Edges:
    Customer -> HAS_TRANSACTION -> Transaction
    Customer -> USES_DEVICE -> Device
    Customer -> USES_CARD -> Card
    Customer -> HAS_CASE -> FraudCase
    FraudCase -> CONTAINS -> Transaction

Important:
- transactions.csv is processed in chunks.
- The original transactions.csv is NOT modified.
- We do not invent the case-pack K1/K2 card mapping.
- Card vertices use a deterministic raw card fingerprint.
- Device vertices use a deterministic device-profile hash:
      DeviceInfo + id_30 + id_31 + id_33
- Only fields useful for the current graph investigation are loaded.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pandas as pd


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "graph_data"

TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
IDENTITY_FILE = DATA_DIR / "identity.csv"
CLOSED_CASES_FILE = DATA_DIR / "closed_cases_history.csv"
CASE_PACK_FILE = DATA_DIR / "case_pack.csv"

CHUNK_SIZE = 100_000


# ============================================================================
# HELPERS
# ============================================================================

def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True,
    )


def clean(value) -> str:
    """
    Normalize a value into a stable string.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }:
        return ""

    return value


def safe_number(value):
    """
    Convert pandas numeric values into plain Python-friendly values.
    """
    if pd.isna(value):
        return ""

    try:
        return float(value)
    except Exception:
        return clean(value)


def sha256_short(value: str) -> str:
    """
    Deterministic compact ID.

    We use a prefix so the graph IDs are human-readable.
    """
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:24]

    return digest


def make_device_id(
    device_info,
    os_name,
    browser,
    screen,
) -> str | None:

    values = [
        clean(device_info),
        clean(os_name),
        clean(browser),
        clean(screen),
    ]

    if not any(values):
        return None

    raw = "|".join(values)

    return "DEV_" + sha256_short(raw)


def make_card_id(row: pd.Series) -> str | None:
    """
    Build a deterministic raw-card fingerprint.

    IMPORTANT:
    This is NOT the case-pack Cxxxxx-K1/K2 identifier.

    We deliberately do not attempt to infer K1/K2.
    """

    fields = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
    ]

    values = [
        clean(row.get(field))
        for field in fields
    ]

    if not any(values):
        return None

    raw = "|".join(values)

    return "CARD_" + sha256_short(raw)


def remove_old_output_files() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in OUTPUT_DIR.glob("*.csv"):
        path.unlink()

    log(
        f"Cleaned graph output directory: {OUTPUT_DIR}"
    )


# ============================================================================
# 1. CUSTOMER VERTEX
# ============================================================================

def generate_customers():

    log("=" * 72)
    log("GENERATING CUSTOMER VERTEX")
    log("=" * 72)

    customers = set()

    # ------------------------------------------------------------------
    # Customers from transactions
    # ------------------------------------------------------------------

    transaction_usecols = [
        "customer_id",
    ]

    for chunk in pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=transaction_usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        values = (
            chunk["customer_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        customers.update(
            x for x in values
            if x
            and x.lower() != "nan"
        )

    # ------------------------------------------------------------------
    # Customers from closed cases
    # ------------------------------------------------------------------

    if CLOSED_CASES_FILE.exists():

        closed = pd.read_csv(
            CLOSED_CASES_FILE,
            usecols=["customer_id"],
            low_memory=False,
        )

        values = (
            closed["customer_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        customers.update(
            x for x in values
            if x
            and x.lower() != "nan"
        )

    # ------------------------------------------------------------------
    # Customers from benchmark cases
    # ------------------------------------------------------------------

    if CASE_PACK_FILE.exists():

        cases = pd.read_csv(
            CASE_PACK_FILE,
            usecols=["customer_id"],
            low_memory=False,
        )

        values = (
            cases["customer_id"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        customers.update(
            x for x in values
            if x
            and x.lower() != "nan"
        )

    df = pd.DataFrame(
        {
            "customer_id": sorted(customers)
        }
    )

    path = OUTPUT_DIR / "customer.csv"

    df.to_csv(
        path,
        index=False,
    )

    log(
        f"customer.csv: {len(df):,} customers"
    )


# ============================================================================
# 2. IDENTITY LOOKUP
# ============================================================================

def load_identity():

    log("=" * 72)
    log("LOADING IDENTITY DATA")
    log("=" * 72)

    identity_columns = [
        "TransactionID",
        "DeviceType",
        "DeviceInfo",
        "id_15",
        "id_23",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
    ]

    identity = pd.read_csv(
        IDENTITY_FILE,
        usecols=lambda c: c in identity_columns,
        low_memory=False,
    )

    identity["TransactionID"] = pd.to_numeric(
        identity["TransactionID"],
        errors="coerce",
    )

    identity = identity.dropna(
        subset=["TransactionID"]
    )

    identity["TransactionID"] = (
        identity["TransactionID"]
        .astype("int64")
    )

    log(
        f"Identity rows: {len(identity):,}"
    )

    return identity


# ============================================================================
# 3. TRANSACTION + EDGE GENERATION
# ============================================================================

def generate_transactions_and_edges(identity):

    log("=" * 72)
    log("GENERATING TRANSACTIONS + EDGES")
    log("=" * 72)

    transaction_path = (
        OUTPUT_DIR / "transaction.csv"
    )

    has_transaction_path = (
        OUTPUT_DIR / "customer_has_transaction.csv"
    )

    uses_device_path = (
        OUTPUT_DIR / "customer_uses_device.csv"
    )

    uses_card_path = (
        OUTPUT_DIR / "customer_uses_card.csv"
    )

    # ------------------------------------------------------------------
    # Remove previous outputs
    # ------------------------------------------------------------------

    for path in [
        transaction_path,
        has_transaction_path,
        uses_device_path,
        uses_card_path,
    ]:

        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Columns needed from transactions.csv
    # ------------------------------------------------------------------

    transaction_columns = [
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

    # ------------------------------------------------------------------
    # Keep track of unique device/card relationships
    # ------------------------------------------------------------------

    customer_device_pairs = set()
    customer_card_pairs = set()

    first_transaction_file = True

    transaction_count = 0

    # ------------------------------------------------------------------
    # Process transactions in chunks
    # ------------------------------------------------------------------

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            TRANSACTIONS_FILE,
            usecols=lambda c: c in transaction_columns,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        log(
            f"Processing transaction chunk {chunk_no}..."
        )

        # --------------------------------------------------------------
        # Normalize TransactionID
        # --------------------------------------------------------------

        chunk["TransactionID"] = pd.to_numeric(
            chunk["TransactionID"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=["TransactionID"]
        )

        chunk["TransactionID"] = (
            chunk["TransactionID"]
            .astype("int64")
        )

        # --------------------------------------------------------------
        # Customer
        # --------------------------------------------------------------

        chunk["customer_id"] = (
            chunk["customer_id"]
            .astype("string")
            .str.strip()
        )

        # --------------------------------------------------------------
        # Join identity information
        # --------------------------------------------------------------

        chunk = chunk.merge(
            identity,
            on="TransactionID",
            how="left",
            suffixes=("", "_identity"),
        )

        # --------------------------------------------------------------
        # Build device IDs
        # --------------------------------------------------------------

        device_ids = []

        for _, row in chunk.iterrows():

            device_id = make_device_id(
                row.get("DeviceInfo"),
                row.get("id_30"),
                row.get("id_31"),
                row.get("id_33"),
            )

            device_ids.append(
                device_id or ""
            )

        chunk["device_id"] = device_ids

        # --------------------------------------------------------------
        # Build card IDs
        # --------------------------------------------------------------

        card_ids = []

        for _, row in chunk.iterrows():

            card_id = make_card_id(row)

            card_ids.append(
                card_id or ""
            )

        chunk["card_id"] = card_ids

        # --------------------------------------------------------------
        # Transaction vertex
        # --------------------------------------------------------------

        transaction_output = pd.DataFrame(
            {
                "transaction_id":
                    chunk["TransactionID"].astype(str),

                "customer_id":
                    chunk["customer_id"].fillna("").astype(str),

                "amount":
                    chunk["TransactionAmt"].apply(
                        safe_number
                    ),

                "timestamp":
                    chunk["ts"].fillna("").astype(str),

                "product_code":
                    chunk["ProductCD"].fillna("").astype(str),

                "channel":
                    chunk["channel"].fillna("").astype(str),

                "risk_score":
                    chunk["risk_score"].apply(
                        safe_number
                    ),

                "billing_region":
                    chunk["addr1"].apply(clean),

                "billing_country":
                    chunk["addr2"].apply(clean),

                "p_email_domain":
                    chunk["P_emaildomain"].apply(clean),

                "r_email_domain":
                    chunk["R_emaildomain"].apply(clean),

                "device_id":
                    chunk["device_id"],

                "card_id":
                    chunk["card_id"],
            }
        )

        transaction_output.to_csv(
            transaction_path,
            mode="w" if first_transaction_file else "a",
            header=first_transaction_file,
            index=False,
        )

        first_transaction_file = False

        # --------------------------------------------------------------
        # Customer -> Transaction
        # --------------------------------------------------------------

        edge_df = pd.DataFrame(
            {
                "from_customer":
                    chunk["customer_id"]
                    .fillna("")
                    .astype(str),

                "to_transaction":
                    chunk["TransactionID"]
                    .astype(str),
            }
        )

        edge_df = edge_df[
            (edge_df["from_customer"] != "")
            & (edge_df["to_transaction"] != "")
        ]

        edge_df.to_csv(
            has_transaction_path,
            mode="a",
            header=not has_transaction_path.exists(),
            index=False,
        )

        # --------------------------------------------------------------
        # Customer -> Device
        # --------------------------------------------------------------

        for customer_id, device_id in zip(
            chunk["customer_id"],
            chunk["device_id"],
        ):

            customer_id = clean(customer_id)
            device_id = clean(device_id)

            if customer_id and device_id:

                customer_device_pairs.add(
                    (
                        customer_id,
                        device_id,
                    )
                )

        # --------------------------------------------------------------
        # Customer -> Card
        # --------------------------------------------------------------

        for customer_id, card_id in zip(
            chunk["customer_id"],
            chunk["card_id"],
        ):

            customer_id = clean(customer_id)
            card_id = clean(card_id)

            if customer_id and card_id:

                customer_card_pairs.add(
                    (
                        customer_id,
                        card_id,
                    )
                )

        transaction_count += len(chunk)

        log(
            f"  transactions processed: "
            f"{transaction_count:,}"
        )

    # ------------------------------------------------------------------
    # Write unique customer-device edges
    # ------------------------------------------------------------------

    device_edges = pd.DataFrame(
        sorted(customer_device_pairs),
        columns=[
            "from_customer",
            "to_device",
        ],
    )

    device_edges.to_csv(
        uses_device_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Write unique customer-card edges
    # ------------------------------------------------------------------

    card_edges = pd.DataFrame(
        sorted(customer_card_pairs),
        columns=[
            "from_customer",
            "to_card",
        ],
    )

    card_edges.to_csv(
        uses_card_path,
        index=False,
    )

    log(
        f"Unique customer-device links: "
        f"{len(device_edges):,}"
    )

    log(
        f"Unique customer-card links: "
        f"{len(card_edges):,}"
    )


# ============================================================================
# 4. DEVICE VERTEX
# ============================================================================

def generate_devices(identity):

    log("=" * 72)
    log("GENERATING DEVICE VERTEX")
    log("=" * 72)

    devices = {}

    for _, row in identity.iterrows():

        device_id = make_device_id(
            row.get("DeviceInfo"),
            row.get("id_30"),
            row.get("id_31"),
            row.get("id_33"),
        )

        if not device_id:
            continue

        if device_id not in devices:

            devices[device_id] = {
                "device_id": device_id,
                "device_type":
                    clean(row.get("DeviceType")),
                "device_info":
                    clean(row.get("DeviceInfo")),
                "os":
                    clean(row.get("id_30")),
                "browser":
                    clean(row.get("id_31")),
                "screen":
                    clean(row.get("id_33")),
                "device_new_found":
                    clean(row.get("id_15")),
                "proxy":
                    clean(row.get("id_23")),
                "match_status":
                    clean(row.get("id_34")),
            }

    df = pd.DataFrame(
        devices.values()
    )

    path = OUTPUT_DIR / "device.csv"

    df.to_csv(
        path,
        index=False,
    )

    log(
        f"device.csv: {len(df):,} devices"
    )


# ============================================================================
# 5. CARD VERTEX
# ============================================================================

def generate_cards():

    log("=" * 72)
    log("GENERATING CARD VERTEX")
    log("=" * 72)

    cards = {}

    card_columns = [
        "customer_id",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
    ]

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            TRANSACTIONS_FILE,
            usecols=lambda c: c in card_columns,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        log(
            f"Processing card chunk {chunk_no}..."
        )

        for _, row in chunk.iterrows():

            card_id = make_card_id(row)

            if not card_id:
                continue

            if card_id not in cards:

                cards[card_id] = {
                    "card_id": card_id,
                    "customer_id":
                        clean(row.get("customer_id")),
                    "card1":
                        clean(row.get("card1")),
                    "card2":
                        clean(row.get("card2")),
                    "card3":
                        clean(row.get("card3")),
                    "card4":
                        clean(row.get("card4")),
                    "card5":
                        clean(row.get("card5")),
                    "card6":
                        clean(row.get("card6")),
                }

    df = pd.DataFrame(
        cards.values()
    )

    path = OUTPUT_DIR / "card.csv"

    df.to_csv(
        path,
        index=False,
    )

    log(
        f"card.csv: {len(df):,} cards"
    )


# ============================================================================
# 6. FRAUD CASE VERTEX
# ============================================================================

def generate_fraud_cases():

    log("=" * 72)
    log("GENERATING FRAUD CASE VERTEX")
    log("=" * 72)

    closed = pd.read_csv(
        CLOSED_CASES_FILE,
        low_memory=False,
    )

    columns = [
        "case_id",
        "customer_id",
        "card_id",
        "opened_at",
        "closed_at",
        "outcome",
        "pattern",
        "first_fraud_txn_id",
        "n_txns",
        "exposure_usd",
        "connected_card_ids",
        "actions_taken",
        "report_filed",
        "analyst_notes",
    ]

    available = [
        c for c in columns
        if c in closed.columns
    ]

    closed = closed[available].copy()

    closed = closed.fillna("")

    # TigerGraph vertex IDs should be clean strings.
    closed["case_id"] = (
        closed["case_id"]
        .astype(str)
        .str.strip()
    )

    path = OUTPUT_DIR / "fraud_case.csv"

    closed.to_csv(
        path,
        index=False,
    )

    log(
        f"fraud_case.csv: {len(closed):,} historical cases"
    )


# ============================================================================
# 7. CUSTOMER -> CASE
# ============================================================================

def generate_customer_case_edges():

    log("=" * 72)
    log("GENERATING CUSTOMER -> CASE EDGES")
    log("=" * 72)

    closed = pd.read_csv(
        CLOSED_CASES_FILE,
        usecols=[
            "case_id",
            "customer_id",
        ],
        low_memory=False,
    )

    closed = closed.fillna("")

    edges = closed.rename(
        columns={
            "customer_id": "from_customer",
            "case_id": "to_case",
        }
    )

    edges = edges[
        (edges["from_customer"] != "")
        & (edges["to_case"] != "")
    ]

    edges = edges.drop_duplicates()

    path = (
        OUTPUT_DIR
        / "customer_has_case.csv"
    )

    edges.to_csv(
        path,
        index=False,
    )

    log(
        f"customer_has_case.csv: "
        f"{len(edges):,} edges"
    )


# ============================================================================
# 8. CASE -> TRANSACTION
# ============================================================================

def generate_case_transaction_edges():

    log("=" * 72)
    log("GENERATING CASE -> TRANSACTION EDGES")
    log("=" * 72)

    closed = pd.read_csv(
        CLOSED_CASES_FILE,
        usecols=[
            "case_id",
            "txn_ids",
        ],
        low_memory=False,
    )

    edges = []

    for _, row in closed.iterrows():

        case_id = clean(
            row.get("case_id")
        )

        txn_ids = clean(
            row.get("txn_ids")
        )

        if not case_id or not txn_ids:
            continue

        for txn_id in txn_ids.split("|"):

            txn_id = clean(txn_id)

            if not txn_id:
                continue

            edges.append(
                {
                    "from_case": case_id,
                    "to_transaction": txn_id,
                }
            )

    df = pd.DataFrame(
        edges
    ).drop_duplicates()

    path = (
        OUTPUT_DIR
        / "fraud_case_contains.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    log(
        f"fraud_case_contains.csv: "
        f"{len(df):,} edges"
    )


# ============================================================================
# 9. BENCHMARK CASES
# ============================================================================

def generate_benchmark_cases():

    log("=" * 72)
    log("GENERATING BENCHMARK CASE FILE")
    log("=" * 72)

    cases = pd.read_csv(
        CASE_PACK_FILE,
        low_memory=False,
    )

    path = (
        OUTPUT_DIR
        / "benchmark_cases.csv"
    )

    cases.to_csv(
        path,
        index=False,
    )

    log(
        f"benchmark_cases.csv: "
        f"{len(cases):,} cases"
    )


# ============================================================================
# 10. SUMMARY
# ============================================================================

def print_summary():

    print()
    print("=" * 72)
    print("GRAPH CSV GENERATION COMPLETE")
    print("=" * 72)
    print()
    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")
    print()

    for path in sorted(
        OUTPUT_DIR.glob("*.csv")
    ):

        try:
            rows = sum(
                1
                for _ in open(
                    path,
                    "r",
                    encoding="utf-8",
                )
            ) - 1

        except Exception:
            rows = "?"

        print(
            f"{path.name:<38} {rows:>10} rows"
        )

    print()
    print("Next step:")
    print(
        "Upload the vertex CSVs to Savanna first, "
        "then configure the edge CSVs."
    )
    print("=" * 72)


# ============================================================================
# MAIN
# ============================================================================

def main():

    start = time.time()

    log("=" * 72)
    log("HHGOA FRAUD — GRAPH CSV GENERATOR")
    log("=" * 72)

    # Validate input files
    required_files = [
        TRANSACTIONS_FILE,
        IDENTITY_FILE,
        CLOSED_CASES_FILE,
        CASE_PACK_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_old_output_files()

    # ------------------------------------------------------------
    # Customer
    # ------------------------------------------------------------

    generate_customers()

    # ------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------

    identity = load_identity()

    # ------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------

    generate_devices(identity)

    # ------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------

    generate_cards()

    # ------------------------------------------------------------
    # Transactions + edges
    # ------------------------------------------------------------

    generate_transactions_and_edges(
        identity
    )

    # ------------------------------------------------------------
    # Historical cases
    # ------------------------------------------------------------

    generate_fraud_cases()

    # ------------------------------------------------------------
    # Customer -> case
    # ------------------------------------------------------------

    generate_customer_case_edges()

    # ------------------------------------------------------------
    # Case -> transaction
    # ------------------------------------------------------------

    generate_case_transaction_edges()

    # ------------------------------------------------------------
    # Benchmark cases
    # ------------------------------------------------------------

    generate_benchmark_cases()

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print_summary()

    elapsed = time.time() - start

    log(
        f"Total runtime: {elapsed / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()