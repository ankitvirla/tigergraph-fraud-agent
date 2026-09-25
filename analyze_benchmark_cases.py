from pathlib import Path
import pandas as pd
import ast
import json

DATA_DIR = Path("data")

CASE_FILE = DATA_DIR / "case_pack.csv"
TXN_FILE = DATA_DIR / "transactions.csv"
IDENTITY_FILE = DATA_DIR / "identity.csv"
CLOSED_FILE = DATA_DIR / "closed_cases_history.csv"


# ------------------------------------------------------------
# 1. Load benchmark cases
# ------------------------------------------------------------

cases = pd.read_csv(CASE_FILE)

target_txns = set(cases["flagged_txn_id"].astype(int))

print("=" * 100)
print("BENCHMARK TRANSACTIONS")
print("=" * 100)

print(f"Cases: {len(cases)}")
print(f"Target transactions: {len(target_txns)}")


# ------------------------------------------------------------
# 2. Columns we need from transactions
# ------------------------------------------------------------

txn_cols = [
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


# ------------------------------------------------------------
# 3. Extract only benchmark transactions
# ------------------------------------------------------------

matches = []

for chunk in pd.read_csv(
    TXN_FILE,
    usecols=txn_cols,
    chunksize=100_000,
    low_memory=False,
):

    found = chunk[
        chunk["TransactionID"].isin(target_txns)
    ]

    if len(found):
        matches.append(found)

transactions = pd.concat(matches, ignore_index=True)

transactions = transactions.sort_values("TransactionID")


print(f"\nFound transactions: {len(transactions)}")

print("\n")
print(transactions.to_string(index=False))


# ------------------------------------------------------------
# 4. Join benchmark case metadata
# ------------------------------------------------------------

benchmark = cases.merge(
    transactions,
    left_on="flagged_txn_id",
    right_on="TransactionID",
    how="left",
    suffixes=("_case", "_txn"),
)


print("\n" + "=" * 100)
print("CASE → TRANSACTION MAPPING")
print("=" * 100)

cols = [
    "case_id",
    "trigger_type",
    "flagged_txn_id",
    "card_id",
    "customer_id_case",

    "TransactionAmt",
    "ProductCD",
    "customer_id_txn",
    "channel",
    "risk_score_txn",
    "ts",

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

print(
    benchmark[cols].to_string(index=False)
)


# ------------------------------------------------------------
# 5. Identity data for benchmark transactions
# ------------------------------------------------------------

identity_cols = [
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


identity = pd.read_csv(
    IDENTITY_FILE,
    usecols=identity_cols,
    low_memory=False,
)

benchmark_identity = identity[
    identity["TransactionID"].isin(target_txns)
].copy()


print("\n" + "=" * 100)
print("BENCHMARK IDENTITY / DEVICE INFORMATION")
print("=" * 100)

print(
    benchmark_identity
    .sort_values("TransactionID")
    .to_string(index=False)
)


# ------------------------------------------------------------
# 6. Historical cases related to benchmark customers/cards
# ------------------------------------------------------------

closed = pd.read_csv(
    CLOSED_FILE,
    low_memory=False,
)

benchmark_customers = set(
    cases["customer_id"].astype(str)
)

benchmark_cards = set(
    cases["card_id"].astype(str)
)


related_cases = closed[
    closed["customer_id"].astype(str).isin(benchmark_customers)
    |
    closed["card_id"].astype(str).isin(benchmark_cards)
].copy()


print("\n" + "=" * 100)
print("RELATED HISTORICAL CASES")
print("=" * 100)

print(f"Related historical cases: {len(related_cases)}")

if len(related_cases):
    
    display_cols = [
        "case_id",
        "customer_id",
        "card_id",
        "opened_at",
        "closed_at",
        "outcome",
        "pattern",
        "first_fraud_txn_id",
        "txn_ids",
        "n_txns",
        "exposure_usd",
        "connected_card_ids",
        "actions_taken",
        "report_filed",
        "analyst_notes",
    ]

    print(
        related_cases[
            display_cols
        ]
        .sort_values(["customer_id", "opened_at"])
        .to_string(index=False)
    )


print("\n" + "=" * 100)
print("DONE")
print("=" * 100)