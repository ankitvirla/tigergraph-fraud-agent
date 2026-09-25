from pathlib import Path
import pandas as pd


DATA_DIR = Path("data")


def file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def profile_csv(name, usecols=None):
    path = DATA_DIR / name

    print("\n" + "=" * 80)
    print(f"{name}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return None

    print(f"File size: {file_size_mb(path):,.2f} MB")

    # Read only the header first
    header = pd.read_csv(path, nrows=0)
    print(f"Columns: {len(header.columns)}")

    print("\nColumns:")
    for i, col in enumerate(header.columns, 1):
        print(f"  {i:3}. {col}")

    print("\nReading data...")

    df = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    print("\nDtypes:")
    print(df.dtypes.to_string())

    print("\nNull counts:")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)

    if len(nulls):
        print(nulls.to_string())
    else:
        print("No nulls.")

    print("\nUnique counts:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique(dropna=True):,}")

    return df


# -------------------------------------------------------------------
# 1. TRANSACTIONS
# -------------------------------------------------------------------

transactions = profile_csv(
    "transactions.csv",
    usecols=[
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
)


if transactions is not None:

    print("\n" + "=" * 80)
    print("TRANSACTION ANALYSIS")
    print("=" * 80)

    print("\nDate range:")
    print("  min:", transactions["ts"].min())
    print("  max:", transactions["ts"].max())

    print("\nChannels:")
    print(transactions["channel"].value_counts(dropna=False).to_string())

    print("\nProduct codes:")
    print(transactions["ProductCD"].value_counts(dropna=False).to_string())

    print("\nRisk score:")
    print(transactions["risk_score"].describe().to_string())

    print("\nTransaction amount:")
    print(transactions["TransactionAmt"].describe().to_string())

    print("\nTop customers by transaction count:")
    print(
        transactions["customer_id"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nTop card1 values:")
    print(
        transactions["card1"]
        .value_counts()
        .head(20)
        .to_string()
    )

    print("\nBilling regions:")
    print(
        transactions["addr1"]
        .value_counts()
        .head(20)
        .to_string()
    )


# -------------------------------------------------------------------
# 2. IDENTITY
# -------------------------------------------------------------------

identity = profile_csv(
    "identity.csv",
    usecols=[
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
)


if identity is not None:

    print("\n" + "=" * 80)
    print("IDENTITY ANALYSIS")
    print("=" * 80)

    print("\nTransactionID uniqueness:")
    print(
        f"  Rows: {len(identity):,}"
    )
    print(
        f"  Unique TransactionID: "
        f"{identity['TransactionID'].nunique():,}"
    )

    for col in [
        "DeviceType",
        "DeviceInfo",
        "id_15",
        "id_23",
        "id_30",
        "id_31",
        "id_33",
        "id_34",
    ]:
        print(f"\n{col}:")
        print(
            identity[col]
            .value_counts(dropna=False)
            .head(20)
            .to_string()
        )


# -------------------------------------------------------------------
# 3. CLOSED CASES
# -------------------------------------------------------------------

closed = profile_csv(
    "closed_cases_history.csv"
)


if closed is not None:

    print("\n" + "=" * 80)
    print("CLOSED CASE ANALYSIS")
    print("=" * 80)

    for col in [
        "outcome",
        "pattern",
        "report_filed",
    ]:
        if col in closed.columns:
            print(f"\n{col}:")
            print(
                closed[col]
                .value_counts(dropna=False)
                .to_string()
            )

    if "exposure_usd" in closed.columns:
        print("\nExposure:")
        print(
            closed["exposure_usd"]
            .describe()
            .to_string()
        )


# -------------------------------------------------------------------
# 4. CASE PACK
# -------------------------------------------------------------------

case_pack = profile_csv(
    "case_pack.csv"
)


if case_pack is not None:

    print("\n" + "=" * 80)
    print("CASE PACK")
    print("=" * 80)

    print(case_pack.to_string(index=False))


print("\n" + "=" * 80)
print("PROFILE COMPLETE")
print("=" * 80)