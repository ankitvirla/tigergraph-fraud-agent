#!/usr/bin/env python3
"""
Audit the case-level probability training dataset before probability-model refinement.

Input:
    analysis/probability_training_dataset.csv

Outputs:
    analysis/probability_training_feature_audit.csv
    analysis/probability_training_risk_score_audit.csv
    analysis/probability_training_temporal_drift.csv
    analysis/probability_training_feature_correlations.csv
    analysis/probability_training_audit_summary.json
    analysis/probability_training_audit_notes.txt

Purpose:
    - Audit label prevalence and temporal distribution.
    - Measure univariate feature/label relationships.
    - Diagnose risk_score direction and calibration behavior.
    - Detect duplicate / highly correlated features.
    - Check feature sparsity and temporal drift.
    - Flag counterintuitive or potentially redundant features for review.

This is an audit only. It does not train or select a production model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


DEFAULT_INPUT = Path("analysis/probability_training_dataset.csv")
DEFAULT_OUTDIR = Path("analysis")

TARGET = "label"
DATE_COL = "opened_at"
RISK_SCORE = "risk_score"

FEATURES = [
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

SPLIT_NAMES = ("train", "validation", "test")


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_auc(y: pd.Series, x: pd.Series) -> float:
    mask = y.notna() & x.notna()
    yy = y.loc[mask]
    xx = x.loc[mask]
    if yy.nunique() < 2 or xx.nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy, xx))
    except Exception:
        return float("nan")


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3 or x.loc[mask].nunique() < 2 or y.loc[mask].nunique() < 2:
        return float("nan")
    try:
        return float(spearmanr(x.loc[mask], y.loc[mask]).statistic)
    except Exception:
        return float("nan")


def corrected_odds_ratio(a: int, b: int, c: int, d: int) -> float:
    # a = feature true & fraud
    # b = feature true & cleared
    # c = feature false & fraud
    # d = feature false & cleared
    # Haldane-Anscombe correction avoids infinite values.
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    return float((aa * dd) / (bb * cc))


def quantile_edges(series: pd.Series, q: int = 10) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return np.array([])
    edges = np.unique(vals.quantile(np.linspace(0, 1, q + 1)).to_numpy())
    if len(edges) < 2:
        return np.array([])
    return edges


def bucket_risk_score(df: pd.DataFrame, q: int = 10) -> pd.DataFrame:
    work = df[[RISK_SCORE, TARGET]].copy()
    work[RISK_SCORE] = pd.to_numeric(work[RISK_SCORE], errors="coerce")
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work = work.dropna()

    edges = quantile_edges(work[RISK_SCORE], q=q)
    if len(edges) < 2:
        return pd.DataFrame()

    # pd.cut requires strictly increasing edges; duplicates were removed above.
    work["bucket"] = pd.cut(
        work[RISK_SCORE],
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )

    rows = []
    grouped = work.groupby("bucket", observed=True)
    for i, (bucket, g) in enumerate(grouped, start=1):
        rows.append(
            {
                "bucket": i,
                "score_range": str(bucket),
                "n": int(len(g)),
                "fraud": int(g[TARGET].sum()),
                "cleared": int((1 - g[TARGET]).sum()),
                "fraud_rate": float(g[TARGET].mean()),
                "mean_risk_score": float(g[RISK_SCORE].mean()),
                "median_risk_score": float(g[RISK_SCORE].median()),
                "min_risk_score": float(g[RISK_SCORE].min()),
                "max_risk_score": float(g[RISK_SCORE].max()),
            }
        )
    return pd.DataFrame(rows)


def exact_duplicate_features(df: pd.DataFrame, features: List[str]) -> List[Dict]:
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            if a not in df.columns or b not in df.columns:
                continue
            aa = pd.to_numeric(df[a], errors="coerce")
            bb = pd.to_numeric(df[b], errors="coerce")
            if aa.equals(bb):
                pairs.append({"feature_a": a, "feature_b": b, "type": "exact_duplicate"})
    return pairs


def build_feature_audit(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    y = pd.to_numeric(df[TARGET], errors="coerce")
    total_fraud = int(y.sum())
    total_cleared = int((1 - y).sum())

    rows = []

    for feature in features:
        if feature not in df.columns:
            rows.append(
                {
                    "feature": feature,
                    "status": "missing",
                }
            )
            continue

        x = pd.to_numeric(df[feature], errors="coerce")
        present = x.fillna(0) != 0
        missing = x.isna()

        tp = int(((present) & (y == 1)).sum())
        fp = int(((present) & (y == 0)).sum())
        fn = int(((~present) & (y == 1)).sum())
        tn = int(((~present) & (y == 0)).sum())

        fraud_present = tp / (tp + fp) if tp + fp else float("nan")
        fraud_absent = fn / (fn + tn) if fn + tn else float("nan")
        risk_difference = (
            fraud_present - fraud_absent
            if not (math.isnan(fraud_present) or math.isnan(fraud_absent))
            else float("nan")
        )

        prevalence = float(present.mean())
        auc = safe_auc(y, x)
        odds_ratio = corrected_odds_ratio(tp, fp, fn, tn)
        spearman = safe_spearman(x, y)

        mean_fraud = float(x[y == 1].mean()) if (y == 1).any() else float("nan")
        mean_cleared = float(x[y == 0].mean()) if (y == 0).any() else float("nan")

        rows.append(
            {
                "feature": feature,
                "status": "ok",
                "dtype": str(df[feature].dtype),
                "n": int(len(df)),
                "missing_count": int(missing.sum()),
                "missing_rate": float(missing.mean()),
                "nonzero_count": int(present.sum()),
                "nonzero_rate": prevalence,
                "fraud_when_nonzero": fraud_present,
                "fraud_when_zero": fraud_absent,
                "fraud_rate_difference": risk_difference,
                "odds_ratio_nonzero_vs_zero": odds_ratio,
                "univariate_auc": auc,
                "spearman_with_label": spearman,
                "mean_value_fraud": mean_fraud,
                "mean_value_cleared": mean_cleared,
                "n_unique": int(x.nunique(dropna=True)),
                "all_zero": bool((x.fillna(0) == 0).all()),
            }
        )

    return pd.DataFrame(rows)


def temporal_drift(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = []

    for split in SPLIT_NAMES:
        g = df[df["_split"] == split]
        if g.empty:
            continue

        for feature in features:
            if feature not in df.columns:
                continue
            x = pd.to_numeric(g[feature], errors="coerce")
            out.append(
                {
                    "split": split,
                    "feature": feature,
                    "n": int(len(g)),
                    "mean": float(x.mean()),
                    "median": float(x.median()),
                    "std": float(x.std(ddof=0)),
                    "nonzero_rate": float((x.fillna(0) != 0).mean()),
                    "missing_rate": float(x.isna().mean()),
                    "q10": float(x.quantile(0.10)),
                    "q25": float(x.quantile(0.25)),
                    "q50": float(x.quantile(0.50)),
                    "q75": float(x.quantile(0.75)),
                    "q90": float(x.quantile(0.90)),
                }
            )

        out.append(
            {
                "split": split,
                "feature": "__LABEL__",
                "n": int(len(g)),
                "mean": float(g[TARGET].mean()),
                "median": float(g[TARGET].median()),
                "std": float(g[TARGET].std(ddof=0)),
                "nonzero_rate": float(g[TARGET].mean()),
                "missing_rate": 0.0,
                "q10": float(g[TARGET].quantile(0.10)),
                "q25": float(g[TARGET].quantile(0.25)),
                "q50": float(g[TARGET].quantile(0.50)),
                "q75": float(g[TARGET].quantile(0.75)),
                "q90": float(g[TARGET].quantile(0.90)),
            }
        )

    return pd.DataFrame(out)


def correlation_audit(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    x = df[features].apply(pd.to_numeric, errors="coerce")
    corr = x.corr(method="spearman")

    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            if a not in corr.index or b not in corr.columns:
                continue
            value = corr.loc[a, b]
            if pd.isna(value):
                continue
            rows.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "spearman_correlation": float(value),
                    "absolute_correlation": float(abs(value)),
                    "flag_high_correlation": bool(abs(value) >= 0.95),
                    "flag_moderate_correlation": bool(abs(value) >= 0.80),
                }
            )

    return pd.DataFrame(rows).sort_values(
        "absolute_correlation", ascending=False
    )


def assign_temporal_splits(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
    if work[DATE_COL].isna().any():
        raise ValueError(f"{DATE_COL} contains invalid/missing timestamps")

    work = work.sort_values(DATE_COL).reset_index(drop=True)
    n = len(work)

    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "validation"
    split[val_end:] = "test"
    work["_split"] = split

    return work


def summarize(df: pd.DataFrame, feature_audit: pd.DataFrame,
              risk_buckets: pd.DataFrame, corr: pd.DataFrame) -> Dict:
    y = pd.to_numeric(df[TARGET], errors="coerce")

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "fraud": int(y.sum()),
            "cleared": int((1 - y).sum()),
            "fraud_rate": float(y.mean()),
            "date_min": str(df[DATE_COL].min()),
            "date_max": str(df[DATE_COL].max()),
        },
        "splits": {},
        "risk_score": {},
        "feature_audit": {},
        "correlation_audit": {},
        "flags": [],
    }

    for split in SPLIT_NAMES:
        g = df[df["_split"] == split]
        if g.empty:
            continue
        summary["splits"][split] = {
            "rows": int(len(g)),
            "fraud": int(g[TARGET].sum()),
            "cleared": int((1 - g[TARGET]).sum()),
            "fraud_rate": float(g[TARGET].mean()),
            "date_min": str(g[DATE_COL].min()),
            "date_max": str(g[DATE_COL].max()),
        }

    rs = pd.to_numeric(df[RISK_SCORE], errors="coerce")
    auc_raw = safe_auc(y, rs)
    auc_inverted = safe_auc(y, 1 - rs)

    summary["risk_score"] = {
        "raw_auc": auc_raw,
        "inverted_auc_diagnostic": auc_inverted,
        "spearman_with_label": safe_spearman(rs, y),
        "mean_fraud": float(rs[y == 1].mean()),
        "mean_cleared": float(rs[y == 0].mean()),
        "median_fraud": float(rs[y == 1].median()),
        "median_cleared": float(rs[y == 0].median()),
        "min": float(rs.min()),
        "max": float(rs.max()),
        "bucket_count": int(len(risk_buckets)),
    }

    fa = feature_audit[feature_audit["status"] == "ok"].copy()
    summary["feature_audit"] = {
        "all_zero_features": fa.loc[fa["all_zero"], "feature"].tolist(),
        "high_missing_features": fa.loc[fa["missing_rate"] > 0.01, "feature"].tolist(),
        "strong_positive_univariate_features": fa.loc[
            fa["odds_ratio_nonzero_vs_zero"] >= 3, "feature"
        ].tolist(),
        "strong_negative_univariate_features": fa.loc[
            fa["odds_ratio_nonzero_vs_zero"] <= (1 / 3), "feature"
        ].tolist(),
    }

    high_corr = corr[corr["flag_high_correlation"]]
    moderate_corr = corr[corr["flag_moderate_correlation"]]
    summary["correlation_audit"] = {
        "high_correlation_pairs": high_corr[
            ["feature_a", "feature_b", "spearman_correlation"]
        ].to_dict("records"),
        "moderate_or_higher_pair_count": int(len(moderate_corr)),
    }

    # Diagnostic flags. These are review prompts, not model-selection decisions.
    if not math.isnan(auc_raw) and auc_raw < 0.25:
        summary["flags"].append(
            "risk_score has very low raw univariate AUC; inspect score direction/semantics before using it as a probability."
        )

    if not math.isnan(auc_inverted) and auc_inverted > 0.75:
        summary["flags"].append(
            "1-risk_score has materially higher diagnostic AUC than risk_score; verify whether risk_score is reversed or represents a different concept."
        )

    if summary["dataset"]["fraud_rate"] > 0.80:
        summary["flags"].append(
            "The historical case dataset has a high fraud prevalence; probability calibration may not transfer directly to a lower-prevalence production population."
        )

    if len(high_corr) > 0:
        summary["flags"].append(
            "Some model features are highly correlated; coefficient signs/importances may be unstable or hard to interpret in isolation."
        )

    return summary


def write_notes(summary: Dict, feature_audit: pd.DataFrame,
                risk_buckets: pd.DataFrame, corr: pd.DataFrame,
                path: Path) -> None:
    lines = []
    lines.append("FRAUD PROBABILITY TRAINING DATA AUDIT")
    lines.append("=" * 72)
    lines.append("")
    lines.append("This audit is diagnostic. It does not select a production model.")
    lines.append("")

    ds = summary["dataset"]
    lines.append(
        f"Rows: {ds['rows']} | Fraud: {ds['fraud']} | Cleared: {ds['cleared']} | "
        f"Fraud rate: {ds['fraud_rate']:.4f}"
    )
    lines.append(f"Date range: {ds['date_min']} -> {ds['date_max']}")
    lines.append("")

    lines.append("TEMPORAL SPLITS")
    lines.append("-" * 72)
    for split, info in summary["splits"].items():
        lines.append(
            f"{split}: rows={info['rows']} fraud={info['fraud']} "
            f"cleared={info['cleared']} fraud_rate={info['fraud_rate']:.4f} "
            f"range={info['date_min']} -> {info['date_max']}"
        )
    lines.append("")

    rs = summary["risk_score"]
    lines.append("RISK_SCORE DIAGNOSTIC")
    lines.append("-" * 72)
    lines.append(f"Raw AUC: {rs['raw_auc']:.6f}")
    lines.append(f"Inverted (1-risk_score) AUC: {rs['inverted_auc_diagnostic']:.6f}")
    lines.append(f"Spearman with label: {rs['spearman_with_label']:.6f}")
    lines.append(f"Mean fraud score: {rs['mean_fraud']:.6f}")
    lines.append(f"Mean cleared score: {rs['mean_cleared']:.6f}")
    lines.append("")
    lines.append("Risk-score quantile buckets:")
    if risk_buckets.empty:
        lines.append("  Unable to create buckets.")
    else:
        for _, r in risk_buckets.iterrows():
            lines.append(
                f"  B{int(r['bucket']):02d}: {r['score_range']} | "
                f"n={int(r['n'])} | fraud_rate={r['fraud_rate']:.4f} | "
                f"mean={r['mean_risk_score']:.4f}"
            )
    lines.append("")

    lines.append("FEATURE FLAGS")
    lines.append("-" * 72)
    for flag in summary["flags"]:
        lines.append(f"- {flag}")
    if not summary["flags"]:
        lines.append("- No automatic diagnostic flags.")
    lines.append("")

    lines.append("ALL-ZERO FEATURES")
    lines.append("- " + ", ".join(summary["feature_audit"]["all_zero_features"]))
    lines.append("")

    lines.append("HIGH CORRELATION PAIRS (|Spearman| >= 0.95)")
    lines.append("-" * 72)
    high = corr[corr["flag_high_correlation"]]
    if high.empty:
        lines.append("None.")
    else:
        for _, r in high.iterrows():
            lines.append(
                f"- {r['feature_a']} <-> {r['feature_b']}: "
                f"{r['spearman_correlation']:.6f}"
            )
    lines.append("")

    lines.append("REVIEW NOTES")
    lines.append("-" * 72)
    lines.append(
        "1. Univariate associations describe the historical case dataset; they are not causal effects."
    )
    lines.append(
        "2. Strong correlations can make multivariate coefficient signs difficult to interpret."
    )
    lines.append(
        "3. Risk-score inversion is a diagnostic only. Verify the upstream definition before changing its meaning."
    )
    lines.append(
        "4. High historical fraud prevalence is important when interpreting probability calibration."
    )
    lines.append(
        "5. HHG-001–020 are not included because they do not provide authoritative labels."
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit probability training dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to probability_training_dataset.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory for audit outputs",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("FRAUD PROBABILITY TRAINING DATA AUDIT")
    log("=" * 72)
    log(f"Input: {args.input}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    required = {TARGET, DATE_COL, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[TARGET].isna().any():
        raise ValueError(f"{TARGET} contains missing values")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0/1")

    log(f"Rows: {len(df):,}")
    log(f"Fraud: {int(df[TARGET].sum()):,}")
    log(f"Cleared: {int((1 - df[TARGET]).sum()):,}")

    df = assign_temporal_splits(df)

    log("Building feature audit...")
    feature_audit = build_feature_audit(df, FEATURES)

    log("Building risk-score quantile audit...")
    risk_buckets = bucket_risk_score(df, q=10)

    log("Building temporal distribution audit...")
    drift = temporal_drift(df, FEATURES)

    log("Building correlation audit...")
    corr = correlation_audit(df, FEATURES)

    duplicates = exact_duplicate_features(df, FEATURES)

    summary = summarize(df, feature_audit, risk_buckets, corr)
    summary["exact_duplicate_pairs"] = duplicates

    # Additional explicit overlap diagnostics.
    summary["expected_feature_overlap_checks"] = {
        "out_of_region_vs_out_of_region_behavior": {
            "exact_equal": bool(
                pd.to_numeric(df["out_of_region"], errors="coerce").equals(
                    pd.to_numeric(df["out_of_region_behavior"], errors="coerce")
                )
            )
        },
        "cnp_vs_cnp_new_device": {
            "cnp_implies_cnp_new_device_rate": float(
                (
                    (pd.to_numeric(df["cnp"], errors="coerce") != 0)
                    & (pd.to_numeric(df["cnp_new_device"], errors="coerce") != 0)
                ).sum()
                / max(
                    1,
                    (pd.to_numeric(df["cnp"], errors="coerce") != 0).sum(),
                )
            )
        },
    }

    # Add duplicate-pair flags to the human-readable flags.
    if duplicates:
        summary["flags"].append(
            f"Detected {len(duplicates)} exact duplicate feature pair(s); review whether both should remain as model inputs."
        )

    # Save CSVs.
    feature_audit_path = args.outdir / "probability_training_feature_audit.csv"
    risk_path = args.outdir / "probability_training_risk_score_audit.csv"
    drift_path = args.outdir / "probability_training_temporal_drift.csv"
    corr_path = args.outdir / "probability_training_feature_correlations.csv"
    summary_path = args.outdir / "probability_training_audit_summary.json"
    notes_path = args.outdir / "probability_training_audit_notes.txt"

    feature_audit.to_csv(feature_audit_path, index=False)
    risk_buckets.to_csv(risk_path, index=False)
    drift.to_csv(drift_path, index=False)
    corr.to_csv(corr_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    write_notes(summary, feature_audit, risk_buckets, corr, notes_path)

    log("")
    log("AUDIT COMPLETE")
    log(f"Feature audit: {feature_audit_path}")
    log(f"Risk score audit: {risk_path}")
    log(f"Temporal drift: {drift_path}")
    log(f"Feature correlations: {corr_path}")
    log(f"Summary: {summary_path}")
    log(f"Notes: {notes_path}")
    log("")
    log("Key diagnostic flags:")
    for flag in summary["flags"]:
        log(f"- {flag}")


if __name__ == "__main__":
    main()
