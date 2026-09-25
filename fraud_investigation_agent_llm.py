#!/usr/bin/env python3
"""
fraud_investigation_agent.py

Deterministic investigation orchestrator.

Flow:
    benchmark/calibrated evidence
        -> live TigerGraph MCP evidence
        -> 18 probability features
        -> HGB + validation-only Platt calibration
        -> investigation uncertainty
        -> next investigation action

Important:
- This agent does not make a final fraud/clear verdict.
- risk_score is an upstream feature, not a probability.
- HHG benchmark cases have no authoritative current-case labels.
- The probability model is the documented case-level historical model.
- Whenever calibrated_evidence.json is available, it is preferred because
  it preserves signal direction and independence_group exactly.
- CSV fallback uses the exact flattened signal names from the calibration
  script and derives direction only from documented signal semantics.
"""

from __future__ import annotations

import argparse
import urllib.request
import urllib.error
import asyncio
import json
import os
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "analysis"

# Prefer JSON because it preserves the full calibrated evidence objects.
JSON_CANDIDATES = [
    ANALYSIS_DIR / "calibrated_evidence.json",
    ANALYSIS_DIR / "benchmark_calibrated_evidence.json",
    ANALYSIS_DIR / "calibrated_cases.json",
]

CSV_CANDIDATES = [
    ANALYSIS_DIR / "calibrated_evidence.csv",
    ANALYSIS_DIR / "benchmark_summary.csv",
]

PROBABILITY_TRAINING_FILE = (
    ANALYSIS_DIR / "probability_training_dataset.csv"
)

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

RANDOM_STATE = 42


# ============================================================================
# Generic helpers
# ============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(
    value: Any,
    default: Optional[float] = None,
) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def clean(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]

    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()

    if isinstance(value, np.generic):
        return clean(value.item())

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def first(
    row: pd.Series,
    names: list[str],
    default: Any = None,
) -> Any:
    for name in names:
        if name not in row.index:
            continue

        value = row[name]

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        return value

    return default


def parse_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "[]"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass

    return [
        x.strip().strip("'\"")
        for x in text.replace(";", ",").split(",")
        if x.strip()
    ]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unwrap_cases(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [
            item for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):
        for key in (
            "cases",
            "calibrated_cases",
            "results",
            "benchmark_cases",
        ):
            value = data.get(key)
            if isinstance(value, list):
                return [
                    item for item in value
                    if isinstance(item, dict)
                ]

        # Single calibrated case.
        if "case_id" in data:
            return [data]

    return []


# ============================================================================
# Calibrated evidence
# ============================================================================

class EvidenceStore:
    """
    Loads the original calibrated evidence.

    JSON is preferred because the calibration script's canonical case object
    contains:
        evidence[]
        evidence_summary{}
        direction
        independence_group

    The CSV is a flattened compatibility fallback.
    """

    def __init__(self):
        self.source_type = None
        self.path = None
        self.cases: dict[str, dict[str, Any]] = {}

        for path in JSON_CANDIDATES:
            if path.exists():
                cases = unwrap_cases(load_json(path))
                if cases:
                    self.source_type = "json"
                    self.path = path
                    for case in cases:
                        case_id = str(case.get("case_id", "")).strip()
                        if case_id:
                            self.cases[case_id] = case
                    return

        for path in CSV_CANDIDATES:
            if path.exists():
                self.source_type = "csv"
                self.path = path
                df = pd.read_csv(path, low_memory=False)

                if "case_id" not in df.columns:
                    continue

                df["case_id"] = df["case_id"].astype(str)

                for _, row in df.iterrows():
                    case_id = str(row["case_id"])
                    self.cases[case_id] = self._csv_case(row)

                if self.cases:
                    return

        searched = (
            [str(p) for p in JSON_CANDIDATES]
            + [str(p) for p in CSV_CANDIDATES]
        )

        raise FileNotFoundError(
            "No calibrated evidence file was found.\n"
            "Searched:\n  - " + "\n  - ".join(searched)
        )

    def get_case(self, case_id: str) -> dict[str, Any]:
        case_id = str(case_id).strip()

        if case_id not in self.cases:
            raise KeyError(
                f"{case_id} was not found in {self.path}."
            )

        return self.cases[case_id]

    @staticmethod
    def _csv_case(row: pd.Series) -> dict[str, Any]:
        """
        Reconstruct the canonical calibrated structure from the exact CSV
        flattening names used by benchmark_evidence_calibration.py.

        Crucially, we do NOT treat all *_strength signals as supporting.
        Direction is assigned only according to the documented calibration
        semantics.
        """

        raw = {
            "case_id": clean(first(row, ["case_id"])),
            "trigger_type": clean(first(row, ["trigger_type"])),
            "flagged_txn_id": clean(
                first(row, ["flagged_txn_id", "txn_id", "transaction_id"])
            ),
            "customer_id": clean(first(row, ["customer_id"])),
            "card_id": clean(first(row, ["card_id"])),
            "opened_at": clean(
                first(row, ["opened_at", "case_opened_at"])
            ),
            "risk_score": safe_float(
                first(row, ["risk_score"])
            ),
            "amount": safe_float(
                first(
                    row,
                    ["amount", "TransactionAmt", "transaction_amount"],
                )
            ),
            "channel": clean(first(row, ["channel"])),
            "ProductCD": clean(first(row, ["ProductCD"])),
        }

        # CSV flattening names -> canonical evidence signal names.
        mapping = {
            "card_testing_strength": (
                "card_testing_pattern",
                "supports_fraud",
            ),
            "cnp_strength": (
                "card_not_present_pattern",
                "supports_fraud",
            ),
            "cnp_new_device_strength": (
                "cnp_new_device_pattern",
                "supports_fraud",
            ),
            "out_of_region_strength": (
                "out_of_region_pattern",
                "supports_fraud",
            ),
            "account_takeover_strength": (
                "account_takeover_pattern",
                "supports_fraud",
            ),
            "new_device_strength": (
                "new_device_indicator",
                "supports_fraud",
            ),
            "proxy_network_strength": (
                "proxy_network_indicator",
                "supports_fraud",
            ),
            "shared_device_profile_strength": (
                "shared_device_profile",
                "supports_fraud",
            ),
            "channel_novelty_strength": (
                "channel_novelty",
                "supports_fraud",
            ),
            "product_novelty_strength": (
                "product_novelty",
                "supports_fraud",
            ),
            "out_of_region_behavior_strength": (
                "out_of_region_behavior",
                "supports_fraud",
            ),
            "post_flagged_activity_strength": (
                "post_flagged_activity",
                "context",
            ),
            "email_novelty_strength": (
                "email_novelty",
                "context",
            ),
            "identity_match_status_strength": (
                "identity_match_status",
                "context",
            ),
            "historical_cleared_signal": (
                "historical_cleared_cases",
                "contradicts_fraud",
            ),
            "repeated_historical_abuse_strength": (
                "repeated_historical_abuse",
                "supports_fraud",
            ),
        }

        evidence = []

        for column, (signal, direction) in mapping.items():
            if column not in row.index:
                continue

            strength = row[column]

            try:
                if pd.isna(strength):
                    continue
            except (TypeError, ValueError):
                pass

            strength = str(strength).strip()

            if not strength or strength.lower() == "nan":
                continue

            evidence.append({
                "signal_type": signal,
                "strength": strength,
                "direction": direction,
                "independence_group": (
                    EvidenceStore._independence_group(signal)
                ),
                "source": "calibrated_evidence.csv",
                "raw": {},
                "explanation": None,
            })

        # The flattened CSV contains authoritative aggregate counts. Use them
        # for counts instead of reconstructing counts from the signal columns.
        strong_total = safe_int(
            first(
                row,
                ["strong_evidence_count", "strong_count"],
            )
        )
        medium_total = safe_int(
            first(
                row,
                ["medium_evidence_count", "medium_count"],
            )
        )
        weak_total = safe_int(
            first(
                row,
                ["weak_evidence_count", "weak_count"],
            )
        )

        supporting_count = safe_int(
            first(row, ["supporting_evidence_count", "supporting_count"])
        )
        contradicting_count = safe_int(
            first(
                row,
                ["contradicting_evidence_count", "contradicting_count"],
            )
        )
        context_count = safe_int(
            first(row, ["context_evidence_count", "context_count"])
        )

        independent_groups = parse_list(
            first(row, ["independent_evidence_groups"])
        )

        independent_count = safe_int(
            first(row, ["independent_evidence_count"]),
            len(independent_groups),
        )

        summary = {
            # Do not call total strong/medium/weak counts "support counts".
            "strong_count": strong_total,
            "medium_count": medium_total,
            "weak_count": weak_total,
            "supporting_count": supporting_count,
            "contradicting_count": contradicting_count,
            "context_count": context_count,
            "medium_contradiction_count": safe_int(
                first(row, ["medium_contradiction_count"])
            ),
            "independent_evidence_groups": independent_groups,
            "independent_evidence_count": independent_count,
        }

        return {
            "case_id": str(raw["case_id"]),
            "raw_context": raw,
            "evidence": evidence,
            "evidence_summary": summary,
            "source_type": "csv_fallback",
            "source_path": str(
                ANALYSIS_DIR / "calibrated_evidence.csv"
            ),
            "mapping_note": (
                "CSV is a flattened representation. Signal direction is "
                "reconstructed only from documented calibration semantics; "
                "aggregate counts are taken directly from the CSV."
            ),
        }

    @staticmethod
    def _independence_group(signal: str) -> Optional[str]:
        groups = {
            "card_testing_pattern": "behavioral_pattern",
            "card_not_present_pattern": "behavioral_pattern",
            "cnp_new_device_pattern": "behavioral_pattern",
            "channel_novelty": "behavioral_pattern",
            "product_novelty": "behavioral_pattern",
            "out_of_region_pattern": "geographic_behavior",
            "account_takeover_pattern": "behavioral_identity",
            "new_device_indicator": "identity_device",
            "proxy_network_indicator": "identity_network",
            "shared_device_profile": "network",
            "out_of_region_behavior": "behavioral_pattern",
            "repeated_historical_abuse": "historical",
            "historical_case_history": "historical",
            "historical_cleared_cases": "historical",
        }
        return groups.get(signal)


# ============================================================================
# Exact canonical JSON evidence normalization
# ============================================================================

def normalize_json_case(case: dict[str, Any]) -> dict[str, Any]:
    raw = dict(case.get("raw_context") or {})
    raw.setdefault("case_id", case.get("case_id"))

    evidence = case.get("evidence") or []

    normalized_evidence = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        normalized_evidence.append({
            "signal_type": item.get("signal_type"),
            "strength": item.get("strength"),
            "direction": item.get("direction"),
            "independence_group": item.get("independence_group"),
            "source": item.get("source"),
            "raw": item.get("raw") or {},
            "explanation": item.get("explanation"),
        })

    summary = dict(case.get("evidence_summary") or {})

    # Canonical calibration output has strong/medium/weak total counts.
    # Build support counts separately from the evidence list.
    supporting = [
        x for x in normalized_evidence
        if x.get("direction") == "supports_fraud"
    ]
    contradicting = [
        x for x in normalized_evidence
        if x.get("direction") == "contradicts_fraud"
    ]
    context = [
        x for x in normalized_evidence
        if x.get("direction") == "context"
    ]

    summary.setdefault("strong_count", sum(
        x.get("strength") == "strong"
        for x in normalized_evidence
    ))
    summary.setdefault("medium_count", sum(
        x.get("strength") == "medium"
        for x in normalized_evidence
    ))
    summary.setdefault("weak_count", sum(
        x.get("strength") == "weak"
        for x in normalized_evidence
    ))

    # These are recomputed from canonical evidence, preventing a stale or
    # inconsistent summary from contaminating the probability features.
    summary["supporting_count"] = len(supporting)
    summary["contradicting_count"] = len(contradicting)
    summary["context_count"] = len(context)

    groups = sorted({
        x.get("independence_group")
        for x in supporting
        if (
            x.get("strength") in {"strong", "medium"}
            and x.get("independence_group")
        )
    })

    summary["independent_evidence_groups"] = groups
    summary["independent_evidence_count"] = len(groups)

    summary["medium_contradiction_count"] = sum(
        x.get("strength") == "medium"
        for x in contradicting
    )

    return {
        "case_id": str(case.get("case_id")),
        "raw_context": raw,
        "evidence": normalized_evidence,
        "evidence_summary": summary,
        "source_type": "canonical_json",
    }


# ============================================================================
# Probability model
# ============================================================================

class ProbabilityModel:
    """
    Rebuild the final documented HGB + Platt model.

    Split:
        chronological 60% train
        20% validation
        20% untouched test

    Platt calibration:
        validation only
    """

    def __init__(
        self,
        path: Path = PROBABILITY_TRAINING_FILE,
    ):
        self.path = path
        self.model = None
        self.platt = None

    def fit(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing probability dataset: {self.path}"
            )

        df = pd.read_csv(
            self.path,
            low_memory=False,
        )

        target = next(
            (
                c for c in
                ["target", "is_fraud", "label", "fraud_label"]
                if c in df.columns
            ),
            None,
        )

        if target is None:
            raise ValueError(
                "No target column found in probability_training_dataset.csv"
            )

        missing = [
            feature
            for feature in FEATURES
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Probability dataset is missing: {missing}"
            )

        if "opened_at" not in df.columns:
            raise ValueError(
                "probability_training_dataset.csv needs opened_at"
            )

        df["opened_at"] = pd.to_datetime(
            df["opened_at"],
            errors="coerce",
        )

        df[target] = pd.to_numeric(
            df[target],
            errors="coerce",
        )

        df = (
            df
            .dropna(subset=["opened_at", target])
            .sort_values("opened_at")
            .reset_index(drop=True)
        )

        train_end = int(len(df) * 0.60)
        validation_end = int(len(df) * 0.80)

        train = df.iloc[:train_end]
        validation = df.iloc[train_end:validation_end]

        X_train = train[FEATURES].apply(
            pd.to_numeric,
            errors="coerce",
        )
        X_validation = validation[FEATURES].apply(
            pd.to_numeric,
            errors="coerce",
        )

        y_train = train[target].astype(int).to_numpy()
        y_validation = validation[target].astype(int).to_numpy()

        self.model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )

        self.model.fit(
            X_train,
            y_train,
        )

        raw_validation = np.clip(
            self.model.predict_proba(X_validation)[:, 1],
            1e-7,
            1.0 - 1e-7,
        )

        logits = np.log(
            raw_validation / (1.0 - raw_validation)
        ).reshape(-1, 1)

        self.platt = LogisticRegression(
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

        self.platt.fit(
            logits,
            y_validation,
        )

    def predict(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:

        if self.model is None or self.platt is None:
            self.fit()

        values = {}

        for feature in FEATURES:
            value = safe_float(
                features.get(feature),
                0.0,
            )

            values[feature] = value

        X = pd.DataFrame(
            [values],
            columns=FEATURES,
        )

        raw_probability = float(
            self.model.predict_proba(X)[0, 1]
        )

        clipped = np.clip(
            raw_probability,
            1e-7,
            1.0 - 1e-7,
        )

        logit = np.log(
            clipped / (1.0 - clipped)
        )

        probability = float(
            self.platt.predict_proba(
                [[logit]]
            )[0, 1]
        )

        return {
            "fraud_probability": probability,
            "raw_model_probability": raw_probability,
            "model_name": (
                "hist_gradient_boosting_"
                "sigmoid_calibrated"
            ),
            "feature_count": len(FEATURES),
            "calibration": "sigmoid_platt",
            "calibration_fit": "historical_validation_only",
            "scope": "case_level_historical_fraud_outcome",
        }


# ============================================================================
# Evidence -> model features
# ============================================================================

def build_features(
    case: dict[str, Any],
) -> dict[str, Any]:

    raw = case["raw_context"]
    summary = case["evidence_summary"]
    evidence = case["evidence"]

    def supporting_signal(name: str) -> int:
        return int(any(
            item.get("signal_type") == name
            and item.get("direction") == "supports_fraud"
            for item in evidence
        ))

    return {
        "risk_score": raw.get("risk_score"),

        "strong_support_count": sum(
            item.get("strength") == "strong"
            and item.get("direction") == "supports_fraud"
            for item in evidence
        ),

        "medium_support_count": sum(
            item.get("strength") == "medium"
            and item.get("direction") == "supports_fraud"
            for item in evidence
        ),

        "weak_support_count": sum(
            item.get("strength") == "weak"
            and item.get("direction") == "supports_fraud"
            for item in evidence
        ),

        "medium_contradiction_count": summary.get(
            "medium_contradiction_count",
            0,
        ),

        "context_evidence_count": summary.get(
            "context_count",
            0,
        ),

        "independent_evidence_count": summary.get(
            "independent_evidence_count",
            0,
        ),

        "card_testing": supporting_signal(
            "card_testing_pattern"
        ),

        "cnp": supporting_signal(
            "card_not_present_pattern"
        ),

        "cnp_new_device": supporting_signal(
            "cnp_new_device_pattern"
        ),

        "account_takeover": supporting_signal(
            "account_takeover_pattern"
        ),

        "out_of_region": supporting_signal(
            "out_of_region_pattern"
        ),

        "new_device": supporting_signal(
            "new_device_indicator"
        ),

        "proxy_network": supporting_signal(
            "proxy_network_indicator"
        ),

        "channel_novelty": supporting_signal(
            "channel_novelty"
        ),

        "product_novelty": supporting_signal(
            "product_novelty"
        ),

        "repeated_historical_abuse": supporting_signal(
            "repeated_historical_abuse"
        ),

        "shared_device_profile": supporting_signal(
            "shared_device_profile"
        ),
    }


def evidence_view(
    case: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:

    evidence = case["evidence"]

    return {
        "strong": [
            x for x in evidence
            if x.get("strength") == "strong"
        ],
        "medium": [
            x for x in evidence
            if x.get("strength") == "medium"
        ],
        "weak": [
            x for x in evidence
            if x.get("strength") == "weak"
        ],
        "not_material": [
            x for x in evidence
            if x.get("strength") == "not_material"
        ],
        "supporting": [
            x for x in evidence
            if x.get("direction") == "supports_fraud"
        ],
        "contradictions": [
            x for x in evidence
            if x.get("direction") == "contradicts_fraud"
        ],
        "context": [
            x for x in evidence
            if x.get("direction") == "context"
        ],
    }


# ============================================================================
# TigerGraph MCP
# ============================================================================

async def run_graph_investigation(
    case: dict[str, Any],
) -> dict[str, Any]:

    from tigergraph_client import TigerGraphMCPClient

    client = TigerGraphMCPClient()

    customer_id = case["raw_context"].get(
        "customer_id"
    )

    transaction_id = case["raw_context"].get(
        "flagged_txn_id"
    )

    if not customer_id:
        raise ValueError(
            f"{case['case_id']} has no customer_id"
        )

    if not transaction_id:
        raise ValueError(
            f"{case['case_id']} has no flagged_txn_id"
        )

    profile, transaction, shared = await asyncio.gather(
        client.get_customer_profile(
            str(customer_id)
        ),
        client.get_transaction_context(
            str(transaction_id)
        ),
        client.get_shared_device_customers(
            str(customer_id)
        ),
    )

    return {
        "customer_profile": profile,
        "transaction_context": transaction,
        "shared_device_network": shared,
    }


def graph_entities(
    value: Any,
    keys: set[str],
) -> list[dict[str, Any]]:

    found = []

    if isinstance(value, dict):
        if keys.intersection(value.keys()):
            found.append(value)

        for child in value.values():
            found.extend(
                graph_entities(child, keys)
            )

    elif isinstance(value, list):
        for child in value:
            found.extend(
                graph_entities(child, keys)
            )

    unique = []
    seen = set()

    for item in found:
        key = json.dumps(
            clean(item),
            sort_keys=True,
            default=str,
        )

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def graph_facts(
    graph: dict[str, Any],
) -> dict[str, Any]:

    profile = graph.get(
        "customer_profile",
        {},
    )

    transaction = graph.get(
        "transaction_context",
        {},
    )

    shared = graph.get(
        "shared_device_network",
        {},
    )

    return {
        "customer_profile_success": bool(
            profile.get("success")
        ) if isinstance(profile, dict) else False,

        "transaction_context_success": bool(
            transaction.get("success")
        ) if isinstance(transaction, dict) else False,

        "shared_device_query_success": bool(
            shared.get("success")
        ) if isinstance(shared, dict) else False,

        "shared_customer_count": len(
            graph_entities(
                shared,
                {"customer_id"},
            )
        ),

        "device_count": len(
            graph_entities(
                shared,
                {"device_id"},
            )
        ),

        "historical_case_count": len(
            graph_entities(
                transaction,
                {"case_id"},
            )
        ),
    }


# ============================================================================
# Uncertainty + next action
# ============================================================================

def assess_uncertainty(
    case: dict[str, Any],
    facts: dict[str, Any],
) -> dict[str, Any]:

    summary = case["evidence_summary"]

    independent = safe_int(
        summary.get("independent_evidence_count")
    )

    supporting = safe_int(
        summary.get("supporting_count")
    )

    contradictions = safe_int(
        summary.get("contradicting_count")
    )

    reasons = []

    if not facts["customer_profile_success"]:
        reasons.append(
            "Customer graph profile was not successfully retrieved."
        )

    if not facts["transaction_context_success"]:
        reasons.append(
            "Transaction graph context was not successfully retrieved."
        )

    if independent == 0 and supporting == 0:
        reasons.append(
            "No supporting evidence group is currently established."
        )

    elif independent == 1:
        reasons.append(
            "Only one independent supporting evidence group is present."
        )

    if contradictions > 0:
        reasons.append(
            f"{contradictions} contradictory evidence signal(s) "
            "are present."
        )

    if (
        not facts["customer_profile_success"]
        or not facts["transaction_context_success"]
    ):
        level = "high"
    elif independent == 0:
        level = "high"
    elif independent == 1 or contradictions > 0:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "independent_evidence_count": independent,
        "supporting_evidence_count": supporting,
        "contradicting_evidence_count": contradictions,
        "reasons": reasons,
    }


def choose_next_action(
    uncertainty: dict[str, Any],
    facts: dict[str, Any],
) -> dict[str, str]:

    if not facts["transaction_context_success"]:
        return {
            "action": "retrieve_transaction_context",
            "reason": (
                "Transaction graph context is unavailable; "
                "retrieve it before completing the investigation."
            ),
        }

    if not facts["customer_profile_success"]:
        return {
            "action": "retrieve_customer_profile",
            "reason": (
                "Customer graph profile is unavailable; "
                "retrieve it before completing the investigation."
            ),
        }

    if uncertainty["level"] == "high":
        return {
            "action": "gather_additional_evidence",
            "reason": (
                "Current supporting evidence is insufficient "
                "to complete the investigation."
            ),
        }

    if uncertainty["level"] == "medium":
        return {
            "action": "review_evidence_and_network_context",
            "reason": (
                "Evidence is currently concentrated in a limited "
                "number of independent evidence groups."
                if uncertainty["contradicting_evidence_count"] == 0
                else (
                    "The case contains contradictory evidence that "
                    "should be reviewed together with network context."
                )
            ),
        }

    return {
        "action": "prepare_investigator_review",
        "reason": (
            "Current evidence and graph context are sufficiently "
            "complete for investigator review."
        ),
    }


# ============================================================================
# Local Ollama LLM explanation layer
# ============================================================================

class OllamaReasoner:
    """Thin local LLM layer. It explains grounded results; it never changes
    the deterministic evidence, probability, uncertainty, or action."""

    def __init__(self, model: Optional[str] = None):
        self.base_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.model = model or os.getenv("OLLAMA_MODEL")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT", "45"))

    def _resolve_model(self) -> Optional[str]:
        if self.model:
            return self.model
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            names = [x.get("name") for x in data.get("models", []) if x.get("name")]
            # Prefer Qwen 2.5 if installed; otherwise use the first local model.
            for name in names:
                if "qwen2.5" in name.lower():
                    self.model = name
                    return name
            if names:
                self.model = names[0]
                return self.model
        except Exception:
            pass
        return None

    def explain(self, investigation: dict[str, Any]) -> dict[str, Any]:
        model = self._resolve_model()
        if not model:
            return {
                "status": "unavailable",
                "provider": "ollama",
                "model": None,
                "message": "Local Ollama model was not reachable; deterministic investigation remains authoritative.",
            }

        evidence = investigation["evidence"]
        risk = investigation["risk_assessment"]
        uncertainty = investigation["uncertainty"]
        action = investigation["next_action"]
        facts = investigation["graph_investigation"]["facts"]

        grounded = {
            "case_id": investigation["case_id"],
            "customer_id": investigation["case"].get("customer_id"),
            "transaction_id": investigation["case"].get("flagged_txn_id"),
            "fraud_probability": risk.get("fraud_probability"),
            "uncertainty": uncertainty,
            "next_action": action,
            "graph_facts": facts,
            "supporting_evidence": evidence.get("supporting", []),
            "contradicting_evidence": evidence.get("contradictions", []),
            "context_evidence": evidence.get("context", []),
            "independent_evidence_groups": investigation["evidence_summary"].get("independent_evidence_groups", []),
        }

        prompt = """You are a fraud investigation assistant. Summarize ONLY the supplied grounded investigation facts.
Do not invent evidence, transactions, relationships, policy rules, or actions. Do not recalculate or alter the fraud probability.
The deterministic system is authoritative for probability, evidence strength, uncertainty, and next action.
Return concise JSON with exactly these keys:
summary, key_findings, uncertainty_explanation, recommended_next_step
where key_findings is an array of 2-5 short strings.

GROUNDED INVESTIGATION:
""" + json.dumps(grounded, ensure_ascii=False, default=str, indent=2)

        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1},
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            text = raw.get("response", "").strip()
            parsed = json.loads(text) if text else {}
            return {
                "status": "success",
                "provider": "ollama",
                "model": model,
                "summary": parsed.get("summary", ""),
                "key_findings": parsed.get("key_findings", []),
                "uncertainty_explanation": parsed.get("uncertainty_explanation", ""),
                "recommended_next_step": parsed.get("recommended_next_step", ""),
            }
        except Exception as exc:
            return {
                "status": "error",
                "provider": "ollama",
                "model": model,
                "message": f"Ollama explanation failed: {type(exc).__name__}",
            }


async def run_llm_explanation(investigation: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(OllamaReasoner().explain, investigation)


# ============================================================================
# Main agent
# ============================================================================

async def investigate_case(
    case_id: str,
    evidence_store: Optional[EvidenceStore] = None,
    probability_model: Optional[ProbabilityModel] = None,
) -> dict[str, Any]:

    case_id = str(case_id).strip()

    trace = [
        "Loading calibrated case evidence."
    ]

    store = evidence_store or EvidenceStore()
    loaded_case = store.get_case(case_id)

    # Normalize canonical JSON; CSV fallback is already normalized.
    if loaded_case.get("source_type") == "canonical_json":
        case = normalize_json_case(
            loaded_case
        )
    else:
        case = loaded_case

    trace.append(
        f"Loaded {case_id} for customer "
        f"{case['raw_context'].get('customer_id')}."
    )

    trace.append(
        f"Evidence source: {store.source_type}."
    )

    trace.append(
        "Running TigerGraph MCP investigation."
    )

    graph = await run_graph_investigation(
        case
    )

    facts = graph_facts(graph)

    trace.extend([
        "Retrieved customer profile.",
        "Retrieved transaction context.",
        "Retrieved shared-device network context.",
        "Merged calibrated evidence with live graph context.",
    ])

    features = build_features(
        case
    )

    trace.append(
        f"Built {len(FEATURES)} probability-model features."
    )

    model = (
        probability_model
        or ProbabilityModel()
    )

    risk = model.predict(
        features
    )

    trace.append(
        "Applied historical HGB + validation-only Platt calibration."
    )

    uncertainty = assess_uncertainty(
        case,
        facts,
    )

    trace.append(
        f"Assessed investigation uncertainty as "
        f"{uncertainty['level']}."
    )

    action = choose_next_action(
        uncertainty,
        facts,
    )

    trace.append(
        f"Selected next action: {action['action']}."
    )

    result = clean({
        "case_id": case_id,
        "status": "investigation_complete",

        "case": case["raw_context"],

        "evidence_source": {
            "type": store.source_type,
            "path": str(store.path),
        },

        "graph_investigation": {
            "facts": facts,
            "raw": graph,
        },

        "evidence": evidence_view(
            case
        ),

        "evidence_summary": case[
            "evidence_summary"
        ],

        "model_features": features,

        "risk_assessment": risk,

        "uncertainty": uncertainty,

        "next_action": action,

        "investigation_trace": trace,

        "scope_notes": [
            (
                "Probability is a case-level estimate based on "
                "historical closed-case outcomes."
            ),
            (
                "HHG-001 through HHG-020 do not have authoritative "
                "current-case labels."
            ),
            (
                "risk_score is an upstream model input and is not "
                "itself a fraud probability."
            ),
            (
                "Independent evidence count uses only supporting "
                "strong/medium evidence and unique independence groups."
            ),
            (
                "Graph relationships are reported from TigerGraph "
                "retrieval; unsupported relationships are not inferred."
            ),
        ],
    })

    trace.append("Generating grounded investigator explanation with local Ollama.")
    result["llm_explanation"] = await run_llm_explanation(result)
    if result["llm_explanation"].get("status") == "success":
        trace.append("Generated grounded LLM explanation.")
    else:
        trace.append("LLM explanation unavailable; deterministic investigation remains authoritative.")
    result["investigation_trace"] = trace

    return result


def print_compact(result: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("FRAUD INVESTIGATION AGENT")
    print("=" * 72)

    print(
        f"Case:        {result['case_id']}"
    )

    print(
        f"Customer:    "
        f"{result['case'].get('customer_id')}"
    )

    print(
        f"Transaction: "
        f"{result['case'].get('flagged_txn_id')}"
    )

    print(
        f"Probability: "
        f"{result['risk_assessment']['fraud_probability']:.6f}"
    )

    print(
        f"Uncertainty: "
        f"{result['uncertainty']['level']}"
    )

    evidence = result["evidence"]

    print(
        "Evidence:    "
        f"strong={len(evidence['strong'])}, "
        f"medium={len(evidence['medium'])}, "
        f"weak={len(evidence['weak'])}, "
        f"supporting={len(evidence['supporting'])}, "
        f"contradictions={len(evidence['contradictions'])}, "
        f"context={len(evidence['context'])}"
    )

    print(
        "Independent: "
        f"{result['uncertainty']['independent_evidence_count']}"
    )

    print(
        f"Next action: "
        f"{result['next_action']['action']}"
    )

    llm = result.get("llm_explanation", {})
    print(
        "LLM:         "
        f"{llm.get('status', 'unknown')} / {llm.get('model', 'n/a')}"
    )
    if llm.get("summary"):
        print(f"Summary:      {llm['summary']}")

    print()
    print("Investigation trace:")

    for index, item in enumerate(
        result["investigation_trace"],
        1,
    ):
        print(
            f"  {index}. {item}"
        )


async def async_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic fraud investigation."
    )

    parser.add_argument(
        "case_id",
        nargs="?",
        default="HHG-001",
    )

    parser.add_argument(
        "--json-only",
        action="store_true",
    )

    args = parser.parse_args()

    result = await investigate_case(
        args.case_id
    )

    if args.json_only:
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print_compact(result)
        print()
        print("FULL JSON")
        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    asyncio.run(async_main())
