from __future__ import annotations

from typing import Any

import pandas as pd

from .constants import N_BOOTSTRAP
from .inference import clustered_bootstrap_ci, risk_difference_and_odds_ratio


SEVERITY_ORDER = ["cosmetic", "minor", "moderate", "serious", "critical"]


def compute_severity_concentration(item_df: pd.DataFrame, *, rng_bootstrap, n_bootstrap: int = N_BOOTSTRAP) -> dict[str, Any]:
    if item_df.empty or "severity" not in item_df:
        return {"status": "ok", "evidence_class": "estimation", "by_severity": {}, "second_derivation": {"match_rate": None, "mismatches": []}}
    data = item_df[item_df["failed"].notna()].copy()
    by_severity = {}
    for severity, group in data.groupby("severity"):
        values = group["failed"].astype(bool).tolist()
        interval = clustered_bootstrap_ci(values, rng=rng_bootstrap, n_bootstrap=n_bootstrap)
        by_severity[str(severity)] = {
            "value": sum(values) / len(values) if values else None,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "n": sum(1 for value in values if value),
            "denominator": len(values),
            "status": "ok",
            "evidence_class": "estimation",
        }
    top = data[data["severity"].isin(["serious", "critical"])]["failed"].astype(bool).tolist()
    rest = data[~data["severity"].isin(["serious", "critical"])]["failed"].astype(bool).tolist()
    comparison = {
        **risk_difference_and_odds_ratio(top, rest, rng=rng_bootstrap, n_bootstrap=n_bootstrap),
        "status": "ok",
        "evidence_class": "estimation",
    }
    second = _second_derivation(item_df)
    suppressed = second["null_second_derivation_count"] > 0
    if suppressed:
        for quantity in by_severity.values():
            quantity["status"] = "suppressed_missing_second_derivation"
            quantity["evidence_class"] = "descriptive"
        comparison["status"] = "suppressed_missing_second_derivation"
        comparison["evidence_class"] = "descriptive"
    return {
        "status": "suppressed_missing_second_derivation" if suppressed else "ok",
        "evidence_class": "descriptive" if suppressed else "estimation",
        "by_severity": by_severity,
        "serious_or_critical_vs_rest": comparison,
        "second_derivation": second,
    }


def _second_derivation(df: pd.DataFrame) -> dict[str, Any]:
    scenarios = df[["scenario", "severity", "severity_second_derivation"]].drop_duplicates()
    mismatches = []
    matches = 0
    comparable = 0
    nulls = 0
    for _, row in scenarios.iterrows():
        second = row.get("severity_second_derivation")
        if pd.isna(second):
            nulls += 1
            continue
        comparable += 1
        if row["severity"] == second:
            matches += 1
        else:
            mismatches.append({"scenario": row["scenario"], "severity": row["severity"], "severity_second_derivation": second})
    return {
        "match_rate": matches / comparable if comparable else None,
        "matches": matches,
        "denominator": comparable,
        "mismatches": mismatches,
        "null_second_derivation_count": nulls,
    }
