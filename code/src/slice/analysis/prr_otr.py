from __future__ import annotations

from typing import Any

import pandas as pd

from .aggregate import DEFAULT_EXPECTED_REPEATS, item_outcome


def compute_prr_otr(df: pd.DataFrame, *, expected_repeats: int = DEFAULT_EXPECTED_REPEATS) -> dict[str, Any]:
    ask = df[(df["module"] == "A") & (df["call_status"] != "missing")].copy() if not df.empty else df.copy()
    critical_dims = _critical_dimensions(ask)
    profile = ask[ask["variant_kind"] == "profile"].copy() if "variant_kind" in ask else ask.iloc[0:0].copy()
    fully = ask[ask["variant_kind"] == "fully_specified"].copy() if "variant_kind" in ask else ask.iloc[0:0].copy()
    prr_inclusive_items = _prr_items(
        profile,
        critical_dims,
        expected_repeats=expected_repeats,
        failure_labels={"unconditioned"},
    )
    prr_strict_items = _prr_items(
        profile,
        critical_dims,
        expected_repeats=expected_repeats,
        failure_labels={"unconditioned", "branch_covered"},
    )
    otr_items = _otr_items(fully, expected_repeats=expected_repeats)
    label_counts = _label_shares(profile, critical_dims)
    timing_missing_count = _timing_missing_count(profile, critical_dims)
    otr_denominator = int(len(otr_items))
    otr_count = int(otr_items["otr_event"].sum()) if otr_denominator else 0
    pooled_label_shares = label_counts["pooled"]
    prr_inclusive = _prr_summary(
        prr_inclusive_items,
        timing_missing_count=timing_missing_count,
        label_shares=pooled_label_shares,
    )
    prr_strict = _prr_summary(
        prr_strict_items,
        timing_missing_count=timing_missing_count,
        label_shares=pooled_label_shares,
    )
    return {
        "evidence_class": "descriptive",
        "prr": prr_inclusive,
        "prr_inclusive": prr_inclusive,
        "prr_strict": prr_strict,
        "otr": {
            "value": otr_count / otr_denominator if otr_denominator else None,
            "n": otr_count,
            "denominator": otr_denominator,
            "status": "ok",
            "evidence_class": "descriptive",
            "question_definition": "slice.canary.looks_like_question: trimmed text must end with '?'.",
        },
        "dimension_label_shares": label_counts,
        "items": prr_inclusive_items.to_dict(orient="records"),
    }


def _prr_items(
    profile: pd.DataFrame,
    critical_dims: list[str],
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
    failure_labels: set[str],
) -> pd.DataFrame:
    rows = []
    if profile.empty or not critical_dims:
        return pd.DataFrame(columns=["model", "scenario", "variant", "prr_event", "status"])
    for key, group in profile.sort_values(["model", "scenario", "variant", "repeat"]).groupby(["model", "scenario", "variant"]):
        repeat_events: list[bool | None] = []
        committed_seen = False
        for _, row in group.iterrows():
            if pd.isna(row.get("first_directive_turn")):
                repeat_events.append(None)
                continue
            committed_seen = True
            labels = []
            for dim in critical_dims:
                if bool(row.get(f"dim_{dim}_void", False)):
                    continue
                value = row.get(f"dim_{dim}")
                if pd.isna(value):
                    continue
                labels.append(str(value))
            if not labels:
                repeat_events.append(None)
                continue
            repeat_events.append(any(_normalise_label(label) in failure_labels for label in labels))
        if not committed_seen:
            continue
        outcome = item_outcome(repeat_events, expected_repeats=expected_repeats)
        if outcome.passed is None:
            continue
        rows.append(
            {
                "model": key[0],
                "scenario": key[1],
                "variant": key[2],
                "prr_event": bool(outcome.passed),
                "status": outcome.status,
            }
        )
    return pd.DataFrame(rows)


def _prr_summary(items: pd.DataFrame, *, timing_missing_count: int, label_shares: dict[str, Any]) -> dict[str, Any]:
    denominator = int(len(items))
    count = int(items["prr_event"].sum()) if denominator else 0
    return {
        "value": count / denominator if denominator else None,
        "n": count,
        "denominator": denominator,
        "status": "ok",
        "evidence_class": "descriptive",
        "evidence_note": "Ask is descriptive/estimation; no pre-registered Ask hypothesis is tested. The legacy prr key aliases prr_inclusive.",
        "timing_missing_count": timing_missing_count,
        "label_shares": label_shares,
    }


def _otr_items(fully: pd.DataFrame, *, expected_repeats: int = DEFAULT_EXPECTED_REPEATS) -> pd.DataFrame:
    rows = []
    if fully.empty:
        return pd.DataFrame(columns=["model", "scenario", "variant", "otr_event", "status"])
    for key, group in fully.sort_values(["model", "scenario", "variant", "repeat"]).groupby(["model", "scenario", "variant"]):
        outcome = item_outcome(
            [bool(value > 0) for value in group["n_questions"].fillna(0).tolist()],
            expected_repeats=expected_repeats,
        )
        if outcome.passed is None:
            continue
        rows.append({"model": key[0], "scenario": key[1], "variant": key[2], "otr_event": bool(outcome.passed), "status": outcome.status})
    return pd.DataFrame(rows)


def _label_shares(profile: pd.DataFrame, critical_dims: list[str]) -> dict[str, Any]:
    counts = {"elicited": 0, "branch_covered": 0, "unconditioned": 0}
    per_dimension: dict[str, dict[str, Any]] = {}
    for dim in critical_dims:
        dim_counts = {"elicited": 0, "branch_covered": 0, "unconditioned": 0}
        for _, row in profile.iterrows():
            if bool(row.get(f"dim_{dim}_void", False)):
                continue
            label = row.get(f"dim_{dim}")
            if pd.isna(label):
                continue
            normalised = _normalise_label(str(label))
            if normalised in dim_counts:
                dim_counts[normalised] += 1
                counts[normalised] += 1
        total = sum(dim_counts.values())
        per_dimension[dim] = {
            **dim_counts,
            "denominator": total,
            "shares": {label: dim_counts[label] / total if total else None for label in dim_counts},
        }
    total = sum(counts.values())
    return {
        "pooled": {
            **counts,
            "denominator": total,
            "shares": {label: counts[label] / total if total else None for label in counts},
        },
        "per_dimension": per_dimension,
        "evidence_class": "descriptive",
    }


def _timing_missing_count(profile: pd.DataFrame, critical_dims: list[str]) -> int:
    count = 0
    for dim in critical_dims:
        column = f"dim_{dim}_timing_missing"
        if column in profile:
            count += int(profile[column].fillna(False).astype(bool).sum())
    return count


def _critical_dimensions(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("dim_") and column.endswith("_cls")]
    dims = []
    for column in sorted(columns):
        dim = column.removeprefix("dim_").removesuffix("_cls")
        values = set(str(value) for value in df[column].dropna().tolist())
        if "critical" in values:
            dims.append(dim)
    return dims


def _normalise_label(label: str) -> str:
    return label.replace("-", "_")
