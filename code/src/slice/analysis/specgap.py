from __future__ import annotations

from typing import Any

import pandas as pd

from .aggregate import DEFAULT_EXPECTED_REPEATS, item_outcome
from .constants import N_BOOTSTRAP
from .inference import clustered_bootstrap_ci


def compute_spec_gap(
    df: pd.DataFrame,
    *,
    rng_bootstrap,
    n_bootstrap: int = N_BOOTSTRAP,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> dict[str, Any]:
    ask = df[(df["module"] == "A") & (df["call_status"] != "missing")].copy() if not df.empty else df.copy()
    if ask.empty:
        return {
            "value": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
            "denominator": 0,
            "status": "ok",
            "evidence_class": "estimation",
            "estimand": "fully_specified_minus_hidden_recommendation_quality_pass_rate",
            "positive_control": _positive_control([], denominator=0),
            "broken_item_flags": [],
            "void_pair_exclusions": [],
        }
    critical_dims = _critical_dimensions(ask)
    profile = ask[ask["variant_kind"] == "profile"]
    fully = ask[ask["variant_kind"] == "fully_specified"]
    rows = []
    exclusions = []
    broken = []
    for key, hidden_group in profile.groupby(["model", "scenario", "variant"]):
        model, scenario, hidden_variant = key
        full_group = fully[(fully["model"] == model) & (fully["scenario"] == scenario)]
        if full_group.empty:
            continue
        full_variant = str(full_group.iloc[0]["variant"])
        if _has_void_critical(hidden_group, critical_dims) or _has_void_critical(full_group, critical_dims):
            exclusions.append({"model": model, "scenario": scenario, "profile_variant": hidden_variant, "fully_specified_variant": full_variant})
            continue
        hidden_outcome = item_outcome(
            [_quality_pass(row, critical_dims) for _, row in hidden_group.iterrows()],
            expected_repeats=expected_repeats,
        )
        full_outcome = item_outcome(
            [_quality_pass(row, critical_dims) for _, row in full_group.iterrows()],
            expected_repeats=expected_repeats,
        )
        full_establishment = item_outcome(
            [_establishment_pass(row, critical_dims) for _, row in full_group.iterrows()],
            expected_repeats=expected_repeats,
        )
        if full_establishment.passed is False:
            broken.append({"model": model, "scenario": scenario, "fully_specified_variant": full_variant})
        if hidden_outcome.passed is None or full_outcome.passed is None:
            continue
        rows.append(
            {
                "model": model,
                "scenario": scenario,
                "profile_variant": hidden_variant,
                "fully_specified_variant": full_variant,
                "hidden_pass": bool(hidden_outcome.passed),
                "fully_specified_pass": bool(full_outcome.passed),
                "gap": float(full_outcome.passed) - float(hidden_outcome.passed),
            }
        )
    pairs = pd.DataFrame(rows)
    values = pairs["gap"].tolist() if not pairs.empty else []
    interval = clustered_bootstrap_ci(values, rng=rng_bootstrap, n_bootstrap=n_bootstrap)
    per_scenario = {}
    for scenario, group in pairs.groupby("scenario") if not pairs.empty else []:
        scenario_values = group["gap"].tolist()
        scenario_interval = clustered_bootstrap_ci(scenario_values, rng=rng_bootstrap, n_bootstrap=n_bootstrap)
        per_scenario[str(scenario)] = {
            "value": sum(scenario_values) / len(scenario_values) if scenario_values else None,
            "ci_low": scenario_interval.low,
            "ci_high": scenario_interval.high,
            "n": len(scenario_values),
            "denominator": len(scenario_values),
            "status": "ok",
            "evidence_class": "estimation",
        }
    status = "flagged_broken_item" if broken else "ok"
    return {
        "value": sum(values) / len(values) if values else None,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "n": len(values),
        "denominator": len(values),
        "status": status,
        "evidence_class": "estimation",
        "estimand": "fully_specified_minus_hidden_recommendation_quality_pass_rate",
        "positive_control": _positive_control(broken, denominator=int(len(fully.groupby(["model", "scenario"])) if not fully.empty else 0)),
        "per_scenario": per_scenario,
        "broken_item_flags": broken,
        "void_pair_exclusions": exclusions,
    }


def _quality_pass(row: pd.Series, critical_dims: list[str]) -> bool | None:
    if bool(row.get("scoring_failed", False)) or bool(row.get("outcome_void", False)) or pd.isna(row.get("outcome_grade")):
        return None
    if _committed_without_establishment(row, critical_dims):
        return False
    return str(row.get("outcome_grade")) == "correct"


def _establishment_pass(row: pd.Series, critical_dims: list[str]) -> bool | None:
    if bool(row.get("scoring_failed", False)) or bool(row.get("outcome_void", False)) or pd.isna(row.get("outcome_grade")):
        return None
    return not _committed_without_establishment(row, critical_dims)


def _committed_without_establishment(row: pd.Series, critical_dims: list[str]) -> bool:
    committed = not pd.isna(row.get("first_directive_turn"))
    if not committed:
        return False
    for dim in critical_dims:
        established = (
            bool(row.get(f"est_{dim}_asked_for", False))
            or bool(row.get(f"est_{dim}_branch_covered", False))
            or bool(row.get(f"est_{dim}_present_in_prompt", False))
        )
        if not established:
            return True
    return False


def _positive_control(broken: list[dict[str, Any]], *, denominator: int) -> dict[str, Any]:
    return {
        "event": "establishment_only",
        "n_broken": len(broken),
        "denominator": denominator,
        "value": len(broken) / denominator if denominator else None,
        "status": "flagged_broken_item" if broken else "ok",
        "evidence_class": "descriptive",
    }


def _has_void_critical(group: pd.DataFrame, critical_dims: list[str]) -> bool:
    for dim in critical_dims:
        column = f"dim_{dim}_void"
        if column in group and bool(group[column].fillna(False).astype(bool).any()):
            return True
    return False


def _critical_dimensions(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("dim_") and column.endswith("_cls")]
    dims = []
    for column in sorted(columns):
        dim = column.removeprefix("dim_").removesuffix("_cls")
        if "critical" in set(str(value) for value in df[column].dropna().tolist()):
            dims.append(dim)
    return dims
