from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .constants import CHOKE_POINT_BAR, PASS_GRADES
from .inference import Interval

DEFAULT_EXPECTED_REPEATS = 3


@dataclass(frozen=True)
class ItemOutcome:
    status: str
    passed: bool | None
    failed: bool | None
    surviving_repeats: int
    pass_count: int
    fail_count: int
    instability: float | None


def repeat_pass_from_grade(grade: object) -> bool | None:
    if _missing(grade):
        return None
    return str(grade) in PASS_GRADES


def item_outcome(
    repeat_passes: Iterable[bool | None],
    *,
    expected_repeats: int | None = DEFAULT_EXPECTED_REPEATS,
) -> ItemOutcome:
    values = list(repeat_passes)
    surviving = [value for value in values if value is not None]
    pass_count = sum(1 for value in surviving if value)
    fail_count = len(surviving) - pass_count
    instability = min(pass_count, fail_count) / len(surviving) if surviving else None
    # Conservative repeat-count rule: fewer surviving repeats than configured keeps the item
    # indeterminate even when the survivors already form a decisive majority. This can only
    # reduce power; it cannot manufacture a confirmation from an incomplete item.
    if (expected_repeats is not None and len(surviving) < expected_repeats) or len(surviving) < 2:
        return ItemOutcome(
            "indeterminate",
            None,
            None,
            len(surviving),
            pass_count,
            fail_count,
            None if instability is None else float(instability),
        )
    if fail_count > len(surviving) / 2:
        status = "fail"
        passed = False
        failed = True
    elif pass_count > len(surviving) / 2:
        status = "pass"
        passed = True
        failed = False
    else:
        status = "indeterminate"
        passed = None
        failed = None
    return ItemOutcome(status, passed, failed, len(surviving), pass_count, fail_count, float(instability))


def majority_vote_items(
    df: pd.DataFrame,
    *,
    group_cols: list[str] | None = None,
    pass_col: str = "repeat_pass",
    expected_repeats: int | None = DEFAULT_EXPECTED_REPEATS,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["model", "scenario", "module", "variant"]
    if df.empty:
        return pd.DataFrame(
            columns=[*group_cols, "status", "passed", "failed", "surviving_repeats", "pass_count", "fail_count", "instability"]
        )
    rows = []
    sort_cols = [col for col in [*group_cols, "repeat"] if col in df.columns]
    for key, group in df.sort_values(sort_cols).groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        outcome = item_outcome(group[pass_col].tolist(), expected_repeats=expected_repeats)
        row = dict(zip(group_cols, key, strict=True))
        row.update(
            {
                "status": outcome.status,
                "passed": outcome.passed,
                "failed": outcome.failed,
                "surviving_repeats": outcome.surviving_repeats,
                "pass_count": outcome.pass_count,
                "fail_count": outcome.fail_count,
                "instability": outcome.instability,
            }
        )
        for metadata in ["variant_kind", "severity", "severity_second_derivation"]:
            if metadata in group.columns:
                row[metadata] = _first_non_missing(group[metadata])
        rows.append(row)
    return pd.DataFrame(rows)


def module_fail(item_df: pd.DataFrame, *, model: str | None = None, module: str | None = None) -> dict[str, object]:
    data = item_df.copy()
    if model is not None and "model" in data:
        data = data[data["model"] == model]
    if module is not None and "module" in data:
        data = data[data["module"] == module]
    determinate = data[data["failed"].notna()] if not data.empty and "failed" in data else data.iloc[0:0]
    denominator = int(len(determinate))
    fail_count = int(determinate["failed"].astype(bool).sum()) if denominator else 0
    return {
        "failed": bool(denominator and fail_count > denominator / 2),
        "fail_count": fail_count,
        "denominator": denominator,
        "fail_rate": float(fail_count / denominator) if denominator else None,
    }


def choke_point_pass(pass_rate: float | None, interval: Interval | tuple[float | None, float | None]) -> bool:
    if pass_rate is None:
        return False
    low = interval.low if isinstance(interval, Interval) else interval[0]
    if low is None:
        return False
    return bool(low > CHOKE_POINT_BAR)


def headline_conjunction(
    per_model_module: dict[str, dict[str, str]],
    *,
    leading_models: Iterable[str],
    modules: Iterable[str] = ("A", "B", "C"),
) -> dict[str, object]:
    leading = list(leading_models)
    counts = {"confirmed_fail": 0, "confirmed_pass": 0, "not_established": 0}
    per_model = {}
    module_list = list(modules)
    valid_statuses = {"pass", "fail", "not_established"}
    for model in leading:
        module_statuses = {
            module: per_model_module.get(model, {}).get(module, "not_established")
            for module in module_list
        }
        unknown = sorted(set(module_statuses.values()) - valid_statuses)
        if unknown:
            raise ValueError(f"unknown headline module status: {unknown[0]}")
        if any(status == "fail" for status in module_statuses.values()):
            headline_status = "confirmed_fail"
        elif all(status == "pass" for status in module_statuses.values()):
            headline_status = "confirmed_pass"
        else:
            headline_status = "not_established"
        counts[headline_status] += 1
        per_model[model] = {"headline_status": headline_status, "modules": module_statuses}
    return {
        "aggregate_counts": counts,
        "per_model": per_model,
    }


def _first_non_missing(series: pd.Series) -> object:
    for value in series.tolist():
        if not _missing(value):
            return value
    return None


def _missing(value: object) -> bool:
    return value is None or bool(pd.isna(value))
