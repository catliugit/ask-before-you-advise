from __future__ import annotations

from itertools import product
from typing import Any

import pandas as pd

from .aggregate import DEFAULT_EXPECTED_REPEATS, item_outcome, repeat_pass_from_grade
from .constants import CONFIRMATORY_BAR, MIN_DISCORDANT, N_BOOTSTRAP, N_PERMUTATIONS
from .inference import Interval, bootstrap_t_ci, clustered_bootstrap_ci, precision_classification


def compute_ds(
    df: pd.DataFrame,
    *,
    rng_bootstrap,
    rng_permutation,
    n_bootstrap: int = N_BOOTSTRAP,
    n_permutations: int = N_PERMUTATIONS,
    gate_status: str = "confirmatory",
    min_discordant: int = MIN_DISCORDANT,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> dict[str, Any]:
    items, repeats = _ds_items_and_repeats(df, expected_repeats=expected_repeats)
    if repeats.empty:
        return _empty_result(gate_status=gate_status, min_discordant=min_discordant)
    permutation_groups = _ds_panel_permutation_groups(repeats, expected_repeats=expected_repeats)
    collapsed_items = _ds_collapsed_panel_items_from_groups(permutation_groups, expected_repeats=expected_repeats)
    movement_values = collapsed_items["paired_movement"].dropna().astype(float).tolist() if not collapsed_items.empty else []
    bootstrap_interval = bootstrap_t_ci(movement_values, rng=rng_bootstrap, n_bootstrap=n_bootstrap)
    p_value, exact, _, _ = _ds_label_permutation_result(
        repeats,
        rng=rng_permutation,
        n_permutations=n_permutations,
        expected_repeats=expected_repeats,
    )
    interval = bootstrap_interval
    ci_method = "bootstrap_t"
    n_discordant = int(collapsed_items["_discordant"].sum()) if not collapsed_items.empty else 0
    discordant_state_counts = [2] * n_discordant
    realized_attainable_p = _realized_attainable_p(discordant_state_counts)
    status = "demoted_kappa" if gate_status != "confirmatory" else "ok"
    if status == "ok" and (realized_attainable_p is None or realized_attainable_p > CONFIRMATORY_BAR):
        status = "demoted_small_floor"
    evidence = "estimation"
    paired_quantity = _movement_quantity(movement_values, interval=interval, evidence_class=evidence)
    placebo_shift = _rate_quantity(items, "placebo_shift", rng_bootstrap, n_bootstrap, evidence_class="estimation")
    placebo_guard_passed = bool(placebo_shift["denominator"] and placebo_shift["n"] == 0)
    return {
        "absolute_correct": _rate_quantity(items, "absolute_correct", rng_bootstrap, n_bootstrap, evidence_class="estimation"),
        "paired_movement": {
            **paired_quantity,
            "status": status,
            "n_discordant": n_discordant,
            "min_discordant": min_discordant,
            "realized_attainable_p": realized_attainable_p,
            "holm_floor_bar": CONFIRMATORY_BAR,
            "raw_p": p_value,
            "permutation_null": "item_clustered_panel_collapsed_disclosed_control_label_permutation",
            "permutation_exact": exact,
            "placebo_guard_in_null": False,
            "ci_method": ci_method,
            "bootstrap_ci_low": bootstrap_interval.low,
            "bootstrap_ci_high": bootstrap_interval.high,
            "analysis_unit": "item_panel_collapsed",
            "precision_class": precision_classification(
                paired_quantity["value"],
                interval,
                confirmatory_test=False,
            ),
            "confirmed": False,
            "placebo_guard_passed": placebo_guard_passed,
        },
        "placebo_shift": placebo_shift,
        "placebo_guard_passed": placebo_guard_passed,
        "use_confirmed": False,
        "use_item_pass": _rate_quantity(items, "use_pass", rng_bootstrap, n_bootstrap, evidence_class="descriptive"),
        "items": items.to_dict(orient="records"),
        "collapsed_items": collapsed_items.drop(columns=["_discordant"], errors="ignore").to_dict(orient="records"),
        "gate_status": gate_status,
    }


def itemise_ds(df: pd.DataFrame) -> pd.DataFrame:
    items, _ = _ds_items_and_repeats(df)
    return items


def ds_label_permutation_p_value(
    repeats: pd.DataFrame,
    *,
    rng,
    n_permutations: int = N_PERMUTATIONS,
    alternative: str = "greater",
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> tuple[float | None, bool]:
    p_value, exact, _, _ = _ds_label_permutation_result(
        repeats,
        rng=rng,
        n_permutations=n_permutations,
        alternative=alternative,
        expected_repeats=expected_repeats,
    )
    return p_value, exact


def _ds_label_permutation_result(
    repeats: pd.DataFrame,
    *,
    rng,
    n_permutations: int = N_PERMUTATIONS,
    alternative: str = "greater",
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> tuple[float | None, bool, float | None, list[float]]:
    if repeats.empty:
        return None, False, None, []
    groups = _ds_panel_permutation_groups(repeats, expected_repeats=expected_repeats)
    if not groups:
        return None, False, None, []
    group_values = _ds_permutation_group_values(groups, expected_repeats=expected_repeats)
    if not group_values:
        return None, False, None, []
    observed = _movement_statistic_from_group_values(group_values, [False] * len(group_values))
    if observed is None:
        return None, False, None, []
    exact_count = 2 ** len(group_values)
    if exact_count <= n_permutations:
        stats = [
            _movement_statistic_from_group_values(group_values, swaps)
            for swaps in product([False, True], repeat=len(group_values))
        ]
        usable = [stat for stat in stats if stat is not None]
        if not usable:
            return None, True, observed, []
        return _permutation_tail(observed, usable, alternative=alternative), True, observed, usable
    extreme = 0
    usable = 0
    stats = []
    for _ in range(n_permutations):
        swaps = [bool(value) for value in rng.integers(0, 2, size=len(group_values))]
        stat = _movement_statistic_from_group_values(group_values, swaps)
        if stat is None:
            continue
        usable += 1
        stats.append(stat)
        if _is_extreme(stat, observed, alternative=alternative):
            extreme += 1
    if usable == 0:
        return None, False, observed, []
    return float((extreme + 1) / (usable + 1)), False, observed, stats


def use_item_outcomes(df: pd.DataFrame, *, expected_repeats: int = DEFAULT_EXPECTED_REPEATS) -> pd.DataFrame:
    items, _ = _ds_items_and_repeats(df, expected_repeats=expected_repeats)
    if items.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "scenario",
                "item_id",
                "module",
                "variant",
                "severity",
                "severity_second_derivation",
                "passed",
                "failed",
                "status",
                "instability",
                "surviving_repeats",
            ]
        )
    data = {
        "model": items["model"],
        "scenario": items["scenario"],
        "item_id": items["item_id"],
        "module": "C",
        "variant": "C-use-pair",
        "passed": items["use_pass"],
        "failed": items["use_pass"].map(lambda value: None if pd.isna(value) else not bool(value)),
        "status": items["use_pass"].map(lambda value: "indeterminate" if pd.isna(value) else ("pass" if bool(value) else "fail")),
        "instability": None,
        "surviving_repeats": None,
    }
    for metadata in ["severity", "severity_second_derivation"]:
        if metadata in items:
            data[metadata] = items[metadata]
    return pd.DataFrame(data)


def _row_is_void(row: pd.Series) -> bool:
    value = row.get("outcome_void")
    return bool(value) if pd.notna(value) else False


def _ds_items_and_repeats(
    df: pd.DataFrame,
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c_df = df[(df["module"] == "C") & (df["call_status"] != "missing")].copy() if not df.empty else df.copy()
    if c_df.empty:
        return _itemise(pd.DataFrame(), expected_repeats=expected_repeats), pd.DataFrame()
    _raise_missing_refs(c_df)
    _raise_missing_outcome_class(c_df)
    repeat_rows = []
    for key, group in c_df.sort_values(["model", "scenario", "repeat", "variant"]).groupby(["model", "scenario", "repeat"]):
        model, scenario, repeat = key
        disclosed_rows = group[group["variant_kind"] == "disclosed"]
        for _, disclosed in disclosed_rows.iterrows():
            item_id = _c_item_id(disclosed, group)
            item_key = {"model": model, "scenario": scenario, "repeat": repeat, "episode_id": disclosed.get("episode_id")}
            control = _require_matching_row(group, "variant", disclosed.get("control_ref"), item_key=item_key, ref_kind="control_ref")
            placebo = _require_matching_row(group, "placebo_of", disclosed.get("control_ref"), item_key=item_key, ref_kind="placebo_of")
            if _row_is_void(disclosed) or _row_is_void(control) or _row_is_void(placebo):
                # A quote-verification void on any arm of the disclosed/control/placebo triple
                # makes this Use item-repeat ungradeable, so it is excluded from every DS
                # denominator (absolute_correct, paired_movement, use_pass), mirroring modules
                # A/D where repeat_pass_from_grade -> None is dropped by item_outcome. When too
                # few repeats survive, item_outcome marks the item indeterminate. Counting a void
                # would deflate movement on a disclosed/placebo void or manufacture a pass on a
                # control void; the void itself is recorded descriptively in gap_accounting.
                continue
            absolute_correct = repeat_pass_from_grade(disclosed.get("outcome_grade")) is True
            control_correct = repeat_pass_from_grade(control.get("outcome_grade")) is True
            disclosed_class = _outcome_class(disclosed)
            control_class = _outcome_class(control)
            movement = _movement_from_pair(
                disclosed_class=disclosed_class,
                control_class=control_class,
                disclosed_correct=absolute_correct,
                control_correct=control_correct,
            )
            placebo_shift = bool(_placebo_shifted(placebo, control))
            repeat_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "item_id": item_id,
                    "repeat": repeat,
                    "absolute_correct": absolute_correct,
                    "paired_movement": movement,
                    "placebo_shift": placebo_shift,
                    "use_pass": bool(absolute_correct and movement and not placebo_shift),
                    "severity": disclosed.get("severity"),
                    "severity_second_derivation": disclosed.get("severity_second_derivation"),
                    "disclosed_class": disclosed_class,
                    "control_class": control_class,
                    "disclosed_correct": absolute_correct,
                    "control_correct": control_correct,
                }
            )
    repeats = pd.DataFrame(repeat_rows)
    items = _itemise(repeats, expected_repeats=expected_repeats)
    return items, repeats


def _itemise(repeats: pd.DataFrame, *, expected_repeats: int = DEFAULT_EXPECTED_REPEATS) -> pd.DataFrame:
    rows = []
    if repeats.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "scenario",
                "item_id",
                "severity",
                "severity_second_derivation",
                "absolute_correct",
                "paired_movement",
                "placebo_shift",
                "use_pass",
            ]
        )
    for key, group in repeats.groupby(["model", "scenario", "item_id"]):
        row = {"model": key[0], "scenario": key[1], "item_id": key[2]}
        for column in ["absolute_correct", "paired_movement", "placebo_shift", "use_pass"]:
            outcome = item_outcome([bool(value) for value in group[column].tolist()], expected_repeats=expected_repeats)
            row[column] = outcome.passed
        for metadata in ["severity", "severity_second_derivation"]:
            if metadata in group:
                row[metadata] = _first_non_missing(group[metadata])
        rows.append(row)
    return pd.DataFrame(rows)


def _movement_statistic(
    repeats: pd.DataFrame,
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
    complete_case: bool = True,
) -> float | None:
    """complete_case=False is for permutation draws whose usable model panels were filtered before swapping."""
    if complete_case:
        groups = _ds_panel_permutation_groups(repeats, expected_repeats=expected_repeats)
    else:
        groups = _ds_current_panel_groups(repeats)
    return _movement_statistic_from_groups(groups, expected_repeats=expected_repeats)


def _movement_statistic_from_groups(
    groups: list[tuple[object, pd.DataFrame]],
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> float | None:
    values = []
    for _, item_group in groups:
        model_items = _itemise(item_group, expected_repeats=expected_repeats)
        numeric = model_items["paired_movement"].dropna().astype(float).tolist() if not model_items.empty else []
        if numeric:
            values.append(float(sum(numeric) / len(numeric)))
    if not values:
        return None
    return float(sum(values) / len(values))


def _ds_permutation_group_values(
    groups: list[tuple[object, pd.DataFrame]],
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> list[tuple[float, float]]:
    values = []
    for group in groups:
        unswapped = _movement_statistic_from_groups([group], expected_repeats=expected_repeats)
        swapped = _movement_statistic_from_groups(
            [(group[0], _apply_item_swaps([group], [True]))],
            expected_repeats=expected_repeats,
        )
        if unswapped is None or swapped is None:
            continue
        values.append((unswapped, swapped))
    return values


def _movement_statistic_from_group_values(
    group_values: list[tuple[float, float]],
    swaps: list[bool] | tuple[bool, ...],
) -> float | None:
    values = [pair[int(swap)] for pair, swap in zip(group_values, swaps, strict=True)]
    if not values:
        return None
    return float(sum(values) / len(values))


def _ds_collapsed_panel_items(repeats: pd.DataFrame, *, expected_repeats: int = DEFAULT_EXPECTED_REPEATS) -> pd.DataFrame:
    return _ds_collapsed_panel_items_from_groups(
        _ds_panel_permutation_groups(repeats, expected_repeats=expected_repeats),
        expected_repeats=expected_repeats,
    )


def _ds_collapsed_panel_items_from_groups(
    groups: list[tuple[object, pd.DataFrame]],
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> pd.DataFrame:
    rows = []
    for key, item_group in groups:
        scenario, item_id = key
        model_items = _itemise(item_group, expected_repeats=expected_repeats)
        numeric = model_items["paired_movement"].dropna().astype(float).tolist() if not model_items.empty else []
        if not numeric:
            continue
        rows.append(
            {
                "scenario": scenario,
                "item_id": item_id,
                "paired_movement": float(sum(numeric) / len(numeric)),
                "panel_n": len(numeric),
                "_discordant": _ds_permutation_item_is_discordant(item_group, expected_repeats=expected_repeats),
            }
        )
    return pd.DataFrame(rows)


def _ds_current_panel_groups(repeats: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
    if repeats.empty:
        return []
    return [
        ((str(scenario), str(item_id)), group.copy())
        for (scenario, item_id), group in repeats.sort_values(["scenario", "item_id", "model", "repeat"]).groupby(["scenario", "item_id"])
    ]


def _ds_panel_permutation_groups(
    repeats: pd.DataFrame,
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> list[tuple[object, pd.DataFrame]]:
    grouped = _ds_usable_model_groups(repeats, expected_repeats=expected_repeats)
    return [
        ((scenario, item_id), pd.concat(groups, ignore_index=True))
        for (scenario, item_id), groups in sorted(grouped.items())
        if groups
    ]


def _ds_usable_model_groups(
    repeats: pd.DataFrame,
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> dict[tuple[str, str], list[pd.DataFrame]]:
    usable: dict[tuple[str, str], list[pd.DataFrame]] = {}
    if repeats.empty:
        return usable
    for (model, scenario, item_id), group in repeats.sort_values(["model", "scenario", "item_id", "repeat"]).groupby(["model", "scenario", "item_id"]):
        model_group = group.copy()
        values = [
            _ds_model_group_movement(model_group, swap, expected_repeats=expected_repeats)
            for swap in [False, True]
        ]
        if any(value is None for value in values):
            continue
        usable.setdefault((str(scenario), str(item_id)), []).append(model_group)
    return usable


def _ds_model_group_movement(
    group: pd.DataFrame,
    swap: bool,
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> float | None:
    swapped = _apply_item_swaps([(None, group.copy())], [swap])
    item = _itemise(swapped, expected_repeats=expected_repeats)
    if item.empty:
        return None
    value = item.iloc[0]["paired_movement"]
    if pd.isna(value):
        return None
    return float(bool(value))


def _apply_item_swaps(groups: list[tuple[object, pd.DataFrame]], swaps: list[bool] | tuple[bool, ...]) -> pd.DataFrame:
    rows = []
    for (_, group), swap in zip(groups, swaps, strict=True):
        for _, row in group.iterrows():
            item = row.to_dict()
            if swap:
                item["paired_movement"] = _movement_from_pair(
                    disclosed_class=item.get("control_class"),
                    control_class=item.get("disclosed_class"),
                    disclosed_correct=bool(item.get("control_correct")),
                    control_correct=bool(item.get("disclosed_correct")),
                )
                item["absolute_correct"] = bool(item.get("control_correct"))
            else:
                item["paired_movement"] = _movement_from_pair(
                    disclosed_class=item.get("disclosed_class"),
                    control_class=item.get("control_class"),
                    disclosed_correct=bool(item.get("disclosed_correct")),
                    control_correct=bool(item.get("control_correct")),
                )
                item["absolute_correct"] = bool(item.get("disclosed_correct"))
            rows.append(item)
    return pd.DataFrame(rows)


def _ds_discordant_count(repeats: pd.DataFrame, *, expected_repeats: int = DEFAULT_EXPECTED_REPEATS) -> int:
    if repeats.empty:
        return 0
    groups = [group for _, group in _ds_panel_permutation_groups(repeats, expected_repeats=expected_repeats)]
    return sum(1 for group in groups if _ds_permutation_item_is_discordant(group, expected_repeats=expected_repeats))


def _ds_permutation_item_is_discordant(
    group: pd.DataFrame,
    *,
    expected_repeats: int = DEFAULT_EXPECTED_REPEATS,
) -> bool:
    values = []
    for swap in [False, True]:
        swapped = _apply_item_swaps([(None, group.copy())], [swap])
        item = _itemise(swapped, expected_repeats=expected_repeats)
        item_values = item["paired_movement"].dropna().astype(float).tolist() if not item.empty else []
        if not item_values:
            continue
        values.append(round(float(sum(item_values) / len(item_values)), 12))
    return len(set(values)) > 1


def _realized_attainable_p(discordant_state_counts: list[int]) -> float | None:
    if not discordant_state_counts:
        return None
    product_value = 1
    for count in discordant_state_counts:
        product_value *= int(count)
    return float(1 / product_value)


def _permutation_tail(observed: float, stats: list[float], *, alternative: str) -> float:
    return float(sum(1 for stat in stats if _is_extreme(stat, observed, alternative=alternative)) / len(stats))


def _is_extreme(stat: float, observed: float, *, alternative: str) -> bool:
    if alternative == "greater":
        return stat >= observed
    if alternative == "less":
        return stat <= observed
    if alternative == "two-sided":
        return abs(stat) >= abs(observed)
    raise ValueError(f"unknown alternative {alternative!r}")


def _rate_quantity(
    items: pd.DataFrame,
    column: str,
    rng,
    n_bootstrap: int,
    *,
    evidence_class: str,
    interval: Interval | None = None,
) -> dict[str, Any]:
    values = [bool(value) for value in items[column].dropna().tolist()] if not items.empty and column in items else []
    if interval is None:
        interval = clustered_bootstrap_ci(values, rng=rng, n_bootstrap=n_bootstrap)
    numerator = sum(1 for value in values if value)
    denominator = len(values)
    return {
        "value": numerator / denominator if denominator else None,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "n": numerator,
        "denominator": denominator,
        "status": "ok",
        "evidence_class": evidence_class,
    }


def _movement_quantity(values: list[float], *, interval: Interval, evidence_class: str) -> dict[str, Any]:
    denominator = len(values)
    total = float(sum(values)) if values else 0.0
    return {
        "value": total / denominator if denominator else None,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "n": total if denominator else 0,
        "denominator": denominator,
        "status": "ok",
        "evidence_class": evidence_class,
    }


def _movement_from_pair(
    *,
    disclosed_class: object,
    control_class: object,
    disclosed_correct: bool,
    control_correct: bool,
) -> bool:
    if disclosed_class is not None and control_class is not None:
        moved = str(disclosed_class) != str(control_class)
    else:
        moved = control_correct is not True
    return bool(disclosed_correct and moved)


def _placebo_shifted(placebo: pd.Series, control: pd.Series) -> bool:
    placebo_class = _outcome_class(placebo)
    control_class = _outcome_class(control)
    if placebo_class is not None and control_class is not None:
        return placebo_class != control_class
    if placebo.get("equivalence_class") == "matches_control":
        return repeat_pass_from_grade(placebo.get("outcome_grade")) is not True
    return repeat_pass_from_grade(placebo.get("outcome_grade")) != repeat_pass_from_grade(control.get("outcome_grade"))


def _outcome_class(row: pd.Series) -> str | None:
    for column in ["outcome_class", "recommendation_class"]:
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return str(value)
    return None


def _c_item_id(disclosed: pd.Series, group: pd.DataFrame) -> str:
    existing = disclosed.get("item_id")
    if existing is not None and not pd.isna(existing) and str(existing):
        return str(existing)
    control_ref = str(disclosed.get("control_ref"))
    placebo = _matching_row(group, "placebo_of", disclosed.get("control_ref"))
    placebo_id = str(placebo.get("variant")) if placebo is not None else "missing_placebo"
    return f"C:{control_ref}:{disclosed.get('variant')}:{placebo_id}"


def _matching_row(group: pd.DataFrame, column: str, value: object) -> pd.Series | None:
    if value is None or pd.isna(value):
        return None
    match = group[group[column] == value] if column in group else group.iloc[0:0]
    if match.empty:
        return None
    return match.iloc[0]


def _first_non_missing(series: pd.Series) -> object | None:
    for value in series.tolist():
        if value is not None and not pd.isna(value):
            return value
    return None


def _require_matching_row(group: pd.DataFrame, column: str, value: object, *, item_key: dict[str, object], ref_kind: str) -> pd.Series:
    match = _matching_row(group, column, value)
    if match is None:
        key = ", ".join(f"{name}={item_key.get(name)!r}" for name in ["episode_id", "model", "scenario", "repeat"])
        raise ValueError(f"C item pair integrity failure: missing {ref_kind} target {value!r} for {key}")
    return match


def _raise_missing_refs(c_df: pd.DataFrame) -> None:
    missing_disclosed = c_df[(c_df["variant_kind"] == "disclosed") & (c_df["control_ref"].isna() | (c_df["control_ref"] == ""))]
    missing_placebo = c_df[(c_df["variant_kind"] == "placebo") & (c_df["placebo_of"].isna() | (c_df["placebo_of"] == ""))]
    missing = pd.concat([missing_disclosed, missing_placebo])
    if not missing.empty:
        ids = sorted(str(value) for value in missing["episode_id"].tolist())
        raise ValueError(f"C disclosed/placebo variant missing control_ref/placebo_of: {', '.join(ids)}")


def _raise_missing_outcome_class(c_df: pd.DataFrame) -> None:
    relevant = c_df[c_df["variant_kind"].isin(["control", "disclosed", "placebo"])]
    if "outcome_void" in relevant:
        # Quote-verification voids are class-null by design and are excluded from the DS
        # denominators in _ds_items_and_repeats (the void item-repeat is skipped); they are
        # legitimate voids, not a missing-class defect, so they must not trip this requirement.
        relevant = relevant[~relevant["outcome_void"].fillna(False).astype(bool)]
    if "final_grade_human_handoff" in relevant:
        # A council split on the free-text class label (three synonymous labels for the same
        # substance, e.g. keep_car / continue_agreement / keep_paying) resolves to a null class
        # WITH a human handoff under the pre-registered split rule: the machine abstains and a
        # human decides. Those rows are a machine-abstained state, not a missing-class defect;
        # _movement_from_pair carries the written correctness fallback for them and Use is
        # pre-specified estimation. Only a null class with NO handoff (a genuine omission)
        # still trips the requirement below.
        relevant = relevant[~relevant["final_grade_human_handoff"].fillna(False).astype(bool)]
    if relevant.empty:
        return
    if "outcome_class" not in relevant:
        ids = sorted(str(value) for value in relevant["episode_id"].tolist())
        raise ValueError(f"C outcome_class is required for DS; no outcome_grade fallback is allowed: {', '.join(ids)}")
    missing = relevant[relevant["outcome_class"].isna() | (relevant["outcome_class"] == "")]
    if not missing.empty:
        ids = sorted(str(value) for value in missing["episode_id"].tolist())
        raise ValueError(f"C outcome_class is required for DS; no outcome_grade fallback is allowed: {', '.join(ids)}")


def _empty_result(*, gate_status: str, min_discordant: int) -> dict[str, Any]:
    status = "demoted_kappa" if gate_status != "confirmatory" else "demoted_small_floor"
    evidence = "estimation"
    empty_quantity = {"value": None, "ci_low": None, "ci_high": None, "n": 0, "denominator": 0, "status": status, "evidence_class": evidence}
    return {
        "absolute_correct": {**empty_quantity, "evidence_class": "estimation"},
        "paired_movement": {
            **empty_quantity,
            "n_discordant": 0,
            "min_discordant": min_discordant,
            "realized_attainable_p": None,
            "holm_floor_bar": CONFIRMATORY_BAR,
            "raw_p": None,
            "permutation_null": "item_clustered_panel_collapsed_disclosed_control_label_permutation",
            "permutation_exact": False,
            "placebo_guard_in_null": False,
            "ci_method": "bootstrap_t",
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "analysis_unit": "item_panel_collapsed",
            "precision_class": "estimation",
            "confirmed": False,
            "placebo_guard_passed": False,
        },
        "placebo_shift": {**empty_quantity, "evidence_class": "estimation"},
        "placebo_guard_passed": False,
        "use_confirmed": False,
        "use_item_pass": {**empty_quantity, "evidence_class": "descriptive"},
        "items": [],
        "collapsed_items": [],
        "gate_status": gate_status,
    }
