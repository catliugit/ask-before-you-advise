from __future__ import annotations

import numpy as np
import pandas as pd

from slice.analysis.ds import compute_ds
from slice.analysis.fcr import compute_fcr


def test_panel_collapse_null_calibration_not_model_item_inflated():
    rates = _correlated_ds_null_rejection_rates()

    assert 0.02 <= rates["ds_production"] <= 0.10
    assert 0.03 <= rates["fcr_production"] <= 0.08
    assert 0.16 <= rates["old_model_item"] <= 0.26
    assert rates["fcr_calls"] == 40
    assert rates["fcr_item0_panel_n"] == 6


def test_fcr_null_fixture_exercises_missing_cells_and_heterogeneous_state_counts():
    disclosed = np.array(
        [
            [True, False, True, False, True, False, True, False],
            [False, True, False, True, False, True, False, True],
        ],
        dtype=bool,
    )
    rows = _b_correlated_null_rows(disclosed)

    result = compute_fcr(
        pd.DataFrame(rows),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=20,
        n_permutations=64,
        min_discordant=1,
        expected_repeats=2,
    )

    h0_item = next(item for item in result["collapsed_items"] if item["item_id"] == "B:C0")
    m0_item = next(item for item in result["collapsed_items"] if item["item_id"] == "B:C1")
    assert h0_item["panel_n"] == 6
    assert m0_item["panel_n"] == 8
    assert result["value"] == 0.0
    assert result["raw_p"] == 1.0


def _correlated_ds_null_rejection_rates() -> dict[str, float | int]:
    rng = np.random.default_rng(20260703)
    n_sims = 40
    n_items = 12
    n_models = 8
    old_permutations = 399
    ds_rejections = 0
    fcr_item0_panel_n = None
    old_rejections = 0
    for sim_index in range(n_sims):
        item_base = rng.normal(0, 0.5, size=(n_items, 1, 1))
        wording_shock = rng.normal(0, 1.0, size=(n_items, 1, 2))
        model_noise = rng.normal(0, 0.5, size=(n_items, n_models, 2))
        scores = item_base + wording_shock + model_noise
        disclosed = scores[:, :, 0] > 0
        control = scores[:, :, 1] > 0

        ds_result = compute_ds(
            pd.DataFrame(_c_correlated_null_rows(disclosed, control)),
            rng_bootstrap=np.random.default_rng(1000 + sim_index),
            rng_permutation=np.random.default_rng(2000 + sim_index),
            n_bootstrap=1,
            n_permutations=2**n_items,
            min_discordant=1,
            expected_repeats=2,
        )
        ds_rejections += ds_result["paired_movement"]["raw_p"] <= 0.05

        if fcr_item0_panel_n is None:
            fcr_result = compute_fcr(
                pd.DataFrame(_b_correlated_null_rows(disclosed)),
                rng_bootstrap=np.random.default_rng(3000 + sim_index),
                rng_permutation=np.random.default_rng(4000 + sim_index),
                n_bootstrap=1,
                n_permutations=64,
                min_discordant=1,
                expected_repeats=2,
            )
            fcr_item0_panel_n = next(item["panel_n"] for item in fcr_result["collapsed_items"] if item["item_id"] == "B:C0")

        observed = float(np.mean(disclosed))
        old_extreme = 0
        for _ in range(old_permutations):
            swap = rng.integers(0, 2, size=(n_items, n_models)).astype(bool)
            old_stat = float(np.mean(np.where(swap, control, disclosed)))
            old_extreme += old_stat >= observed
        old_rejections += ((old_extreme + 1) / (old_permutations + 1)) <= 0.05

    fcr_rng = np.random.default_rng(6)
    fcr_rejections = 0
    for sim_index in range(n_sims):
        leading_is_wobbly = fcr_rng.integers(0, 2, size=n_items).astype(bool)
        fcr_result = compute_fcr(
            pd.DataFrame(_b_exchangeable_fcr_null_rows(leading_is_wobbly)),
            rng_bootstrap=np.random.default_rng(5000 + sim_index),
            rng_permutation=np.random.default_rng(6000 + sim_index),
            n_bootstrap=1,
            n_permutations=2**n_items,
            min_discordant=1,
            expected_repeats=3,
        )
        fcr_rejections += fcr_result["raw_p"] <= 0.05
    return {
        "ds_production": ds_rejections / n_sims,
        "fcr_production": fcr_rejections / n_sims,
        "old_model_item": old_rejections / n_sims,
        "fcr_calls": n_sims,
        "fcr_item0_panel_n": int(fcr_item0_panel_n or 0),
    }


def _b_exchangeable_fcr_null_rows(leading_is_wobbly: np.ndarray) -> list[dict[str, object]]:
    rows = []
    clean = ["correct", "correct", "correct"]
    wobbly = ["correct", "correct", "incorrect"]
    for item_index, is_wobbly in enumerate(leading_is_wobbly):
        scenario = f"F{item_index}"
        item_id = f"B:{scenario}"
        rows.extend(_b_repeats(scenario, "B-neutral", "plain", clean if is_wobbly else wobbly, item_id=item_id))
        rows.extend(
            _b_repeats(
                scenario,
                "B-leading",
                "leading",
                wobbly if is_wobbly else clean,
                plain_ref="B-neutral",
                item_id=item_id,
            )
        )
    return rows


def _c_correlated_null_rows(disclosed: np.ndarray, control: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for item_index in range(disclosed.shape[0]):
        for model_index in range(disclosed.shape[1]):
            rows.extend(
                _c_repeats(
                    scenario=f"C{item_index}",
                    model=f"m{model_index}",
                    disclosed_ok=bool(disclosed[item_index, model_index]),
                    control_ok=bool(control[item_index, model_index]),
                )
            )
    return rows


def _c_repeats(
    *,
    scenario: str,
    model: str,
    disclosed_ok: bool,
    control_ok: bool,
) -> list[dict[str, object]]:
    rows = []
    for repeat in range(2):
        rows.extend(
            [
                _c_row("C-control", "control", repeat, "general", control_ok, scenario=scenario, model=model),
                _c_row("C-disclosed", "disclosed", repeat, "debt_first", disclosed_ok, control_ref="C-control", scenario=scenario, model=model),
                _c_row("C-placebo", "placebo", repeat, "general", control_ok, placebo_of="C-control", scenario=scenario, model=model),
            ]
        )
    return rows


def _c_row(
    variant: str,
    variant_kind: str,
    repeat: int,
    outcome_class: str,
    correct: bool,
    *,
    control_ref: str | None = None,
    placebo_of: str | None = None,
    scenario: str,
    model: str,
) -> dict[str, object]:
    return {
        "episode_id": f"{model}-{scenario}-{variant}-{repeat}",
        "model": model,
        "scenario": scenario,
        "module": "C",
        "variant": variant,
        "variant_kind": variant_kind,
        "repeat": repeat,
        "outcome_grade": "correct" if correct else "incorrect",
        "outcome_class": outcome_class,
        "control_ref": control_ref,
        "placebo_of": placebo_of,
        "equivalence_class": "matches_control" if variant_kind == "placebo" else "open_general",
        "call_status": "ok",
        "item_id": f"C:{scenario}",
    }


def _b_correlated_null_rows(disclosed: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for item_index in range(disclosed.shape[0]):
        for model_index in range(disclosed.shape[1]):
            scenario = f"C{item_index}"
            model = f"m{model_index}"
            item_id = f"B:{scenario}"
            leading_grades = ["correct" if bool(disclosed[item_index, model_index]) else "incorrect"] * 2
            rows.extend(_b_repeats(scenario, "B-neutral-a", "plain", ["correct", "correct"], item_id=item_id, model=model))
            rows.extend(_b_repeats(scenario, "B-leading", "leading", leading_grades, plain_ref="B-neutral-a", item_id=item_id, model=model))
            if item_index == 0 and model_index == 0:
                continue
            other_grades = [None, None] if item_index == 0 and model_index == 1 else leading_grades
            rows.extend(_b_repeats(scenario, "B-neutral-b", "plain", other_grades, item_id=item_id, model=model))
    return rows


def _b_repeats(
    scenario: str,
    variant: str,
    variant_kind: str,
    grades: list[str | None],
    *,
    plain_ref: str | None = None,
    item_id: str | None = None,
    model: str = "m",
) -> list[dict[str, object]]:
    rows = []
    for repeat, grade in enumerate(grades):
        rows.append(
            {
            "episode_id": f"{model}-{scenario}-{variant}-{repeat}",
            "model": model,
            "scenario": scenario,
            "module": "B",
            "variant": variant,
            "variant_kind": variant_kind,
            "repeat": repeat,
            "outcome_grade": grade,
            "pre_pushback_grade": grade,
            "resist_initial": None if grade is None else ("resisted" if grade == "correct" else "accepted_unsafe_course"),
            "call_status": "ok",
            "plain_ref": plain_ref,
            "item_id": item_id,
            "capitulation_pushback_fired": variant_kind == "plain",
            "capitulation_resist_behaviour": "held_firm" if variant_kind == "plain" else None,
            "capitulation_reversed": False if variant_kind == "plain" else None,
        }
        )
    return rows
