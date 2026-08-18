from __future__ import annotations

import numpy as np
import pandas as pd

from slice.analysis.ds import compute_ds
from slice.analysis.fcr import compute_fcr
from slice.analysis.inference import (
    Interval,
    clustered_bootstrap_ci,
    holm_step_down,
    min_discordant_for_confirmation,
    permutation_inverted_ci_from_stats,
    precision_classification,
)
from slice.analysis.constants import CONFIRMATORY_BAR, MIN_DISCORDANT


def test_min_discordant_floor_uses_holm_attainable_p():
    floor = min_discordant_for_confirmation([2] * 12, CONFIRMATORY_BAR)

    assert floor == MIN_DISCORDANT == 5


def test_permutation_inverted_ci_uses_null_reference_space():
    interval = permutation_inverted_ci_from_stats(0.4, [-0.1, 0.0, 0.2, 0.3])

    assert interval.low < 0.4 < interval.high


def test_production_bootstrap_t_ci_covers_known_effect_at_twelve_clusters():
    rng = np.random.default_rng(20260703)
    n_sims = 100
    ds_effect = 0.55
    fcr_effect = 0.45
    ds_covered = 0
    fcr_covered = 0
    ds_methods = set()
    fcr_methods = set()
    for sim_index in range(n_sims):
        ds = compute_ds(
            pd.DataFrame(_known_effect_ds_rows(rng, ds_effect)),
            rng_bootstrap=np.random.default_rng(1000 + sim_index),
            rng_permutation=np.random.default_rng(2000 + sim_index),
            n_bootstrap=399,
            n_permutations=2**12,
            min_discordant=1,
            expected_repeats=2,
        )["paired_movement"]
        fcr = compute_fcr(
            pd.DataFrame(_known_effect_fcr_rows(rng, fcr_effect)),
            rng_bootstrap=np.random.default_rng(3000 + sim_index),
            rng_permutation=np.random.default_rng(4000 + sim_index),
            n_bootstrap=399,
            n_permutations=2**12,
            min_discordant=1,
            expected_repeats=2,
        )
        ds_covered += ds["ci_low"] <= ds_effect <= ds["ci_high"]
        fcr_covered += fcr["ci_low"] <= fcr_effect <= fcr["ci_high"]
        ds_methods.add(ds["ci_method"])
        fcr_methods.add(fcr["ci_method"])
        assert ds["raw_p"] is not None
        assert fcr["raw_p"] is not None

    assert ds_methods == {"bootstrap_t"}
    assert fcr_methods == {"bootstrap_t"}
    assert 0.93 <= ds_covered / n_sims <= 0.98
    assert 0.93 <= fcr_covered / n_sims <= 0.98


def test_bootstrap_ci_brackets_known_effect_and_is_seeded():
    first = clustered_bootstrap_ci([0, 1, 1, 1], rng=np.random.default_rng(4), n_bootstrap=200)
    second = clustered_bootstrap_ci([0, 1, 1, 1], rng=np.random.default_rng(4), n_bootstrap=200)
    assert first.low <= 0.75 <= first.high
    assert first == second


def test_holm_two_p_values_step_down():
    result = holm_step_down({"fcr": 0.01, "ds": 0.04})
    assert result["fcr"]["alpha_bar"] == 0.025
    assert result["fcr"]["confirmed"] is True
    assert result["ds"]["alpha_bar"] == 0.05
    assert result["ds"]["confirmed"] is True


def test_holm_fixed_family_does_not_shrink_after_demotion():
    result = holm_step_down({"fcr": 0.03, "ds": None}, family_size=2)

    assert result["fcr"]["alpha_bar"] == 0.025
    assert result["fcr"]["adjusted_p"] == 0.06
    assert result["fcr"]["confirmed"] is False
    assert result["ds"]["confirmed"] is False


def test_holm_single_fcr_family_uses_full_alpha_bar():
    result = holm_step_down({"resist_fcr_excess": 0.03}, family_size=1)

    assert result["resist_fcr_excess"]["alpha_bar"] == CONFIRMATORY_BAR == 0.05
    assert result["resist_fcr_excess"]["adjusted_p"] == 0.03
    assert result["resist_fcr_excess"]["confirmed"] is True


def test_precision_classifier():
    assert precision_classification(0.25, Interval(0.0, 0.7)) == "estimation"
    assert precision_classification(0.45, Interval(0.42, 0.48)) == "precision-bounded"
    assert precision_classification(0.01, Interval(-0.1, 0.1), confirmatory_test=True) == "confirmation"


def _known_effect_ds_rows(rng: np.random.Generator, effect: float) -> list[dict[str, object]]:
    rows = []
    for item_index in range(12):
        moved = bool(rng.random() < effect)
        for repeat in range(2):
            rows.extend(
                [
                    _c_row(f"C{item_index}", "C-control", "control", repeat, "general", False),
                    _c_row(f"C{item_index}", "C-disclosed", "disclosed", repeat, "debt_first", moved, control_ref="C-control"),
                    _c_row(f"C{item_index}", "C-placebo", "placebo", repeat, "general", False, placebo_of="C-control"),
                ]
            )
    return rows


def _c_row(
    scenario: str,
    variant: str,
    variant_kind: str,
    repeat: int,
    outcome_class: str,
    correct: bool,
    *,
    control_ref: str | None = None,
    placebo_of: str | None = None,
) -> dict[str, object]:
    return {
        "episode_id": f"{scenario}-{variant}-{repeat}",
        "model": "m",
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


def _known_effect_fcr_rows(rng: np.random.Generator, effect: float) -> list[dict[str, object]]:
    rows = []
    for item_index in range(12):
        failed = bool(rng.random() < effect)
        scenario = f"B{item_index}"
        item_id = f"B:{scenario}"
        rows.extend(_b_repeats(scenario, "B-neutral", "plain", [True, True], item_id=item_id))
        rows.extend(_b_repeats(scenario, "B-leading", "leading", [not failed, not failed], plain_ref="B-neutral", item_id=item_id))
    return rows


def _b_repeats(
    scenario: str,
    variant: str,
    variant_kind: str,
    passes: list[bool],
    *,
    plain_ref: str | None = None,
    item_id: str,
) -> list[dict[str, object]]:
    rows = []
    for repeat, passed in enumerate(passes):
        grade = "correct" if passed else "incorrect"
        rows.append(
            {
                "episode_id": f"{scenario}-{variant}-{repeat}",
                "model": "m",
                "scenario": scenario,
                "module": "B",
                "variant": variant,
                "variant_kind": variant_kind,
                "repeat": repeat,
                "outcome_grade": grade,
                "pre_pushback_grade": grade,
                "resist_initial": "resisted" if passed else "accepted_unsafe_course",
                "call_status": "ok",
                "plain_ref": plain_ref,
                "item_id": item_id,
                "capitulation_pushback_fired": variant_kind == "plain",
                "capitulation_resist_behaviour": "held_firm" if variant_kind == "plain" else None,
                "capitulation_reversed": False if variant_kind == "plain" else None,
            }
        )
    return rows
