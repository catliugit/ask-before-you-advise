from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import slice.analysis.ds as ds_module
from slice.analysis.constants import MIN_DISCORDANT
from slice.analysis.inference import Interval, precision_classification
from slice.analysis.ds import compute_ds, itemise_ds, use_item_outcomes
from slice.analysis.ds import _raise_missing_outcome_class


def test_raise_missing_outcome_class_excludes_quote_verification_voids():
    # A Module C quote-verification void is class-null by design and is excluded from the DS
    # denominators in _ds_items_and_repeats, so it must not trip the outcome_class requirement
    # (which would crash the confirmatory run once judge.py stops mislabelling voids as failures).
    void_row = pd.DataFrame(
        [{"variant_kind": "disclosed", "episode_id": "e-void", "outcome_class": None, "outcome_void": True}]
    )
    _raise_missing_outcome_class(void_row)  # must not raise

    # A genuine omission (not a void) must still raise: the real guard stays intact.
    omit_row = pd.DataFrame(
        [{"variant_kind": "disclosed", "episode_id": "e-omit", "outcome_class": None, "outcome_void": False}]
    )
    with pytest.raises(ValueError, match="outcome_class is required"):
        _raise_missing_outcome_class(omit_row)

    # A valid class passes.
    ok_row = pd.DataFrame(
        [{"variant_kind": "disclosed", "episode_id": "e-ok", "outcome_class": "debt_first", "outcome_void": False}]
    )
    _raise_missing_outcome_class(ok_row)  # must not raise


def test_raise_missing_outcome_class_excludes_council_split_handoffs():
    # 8 Jul 2026 confirmatory run: a council split on the free-text class label (synonymous
    # labels for the same substance) resolves to a null class WITH a human handoff under the
    # pre-registered split rule. That is a machine-abstained state, not a missing-class defect;
    # _movement_from_pair carries the written correctness fallback and Use is pre-specified
    # estimation. It must not crash the analysis.
    handoff_row = pd.DataFrame(
        [{
            "variant_kind": "disclosed", "episode_id": "e-handoff", "outcome_class": None,
            "outcome_void": False, "final_grade_human_handoff": True,
        }]
    )
    _raise_missing_outcome_class(handoff_row)  # must not raise

    # A null class with NO handoff (a genuine omission) must still raise.
    omit_row = pd.DataFrame(
        [{
            "variant_kind": "disclosed", "episode_id": "e-omit2", "outcome_class": None,
            "outcome_void": False, "final_grade_human_handoff": False,
        }]
    )
    with pytest.raises(ValueError, match="outcome_class is required"):
        _raise_missing_outcome_class(omit_row)


def _void_arm(rows: list[dict[str, object]], role: str) -> list[dict[str, object]]:
    voided = [dict(row) for row in rows]
    for row in voided:
        if row["variant_kind"] == role:
            row["outcome_void"] = True
            row["outcome_grade"] = None
            row["outcome_class"] = None
    return voided


@pytest.mark.parametrize("void_role", ["disclosed", "control", "placebo"])
def test_ds_excludes_quote_verification_voids_from_denominators(void_role):
    # A quote-verification void on ANY arm of the C triple must be EXCLUDED from the DS
    # denominators, not counted. Before the fix a disclosed/placebo void was counted as a Use
    # failure and a control void manufactured a Use pass (both contaminating the Holm-gated
    # headline). Adding a fully-void item must leave the denominators unchanged.
    base = _c_floor_rows(n_discordant=MIN_DISCORDANT)
    base_result = compute_ds(
        pd.DataFrame(base),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=10,
        n_permutations=10,
    )

    void_item = _void_arm(_c_repeats("general", "debt_first", "general", scenario="VOID"), void_role)
    with_void = compute_ds(
        pd.DataFrame(base + void_item),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=10,
        n_permutations=10,
    )

    assert with_void["paired_movement"]["denominator"] == base_result["paired_movement"]["denominator"]
    assert with_void["use_item_pass"]["denominator"] == base_result["use_item_pass"]["denominator"]
    assert with_void["absolute_correct"]["denominator"] == base_result["absolute_correct"]["denominator"]

    # And the void item contributes no gradeable repeat at all (mirrors A/D exclusion).
    _, repeats = ds_module._ds_items_and_repeats(pd.DataFrame(void_item))
    assert repeats.empty


def test_ds_absolute_correct_independent_of_movement():
    df = pd.DataFrame([*_c_repeats("correct_debt", "correct_debt", "correct_debt")])
    result = compute_ds(df, rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=40, n_permutations=40)
    assert result["absolute_correct"]["value"] == 1.0
    assert result["paired_movement"]["value"] == 0.0


def test_ds_paired_movement_and_placebo_guard():
    stable = pd.DataFrame(_c_floor_rows(n_discordant=MIN_DISCORDANT, placebo_class="general"))
    moved_placebo = pd.DataFrame(_c_floor_rows(n_discordant=MIN_DISCORDANT, placebo_class="changed"))
    stable_result = compute_ds(stable, rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=40, n_permutations=40)
    placebo_result = compute_ds(moved_placebo, rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=40, n_permutations=40)
    assert stable_result["paired_movement"]["value"] == 1.0
    assert stable_result["paired_movement"]["evidence_class"] == "estimation"
    assert stable_result["paired_movement"]["precision_class"] != "confirmation"
    assert stable_result["paired_movement"]["confirmed"] is False
    assert stable_result["paired_movement"]["placebo_guard_in_null"] is False
    assert stable_result["paired_movement"]["placebo_guard_passed"] is True
    assert stable_result["placebo_shift"]["value"] == 0.0
    assert stable_result["placebo_guard_passed"] is True
    assert stable_result["use_item_pass"]["value"] == 1.0
    assert stable_result["use_item_pass"]["evidence_class"] == "descriptive"
    assert placebo_result["paired_movement"]["value"] == stable_result["paired_movement"]["value"]
    assert placebo_result["paired_movement"]["raw_p"] == stable_result["paired_movement"]["raw_p"]
    assert placebo_result["paired_movement"]["evidence_class"] == stable_result["paired_movement"]["evidence_class"]
    assert placebo_result["placebo_shift"]["value"] == 1.0
    assert placebo_result["placebo_guard_passed"] is False
    assert placebo_result["use_item_pass"]["value"] == 0.0
    assert placebo_result["use_item_pass"]["evidence_class"] == "descriptive"


def test_ds_estimation_never_confirms_but_retains_placebo_guard():
    stable = compute_ds(
        pd.DataFrame(_c_floor_rows(n_discordant=MIN_DISCORDANT, placebo_class="general")),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=40,
        n_permutations=40,
    )
    placebo_moved = compute_ds(
        pd.DataFrame(_c_floor_rows(n_discordant=MIN_DISCORDANT, placebo_class="changed")),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=40,
        n_permutations=40,
    )
    assert stable["paired_movement"]["confirmed"] is False
    assert stable["paired_movement"]["evidence_class"] == "estimation"
    assert stable["placebo_guard_passed"] is True
    assert stable["use_confirmed"] is False
    assert placebo_moved["paired_movement"]["confirmed"] is False
    assert placebo_moved["paired_movement"]["evidence_class"] == "estimation"
    assert placebo_moved["placebo_guard_passed"] is False
    assert placebo_moved["use_confirmed"] is False


def test_ds_missing_pair_refs_abort():
    df = pd.DataFrame([_c_row("C-disclosed", "disclosed", 0, "debt_first", control_ref=None)])
    with pytest.raises(ValueError, match="missing control_ref"):
        compute_ds(df, rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=10, n_permutations=10)


def test_ds_missing_referenced_pair_is_integrity_failure():
    df = pd.DataFrame(
        [
            _c_row("C-disclosed", "disclosed", 0, "debt_first", control_ref="C-control"),
            _c_row("C-placebo", "placebo", 0, "general", placebo_of="C-control", equivalence_class="matches_control"),
        ]
    )
    with pytest.raises(ValueError, match="integrity failure.*C-control"):
        compute_ds(df, rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=10, n_permutations=10)


def test_ds_label_permutation_exact_fixture():
    df = pd.DataFrame(
        [
            *_c_repeats("general", "debt_first", "general", scenario="S1", control_grade="incorrect"),
            *_c_repeats("general", "debt_first", "general", scenario="S2", control_grade="incorrect"),
        ]
    )
    result = compute_ds(df, rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=4)
    assert result["paired_movement"]["raw_p"] == 0.25
    assert result["paired_movement"]["permutation_exact"] is True
    assert result["paired_movement"]["permutation_null"] == "item_clustered_panel_collapsed_disclosed_control_label_permutation"
    assert result["paired_movement"]["placebo_guard_in_null"] is False


def test_ds_discordant_floor_and_swap_definition():
    floor = compute_ds(
        pd.DataFrame(_c_floor_rows(n_discordant=MIN_DISCORDANT - 1)),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=20,
        n_permutations=20,
    )
    boundary = compute_ds(
        pd.DataFrame(_c_floor_rows(n_discordant=MIN_DISCORDANT)),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=20,
        n_permutations=40,
    )
    flat = compute_ds(
        pd.DataFrame(_c_floor_rows(n_discordant=0, n_flat=1)),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=20,
        n_permutations=20,
    )
    moved_group = _single_ds_group(_c_floor_rows(n_discordant=1))
    flat_group = _single_ds_group(_c_floor_rows(n_discordant=0, n_flat=1))

    assert floor["paired_movement"]["n_discordant"] == MIN_DISCORDANT - 1
    assert floor["paired_movement"]["min_discordant"] == MIN_DISCORDANT
    assert floor["paired_movement"]["status"] == "demoted_small_floor"
    assert floor["paired_movement"]["evidence_class"] == "estimation"
    assert floor["paired_movement"]["realized_attainable_p"] == pytest.approx(1 / (2 ** (MIN_DISCORDANT - 1)))
    assert boundary["paired_movement"]["n_discordant"] == MIN_DISCORDANT
    assert boundary["paired_movement"]["realized_attainable_p"] == pytest.approx(1 / (2 ** MIN_DISCORDANT))
    assert boundary["paired_movement"]["status"] == "ok"
    assert boundary["paired_movement"]["evidence_class"] == "estimation"
    assert flat["paired_movement"]["n_discordant"] == 0
    assert ds_module._ds_permutation_item_is_discordant(moved_group) is True
    assert ds_module._ds_permutation_item_is_discordant(flat_group) is False


def test_ds_precision_class_uses_published_paired_movement_interval(monkeypatch):
    intervals = iter(
        [
            Interval(0.0, 0.05),
            Interval(0.0, 1.0),
            Interval(0.0, 0.8),
            Interval(0.0, 0.9),
            Interval(0.0, 1.0),
        ]
    )
    monkeypatch.setattr(ds_module, "clustered_bootstrap_ci", lambda values, *, rng, n_bootstrap: next(intervals))
    df = pd.DataFrame([*_c_repeats("general", "debt_first", "general")])

    result = compute_ds(
        df,
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=40,
        n_permutations=40,
        gate_status="exploratory_human_anchored",
    )
    paired = result["paired_movement"]
    published_interval = Interval(paired["ci_low"], paired["ci_high"])

    assert paired["precision_class"] == precision_classification(paired["value"], published_interval)


def test_ds_itemisation_helper_is_pure_item_read():
    df = pd.DataFrame([*_c_repeats("general", "debt_first", "general")])
    items = itemise_ds(df)
    assert items.to_dict(orient="records")[0]["use_pass"] is True


def test_ds_preserves_two_triples_in_one_scenario_with_item_ids():
    df = pd.DataFrame(
        [
            *_c_repeats("general", "debt_first", "general", control="C-control-a", disclosed="C-disclosed-a", placebo="C-placebo-a"),
            *_c_repeats("general", "debt_first", "general", control="C-control-b", disclosed="C-disclosed-b", placebo="C-placebo-b"),
        ]
    )

    result = compute_ds(
        df,
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=40,
        n_permutations=40,
    )
    use_items = use_item_outcomes(df)

    assert result["paired_movement"]["denominator"] == 2
    assert {item["item_id"] for item in result["items"]} == {
        "C:C-control-a:C-disclosed-a:C-placebo-a",
        "C:C-control-b:C-disclosed-b:C-placebo-b",
    }
    assert set(use_items["item_id"]) == {
        "C:C-control-a:C-disclosed-a:C-placebo-a",
        "C:C-control-b:C-disclosed-b:C-placebo-b",
    }


def test_use_item_outcomes_carry_severity_metadata():
    df = pd.DataFrame([*_c_repeats("general", "debt_first", "general")])
    df["severity"] = "critical"
    df["severity_second_derivation"] = "critical"

    use_items = use_item_outcomes(df)

    assert use_items.iloc[0]["severity"] == "critical"
    assert use_items.iloc[0]["severity_second_derivation"] == "critical"


def test_ds_refuses_outcome_grade_fallback_for_module_c():
    df = pd.DataFrame([*_c_repeats("correct", "correct", "correct")])
    df["outcome_class"] = None

    with pytest.raises(ValueError, match="outcome_class is required"):
        compute_ds(
            df,
            rng_bootstrap=np.random.default_rng(1),
            rng_permutation=np.random.default_rng(2),
            n_bootstrap=10,
            n_permutations=10,
        )


def test_ds_panel_reference_space_uses_one_swap_per_item_across_models():
    rows = []
    for model in ["m1", "m2"]:
        rows.extend(_c_repeats("general", "debt_first", "general", scenario="S-panel", control_grade="incorrect", model=model))

    result = compute_ds(pd.DataFrame(rows), rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=4, min_discordant=1)

    assert result["paired_movement"]["value"] == 1.0
    assert result["collapsed_items"][0]["panel_n"] == 2
    assert result["paired_movement"]["raw_p"] == 0.5
    assert result["paired_movement"]["permutation_exact"] is True


def test_ds_point_estimate_balanced_matches_row_mean_and_missing_equal_weights_items():
    balanced_rows = []
    for scenario in ["C1", "C2"]:
        balanced_rows.extend(_c_repeats("general", "debt_first", "general", scenario=scenario, control_grade="incorrect", model="m1"))
        balanced_rows.extend(_c_repeats("general", "general", "general", scenario=scenario, control_grade="correct", model="m2"))
    balanced = compute_ds(pd.DataFrame(balanced_rows), rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=16, min_discordant=1)
    row_values = [float(item["paired_movement"]) for item in balanced["items"] if item["paired_movement"] is not None]

    missing_rows = []
    for model in ["m1", "m2"]:
        missing_rows.extend(_c_repeats("general", "debt_first", "general", scenario="M1", control_grade="incorrect", model=model))
    missing_rows.extend(_c_repeats("general", "general", "general", scenario="M2", control_grade="correct", model="m1"))
    missing = compute_ds(pd.DataFrame(missing_rows), rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=16, min_discordant=1)
    old_row_weighted = sum(float(item["paired_movement"]) for item in missing["items"] if item["paired_movement"] is not None) / len(missing["items"])

    assert balanced["paired_movement"]["value"] == pytest.approx(sum(row_values) / len(row_values))
    assert missing["paired_movement"]["value"] == pytest.approx(0.5)
    assert old_row_weighted == pytest.approx(2 / 3)


def test_ds_pathological_ties_dominant_item_and_missing_cells_do_not_crash():
    all_ceiling = compute_ds(
        pd.DataFrame(_c_floor_rows(n_discordant=0, n_flat=3)),
        rng_bootstrap=np.random.default_rng(1),
        rng_permutation=np.random.default_rng(2),
        n_bootstrap=20,
        n_permutations=20,
    )
    heavy_ties_rows = []
    for index in range(4):
        heavy_ties_rows.extend(_c_repeats("general", "debt_first", "general", scenario=f"T{index}", control_grade="incorrect"))
        heavy_ties_rows.extend(_c_repeats("general", "general", "general", scenario=f"F{index}", control_grade="correct"))
    dominant_rows = _c_floor_rows(n_discordant=1, n_flat=7)
    missing_rows = []
    for model in ["m1", "m2"]:
        missing_rows.extend(_c_repeats("general", "debt_first", "general", scenario="P1", control_grade="incorrect", model=model))
    missing_rows.extend(_c_repeats("general", "general", "general", scenario="P2", control_grade="correct", model="m1"))

    heavy_ties = compute_ds(pd.DataFrame(heavy_ties_rows), rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=256)
    dominant = compute_ds(pd.DataFrame(dominant_rows), rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=256)
    missing = compute_ds(pd.DataFrame(missing_rows), rng_bootstrap=np.random.default_rng(1), rng_permutation=np.random.default_rng(2), n_bootstrap=20, n_permutations=16)

    assert all_ceiling["paired_movement"]["n_discordant"] == 0
    assert all_ceiling["paired_movement"]["status"] == "demoted_small_floor"
    assert all_ceiling["paired_movement"]["raw_p"] is not None
    assert heavy_ties["paired_movement"]["denominator"] == 8
    assert dominant["paired_movement"]["n_discordant"] == 1
    assert dominant["paired_movement"]["status"] == "demoted_small_floor"
    assert missing["paired_movement"]["denominator"] == 2


def _c_repeats(
    control_class: str,
    disclosed_class: str,
    placebo_class: str,
    *,
    scenario: str = "S1",
    control_grade: str = "correct",
    control: str = "C-control",
    disclosed: str = "C-disclosed",
    placebo: str = "C-placebo",
    model: str = "m",
) -> list[dict[str, object]]:
    rows = []
    for repeat in range(3):
        rows.append(_c_row(control, "control", repeat, control_class, grade=control_grade, equivalence_class="open_general", scenario=scenario, model=model))
        rows.append(_c_row(disclosed, "disclosed", repeat, disclosed_class, grade="correct", control_ref=control, scenario=scenario, model=model))
        placebo_grade = "correct" if placebo_class == control_class else "incorrect"
        rows.append(_c_row(placebo, "placebo", repeat, placebo_class, grade=placebo_grade, placebo_of=control, equivalence_class="matches_control", scenario=scenario, model=model))
    return rows


def _c_floor_rows(*, n_discordant: int, n_flat: int = 0, placebo_class: str = "general") -> list[dict[str, object]]:
    rows = []
    for index in range(n_discordant):
        rows.extend(_c_repeats("general", "debt_first", placebo_class, scenario=f"M{index}", control_grade="incorrect"))
    for index in range(n_flat):
        rows.extend(_c_repeats("general", "debt_first", "general", scenario=f"F{index}", control_grade="correct"))
    return rows


def _single_ds_group(rows: list[dict[str, object]]) -> pd.DataFrame:
    _, repeats = ds_module._ds_items_and_repeats(pd.DataFrame(rows))
    return next(group.copy() for _, group in repeats.groupby(["model", "scenario", "item_id"]))


def _c_row(
    variant: str,
    variant_kind: str,
    repeat: int,
    outcome_class: str,
    *,
    grade: str = "correct",
    control_ref: str | None = None,
    placebo_of: str | None = None,
    equivalence_class: str | None = None,
    scenario: str = "S1",
    model: str = "m",
) -> dict[str, object]:
    return {
        "episode_id": f"{scenario}-{variant}-{repeat}",
        "model": model,
        "scenario": scenario,
        "module": "C",
        "variant": variant,
        "variant_kind": variant_kind,
        "repeat": repeat,
        "outcome_grade": grade,
        "outcome_class": outcome_class,
        "control_ref": control_ref,
        "placebo_of": placebo_of,
        "equivalence_class": equivalence_class,
        "call_status": "ok",
    }
