"""Drift guard: every coded variable named in grading-codebook.md Section 10 must map to a
schema field, so the pipeline can store what the codebook says to measure.

This is the cold-read drift guard: the codebook
defines coded variables the schema must be able to persist. If a future codebook revision adds a
coded variable, this test should be extended; if the schema drops a field, this test should fail.
"""

import typing

import pandas as pd

from slice.analysis.ds import itemise_ds
from slice.schema import Capitulation, Pass1Dimension, Pass2


def _literal_members(annotation: object) -> set[str]:
    """Flatten the string members of an Optional[Literal[...]] annotation."""
    members: set[str] = set()
    for arg in typing.get_args(annotation):
        if arg is type(None):
            continue
        sub = typing.get_args(arg)
        if sub:
            members |= {s for s in sub if isinstance(s, str)}
        elif isinstance(arg, str):
            members.add(arg)
    return members


def test_section10_ask_axis_has_storage():
    # Ask fact label (per critical dimension) and the late_asked timing flag.
    assert "label" in Pass1Dimension.model_fields
    assert "asked_turn" in Pass1Dimension.model_fields
    assert "late_asked" in Pass1Dimension.model_fields
    assert "timing_missing" in Pass1Dimension.model_fields
    assert _literal_members(Pass1Dimension.model_fields["label"].annotation) == {
        "elicited",
        "branch_covered",
        "unconditioned",
    }


def test_section10_use_axis_withdrawn_keeps_only_mechanical_diagnostic_storage():
    # 2026-07-07 mechanical-override-fix SPEC v2 C3 + codebook edit:
    # the Use mark was withdrawn from grading-codebook.md because prompts never ask judges for it.
    assert "disclosed_fact_use" not in Pass2.model_fields
    assert "mechanical_disclosed_fact_use" in Pass2.model_fields


def test_section10_resist_axis_has_storage():
    # Resist is two marks: first-contact (resist_initial) and post-pushback (resist_behaviour).
    assert "resist_initial" in Pass2.model_fields
    assert "resist_behaviour" in Pass2.model_fields
    assert _literal_members(Pass2.model_fields["resist_initial"].annotation) >= {
        "resisted",
        "accepted_unsafe_course",
        "not_applicable",
    }
    assert _literal_members(Pass2.model_fields["resist_behaviour"].annotation) >= {
        "held_firm",
        "caved",
        "not_triggered",
        "not_applicable",
    }


def test_section10_outcome_boundary_signposting_have_storage():
    for field in (
        "outcome_grade",
        "deferral_score",
        "signposting_mark",
        "accompanied_by_refusal",
        "context_correct",
    ):
        assert field in Pass2.model_fields


def test_mechanical_override_fix_renamed_diagnostics_are_schema_fields():
    # 2026-07-07 mechanical-override-fix SPEC v2 C1/C4/C5: diagnostics are stored
    # under mechanical_* names and scenario rule metadata is track_rule.
    for field in (
        "mechanical_deferral_score",
        "mechanical_disclosed_fact_use",
        "mechanical_signposting_level",
        "mechanical_accompanied_by_refusal",
        "mechanical_named_correct_resource",
        "track_rule",
        "judge_cited_rule",
        "judge_safety_flag",
    ):
        assert field in Pass2.model_fields
    assert "named_correct_resource" not in Pass2.model_fields
    assert "cited_rule" not in Pass2.model_fields


def test_section10_placebo_shift_has_storage():
    # Placebo shift is the one Section 10 variable stored as a DS item-level computation, not Pass2.
    assert "placebo_shift" not in Pass2.model_fields

    stable_rows = [
        _c_row("C-control", "control", 0, "general", equivalence_class="open_general"),
        _c_row("C-disclosed", "disclosed", 0, "debt_first", control_ref="C-control"),
        _c_row("C-placebo", "placebo", 0, "general", placebo_of="C-control", equivalence_class="matches_control"),
        _c_row("C-control", "control", 1, "general", equivalence_class="open_general"),
        _c_row("C-disclosed", "disclosed", 1, "debt_first", control_ref="C-control"),
        _c_row("C-placebo", "placebo", 1, "general", placebo_of="C-control", equivalence_class="matches_control"),
    ]
    stable_items = itemise_ds(pd.DataFrame(stable_rows))

    assert "placebo_shift" in stable_items.columns
    assert bool(stable_items.iloc[0]["placebo_shift"]) is False

    indeterminate_rows = [
        _c_row("C-control", "control", 0, "general", equivalence_class="open_general"),
        _c_row("C-disclosed", "disclosed", 0, "debt_first", control_ref="C-control"),
        _c_row("C-placebo", "placebo", 0, "general", placebo_of="C-control", equivalence_class="matches_control"),
        _c_row("C-control", "control", 1, "general", equivalence_class="open_general"),
        _c_row("C-disclosed", "disclosed", 1, "debt_first", control_ref="C-control"),
        _c_row("C-placebo", "placebo", 1, "changed", placebo_of="C-control", equivalence_class="matches_control"),
    ]
    indeterminate_items = itemise_ds(pd.DataFrame(indeterminate_rows))

    # A genuinely indeterminate placebo item is null, not coerced into a non-placebo not_applicable bucket.
    assert pd.isna(indeterminate_items.iloc[0]["placebo_shift"])


def test_episode_resist_record_mirrors_the_two_marks():
    # The per-episode resist record must be able to carry both marks too.
    assert "resist_initial" in Capitulation.model_fields
    assert _literal_members(Capitulation.model_fields["resist_behaviour"].annotation) >= {
        "held_firm",
        "caved",
        "not_triggered",
        "not_applicable",
    }


def _c_row(
    variant: str,
    variant_kind: str,
    repeat: int,
    outcome_class: str,
    *,
    control_ref: str | None = None,
    placebo_of: str | None = None,
    equivalence_class: str | None = None,
) -> dict[str, object]:
    return {
        "episode_id": f"S1-{variant}-{repeat}",
        "model": "m",
        "scenario": "S1",
        "module": "C",
        "variant": variant,
        "variant_kind": variant_kind,
        "repeat": repeat,
        "outcome_grade": "correct",
        "outcome_class": outcome_class,
        "control_ref": control_ref,
        "placebo_of": placebo_of,
        "equivalence_class": equivalence_class,
        "call_status": "ok",
    }
