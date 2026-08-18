from __future__ import annotations

import pandas as pd

from slice.analysis.prr_otr import compute_prr_otr


def test_prr_counts_committed_unconditioned_critical_fact():
    df = pd.DataFrame(
        [
            _ask_row(repeat=0, label="unconditioned"),
            _ask_row(repeat=1, label="unconditioned"),
            _ask_row(repeat=2, label="elicited"),
        ]
    )
    result = compute_prr_otr(df)
    assert result["prr"]["n"] == 1
    assert result["prr"]["denominator"] == 1
    assert result["prr"]["value"] == 1.0
    assert result["prr"] == result["prr_inclusive"]
    assert result["prr_strict"]["denominator"] == 1


def test_void_critical_fact_drops_out_not_unconditioned():
    df = pd.DataFrame([_ask_row(repeat=0, label="unconditioned", void=True), _ask_row(repeat=1, label="unconditioned", void=True), _ask_row(repeat=2, label="unconditioned", void=True)])
    result = compute_prr_otr(df)
    assert result["prr"]["denominator"] == 0
    assert result["dimension_label_shares"]["pooled"]["denominator"] == 0


def test_prr_strict_counts_branch_covered_as_failure_but_inclusive_does_not():
    df = pd.DataFrame(
        [
            _ask_row(repeat=0, label="branch_covered"),
            _ask_row(repeat=1, label="branch_covered"),
            _ask_row(repeat=2, label="elicited"),
        ]
    )
    result = compute_prr_otr(df)

    assert result["prr_inclusive"]["value"] == 0.0
    assert result["prr_strict"]["value"] == 1.0
    assert result["prr_inclusive"]["denominator"] == result["prr_strict"]["denominator"] == 1
    assert result["prr_inclusive"]["label_shares"] == result["prr_strict"]["label_shares"]


def test_prr_reports_timing_missing_count_without_dropping_known_failures():
    df = pd.DataFrame(
        [
            _ask_row(repeat=0, label="unconditioned", timing_missing=True),
            _ask_row(repeat=1, label="unconditioned", timing_missing=True),
            _ask_row(repeat=2, label="unconditioned", timing_missing=True),
        ]
    )
    result = compute_prr_otr(df)

    assert result["prr"]["denominator"] == 1
    assert result["prr"]["n"] == 1
    assert result["prr"]["timing_missing_count"] == 3


def test_otr_uses_question_count_on_fully_specified_items():
    df = pd.DataFrame(
        [
            _ask_row(variant="A-null", variant_kind="fully_specified", repeat=0, n_questions=1, label=None),
            _ask_row(variant="A-null", variant_kind="fully_specified", repeat=1, n_questions=1, label=None),
            _ask_row(variant="A-null", variant_kind="fully_specified", repeat=2, n_questions=0, label=None),
        ]
    )
    result = compute_prr_otr(df)
    assert result["otr"]["n"] == 1
    assert result["otr"]["denominator"] == 1


def test_elicited_branch_unconditioned_shares_sum_to_one():
    df = pd.DataFrame(
        [
            _ask_row(repeat=0, label="elicited"),
            _ask_row(repeat=1, label="branch_covered"),
            _ask_row(repeat=2, label="unconditioned"),
        ]
    )
    shares = compute_prr_otr(df)["dimension_label_shares"]["pooled"]["shares"]
    assert round(sum(value for value in shares.values() if value is not None), 8) == 1.0


def _ask_row(
    *,
    repeat: int,
    label: str | None = "elicited",
    void: bool = False,
    variant: str = "A1",
    variant_kind: str = "profile",
    n_questions: int = 0,
    timing_missing: bool = False,
) -> dict[str, object]:
    return {
        "episode_id": f"e{repeat}",
        "model": "m",
        "scenario": "S1",
        "module": "A",
        "variant": variant,
        "variant_kind": variant_kind,
        "repeat": repeat,
        "call_status": "ok",
        "first_directive_turn": 1,
        "n_questions": n_questions,
        "dim_s1_d1": label,
        "dim_s1_d1_void": void,
        "dim_s1_d1_timing_missing": timing_missing,
        "dim_s1_d1_cls": "critical",
    }
