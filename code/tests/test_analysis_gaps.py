from __future__ import annotations

import pandas as pd

from slice.analysis.gaps import gap_accounting


def test_gap_accounting_counts_void_missing_and_reruns():
    df = pd.DataFrame(
        [
            {
                "episode_id": "e1",
                "model": "m",
                "scenario": "s",
                "module": "A",
                "variant": "A1",
                "repeat": 0,
                "call_status": "ok",
                "dim_s1_d1_void": True,
                "outcome_void": True,
                "outcome_grade": "correct",
                "scoring_failed": False,
                "rerun_due_to_persona_leak": True,
                "rerun_count": 2,
            },
            {
                "episode_id": "e2",
                "model": "m",
                "scenario": "s",
                "module": "A",
                "variant": "A1",
                "repeat": 1,
                "call_status": "missing",
                "dim_s1_d1_void": False,
                "outcome_void": False,
                "outcome_grade": "correct",
                "scoring_failed": False,
                "rerun_due_to_persona_leak": False,
                "rerun_count": 0,
            },
            {
                "episode_id": "e3",
                "model": "m",
                "scenario": "s",
                "module": "A",
                "variant": "A1",
                "repeat": 2,
                "call_status": "ok",
                "dim_s1_d1_void": False,
                "outcome_void": False,
                "outcome_grade": "correct",
                "scoring_failed": False,
                "rerun_due_to_persona_leak": False,
                "rerun_count": 0,
            },
        ]
    )
    result = gap_accounting(df)
    assert result["void_verdicts"]["n"] == 1
    assert result["void_verdicts"]["denominator"] == 2
    assert result["void_verdicts"]["value"] == 0.5
    assert result["void_verdicts"]["dimension_void_counts"]["dim_s1_d1_void"] == 1
    assert result["missing_cells"]["denominator"] == 3
    assert result["missing_cells"]["cells"][0]["episode_id"] == "e2"
    assert result["persona_leak_reruns"]["value"] == 1 / 3
    assert result["persona_leak_reruns"]["rerun_count_ge_2"][0]["episode_id"] == "e1"


def test_missing_null_grade_is_not_void():
    df = pd.DataFrame(
        [
            {
                "episode_id": "missing",
                "model": "m",
                "scenario": "s",
                "module": "A",
                "variant": "A1",
                "repeat": 0,
                "call_status": "missing",
                "outcome_void": False,
                "outcome_grade": None,
                "scoring_failed": False,
            }
        ]
    )
    result = gap_accounting(df)
    assert result["void_verdicts"]["n"] == 0
    assert result["void_verdicts"]["denominator"] == 0
    assert result["missing_cells"]["n"] == 1
