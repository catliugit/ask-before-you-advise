from __future__ import annotations

import numpy as np
import pandas as pd

from slice.analysis.specgap import compute_spec_gap


def test_spec_gap_positive_with_prompt_supplied_fact_established():
    df = pd.DataFrame(
        [
            *_a_repeats("A1", "profile", "incorrect", present=False, asked=False, branch=False),
            *_a_repeats("A-null", "fully_specified", "correct", present=True, asked=False, branch=False),
        ]
    )
    result = compute_spec_gap(df, rng_bootstrap=np.random.default_rng(1), n_bootstrap=40)
    assert result["value"] == 1.0
    assert result["evidence_class"] == "estimation"
    assert result["broken_item_flags"] == []
    assert result["positive_control"]["event"] == "establishment_only"
    assert result["positive_control"]["status"] == "ok"


def test_spec_gap_incorrect_but_established_fully_specified_item_is_not_positive_control_broken():
    df = pd.DataFrame(
        [
            *_a_repeats("A1", "profile", "correct", present=True),
            *_a_repeats("A-null", "fully_specified", "incorrect", present=True),
        ]
    )
    result = compute_spec_gap(df, rng_bootstrap=np.random.default_rng(1), n_bootstrap=40)
    assert result["status"] == "ok"
    assert result["positive_control"]["status"] == "ok"
    assert result["broken_item_flags"] == []
    assert result["value"] == -1.0


def test_spec_gap_broken_fully_specified_establishment_event_is_flagged():
    df = pd.DataFrame(
        [
            *_a_repeats("A1", "profile", "correct", present=True),
            *_a_repeats("A-null", "fully_specified", "correct", present=False),
        ]
    )
    result = compute_spec_gap(df, rng_bootstrap=np.random.default_rng(1), n_bootstrap=40)
    assert result["status"] == "flagged_broken_item"
    assert result["positive_control"]["event"] == "establishment_only"
    assert result["broken_item_flags"][0]["fully_specified_variant"] == "A-null"


def test_spec_gap_void_in_either_arm_excludes_pair():
    df = pd.DataFrame(
        [
            *_a_repeats("A1", "profile", "incorrect", present=False),
            *_a_repeats("A-null", "fully_specified", "correct", present=True, void=True),
        ]
    )
    result = compute_spec_gap(df, rng_bootstrap=np.random.default_rng(1), n_bootstrap=40)
    assert result["denominator"] == 0
    assert result["void_pair_exclusions"][0]["profile_variant"] == "A1"


def _a_repeats(
    variant: str,
    variant_kind: str,
    grade: str,
    *,
    present: bool,
    asked: bool = False,
    branch: bool = False,
    void: bool = False,
) -> list[dict[str, object]]:
    return [
        {
            "episode_id": f"{variant}-{repeat}",
            "model": "m",
            "scenario": "S1",
            "module": "A",
            "variant": variant,
            "variant_kind": variant_kind,
            "repeat": repeat,
            "call_status": "ok",
            "first_directive_turn": 1,
            "outcome_grade": grade,
            "outcome_void": False,
            "scoring_failed": False,
            "dim_s1_d1": "elicited" if asked else "unconditioned",
            "dim_s1_d1_void": void,
            "dim_s1_d1_cls": "critical",
            "est_s1_d1_present_in_prompt": present,
            "est_s1_d1_asked_for": asked,
            "est_s1_d1_branch_covered": branch,
        }
        for repeat in range(3)
    ]
