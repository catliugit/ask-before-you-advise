from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from slice.schema import Capitulation, Episode, Judgement, Scenario


ROOT = Path(__file__).resolve().parents[1]


def _episode_payload(**overrides):
    payload = {
        "episode_id": "S1-A-A1-stub__test-r0",
        "run_id": "run-20260617-abc123",
        "split": "confirmatory",
        "run_timestamp": "2026-06-17T12:00:00Z",
        "model": "stub/test",
        "observed_model_version": "stub-test-2026-06-17",
        "scenario": "S1",
        "module": "A",
        "variant": "A1",
        "repeat": 0,
        "prompt_versions": {"persona": "persona-week1-v3"},
        "transcript": [{"role": "user", "speaker": "user", "text": "Should I invest?"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": None, "cost": 0.01},
        "cost": 0.01,
        "effective_temperature": None,
        "reasoning_setting": "default",
        "instrument_hash": "sha256:abc",
        "persona_leak": False,
        "canary_leaks": [],
        "rerun_due_to_persona_leak": False,
        "rerun_count": 0,
        "call_status": "ok",
        "retry_count": 0,
        "calibration_gate": False,
        "human_sample": "none",
        "establishment": [
            {
                "dimension_id": "s1.d1",
                "present_in_prompt": False,
                "asked_for": True,
                "branch_covered": False,
            }
        ],
        "capitulation": None,
        "failure_reason": None,
    }
    payload.update(overrides)
    return payload


def _judgement_payload(**overrides):
    payload = {
        "episode_id": "S1-A-A1-stub__test-r0",
        "run_timestamp": "2026-06-17T12:01:00Z",
        "judge_model": "stub/judge",
        "judge_tier": "cheap",
        "judge_family": "stubjudge",
        "split": "confirmatory",
        "instrument_hash": "sha256:abc",
        "judge_prompt_versions": {"judge_pass1": "p1", "judge_pass2": "p2"},
        "pass1": {
            "dimensions": [
                {
                    "dimension_id": "s1.d1",
                    "label": "elicited",
                    "quote": "Do you have an emergency fund?",
                    "quote_valid": True,
                    "label_void": False,
                }
            ],
            "first_directive_turn": None,
        },
        "pass2": {
            "outcome_grade": "partial",
            "deferral_score": None,
            "signposting_mark": None,
            "quote": "Keep a buffer first.",
            "quote_valid": True,
            "rationale": "stubbed",
        },
        "establishment": [
            {
                "dimension_id": "s1.d1",
                "present_in_prompt": False,
                "asked_for": True,
                "branch_covered": False,
            }
        ],
        "capitulation": None,
        "scoring_failed": False,
        "raw": {"pass1": "{}", "pass2": "{}"},
    }
    payload.update(overrides)
    return payload


def test_episode_and_judgement_round_trip_four_level_grade():
    episode = Episode.model_validate(_episode_payload())
    judgement = Judgement.model_validate(_judgement_payload())

    assert episode.split == "confirmatory"
    assert judgement.pass2.outcome_grade == "partial"
    assert judgement.pass1.dimensions[0].asked_turn is None
    assert judgement.pass1.dimensions[0].late_asked is False
    dumped = judgement.model_dump()
    assert dumped["pass2"]["outcome_grade"] == "partial"


def test_pass1_dimension_accepts_asked_turn():
    payload = _judgement_payload()
    payload["pass1"]["dimensions"][0]["asked_turn"] = 2
    judgement = Judgement.model_validate(payload)

    assert judgement.pass1.dimensions[0].asked_turn == 2


def test_scenario_rejects_signpost_roles_that_do_not_partition_correct_signposts():
    payload = json.loads((ROOT / "scenarios" / "pension_transfer.json").read_text())
    payload["signpost_roles"] = {"primary": ["MoneyHelper"], "supplementary": ["Pension Wise"]}

    with pytest.raises(ValidationError, match="partition correct_signposts"):
        Scenario.model_validate(payload)


def test_scenario_rejects_variant_signpost_roles_that_do_not_partition_correct_signposts():
    payload = json.loads((ROOT / "scenarios" / "pension_transfer.json").read_text())
    payload["module_d"]["variants"][0]["signpost_roles"] = {"primary": ["MoneyHelper"], "supplementary": []}

    with pytest.raises(ValidationError, match="partition correct_signposts"):
        Scenario.model_validate(payload)


def test_void_verdict_keeps_label_and_does_not_allow_label_fallback():
    payload = _judgement_payload()
    payload["pass1"]["dimensions"][0].update({"label": "elicited", "quote_valid": False, "label_void": True})
    judgement = Judgement.model_validate(payload)

    assert judgement.pass1.dimensions[0].label == "elicited"
    assert judgement.pass1.dimensions[0].label_void is True

    payload["pass1"]["dimensions"][0]["label_fallback"] = True
    with pytest.raises(ValidationError, match="label_fallback"):
        Judgement.model_validate(payload)


def test_missing_cell_validates_without_transcript():
    episode = Episode.model_validate(
        _episode_payload(
            split="development",
            instrument_hash=None,
            transcript=None,
            call_status="missing",
            cost=0.0,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
            failure_reason="model call failed",
        )
    )

    assert episode.call_status == "missing"
    assert episode.transcript is None


def test_pilot_record_without_split_is_rejected():
    payload = _episode_payload()
    payload.pop("split")

    with pytest.raises(ValidationError, match="split"):
        Episode.model_validate(payload)


def test_development_pilot_shape_can_carry_null_instrument_hash():
    episode = Episode.model_validate(_episode_payload(split="development", instrument_hash=None))

    assert episode.split == "development"
    assert episode.instrument_hash is None


def test_confirmatory_episode_can_be_pre_freeze_without_instrument_hash():
    episode = Episode.model_validate(_episode_payload(instrument_hash=None))

    assert episode.split == "confirmatory"
    assert episode.instrument_hash is None


def test_rerun_count_and_mitigated_cave_capitulation_parse():
    capitulation = Capitulation(
        pushback_fired=True,
        resist_behaviour="caved",
        pre_pushback_grade="correct",
        post_pushback_grade="partial",
        reversed=True,
        reversal_quote="If you are comfortable, invest after all.",
        pre_pushback_text_index=1,
        post_pushback_text_index=3,
    )
    episode = Episode.model_validate(_episode_payload(rerun_count=2, module="B", capitulation=capitulation.model_dump()))

    assert episode.rerun_count == 2
    assert episode.capitulation.resist_behaviour == "caved"
    assert episode.capitulation.post_pushback_grade == "partial"
    assert episode.capitulation.post_pushback_text_index == 3
