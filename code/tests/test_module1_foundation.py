from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from slice.gate import cheap_panel_must_escalate, safety_critical_must_escalate
from slice.phase_roles import is_calibration_gate_record, is_rule_fitting_record, is_safety_critical_record
from slice.schema import (
    CalibrationGateRecord,
    CheapPanelAgreement,
    Episode,
    H0HumanLabel,
    H1AdjudicatedLabel,
    HumanValidationStats,
    Judgement,
    RedTeamProbeRecord,
    load_config,
    load_model_panel,
)


ROOT = Path(__file__).resolve().parents[1]


def _config_data() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return path


def _episode_payload(**overrides) -> dict:
    payload = {
        "episode_id": "S1-A-A1-stub__test-r0",
        "run_id": "run-20260617-abc123",
        "split": "confirmatory",
        "phase": "confirmatory",
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
        "call_status": "ok",
    }
    payload.update(overrides)
    return payload


def _judgement_payload(**overrides) -> dict:
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
        "scoring_failed": False,
        "raw": {"pass1": "{}", "pass2": "{}"},
    }
    payload.update(overrides)
    return payload


def _assert_round_trips(model_cls, payload: dict) -> dict:
    model = model_cls.model_validate(payload)
    dumped = model.model_dump()
    assert model_cls.model_validate(dumped).model_dump() == dumped
    return dumped


def test_real_config_loads_with_three_family_cheap_panel():
    config = load_config(ROOT / "config.yaml")
    panel = load_model_panel(ROOT / "model_panel.yaml")

    families = {panel.entry_for_role(slug, "cheap_panel").family for slug in config.cheap_panel_models}

    # 7 Jul 2026 (run prep): the panel is now the calibrated trio from the 6 Jul
    # qualification (re-confirmed post-repair); gpt-5.4-mini passed but is benched.
    assert config.cheap_panel_models == [
        "deepseek/deepseek-v4-pro",
        "google/gemini-3-flash-preview",
        "minimax/minimax-m3",
    ]
    assert config.cheap_panel_size == 3
    assert len(families) == 3


def test_cheap_panel_rejects_too_few_distinct_families(tmp_path):
    data = _config_data()
    data["cheap_panel_models"] = ["stub/cheap-a", "stub/cheap-dup-a", "stub/cheap-b"]
    data["config_root"] = str(ROOT)

    with pytest.raises(ValueError, match="distinct families"):
        load_config(_write_config(tmp_path, data))


def test_cheap_panel_rejects_single_entry(tmp_path):
    data = _config_data()
    data["cheap_panel_models"] = ["stub/cheap-a"]
    data["config_root"] = str(ROOT)

    with pytest.raises(ValueError, match="cheap_panel"):
        load_config(_write_config(tmp_path, data))


def test_cheap_panel_rejects_duplicate_slugs(tmp_path):
    data = _config_data()
    data["cheap_panel_models"] = ["stub/cheap-a", "stub/cheap-a", "stub/cheap-b"]
    data["config_root"] = str(ROOT)

    with pytest.raises(ValueError, match="duplicate"):
        load_config(_write_config(tmp_path, data))


def test_empty_cheap_panel_default_is_accepted(tmp_path):
    data = _config_data()
    data.pop("cheap_panel_models", None)
    data["config_root"] = str(ROOT)

    config = load_config(_write_config(tmp_path, data))

    assert config.cheap_panel_models == []
    assert config.cheap_panel_size == 0


def test_cheap_panel_judgement_requires_confidence():
    with pytest.raises(ValidationError, match="cheap_panel.*confidence"):
        Judgement.model_validate(_judgement_payload(judge_tier="cheap_panel"))

    judgement = Judgement.model_validate(_judgement_payload(judge_tier="cheap_panel", confidence=0.9))

    assert judgement.judge_tier == "cheap_panel"
    assert judgement.confidence == 0.9


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_judgement_confidence_must_be_between_zero_and_one(confidence):
    with pytest.raises(ValidationError, match="confidence"):
        Judgement.model_validate(_judgement_payload(judge_tier="cheap_panel", confidence=confidence))


def test_judgement_defaults_and_legacy_cheap_tier_still_validate():
    legacy = Judgement.model_validate(_judgement_payload())
    revised = Judgement.model_validate(_judgement_payload(pre_deliberation=False))

    assert legacy.judge_tier == "cheap"
    assert legacy.confidence is None
    assert legacy.pre_deliberation is True
    assert revised.pre_deliberation is False


def test_calibration_gate_record_round_trips_for_both_verdicts():
    for verdict in ["trusted", "escalate_whole_scenario"]:
        dumped = _assert_round_trips(
            CalibrationGateRecord,
            {
                "scenario_id": "S1",
                "run_timestamp": "2026-06-17T12:00:00Z",
                "audit_n_apparent_pass": 60,
                "audit_n_non_pass": 30,
                "false_safe_errors": 0,
                "routine_disagree_pct": 0.05,
                "verdict": verdict,
                "human_items_audited": 10,
                "council_items_audited": 90,
            },
        )
        assert dumped["verdict"] == verdict


def test_h1_adjudicated_label_requires_ai_reveal_after_revision():
    payload = {
        "episode_id": "e1",
        "coder_id": "coder-a",
        "revised_at": "2026-06-17T12:00:00Z",
        "outcome_grade": "partial",
        "ai_grade_revealed_at": "2026-06-17T12:00:00Z",
    }
    with pytest.raises(ValidationError, match="ai_grade_revealed_at"):
        H1AdjudicatedLabel.model_validate(payload)

    label = H1AdjudicatedLabel.model_validate(
        {
            **payload,
            "ai_grade_revealed_at": "2026-06-17T12:00:01Z",
            "flip_direction": "toward_ai",
        }
    )

    assert label.flip_direction == "toward_ai"


def test_human_label_models_accept_resist_fields_with_safe_defaults():
    h0_default = H0HumanLabel.model_validate(
        {
            "episode_id": "e1",
            "coder_id": "coder-a",
            "coded_at": "2026-06-17T12:00:00Z",
            "outcome_grade": "correct",
        }
    )
    h1_default = H1AdjudicatedLabel.model_validate(
        {
            "episode_id": "e1",
            "coder_id": "coder-a",
            "revised_at": "2026-06-17T12:00:00Z",
            "outcome_grade": "partial",
        }
    )
    h0_resist = H0HumanLabel.model_validate(
        {
            "episode_id": "e2",
            "coder_id": "coder-a",
            "coded_at": "2026-06-17T12:00:00Z",
            "outcome_grade": "incorrect",
            "resist_initial": "accepted_unsafe_course",
            "resist_behaviour": "caved",
        }
    )
    h1_resist = H1AdjudicatedLabel.model_validate(
        {
            "episode_id": "e2",
            "coder_id": "coder-b",
            "revised_at": "2026-06-17T12:00:00Z",
            "outcome_grade": "incorrect",
            "resist_initial": "resisted",
            "resist_behaviour": "held_firm",
        }
    )

    assert h0_default.resist_initial is None
    assert h0_default.resist_behaviour is None
    assert h1_default.resist_initial is None
    assert h1_default.resist_behaviour is None
    assert h0_resist.resist_initial == "accepted_unsafe_course"
    assert h0_resist.resist_behaviour == "caved"
    assert h1_resist.resist_initial == "resisted"
    assert h1_resist.resist_behaviour == "held_firm"


def test_new_human_and_panel_carrier_models_round_trip():
    _assert_round_trips(
        H0HumanLabel,
        {
            "episode_id": "e1",
            "coder_id": "coder-a",
            "coded_at": "2026-06-17T12:00:00Z",
            "outcome_grade": "correct",
        },
    )
    _assert_round_trips(
        CheapPanelAgreement,
        {
            "episode_id": "e1",
            "cheap_judgement_ids": ["j1", "j2", "j3"],
            "verdict_agree": True,
            "signposting_agree": True,
            "refusal_score_agree": True,
            "decisive_fact_agree": True,
            "mean_confidence": 0.92,
            "any_harm_flagged": False,
            "any_quote_missing": False,
            "escalated": False,
            "finalised_by": "cheap_panel",
        },
    )
    _assert_round_trips(HumanValidationStats, {})
    _assert_round_trips(
        RedTeamProbeRecord,
        {
            "run_timestamp": "2026-06-17T12:00:00Z",
            "n_probes": 30,
            "n_cleared_by_cheap": 1,
            "cheap_pass_rate": 0.03,
        },
    )
    _assert_round_trips(
        RedTeamProbeRecord,
        {
            "run_timestamp": "2026-06-17T12:00:00Z",
            "n_probes": 30,
            "n_cleared_by_cheap": 1,
            "cheap_pass_rate": 0.03,
            "cheap_pass_rate_ucb": 0.12,
            "n_safe_probes": 10,
            "n_over_flagged": 2,
            "over_flag_rate": 0.2,
            "over_flag_rate_ucb": 0.45,
            "n_prosecutor_harmful": 30,
            "tripwire_fire_rate_harmful": 0.9,
            "n_prosecutor_safe": 10,
            "tripwire_fire_rate_safe": 0.1,
            "tripwire_discriminates": True,
        },
    )


def test_calibration_gate_phase_predicate():
    assert is_calibration_gate_record({"phase": "calibration_gate"}) is True
    assert is_calibration_gate_record({"calibration_gate": True}) is True
    assert is_calibration_gate_record({"phase": "confirmatory"}) is False
    assert is_calibration_gate_record({"phase": "development"}) is False


def test_calibration_gate_flag_overrides_development_phase_for_role_predicates():
    assert is_calibration_gate_record({"phase": "development", "calibration_gate": True}) is True
    assert is_rule_fitting_record({"phase": "development", "calibration_gate": True}) is False
    assert is_calibration_gate_record({"phase": "calibration_gate"}) is True
    assert is_rule_fitting_record({"phase": "development"}) is True


def test_safety_critical_predicate_uses_module_d_and_harm_flag():
    assert is_safety_critical_record({"module": "D"}) is True
    assert is_safety_critical_record({"module": "D", "variant": "D-urgent"}) is True
    assert is_safety_critical_record({"module": "A", "any_harm_flagged": True}) is True
    assert is_safety_critical_record({"module": "A", "variant": "A1"}) is False
    assert is_safety_critical_record({"module": "B", "variant": "B-neutral"}) is False


def test_gate_escalation_helpers():
    assert safety_critical_must_escalate({"module": "D"}) is True
    assert safety_critical_must_escalate({"module": "A"}) is False
    assert cheap_panel_must_escalate({"escalated": True, "escalation_triggers": []}) is True
    assert cheap_panel_must_escalate({"escalated": False, "escalation_triggers": ["low_confidence"]}) is True
    assert cheap_panel_must_escalate({"escalated": False, "escalation_triggers": []}) is False


def test_literal_migration_shim_accepts_remaining_legacy_values_and_rejects_old_phase():
    cheap_panel = Judgement.model_validate(_judgement_payload(judge_tier="cheap_panel", confidence=0.9))
    cheap = Judgement.model_validate(_judgement_payload(judge_tier="cheap"))
    calibration_gate = Episode.model_validate(_episode_payload(phase="calibration_gate"))

    assert cheap_panel.judge_tier == "cheap_panel"
    assert cheap.judge_tier == "cheap"
    assert calibration_gate.phase == "calibration_gate"
    with pytest.raises(ValidationError):
        Episode.model_validate(_episode_payload(phase="_".join(["marker", "validation"]), calibration_gate=True))
