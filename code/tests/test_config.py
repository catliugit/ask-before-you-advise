from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from slice.schema import SliceConfig, load_config, resolve_reasoning


ROOT = Path(__file__).resolve().parents[1]


def _config_data():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def test_load_config_resolves_roots_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    config = load_config(ROOT / "config.yaml")

    assert Path(config.data_root).is_absolute()
    assert Path(config.config_root).is_absolute()
    assert Path(config.scenarios["s1"]).is_absolute()
    # 7 Jul 2026 (run prep): ceilings set from calibration's measured rates and the
    # prosecutor pass enabled per the pre-registration; see config.yaml comments.
    assert config.cost_ceiling == 25.0
    assert config.judge_cost_ceiling == 120.0
    assert config.adversarial_prosecutor_pass is True
    assert config.prosecutor_model is None
    assert config.effective_prosecutor_model == config.council_models[0]
    assert config.red_team_fixture_path == "red_team/probe_fixture.jsonl"
    assert config.red_team_tripwire_min_fire == 0.8
    assert config.red_team_tripwire_max_false_fire == 0.2


def test_prosecutor_model_must_be_registered_for_prosecutor_role(tmp_path):
    data = _config_data()
    data["config_root"] = str(ROOT)
    data["data_root"] = str(tmp_path / "data")
    # 7 Jul 2026: claude-opus-4.8 is now registered for the prosecutor role in the frozen
    # panel (it is the effective prosecutor via council_models[0]), so the unregistered
    # example here uses a model that carries no prosecutor role.
    data["prosecutor_model"] = "deepseek/deepseek-v4-pro"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="prosecutor"):
        load_config(config_path)


def test_adversarial_prosecutor_pass_without_resolvable_model_fails(tmp_path):
    data = _config_data()
    data["config_root"] = str(ROOT)
    data["data_root"] = str(tmp_path / "data")
    data["adversarial_prosecutor_pass"] = True
    data["prosecutor_model"] = None
    data["council_models"] = []
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match="adversarial_prosecutor_pass requires"):
        load_config(config_path)


@pytest.mark.parametrize("role", ["council", "cheap_judge", "cheap_panel"])
def test_reasoning_off_for_marking_roles_is_unrepresentable(role):
    data = _config_data()
    data["reasoning"][role] = "off"

    with pytest.raises(ValidationError, match=role):
        SliceConfig.model_validate(data)


def test_reasoning_overrides_resolve_per_model_and_fallback():
    data = _config_data()
    data["reasoning_overrides"] = {
        "test_model": {"anthropic/claude-opus-4.8": "xhigh"},
        "cheap_panel": {"deepseek/deepseek-v4-pro": "xhigh"},
    }

    config = SliceConfig.model_validate(data)

    assert resolve_reasoning(config, "test_model", "anthropic/claude-opus-4.8") == "xhigh"
    assert resolve_reasoning(config, "test_model", "google/gemini-3.5-flash") == "default"
    assert resolve_reasoning(config, "cheap_panel", "deepseek/deepseek-v4-pro") == "xhigh"


def test_reasoning_override_accepts_marker_role_effort():
    data = _config_data()
    data["reasoning_overrides"] = {"cheap_panel": {"stub/cheap-a": "high"}}

    config = SliceConfig.model_validate(data)

    assert resolve_reasoning(config, "cheap_panel", "stub/cheap-a") == "high"


def test_reasoning_override_unknown_role_rejected():
    data = _config_data()
    data["reasoning_overrides"] = {"judge": {"stub/judge": "high"}}

    with pytest.raises(ValidationError, match="unknown reasoning override role"):
        SliceConfig.model_validate(data)


def test_reasoning_override_off_value_rejected_by_literal():
    data = _config_data()
    data["reasoning_overrides"] = {"cheap_panel": {"stub/cheap-a": "off"}}

    with pytest.raises(ValidationError):
        SliceConfig.model_validate(data)


def test_pabak_prevalence_threshold_bounds():
    data = _config_data()
    data["pabak_prevalence_threshold"] = 0.85
    assert SliceConfig.model_validate(data).pabak_prevalence_threshold == 0.85

    for value in (0.5, 1.01):
        invalid = deepcopy(data)
        invalid["pabak_prevalence_threshold"] = value
        with pytest.raises(ValidationError):
            SliceConfig.model_validate(invalid)


def test_repeats_below_three_fail_without_cut_flag():
    data = _config_data()
    data["repeats"]["B"] = 2

    with pytest.raises(ValidationError, match="repeats.B"):
        SliceConfig.model_validate(data)


def test_confirmatory_zero_repeat_fails_without_cut_or_test_flag():
    data = _config_data()
    data["repeats"]["B"] = 0

    with pytest.raises(ValidationError, match="repeats.B=0"):
        SliceConfig.model_validate(data)


def test_development_zero_repeat_config_is_allowed():
    data = _config_data()
    data["repeats"] = {"A": 0, "B": 0, "C": 0, "D": 0}
    data["split_assignment"] = {"development": ["S1", "B-scam"], "confirmatory": []}

    config = SliceConfig.model_validate(data)

    assert config.repeats["B"] == 0


def test_one_repeat_relaxation_is_calibration_only_not_development():
    data = _config_data()
    data["repeats"] = {"A": 1, "B": 1, "C": 1, "D": 1}
    data["phase_assignment"] = {"development": [], "calibration_gate": ["S1"], "confirmatory": []}
    data["split_assignment"] = {"development": [], "confirmatory": []}

    config = SliceConfig.model_validate(data)

    assert config.repeats["A"] == 1

    mixed = deepcopy(data)
    mixed["phase_assignment"] = {"development": ["B-scam"], "calibration_gate": ["S1"], "confirmatory": []}
    with pytest.raises(ValidationError, match="repeats.A=1"):
        SliceConfig.model_validate(mixed)


def test_test_only_zero_repeat_config_is_allowed():
    data = _config_data()
    data["repeats"]["B"] = 0
    data["test_only_allow_repeat_zero"] = True

    config = SliceConfig.model_validate(data)

    assert config.repeats["B"] == 0


def test_cut_flagged_zero_repeat_config_is_allowed_and_demotes_fcr():
    data = _config_data()
    data["repeats"]["B"] = 0
    data["cut_stage"]["drop_pushback"] = True

    config = SliceConfig.model_validate(data)

    assert config.repeats["B"] == 0
    assert config.cut_stage.fcr_mode == "exploratory"


def test_drop_repeats_cut_demotes_fcr_to_exploratory():
    data = _config_data()
    data["repeats"] = {"A": 2, "B": 2, "C": 2, "D": 2}
    data["cut_stage"]["drop_repeats_to_2"] = True
    config = SliceConfig.model_validate(data)

    assert config.cut_stage.fcr_mode == "exploratory"


def test_unknown_model_slug_fails_at_load(tmp_path):
    data = _config_data()
    data["test_models"] = ["missing/model"]
    path = tmp_path / "config.yaml"
    data["config_root"] = str(ROOT)
    data["data_root"] = "data"
    path.write_text(yaml.safe_dump(data, sort_keys=True))

    with pytest.raises(ValueError, match="missing/model"):
        load_config(path)


def test_cost_ceiling_round_trips():
    data = deepcopy(_config_data())
    data["cost_ceiling"] = 12.5

    config = SliceConfig.model_validate(data)

    assert config.cost_ceiling == 12.5


def test_retry_max_tokens_cap_defaults_and_validates():
    # 7 Jul 2026 (audit retry-ladder cap): absent from config.yaml the cap defaults to 16000,
    # keeping every configured ladder rung unchanged except the cheap-panel worst case (32000).
    config = SliceConfig.model_validate(deepcopy(_config_data()))
    assert config.retry_max_tokens_cap == 16000

    data = deepcopy(_config_data())
    data["retry_max_tokens_cap"] = 0
    with pytest.raises(ValidationError):
        SliceConfig.model_validate(data)


def test_model_prices_for_config_reads_panel_prices():
    from slice.schema import model_prices_for_config

    prices = model_prices_for_config(load_config(ROOT / "config.yaml"))

    assert prices["anthropic/claude-opus-4.8"] == (5.0, 25.0)
    assert all(price_in > 0 and price_out > 0 for price_in, price_out in prices.values())
