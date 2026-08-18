from __future__ import annotations

import glob as glob_module
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from slice.calibration_gate import load_calibration_verdicts
from slice import freeze
from slice.cli import main as cli_main
from slice.metrics import _load_gate_verdict
from slice.schema import FROZEN_HASH_INPUTS, load_config, load_prompt_file

CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent


def test_hash_is_deterministic_sorted_and_cwd_independent(tmp_path, monkeypatch):
    _, config_path = _copy_freeze_tree(tmp_path)
    first_hash, first_entries = freeze.compute_instrument_hash(config_path)
    second_hash, second_entries = freeze.compute_instrument_hash(config_path)

    monkeypatch.chdir(tmp_path)
    third_hash, _ = freeze.compute_instrument_hash(config_path)

    assert first_hash == second_hash == third_hash
    assert len(first_hash) == 64
    assert first_hash == first_hash.lower()
    paths = [entry.path for entry in first_entries]
    assert paths == sorted(paths, key=lambda value: value.encode("utf-8"))
    assert [entry.to_dict() for entry in first_entries] == [entry.to_dict() for entry in second_entries]


def test_frozen_hash_inputs_cover_required_current_files(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    _, entries = freeze.compute_instrument_hash(config_path)
    paths = {entry.path for entry in entries}
    expected_protocol = {
        (path.relative_to(repo)).as_posix()
        for path in sorted((repo / "code/src/slice").glob("*.py"))
        if path.name != "cli.py"
    }
    expected_analysis = {
        (path.relative_to(repo)).as_posix()
        for path in sorted((repo / "code/src/slice/analysis").glob("*.py"))
    }

    assert len(FROZEN_HASH_INPUTS) == len(set(FROZEN_HASH_INPUTS))
    assert len(paths) == len(entries)
    assert "code/config.yaml" in paths
    assert "code/scenarios/s1.json" in paths
    assert "code/scenarios/boundary_scam.json" in paths
    assert "code/prompts/judge_pass2_boundary.md" in paths
    assert "code/prompts/judge_resist_initial.md" in paths
    assert "code/prompts/council_coder.md" in paths
    assert "code/prompts/prosecutor_pass.md" in FROZEN_HASH_INPUTS
    assert "code/prompts/prosecutor_pass.md" in paths
    assert expected_protocol <= paths
    assert expected_analysis <= paths
    assert "code/src/slice/freeze.py" in paths
    assert "code/src/slice/cli.py" not in paths


@pytest.mark.parametrize(
    "relative_path",
    [
        "code/scenarios/s1.json",
        "code/prompts/judge_pass1.md",
        "code/prompts/judge_resist_initial.md",
        "code/prompts/prosecutor_pass.md",
        "grading-codebook.md",
        "decision-rules.md",
        "severity-rubric.md",
        "code/config.yaml",
        "code/model_panel.yaml",
        "pre-registration.md",
        "code/src/slice/metrics.py",
        "code/src/slice/etl.py",
        "code/src/slice/kappa.py",
        "code/src/slice/kappa_gate.py",
        "code/src/slice/freeze.py",
        "code/src/slice/gate.py",
        "code/src/slice/judge.py",
        "code/src/slice/canary.py",
        "code/compute_kappa.py",
        "code/src/slice/analysis/fcr.py",
        "code/src/slice/analysis/inference.py",
    ],
)
def test_one_byte_change_to_each_hashed_group_changes_hash(tmp_path, relative_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    baseline, _ = freeze.compute_instrument_hash(config_path)
    path = repo / relative_path
    original = path.read_bytes()

    path.write_bytes(original + b" ")
    try:
        changed, _ = freeze.compute_instrument_hash(config_path)
    finally:
        path.write_bytes(original)

    assert changed != baseline


def test_glob_expansion_is_sorted(monkeypatch, tmp_path):
    _, config_path = _copy_freeze_tree(tmp_path)
    baseline, _ = freeze.compute_instrument_hash(config_path)
    real_glob = freeze.glob

    def reversed_glob(pattern: str) -> list[str]:
        return list(reversed(real_glob(pattern)))

    monkeypatch.setattr(freeze, "glob", reversed_glob)
    assert freeze.compute_instrument_hash(config_path)[0] == baseline


def test_missing_declared_file_fails_loudly(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    (repo / "code/prompts/persona.md").unlink()

    with pytest.raises(freeze.MissingFrozenInputError, match="code/prompts/persona.md"):
        freeze.compute_instrument_hash(config_path)


def test_prosecutor_prompt_exists_and_parses_as_frozen_input():
    path = CODE_ROOT / "prompts" / "prosecutor_pass.md"

    prompt = load_prompt_file(path)

    assert path.exists()
    assert prompt.version == "prosecutor-pass-v1"
    assert "tripwire" in prompt.text


def test_line_endings_are_normalised_but_semantic_byte_changes_are_not(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    path = repo / "grading-codebook.md"
    original = path.read_bytes()
    baseline, _ = freeze.compute_instrument_hash(config_path)

    path.write_bytes(original.replace(b"\n", b"\r\n"))
    assert freeze.compute_instrument_hash(config_path)[0] == baseline

    path.write_bytes(original.replace(b"correct", b"corect", 1))
    assert freeze.compute_instrument_hash(config_path)[0] != baseline


def test_freeze_record_round_trips_and_verify_detects_drift_and_deviation(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    record = freeze.write_freeze_record(
        config_path,
        date_stamp="2026-06-18T09:00:00Z",
        external_timestamp={
            "method": "github-commit",
            "reference": "https://github.example/freeze-commit",
            "verified_instructions": "operator-provided test pointer",
        },
        enforce_preflight=False,
    )
    record_path = repo / "freeze_record.json"
    reloaded = json.loads(record_path.read_text())
    recomputed, entries = freeze.compute_instrument_hash(config_path)

    assert reloaded["instrument_hash"] == record["instrument_hash"] == recomputed
    assert reloaded["date_stamp"] == "2026-06-18T09:00:00Z"
    assert reloaded["freeze_timestamp"] == "2026-06-18T09:00:00Z"
    assert reloaded["files"] == [entry.to_dict() for entry in entries]
    assert reloaded["model_panel_snapshot"]["leaderboard_source"]
    assert reloaded["external_timestamp"]["reference"] == "https://github.example/freeze-commit"
    assert reloaded["deviations"] == []
    assert freeze.verify(config_path, enforce_preflight=False).ok is True

    pre_reg = repo / "pre-registration.md"
    pre_reg.write_bytes(pre_reg.read_bytes() + b" ")
    drift = freeze.verify(config_path, enforce_preflight=False)
    assert drift.ok is False
    assert drift.status == "MISMATCH"
    assert "pre-registration.md" in drift.drifted_files

    with pytest.raises(freeze.FreezeError, match="not a frozen input"):
        freeze.append_deviation(
            config_path,
            file="README.md",
            what_changed="bad deviation",
            why="not in the frozen inventory",
            date_stamp="2026-06-18T09:30:00Z",
        )
    with pytest.raises(freeze.FreezeError, match="has not drifted"):
        freeze.append_deviation(
            config_path,
            file="grading-codebook.md",
            what_changed="bad deviation",
            why="names the wrong frozen file",
            date_stamp="2026-06-18T09:45:00Z",
        )

    new_hash = freeze.append_deviation(
        config_path,
        file="pre-registration.md",
        what_changed="test mutation",
        why="exercise deviation chain",
        date_stamp="2026-06-18T10:00:00Z",
    )
    after_deviation = json.loads(record_path.read_text())
    assert after_deviation["instrument_hash"] == record["instrument_hash"]
    assert after_deviation["deviations"][-1]["new_instrument_hash"] == new_hash
    assert freeze.verify(config_path, enforce_preflight=False).ok is True

    codebook = repo / "grading-codebook.md"
    codebook.write_bytes(codebook.read_bytes() + b" ")
    second_drift = freeze.verify(config_path, enforce_preflight=False)
    assert second_drift.ok is False
    assert second_drift.drifted_files == ("grading-codebook.md",)


def test_confirmatory_hash_stamping_and_gate_boundary(tmp_path):
    _, config_path = _copy_freeze_tree(tmp_path)
    record = freeze.write_freeze_record(config_path, date_stamp="2026-06-18T09:00:00Z", enforce_preflight=False)
    expected = record["instrument_hash"]

    confirmatory = {"episode_id": "ok", "split": "confirmatory", "instrument_hash": None}
    development = {"episode_id": "dev", "split": "development", "instrument_hash": "stale"}
    assert freeze.stamp_instrument_hash(confirmatory, config_path, enforce_preflight=False)["instrument_hash"] == expected
    assert freeze.stamp_instrument_hash(development, config_path, enforce_preflight=False)["instrument_hash"] is None
    freeze.assert_frozen_confirmatory_episodes(
        [
            {"episode_id": "ok", "split": "confirmatory", "instrument_hash": expected},
            {"episode_id": "dev", "split": "development", "instrument_hash": None},
        ],
        expected,
    )

    with pytest.raises(freeze.FrozenRunHashError, match="missing or mismatched"):
        freeze.assert_frozen_confirmatory_episodes(
            [{"episode_id": "bad-null", "split": "confirmatory", "instrument_hash": None}],
            expected,
        )
    with pytest.raises(freeze.FrozenRunHashError, match="bad-mismatch"):
        freeze.assert_frozen_confirmatory_episodes(
            [{"episode_id": "bad-mismatch", "split": "confirmatory", "instrument_hash": "other"}],
            expected,
        )


def test_confirmatory_stamp_requires_freeze_record(tmp_path):
    _, config_path = _copy_freeze_tree(tmp_path)

    with pytest.raises(freeze.FrozenRunHashError, match="no freeze_record.json"):
        freeze.stamp_instrument_hash({"episode_id": "e1", "split": "confirmatory"}, config_path, enforce_preflight=False)


def test_freeze_cli_writes_and_verifies_record(tmp_path, capsys):
    repo, config_path = _copy_freeze_tree(tmp_path)
    _make_freeze_ready(repo, config_path)

    cli_main(["freeze", "--config", str(config_path), "--date-stamp", "2026-06-18T09:00:00Z"])
    assert (repo / "freeze_record.json").exists()

    cli_main(["freeze", "--config", str(config_path), "--verify"])
    captured = capsys.readouterr()
    assert "instrument_hash" in captured.out
    assert "'ok': True" in captured.out

    result = subprocess.run(
        [sys.executable, str(CODE_ROOT / "freeze.py"), "--config", str(config_path), "--verify"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert '"ok": true' in result.stdout


def test_freeze_cli_refuses_preflight_violations(tmp_path):
    _, config_path = _copy_freeze_tree(tmp_path)

    with pytest.raises(freeze.FreezeError, match="freeze preflight failed"):
        cli_main(["freeze", "--config", str(config_path), "--date-stamp", "2026-06-18T09:00:00Z"])


def test_build_calibration_distillate_bytes_ignore_input_timestamps():
    gate = _gate_decisions("PASS")
    calibration = {
        "S1": {
            "verdict": "trusted",
            "audit_n_apparent_pass": 3,
            "audit_n_non_pass": 1,
            "false_safe_errors": 0,
            "routine_disagree_pct": 0.0,
            "run_timestamp": "2026-06-18T09:00:00Z",
        }
    }
    changed_gate = {**gate, "computed_at": "2026-06-18T10:00:00Z"}
    changed_calibration = {
        "S1": {
            **calibration["S1"],
            "run_timestamp": "2026-06-18T10:00:00Z",
        }
    }

    first = _canonical_distillate_bytes(freeze.build_calibration_distillate(gate, calibration))
    second = _canonical_distillate_bytes(freeze.build_calibration_distillate(changed_gate, changed_calibration))

    assert first == second
    assert b"2026-06-18T09:00:00Z" not in first
    assert b"2026-06-18T10:00:00Z" not in first


def test_validate_calibration_distillate_rejects_malformed_shape(tmp_path):
    _, config_path = _copy_freeze_tree(tmp_path)

    violations = freeze.validate_calibration_distillate({"S1": {}}, load_config(config_path))
    codes = {violation["code"] for violation in violations}

    assert "calibration_frozen_schema" in codes
    assert "calibration_frozen_permission" in codes
    assert "calibration_frozen_gate" in codes


def _zero_count_permission() -> dict[str, object]:
    return {
        "scenario_id": "stub",
        "verdict": "escalate_whole_scenario",
        "audit_n_apparent_pass": 0,
        "audit_n_non_pass": 0,
        "false_safe_errors": 0,
        "routine_disagree_pct": 0.0,
        "human_items_audited": 0,
        "council_items_audited": 0,
    }


def test_validate_calibration_distillate_rejects_all_zero_audit_count_stub(tmp_path):
    # 7 Jul 2026 audit guard: the 2 July placeholder calibration_frozen.json carried all-zero
    # audit counts in every permission. A freeze must never take such a stub for the real
    # calibration distillate.
    _, config_path = _copy_freeze_tree(tmp_path)
    config = load_config(config_path)
    stub = freeze.build_calibration_distillate(
        _gate_decisions("INSUFFICIENT_N"),
        {
            scenario_id: _zero_count_permission()
            for scenario_id in config.effective_phase_assignment.confirmatory
        },
    )

    violations = freeze.validate_calibration_distillate(stub, config)
    codes = {violation["code"] for violation in violations}

    assert "calibration_frozen_stub" in codes
    assert codes == {"calibration_frozen_stub"}


def test_validate_calibration_distillate_accepts_mixed_real_and_zero_count_permissions(tmp_path):
    # A single zero-count scenario beside real audited counts is legitimate (e.g. a scenario
    # added to the confirmatory set after calibration); the stub guard fires only when EVERY
    # permission is all-zero.
    _, config_path = _copy_freeze_tree(tmp_path)
    config = load_config(config_path)
    verdicts: dict[str, dict[str, object]] = {}
    confirmatory = list(config.effective_phase_assignment.confirmatory)
    for scenario_id in confirmatory[:-1]:
        verdicts[scenario_id] = {
            **_zero_count_permission(),
            "scenario_id": scenario_id,
            "verdict": "trusted",
            "audit_n_apparent_pass": 60,
            "audit_n_non_pass": 30,
            "council_items_audited": 88,
        }
    verdicts[confirmatory[-1]] = {**_zero_count_permission(), "scenario_id": confirmatory[-1]}
    distillate = freeze.build_calibration_distillate(_gate_decisions("PASS"), verdicts)

    violations = freeze.validate_calibration_distillate(distillate, config)

    assert violations == []


def test_repo_calibration_frozen_is_the_real_distillate_not_a_stub():
    # Regression pin (7 Jul 2026): code/calibration_frozen.json is the regenerated 6 July
    # calibration distillate. It must validate cleanly and carry real audit counts.
    distillate = json.loads((CODE_ROOT / "calibration_frozen.json").read_text())

    violations = freeze.validate_calibration_distillate(distillate, load_config(CODE_ROOT / "config.yaml"))

    assert violations == []
    permissions = distillate["calibration_permissions"]
    assert permissions["B-pension-transfer"]["verdict"] == "escalate_whole_scenario"
    assert permissions["B-scam"]["verdict"] == "escalate_whole_scenario"
    assert any(
        record.get("audit_n_apparent_pass") or record.get("audit_n_non_pass")
        for record in permissions.values()
    )


def test_frozen_routing_uses_distillate_and_prefreeze_uses_live_sidecars(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    config = load_config(config_path)
    data_root = Path(config.data_root)
    (data_root / "outputs").mkdir(parents=True)
    (data_root / "outputs" / "calibration_verdicts.json").write_text(
        json.dumps({"S1": {"scenario_id": "S1", "verdict": "trusted"}}, sort_keys=True)
    )
    (data_root / "outputs" / "gate_verdict.json").write_text(
        json.dumps(_gate_decisions("DEMOTE_TO_ESTIMATION"), sort_keys=True)
    )
    frozen = freeze.build_calibration_distillate(
        _gate_decisions("PASS"),
        {"S1": {"scenario_id": "S1", "verdict": "escalate_whole_scenario"}},
    )
    (repo / "code" / "calibration_frozen.json").write_text(json.dumps(frozen, sort_keys=True))

    assert load_calibration_verdicts(config)["S1"]["verdict"] == "trusted"
    assert _load_gate_verdict(config, data_root)["per_module"]["A"]["verdict"] == "DEMOTE_TO_ESTIMATION"

    _make_freeze_ready(repo, config_path)
    frozen_config = load_config(config_path)
    assert load_calibration_verdicts(frozen_config)["S1"]["verdict"] == "escalate_whole_scenario"
    assert _load_gate_verdict(frozen_config, data_root)["per_module"]["A"]["verdict"] == "PASS"


def test_freeze_preflight_enumerates_all_violation_classes(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["prompt_versions"]["persona"] = "persona-config-mismatch"
    config["red_team_n"] = 999
    # 7 Jul 2026: the live config now carries real ceilings (run prep), so this test
    # must null them itself to keep exercising the twin-gate violation class.
    config["cost_ceiling"] = None
    config["judge_cost_ceiling"] = None
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    (repo / "grading-codebook.md").unlink()
    (repo / "code" / "calibration_frozen.json").write_text(json.dumps({"S1": {}}, sort_keys=True))

    preflight = freeze.freeze_preflight(config_path)
    codes = {violation["code"] for violation in preflight["violations"]}

    assert preflight["ok"] is False
    assert {
        "missing_frozen_input",
        "model_panel_freeze_day",
        "model_panel_pinned_version",
        "cost_ceiling",
        "prompt_version_mismatch",
        "red_team_n_mismatch",
        "calibration_frozen_schema",
        "calibration_frozen_permission",
        "calibration_frozen_gate",
    } <= codes


@pytest.mark.parametrize("value", [None, 0, -1.0, True])
def test_freeze_preflight_requires_positive_judge_cost_ceiling(tmp_path, value):
    repo, config_path = _copy_freeze_tree(tmp_path)
    _make_freeze_ready(repo, config_path)
    config = _config_copy(load_config(config_path), judge_cost_ceiling=value)

    preflight = freeze.freeze_preflight(config)
    codes = {violation["code"] for violation in preflight["violations"]}

    assert "judge_cost_ceiling" in codes


def test_freeze_preflight_accepts_positive_judge_cost_ceiling(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    _make_freeze_ready(repo, config_path)

    preflight = freeze.freeze_preflight(config_path)
    codes = {violation["code"] for violation in preflight["violations"]}

    assert "judge_cost_ceiling" not in codes


def test_freeze_preflight_reports_red_team_n_fixture_count_mismatch(tmp_path):
    repo, config_path = _copy_freeze_tree(tmp_path)
    _make_freeze_ready(repo, config_path)
    config = yaml.safe_load(config_path.read_text())
    config["red_team_n"] = 29
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))

    preflight = freeze.freeze_preflight(config_path)
    mismatch = [violation for violation in preflight["violations"] if violation["code"] == "red_team_n_mismatch"]

    assert len(mismatch) == 1
    assert "fixture total item count is 28" in mismatch[0]["message"]
    assert mismatch[0]["details"] == {"n_probes": 14, "n_safe": 14}


def _copy_freeze_tree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    for source in _source_frozen_inputs():
        relative = source.relative_to(REPO_ROOT)
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    config_target = repo / "code/config.yaml"
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CODE_ROOT / "config.yaml", config_target)
    # 7 Jul 2026 (freeze day): the repo model_panel.yaml now carries freeze_day + pins for the
    # confirmatory run. The copied tree must start PRE-freeze — tests that need the frozen
    # state freeze it explicitly via _make_freeze_ready (same pattern as the ceilings note in
    # test_freeze_preflight_enumerates_all_violation_classes).
    panel_path = repo / "code/model_panel.yaml"
    panel = yaml.safe_load(panel_path.read_text())
    panel["freeze_day"] = None
    for entry in panel.get("entries", []):
        entry["pinned_version"] = None
    panel_path.write_text(yaml.safe_dump(panel, sort_keys=True))
    return repo, config_target


def _make_freeze_ready(repo: Path, config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    config["cost_ceiling"] = 1.0
    config["judge_cost_ceiling"] = 1.0
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))

    panel_path = repo / "code/model_panel.yaml"
    panel = yaml.safe_load(panel_path.read_text())
    panel["freeze_day"] = "2026-06-18"
    for entry in panel.get("entries", []):
        entry["pinned_version"] = f"{entry.get('slug', 'model')}@2026-06-18"
    panel_path.write_text(yaml.safe_dump(panel, sort_keys=True))

    calibration_path = repo / "code" / "calibration_frozen.json"
    calibration = json.loads(calibration_path.read_text())
    permissions = calibration.setdefault("calibration_permissions", {})
    for scenario_id in config.get("split_assignment", {}).get("confirmatory", []):
        permissions.setdefault(
            scenario_id,
            {
                "audit_n_apparent_pass": 0,
                "audit_n_non_pass": 0,
                "cheap_grading_permitted": False,
                "council_items_audited": 0,
                "false_safe_errors": 0,
                "human_items_audited": 0,
                "routine_disagree_pct": 0.0,
                "verdict": "escalate_whole_scenario",
            },
        )
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")


def _config_copy(config, **updates):
    if hasattr(config, "model_copy"):
        return config.model_copy(update=updates)
    return config.copy(update=updates)


def _gate_decisions(verdict: str) -> dict[str, object]:
    return {
        "computed_at": "2026-06-18T09:00:00Z",
        "per_module": {module: {"verdict": verdict, "n": 10} for module in ["A", "B", "C", "D"]},
        "council_vs_human": {module: {"verdict": verdict, "n": 10} for module in ["A", "B", "C", "D"]},
        "false_clear": {
            "boundary": {
                "binding": True,
                "below_n_floor": False,
                "false_clear_count": 0,
                "n_compared": 10,
                "n_human_dangerous": 10,
            },
            "boundary_safety_verdict": "PASS",
            "safety_set_widened_required": False,
        },
        "boundary_safety_verdict": "PASS",
        "safety_set_widened_required": False,
    }


def _canonical_distillate_bytes(distillate: dict[str, object]) -> bytes:
    return (json.dumps(distillate, indent=2, sort_keys=True) + "\n").encode()


def _source_frozen_inputs() -> list[Path]:
    paths: list[Path] = []
    for declared in FROZEN_HASH_INPUTS:
        pattern = REPO_ROOT / declared
        if any(char in declared for char in "*?["):
            matches = sorted(glob_module.glob(str(pattern)))
            assert matches, declared
            paths.extend(Path(match) for match in matches)
        else:
            path = pattern
            assert path.exists(), declared
            paths.append(path)
    return sorted({path.resolve() for path in paths}, key=lambda value: value.relative_to(REPO_ROOT).as_posix())
