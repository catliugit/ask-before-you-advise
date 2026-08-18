from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from slice.calibration_gate import (
    build_calibration_verdicts,
    load_calibration_verdicts,
    write_calibration_verdicts,
)
from slice.schema import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_build_calibration_verdicts_trusts_apparent_pass_with_agreeing_reference(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    data_root = tmp_path / "data"
    episode = _episode("S1-calibration-C-r1")
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", [episode])
    _write_jsonl(
        data_root / "judgements.jsonl",
        [
            *_judgements(episode["episode_id"], "cheap_panel", "correct"),
            *_judgements(episode["episode_id"], "council", "correct"),
        ],
    )

    verdicts = build_calibration_verdicts(config_path)
    output_path = write_calibration_verdicts(config_path)
    loaded = load_calibration_verdicts(load_config(config_path))

    assert verdicts["S1"]["verdict"] == "trusted"
    assert verdicts["S1"]["audit_n_apparent_pass"] == 1
    assert verdicts["S1"]["council_items_audited"] == 1
    assert output_path == data_root / "outputs" / "calibration_verdicts.json"
    assert loaded is not None and loaded["S1"]["verdict"] == "trusted"


def test_build_calibration_verdicts_escalates_false_safe_apparent_pass(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    data_root = tmp_path / "data"
    episode = _episode("S1-calibration-C-r2")
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", [episode])
    _write_jsonl(
        data_root / "judgements.jsonl",
        [
            *_judgements(episode["episode_id"], "cheap_panel", "correct"),
            *_judgements(episode["episode_id"], "council", "harmful"),
        ],
    )

    verdicts = build_calibration_verdicts(config_path)

    assert verdicts["S1"]["verdict"] == "escalate_whole_scenario"
    assert verdicts["S1"]["false_safe_errors"] == 1


def test_build_calibration_verdicts_escalates_safety_split_false_safe_apparent_pass(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    data_root = tmp_path / "data"
    episode = _episode("S1-calibration-D-r4", module="D", variant="D-neutral")
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", [episode])
    _write_jsonl(
        data_root / "judgements.jsonl",
        [
            *_judgements(episode["episode_id"], "cheap_panel", "correct"),
            _judgement(episode["episode_id"], "council", "stub/council-a", "correct"),
            _judgement(episode["episode_id"], "council", "stub/council-b", "correct"),
            _judgement(episode["episode_id"], "council", "stub/council-c", "harmful"),
        ],
    )

    verdicts = build_calibration_verdicts(config_path)

    # The unanimous-harmful test covers the obvious miss; this covers the buried harmful dissent.
    assert verdicts["S1"]["verdict"] == "escalate_whole_scenario"
    assert verdicts["S1"]["false_safe_errors"] >= 1


def test_build_calibration_verdicts_harm_flag_marks_non_d_safety_axis_and_no_flag_does_not(tmp_path):
    flagged = _split_calibration_verdict(tmp_path / "flagged", safety_flag="harm")
    unflagged = _split_calibration_verdict(tmp_path / "unflagged", safety_flag="none")

    assert flagged["S1"]["verdict"] == "escalate_whole_scenario"
    assert flagged["S1"]["false_safe_errors"] == 1
    assert unflagged["S1"]["verdict"] == "trusted"
    assert unflagged["S1"]["false_safe_errors"] == 0


def test_build_calibration_verdicts_escalates_zero_reference_sample(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    data_root = tmp_path / "data"
    episode = _episode("S1-calibration-C-r3")
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", [episode])
    _write_jsonl(
        data_root / "judgements.jsonl",
        _judgements(episode["episode_id"], "cheap_panel", "correct"),
    )

    verdicts = build_calibration_verdicts(config_path)

    assert verdicts["S1"]["verdict"] == "escalate_whole_scenario"
    assert verdicts["S1"]["human_items_audited"] == 0
    assert verdicts["S1"]["council_items_audited"] == 0


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data",
                "config_root": str(ROOT),
                "model_panel_path": "model_panel.yaml",
                "scenario_paths": {"S1": "scenarios/s1.json"},
                "test_models": ["stub/test"],
                "persona_model": "stub/persona",
                "council_models": ["stub/council-a", "stub/council-b", "stub/council-c"],
                "cheap_panel_models": ["stub/cheap-a", "stub/cheap-b", "stub/cheap-c"],
                "repeats": {"A": 0, "B": 0, "C": 0, "D": 0},
                "turn_cap": 6,
                "max_concurrency": 1,
                "prompt_versions": {
                    "persona": "persona-week1-v3",
                    "judge_pass1": "judge-pass1-general-v2",
                    "judge_pass2": "judge-pass2-general-v3",
                },
                "test_only_allow_repeat_zero": True,
                "split_assignment": {"development": ["S1"], "confirmatory": []},
            },
            sort_keys=True,
        )
    )
    return path


def _episode(episode_id: str, *, module: str = "C", variant: str = "C-control") -> dict:
    return {
        "episode_id": episode_id,
        "scenario": "S1",
        "module": module,
        "variant": variant,
        "split": "development",
        "phase": "calibration_gate",
        "call_status": "ok",
        "instrument_hash": None,
    }


def _judgements(episode_id: str, tier: str, outcome: str) -> list[dict]:
    return [
        _judgement(episode_id, tier, f"stub/{tier}-{index}", outcome)
        for index in range(3)
    ]


def _judgement(
    episode_id: str,
    tier: str,
    model: str,
    outcome: str,
    *,
    safety_flag: str = "none",
) -> dict:
    return {
        "episode_id": episode_id,
        "judge_tier": tier,
        "judge_model": model,
        "pass2": {"outcome_grade": outcome, "safety_flag": safety_flag},
    }


def _split_calibration_verdict(root: Path, *, safety_flag: str) -> dict[str, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    config_path = _write_config(root / "config.yaml")
    data_root = root / "data"
    episode = _episode(f"S1-calibration-C-split-{safety_flag}")
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", [episode])
    _write_jsonl(
        data_root / "judgements.jsonl",
        [
            *[
                _judgement(
                    episode["episode_id"],
                    "cheap_panel",
                    f"stub/cheap-{index}",
                    "correct",
                    safety_flag=safety_flag if index == 0 else "none",
                )
                for index in range(3)
            ],
            _judgement(episode["episode_id"], "council", "stub/council-a", "correct"),
            _judgement(episode["episode_id"], "council", "stub/council-b", "correct"),
            _judgement(episode["episode_id"], "council", "stub/council-c", "incorrect"),
        ],
    )
    return build_calibration_verdicts(config_path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
