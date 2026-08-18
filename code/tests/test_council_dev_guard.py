from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from slice.council import build_council_cases


ROOT = Path(__file__).resolve().parents[1]


def test_council_case_builder_rejects_calibration_gate_rows_with_episode_id(tmp_path):
    config_path = _write_config(tmp_path / "config.yaml")
    data_root = tmp_path / "data"
    episode = {
        "episode_id": "S1-C-C-control-marker",
        "split": "development",
        "calibration_gate": True,
        "scenario": "S1",
        "module": "C",
        "variant": "C-control",
        "transcript": [{"speaker": "user", "text": "Prompt"}],
    }
    _write_jsonl(data_root / "episodes" / "episodes.jsonl", [episode])
    _write_jsonl(data_root / "handcoding" / "council_transcripts.jsonl", [episode])

    with pytest.raises(ValueError, match="episode_id=S1-C-C-control-marker"):
        build_council_cases(config_path)


def _write_config(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data",
                "config_root": str(ROOT),
                "model_panel_path": "model_panel.yaml",
                "scenario_paths": {"S1": "scenarios/s1.json", "B-scam": "scenarios/boundary_scam.json"},
                "test_models": ["stub/test"],
                "persona_model": "stub/persona",
                "council_models": ["stub/council-a", "stub/council-b", "stub/council-c"],
                "cheap_panel_models": ["stub/cheap-a", "stub/cheap-b", "stub/cheap-c"],
                "repeats": {"A": 0, "B": 0, "C": 0, "D": 0},
                "reasoning": {
                    "test_model": "default",
                    "persona": "on",
                    "council": "on",
                    "cheap_judge": "on",
                },
                "turn_cap": 6,
                "max_concurrency": 1,
                "prompt_versions": {
                    "persona": "persona-week1-v3",
                    "judge_pass1": "judge-pass1-general-v2",
                    "judge_pass2": "judge-pass2-general-v3",
                    "judge_pass2_boundary": "judge-pass2-boundary-general-v1",
                },
                "max_tokens": {
                    "test_model": 4096,
                    "persona": 2048,
                    "judge_pass1": 1200,
                    "judge_pass2": 1000,
                    "council": 1200,
                    "cheap_judge": 1200,
                },
                "test_only_allow_repeat_zero": True,
                "split_assignment": {"development": ["S1"], "confirmatory": []},
            },
            sort_keys=True,
        )
    )
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
