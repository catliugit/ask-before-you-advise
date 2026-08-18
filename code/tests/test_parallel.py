from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import yaml

from slice._jsonl import append_jsonl
from slice.freeze import write_freeze_record
from slice.judge import score_all
from slice.runner import run_all


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeResult:
    text: str
    usage: dict[str, int]
    cost_estimate: float = 0.0
    request_id: str = "parallel-mock"
    model: str = "mock"
    latency_ms: float = 1.0
    sent_reasoning: str = "on"


class DeterministicClient:
    def __init__(self) -> None:
        self.calls = []
        self._lock = threading.Lock()

    def chat(self, model, messages, *, temperature=None, max_tokens=2048, cache_scope=None, timestamp=None):
        with self._lock:
            self.calls.append(
                {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "cache_scope": cache_scope,
                    "timestamp": timestamp,
                }
            )
        return FakeResult(
            text=self._response(model, messages),
            usage={"prompt_tokens": len(messages), "completion_tokens": 1},
            model=model,
        )

    def _response(self, model, messages):
        if model.startswith("stub/cheap"):
            payload = json.loads(messages[-1]["content"])
            if "critical_dimensions" in payload:
                return json.dumps(
                    {
                        "first_directive_turn": None,
                        "dimensions": [
                            {"dimension_id": dim["id"], "label": "unconditioned", "quote": ""}
                            for dim in payload["critical_dimensions"]
                        ],
                    },
                    sort_keys=True,
                )
            if "first_user_message" in payload:
                return json.dumps(
                    {
                        "resist_initial": "resisted",
                        "quote": payload["first_assistant_answer"],
                        "confidence": 0.9,
                    },
                    sort_keys=True,
                )
            return json.dumps(
                {
                    "outcome_grade": "correct",
                    "outcome_class": "general_conditional" if payload["module"] == "C" else None,
                    "deferral_score": 3 if payload["module"] == "D" else None,
                    "confidence": 0.9,
                    "safety_flag": "none",
                    "quote": payload["transcript"].splitlines()[-1],
                    "rationale": "stubbed",
                },
                sort_keys=True,
            )
        if model == "stub/persona":
            return "I'm not sure."
        return "You should keep the money accessible until the missing facts are clear."


def test_parallel_run_and_score_match_serial_sets(tmp_path):
    serial_config = _write_config(tmp_path / "serial" / "config.yaml", max_concurrency=1)
    parallel_config = _write_config(tmp_path / "parallel" / "config.yaml", max_concurrency=4)

    serial_episodes = run_all(serial_config, client=DeterministicClient(), enforce_preflight=False)
    serial_judgements = score_all(serial_config, client=DeterministicClient())
    parallel_episodes = run_all(parallel_config, client=DeterministicClient(), enforce_preflight=False)
    parallel_judgements = score_all(parallel_config, client=DeterministicClient())

    assert _jsonl_set(parallel_episodes) == _jsonl_set(serial_episodes)
    assert _jsonl_set(parallel_judgements) == _jsonl_set(serial_judgements)


def test_concurrent_jsonl_appends_are_well_formed_and_complete(tmp_path):
    path = tmp_path / "events.jsonl"
    payload = "x" * 2000

    def write_row(index: int) -> None:
        append_jsonl(path, {"index": index, "payload": payload})

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(write_row, range(250)))

    lines = path.read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(lines) == 250
    assert {row["index"] for row in rows} == set(range(250))
    assert all(row["payload"] == payload for row in rows)


def _write_config(path: Path, *, max_concurrency: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data",
                "config_root": str(ROOT),
                "model_panel_path": "model_panel.yaml",
                "scenario_paths": {"S1": "scenarios/s1.json", "B-scam": "scenarios/boundary_scam.json"},
                "test_models": ["stub/test-a", "stub/test-b"],
                "persona_model": "stub/persona",
                "council_models": ["stub/council-a", "stub/council-b", "stub/council-c"],
                "cheap_panel_models": ["stub/cheap-a", "stub/cheap-b", "stub/cheap-c"],
                "repeats": {"A": 3, "B": 0, "C": 0, "D": 0},
                "turn_cap": 6,
                "max_concurrency": max_concurrency,
                "prompt_versions": {
                    "persona": "persona-week1-v3",
                    "judge_pass1": "judge-pass1-general-v2",
                    "judge_pass2": "judge-pass2-general-v3",
                    "judge_resist_initial": "judge-resist-initial-v1",
                },
                "max_tokens": {
                    "test_model": 4096,
                    "persona": 2048,
                    "judge_pass1": 1200,
                    "judge_pass2": 1000,
                    "judge_resist_initial": 400,
                },
                "test_only_allow_repeat_zero": True,
                "split_assignment": {"development": [], "confirmatory": ["S1", "B-scam"]},
            },
            sort_keys=True,
        )
    )
    write_freeze_record(path, date_stamp="2026-06-18T09:00:00Z", record_dir=path.parent, enforce_preflight=False)
    _write_trusted_calibration(path.parent / "data", scenarios=("S1", "B-scam"))
    return path


def _write_trusted_calibration(data_root: Path, scenarios=("S1",)) -> None:
    outputs = data_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "calibration_verdicts.json").write_text(
        json.dumps(
            {
                scenario: {
                    "scenario_id": scenario,
                    "run_timestamp": "2026-06-18T09:00:00Z",
                    "instrument_hash": None,
                    "verdict": "trusted",
                    "audit_n_apparent_pass": 1,
                    "audit_n_non_pass": 0,
                    "false_safe_errors": 0,
                    "routine_disagree_pct": 0.0,
                    "human_items_audited": 0,
                    "council_items_audited": 1,
                }
                for scenario in scenarios
            },
            sort_keys=True,
        )
    )


def _jsonl_set(path: Path) -> set[str]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row.pop("run_timestamp", None)
        row.pop("run_id", None)
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return set(rows)
