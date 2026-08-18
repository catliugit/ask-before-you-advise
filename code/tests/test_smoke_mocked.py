from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from slice.etl import build_features
from slice.freeze import write_freeze_record
from slice.handcode import export_handcode_pack
from slice.judge import score_all
from slice.metrics import compute_metrics
from slice.runner import run_all


ROOT = Path(__file__).resolve().parents[1]

def _prefreeze_panel_path(target_dir):
    # 7 Jul 2026 (freeze day): the repo model_panel.yaml now carries freeze_day + pins for the
    # confirmatory run, which switches on frozen-only behaviour (pin assertions, distillate
    # routing). These tests exercise pre-freeze behaviour, so they point at a copy with
    # freeze_day nulled (same pattern as the pre-freeze normalisation in test_freeze.py).
    panel = yaml.safe_load((ROOT / "model_panel.yaml").read_text())
    panel["freeze_day"] = None
    target = Path(target_dir) / "model_panel_prefreeze.yaml"
    target.write_text(yaml.safe_dump(panel, sort_keys=True))
    return str(target)



@dataclass
class FakeResult:
    text: str
    usage: dict[str, int]
    cost_estimate: float = 0.0
    request_id: str = "mock-request"
    model: str = "mock"
    latency_ms: float = 1.0
    sent_reasoning: str = "on"


class SmokeClient:
    def __init__(self, cost_log_path: Path) -> None:
        self.cost_log_path = cost_log_path
        self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def chat(self, model, messages, *, temperature=None, max_tokens=2048, cache_scope=None, timestamp=None):
        self.calls.append(
            {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "cache_scope": cache_scope,
            }
        )
        text = self._response(model, messages)
        self.cost_log_path.open("a").write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "model": model,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "cost_estimate": 0.0,
                    "cached": False,
                    "request_id": "mock-request",
                },
                sort_keys=True,
            )
            + "\n"
        )
        return FakeResult(text=text, usage={"prompt_tokens": 1, "completion_tokens": 1}, model=model)

    def _response(self, model, messages):
        if model in {"stub/cheap-a", "stub/cheap-b", "stub/cheap-c", "stub/council-a", "stub/council-b", "stub/council-c"}:
            payload = json.loads(messages[-1]["content"])
            if "critical_dimensions" in payload:
                quote = payload["transcript"].splitlines()[0]
                return json.dumps(
                    {
                        "dimensions": [
                            {"dimension_id": dim["id"], "label": "unconditioned", "quote": quote}
                            for dim in payload["critical_dimensions"]
                        ]
                    }
                )
            if "first_user_message" in payload:
                return json.dumps(
                    {
                        "resist_initial": "resisted",
                        "quote": payload["first_assistant_answer"],
                        "confidence": 0.9,
                    }
                )
            quote = payload["transcript"].splitlines()[-1].partition(": ")[2]
            return json.dumps(
                {
                    "outcome_grade": "correct",
                    "outcome_class": "general_conditional" if payload["module"] == "C" else None,
                    "deferral_score": 3 if payload["module"] == "D" else None,
                    "confidence": 0.9,
                    "safety_flag": "none",
                    "quote": quote,
                    "rationale": "stubbed",
                }
            )
        if model == "stub/persona":
            return "I'm not sure"

        prompt = messages[-1]["content"]
        all_text = "\n".join(message["content"] for message in messages)
        if "safe account" in all_text.lower():
            return "Do not transfer it; this is a safe account scam, hang up and call 159 or the number on your card."
        if "24.9% APR" in all_text:
            return "You should clear the 24.9% credit card first before investing."
        if "£11,400" in all_text and "20-year horizon" in all_text:
            return "Investing is suitable in a low-cost diversified fund inside a stocks-and-shares ISA."
        if len(messages) == 1 and "Should I invest it for better returns?" in prompt:
            return "Is this your only savings or do you have an emergency fund elsewhere?"
        return "You should keep the money accessible until the missing facts are clear."


def test_mocked_end_to_end_pipeline_produces_acceptance_artifacts():
    runtime = ROOT / "data" / "test_smoke_mocked"
    if runtime.exists():
        shutil.rmtree(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    config_path = runtime / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data_root": "data",
                "config_root": str(ROOT),
                "model_panel_path": _prefreeze_panel_path(runtime),
                "scenario_paths": {"S1": "scenarios/s1.json", "B-scam": "scenarios/boundary_scam.json"},
                "test_models": ["stub/test-a", "stub/test-b"],
                "persona_model": "stub/persona",
                "council_models": ["stub/council-a", "stub/council-b", "stub/council-c"],
                "cheap_panel_models": ["stub/cheap-a", "stub/cheap-b", "stub/cheap-c"],
                "repeats": {"A": 3, "B": 3, "C": 3, "D": 3},
                "turn_cap": 6,
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
                "split_assignment": {"development": [], "confirmatory": ["S1", "B-scam"]},
            },
            sort_keys=True,
        )
    )
    client = SmokeClient(runtime / "data" / "cost_log.jsonl")
    _write_trusted_calibration(runtime / "data")

    episodes_path = run_all(config_path, client=client, enforce_preflight=False)
    judgements_path = score_all(config_path, client=client)
    _stamp_frozen_hash(runtime / "data")
    features_path = build_features(config_path)
    metrics = compute_metrics(config_path)
    handcode_dir = export_handcode_pack(config_path)

    assert episodes_path.exists()
    assert judgements_path.exists()
    assert features_path.exists()
    assert (runtime / "data" / "metrics.json").exists()
    assert (runtime / "data" / "RESULTS.md").exists()
    assert (runtime / "data" / "cost_log.jsonl").exists()
    assert (handcode_dir / "transcripts.jsonl").exists()
    assert (handcode_dir / "coding_template.csv").exists()
    assert (handcode_dir / "instructions.md").exists()

    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    assert len(episodes) == 66
    assert all("prompt_versions" in episode for episode in episodes)
    assert all(episode["prompt_versions"]["persona"] == "persona-week1-v3" for episode in episodes)

    features = pd.read_parquet(features_path)
    assert len(features) == 66
    assert metrics["n_episodes"] == 66
    assert not any(call["model"].startswith("anthropic/") for call in client.calls)


def _write_trusted_calibration(data_root: Path) -> None:
    write_freeze_record(
        data_root.parent / "config.yaml",
        date_stamp="2026-06-18T09:00:00Z",
        record_dir=data_root.parent,
        enforce_preflight=False,
    )
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
                for scenario in ["S1", "B-scam"]
            },
            sort_keys=True,
        )
    )


def _stamp_frozen_hash(data_root: Path, value: str | None = None) -> None:
    if value is None:
        value = json.loads((data_root.parent / "freeze_record.json").read_text())["instrument_hash"]
    for path in [data_root / "episodes" / "episodes.jsonl", data_root / "judgements.jsonl"]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            row["instrument_hash"] = value
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
