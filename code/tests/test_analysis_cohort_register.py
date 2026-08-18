from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from slice.metrics import select_cohort
from slice.schema import ModelPanel


ROOT = Path(__file__).resolve().parents[1]


def test_development_pilot_hash_null_yields_empty_confirmatory():
    df = pd.DataFrame([{"episode_id": "dev1", "split": "development", "instrument_hash": None, "call_status": "ok"}])
    assert select_cohort(df, "confirmatory", require_frozen=True).empty
    assert len(select_cohort(df, "development", require_frozen=False)) == 1


def test_confirmatory_null_or_mismatched_hash_aborts_with_episode_id():
    df = pd.DataFrame(
        [
            {"episode_id": "bad-null", "split": "confirmatory", "instrument_hash": None, "call_status": "ok"},
            {"episode_id": "bad-mismatch", "split": "confirmatory", "instrument_hash": "hash2", "call_status": "ok"},
        ]
    )
    with pytest.raises(ValueError, match="bad-null"):
        select_cohort(df, "confirmatory", require_frozen=True, expected_hash="hash1")


def test_confirmatory_cohort_requires_expected_frozen_hash():
    df = pd.DataFrame([{"episode_id": "e1", "split": "confirmatory", "instrument_hash": "hash1", "call_status": "ok"}])
    with pytest.raises(ValueError, match="freeze_record.json"):
        select_cohort(df, "confirmatory", require_frozen=True)


def test_frozen_panel_pin_checks_drifted_cheap_panel_on_escalated_episode():
    panel = _frozen_panel()
    df = pd.DataFrame(
        [
            {
                "episode_id": "escalated",
                "split": "confirmatory",
                "instrument_hash": "hash1",
                "call_status": "ok",
                "model": "anthropic/claude-opus-4.8",
                "observed_model_version": "pinned-v1",
                "judge_tier": "council",
                "judge_model": "anthropic/claude-opus-4.8",
                "judge_observed_model_version": "pinned-v1",
                "grading_role_model_versions": json.dumps(
                    {
                        "cheap_panel": [
                            {
                                "model": "google/gemini-3-flash-preview",
                                "observed_version": "drifted-cheap-v2",
                            }
                        ],
                        "council": [
                            {
                                "model": "anthropic/claude-opus-4.8",
                                "observed_version": "pinned-v1",
                            }
                        ],
                    },
                    sort_keys=True,
                ),
            }
        ]
    )

    with pytest.raises(ValueError, match="cheap_panel.*drifted-cheap-v2.*pinned-v1"):
        select_cohort(df, "confirmatory", require_frozen=True, expected_hash="hash1", panel=panel, config=object())


def _frozen_panel() -> ModelPanel:
    data = yaml.safe_load((ROOT / "model_panel.yaml").read_text())
    data["freeze_day"] = "2026-06-18"
    for entry in data["entries"]:
        entry["pinned_version"] = "pinned-v1"
    return ModelPanel.model_validate(data)
