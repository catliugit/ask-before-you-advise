from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from slice.schema import ModelPanel, SwapEntry, load_model_panel


ROOT = Path(__file__).resolve().parents[1]


def test_model_panel_loads_and_resolves_roles():
    panel = load_model_panel(ROOT / "model_panel.yaml")

    assert panel.entry_for_role("anthropic/claude-opus-4.8", "test").family == "anthropic"
    assert any(entry.leading for entry in panel.entries)


def test_model_panel_rejects_duplicate_slug_role():
    data = yaml.safe_load((ROOT / "model_panel.yaml").read_text())
    data["entries"].append(dict(data["entries"][0]))

    with pytest.raises(ValidationError, match="duplicate model panel slug-role"):
        ModelPanel.model_validate(data)


def test_model_panel_requires_rq4_axes_on_test_entry():
    data = yaml.safe_load((ROOT / "model_panel.yaml").read_text())
    del data["entries"][0]["western_or_chinese"]

    with pytest.raises(ValidationError, match="western_or_chinese"):
        ModelPanel.model_validate(data)


def test_model_panel_warns_without_leading_entry_but_loads():
    data = yaml.safe_load((ROOT / "model_panel.yaml").read_text())
    for entry in data["entries"]:
        entry["leading"] = False

    with pytest.warns(UserWarning, match="no leading entry"):
        panel = ModelPanel.model_validate(data)

    assert not any(entry.leading for entry in panel.entries)


def test_swap_log_append_round_trips():
    panel = load_model_panel(ROOT / "model_panel.yaml")
    record = SwapEntry(
        date="2026-06-17",
        cell="western_flagship",
        removed_slug="old/model",
        added_slug="new/model",
        reason="withdrawn before freeze",
    )
    dumped = panel.model_dump()
    dumped["swap_log"].append(record.model_dump())
    reloaded = ModelPanel.model_validate(dumped)

    assert reloaded.swap_log[-1].added_slug == "new/model"
