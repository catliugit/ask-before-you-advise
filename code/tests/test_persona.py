from __future__ import annotations

from pathlib import Path

from slice.persona import build_persona_messages, build_persona_system_prompt
from slice.schema import load_scenario


ROOT = Path(__file__).resolve().parents[1]


def test_persona_prompt_builds_active_profile_with_new_rules():
    scenario = load_scenario(ROOT / "scenarios" / "s1.json")
    variant = scenario.module_a.variants[0]

    version, system_prompt = build_persona_system_prompt(ROOT / "prompts" / "persona.md", variant)
    messages = build_persona_messages(system_prompt, "You should keep it in cash.")

    assert version == "persona-week1-v3"
    assert "Active profile:" in system_prompt
    assert variant.profile in system_prompt
    assert "Never volunteer a fact you were not asked about." in system_prompt
    assert "without asking you anything" in system_prompt
    assert "add no new facts" in system_prompt
    assert "do not ask the assistant your own questions" in system_prompt.lower()
    assert "- the £8,000 is their only savings" in system_prompt
    assert messages == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "You should keep it in cash."},
    ]
