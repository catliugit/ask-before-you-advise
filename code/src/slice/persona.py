from __future__ import annotations

from pathlib import Path

from .schema import ModuleAVariant, ProfileFact, load_prompt_file


FORCING_MOVE = "I don't have more details to hand, what should I do?"


def build_persona_system_prompt(prompt_path: str | Path, variant: ModuleAVariant) -> tuple[str, str]:
    prompt = load_prompt_file(prompt_path)
    facts = "\n".join(f"- {fact.text}" for fact in variant.facts)
    profile = variant.profile or ""
    system = (
        f"{prompt.text}\n\n"
        f"Active profile:\n{profile}\n\n"
        f"Profile facts:\n{facts}"
    )
    return prompt.version, system


def build_persona_messages(system_prompt: str, model_turn: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_turn},
    ]


def unasked_facts(variant: ModuleAVariant, elicited_dimension_ids: set[str]) -> list[ProfileFact]:
    return [fact for fact in variant.facts if fact.dimension_id not in elicited_dimension_ids]

