from __future__ import annotations

from typing import Any

from .phase_roles import is_safety_critical_record

PASS_GATE_VERDICT = "PASS"
DEMOTE_TO_ESTIMATION = "DEMOTE_TO_ESTIMATION"
BLOCKING_GATE_VERDICTS = frozenset({"BELOW", "DEMOTE_TO_ESTIMATION", "INSUFFICIENT_N", "UNDEFINED"})
GATE_VERDICTS = frozenset({PASS_GATE_VERDICT, *BLOCKING_GATE_VERDICTS})


class UnrecognisedGateVerdictError(ValueError):
    pass


def bulk_scoring_allowed(gate_verdict: dict[str, Any], module: str) -> bool:
    module_verdict = gate_verdict.get("per_module", {}).get(module, {}).get("verdict")
    if module_verdict not in GATE_VERDICTS:
        raise UnrecognisedGateVerdictError(
            f"unrecognised gate verdict for module {module}: {module_verdict!r}"
        )
    return module_verdict == PASS_GATE_VERDICT


def safety_critical_must_escalate(episode: dict[str, Any]) -> bool:
    return is_safety_critical_record(episode)


def cheap_panel_must_escalate(agreement: dict[str, Any]) -> bool:
    return agreement.get("escalated") is True or bool(agreement.get("escalation_triggers"))
