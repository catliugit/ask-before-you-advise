from __future__ import annotations

from typing import Any


HUMAN_PHASES = {"human_dev", "human_test"}


def phase_of(record: dict[str, Any]) -> str | None:
    phase = record.get("phase")
    return str(phase) if phase is not None else None


def human_sample_part(record: dict[str, Any]) -> str:
    part = record.get("human_sample")
    if part in {"dev", "test"}:
        return str(part)
    phase = phase_of(record)
    if phase == "human_dev":
        return "dev"
    if phase == "human_test":
        return "test"
    return "none"


def is_confirmatory_record(record: dict[str, Any]) -> bool:
    phase = phase_of(record)
    if phase is not None:
        return phase == "confirmatory"
    return record.get("split") == "confirmatory"


def is_calibration_gate_record(record: dict[str, Any]) -> bool:
    if phase_of(record) == "calibration_gate":
        return True
    return bool(record.get("calibration_gate", False))


def is_safety_critical_record(record: dict[str, Any]) -> bool:
    return record.get("module") == "D" or record.get("any_harm_flagged") is True


def is_human_sample_record(record: dict[str, Any]) -> bool:
    phase = phase_of(record)
    if phase in HUMAN_PHASES:
        return True
    if phase in {"confirmatory", "calibration_gate"}:
        return False
    return human_sample_part(record) in {"dev", "test"}


def is_rule_fitting_record(record: dict[str, Any]) -> bool:
    if is_calibration_gate_record(record):
        return False
    phase = phase_of(record)
    if phase is not None:
        return phase == "development" and human_sample_part(record) == "none"
    return (
        record.get("split") == "development"
        and not record.get("calibration_gate", False)
        and human_sample_part(record) == "none"
    )


def is_cheap_pregate_record(record: dict[str, Any]) -> bool:
    return is_calibration_gate_record(record) or is_human_sample_record(record)


def is_duplicate_item(code: str, duplicate_map: dict[str, str] | None = None) -> bool:
    """Return True when a coder-facing handcode pack code is a duplicate presentation."""

    return code in (duplicate_map or {})


def is_h0_item(code: str, duplicate_map: dict[str, str] | None = None) -> bool:
    """Return True for non-duplicate H0 handcode pack codes.

    The input is a coder-facing pack code. Over pack codes, this is the complement of
    is_duplicate_item; 3c-iv uses that contract before pairing duplicate labels.
    """

    return code not in (duplicate_map or {})


def duplicate_source_code(code: str, duplicate_map: dict[str, str] | None = None) -> str | None:
    """Return the source stable_code for a duplicate pack code, else None."""

    return (duplicate_map or {}).get(code)


def is_h1_item(code: str, masked_map: dict[str, str] | None = None) -> bool:
    """Return True when a coder-facing masked review code is an H1 item."""

    return code in (masked_map or {})


def masked_source_code(code: str, masked_map: dict[str, str] | None = None) -> str | None:
    """Return the source H0 stable_code for a masked review code, else None."""

    return (masked_map or {}).get(code)
