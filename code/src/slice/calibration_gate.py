from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .calibration import run_calibration_gate
from .escalation import is_harm_flagged
from .freeze import calibration_verdicts_from_distillate, load_calibration_distillate
from .kappa import normalise_label
from .phase_roles import is_calibration_gate_record, is_safety_critical_record
from .resolution import council_majority, most_dangerous_council_label, resolve_council_field
from .schema import load_config, model_to_dict


def build_calibration_verdicts(config_path: str | Path) -> dict[str, dict[str, Any]]:
    config = load_config(config_path)
    data_root = Path(config.data_root)
    episodes = [
        episode
        for episode in _read_jsonl(data_root / "episodes" / "episodes.jsonl")
        if is_calibration_gate_record(episode) and episode.get("call_status", "ok") != "missing"
    ]
    judgements = _judgements_by_episode(_read_jsonl(data_root / "judgements.jsonl"))
    human_labels = _load_human_calibration_labels(data_root / "handcoding" / "calibration_human.csv")

    items_by_scenario: dict[str, list[dict[str, Any]]] = {}
    episodes_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        cheap_judgements = judgements.get(episode_id, {}).get("cheap_panel", [])
        council_judgements = judgements.get(episode_id, {}).get("council", [])
        cheap_outcome = council_majority(cheap_judgements, field="outcome")
        episode["any_harm_flagged"] = episode.get("any_harm_flagged") or is_harm_flagged(cheap_judgements)
        is_safety_axis = is_safety_critical_record(episode)
        if is_safety_axis:
            # Pre-reg: council disagreement on safety labels breaks toward the more dangerous reading.
            council_outcome = resolve_council_field(
                council_judgements,
                "outcome",
                safety_label=True,
                danger_order=most_dangerous_council_label,
            )["label"]
        else:
            council_outcome = council_majority(council_judgements, field="outcome")["label"]
        scenario_id = str(episode["scenario"])
        items_by_scenario.setdefault(scenario_id, []).append(
            {
                "episode_id": episode_id,
                "scenario_id": scenario_id,
                "is_apparent_pass": cheap_outcome["basis"] == "unanimous"
                and cheap_outcome["label"] == "correct",
                "cheap_outcome": cheap_outcome["label"],
                "council_outcome": council_outcome,
                "human_outcome": human_labels.get(episode_id),
                "is_safety_axis": is_safety_axis,
            }
        )
        episodes_by_scenario.setdefault(scenario_id, []).append(episode)

    records: dict[str, dict[str, Any]] = {}
    run_timestamp = config.run_timestamp
    for scenario_id in sorted(items_by_scenario):
        record = run_calibration_gate(
            scenario_id,
            items_by_scenario[scenario_id],
            run_timestamp=run_timestamp,
            instrument_hash=_instrument_hash(episodes_by_scenario[scenario_id]),
            false_safe_tolerance=config.calibration_false_safe_tolerance,
            disagree_threshold=config.calibration_trust_threshold_disagree_pct,
        )
        if record.council_items_audited + record.human_items_audited == 0:
            if hasattr(record, "model_copy"):
                record = record.model_copy(update={"verdict": "escalate_whole_scenario"})
            else:  # pragma: no cover - pydantic v1 fallback
                record = record.copy(update={"verdict": "escalate_whole_scenario"})
        records[scenario_id] = model_to_dict(record)
    return records


def write_calibration_verdicts(
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    output = Path(output_path) if output_path is not None else Path(config.data_root) / "outputs" / "calibration_verdicts.json"
    verdicts = build_calibration_verdicts(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verdicts, indent=2, sort_keys=True) + "\n")
    return output


def load_calibration_verdicts(config: Any) -> dict[str, dict[str, Any]] | None:
    frozen = load_calibration_distillate(config)
    if frozen is not None:
        return calibration_verdicts_from_distillate(frozen)
    path = Path(config.data_root) / "outputs" / "calibration_verdicts.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def calibration_permits_cheap_grading(
    verdicts: dict[str, dict[str, Any]] | None,
    scenario_id: str,
) -> bool:
    """Answer whether the cheap panel may run, not whether a cheap label may be finalised."""

    return verdicts is not None and scenario_id in verdicts


def calibration_block_reason(
    verdicts: dict[str, dict[str, Any]] | None,
    scenario_id: str,
) -> str | None:
    if calibration_permits_cheap_grading(verdicts, scenario_id):
        return None
    if verdicts is None:
        return "calibration_verdicts.json is absent"
    return f"no calibration verdict for scenario {scenario_id}; run the calibration gate before confirmatory scoring"


def scenario_escalate_whole(
    verdicts: dict[str, dict[str, Any]] | None,
    scenario_id: str,
) -> bool:
    if verdicts is None:
        return False
    return verdicts.get(scenario_id, {}).get("verdict") == "escalate_whole_scenario"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _judgements_by_episode(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        tier = row.get("judge_tier")
        if tier not in {"cheap_panel", "council"}:
            continue
        episode_id = row.get("episode_id")
        if not episode_id:
            continue
        grouped.setdefault(str(episode_id), {}).setdefault(str(tier), []).append(row)
    return grouped


def _load_human_calibration_labels(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            episode_id = row.get("episode_id")
            label = normalise_label(row.get("human_outcome_grade"))
            if episode_id and label:
                labels[str(episode_id)] = label
    return labels


def _instrument_hash(episodes: list[dict[str, Any]]) -> str | None:
    hashes = {episode.get("instrument_hash") for episode in episodes if episode.get("instrument_hash")}
    return next(iter(hashes)) if len(hashes) == 1 else None
