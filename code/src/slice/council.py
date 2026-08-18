from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .escalation import harm_flagged_episode_ids
from .phase_roles import is_rule_fitting_record, is_safety_critical_record
from .resolution import apply_safety_resolution, majority_from_labels, most_dangerous_council_label
from .schema import Scenario, load_config, load_scenario, resolve_from_config


COUNCIL_CODED_FIELDS = {
    "outcome": "final outcome grade",
    "deferral": "boundary deferral score",
}


def run_council(config_path: str | Path) -> Path:
    """Resolve first-round council labels into one development-sample label per field.

    This is the config-driven counterpart to the historical council scripts. It reads the
    development rule-fitting records, rejects confirmatory and calibration-gate rows, and
    writes the same one-row-per-(transcript, field) shape used by the kappa layer.
    """

    config_path = Path(config_path)
    config = load_config(config_path)
    data_root = Path(config.data_root)
    handcoding_dir = data_root / "handcoding"
    records, strict_records = _load_council_records(data_root)
    records = _development_rule_fitting_records(records, strict=strict_records)
    harm_flagged = harm_flagged_episode_ids(_read_jsonl(data_root / "judgements.jsonl"))
    for record in records:
        record["any_harm_flagged"] = record.get("any_harm_flagged") or str(record.get("episode_id")) in harm_flagged
    scenarios = _load_scenarios(config)

    council_labels = _load_council_judgement_labels(data_root, config.council_models)
    rows = _synthesise_council_judgements(records, scenarios, council_labels, config.council_models)
    output_path = handcoding_dir / "council_labels.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "code",
                "episode_id",
                "module",
                "variant",
                "field",
                "council_label",
                "basis",
                "minority_report",
                "human_handoff",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    pre_deliberation_path = handcoding_dir / "council_pre_deliberation.csv"
    with pre_deliberation_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "code",
                "episode_id",
                "module",
                "variant",
                "field",
                "coder",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(
            _synthesise_pre_deliberation(records, scenarios, council_labels, config.council_models)
        )
    split_summary_path = handcoding_dir / "council_split_summary.json"
    split_summary_path.write_text(
        json.dumps(_council_split_summary(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output_path


def build_council_cases(config_path: str | Path) -> Path:
    """Write anonymised disagreement cases for council deliberation."""

    config_path = Path(config_path)
    config = load_config(config_path)
    data_root = Path(config.data_root)
    handcoding_dir = data_root / "handcoding"
    records, strict_records = _load_council_records(data_root)
    records = _development_rule_fitting_records(records, strict=strict_records)
    scenarios = _load_scenarios(config)
    coders = _load_round1_coders(handcoding_dir, config.council_models)

    cases: list[dict[str, Any]] = []
    anon_map: dict[str, dict[str, str]] = {}
    for record in records:
        code = _record_code(record)
        slots = _anonymised_slots(config.council_models, code)
        slot_of = {real: f"Coder {index + 1}" for index, real in enumerate(slots)}
        anon_map[code] = {f"Coder {index + 1}": real for index, real in enumerate(slots)}
        disagreements = []
        for field in scored_fields(record, scenarios[record["scenario"]]):
            labels = {
                model: coders.get(model, {}).get(code, {}).get(field, "").strip()
                for model in config.council_models
            }
            present = {label for label in labels.values() if label}
            if len(present) > 1:
                disagreements.append(
                    {
                        "field": field,
                        "label_name": _field_name(field, scenarios[record["scenario"]]),
                        "anon_labels": {
                            slot_of[model]: label for model, label in labels.items() if label
                        },
                    }
                )
        if disagreements:
            cases.append(
                {
                    "code": code,
                    "episode_id": record.get("episode_id", code),
                    "module": record["module"],
                    "variant": record["variant"],
                    "transcript": _transcript_text(record),
                    "disagreements": disagreements,
                }
            )

    cases_path = handcoding_dir / "council_cases.json"
    anon_path = handcoding_dir / "council_anon_map.json"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2, sort_keys=True))
    anon_path.write_text(json.dumps(anon_map, ensure_ascii=False, indent=2, sort_keys=True))
    return cases_path


def scored_fields(record: dict[str, Any], scenario: Scenario) -> list[str]:
    module_name = record["module"]
    if module_name == "A":
        variant = _variant(scenario, "A", record["variant"])
        if variant.variant_kind == "fully_specified":
            return ["outcome"]
        return [dimension.id for dimension in scenario.dimensions if dimension.cls == "critical"] + ["outcome"]
    if module_name == "D":
        return ["deferral"]
    return ["outcome"]


def _load_council_records(data_root: Path) -> tuple[list[dict[str, Any]], bool]:
    episodes = _read_jsonl(data_root / "episodes" / "episodes.jsonl")
    council_transcripts = data_root / "handcoding" / "council_transcripts.jsonl"
    if council_transcripts.exists():
        return _join_handcoding_records(_read_jsonl(council_transcripts), episodes), True
    handcoding = data_root / "handcoding" / "transcripts.jsonl"
    if handcoding.exists():
        return _join_handcoding_records(
            _read_jsonl(handcoding),
            episodes,
            skip_codes=_duplicate_pack_codes(data_root),
        ), False
    return episodes, False


def _duplicate_pack_codes(data_root: Path) -> set[str]:
    path = data_root / "handcoding" / "handcode_pack_manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError):
        return set()
    if not isinstance(manifest, dict):
        return set()
    duplicate_map = manifest.get("duplicate_map") or {}
    if not isinstance(duplicate_map, dict):
        return set()
    return {str(code) for code in duplicate_map}


def _join_handcoding_records(
    handcoding_records: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    skip_codes: set[str] = frozenset(),
) -> list[dict[str, Any]]:
    episodes_by_id = {str(episode.get("episode_id")): episode for episode in episodes if episode.get("episode_id")}
    episodes_by_code = {
        _stable_handcode(episode["episode_id"]): episode
        for episode in episodes
        if episode.get("episode_id")
    }
    joined = []
    for record in handcoding_records:
        if str(record.get("code")) in skip_codes:
            continue
        episode = None
        if record.get("episode_id"):
            episode = episodes_by_id.get(str(record["episode_id"]))
        if episode is None and record.get("code"):
            code = str(record["code"])
            episode = episodes_by_code.get(code) or episodes_by_id.get(code)
        if episode is None:
            if "split" not in record or "calibration_gate" not in record:
                raise ValueError(
                    "handcoding transcript could not be joined to episodes.jsonl; "
                    f"episode_id={record.get('episode_id', '<missing>')} code={record.get('code', '<missing>')}"
                )
            joined.append(dict(record))
            continue
        merged = dict(episode)
        merged.update(record)
        joined.append(merged)
    return joined


def _development_rule_fitting_records(records: list[dict[str, Any]], *, strict: bool = False) -> list[dict[str, Any]]:
    if strict:
        for record in records:
            if not _rule_fitting_record(record):
                raise ValueError(
                    "council rule-fitting records must be phase=development, split=development, "
                    "calibration_gate=false, and outside the human sample; "
                    f"episode_id={record.get('episode_id', '<missing>')}"
                )
    return [
        record
        for record in records
        if _rule_fitting_record(record)
    ]


def _rule_fitting_record(record: dict[str, Any]) -> bool:
    return is_rule_fitting_record(record)


def _load_scenarios(config: Any) -> dict[str, Scenario]:
    return {
        scenario_id: load_scenario(resolve_from_config(config, path, root="config"))
        for scenario_id, path in config.scenario_paths.items()
    }


def _load_round1_coders(handcoding_dir: Path, council_models: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    return {
        model: _load_round1_csv(handcoding_dir / f"{_safe_model_name(model)}_coding.csv")
        for model in council_models
    }


def _load_round1_csv(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            code = row.get("code") or row.get("episode_id")
            if not code:
                continue
            labels = {
                "outcome": row.get("human_outcome_grade", ""),
                "deferral": row.get("human_deferral_score", ""),
            }
            for key, value in row.items():
                if key.startswith("human_dim_"):
                    labels[_dimension_from_column(key)] = value
            rows[code] = labels
    return rows


def _load_council_judgement_labels(
    data_root: Path,
    council_models: list[str],
) -> dict[str, dict[str, dict[str, str]]]:
    labels: dict[str, dict[str, dict[str, str]]] = {}
    council_model_set = set(council_models)
    for row in _read_jsonl(data_root / "judgements.jsonl"):
        if row.get("judge_tier") != "council":
            continue
        model = row.get("judge_model")
        if model not in council_model_set:
            continue
        episode_id = row.get("episode_id")
        if not episode_id:
            continue
        labels.setdefault(episode_id, {})[model] = _labels_from_judgement(row)
    return labels


def _labels_from_judgement(row: dict[str, Any]) -> dict[str, str]:
    labels: dict[str, str] = {}
    pass2 = row.get("pass2", {})
    if pass2.get("outcome_grade") is not None:
        labels["outcome"] = str(pass2["outcome_grade"])
    if pass2.get("deferral_score") is not None:
        labels["deferral"] = str(int(pass2["deferral_score"]))
    for verdict in row.get("pass1", {}).get("dimensions", []):
        dimension_id = verdict.get("dimension_id")
        label = verdict.get("label")
        if dimension_id and label:
            labels[str(dimension_id)] = str(label)
    return labels


def _synthesise_council_judgements(
    records: list[dict[str, Any]],
    scenarios: dict[str, Scenario],
    council_labels: dict[str, dict[str, dict[str, str]]],
    council_models: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        code = _record_code(record)
        episode_id = str(record.get("episode_id", code))
        for field in scored_fields(record, scenarios[record["scenario"]]):
            labels: list[str] = []
            missing: list[str] = []
            for model in council_models:
                label = council_labels.get(episode_id, {}).get(model, {}).get(field, "").strip()
                if label:
                    labels.append(label)
                else:
                    missing.append(model)
            if missing:
                raise ValueError(
                    f"missing council labels for episode_id={episode_id} field={field}: "
                    + ", ".join(missing)
                )
            majority = majority_from_labels(labels)
            safety_label = is_safety_critical_record(record) and field in {"outcome", "deferral"}
            result = apply_safety_resolution(
                majority,
                field,
                safety_label=safety_label,
                danger_order=most_dangerous_council_label,
            )
            rows.append(
                {
                    "code": code,
                    "episode_id": episode_id,
                    "module": record["module"],
                    "variant": record["variant"],
                    "field": field,
                    "council_label": result["label"] or "",
                    "basis": result["basis"],
                    "minority_report": ";".join(result["minority"]),
                    "human_handoff": str(result["human_handoff"]),
                }
            )
    return rows


def _synthesise_pre_deliberation(
    records: list[dict[str, Any]],
    scenarios: dict[str, Scenario],
    council_labels: dict[str, dict[str, dict[str, str]]],
    council_models: list[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        code = _record_code(record)
        episode_id = str(record.get("episode_id", code))
        for field in scored_fields(record, scenarios[record["scenario"]]):
            for model in council_models:
                label = council_labels.get(episode_id, {}).get(model, {}).get(field, "").strip()
                if not label:
                    continue
                rows.append(
                    {
                        "code": code,
                        "episode_id": episode_id,
                        "module": record["module"],
                        "variant": record["variant"],
                        "field": field,
                        "coder": model,
                        "label": label,
                    }
                )
    return rows


def _council_split_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    split_bases = {"human_handoff", "safety_break"}
    fields_total = len(rows)
    splits = sum(1 for row in rows if row["basis"] in split_bases)
    by_module: dict[str, dict[str, int]] = {}
    for row in rows:
        module = row["module"]
        module_counts = by_module.setdefault(module, {"fields_total": 0, "splits": 0})
        module_counts["fields_total"] += 1
        if row["basis"] in split_bases:
            module_counts["splits"] += 1
    return {
        "fields_total": fields_total,
        "splits": splits,
        "split_rate": splits / fields_total if fields_total else 0.0,
        "by_module": {
            module: {
                "fields_total": counts["fields_total"],
                "splits": counts["splits"],
                "split_rate": counts["splits"] / counts["fields_total"] if counts["fields_total"] else 0.0,
            }
            for module, counts in sorted(by_module.items())
        },
        "note": "confirmatory split rate is the final_grade_human_handoff rate in features.parquet (part 1)",
    }


def _synthesise_round1(
    records: list[dict[str, Any]],
    scenarios: dict[str, Scenario],
    coders: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for record in records:
        code = _record_code(record)
        for field in scored_fields(record, scenarios[record["scenario"]]):
            labels = [
                coder_labels.get(code, {}).get(field, "").strip()
                for coder_labels in coders.values()
            ]
            present = [label for label in labels if label]
            counts = Counter(present)
            if not present:
                label, basis, minority = "", "minority-report", []
            elif len(counts) == 1:
                label, basis, minority = present[0], "unanimous", []
            else:
                label, count = counts.most_common(1)[0]
                if count >= 2:
                    basis = "deliberated-majority"
                    minority = sorted({item for item in present if item != label})
                else:
                    label, basis, minority = "", "minority-report", sorted(set(present))
            rows.append(
                {
                    "code": code,
                    "episode_id": record.get("episode_id", code),
                    "module": record["module"],
                    "variant": record["variant"],
                    "field": field,
                    "council_label": label,
                    "basis": basis,
                    "minority_report": ";".join(minority),
                }
            )
    return rows


def _variant(scenario: Scenario, module_name: str, variant_id: str) -> Any:
    module = {
        "A": scenario.module_a,
        "B": scenario.module_b,
        "C": scenario.module_c,
        "D": scenario.module_d,
    }[module_name]
    if module is None:
        raise ValueError(f"scenario={scenario.id} has no module {module_name}")
    for variant in module.variants:
        if variant.id == variant_id:
            return variant
    raise ValueError(f"scenario={scenario.id} module={module_name} has no variant {variant_id}")


def _field_name(field: str, scenario: Scenario) -> str:
    if field in COUNCIL_CODED_FIELDS:
        return COUNCIL_CODED_FIELDS[field]
    for dimension in scenario.dimensions:
        if dimension.id == field:
            return dimension.name
    return field


def _record_code(record: dict[str, Any]) -> str:
    return str(record.get("code") or record.get("episode_id"))


def _transcript_text(record: dict[str, Any]) -> str:
    lines = []
    for turn in record.get("transcript") or []:
        speaker = turn.get("speaker", turn.get("role", "unknown"))
        lines.append(f"{speaker}: {turn.get('text', '').strip()}")
    return "\n".join(lines)


def _anonymised_slots(council_models: list[str], code: str) -> list[str]:
    if not council_models:
        return []
    rotation = int(hashlib.md5(code.encode()).hexdigest(), 16) % len(council_models)
    return council_models[rotation:] + council_models[:rotation]


def _safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _stable_handcode(episode_id: str) -> str:
    return "T" + hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:10].upper()


def _dimension_from_column(column: str) -> str:
    return column.removeprefix("human_dim_").replace("_", ".")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
