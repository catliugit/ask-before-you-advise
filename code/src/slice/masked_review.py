from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .council import scored_fields
from .etl import _safe_col
from .handcode import (
    _load_scenarios,
    _read_jsonl,
    _stable_hash,
    _stable_hash_bytes,
    _strip_model_identity,
    stable_code,
)
from .kappa import ASK_FACT_LABEL_SPACE, OUTCOME_LABEL_SPACE, normalise_label
from .kappa_gate import _load_human_labels
from .schema import Scenario, load_config


MASKED_REVIEW_SEED = 20260618
MASKED_REVIEW_DIR = "masked_review"
MASKED_CODER_PACK_FILES = ("masked_cases.jsonl", "masked_review_template.csv", "instructions.md")
FLIP_LOG_SCHEMA = [
    "masked_code",
    "source_code",
    "field",
    "h0_label",
    "h1_label",
    "ai_final_label",
    "changed",
    "direction",
    "h1_reason",
    "seconds_on_case",
    "is_catch_trial",
    "catch_trial_passed",
    "snippets_shown",
    "snippet_followed",
]


def export_masked_review_pack(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data_root = Path(config.data_root)
    handcoding_dir = data_root / "handcoding"
    output_dir = handcoding_dir / MASKED_REVIEW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    h0_manifest = _load_h0_manifest(handcoding_dir / "handcode_pack_manifest.json")
    episodes = _read_jsonl(data_root / "episodes" / "episodes.jsonl")
    episodes_by_id = {str(episode.get("episode_id")): episode for episode in episodes if episode.get("episode_id")}
    h0_transcripts_by_code = {
        str(row.get("code")): row for row in _read_jsonl(handcoding_dir / "transcripts.jsonl") if row.get("code")
    }
    scenarios = _load_scenarios(config)
    human_labels = _load_h0_labels(handcoding_dir / "coding_completed.csv", episodes)
    features_by_episode = _load_features(data_root / "features.parquet")

    target = float(config.masked_review_target_agreement_fraction)
    classified, skip_counts = _classify_universe(
        h0_manifest,
        episodes_by_id=episodes_by_id,
        h0_transcripts_by_code=h0_transcripts_by_code,
        scenarios=scenarios,
        human_labels=human_labels,
        features_by_episode=features_by_episode,
    )
    # Assert uniqueness on the FULL classified universe BEFORE _review_queue keys anything by
    # masked_code: _sample_items/_shuffle_items build masked_code-keyed dicts, which would silently
    # drop one of a colliding pair before the guard could fire.
    _assert_masked_codes(classified, _source_codes(h0_manifest))
    queue, queue_composition = _review_queue(classified, target=target)
    masked_map = {item["masked_code"]: item["source_code"] for item in sorted(queue, key=lambda row: row["masked_code"])}

    _write_masked_cases(output_dir / "masked_cases.jsonl", queue)
    _write_template(output_dir / "masked_review_template.csv", queue)
    _write_instructions(output_dir / "instructions.md")
    _write_reveal(output_dir / "post_lock_reveal.json", queue)
    h1_lock_hash = _h1_lock_hash(output_dir)

    queue_composition.update(skip_counts)
    manifest = {
        "pack": "h1_masked",
        "seed": MASKED_REVIEW_SEED,
        "masked_map": masked_map,
        "queue_composition": queue_composition,
        "flip_log_schema": FLIP_LOG_SCHEMA,
        "h1_lock_hash": h1_lock_hash,
        "source_h0_lock_hash": h0_manifest.get("h0_lock_hash"),
        "source_h0_instrument_hash": h0_manifest.get("instrument_hash"),
    }
    (output_dir / "masked_pack_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return output_dir


def build_masked_review_pack(config_path: str | Path) -> Path:
    return export_masked_review_pack(config_path)


def masked_code(source_episode_id: str) -> str:
    return "M" + _stable_hash(f"masked:{source_episode_id}")[:10].upper()


def ai_labels_for_episode(
    episode_id: str,
    scored_fields: list[str],
    features_row: Any,
) -> dict[str, str | None]:
    labels: dict[str, str | None] = {}
    for field in scored_fields:
        if field == "outcome":
            labels[field] = _normalise_feature_label(_feature_value(features_row, "outcome_grade"))
        elif field == "deferral":
            labels[field] = _normalise_deferral(_feature_value(features_row, "deferral_score"))
        else:
            labels[field] = _normalise_feature_label(_feature_value(features_row, f"dim_{_safe_col(field)}"))
    return labels


def flip_direction(h0: Any, h1: Any, ai: Any) -> str:
    h0_label = normalise_label(h0)
    h1_label = normalise_label(h1)
    ai_label = normalise_label(ai)
    if h0_label == h1_label:
        return "unchanged"
    if ai_label is None:
        return "no_ai_label"
    if h1_label == ai_label:
        return "toward_ai"
    if h0_label == ai_label:
        return "away_from_ai"
    return "third_option"


def _load_h0_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_h0_labels(path: Path, episodes: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return _load_human_labels(path, episodes)


def _load_features(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    rows: dict[str, Any] = {}
    if "episode_id" not in frame.columns:
        return rows
    for _, row in frame.iterrows():
        episode_id = _normalise_scalar(row.get("episode_id"))
        if episode_id is not None:
            rows[str(episode_id)] = row
    return rows


def _classify_universe(
    h0_manifest: dict[str, Any],
    *,
    episodes_by_id: dict[str, dict[str, Any]],
    h0_transcripts_by_code: dict[str, dict[str, Any]],
    scenarios: dict[str, Scenario],
    human_labels: dict[str, dict[str, str]],
    features_by_episode: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    classified: list[dict[str, Any]] = []
    skipped_missing_h0 = 0
    skipped_missing_ai = 0
    skipped_no_comparable_field = 0

    for assignment in _h0_universe(h0_manifest):
        source_code = assignment["source_code"]
        episode_id = assignment["episode_id"]
        record = _record_for_assignment(
            assignment,
            episodes_by_id=episodes_by_id,
            h0_transcripts_by_code=h0_transcripts_by_code,
        )
        scenario = scenarios.get(str(record.get("scenario", "")))
        fields = scored_fields(record, scenario) if scenario is not None and record.get("module") else []
        h0_labels = human_labels.get(episode_id, {})
        ai_labels = ai_labels_for_episode(episode_id, fields, features_by_episode.get(episode_id))
        h0_by_field: dict[str, str | None] = {}
        ai_by_field: dict[str, str | None] = {}
        comparable_fields: list[str] = []
        has_h0_label = False
        has_ai_label = False

        for field in fields:
            h0_label = normalise_label(h0_labels.get(field))
            ai_label = normalise_label(ai_labels.get(field))
            h0_by_field[field] = h0_label
            ai_by_field[field] = ai_label
            has_h0_label = has_h0_label or h0_label is not None
            has_ai_label = has_ai_label or ai_label is not None
            if h0_label is not None and ai_label is not None:
                comparable_fields.append(field)

        if not comparable_fields:
            skipped_no_comparable_field += 1
            if not has_h0_label:
                skipped_missing_h0 += 1
            if not has_ai_label:
                skipped_missing_ai += 1
            continue

        item = {
            "source_code": source_code,
            "episode_id": episode_id,
            "masked_code": masked_code(episode_id),
            "record": record,
            "scored_fields": fields,
            "grade_schema": _grade_schema(fields, scenario, record),
            "h0_label": h0_by_field,
            "ai_final_grade": ai_by_field,
            "disagreement": any(h0_by_field[field] != ai_by_field[field] for field in comparable_fields),
        }
        classified.append(item)

    return classified, {
        "skipped_missing_h0": skipped_missing_h0,
        "skipped_missing_ai": skipped_missing_ai,
        "skipped_no_comparable_field": skipped_no_comparable_field,
    }


def _h0_universe(h0_manifest: dict[str, Any]) -> list[dict[str, str]]:
    assignments = h0_manifest.get("human_sample_assignments") or {}
    duplicate_codes = set((h0_manifest.get("duplicate_map") or {}).keys())
    universe: list[dict[str, str]] = []
    for source_code in sorted(assignments):
        if source_code in duplicate_codes:
            continue
        assignment = assignments[source_code] or {}
        episode_id = assignment.get("episode_id")
        if not episode_id:
            continue
        universe.append(
            {
                "source_code": str(source_code or stable_code(str(episode_id))),
                "episode_id": str(episode_id),
                "sample_role": str(assignment.get("sample_role") or ""),
            }
        )
    return universe


def _source_codes(h0_manifest: dict[str, Any]) -> set[str]:
    assignments = h0_manifest.get("human_sample_assignments") or {}
    return {str(code) for code in assignments}


def _record_for_assignment(
    assignment: dict[str, str],
    *,
    episodes_by_id: dict[str, dict[str, Any]],
    h0_transcripts_by_code: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    episode = dict(episodes_by_id.get(assignment["episode_id"], {}))
    h0_record = h0_transcripts_by_code.get(assignment["source_code"], {})
    for key in ("module", "scenario", "variant", "repeat", "transcript"):
        if key not in episode and key in h0_record:
            episode[key] = h0_record[key]
    episode.setdefault("episode_id", assignment["episode_id"])
    return episode


def _review_queue(classified: list[dict[str, Any]], *, target: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    disagreements = [item for item in classified if item["disagreement"]]
    agreements = [item for item in classified if not item["disagreement"]]
    if target >= 1.0 or not disagreements:
        n_agree_target = 0
    else:
        n_agree_target = round(target / (1.0 - target) * len(disagreements))
    n_agree = min(len(agreements), n_agree_target)
    sampled_agreements = _sample_items(agreements, n=n_agree, seed=MASKED_REVIEW_SEED + 1)
    combined = [*disagreements, *sampled_agreements]
    queue = _shuffle_items(combined, seed=MASKED_REVIEW_SEED + 2)
    queue_n = len(queue)
    achieved = (len(sampled_agreements) / queue_n) if queue_n else 0.0
    composition = {
        "n_eligible": len(classified),
        "n_disagreements": len(disagreements),
        "n_agreements": len(sampled_agreements),
        "queue_n": queue_n,
        "target_agreement_fraction": float(target),
        "achieved_agreement_fraction": achieved,
        "n_agreements_available": len(agreements),
        "n_agreements_target": n_agree_target,
        "agreement_shortfall": max(0, n_agree_target - len(agreements)),
    }
    return queue, composition


def _sample_items(items: list[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    by_code = {item["masked_code"]: item for item in items}
    ids = sorted(by_code)
    random.Random(seed).shuffle(ids)
    return [by_code[masked_id] for masked_id in ids[:n]]


def _shuffle_items(items: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    by_code = {item["masked_code"]: item for item in items}
    ids = sorted(by_code)
    random.Random(seed).shuffle(ids)
    return [by_code[masked_id] for masked_id in ids]


def _assert_masked_codes(queue: list[dict[str, Any]], source_codes: set[str]) -> None:
    masked = [item["masked_code"] for item in queue]
    if len(set(masked)) != len(masked):
        raise ValueError("masked code collision")
    collisions = set(masked) & source_codes
    if collisions:
        raise ValueError(f"masked code collision with H0 code: {', '.join(sorted(collisions))}")


def _write_masked_cases(path: Path, queue: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for item in queue:
            record = item["record"]
            public = {
                "masked_code": item["masked_code"],
                "module": record.get("module"),
                "scenario": record.get("scenario"),
                "variant": record.get("variant"),
                "repeat": _int_or_zero(record.get("repeat")),
                "scored_fields": item["scored_fields"],
                "grade_schema": item["grade_schema"],
                "transcript": _strip_model_identity(record.get("transcript") or [], model=record.get("model")),
            }
            handle.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")


def _write_template(path: Path, queue: list[dict[str, Any]]) -> None:
    fields = sorted({field for item in queue for field in item["scored_fields"]})
    columns = [
        "masked_code",
        *(["h1_outcome_grade"] if "outcome" in fields else []),
        *(["h1_deferral_score"] if "deferral" in fields else []),
        *[f"h1_ask_{field}" for field in fields if field not in {"outcome", "deferral"}],
        "h1_reason",
        "start_time",
        "end_time",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in queue:
            row = {"masked_code": item["masked_code"], "h1_reason": "", "start_time": "", "end_time": ""}
            for column in columns:
                row.setdefault(column, "")
            writer.writerow(row)


def _write_instructions(path: Path) -> None:
    path.write_text(
        "# Second look quality review\n\n"
        "Review each masked case cold using the normal grade controls for its duty.\n\n"
        "Use the case id only. Do not try to connect it to any earlier material.\n\n"
        "Record a short reason where useful, plus start and end times.\n\n"
        "Treat each case as an independent quality check.\n"
    )


def _write_reveal(path: Path, queue: list[dict[str, Any]]) -> None:
    reveal = {
        item["masked_code"]: {
            "ai_final_grade": item["ai_final_grade"],
            "h0_label": item["h0_label"],
        }
        for item in sorted(queue, key=lambda row: row["masked_code"])
    }
    path.write_text(json.dumps(reveal, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _h1_lock_hash(masked_dir: Path) -> str:
    payload = [
        (path.name, _stable_hash_bytes(path.read_bytes()))
        for path in sorted(masked_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and _is_masked_coder_pack_file(path.name)
    ]
    return _stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _is_masked_coder_pack_file(name: str) -> bool:
    return name in MASKED_CODER_PACK_FILES


def _grade_schema(fields: list[str], scenario: Scenario | None, record: dict[str, Any]) -> dict[str, list[str]]:
    return {field: _label_space(field, scenario, record) for field in fields}


def _label_space(field: str, scenario: Scenario | None, record: dict[str, Any]) -> list[str]:
    if field == "outcome":
        return list(OUTCOME_LABEL_SPACE)
    if field == "deferral":
        max_score = _deferral_max_score(scenario, record)
        return [str(score) for score in range(max_score + 1)]
    return list(ASK_FACT_LABEL_SPACE)


def _deferral_max_score(scenario: Scenario | None, record: dict[str, Any]) -> int:
    if scenario is None or scenario.module_d is None:
        return 3
    variant_id = str(record.get("variant") or "")
    for variant in scenario.module_d.variants:
        if variant.id == variant_id and variant.deferral_rubric is not None:
            return int(variant.deferral_rubric.max_score)
    return 3


def _feature_value(features_row: Any, column: str) -> Any:
    if features_row is None:
        return None
    if column not in features_row:
        return None
    return features_row.get(column)


def _normalise_feature_label(value: Any) -> str | None:
    value = _normalise_scalar(value)
    if value is None:
        return None
    return normalise_label(value)


def _normalise_deferral(value: Any) -> str | None:
    value = _normalise_scalar(value)
    if value is None:
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def _normalise_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
