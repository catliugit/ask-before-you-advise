from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .canary import (
    _extract_match_values,
    _phrase_present,
    _sentence_segment_spans,
    _word_norm,
    detect_leak,
    elicited,
    find_phrase_spans,
    find_value_spans,
    looks_like_question,
    question_segment_spans,
)
from .handcode import _load_scenarios, _read_jsonl, _stable_hash, _stable_hash_bytes
from .schema import Dimension, Scenario, Variant, load_config


NAV_AID_VERSION = "nav-aid-v1"
NAV_AID_SEED = 20260625

RECOMMENDATION_LEXICON = (
    "i'd recommend",
    "i recommend",
    "i suggest",
    "i'd suggest",
    "you should",
    "i would",
    "my advice",
    "go with",
    "the best option",
    "you could",
    "put it in",
    "pay off",
    "clear the",
)

REFUSAL_SIGNPOST_LEXICON = (
    "i can't recommend",
    "i won't",
    "i would not",
    "this looks like a scam",
    "be very careful",
    "fca",
    "financial conduct authority",
    "citizens advice",
    "pension wise",
    "moneyhelper",
    "speak to a",
    "regulated adviser",
    "report it",
    "warning",
)

BUCKET_LABELS = {
    "request": "What the person asked for",
    "facts_disclosed": "Facts the person gave",
    "assistant_questions": "Where the assistant asked questions",
    "assistant_advice": "What the assistant recommended",
    "assistant_caveats": "Where the assistant hedged, declined, or pointed elsewhere",
    "after_pushback": "After the person pushed back",
}
BUCKET_ORDER = tuple(BUCKET_LABELS)

PERSON_SPEAKERS = {"user", "persona"}
ASSISTANT_SPEAKER = "assistant"
BANNED_PRODUCER_TOKENS = {
    "correct",
    "partial",
    "incorrect",
    "harmful",
    "elicited",
    "branch",
    "unconditioned",
    "held",
    "caved",
    "refused",
    "good",
    "bad",
    "right",
    "wrong",
    "missing",
    "ignored",
    "passed",
    "failed",
}


def export_nav_aids(config_path: str | Path) -> Path:
    config = load_config(config_path)
    review_dir = Path(config.data_root) / "handcoding" / "masked_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    cases = _safe_read_jsonl(review_dir / "masked_cases.jsonl")
    scenarios = _safe_load_scenarios(config)
    source_h1_lock_hash = _source_h1_lock_hash(review_dir / "masked_pack_manifest.json")

    nav_aids: dict[str, Any] = {}
    field_coverage: dict[str, dict[str, list[str]]] = {}
    skipped = 0
    unlocated_disclosures = 0

    for index, case in enumerate(sorted(cases, key=_case_sort_key)):
        masked_code = str(case.get("masked_code") or f"UNMASKED-{index:04d}")
        scenario = scenarios.get(str(case.get("scenario") or ""))
        variant = _variant_for(scenario, str(case.get("module") or ""), str(case.get("variant") or ""))
        aid, coverage, case_skipped, unlocated = _case_nav_aid(
            case,
            scenario=scenario,
            variant=variant,
            max_items=int(config.nav_aid_max_items_per_bucket),
        )
        nav_aids[masked_code] = aid
        field_coverage[masked_code] = coverage
        skipped += int(case_skipped)
        unlocated_disclosures += unlocated

    _assert_producer_neutral(nav_aids)

    nav_aid_bytes = _json_bytes(nav_aids)
    nav_aid_path = review_dir / "nav_aid.json"
    nav_aid_path.write_bytes(nav_aid_bytes)

    manifest = {
        "nav_aid_version": NAV_AID_VERSION,
        "nav_aid_hash": _stable_hash_bytes(nav_aid_bytes),
        "lexicon_hash": _lexicon_hash(),
        "aided_fraction": float(config.nav_aid_aided_fraction),
        "aid_arm": _assign_aid_arms(cases, float(config.nav_aid_aided_fraction)),
        "field_coverage": field_coverage,
        "unlocated_disclosures": unlocated_disclosures,
        "skipped": skipped,
        "source_h1_lock_hash": source_h1_lock_hash,
        "seed": NAV_AID_SEED,
    }
    (review_dir / "nav_aid_manifest.json").write_bytes(_json_bytes(manifest))
    return review_dir


def _case_nav_aid(
    case: dict[str, Any],
    *,
    scenario: Scenario | None,
    variant: Variant | None,
    max_items: int,
) -> tuple[dict[str, Any], dict[str, list[str]], bool, int]:
    transcript = _transcript(case)
    scored_fields = [str(field) for field in case.get("scored_fields") or []]
    grade_fields = _grade_fields(case, scored_fields)
    person_turns = _turns_with_speaker(transcript, PERSON_SPEAKERS)
    assistant_turns = _turns_with_speaker(transcript, {ASSISTANT_SPEAKER})
    case_skipped = scenario is None or variant is None or not person_turns or not assistant_turns
    unlocated_disclosures = 0
    buckets: list[dict[str, Any]] = []

    request_items = _request_items(transcript, person_turns)
    buckets.append(_bucket("request", request_items, max_items))

    facts_items, unlocated = _facts_items(transcript, person_turns, scenario, variant)
    unlocated_disclosures += unlocated
    buckets.append(_bucket("facts_disclosed", facts_items, max_items))

    ask_field_ids = [field for field in scored_fields if field not in {"outcome", "deferral"}]
    if ask_field_ids:
        buckets.append(
            _bucket(
                "assistant_questions",
                _assistant_question_items(transcript, assistant_turns, ask_field_ids, scenario),
                max_items,
            )
        )

    if "outcome" in scored_fields:
        buckets.append(
            _bucket(
                "assistant_advice",
                _lexicon_segment_items(
                    transcript,
                    assistant_turns,
                    RECOMMENDATION_LEXICON,
                    relates_to="outcome",
                ),
                max_items,
            )
        )

    if "deferral" in scored_fields or (
        scenario is not None and scenario.signpost_context != "general_suitability"
    ):
        buckets.append(
            _bucket(
                "assistant_caveats",
                _lexicon_segment_items(
                    transcript,
                    assistant_turns,
                    REFUSAL_SIGNPOST_LEXICON,
                    relates_to="deferral" if "deferral" in scored_fields else None,
                ),
                max_items,
            )
        )

    pushback_turn = _pushback_turn_index(transcript, person_turns, variant)
    if pushback_turn is not None:
        buckets.append(_bucket("after_pushback", _after_pushback_items(transcript, pushback_turn), max_items))

    applicable_keys = [bucket["key"] for bucket in buckets]
    coverage = _field_coverage(grade_fields, applicable_keys)
    return {"nav_aid_version": NAV_AID_VERSION, "buckets": buckets}, coverage, case_skipped, unlocated_disclosures


def _request_items(transcript: list[dict[str, str]], person_turns: list[int]) -> list[dict[str, Any]]:
    if not person_turns:
        return []
    turn = person_turns[0]
    text = transcript[turn]["text"]
    return [_item(transcript, turn, 0, len(text), value=None, relates_to=None)]


def _facts_items(
    transcript: list[dict[str, str]],
    person_turns: list[int],
    scenario: Scenario | None,
    variant: Variant | None,
) -> tuple[list[dict[str, Any]], int]:
    material, full_phrase_by_marker = _match_material(scenario, variant)
    if not material:
        return [], 0

    items: list[dict[str, Any]] = []
    unlocated = 0
    for turn in person_turns:
        text = transcript[turn]["text"]
        for leak in detect_leak(text, material):
            dimension_id = str(leak["dimension_id"])
            value = str(leak["value"])
            full_phrase = full_phrase_by_marker.get((dimension_id, value), False)
            spans = find_phrase_spans(text, value) if full_phrase else find_value_spans(text, value)
            if not spans:
                unlocated += 1
                continue
            for start, end in spans:
                items.append(_item(transcript, turn, start, end, value=value, relates_to=dimension_id))
    return items, unlocated


def _assistant_question_items(
    transcript: list[dict[str, str]],
    assistant_turns: list[int],
    ask_field_ids: list[str],
    scenario: Scenario | None,
) -> list[dict[str, Any]]:
    dimensions = _dimensions_by_id(scenario)
    scored_dimensions = [dimensions[field] for field in ask_field_ids if field in dimensions]
    items: list[dict[str, Any]] = []
    for turn in assistant_turns:
        for (start, end), segment in question_segment_spans(transcript[turn]["text"]):
            matched = [dimension for dimension in scored_dimensions if elicited(segment, dimension)]
            if matched:
                for dimension in matched:
                    items.append(_item(transcript, turn, start, end, value=None, relates_to=dimension.id))
            else:
                items.append(_item(transcript, turn, start, end, value=None, relates_to=None))
    return items


def _lexicon_segment_items(
    transcript: list[dict[str, str]],
    assistant_turns: list[int],
    lexicon: tuple[str, ...],
    *,
    relates_to: str | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for turn in assistant_turns:
        for start, end, segment in _sentence_segment_spans(transcript[turn]["text"]):
            for token in lexicon:
                if _phrase_present(_word_norm(segment), token):
                    items.append(_item(transcript, turn, start, end, value=token, relates_to=relates_to))
    return items


def _after_pushback_items(transcript: list[dict[str, str]], pushback_turn: int) -> list[dict[str, Any]]:
    items = [
        _item(
            transcript,
            pushback_turn,
            0,
            len(transcript[pushback_turn]["text"]),
            value=None,
            relates_to=None,
        )
    ]
    for turn in range(pushback_turn + 1, len(transcript)):
        if transcript[turn]["speaker"] == ASSISTANT_SPEAKER:
            items.append(
                _item(
                    transcript,
                    turn,
                    0,
                    len(transcript[turn]["text"]),
                    value=None,
                    relates_to=None,
                )
            )
    return items


def _match_material(
    scenario: Scenario | None,
    variant: Variant | None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], bool]]:
    material: list[dict[str, Any]] = []
    full_phrase_by_marker: dict[tuple[str, str], bool] = {}
    seen: set[tuple[str, str]] = set()
    sources: list[Any] = []
    if scenario is not None:
        sources.extend(scenario.dimensions)
    if variant is not None:
        sources.extend(variant.facts)

    for source in sources:
        dimension_id = _dimension_id(source)
        if not dimension_id:
            continue
        for value, full_phrase in _extract_match_values(source):
            marker = (dimension_id, value)
            if marker in seen:
                continue
            seen.add(marker)
            full_phrase_by_marker[marker] = full_phrase
            if full_phrase:
                material.append({"dimension_id": dimension_id, "qualitative_marker": value, "canary_values": []})
            else:
                material.append({"dimension_id": dimension_id, "canary_values": [value], "qualitative_marker": None})
    return material, full_phrase_by_marker


def _pushback_turn_index(
    transcript: list[dict[str, str]],
    person_turns: list[int],
    variant: Variant | None,
) -> int | None:
    pushback = variant.pushback if variant is not None else None
    if not pushback:
        return None
    for turn in person_turns:
        text = transcript[turn]["text"]
        if find_phrase_spans(text, pushback) or _phrase_present(_word_norm(text), pushback):
            return turn
    return None


def _bucket(key: str, items: list[dict[str, Any]], max_items: int) -> dict[str, Any]:
    ordered = sorted(items, key=_item_sort_key)
    truncated = len(ordered) > max_items
    capped = ordered[:max_items]
    return {
        "key": key,
        "label": BUCKET_LABELS[key],
        "no_match": not capped,
        "truncated": truncated,
        "items": capped,
    }


def _item(
    transcript: list[dict[str, str]],
    turn: int,
    start: int,
    end: int,
    *,
    value: str | None,
    relates_to: str | None,
) -> dict[str, Any]:
    text = transcript[turn]["text"]
    return {
        "turn": turn,
        "speaker": transcript[turn]["speaker"],
        "char_start": start,
        "char_end": end,
        "value": value,
        "text": text[start:end],
        "relates_to": relates_to,
    }


def _item_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["turn"],
        item["char_start"],
        item["char_end"],
        item.get("relates_to") or "",
        item.get("value") or "",
    )


def _field_coverage(fields: list[str], applicable_keys: list[str]) -> dict[str, list[str]]:
    applicable = set(applicable_keys)
    coverage: dict[str, list[str]] = {}
    for field in fields:
        if field == "outcome":
            candidates = (
                "request",
                "facts_disclosed",
                "assistant_advice",
                "assistant_caveats",
                "after_pushback",
            )
        elif field == "deferral":
            candidates = ("request", "assistant_caveats", "after_pushback")
        else:
            candidates = ("request", "facts_disclosed", "assistant_questions")
        keys = [key for key in candidates if key in applicable]
        coverage[field] = keys or [applicable_keys[0]] if applicable_keys else []
    return coverage


def _assign_aid_arms(cases: list[dict[str, Any]], fraction: float) -> dict[str, str]:
    by_module: dict[str, list[str]] = {}
    for case in cases:
        masked_code = case.get("masked_code")
        if not masked_code:
            continue
        by_module.setdefault(str(case.get("module") or "unknown"), []).append(str(masked_code))

    aid_arm: dict[str, str] = {}
    for module, masked_codes in sorted(by_module.items()):
        ordered = sorted(masked_codes)
        shuffled = list(ordered)
        seed = int(_stable_hash(f"{NAV_AID_SEED}:{module}")[:16], 16)
        random.Random(seed).shuffle(shuffled)
        aided = set(shuffled[: round(fraction * len(shuffled))])
        for masked_code in ordered:
            aid_arm[masked_code] = "aided" if masked_code in aided else "unaided"
    return aid_arm


def _lexicon_hash() -> str:
    payload = {
        "version": NAV_AID_VERSION,
        "bucket_keys": BUCKET_ORDER,
        "recommendation": RECOMMENDATION_LEXICON,
        "refusal_signpost": REFUSAL_SIGNPOST_LEXICON,
    }
    return _stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _assert_producer_neutral(nav_aids: dict[str, Any]) -> None:
    producer_strings = [
        *BUCKET_ORDER,
        *BUCKET_LABELS.values(),
        *RECOMMENDATION_LEXICON,
        *REFUSAL_SIGNPOST_LEXICON,
    ]
    for case in nav_aids.values():
        for bucket in case.get("buckets", []):
            producer_strings.append(str(bucket.get("key") or ""))
            producer_strings.append(str(bucket.get("label") or ""))
            for item in bucket.get("items", []):
                relates_to = item.get("relates_to")
                if relates_to is not None:
                    producer_strings.append(str(relates_to))

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(token) for token in sorted(BANNED_PRODUCER_TOKENS)) + r")(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    violations = sorted({value for value in producer_strings if pattern.search(value)})
    if violations:
        raise AssertionError(f"producer string contains banned token: {violations}")


def _transcript(case: dict[str, Any]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for turn in case.get("transcript") or []:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("speaker") or "unknown")
        text = str(turn.get("text") or "")
        turns.append({"speaker": speaker, "text": text})
    return turns


def _turns_with_speaker(transcript: list[dict[str, str]], speakers: set[str]) -> list[int]:
    return [index for index, turn in enumerate(transcript) if turn["speaker"] in speakers]


def _grade_fields(case: dict[str, Any], scored_fields: list[str]) -> list[str]:
    grade_schema = case.get("grade_schema")
    if isinstance(grade_schema, dict):
        return [str(field) for field in grade_schema]
    return list(scored_fields)


def _dimensions_by_id(scenario: Scenario | None) -> dict[str, Dimension]:
    if scenario is None:
        return {}
    return {dimension.id: dimension for dimension in scenario.dimensions}


def _dimension_id(source: Any) -> str | None:
    dimension_id = getattr(source, "dimension_id", None)
    if dimension_id is None and isinstance(source, dict):
        dimension_id = source.get("dimension_id")
    if dimension_id is None:
        dimension_id = getattr(source, "id", None)
    if dimension_id is None and isinstance(source, dict):
        dimension_id = source.get("id")
    return str(dimension_id) if dimension_id else None


def _variant_for(scenario: Scenario | None, module: str, variant_id: str) -> Variant | None:
    if scenario is None or not module or not variant_id:
        return None
    module_obj = {
        "A": scenario.module_a,
        "B": scenario.module_b,
        "C": scenario.module_c,
        "D": scenario.module_d,
    }.get(module)
    if module_obj is None:
        return None
    for variant in module_obj.variants:
        if variant.id == variant_id:
            return variant
    return None


def _safe_read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return _read_jsonl(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _safe_load_scenarios(config: Any) -> dict[str, Scenario]:
    try:
        return _load_scenarios(config)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _source_h1_lock_hash(path: Path) -> str | None:
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    value = manifest.get("h1_lock_hash") if isinstance(manifest, dict) else None
    return str(value) if value is not None else None


def _case_sort_key(case: dict[str, Any]) -> str:
    return str(case.get("masked_code") or "")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
