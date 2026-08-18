from __future__ import annotations

import re
import unicodedata
from typing import Any

_TRAILING_QUESTION_MARKUP = " \t\r\n*_\"')]}"


def _word_norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9£.%\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_question(text: str) -> bool:
    """Conservative question detector: only a trailing question mark counts."""
    return unicodedata.normalize("NFKC", text).rstrip(_TRAILING_QUESTION_MARKUP).endswith("?")


def elicited(model_turn: str, dimension: Any, *, require_question: bool = True) -> bool:
    """
    Return whether this single assistant turn asks about the dimension topic.

    A topic only counts as elicited inside a question by default. This prevents
    bare paraphrases such as ISA, goal, or APR from matching recommendation
    sentences that merely mention the topic.
    """
    segments = _question_segments(model_turn) if require_question else [model_turn]
    if require_question and not segments:
        return False

    paraphrases = getattr(dimension, "paraphrases", None)
    if paraphrases is None and isinstance(dimension, dict):
        paraphrases = dimension.get("paraphrases", [])
    for segment in segments:
        text = _word_norm(segment)
        for phrase in paraphrases or []:
            if _phrase_present(text, phrase):
                return True
    return False


def present_in_prompt(prompt_text: str, dimension: Any) -> bool:
    """Return True when a dimension's canary value or marker is in the opening prompt."""
    return any(
        _value_present(prompt_text, value, full_phrase=full_phrase)
        for value, full_phrase in _extract_match_values(dimension)
    )


def detect_leak(text: str, facts: list[Any]) -> list[dict[str, str]]:
    leaks: list[dict[str, str]] = []
    for fact in facts:
        dimension_id = _field(fact, "dimension_id", _field(fact, "id"))
        seen_values: set[tuple[str, bool]] = set()
        for value, full_phrase in _extract_match_values(fact):
            marker = (value, full_phrase)
            if marker in seen_values:
                continue
            seen_values.add(marker)
            if _value_present(text, value, full_phrase=full_phrase):
                leaks.append({"dimension_id": str(dimension_id), "value": value})
    return leaks


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _extract_match_values(item: Any) -> list[tuple[str, bool]]:
    values = _field(item, "canary_values")
    if values is None:
        canary_value = _field(item, "canary_value")
        values = [canary_value] if canary_value else []
    elif isinstance(values, str):
        values = [values]
    else:
        values = list(values)

    qualitative_marker = _field(item, "qualitative_marker")
    matches = [(value, False) for value in values if isinstance(value, str) and value]
    if qualitative_marker:
        matches.append((qualitative_marker, True))
    return matches


def _question_segments(text: str) -> list[str]:
    return [segment for segment in _sentence_segments(text) if looks_like_question(segment)]


def question_segment_spans(text: str) -> list[tuple[tuple[int, int], str]]:
    return [
        ((start, end), segment)
        for start, end, segment in _sentence_segment_spans(text)
        if looks_like_question(segment)
    ]


def _sentence_segments(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    return [
        match.group(0).strip()
        for match in re.finditer(r"[^?.!\n]+[?.!]?|[?.!]", text)
        if match.group(0).strip()
    ]


def _sentence_segment_spans(text: str) -> list[tuple[int, int, str]]:
    return [
        (start, end, text[start:end])
        for match in re.finditer(r"[^?.!\n]+[?.!]?|[?.!]", text)
        for start, end in [_trimmed_span(text, match.start(), match.end())]
        if start < end
    ]


def _phrase_present(normalised_text: str, phrase: str) -> bool:
    normalised_text = _phrase_norm(normalised_text)
    normalised_phrase = _phrase_norm(phrase)
    if not normalised_phrase:
        return False
    return f" {normalised_phrase} " in f" {normalised_text} "


def _phrase_norm(text: str) -> str:
    return re.sub(r"\s+", " ", _word_norm(text).replace(".", " ")).strip()


def find_value_spans(text: str, value: str, *, full_phrase: bool = False) -> list[tuple[int, int]]:
    if full_phrase:
        return find_phrase_spans(text, value)

    pattern = _value_pattern(value, full_phrase=False)
    if pattern is None:
        return []
    return [match.span() for match in pattern.finditer(text)]


def find_phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    if not _phrase_present(_word_norm(text), phrase):
        return []
    pattern = _phrase_pattern(phrase)
    if pattern is None:
        return []
    return [match.span() for match in pattern.finditer(text)]


def _value_present(text: str, value: str, *, full_phrase: bool = False) -> bool:
    if full_phrase:
        return _phrase_present(_word_norm(text), value)

    text = unicodedata.normalize("NFKC", text)
    pattern = _value_pattern(value, full_phrase=False)
    if pattern is None:
        # A value that normalises to empty historically matched (substring "" in text); preserve that.
        return True
    return pattern.search(text) is not None


def _value_pattern(value: str, full_phrase: bool = False) -> re.Pattern[str] | None:
    if full_phrase:
        return None
    value = unicodedata.normalize("NFKC", value)
    if "%" in value:
        numeric = re.escape(value.replace("%", "").strip())
        pattern = rf"(?<![\d.]){numeric}\s*%(?!\d)"
        return re.compile(pattern, flags=re.IGNORECASE)

    digits = re.sub(r"\D", "", value)
    if digits:
        if len(digits) > 3:
            number_pattern = rf"{re.escape(digits[:-3])}\s*,?\s*{re.escape(digits[-3:])}"
        else:
            number_pattern = re.escape(digits)
        pattern = rf"(?<![\d.])(?:£\s*)?{number_pattern}(?![\d.])"
        return re.compile(pattern, flags=re.IGNORECASE)

    normalised = _word_norm(value)
    if not normalised:
        return None
    return _flexible_raw_pattern(normalised, phrase=False)


def _phrase_pattern(phrase: str) -> re.Pattern[str] | None:
    normalised = _phrase_norm(phrase)
    if not normalised:
        return None
    return _flexible_raw_pattern(normalised, phrase=True)


def _flexible_raw_pattern(normalised: str, *, phrase: bool) -> re.Pattern[str]:
    tokens = [re.escape(token) for token in normalised.split() if token]
    separator = r"[^A-Za-z0-9£%]+" if phrase else r"[^A-Za-z0-9£.%]+"
    body = separator.join(tokens)
    if phrase:
        # Anchor at token boundaries so a marker is not located inside a larger token
        # (e.g. "office marker" must not match inside "xoffice markerx"); such a normalisation-only
        # match with no clean raw span is returned as [] and counted as unlocated, per the spec.
        body = rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])"
    return re.compile(body, flags=re.IGNORECASE)


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end
