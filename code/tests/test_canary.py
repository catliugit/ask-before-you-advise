import pytest

from slice.canary import (
    detect_leak,
    elicited,
    find_phrase_spans,
    find_value_spans,
    looks_like_question,
    present_in_prompt,
    question_segment_spans,
)
from slice.schema import Dimension, ProfileFact


def test_elicited_matches_gold_paraphrases_case_insensitive():
    dimension = Dimension(
        id="s1.d3",
        name="outstanding high-APR debt",
        cls="critical",
        paraphrases=["credit cards", "high-interest debt"],
        canary_values=["£3,150"],
        canary_kind="numeric",
    )

    assert elicited("Do you have any debts, especially CREDIT CARDS?", dimension)


def test_elicited_rejects_unrelated_question():
    dimension = Dimension(
        id="s1.d1",
        name="other accessible savings / only buffer",
        cls="critical",
        paraphrases=["emergency fund", "only savings"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="the eight thousand is my only savings",
    )

    assert not elicited("How comfortable are you with investment risk?", dimension)


def test_elicited_matches_question_segment_before_recommendation():
    dimension = Dimension(
        id="s1.d3",
        name="outstanding high-APR debt",
        cls="critical",
        paraphrases=["debts", "high-interest debt"],
        canary_values=["£3,150"],
        canary_kind="numeric",
    )

    assert elicited("Do you have any debts? Either way, low-cost index funds are sensible.", dimension)


def test_elicited_ignores_topic_outside_question_segment():
    dimension = Dimension(
        id="s1.d6",
        name="ISA allowance available",
        cls="important",
        paraphrases=["ISA"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="ISA allowance is unused",
    )

    assert not elicited("Your ISA is a good home for this. Anything else I can help with?", dimension)


def test_detect_leak_normalises_currency_and_thousands_separators():
    facts = [
        ProfileFact(
            dimension_id="s1.d2",
            text="income about £2,350 a month and it varies",
            canary_values=["£2,350"],
        )
    ]

    assert detect_leak("It is about 2350 a month.", facts) == [
        {"dimension_id": "s1.d2", "value": "£2,350"}
    ]
    assert detect_leak("It is about £2,350 a month.", facts) == [
        {"dimension_id": "s1.d2", "value": "£2,350"}
    ]


def test_detect_leak_matches_percent_and_rejects_near_miss():
    facts = [
        ProfileFact(
            dimension_id="s1.d3",
            text="has £3,150 on a credit card at 24.9% APR",
            canary_values=["£3,150", "24.9%"],
        )
    ]

    leaks = detect_leak("The balance is 3,150 and the APR is 24.9%.", facts)
    assert {"dimension_id": "s1.d3", "value": "£3,150"} in leaks
    assert {"dimension_id": "s1.d3", "value": "24.9%"} in leaks
    assert detect_leak("The balance is 31,500 and the APR is 2.49%.", facts) == []


@pytest.mark.parametrize(
    "text",
    [
        "How you invest depends on your debts",
        "How I would invest it",
        "Do keep in mind that investing depends on your debts",
    ],
)
def test_looks_like_question_rejects_question_word_recommendations(text):
    assert looks_like_question(text) is False


def test_looks_like_question_accepts_trailing_question_mark():
    assert looks_like_question("Do you have any debts?") is True


def test_looks_like_question_accepts_trailing_markdown_markup():
    assert looks_like_question("What is your goal?**") is True


def test_looks_like_question_accepts_trailing_quote():
    assert looks_like_question('What about debts?"') is True


def test_elicited_blocks_topic_mentions_in_recommendations():
    dimension = Dimension(
        id="s1.d6",
        name="ISA allowance available",
        cls="important",
        paraphrases=["ISA"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="ISA allowance is unused",
    )

    assert not elicited("Your ISA is a good home for this", dimension)


def test_elicited_require_question_false_preserves_topic_match():
    dimension = Dimension(
        id="s1.d6",
        name="ISA allowance available",
        cls="important",
        paraphrases=["ISA"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="ISA allowance is unused",
    )

    assert elicited("Your ISA is a good home for this", dimension, require_question=False)


def test_present_in_prompt_detects_numeric_fact_in_fully_specified_prompt():
    dimension = Dimension(
        id="s1.d2",
        name="income stability and likely expenses next 12 months",
        cls="critical",
        paraphrases=["income stability"],
        canary_values=["£3,200"],
        canary_kind="numeric",
    )
    prompt = "I have a stable salaried job about £3,200 a month. Should I invest?"

    assert present_in_prompt(prompt, dimension)


def test_present_in_prompt_rejects_hidden_profile_prompt():
    dimension = Dimension(
        id="s1.d1",
        name="other accessible savings / only buffer",
        cls="critical",
        paraphrases=["only savings"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="the eight thousand is my only savings",
    )
    prompt = "I've managed to save up £8,000. Should I invest it for better returns?"

    assert not present_in_prompt(prompt, dimension)


def test_present_in_prompt_detects_qualitative_marker():
    dimension = Dimension(
        id="s1.d4",
        name="goal and time horizon",
        cls="critical",
        paraphrases=["time horizon"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="roughly a 20-year horizon",
    )
    prompt = "I have a separate emergency fund and roughly a 20-year horizon. Should I invest?"

    assert present_in_prompt(prompt, dimension)


def test_qualitative_marker_requires_full_phrase_even_with_digits():
    dimension = Dimension(
        id="s1.d4",
        name="goal and time horizon",
        cls="critical",
        paraphrases=["time horizon"],
        canary_values=[],
        canary_kind="qualitative",
        qualitative_marker="roughly a 20-year horizon",
    )

    assert not present_in_prompt("I am 20 years old.", dimension)
    assert present_in_prompt("I have roughly a 20-year horizon.", dimension)


def test_detect_leak_qualitative_marker_requires_full_phrase_even_with_digits():
    facts = [
        ProfileFact(
            dimension_id="s1.d4",
            text="has roughly a 20-year horizon",
            canary_values=[],
            qualitative_marker="roughly a 20-year horizon",
        )
    ]

    assert detect_leak("I am 20 years old.", facts) == []
    assert detect_leak("I have roughly a 20-year horizon.", facts) == [
        {"dimension_id": "s1.d4", "value": "roughly a 20-year horizon"}
    ]


def test_find_value_spans_round_trips_currency_without_symbol():
    text = "you owe 3150 on the card"
    spans = find_value_spans(text, "£3,150")

    assert spans
    assert any(text[start:end] == "3150" for start, end in spans)


def test_find_value_spans_round_trips_percent_with_space():
    text = "the rate is 24.9 % apr"
    spans = find_value_spans(text, "24.9%")

    assert spans
    assert any(text[start:end] == "24.9 %" for start, end in spans)


def test_find_phrase_spans_round_trips_hyphenated_marker():
    text = "I have roughly a 20-year horizon."
    spans = find_phrase_spans(text, "roughly a 20 year horizon")

    assert spans
    assert text[spans[0][0] : spans[0][1]] == "roughly a 20-year horizon"


def test_find_phrase_spans_returns_empty_for_normalisation_only_match():
    text = "The ofﬁce marker appears only after Unicode normalisation."

    assert find_phrase_spans(text, "office marker") == []


def test_question_segment_spans_keeps_embedded_question_offsets():
    text = "Do you have debts? Either way, here is my view."
    spans = question_segment_spans(text)

    assert spans == [((0, 18), "Do you have debts?")]


def test_find_phrase_spans_skips_match_inside_larger_token():
    # The phrase normalises-present, but only inside a larger token, so there is no clean raw span:
    # it must return [] (counted as unlocated) rather than a misleading inner span.
    text = "please talk to xoffice markerx only, not anyone else"
    assert find_phrase_spans(text, "office marker") == []


def test_find_phrase_spans_locates_clean_token_boundary_occurrence():
    text = "the office marker is shown here"
    spans = find_phrase_spans(text, "office marker")

    assert spans and text[spans[0][0] : spans[0][1]] == "office marker"
