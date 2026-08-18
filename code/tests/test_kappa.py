from __future__ import annotations

import pytest

from slice.kappa import (
    CHEAP_VS_HUMAN_BAR,
    KappaResult,
    OUTCOME_LABEL_SPACE,
    cohens_kappa,
    kappa_report,
    normalise_label,
    pabak,
    per_class_agreement,
    verdict_of,
)


def _expand_pairs(parts):
    pairs = []
    for pair, count in parts:
        pairs.extend([pair] * count)
    return pairs


def test_cohens_kappa_matches_hand_computed_binary_and_three_class_fixtures():
    binary = [("yes", "yes")] * 2 + [("yes", "no"), ("no", "yes")] + [("no", "no")] * 6
    assert cohens_kappa(binary) == pytest.approx(0.5238095238)

    three_class = [("a", "a")] * 2 + [("b", "b")] * 2 + [("c", "c"), ("a", "b"), ("b", "c")]
    assert cohens_kappa(three_class) == pytest.approx(0.5625)


def test_pabak_uses_fixed_label_space_size_even_when_labels_are_absent():
    binary = [("yes", "yes")] * 8 + [("yes", "no"), ("no", "yes")]
    assert pabak(binary, label_space=["yes", "no"]) == pytest.approx(0.6)

    three_class = [("a", "a")] * 2 + [("b", "b")] * 2 + [("c", "c"), ("a", "b"), ("b", "c")]
    assert pabak(three_class, label_space=["a", "b", "c"]) == pytest.approx(4 / 7)
    assert pabak(three_class, label_space=["a", "b", "c", "d"]) == pytest.approx(13 / 21)


def test_high_prevalence_axis_passes_on_pabak_and_class_agreement_lows():
    pairs = [("correct", "correct")] * 95 + [("harmful", "harmful")] * 5

    report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )

    assert report.anchor_base_rate_max == pytest.approx(0.95)
    assert report.gated_metric == "pabak"
    assert report.pabak_ci_low == pytest.approx(1.0)
    assert report.pos_agreement_ci_low == pytest.approx(1.0)
    assert report.neg_agreement_ci_low == pytest.approx(1.0)
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "PASS"


def test_high_prevalence_rare_positive_disagreement_fails_joint_pabak_gate():
    harmful_misses = [("correct", "correct")] * 95 + [("harmful", "correct")] * 5

    report = kappa_report(
        harmful_misses,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )

    assert report.gated_metric == "pabak"
    assert report.pabak_ci_low is not None
    assert report.pos_agreement == 0.0
    assert report.pos_agreement_ci_low == 0.0
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "DEMOTE_TO_ESTIMATION"


def test_balanced_axis_still_gates_on_kappa_lower_ci():
    pairs = [("correct", "correct")] * 12 + [("harmful", "harmful")] * 12 + [("harmful", "correct")] * 4

    report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )

    assert report.anchor_base_rate_max < 0.85
    assert report.gated_metric == "kappa"
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == (
        "PASS" if report.kappa_ci_low is not None and report.kappa_ci_low >= CHEAP_VS_HUMAN_BAR else "DEMOTE_TO_ESTIMATION"
    )


def test_pabak_threshold_is_configurable():
    pairs = [("correct", "correct")] * 80 + [("harmful", "harmful")] * 20

    default_report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )
    lowered_report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
        pabak_prevalence_threshold=0.75,
    )

    assert default_report.anchor_base_rate_max == pytest.approx(0.8)
    assert default_report.gated_metric == "kappa"
    assert lowered_report.gated_metric == "pabak"
    assert verdict_of(lowered_report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2, pabak_prevalence_threshold=0.75) == "PASS"


def test_pabak_prevalence_uses_anchor_side_marginal_not_marker_side():
    pairs = [("correct", "correct")] * 50 + [("harmful", "correct")] * 50

    report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
        pabak_prevalence_threshold=0.7,
    )

    assert report.anchor_base_rate_max == pytest.approx(0.5)
    assert report.gated_metric == "kappa"


def test_outcome_report_uses_weighted_kappa_lower_ci_and_confusion_matrix_for_gate():
    pairs = _expand_pairs(
        [
            (("correct", "correct"), 9),
            (("partial", "partial"), 7),
            (("incorrect", "incorrect"), 6),
            (("harmful", "harmful"), 3),
            (("correct", "partial"), 1),
            (("incorrect", "harmful"), 1),
            (("correct", "incorrect"), 1),
        ]
    )

    report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )

    assert report.n == 28
    assert report.gated_statistic == "weighted_kappa"
    assert report.weighted_kappa == pytest.approx(0.9019, abs=1e-3)
    assert report.gated_value == pytest.approx(0.9019, abs=1e-3)
    assert report.gated_value > CHEAP_VS_HUMAN_BAR
    assert report.kappa_ci_low == pytest.approx(0.7183, abs=1e-3)
    assert report.confusion_matrix == [
        [9, 1, 1, 0],
        [0, 7, 0, 0],
        [0, 0, 6, 1],
        [0, 0, 0, 3],
    ]
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "DEMOTE_TO_ESTIMATION"


def test_kappa_report_filters_off_space_labels_before_n_and_n_floor():
    report = kappa_report(
        [
            ("correct", "correct"),
            ("harmful", "harmful"),
            ("bogus", "correct"),
            ("partial", "bogus"),
            ("bogus", "bogus"),
        ],
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )

    assert report.n == 2
    assert report.confusion_matrix == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=3) == "INSUFFICIENT_N"


def test_clear_imperfect_outcome_fixture_passes_lower_ci_gate():
    pairs = _expand_pairs(
        [
            (("correct", "correct"), 30),
            (("partial", "partial"), 20),
            (("incorrect", "incorrect"), 20),
            (("harmful", "harmful"), 18),
            (("correct", "partial"), 3),
            (("incorrect", "partial"), 3),
            (("harmful", "incorrect"), 3),
        ]
    )

    report = kappa_report(
        pairs,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        ordinal=True,
    )

    assert report.n == 97
    assert report.gated_value == pytest.approx(0.9634, abs=1e-3)
    assert report.kappa_ci_low == pytest.approx(0.9338, abs=1e-3)
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "PASS"


def test_per_coder_constant_disagreement_is_undefined_by_degeneracy():
    report = kappa_report(
        [("correct", "partial")] * 30,
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
    )

    assert report.single_class is True
    assert verdict_of(report, threshold=CHEAP_VS_HUMAN_BAR, n_floor=30) == "UNDEFINED"


def test_per_class_agreement_reports_yes_and_no_cases_separately():
    result = per_class_agreement(
        [("yes", "yes"), ("yes", "no"), ("no", "no"), ("maybe", "no")],
        "yes",
    )

    assert result == {"yes_case": pytest.approx(0.5), "no_case": pytest.approx(2 / 3)}
    assert per_class_agreement([("a", "a")], "missing")["yes_case"] is None


def test_normalise_label_aligns_codebook_and_foundation_spellings():
    assert normalise_label(" branch-covered ") == "branch_covered"
    report = kappa_report(
        [("branch-covered", "branch_covered"), ("elicited", "elicited")],
        axis="ask_fact",
        label_space=["elicited", "branch_covered", "unconditioned"],
        positive_label="unconditioned",
    )
    assert report.observed_agreement == 1.0


def test_verdict_of_pass_demote_insufficient_and_undefined_cases():
    passing = kappa_report(
        [("a", "a")] * 9 + [("b", "b")],
        axis="ask_fact",
        label_space=["a", "b", "c"],
        positive_label="b",
    )
    assert verdict_of(passing, threshold=CHEAP_VS_HUMAN_BAR) == "PASS"
    with pytest.raises(TypeError):
        verdict_of(passing)

    missing_ci = KappaResult(
        axis="outcome",
        label_space=OUTCOME_LABEL_SPACE,
        positive_label="harmful",
        n=20,
        observed_agreement=0.8,
        kappa=0.65,
        pabak=0.8,
        weighted_kappa=None,
        gated_statistic="cohens_kappa",
        gated_value=0.65,
        gated_metric="kappa",
        kappa_ci_low=None,
        kappa_ci_high=None,
        pabak_ci_low=None,
        pabak_ci_high=None,
        confusion_matrix=[[8, 2], [2, 8]],
        base_rate_max=0.5,
        anchor_base_rate_max=0.5,
        pos_agreement=None,
        neg_agreement=None,
        yes_case_agreement=None,
        no_case_agreement=None,
        pos_agreement_ci_low=None,
        pos_agreement_ci_high=None,
        neg_agreement_ci_low=None,
        neg_agreement_ci_high=None,
        single_class=False,
    )
    assert verdict_of(missing_ci, threshold=CHEAP_VS_HUMAN_BAR) == "INSUFFICIENT_N"

    too_small = kappa_report([("a", "a")], axis="ask_fact", label_space=["a", "b"], positive_label="b")
    assert verdict_of(too_small, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "INSUFFICIENT_N"

    single_class = kappa_report(
        [("a", "a")] * 20,
        axis="ask_fact",
        label_space=["a", "b", "c"],
        positive_label="b",
    )
    assert verdict_of(single_class, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "UNDEFINED"

    redundant_agreement_guard = KappaResult(
        axis="ask_fact",
        label_space=("a", "b"),
        positive_label="b",
        n=20,
        observed_agreement=0.0,
        kappa=None,
        pabak=None,
        weighted_kappa=None,
        gated_statistic="cohens_kappa",
        gated_value=None,
        gated_metric="pabak",
        kappa_ci_low=1.0,
        kappa_ci_high=1.0,
        pabak_ci_low=1.0,
        pabak_ci_high=1.0,
        confusion_matrix=[[0, 20], [0, 0]],
        base_rate_max=1.0,
        anchor_base_rate_max=1.0,
        pos_agreement=None,
        neg_agreement=None,
        yes_case_agreement=None,
        no_case_agreement=None,
        pos_agreement_ci_low=None,
        pos_agreement_ci_high=None,
        neg_agreement_ci_low=None,
        neg_agreement_ci_high=None,
        single_class=True,
    )
    assert verdict_of(redundant_agreement_guard, threshold=CHEAP_VS_HUMAN_BAR, n_floor=2) == "UNDEFINED"
