import pytest

from slice.kappa import (
    CHEAP_VS_HUMAN_BAR,
    COUNCIL_VS_HUMAN_BAR,
    OUTCOME_LABEL_SPACE,
    bootstrap_ci,
    cohens_kappa,
    confusion_matrix,
    fleiss_kappa,
    weighted_kappa,
)


def test_weighted_kappa_matches_hand_computed_golden_values():
    perfect = [("correct", "correct"), ("harmful", "harmful")]
    assert weighted_kappa(perfect, label_order=OUTCOME_LABEL_SPACE) == pytest.approx(1.0)

    base = (
        [("correct", "correct")] * 10
        + [("partial", "partial")] * 10
        + [("incorrect", "incorrect")] * 10
        + [("harmful", "harmful")] * 10
    )
    adjacent = base + [("correct", "partial")] * 4 + [("partial", "correct")] * 4
    far = base + [("correct", "harmful")] * 4 + [("harmful", "correct")] * 4

    adjacent_value = weighted_kappa(adjacent, label_order=OUTCOME_LABEL_SPACE)
    far_value = weighted_kappa(far, label_order=OUTCOME_LABEL_SPACE)
    assert adjacent_value == pytest.approx(0.931818, abs=1e-6)
    assert far_value == pytest.approx(0.470588, abs=1e-6)
    assert adjacent_value > far_value

    fixture = [("correct", "correct")] * 9 + [("incorrect", "incorrect")] * 9 + [("incorrect", "partial")] * 2
    assert weighted_kappa(fixture, label_order=OUTCOME_LABEL_SPACE) == pytest.approx(0.947368, abs=1e-6)
    assert weighted_kappa(fixture, label_order=OUTCOME_LABEL_SPACE, weights="linear") == pytest.approx(
        0.900000, abs=1e-6
    )
    assert cohens_kappa(fixture) == pytest.approx(0.818182, abs=1e-6)


def test_weighted_kappa_returns_none_for_degenerate_inputs():
    assert weighted_kappa([("correct", "correct")], label_order=OUTCOME_LABEL_SPACE) is None
    assert (
        weighted_kappa(
            [("correct", "correct"), ("correct", "correct")],
            label_order=OUTCOME_LABEL_SPACE,
        )
        is None
    )


def test_bootstrap_ci_is_seeded_contains_point_and_flags_thin_draws():
    statistic = lambda pairs: weighted_kappa(pairs, label_order=OUTCOME_LABEL_SPACE)
    stable = (
        [("correct", "correct")] * 10
        + [("partial", "partial")] * 10
        + [("incorrect", "incorrect")] * 10
        + [("harmful", "harmful")] * 10
        + [("correct", "partial")] * 4
        + [("partial", "correct")] * 4
    )

    first = bootstrap_ci(stable, statistic, seed=20260624)
    second = bootstrap_ci(stable, statistic, seed=20260624)
    assert first == second

    point = statistic(stable)
    low, high = first
    assert low is not None
    assert high is not None
    assert point is not None
    assert low <= point <= high

    low_variance = (
        [("correct", "correct")] * 20
        + [("partial", "partial")] * 20
        + [("incorrect", "incorrect")] * 20
        + [("harmful", "harmful")] * 20
        + [("correct", "partial")] * 2
        + [("partial", "correct")] * 2
    )
    high_variance = (
        [("correct", "correct")] * 4
        + [("harmful", "harmful")] * 4
        + [("correct", "harmful")] * 3
        + [("harmful", "correct")] * 3
        + [("partial", "incorrect")] * 2
        + [("incorrect", "partial")] * 2
    )
    low_variance_ci = bootstrap_ci(low_variance, statistic, seed=20260624)
    high_variance_ci = bootstrap_ci(high_variance, statistic, seed=20260624)
    assert low_variance_ci[0] is not None
    assert low_variance_ci[1] is not None
    assert high_variance_ci[0] is not None
    assert high_variance_ci[1] is not None
    assert high_variance_ci[1] - high_variance_ci[0] > low_variance_ci[1] - low_variance_ci[0]
    assert high_variance_ci[0] < statistic(high_variance)

    assert bootstrap_ci([("correct", "correct")], statistic, seed=20260624) == (None, None)

    thin = [("correct", "correct")] * 8 + [("partial", "partial"), ("harmful", "harmful")]

    def requires_both_rare_labels(pairs):
        labels = {label for pair in pairs for label in pair}
        if {"partial", "harmful"} - labels:
            return None
        return weighted_kappa(pairs, label_order=OUTCOME_LABEL_SPACE)

    assert requires_both_rare_labels(thin) is not None
    assert bootstrap_ci(thin, requires_both_rare_labels, seed=20260624) == (None, None)


def test_confusion_matrix_counts_in_label_space_order_and_drops_off_space_labels():
    pairs = [
        ("correct", "correct"),
        ("correct", "partial"),
        ("partial", "correct"),
        ("incorrect", "harmful"),
        ("harmful", "incorrect"),
        ("harmful", "harmful"),
        ("outside", "correct"),
        ("correct", "outside"),
        ("", "correct"),
        (None, "correct"),
    ]

    assert confusion_matrix(pairs, label_space=OUTCOME_LABEL_SPACE) == [
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
    ]


def test_fleiss_kappa_matches_assignment_weighted_golden_values():
    unanimous = [{"correct": 3}, {"harmful": 2}, {"correct": 4}]
    assert fleiss_kappa(unanimous) == pytest.approx(1.0)

    varying_n = [{"correct": 3}, {"correct": 2, "harmful": 1}, {"harmful": 2}, {"correct": 1, "harmful": 3}]
    assert fleiss_kappa(varying_n) == pytest.approx(0.416667, abs=1e-6)

    fixed_3 = [{"correct": 3}, {"correct": 2, "partial": 1}, {"partial": 3}, {"correct": 1, "partial": 2}]
    assert fleiss_kappa(fixed_3) == pytest.approx(0.333333, abs=1e-6)


def test_fleiss_kappa_returns_none_for_degenerate_inputs():
    assert fleiss_kappa([{"correct": 3}]) is None
    assert fleiss_kappa([{"correct": 3}, {"correct": 2}]) is None


def test_kappa_core_threshold_constants_are_pinned():
    assert CHEAP_VS_HUMAN_BAR == 0.75
    assert COUNCIL_VS_HUMAN_BAR == 0.80
