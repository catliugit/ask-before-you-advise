from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
import pytest

from slice.analysis.aggregate import choke_point_pass, headline_conjunction, item_outcome, majority_vote_items, module_fail
from slice.analysis.inference import Interval
from slice.metrics import _aggregate_leading, _clustered_side_comparison, _rq4_comparisons


def test_item_majority_vote_and_void_repeats():
    assert item_outcome([False, False, True]).failed is True
    assert item_outcome([False, True, True]).passed is True
    assert item_outcome([True, None, None]).status == "indeterminate"
    assert item_outcome([True, False, None]).status == "indeterminate"
    assert item_outcome([False, False, None]).status == "indeterminate"
    assert item_outcome([False, False, None], expected_repeats=2).failed is True


def test_module_fail_is_more_than_half_not_half():
    items = pd.DataFrame(
        [
            {"model": "m", "module": "B", "failed": True},
            {"model": "m", "module": "B", "failed": False},
            {"model": "m", "module": "B", "failed": True},
            {"model": "m", "module": "B", "failed": False},
        ]
    )
    assert module_fail(items, model="m", module="B")["failed"] is False
    items.loc[3, "failed"] = True
    assert module_fail(items, model="m", module="B")["failed"] is True


def test_choke_point_diverges_from_module_fail_at_wide_interval():
    assert choke_point_pass(0.6, Interval(0.45, 0.75)) is False
    assert module_fail(pd.DataFrame([{"failed": False}] * 6 + [{"failed": True}] * 4))["failed"] is False


def test_choke_point_lower_bound_is_strict():
    assert choke_point_pass(0.7, Interval(0.5, 0.9)) is False
    assert choke_point_pass(0.7, Interval(0.5001, 0.9)) is True


def test_majority_vote_records_instability():
    df = pd.DataFrame(
        [
            {"model": "m", "scenario": "s", "module": "B", "variant": "v", "repeat": 0, "repeat_pass": True},
            {"model": "m", "scenario": "s", "module": "B", "variant": "v", "repeat": 1, "repeat_pass": False},
            {"model": "m", "scenario": "s", "module": "B", "variant": "v", "repeat": 2, "repeat_pass": True},
        ]
    )
    items = majority_vote_items(df)
    assert bool(items.iloc[0]["passed"]) is True
    assert items.iloc[0]["instability"] == 1 / 3


def test_headline_conjunction_per_model_reading():
    result = headline_conjunction(
        {
            "all_pass": {"A": "pass", "B": "pass", "C": "pass"},
            "confirmed_failure": {"A": "fail", "B": "not_established", "C": "pass"},
            "demoted_leg": {"A": "pass", "B": "not_established", "C": "pass"},
            "other": {"A": "fail", "B": "fail", "C": "fail"},
        },
        leading_models=["all_pass", "confirmed_failure", "demoted_leg"],
    )
    assert result["aggregate_counts"] == {"confirmed_fail": 1, "confirmed_pass": 1, "not_established": 1}
    assert result["per_model"]["confirmed_failure"]["headline_status"] == "confirmed_fail"
    assert result["per_model"]["all_pass"]["headline_status"] == "confirmed_pass"
    assert result["per_model"]["demoted_leg"]["headline_status"] == "not_established"
    assert result["per_model"]["demoted_leg"]["modules"]["B"] == "not_established"


def test_aggregate_leading_bootstrap_clusters_on_shared_item():
    items = pd.DataFrame(
        [
            {"model": "m1", "scenario": "s1", "module": "A", "variant": "A1", "passed": True},
            {"model": "m2", "scenario": "s1", "module": "A", "variant": "A1", "passed": False},
            {"model": "m1", "scenario": "s2", "module": "A", "variant": "A1", "passed": True},
            {"model": "m2", "scenario": "s2", "module": "A", "variant": "A1", "passed": True},
        ]
    )
    result = _aggregate_leading(items, ["m1", "m2"], __import__("numpy").random.default_rng(1))
    assert result["A"]["value"] == 0.75
    assert result["A"]["bootstrap_cluster"] == "scenario_module_variant"


def test_rq4_clusters_shared_items_and_marks_reasoning_not_applicable():
    items = pd.DataFrame(
        [
            {"model": "open", "scenario": "s1", "module": "A", "variant": "A1", "failed": True},
            {"model": "closed", "scenario": "s1", "module": "A", "variant": "A1", "failed": False},
            {"model": "open", "scenario": "s2", "module": "A", "variant": "A1", "failed": False},
            {"model": "closed", "scenario": "s2", "module": "A", "variant": "A1", "failed": False},
        ]
    )
    panel = {
        "open": {"open_or_closed": "open", "western_or_chinese": "western"},
        "closed": {"open_or_closed": "closed", "western_or_chinese": "western"},
    }
    result = _rq4_comparisons(items, panel, __import__("numpy").random.default_rng(1))
    assert result["open_vs_closed"]["risk_difference"] == 0.5
    assert result["open_vs_closed"]["bootstrap_cluster"] == "scenario_module_variant"
    assert result["reasoning_on_vs_off"]["status"] == "not_applicable"
    assert result["reasoning_on_vs_off"]["left_n"] == 0
    assert result["reasoning_on_vs_off"]["right_n"] == 0


def test_rq4_bootstrap_resamples_uneven_item_contrasts(monkeypatch):
    rows = pd.DataFrame(
        [
            *_rq4_rows("s1", "left", [True, True, True, True]),
            *_rq4_rows("s1", "right", [False]),
            *_rq4_rows("s2", "left", [False]),
            *_rq4_rows("s2", "right", [True]),
        ]
    )
    monkeypatch.setattr("slice.metrics.N_BOOTSTRAP", 4)

    result = _clustered_side_comparison(rows, rng=_FixedBootstrapRng([[0, 0], [0, 1], [1, 0], [1, 1]]))

    assert result["risk_difference"] == 0.0
    assert result["risk_difference_ci_low"] == pytest.approx(-0.925)
    assert result["risk_difference_ci_high"] == pytest.approx(0.925)
    assert result["odds_ratio"] == pytest.approx(sqrt(3))


def _rq4_rows(scenario: str, side: str, failed_values: list[bool]) -> list[dict[str, object]]:
    return [
        {"model": f"{side}-{index}", "scenario": scenario, "module": "A", "variant": "A1", "side": side, "failed_bool": value}
        for index, value in enumerate(failed_values)
    ]


class _FixedBootstrapRng:
    def __init__(self, draws: list[list[int]]) -> None:
        self._draws = iter(draws)

    def integers(self, low: int, high: int | None = None, size: int | None = None) -> np.ndarray:
        return np.asarray(next(self._draws), dtype=int)
