from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import exp, log
from typing import Callable, Iterable

import numpy as np

from .constants import ALPHA, PRECISION_TARGET_MAX_WIDTH_PTS, min_discordant_for_confirmation


@dataclass(frozen=True)
class Interval:
    low: float | None
    high: float | None

    @property
    def width(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return self.high - self.low


def explicit_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def named_rng(seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    child_seed = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(child_seed)


def mean_or_none(values: Iterable[float | int | bool | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None and not _is_nan(value)]
    if not numeric:
        return None
    return float(np.mean(numeric))


def clustered_bootstrap_ci(
    values: Iterable[float | int | bool],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    statistic: Callable[[np.ndarray], float] | None = None,
    alpha: float = ALPHA,
) -> Interval:
    data = np.asarray([float(value) for value in values], dtype=float)
    if data.size == 0:
        return Interval(None, None)
    if statistic is None:
        statistic = lambda sample: float(np.mean(sample))
    if data.size == 1:
        point = statistic(data)
        return Interval(point, point)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample_index = rng.integers(0, data.size, size=data.size)
        draws[index] = statistic(data[sample_index])
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(low), float(high))


def clustered_bootstrap_ci_by_cluster(
    clusters: Iterable[Iterable[float | int | bool]],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    statistic: Callable[[np.ndarray], float] | None = None,
    alpha: float = ALPHA,
) -> Interval:
    cluster_arrays = [np.asarray([float(value) for value in cluster], dtype=float) for cluster in clusters]
    cluster_arrays = [cluster for cluster in cluster_arrays if cluster.size > 0]
    if not cluster_arrays:
        return Interval(None, None)
    data = np.concatenate(cluster_arrays)
    if statistic is None:
        statistic = lambda sample: float(np.mean(sample))
    if len(cluster_arrays) == 1:
        point = statistic(data)
        return Interval(point, point)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample_index = rng.integers(0, len(cluster_arrays), size=len(cluster_arrays))
        sample = np.concatenate([cluster_arrays[item] for item in sample_index])
        draws[index] = statistic(sample)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(low), float(high))


def bootstrap_t_ci(
    values: Iterable[float | int | bool],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
    statistic: Callable[[np.ndarray], float] | None = None,
    alpha: float = ALPHA,
) -> Interval:
    data = np.asarray([float(value) for value in values], dtype=float)
    if data.size == 0:
        return Interval(None, None)
    if statistic is None:
        statistic = lambda sample: float(np.mean(sample))
    point = statistic(data)
    if data.size == 1:
        return Interval(point, point)
    se_hat = _bootstrap_t_standard_error(data, statistic)
    if se_hat <= 0:
        return Interval(point, point)
    pivots = []
    for _ in range(n_bootstrap):
        sample_index = rng.integers(0, data.size, size=data.size)
        sample = data[sample_index]
        sample_se = _bootstrap_t_standard_error(sample, statistic)
        if sample_se <= 0:
            continue
        pivots.append((statistic(sample) - point) / sample_se)
    if not pivots:
        return Interval(point, point)
    lower_pivot, upper_pivot = np.quantile(pivots, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point - upper_pivot * se_hat), float(point - lower_pivot * se_hat))


def permutation_inverted_ci_from_stats(
    observed: float | None,
    null_stats: Iterable[float | int],
    *,
    alpha: float = ALPHA,
) -> Interval:
    if observed is None:
        return Interval(None, None)
    stats = np.asarray([float(value) for value in null_stats], dtype=float)
    if stats.size == 0:
        return Interval(None, None)
    low_null, high_null = np.quantile(stats, [1 - alpha / 2, alpha / 2])
    return Interval(float(observed - low_null), float(observed - high_null))


def holm_step_down(
    p_values: dict[str, float | None],
    *,
    alpha: float = ALPHA,
    family_size: int | None = None,
) -> dict[str, dict[str, float | bool | None]]:
    usable = [(name, p) for name, p in p_values.items() if p is not None]
    ordered = sorted(usable, key=lambda item: item[1])
    m = len(ordered) if family_size is None else family_size
    if m < len(ordered):
        raise ValueError("family_size cannot be smaller than the number of non-missing p-values")
    adjusted: dict[str, dict[str, float | bool | None]] = {
        name: {"raw_p": p, "adjusted_p": None, "alpha_bar": None, "confirmed": False}
        for name, p in p_values.items()
    }
    running_adjusted = 0.0
    still_confirmed = True
    for rank, (name, p_value) in enumerate(ordered, start=1):
        factor = m - rank + 1
        alpha_bar = alpha / factor
        running_adjusted = max(running_adjusted, min(1.0, p_value * factor))
        confirmed = still_confirmed and p_value <= alpha_bar
        if not confirmed:
            still_confirmed = False
        adjusted[name] = {
            "raw_p": float(p_value),
            "adjusted_p": float(running_adjusted),
            "alpha_bar": float(alpha_bar),
            "confirmed": bool(confirmed),
        }
    return adjusted


def risk_difference_and_odds_ratio(
    group_a: Iterable[bool | int | float],
    group_b: Iterable[bool | int | float],
    *,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | None]:
    a = np.asarray([float(value) for value in group_a], dtype=float)
    b = np.asarray([float(value) for value in group_b], dtype=float)
    if a.size == 0 or b.size == 0:
        return {
            "risk_difference": None,
            "risk_difference_ci_low": None,
            "risk_difference_ci_high": None,
            "odds_ratio": None,
            "odds_ratio_ci_low": None,
            "odds_ratio_ci_high": None,
        }
    rd = float(np.mean(a) - np.mean(b))
    odds = _odds_ratio(float(np.sum(a)), float(a.size - np.sum(a)), float(np.sum(b)), float(b.size - np.sum(b)))
    rd_draws = []
    or_draws = []
    for _ in range(n_bootstrap):
        sample_a = a[rng.integers(0, a.size, size=a.size)]
        sample_b = b[rng.integers(0, b.size, size=b.size)]
        rd_draws.append(float(np.mean(sample_a) - np.mean(sample_b)))
        or_draws.append(
            _odds_ratio(
                float(np.sum(sample_a)),
                float(sample_a.size - np.sum(sample_a)),
                float(np.sum(sample_b)),
                float(sample_b.size - np.sum(sample_b)),
            )
        )
    rd_low, rd_high = np.quantile(rd_draws, [0.025, 0.975])
    or_low, or_high = np.quantile(or_draws, [0.025, 0.975])
    return {
        "risk_difference": rd,
        "risk_difference_ci_low": float(rd_low),
        "risk_difference_ci_high": float(rd_high),
        "odds_ratio": odds,
        "odds_ratio_ci_low": float(or_low),
        "odds_ratio_ci_high": float(or_high),
    }


def precision_classification(
    effect: float | None,
    interval: Interval | tuple[float | None, float | None] | None,
    *,
    confirmatory_test: bool = False,
) -> str:
    if confirmatory_test:
        return "confirmation"
    if interval is not None:
        low, high = _interval_parts(interval)
        if low is not None and high is not None and (high - low) <= PRECISION_TARGET_MAX_WIDTH_PTS / 100:
            return "precision-bounded"
    return "estimation"


def _odds_ratio(a_success: float, a_fail: float, b_success: float, b_fail: float) -> float:
    # Haldane-Anscombe correction keeps zero cells finite and deterministic.
    return float(((a_success + 0.5) / (a_fail + 0.5)) / ((b_success + 0.5) / (b_fail + 0.5)))


def _interval_parts(interval: Interval | tuple[float | None, float | None]) -> tuple[float | None, float | None]:
    if isinstance(interval, Interval):
        return interval.low, interval.high
    return interval


def _bootstrap_t_standard_error(data: np.ndarray, statistic: Callable[[np.ndarray], float]) -> float:
    if data.size < 2:
        return 0.0
    if statistic(data) == float(np.mean(data)):
        return float(np.std(data, ddof=1) / np.sqrt(data.size))
    jackknife = np.asarray(
        [statistic(np.delete(data, index)) for index in range(data.size)],
        dtype=float,
    )
    if jackknife.size < 2:
        return 0.0
    return float(np.sqrt((data.size - 1) / data.size * np.sum((jackknife - np.mean(jackknife)) ** 2)))


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False
