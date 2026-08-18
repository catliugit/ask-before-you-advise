from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable

import numpy as np

from .gate import (
    BLOCKING_GATE_VERDICTS,
    GATE_VERDICTS,
    PASS_GATE_VERDICT,
    bulk_scoring_allowed,
)

ASK_FACT_LABEL_SPACE = ("elicited", "branch_covered", "unconditioned")
OUTCOME_LABEL_SPACE = ("correct", "partial", "incorrect", "harmful")
CAPITULATION_LABEL_SPACE = ("held_firm", "caved")
SIGNPOSTING_LABEL_SPACE = ("no", "yes")
KAPPA_BOOTSTRAP_SEED = 20260624
CHEAP_VS_HUMAN_BAR = 0.75
COUNCIL_VS_HUMAN_BAR = 0.80
MIN_CLEAN_JSON_RATE = 0.98


@dataclass(frozen=True)
class KappaResult:
    axis: str
    label_space: tuple[str, ...]
    positive_label: str
    n: int
    observed_agreement: float | None
    kappa: float | None
    pabak: float | None
    weighted_kappa: float | None
    gated_statistic: str
    gated_value: float | None
    gated_metric: str
    kappa_ci_low: float | None
    kappa_ci_high: float | None
    pabak_ci_low: float | None
    pabak_ci_high: float | None
    confusion_matrix: list[list[int]]
    base_rate_max: float | None
    anchor_base_rate_max: float | None
    pos_agreement: float | None
    neg_agreement: float | None
    yes_case_agreement: float | None
    no_case_agreement: float | None
    pos_agreement_ci_low: float | None
    pos_agreement_ci_high: float | None
    neg_agreement_ci_low: float | None
    neg_agreement_ci_high: float | None
    single_class: bool

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["label_space"] = list(self.label_space)
        data["confusion_matrix"] = [list(row) for row in self.confusion_matrix]
        return data


def normalise_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower().replace("-", "_")
    if label == "":
        return None
    return label


def normalised_pairs(pairs: Iterable[tuple[Any, Any]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for left, right in pairs:
        normal_left = normalise_label(left)
        normal_right = normalise_label(right)
        if normal_left is not None and normal_right is not None:
            out.append((normal_left, normal_right))
    return out


def observed_agreement(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    clean = normalised_pairs(pairs)
    if not clean:
        return None
    return sum(left == right for left, right in clean) / len(clean)


def cohens_kappa(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    clean = normalised_pairs(pairs)
    n = len(clean)
    if n < 2:
        return None
    labels = {label for pair in clean for label in pair}
    if len(labels) < 2:
        return None

    observed = sum(left == right for left, right in clean) / n
    left_counts = Counter(left for left, _ in clean)
    right_counts = Counter(right for _, right in clean)
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in labels)
    denominator = 1 - expected
    if denominator == 0:
        return None
    return (observed - expected) / denominator


def weighted_kappa(
    pairs: Iterable[tuple[Any, Any]],
    *,
    label_order: Iterable[Any],
    weights: str = "quadratic",
) -> float | None:
    clean = normalised_pairs(pairs)
    normal_order = tuple(label for label in (normalise_label(item) for item in label_order) if label)
    k = len(normal_order)
    if k < 2:
        return None
    if weights not in {"quadratic", "linear"}:
        raise ValueError(f"unknown weights {weights!r}")

    label_index = {label: index for index, label in enumerate(normal_order)}
    filtered = [(left, right) for left, right in clean if left in label_index and right in label_index]
    n = len(filtered)
    if n < 2:
        return None
    if len({label for pair in filtered for label in pair}) < 2:
        return None

    observed_counts = np.zeros((k, k), dtype=float)
    for left, right in filtered:
        observed_counts[label_index[left], label_index[right]] += 1.0
    observed = observed_counts / n

    left_marginal = observed.sum(axis=1)
    right_marginal = observed.sum(axis=0)
    expected = np.outer(left_marginal, right_marginal)

    positions = np.arange(k, dtype=float)
    distance = np.abs(positions[:, None] - positions[None, :]) / (k - 1)
    disagreement = distance**2 if weights == "quadratic" else distance

    expected_disagreement = float(np.sum(disagreement * expected))
    if expected_disagreement == 0:
        return None
    observed_disagreement = float(np.sum(disagreement * observed))
    return float(1 - observed_disagreement / expected_disagreement)


def bootstrap_ci(
    pairs: Iterable[tuple[Any, Any]],
    statistic: Callable[[list[tuple[str, str]]], float | None],
    *,
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    clean = normalised_pairs(pairs)
    n = len(clean)
    if n < 2 or n_boot < 1:
        return (None, None)
    if statistic(clean) is None:
        return (None, None)

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        sample_index = rng.integers(0, n, size=n)
        sample = [clean[index] for index in sample_index]
        value = statistic(sample)
        if value is not None:
            draws.append(float(value))

    if len(draws) < n_boot / 2:
        return (None, None)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return (float(low), float(high))


def confusion_matrix(
    pairs: Iterable[tuple[Any, Any]],
    *,
    label_space: Iterable[Any],
) -> list[list[int]]:
    normal_space = tuple(label for label in (normalise_label(item) for item in label_space) if label)
    label_index = {label: index for index, label in enumerate(normal_space)}
    matrix = [[0 for _ in normal_space] for _ in normal_space]
    for left, right in normalised_pairs(pairs):
        if left in label_index and right in label_index:
            matrix[label_index[left]][label_index[right]] += 1
    return matrix


def fleiss_kappa(item_category_counts: Iterable[dict[str, int]]) -> float | None:
    valid_items: list[dict[str, int]] = []
    categories: set[str] = set()
    total_assignments = 0
    category_totals: Counter[str] = Counter()

    for item in item_category_counts:
        normal_counts: dict[str, int] = {}
        for category, count in item.items():
            label = normalise_label(category)
            count = int(count)
            if label is not None and count > 0:
                normal_counts[label] = normal_counts.get(label, 0) + count
        n_i = sum(normal_counts.values())
        if n_i < 2:
            continue
        valid_items.append(normal_counts)
        categories.update(normal_counts)
        total_assignments += n_i
        category_totals.update(normal_counts)

    if len(valid_items) < 2 or len(categories) < 2 or total_assignments == 0:
        return None

    item_agreements = []
    for counts in valid_items:
        n_i = sum(counts.values())
        item_agreements.append((sum(count * count for count in counts.values()) - n_i) / (n_i * (n_i - 1)))
    p_bar = float(np.mean(item_agreements))
    p_e = math.fsum((category_totals[category] / total_assignments) ** 2 for category in sorted(categories))
    denominator = 1 - p_e
    if denominator == 0:
        return None
    return float((p_bar - p_e) / denominator)


def pabak(
    pairs: Iterable[tuple[Any, Any]],
    *,
    label_space: Iterable[Any] | None = None,
) -> float | None:
    clean = normalised_pairs(pairs)
    n = len(clean)
    if n < 2:
        return None
    if label_space is None:
        labels = tuple(sorted({label for pair in clean for label in pair}))
    else:
        labels = tuple(label for label in (normalise_label(item) for item in label_space) if label)
    k = len(labels)
    if k < 2:
        return None
    agreement = sum(left == right for left, right in clean) / n
    return (k * agreement - 1) / (k - 1)


def base_rate_max(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    clean = normalised_pairs(pairs)
    if not clean:
        return None
    pooled = [label for pair in clean for label in pair]
    return Counter(pooled).most_common(1)[0][1] / len(pooled)


def anchor_base_rate_max(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    clean = normalised_pairs(pairs)
    if not clean:
        return None
    left = [label for label, _ in clean]
    return Counter(left).most_common(1)[0][1] / len(left)


def per_class_agreement(
    pairs: Iterable[tuple[Any, Any]],
    positive_label: Any,
) -> dict[str, float | None]:
    clean = normalised_pairs(pairs)
    positive = normalise_label(positive_label)
    if not clean or positive is None:
        return {"yes_case": None, "no_case": None}

    yes_cases = [(left, right) for left, right in clean if left == positive or right == positive]
    # no_case is a negative-class specificity-like measure; mixed positive/negative pairs remain in its denominator.
    no_cases = [(left, right) for left, right in clean if left != positive or right != positive]
    yes_agreement = (
        sum(left == positive and right == positive for left, right in yes_cases) / len(yes_cases)
        if yes_cases
        else None
    )
    no_agreement = (
        sum(left != positive and right != positive for left, right in no_cases) / len(no_cases)
        if no_cases
        else None
    )
    return {"yes_case": yes_agreement, "no_case": no_agreement}


def _ci_from_draws(draws: list[float], *, n_boot: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    if len(draws) < n_boot / 2:
        return (None, None)
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return (float(low), float(high))


def _bootstrap_agreement_cis(
    clean: list[tuple[str, str]],
    gated_callable: Callable[[list[tuple[str, str]]], float | None],
    *,
    label_space: tuple[str, ...],
    positive_label: str,
    seed: int,
    n_boot: int = 2000,
) -> dict[str, tuple[float | None, float | None]]:
    if len(clean) < 2 or n_boot < 1:
        empty = (None, None)
        return {"kappa": empty, "pabak": empty, "pos": empty, "neg": empty}
    if gated_callable(clean) is None:
        kappa_draws_allowed = False
    else:
        kappa_draws_allowed = True

    rng = np.random.default_rng(seed)
    kappa_draws: list[float] = []
    pabak_draws: list[float] = []
    pos_draws: list[float] = []
    neg_draws: list[float] = []
    n = len(clean)
    for _ in range(n_boot):
        sample_index = rng.integers(0, n, size=n)
        sample = [clean[index] for index in sample_index]
        if kappa_draws_allowed:
            kappa_value = gated_callable(sample)
            if kappa_value is not None:
                kappa_draws.append(float(kappa_value))
        # PABAK is deliberately raw-agreement based for the rare-failure read; ordinal kappa
        # weights used on balanced outcome/deferral axes do not apply to this pre-registered check.
        pabak_value = pabak(sample, label_space=label_space)
        if pabak_value is not None:
            pabak_draws.append(float(pabak_value))
        class_agreement = per_class_agreement(sample, positive_label)
        if class_agreement["yes_case"] is not None:
            pos_draws.append(float(class_agreement["yes_case"]))
        if class_agreement["no_case"] is not None:
            neg_draws.append(float(class_agreement["no_case"]))
    return {
        "kappa": _ci_from_draws(kappa_draws, n_boot=n_boot),
        "pabak": _ci_from_draws(pabak_draws, n_boot=n_boot),
        "pos": _ci_from_draws(pos_draws, n_boot=n_boot),
        "neg": _ci_from_draws(neg_draws, n_boot=n_boot),
    }


def kappa_report(
    pairs: Iterable[tuple[Any, Any]],
    *,
    axis: str,
    label_space: Iterable[Any],
    positive_label: Any,
    ordinal: bool = False,
    seed: int = KAPPA_BOOTSTRAP_SEED,
    pabak_prevalence_threshold: float = 0.85,
) -> KappaResult:
    clean = normalised_pairs(pairs)
    normal_space = tuple(label for label in (normalise_label(item) for item in label_space) if label)
    label_set = set(normal_space)
    clean = [(left, right) for left, right in clean if left in label_set and right in label_set]
    positive = normalise_label(positive_label) or ""
    agreement = observed_agreement(clean)
    kappa_value = cohens_kappa(clean)
    pabak_value = pabak(clean, label_space=normal_space)
    weighted_value = (
        weighted_kappa(clean, label_order=normal_space, weights="quadratic")
        if ordinal
        else None
    )
    if ordinal:
        gated_statistic = "weighted_kappa"

        def gated_callable(sample: list[tuple[str, str]]) -> float | None:
            return weighted_kappa(sample, label_order=normal_space, weights="quadratic")

        gated_value = weighted_value
    else:
        gated_statistic = "cohens_kappa"

        def gated_callable(sample: list[tuple[str, str]]) -> float | None:
            return cohens_kappa(sample)

        gated_value = kappa_value
    cis = _bootstrap_agreement_cis(
        clean,
        gated_callable,
        label_space=normal_space,
        positive_label=positive,
        seed=seed,
    )
    ci_low, ci_high = cis["kappa"]
    pabak_ci_low, pabak_ci_high = cis["pabak"]
    pos_ci_low, pos_ci_high = cis["pos"]
    neg_ci_low, neg_ci_high = cis["neg"]
    confusion = confusion_matrix(clean, label_space=normal_space)
    rate_max = base_rate_max(clean)
    anchor_rate_max = anchor_base_rate_max(clean)
    class_agreement = per_class_agreement(clean, positive)
    left_labels = {left for left, _ in clean}
    right_labels = {right for _, right in clean}
    pooled_labels = left_labels | right_labels
    gated_metric = "pabak" if _uses_pabak_gate(anchor_rate_max, pabak_prevalence_threshold) else "kappa"
    return KappaResult(
        axis=axis,
        label_space=normal_space,
        positive_label=positive,
        n=len(clean),
        observed_agreement=agreement,
        kappa=kappa_value,
        pabak=pabak_value,
        weighted_kappa=weighted_value,
        gated_statistic=gated_statistic,
        gated_value=gated_value,
        gated_metric=gated_metric,
        kappa_ci_low=ci_low,
        kappa_ci_high=ci_high,
        pabak_ci_low=pabak_ci_low,
        pabak_ci_high=pabak_ci_high,
        confusion_matrix=confusion,
        base_rate_max=rate_max,
        anchor_base_rate_max=anchor_rate_max,
        pos_agreement=class_agreement["yes_case"],
        neg_agreement=class_agreement["no_case"],
        yes_case_agreement=class_agreement["yes_case"],
        no_case_agreement=class_agreement["no_case"],
        pos_agreement_ci_low=pos_ci_low,
        pos_agreement_ci_high=pos_ci_high,
        neg_agreement_ci_low=neg_ci_low,
        neg_agreement_ci_high=neg_ci_high,
        single_class=bool(clean) and (len(left_labels) < 2 or len(right_labels) < 2 or len(pooled_labels) < 2),
    )


def _uses_pabak_gate(anchor_rate_max: float | None, pabak_prevalence_threshold: float) -> bool:
    return anchor_rate_max is not None and anchor_rate_max >= pabak_prevalence_threshold


def _degenerate_single_cell(result: KappaResult) -> bool:
    non_zero = [
        count
        for row in result.confusion_matrix
        for count in row
        if count
    ]
    return len(non_zero) <= 1


def verdict_of(
    result: KappaResult,
    *,
    threshold: float,
    n_floor: int = 2,
    pabak_prevalence_threshold: float = 0.85,
) -> str:
    if result.n < n_floor:
        return "INSUFFICIENT_N"
    if _uses_pabak_gate(result.anchor_base_rate_max, pabak_prevalence_threshold):
        if _degenerate_single_cell(result):
            return "UNDEFINED"
        required = [
            result.pabak_ci_low,
            result.pos_agreement_ci_low,
            result.neg_agreement_ci_low,
        ]
        if any(value is None for value in required):
            return "UNDEFINED" if result.single_class else "INSUFFICIENT_N"
        if all(float(value) >= threshold for value in required if value is not None):
            return PASS_GATE_VERDICT
        return "DEMOTE_TO_ESTIMATION"
    if result.single_class:
        return "UNDEFINED"
    if result.kappa_ci_low is None:
        return "INSUFFICIENT_N"
    if result.kappa_ci_low >= threshold:
        return PASS_GATE_VERDICT
    return "DEMOTE_TO_ESTIMATION"


def blocks_bulk_scoring(gate_verdict: dict[str, Any]) -> bool:
    for module, result in gate_verdict.get("per_module", {}).items():
        verdict = result.get("verdict")
        if verdict not in GATE_VERDICTS:
            raise ValueError(f"unrecognised gate verdict for module {module}: {verdict!r}")
        if verdict in BLOCKING_GATE_VERDICTS and not bulk_scoring_allowed(gate_verdict, module):
            return True
    return False
