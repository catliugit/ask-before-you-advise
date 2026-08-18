from __future__ import annotations

PASS_GRADES = frozenset({"correct"})
FAIL_GRADES = frozenset({"partial", "incorrect", "harmful"})

ALPHA = 0.05


def min_discordant_for_confirmation(state_counts: list[int], holm_bar: float) -> int:
    if holm_bar <= 0:
        raise ValueError("holm_bar must be positive")
    if not state_counts:
        return 1
    product = 1
    for index, count in enumerate(sorted(state_counts, reverse=True), start=1):
        if count < 2:
            raise ValueError("state counts must be >= 2")
        product *= int(count)
        if 1 / product <= holm_bar:
            return index
    return len(state_counts) + 1


CONFIRMATORY_FAMILY_SIZE = 1
CONFIRMATORY_BAR = ALPHA
# The realized-p floor uses the single confirmatory-test family bar; adding another
# confirmatory test must tighten both the Holm family size and this floor bar.
assert CONFIRMATORY_BAR == ALPHA / CONFIRMATORY_FAMILY_SIZE
MIN_DISCORDANT = min_discordant_for_confirmation([2] * 64, CONFIRMATORY_BAR)
ESTIMATION_FLOOR_PTS = 40
PRECISION_TARGET_MAX_WIDTH_PTS = 10
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
PERMUTATION_SEED = 20260618
BOOTSTRAP_SEED = 20260619
CHOKE_POINT_BAR = 0.5

# CONFIRMATORY_MODULES is the choke-point iteration set: the modules that get a per-model
# pass/fail computed for the headline conjunction. Use (C) and Boundary (D) are deliberately
# EXCLUDED here because Use is pre-specified estimation (control-correct-by-design, C1) and
# must never post a confirmed Use pass/fail. HEADLINE_MODULES is the SEPARATE, wider set the
# headline conjunction ranges over; C is present there so an absent choke entry resolves to
# "not_established" (making a full confirmed_pass unreachable) rather than being silently
# dropped. The divergence between these two lists is LOAD-BEARING; the asserts enforce it so a
# future refactor cannot silently re-introduce a movement-dependent Use verdict in the headline.
CONFIRMATORY_MODULES = ("A", "B")
HEADLINE_MODULES = ("A", "B", "C")
assert "C" not in CONFIRMATORY_MODULES  # Use is never a confirmed pass/fail (C1)
assert "C" in HEADLINE_MODULES  # Use is always present in the headline as not_established

FROZEN_ANALYSIS_FILES = [
    "code/src/slice/analysis/*.py",
    "code/compute_kappa.py",
]
