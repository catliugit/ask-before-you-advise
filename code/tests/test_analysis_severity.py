from __future__ import annotations

import numpy as np
import pandas as pd

from slice.analysis.severity import compute_severity_concentration


def test_severity_rates_and_second_derivation_match_rate():
    items = pd.DataFrame(
        [
            {"scenario": "s1", "severity": "critical", "severity_second_derivation": "critical", "failed": True},
            {"scenario": "s2", "severity": "minor", "severity_second_derivation": "minor", "failed": False},
        ]
    )
    result = compute_severity_concentration(items, rng_bootstrap=np.random.default_rng(1), n_bootstrap=40)
    assert result["by_severity"]["critical"]["value"] == 1.0
    assert result["second_derivation"]["match_rate"] == 1.0
    assert result["evidence_class"] == "estimation"


def test_severity_missing_second_derivation_suppresses_to_descriptive():
    items = pd.DataFrame(
        [{"scenario": "s1", "severity": "critical", "severity_second_derivation": None, "failed": True}]
    )
    result = compute_severity_concentration(items, rng_bootstrap=np.random.default_rng(1), n_bootstrap=40)
    assert result["status"] == "suppressed_missing_second_derivation"
    assert result["evidence_class"] == "descriptive"
    assert result["by_severity"]["critical"]["status"] == "suppressed_missing_second_derivation"
    assert result["by_severity"]["critical"]["evidence_class"] == "descriptive"
    assert result["serious_or_critical_vs_rest"]["status"] == "suppressed_missing_second_derivation"
    assert result["serious_or_critical_vs_rest"]["evidence_class"] == "descriptive"
