from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def stable_json(data: Any) -> str:
    return json.dumps(_normalise(data), ensure_ascii=False, indent=2, sort_keys=True)


def stable_hash(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def write_outputs(data_root: Path, results: dict[str, Any]) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    manifest = dict(payload.get("run_manifest", {}))
    manifest["results_hash"] = stable_hash({key: value for key, value in payload.items() if key != "run_manifest"})
    payload["run_manifest"] = manifest
    text = stable_json(payload)
    (data_root / "analysis_results.json").write_text(text + "\n")
    (data_root / "metrics.json").write_text(text + "\n")
    (data_root / "RESULTS.md").write_text(results_markdown(payload))


def results_markdown(results: dict[str, Any]) -> str:
    headline = results.get("confirmatory", {}).get("headline", {})
    headline_counts = headline.get("aggregate_counts", {})
    confirmatory_keys = ", ".join(key for key in results.get("confirmatory", {}) if key not in {"headline", "holm", "holm_note"}) or "none"
    estimation_keys = ", ".join(results.get("estimation", {}).keys()) or "none"
    descriptive_keys = ", ".join(results.get("descriptive", {}).keys()) or "none"
    lines = [
        "# Results",
        "",
        "## Confirmatory",
        "",
        "Headline counts: "
        f"confirmed_fail={headline_counts.get('confirmed_fail', 0)}, "
        f"confirmed_pass={headline_counts.get('confirmed_pass', 0)}, "
        f"not_established={headline_counts.get('not_established', 0)}",
        f"Measures: {confirmatory_keys}",
        f"Holm note: {results.get('confirmatory', {}).get('holm_note')}",
        "",
        "## Estimation",
        "",
        f"Measures: {estimation_keys}",
        f"Specification Gap: {results.get('estimation', {}).get('specification_gap', {}).get('value')}",
        "",
        "## Descriptive",
        "",
        f"Measures: {descriptive_keys}",
        f"PRR: {results.get('descriptive', {}).get('ask', {}).get('prr', {}).get('value')}",
        f"OTR: {results.get('descriptive', {}).get('ask', {}).get('otr', {}).get('value')}",
        "",
    ]
    return "\n".join(lines)


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalise(inner) for key, inner in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalise(inner) for inner in value]
    if isinstance(value, tuple):
        return [_normalise(inner) for inner in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value) and value is not None and not isinstance(value, (str, bytes)):
        return None
    return value
