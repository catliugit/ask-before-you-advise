from __future__ import annotations

from typing import Any

import pandas as pd


def gap_accounting(df: pd.DataFrame) -> dict[str, Any]:
    total = int(len(df))
    void_count = _void_count(df)
    non_missing = _non_missing(df)
    missing = _missing_cells(df)
    rerun_count = int(df.get("rerun_due_to_persona_leak", pd.Series(False, index=df.index)).fillna(False).astype(bool).sum()) if total else 0
    double_leak = df[df.get("rerun_count", pd.Series(0, index=df.index)).fillna(0).astype(int) >= 2] if total else df.iloc[0:0]
    return {
        "void_verdicts": {
            "n": void_count,
            "denominator": len(non_missing),
            "value": void_count / len(non_missing) if len(non_missing) else None,
            "status": "ok",
            "evidence_class": "descriptive",
            "dimension_void_counts": _dimension_void_counts(non_missing),
        },
        "missing_cells": {
            "n": len(missing),
            "denominator": total,
            "value": len(missing) / total if total else None,
            "status": "ok",
            "evidence_class": "descriptive",
            "cells": missing,
        },
        "persona_leak_reruns": {
            "n": rerun_count,
            "denominator": total,
            "value": rerun_count / total if total else None,
            "status": "ok",
            "evidence_class": "descriptive",
            "rerun_count_ge_2": [_row_key(row) for _, row in double_leak.iterrows()],
        },
    }


def _void_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    data = _non_missing(df)
    if data.empty:
        return 0
    outcome_void = data.get("outcome_void", pd.Series(False, index=data.index)).fillna(False).astype(bool)
    null_grade = data.get("outcome_grade", pd.Series(None, index=data.index)).isna()
    scoring_failed = data.get("scoring_failed", pd.Series(False, index=data.index)).fillna(False).astype(bool)
    return int((outcome_void | null_grade | scoring_failed).sum())


def _dimension_void_counts(df: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(df[column].fillna(False).astype(bool).sum())
        for column in sorted(df.columns)
        if column.startswith("dim_") and column.endswith("_void")
    }


def _non_missing(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "call_status" not in df:
        return df
    return df[df["call_status"] != "missing"]


def _missing_cells(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty or "call_status" not in df:
        return []
    return [_row_key(row) for _, row in df[df["call_status"] == "missing"].iterrows()]


def _row_key(row: pd.Series) -> dict[str, Any]:
    repeat = row.get("repeat")
    return {
        "episode_id": row.get("episode_id"),
        "model": row.get("model"),
        "scenario": row.get("scenario"),
        "module": row.get("module"),
        "variant": row.get("variant"),
        "repeat": int(repeat) if repeat is not None and not pd.isna(repeat) else None,
    }
