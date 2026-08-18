from __future__ import annotations

from slice.phase_roles import (
    duplicate_source_code,
    is_duplicate_item,
    is_h0_item,
    is_h1_item,
    masked_source_code,
)


def test_handcode_pack_duplicate_helpers_classify_codes():
    duplicate_map = {"T1111111111": "T2222222222"}

    assert is_duplicate_item("T1111111111", duplicate_map) is True
    assert is_h0_item("T1111111111", duplicate_map) is False
    assert duplicate_source_code("T1111111111", duplicate_map) == "T2222222222"

    assert is_duplicate_item("T2222222222", duplicate_map) is False
    assert is_h0_item("T2222222222", duplicate_map) is True
    assert duplicate_source_code("T2222222222", duplicate_map) is None

    assert is_duplicate_item("T1111111111") is False
    assert is_h0_item("T1111111111") is True
    assert duplicate_source_code("T1111111111") is None
    assert is_duplicate_item("T1111111111", {}) is False
    assert is_h0_item("T1111111111", {}) is True
    assert duplicate_source_code("T1111111111", {}) is None


def test_masked_review_h1_helpers_classify_codes():
    masked_map = {"M1111111111": "T2222222222"}

    assert is_h1_item("M1111111111", masked_map) is True
    assert masked_source_code("M1111111111", masked_map) == "T2222222222"

    assert is_h1_item("M3333333333", masked_map) is False
    assert masked_source_code("M3333333333", masked_map) is None
    assert is_h1_item("M1111111111") is False
    assert masked_source_code("M1111111111") is None
    assert is_h1_item("M1111111111", {}) is False
    assert masked_source_code("M1111111111", {}) is None
