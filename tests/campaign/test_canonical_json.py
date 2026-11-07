"""Tests for canonical JSON serialization and normalization in xtrax.campaign."""

import math
import unicodedata

import numpy as np
import pytest

from xtrax.campaign.manifest import canonical_json_bytes, normalize_json_value


def test_float_negative_zero_normalization() -> None:
    # Python floats
    norm_neg = normalize_json_value(-0.0)
    norm_pos = normalize_json_value(0.0)
    assert math.copysign(1.0, norm_neg) == 1.0
    assert math.copysign(1.0, norm_pos) == 1.0
    assert canonical_json_bytes({"val": -0.0}) == canonical_json_bytes({"val": 0.0})

    # NumPy floating
    np_neg32 = normalize_json_value(np.float32(-0.0))
    np_pos32 = normalize_json_value(np.float32(0.0))
    assert math.copysign(1.0, np_neg32) == 1.0
    assert math.copysign(1.0, np_pos32) == 1.0
    assert canonical_json_bytes({"val": np.float32(-0.0)}) == canonical_json_bytes({"val": 0.0})

    np_neg64 = normalize_json_value(np.float64(-0.0))
    assert math.copysign(1.0, np_neg64) == 1.0
    assert canonical_json_bytes({"val": np.float64(-0.0)}) == canonical_json_bytes({"val": 0.0})


def test_mixed_key_sorting_without_type_error() -> None:
    # In Python 3, sorted on mixed-type keys raises TypeError unless key=lambda kv: str(kv[0])
    mixed_dict = {10: "int_key", "2": "str_key", "apple": "str_key_2"}
    normalized = normalize_json_value(mixed_dict)
    assert list(normalized.keys()) == ["10", "2", "apple"]

    b1 = canonical_json_bytes({1: "val", "2": "val2"})
    b2 = canonical_json_bytes({"2": "val2", 1: "val"})
    assert b1 == b2


def test_rejection_of_non_finite_floats() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"nan_val": float("nan")})

    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"inf_val": float("inf")})

    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json_bytes({"neg_inf_val": float("-inf")})


def test_unicode_nfc_normalization() -> None:
    composed = "café"  # \u00e9
    decomposed = "cafe\u0301"  # e + combining acute accent
    assert composed != decomposed
    assert unicodedata.normalize("NFC", decomposed) == composed

    norm_c = normalize_json_value(composed)
    norm_d = normalize_json_value(decomposed)
    assert norm_c == norm_d

    b_c = canonical_json_bytes({"word": composed})
    b_d = canonical_json_bytes({"word": decomposed})
    assert b_c == b_d


def test_bytes_and_numpy_types_normalization() -> None:
    raw_bytes = b"sample_bytes"
    assert normalize_json_value(raw_bytes) == "sample_bytes"

    np_arr = np.array([[-0.0, 1.0], [2.0, 3.0]])
    norm_arr = normalize_json_value(np_arr)
    assert norm_arr == [[0.0, 1.0], [2.0, 3.0]]
    assert math.copysign(1.0, norm_arr[0][0]) == 1.0

    np_int = np.int64(42)
    assert normalize_json_value(np_int) == 42


def test_canonical_json_formatting() -> None:
    data = {"b": 2, "a": 1, "c": [1, 2]}
    raw = canonical_json_bytes(data)
    # Strictly sorted keys, separators (',', ':'), no trailing newline
    assert raw == b'{"a":1,"b":2,"c":[1,2]}'
