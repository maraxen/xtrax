"""AC9: config_hash stability and error-safety tests."""

from __future__ import annotations

import datetime

from xtrax.cli.hash import config_hash


def test_same_keys_different_order_hash_equal() -> None:
    """AC9 pin: sort_keys makes key ordering irrelevant."""
    d1 = {"b": 2, "a": 1, "c": {"z": 9, "y": 8}}
    d2 = {"a": 1, "c": {"y": 8, "z": 9}, "b": 2}
    assert config_hash(d1) == config_hash(d2)


def test_different_values_hash_differ() -> None:
    d1 = {"a": 1}
    d2 = {"a": 2}
    assert config_hash(d1) != config_hash(d2)


def test_datetime_value_hashes_without_error() -> None:
    """AC9 pin: default=str handles TOML datetimes (non-JSON-native)."""
    d = {"created": datetime.datetime(2026, 6, 23), "lr": 1e-3}
    result = config_hash(d)
    assert isinstance(result, str)
    assert len(result) == 12


def test_nested_kwargs_hash_without_error() -> None:
    """AC9 pin: nested kwargs dict hashes correctly."""
    d = {
        "schema_version": 1,
        "model": {"path": "pkg:Model", "kwargs": {"hidden": 128}},
        "optimizer": {"path": "pkg:opt", "kwargs": {"lr": 1e-3}},
    }
    result = config_hash(d)
    assert isinstance(result, str)
    assert len(result) == 12


def test_hash_length_is_12() -> None:
    result = config_hash({"x": 1})
    assert len(result) == 12


def test_hash_is_lowercase_hex() -> None:
    result = config_hash({"x": 1})
    assert all(c in "0123456789abcdef" for c in result)
