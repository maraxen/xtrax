"""Tests for xtrax.run.zarr_integrity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr

from xtrax.run.zarr_integrity import (
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    fsync_tree,
    normalize_json_value,
    zarr_content_digest,
)


def _make_store(tmp_path: Path, name: str = "test.zarr") -> Path:
    store_path = tmp_path / name
    root = zarr.open_group(str(store_path), mode="a")
    arr = root.create_array(name="data", shape=(3,), dtype="int32")
    arr[...] = np.array([1, 2, 3], dtype=np.int32)
    arr.attrs["label"] = "alpha"
    return store_path


def test_canonical_json_bytes_is_deterministic_and_sorted() -> None:
    a = canonical_json_bytes({"b": 1, "a": 2})
    b = canonical_json_bytes({"a": 2, "b": 1})
    assert a == b
    assert a == b'{"a":2,"b":1}'


def test_normalize_json_value_handles_numpy_scalars_and_arrays() -> None:
    assert normalize_json_value(np.int32(5)) == 5
    assert normalize_json_value(np.array([1, 2, 3])) == [1, 2, 3]
    assert normalize_json_value(b"hi") == "hi"
    assert normalize_json_value({"z": 1, "a": 2}) == {"a": 2, "z": 1}


def test_digest_is_deterministic(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    assert zarr_content_digest(store_path) == zarr_content_digest(store_path)


def test_digest_changes_when_array_data_changes(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    original = zarr_content_digest(store_path)

    root = zarr.open_group(str(store_path), mode="a")
    root["data"][...] = np.array([9, 9, 9], dtype=np.int32)

    assert zarr_content_digest(store_path) != original


def test_digest_changes_when_attrs_change(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    original = zarr_content_digest(store_path)

    root = zarr.open_group(str(store_path), mode="a")
    root["data"].attrs["label"] = "beta"

    assert zarr_content_digest(store_path) != original


def test_digest_covers_nested_groups(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    original = zarr_content_digest(store_path)

    root = zarr.open_group(str(store_path), mode="a")
    sub = root.require_group("nested")
    arr = sub.create_array(name="extra", shape=(1,), dtype="int32")
    arr[...] = np.array([42], dtype=np.int32)

    assert zarr_content_digest(store_path) != original


def test_digest_stable_across_reopen(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    first = zarr_content_digest(store_path)
    second = zarr_content_digest(store_path)
    assert first == second


def test_digest_raises_import_error_without_zarr(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    with (
        patch.dict("sys.modules", {"zarr": None}),
        pytest.raises(ImportError, match="xtrax\\[io\\]"),
    ):
        zarr_content_digest(store_path)


def test_fsync_tree_does_not_raise_on_real_store(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    fsync_tree(store_path)  # should not raise


def test_fsync_tree_syncs_every_file_and_directory(tmp_path: Path) -> None:
    store_path = _make_store(tmp_path)
    root = zarr.open_group(str(store_path), mode="a")
    sub = root.require_group("nested")
    arr = sub.create_array(name="extra", shape=(1,), dtype="int32")
    arr[...] = np.array([1], dtype=np.int32)

    all_files = [p for p in store_path.rglob("*") if p.is_file()]
    all_dirs = [p for p in store_path.rglob("*") if p.is_dir()]
    assert all_files, "fixture should have produced at least one chunk/metadata file"

    with (
        patch("xtrax.run.zarr_integrity.fsync_file") as mock_fsync_file,
        patch("xtrax.run.zarr_integrity.fsync_directory") as mock_fsync_dir,
    ):
        fsync_tree(store_path)
        synced_files = {call.args[0] for call in mock_fsync_file.call_args_list}
        synced_dirs = {call.args[0] for call in mock_fsync_dir.call_args_list}
        assert synced_files == set(all_files)
        assert synced_dirs == {*all_dirs, store_path}


def test_fsync_file_and_directory_do_not_raise(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hi")
    fsync_file(f)
    fsync_directory(tmp_path)
