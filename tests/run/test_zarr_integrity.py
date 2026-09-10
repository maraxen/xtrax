"""Tests for xtrax.run.zarr_integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr

from xtrax.run import SinkSpec, ZarrStagingSink
from xtrax.run.zarr_integrity import (
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    fsync_tree,
    normalize_json_value,
    update_zarr_node_digest,
    zarr_content_digest,
)


def _make_store(tmp_path: Path, name: str = "test.zarr") -> Path:
    store_path = tmp_path / name
    root = zarr.open_group(str(store_path), mode="a")
    arr = root.create_array(name="data", shape=(3,), dtype="int32")
    arr[...] = np.array([1, 2, 3], dtype=np.int32)
    arr.attrs["label"] = "alpha"
    return store_path


def _build_sink_store(run_id: str, output_dir: Path, *, finalize: bool = False) -> Path:
    """Build a real ZarrStagingSink store at ``output_dir`` under ``run_id``,
    staging identical array content regardless of ``run_id``."""
    sink = ZarrStagingSink(SinkSpec(run_id=run_id, output_dir=output_dir, format="zarr"))
    sink.stage(("k",), data=np.array([1, 2, 3], dtype=np.int32))
    sink.drain()
    if finalize:
        sink.finalize()
    return output_dir


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


# --- #5013: zarr_content_digest / update_zarr_node_digest provenance exclusion ---
#
# ZarrStagingSink stamps run-varying provenance (root: git_sha/git_branch/git_dirty/
# run_id/created_at; every drained key's own group: run_id/git_sha) into every store
# it writes. zarr_content_digest folds every node's attrs into its sha256, so two
# structurally-identical writes currently never compare equal -- contradicting its
# own docstring ("unaffected by ... which process/session wrote the store, only by
# the store's logical content"). These tests pin the fix: a keyword-only
# ``include_provenance: bool = False`` on both functions, default-excluding
# provenance; root group skips all five core fields, non-root GROUPS skip only
# run_id/git_sha, ARRAYS skip nothing.


def test_digest_equal_across_different_run_ids_by_default(tmp_path: Path) -> None:
    """Two ZarrStagingSink instances with different run_ids, writing identical
    staged content to different output paths, must produce EQUAL digests under
    the default include_provenance=False."""
    path_a = _build_sink_store("run-aaaaaaaaaaaa", tmp_path / "a.zarr")
    path_b = _build_sink_store("run-bbbbbbbbbbbb", tmp_path / "b.zarr")

    assert zarr_content_digest(path_a) == zarr_content_digest(path_b)


def test_digest_differs_across_different_run_ids_when_provenance_included(
    tmp_path: Path,
) -> None:
    """The same pair of stores, but with include_provenance=True, must produce
    DIFFERENT digests -- run_id (and created_at) genuinely differ between them."""
    path_a = _build_sink_store("run-aaaaaaaaaaaa", tmp_path / "a.zarr")
    path_b = _build_sink_store("run-bbbbbbbbbbbb", tmp_path / "b.zarr")

    digest_a = zarr_content_digest(path_a, include_provenance=True)
    digest_b = zarr_content_digest(path_b, include_provenance=True)
    assert digest_a != digest_b


def test_root_run_id_attr_excluded_by_default_but_array_run_id_attr_is_hashed(
    tmp_path: Path,
) -> None:
    """A hand-built store whose ROOT group carries an attr literally named
    ``run_id`` has that attr excluded from the digest by default -- but the
    identical attr NAME placed on a non-root ARRAY is still hashed (arrays skip
    nothing). This pins the array/group asymmetry in the exclusion scope."""
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(store_path), mode="a")
    root.attrs["run_id"] = "run-aaa"
    arr = root.create_array(name="data", shape=(3,), dtype="int32")
    arr[...] = np.array([1, 2, 3], dtype=np.int32)
    arr.attrs["run_id"] = "run-aaa"

    original = zarr_content_digest(store_path)

    # Root group's run_id: excluded by default -- changing it must NOT move the digest.
    root.attrs["run_id"] = "run-bbb"
    assert zarr_content_digest(store_path) == original

    # Array's run_id: arrays skip nothing -- changing it MUST move the digest.
    arr.attrs["run_id"] = "run-ccc"
    assert zarr_content_digest(store_path) != original


def test_non_root_group_run_id_attr_excluded_by_default_known_cost(tmp_path: Path) -> None:
    """Documents a known, accepted cost of the fix: exclusion is by attr NAME,
    unconditionally -- not by provenance origin. A hand-built store's non-root
    GROUP carrying a domain-meaningful attr that happens to be literally named
    ``run_id`` loses that attr's contribution to the digest by default, exactly
    like ZarrStagingSink's own per-key provenance pointer would. Callers who need
    such an attr digested must pass include_provenance=True or rename the attr."""
    store_path = tmp_path / "store.zarr"
    root = zarr.open_group(str(store_path), mode="a")
    sub = root.require_group("nested")
    arr = sub.create_array(name="data", shape=(1,), dtype="int32")
    arr[...] = np.array([1], dtype=np.int32)
    sub.attrs["run_id"] = "domain-meaningful-value-1"

    original = zarr_content_digest(store_path)
    sub.attrs["run_id"] = "domain-meaningful-value-2"
    assert zarr_content_digest(store_path) == original, (
        "non-root GROUP attrs named run_id/git_sha are excluded from the digest "
        "by default (include_provenance=False) -- a caller-authored attr sharing "
        "that name pays the same cost as the sink's own provenance pointer."
    )


def test_root_exclusion_applies_when_caller_passes_zarrs_own_root_path(tmp_path: Path) -> None:
    """`update_zarr_node_digest` is public API, and zarr's root reports ``path == ""``.

    Regression guard: the root branch originally matched the literal ``"/"``, which
    `zarr_content_digest` passes internally but which zarr itself never reports --
    `root.path` is ``""`` and only `root.name` is ``"/"``. A caller writing the
    natural `update_zarr_node_digest(h, root, root.path)` therefore fell into the
    NON-root branch and kept created_at/git_branch/git_dirty in the digest, silently
    reintroducing #5013 through the public API with no error.
    """
    path_a = _build_sink_store("run-aaaaaaaaaaaa", tmp_path / "a.zarr")
    path_b = _build_sink_store("run-bbbbbbbbbbbb", tmp_path / "b.zarr")

    def digest_via_root_path(store: Path) -> str:
        root = zarr.open_group(str(store), mode="r")
        h = hashlib.sha256()
        update_zarr_node_digest(h, root, root.path)
        return h.hexdigest()

    assert digest_via_root_path(path_a) == digest_via_root_path(path_b)


def test_provenance_exclusion_survives_finalize_consolidated_metadata(tmp_path: Path) -> None:
    """The shape a done-marker digest is actually taken over.

    After `finalize()`, `zarr.consolidate_metadata()` has run and a reader serves
    attrs from the consolidated document rather than per-node `zarr.json`. Every
    other sink-backed test here stops at `drain()`, so this is the one path a
    regression in attr plumbing could hide in.
    """
    path_a = _build_sink_store("run-cccccccccccc", tmp_path / "c.zarr", finalize=True)
    path_b = _build_sink_store("run-dddddddddddd", tmp_path / "d.zarr", finalize=True)

    assert zarr_content_digest(path_a) == zarr_content_digest(path_b)
    assert zarr_content_digest(path_a, include_provenance=True) != zarr_content_digest(
        path_b, include_provenance=True
    )
