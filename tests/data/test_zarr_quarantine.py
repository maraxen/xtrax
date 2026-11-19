"""Tests for zarr commit chunks, content verification, and quarantine (#3117)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.run import zarr_commit as zc
from xtrax.run.ident import new_run_id
from xtrax.run.sink import SinkSpec
from xtrax.run.zarr_sink import ZarrStagingSink

KEY = ("chunks", "k0")


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    assert zc.create_store(
        store,
        identity_payload={"type": "verify-dup-quarantine"},
        creator_run_id="creator",
        prefixes=(("chunks",),),
        writer_id="w",
    )
    return store


def test_write_staged_group_custom_tuple_chunks(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    data = np.arange(16, dtype=np.int32).reshape((4, 4))
    zc.write_staged_group(staged, {"arr": data}, chunks=(2, 2))

    grp = zarr.open_group(str(staged), mode="r")
    assert grp["arr"].chunks == (2, 2)


def test_write_staged_group_mapping_chunks(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    data_a = np.arange(10, dtype=np.int32)
    data_b = np.arange(16, dtype=np.int32).reshape((4, 4))
    zc.write_staged_group(
        staged,
        {"a": data_a, "b": data_b},
        chunks={"a": (5,), "b": (2, 4)},
    )

    grp = zarr.open_group(str(staged), mode="r")
    assert grp["a"].chunks == (5,)
    assert grp["b"].chunks == (2, 4)


def test_write_staged_group_rank_mismatch_raises(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    data = np.arange(16, dtype=np.int32).reshape((4, 4))
    with pytest.raises(ValueError, match="rank"):
        zc.write_staged_group(staged, {"arr": data}, chunks=(2, 2, 2))


def test_quarantine_force_true_bypasses_race_rollback(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged = zc.staging_root(store) / "w" / "a"
    zc.write_staged_group(staged, {"arr": np.arange(5)})
    zc.commit_key(store, KEY, staged, input_digest="d1", run_id="r", env={})

    # Without force=True, quarantining a valid key with expected_input_digest="d1"
    # raises QuarantineRaceError
    with pytest.raises(zc.QuarantineRaceError):
        zc.quarantine(store, KEY, expected_input_digest="d1")

    # With force=True, it unconditionally quarantines the valid key
    quarantined = zc.quarantine(store, KEY, expected_input_digest="d1", force=True)
    assert quarantined is not None
    assert quarantined.exists()
    assert not (store / "chunks" / "k0").exists()


def test_duplicate_content_divergence_quarantines_both_versions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged_a = zc.staging_root(store) / "w" / "a"
    zc.write_staged_group(staged_a, {"arr": np.arange(5, dtype=np.int32)})
    first = zc.commit_key(store, KEY, staged_a, input_digest="d1", run_id="r1", env={})
    assert isinstance(first, zc.Committed)

    # Second commit with same input_digest but different content and verify_duplicate=True
    staged_b = zc.staging_root(store) / "w" / "b"
    zc.write_staged_group(staged_b, {"arr": np.zeros(5, dtype=np.int32)})

    with pytest.raises(zc.CommitConflictError, match="different content"):
        zc.commit_key(
            store,
            KEY,
            staged_b,
            input_digest="d1",
            run_id="r2",
            env={},
            verify_duplicate=True,
            chunks=(2,),
        )

    # Verify both committed key and staged dir were preserved into quarantine
    q_root = zc.quarantine_root(store) / "chunks"
    assert q_root.is_dir()
    quarantined_items = list(q_root.iterdir())
    assert len(quarantined_items) == 2

    # One item is the committed group (named k0-<uuid8>)
    committed_q = [p for p in quarantined_items if "-staged-" not in p.name]
    assert len(committed_q) == 1
    assert (committed_q[0] / "arr").is_dir()

    # One item is the staged group (named k0-staged-<uuid8>)
    staged_q = [p for p in quarantined_items if "-staged-" in p.name]
    assert len(staged_q) == 1
    assert (staged_q[0] / "arr").is_dir()

    # Neither target in store nor staged dir in staging exists anymore
    assert not (store / "chunks" / "k0").exists()
    assert not staged_b.exists()


def test_duplicate_same_content_cleans_up_staged(tmp_path: Path) -> None:
    store = _store(tmp_path)
    staged_a = zc.staging_root(store) / "w" / "a"
    zc.write_staged_group(staged_a, {"arr": np.arange(5, dtype=np.int32)})
    zc.commit_key(store, KEY, staged_a, input_digest="d1", run_id="r1", env={})

    staged_b = zc.staging_root(store) / "w" / "b"
    zc.write_staged_group(staged_b, {"arr": np.arange(5, dtype=np.int32)})
    second = zc.commit_key(
        store,
        KEY,
        staged_b,
        input_digest="d1",
        run_id="r2",
        env={},
        verify_duplicate=True,
    )
    assert isinstance(second, zc.Duplicate)
    assert not staged_b.exists()
    assert (store / "chunks" / "k0").exists()


def test_sink_spec_and_zarr_sink_chunks(tmp_path: Path) -> None:
    store = tmp_path / "sink.zarr"
    spec = SinkSpec(
        run_id=new_run_id(),
        output_dir=store,
        format="zarr",
        open_mode="create_or_join",
        store_identity={"kind": "test-chunks"},
        prefixes=(("tiling",),),
        chunks=(2, 2),
    )
    assert spec.chunks == (2, 2)
    sink = ZarrStagingSink(spec)
    key = ("tiling", "t0")
    data = np.arange(16, dtype=np.float32).reshape((4, 4))
    sink.stage(key, input_digest="d1", data=data)
    sink.drain()

    grp = zarr.open_group(str(store / "tiling" / "t0"), mode="r")
    assert grp["data"].chunks == (2, 2)
