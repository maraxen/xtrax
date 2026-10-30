"""zarr_commit: content check on Duplicate, and quarantine (spec 261030_ledger-pack-fileguard §3.5).

``commit_key`` decides ``Duplicate`` on the input digest alone. With ``verify_duplicate=True`` a
second commit of the same input with different content raises ``CommitConflictError`` instead
(alynxr ``fold_stats/array_commit.py`` R-S6, lifted). ``quarantine`` moves a damaged key out of
the store, never deletes it, and restores a key that turns out to be valid (a racing writer).
"""

from pathlib import Path

import numpy as np
import pytest

from xtrax.run import zarr_commit as zc
from xtrax.run.ident import new_run_id
from xtrax.run.sink import SinkSpec
from xtrax.run.zarr_sink import ZarrStagingSink

KEY = ("chunks", "k0")


def _store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    assert zc.create_store(
        store,
        identity_payload={"type": "verify-dup"},
        creator_run_id="creator",
        prefixes=(("chunks",),),
        writer_id="w",
    )
    return store


def _commit(store: Path, values, *, name: str, digest: str = "d1", **kw):
    staged = zc.staging_root(store) / "w" / name
    zc.write_staged_group(staged, {"arr": np.asarray(values)})
    return zc.commit_key(store, KEY, staged, input_digest=digest, run_id="r", env={}, **kw)


def _flip_last_chunk_byte(key_dir: Path) -> None:
    chunk = key_dir / "arr" / "c" / "0"
    data = bytearray(chunk.read_bytes())
    data[-1] ^= 0x01
    chunk.write_bytes(bytes(data))


# --- verify_duplicate ---------------------------------------------------------


def test_default_keeps_input_only_duplicate_decision(tmp_path):
    store = _store(tmp_path)
    first = _commit(store, np.arange(5), name="a")
    second = _commit(store, np.zeros(5), name="b")  # same input, different content
    assert isinstance(second, zc.Duplicate)
    assert second.record == first.record


def test_verify_duplicate_same_content_is_duplicate(tmp_path):
    store = _store(tmp_path)
    _commit(store, np.arange(5), name="a")
    out = _commit(store, np.arange(5), name="b", verify_duplicate=True)
    assert isinstance(out, zc.Duplicate)


def test_negative_control_verify_duplicate_different_content_raises(tmp_path):
    store = _store(tmp_path)
    first = _commit(store, np.arange(5), name="a")
    with pytest.raises(zc.CommitConflictError, match="different content") as exc:
        _commit(store, np.zeros(5), name="b", verify_duplicate=True)
    assert exc.value.stored_record == first.record
    # The committed key is untouched and the staged group is gone.
    found = zc.lookup(store, KEY, "d1")
    assert isinstance(found, zc.Reuse)
    assert found.record == first.record
    assert not (zc.staging_root(store) / "w" / "b").exists()


def test_verify_duplicate_different_input_still_raises_as_before(tmp_path):
    store = _store(tmp_path)
    _commit(store, np.arange(5), name="a")
    with pytest.raises(zc.CommitConflictError, match="different input_digest"):
        _commit(store, np.arange(5), name="b", digest="d2", verify_duplicate=True)


# --- quarantine ---------------------------------------------------------------


def test_quarantine_root_is_a_sibling_of_the_store(tmp_path):
    store = tmp_path / "s.zarr"
    assert zc.quarantine_root(store) == tmp_path / "s.zarr.quarantine"


def test_quarantine_moves_a_corrupt_key_out_and_keeps_its_bytes(tmp_path):
    store = _store(tmp_path)
    _commit(store, np.arange(5), name="a")
    target = store / "chunks" / "k0"
    _flip_last_chunk_byte(target)
    assert isinstance(zc.lookup(store, KEY, "d1"), zc.Corrupt)
    files_before = sorted(p.relative_to(target) for p in target.rglob("*"))

    moved = zc.quarantine(store, KEY)

    assert moved is not None
    assert moved.parent == zc.quarantine_root(store) / "chunks"
    assert moved.name.startswith("k0-")
    assert sorted(p.relative_to(moved) for p in moved.rglob("*")) == files_before
    assert isinstance(zc.lookup(store, KEY, "d1"), zc.Missing)
    assert zc.committed_keys(store) == []
    # The key can be committed again after quarantine.
    assert isinstance(_commit(store, np.arange(5), name="again"), zc.Committed)


def test_quarantine_moves_an_uncommitted_group(tmp_path):
    store = _store(tmp_path)
    zc.write_staged_group(store / "chunks" / "k0", {"arr": np.arange(3)})  # no commit record
    assert zc.quarantine(store, KEY) is not None
    assert not (store / "chunks" / "k0").exists()


def test_quarantine_of_a_missing_key_returns_none(tmp_path):
    store = _store(tmp_path)
    assert zc.quarantine(store, KEY) is None
    assert not zc.quarantine_root(store).exists() or not any(zc.quarantine_root(store).rglob("*"))


def test_negative_control_quarantine_restores_a_valid_key_and_raises(tmp_path):
    store = _store(tmp_path)
    first = _commit(store, np.arange(5), name="a")
    with pytest.raises(zc.QuarantineRaceError, match="put back untouched"):
        zc.quarantine(store, KEY)
    found = zc.lookup(store, KEY, "d1")
    assert isinstance(found, zc.Reuse)
    assert found.record == first.record


def test_quarantine_race_with_a_newer_commit_keeps_both(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _commit(store, np.arange(5), name="a")
    target = store / "chunks" / "k0"
    real_valid = zc._group_is_valid

    def racing_valid(path):
        # While the old (valid) key sits in quarantine, a racing writer commits a new one.
        if path != target and not target.exists():
            _commit(store, np.arange(5), name="racer")
        return real_valid(path)

    monkeypatch.setattr(zc, "_group_is_valid", racing_valid)
    with pytest.raises(zc.QuarantineRaceError, match="newer one is now in place"):
        zc.quarantine(store, KEY)
    assert isinstance(zc.lookup(store, KEY, "d1"), zc.Reuse)
    assert len(list((zc.quarantine_root(store) / "chunks").iterdir())) == 1


# --- sink pass-through --------------------------------------------------------


def _sink(store: Path, **kw) -> ZarrStagingSink:
    return ZarrStagingSink(
        SinkSpec(
            run_id=new_run_id(),
            output_dir=store,
            format="zarr",
            flush_every=1000,
            open_mode="create_or_join",
            store_identity={"kind": "verify-dup-sink"},
            prefixes=(("shards",),),
            **kw,
        )
    )


def test_sink_spec_verify_duplicate_defaults_off(tmp_path):
    assert SinkSpec(run_id="r").verify_duplicate is False
    with pytest.raises(TypeError):
        SinkSpec(run_id="r", verify_duplicate="yes")  # ty: ignore[invalid-argument-type]


def test_negative_control_durable_sink_raises_on_same_input_different_content(tmp_path):
    store = tmp_path / "sink.zarr"
    key = ("shards", "s0")
    first = _sink(store)
    first.stage(key, input_digest="d1", x=np.arange(5))
    first.drain()

    lax = _sink(store)
    lax.stage(key, input_digest="d1", x=np.zeros(5))
    assert isinstance(lax.drain()[key], zc.Duplicate)

    strict = _sink(store, verify_duplicate=True)
    strict.stage(key, input_digest="d1", x=np.zeros(5))
    with pytest.raises(zc.CommitConflictError, match="different content"):
        strict.drain()
