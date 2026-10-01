"""Tests for ZarrStagingSink durable mode (``open_mode="create_or_join"``) -- spec U1/U3.

The multi-process concurrency test (T-X3) lives in ``test_zarr_sink_join_concurrency.py``.
Exclusive-mode regression (T-X6) is covered by the unmodified ``test_zarr_sink.py``,
``test_zarr_sink_errors.py`` and ``test_sink.py``.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from xtrax.run import zarr_commit as zc
from xtrax.run.ident import new_run_id
from xtrax.run.sink import SinkSpec, derive_sink_spec
from xtrax.run.spec import RunSpec
from xtrax.run.zarr_sink import ZarrStagingSink

IDENTITY: dict[str, Any] = {"kind": "durable-sink-test", "version": 1}
PREFIXES = (("shards",),)
WRITERS = "_xtrax_writers"


def _spec(store: Path, run_id: str | None = None, **overrides: Any) -> SinkSpec:  # noqa: ANN401
    kwargs: dict[str, Any] = {
        "run_id": run_id or new_run_id(),
        "output_dir": store,
        "format": "zarr",
        "flush_every": 1000,
        "open_mode": "create_or_join",
        "store_identity": IDENTITY,
        "prefixes": PREFIXES,
    }
    kwargs.update(overrides)
    return SinkSpec(**kwargs)


def _sink(store: Path, run_id: str | None = None, **overrides: Any) -> ZarrStagingSink:  # noqa: ANN401
    return ZarrStagingSink(_spec(store, run_id, **overrides))


def _root_bytes(store: Path) -> bytes:
    return (store / "zarr.json").read_bytes()


def _shard_records(store: Path) -> dict[tuple[str, ...], zc.CommitRecord]:
    """Every committed key outside the writers prefix, with its verified record."""
    out: dict[tuple[str, ...], zc.CommitRecord] = {}
    for key in zc.committed_keys(store, ("shards",)):
        record = zc.read_record(store / zc.key_path(key))
        assert record is not None
        out[key] = record
    return out


# --------------------------------------------------------------------------------------
# 1. SinkSpec validation
# --------------------------------------------------------------------------------------


class TestSinkSpecDurableFields:
    def test_defaults_are_exclusive(self) -> None:
        spec = SinkSpec(run_id="r")
        assert spec.open_mode == "exclusive"
        assert spec.store_identity is None
        assert spec.prefixes == ()

    def test_rejects_unknown_open_mode(self) -> None:
        # Mutate + re-run __post_init__: the test env's beartype hook would otherwise
        # reject the Literal at the call boundary before the plain-Python backstop runs.
        spec = SinkSpec(run_id="r", format="zarr")
        spec.open_mode = "append"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="open_mode"):
            spec.__post_init__()

    def test_create_or_join_requires_store_identity(self) -> None:
        with pytest.raises(ValueError, match="store_identity"):
            SinkSpec(run_id="r", format="zarr", open_mode="create_or_join")

    def test_create_or_join_requires_zarr_format(self) -> None:
        with pytest.raises(ValueError, match="zarr"):
            SinkSpec(
                run_id="r", format="jsonl", open_mode="create_or_join", store_identity=IDENTITY
            )

    def test_empty_identity_mapping_is_still_an_identity(self) -> None:
        spec = SinkSpec(run_id="r", format="zarr", open_mode="create_or_join", store_identity={})
        assert spec.store_identity == {}

    def test_prefixes_normalized_to_tuple_of_tuples_of_str(self) -> None:
        spec = SinkSpec(run_id="r")
        spec.prefixes = [["a", "b"], ("c",), (1, 2)]  # type: ignore[assignment]
        spec.__post_init__()
        assert spec.prefixes == (("a", "b"), ("c",), ("1", "2"))
        assert isinstance(spec.prefixes, tuple)
        assert all(isinstance(p, tuple) for p in spec.prefixes)

    def test_bare_string_prefix_rejected(self) -> None:
        spec = SinkSpec(run_id="r")
        spec.prefixes = ["shards"]  # type: ignore[assignment]
        with pytest.raises(TypeError, match="prefixes"):
            spec.__post_init__()

    def test_derive_sink_spec_forwards_durable_fields(self, tmp_path: Path) -> None:
        derived = derive_sink_spec(
            RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None),
            output_dir=tmp_path / "s",
            open_mode="create_or_join",
            store_identity=IDENTITY,
            prefixes=[("shards",)],
        )
        assert derived.open_mode == "create_or_join"
        assert derived.store_identity == IDENTITY
        assert derived.prefixes == (("shards",),)

    def test_derive_sink_spec_default_stays_exclusive(self, tmp_path: Path) -> None:
        derived = derive_sink_spec(
            RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None), output_dir=tmp_path / "s"
        )
        assert (derived.open_mode, derived.store_identity, derived.prefixes) == (
            "exclusive",
            None,
            (),
        )

    def test_derive_sink_spec_create_or_join_without_identity_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="store_identity"):
            derive_sink_spec(
                RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None),
                output_dir=tmp_path / "s",
                open_mode="create_or_join",
            )


# --------------------------------------------------------------------------------------
# 2. Round trip / duplicate / conflict
# --------------------------------------------------------------------------------------


class TestDurableRoundTrip:
    def test_commit_lookup_duplicate_and_conflict(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        key = ("shards", "a")
        sink1 = _sink(store, "run-one")
        sink1.stage(key, {"note": "hi"}, input_digest="d1", x=np.arange(5))
        outcomes = sink1.drain()
        assert list(outcomes) == [key]
        committed = outcomes[key]
        assert isinstance(committed, zc.Committed)
        assert committed.record.input_digest == "d1"
        assert committed.record.run_id == "run-one"
        assert len(sink1) == 0

        result = sink1.lookup(key, "d1")
        assert isinstance(result, zc.Reuse)
        assert result.record == committed.record
        group = result.open()
        np.testing.assert_array_equal(group["x"][:], np.arange(5))
        assert group.attrs["note"] == "hi"
        assert group.attrs["run_id"] == "run-one"
        assert isinstance(sink1.lookup(key, "other-digest"), zc.Stale)
        assert isinstance(sink1.lookup(("shards", "nope"), "d1"), zc.Missing)

        # A second sink (new run_id) re-staging the same key + digest gets Duplicate.
        sink2 = _sink(store, "run-two")
        sink2.stage(key, input_digest="d1", x=np.arange(5))
        dup = sink2.drain()
        assert isinstance(dup[key], zc.Duplicate)
        assert dup[key].record == committed.record  # the original record, not run-two's
        assert len(sink2) == 0

        # Same key, different digest -> CommitConflictError; the key stays buffered.
        sink3 = _sink(store, "run-three")
        sink3.stage(key, input_digest="d-different", x=np.zeros(5))
        with pytest.raises(zc.CommitConflictError) as exc:
            sink3.drain()
        assert exc.value.key == key
        assert exc.value.stored_record is not None
        assert exc.value.stored_record.input_digest == "d1"
        assert len(sink3) == 1
        # ...and the committed content is untouched.
        np.testing.assert_array_equal(sink1.lookup(key, "d1").open()["x"][:], np.arange(5))  # type: ignore[union-attr]

    def test_conflict_midway_leaves_only_failing_and_later_keys_buffered(
        self, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        taken = ("shards", "taken")
        first = _sink(store)
        first.stage(taken, input_digest="orig", x=np.ones(2))
        first.drain()

        sink = _sink(store)
        sink.stage(("shards", "ok1"), input_digest="d", x=np.ones(2))
        sink.stage(taken, input_digest="changed", x=np.zeros(2))
        sink.stage(("shards", "after"), input_digest="d", x=np.ones(2))
        with pytest.raises(zc.CommitConflictError):
            sink.drain()
        # ok1 was committed before the conflict and is out of the buffer; the failing
        # key and the one after it remain buffered.
        assert len(sink) == 2
        assert isinstance(sink.lookup(("shards", "ok1"), "d"), zc.Reuse)
        assert isinstance(sink.lookup(("shards", "after"), "d"), zc.Missing)
        # Drop the offender and the rest drains cleanly; the failed drain's earlier
        # outcome (ok1) is still unreported, so this drain returns it too.
        sink.take(taken)
        out = sink.drain()
        assert list(out) == [("shards", "ok1"), ("shards", "after")]
        assert all(isinstance(o, zc.Committed) for o in out.values())

    def test_attr_only_and_empty_array_shards(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        sink.stage(("shards", "attrs"), {"k": 1}, input_digest="d")
        sink.stage(("shards", "empty"), input_digest="d", z=np.zeros((0, 3)))
        out = sink.drain()
        assert all(isinstance(o, zc.Committed) for o in out.values())
        empty = sink.lookup(("shards", "empty"), "d")
        assert isinstance(empty, zc.Reuse)
        assert empty.open()["z"].shape == (0, 3)

    def test_committed_keys_lists_shards_and_writers(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store", "run-x")
        sink.stage(("shards", "b"), input_digest="d", v=np.ones(1))
        sink.stage(("shards", "a"), input_digest="d", v=np.ones(1))
        sink.drain()
        assert sink.committed_keys(("shards",)) == [("shards", "a"), ("shards", "b")]
        assert sink.committed_keys((WRITERS,)) == [(WRITERS, "run-x")]

    def test_exclusive_drain_returns_empty_dict(self, tmp_path: Path) -> None:
        sink = ZarrStagingSink(SinkSpec(run_id="r", output_dir=tmp_path / "e", format="zarr"))
        sink.stage((0,), v=np.ones(1))
        assert sink.drain() == {}
        assert sink.drain() == {}


# --------------------------------------------------------------------------------------
# 3. stage() argument rules
# --------------------------------------------------------------------------------------


class TestStageArguments:
    def test_durable_stage_requires_input_digest(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        with pytest.raises(ValueError, match="input_digest") as exc:
            sink.stage(("shards", "needs-digest"), x=np.ones(1))
        assert "needs-digest" in str(exc.value)
        assert len(sink) == 0

    @pytest.mark.parametrize(
        "kwarg",
        [
            {"input_digest": "d"},
            {"input_payload": {"a": 1}},
            {"commit_meta": {"a": 1}},
            {"env_extra": {"a": 1}},
        ],
    )
    def test_exclusive_rejects_durable_only_arguments(
        self, tmp_path: Path, kwarg: dict[str, Any]
    ) -> None:
        sink = ZarrStagingSink(
            SinkSpec(run_id="r", output_dir=tmp_path / "e", format="zarr", flush_every=100)
        )
        with pytest.raises(ValueError, match="durable-only"):
            sink.stage((0,), x=np.ones(1), **kwarg)
        assert len(sink) == 0

    def test_env_extra_merged_into_record_env_numerics(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        key = ("shards", "e")
        sink.stage(
            key,
            input_digest="d",
            input_payload={"seed": 7},
            commit_meta={"tag": "t"},
            env_extra={"inner_width": 8},
            x=np.ones(2),
        )
        outcome = sink.drain()[key]
        record = outcome.record
        assert record.env["numerics"]["inner_width"] == 8
        # Process-level fields are still there alongside the per-key one.
        assert {"platform", "device_kind", "XLA_FLAGS"} <= set(record.env["numerics"])
        assert "info" in record.env
        assert record.input_payload == {"seed": 7}
        assert record.meta == {"tag": "t"}
        # ...and that is what is on disk, not just what was returned.
        on_disk = zc.read_record(tmp_path / "store" / "shards" / "e")
        assert on_disk == record

    def test_env_extra_does_not_leak_across_keys(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        sink.stage(("shards", "a"), input_digest="d", env_extra={"w": 1}, x=np.ones(1))
        sink.stage(("shards", "b"), input_digest="d", x=np.ones(1))
        out = sink.drain()
        assert out[("shards", "a")].record.env["numerics"]["w"] == 1
        assert "w" not in out[("shards", "b")].record.env["numerics"]

    def test_env_extra_collision_with_process_field_raises(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        with pytest.raises(ValueError, match="platform"):
            sink.stage(("shards", "c"), input_digest="d", env_extra={"platform": "x"}, x=np.ones(1))
        assert len(sink) == 0

    def test_reserved_xtrax_attrs_still_rejected(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        with pytest.raises(ValueError, match="xtrax"):
            sink.stage(("shards", "r"), {"xtrax.commit": {}}, input_digest="d", x=np.ones(1))
        with pytest.raises(ValueError, match="reserved"):
            sink.stage(("shards", "r"), {"run_id": "spoof"}, input_digest="d", x=np.ones(1))
        assert len(sink) == 0

    def test_writers_prefix_is_reserved_for_the_sink(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        with pytest.raises(ValueError, match=WRITERS):
            sink.stage((WRITERS, "forged"), input_digest="d")

    def test_empty_key_rejected(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        with pytest.raises(ValueError, match="key"):
            sink.stage((), input_digest="d")

    def test_flush_every_one_autoflush_commits_durably(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store, flush_every=1)
        sink.stage(("shards", "auto"), input_digest="d", x=np.arange(3))
        assert len(sink) == 0  # flushed without an explicit drain()
        result = sink.lookup(("shards", "auto"), "d", verify=True)
        assert isinstance(result, zc.Reuse)  # i.e. it carries a verified commit record
        np.testing.assert_array_equal(result.open()["x"][:], np.arange(3))

    def test_autoflush_conflict_keeps_payload_buffered(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        key = ("shards", "auto")
        _sink(store, flush_every=1).stage(key, input_digest="d1", x=np.arange(3))
        sink = _sink(store, flush_every=1)
        with pytest.raises(zc.CommitConflictError):
            sink.stage(key, input_digest="d2", x=np.arange(3))
        assert len(sink) == 1

    def test_schema_required_fields_across_existing_group(self, tmp_path: Path) -> None:
        """_validate_required_before_drain tolerates a key whose group doesn't exist yet."""
        schema = {"required": ["trial"], "properties": {"trial": {"type": "integer"}}}
        sink = _sink(tmp_path / "store", extension_schema=schema)
        sink.stage(("shards", "s"), {"other": 1}, input_digest="d", x=np.ones(1))
        with pytest.raises(ValueError, match="trial"):
            sink.drain()
        assert len(sink) == 1  # buffer intact
        sink.stage(("shards", "s"), {"trial": 3}, input_digest="d")
        out = sink.drain()
        assert isinstance(out[("shards", "s")], zc.Committed)
        group = zarr.open_group(str(tmp_path / "store" / "shards" / "s"), mode="r")
        assert group.attrs["trial"] == 3
        assert group.attrs["other"] == 1

    def test_unknown_prefix_raises_and_keeps_buffer(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store")
        sink.stage(("nowhere", "k"), input_digest="d", x=np.ones(1))
        with pytest.raises(zc.UnknownPrefixError):
            sink.drain()
        assert len(sink) == 1


# --------------------------------------------------------------------------------------
# 4. The root is never rewritten
# --------------------------------------------------------------------------------------


class TestRootNeverRewritten:
    def test_root_bytes_unchanged_across_joins_and_drains(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        creator = _sink(store)
        baseline = _root_bytes(store)
        assert baseline  # the creator really did write a root

        sinks = [creator] + [_sink(store) for _ in range(3)]
        assert _root_bytes(store) == baseline, "joining rewrote the root"
        for i, sink in enumerate(sinks):
            for j in range(2):
                sink.stage(("shards", f"s{i}-{j}"), input_digest="d", x=np.full(4, i * 10 + j))
            outcomes = sink.drain()
            assert len(outcomes) == 2
            assert _root_bytes(store) == baseline, f"drain by sink {i} rewrote the root"

        # A reserved stamp on a committed key must not touch the root either.
        creator.stamp_reserved(("shards", "s0-0"), "note", {"ok": True})
        assert _root_bytes(store) == baseline
        for sink in sinks:
            sink.close()
        assert _root_bytes(store) == baseline
        assert len(zc.committed_keys(store, ("shards",))) == 8

    def test_root_carries_only_the_store_record(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store, "run-creator")
        sink.stage(("shards", "k"), input_digest="d", x=np.ones(1))
        sink.drain()
        attrs = dict(zarr.open_group(str(store), mode="r").attrs)
        assert set(attrs) == {zc.STORE_ATTR}
        assert attrs[zc.STORE_ATTR]["creator_run_id"] == "run-creator"
        assert sink.store_record["creator_run_id"] == "run-creator"
        assert sink.store_record["identity_payload"] == IDENTITY

    def test_target_never_opened_in_append_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        modes: list[str] = []
        real = zarr.open_group

        def spy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            modes.append(str(kwargs.get("mode", "r+")))
            return real(*args, **kwargs)

        monkeypatch.setattr(zarr, "open_group", spy)
        store = tmp_path / "store"
        _sink(store).stage(("shards", "k"), input_digest="d", x=np.ones(1))
        _sink(store)
        assert "a" not in modes


# --------------------------------------------------------------------------------------
# 5. Writer records
# --------------------------------------------------------------------------------------


class TestWriterRecords:
    def test_record_committed_before_init_returns(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store, "run-w0")
        assert zc.committed_keys(store, (WRITERS,)) == [(WRITERS, "run-w0")]
        record = zc.read_record(store / WRITERS / "run-w0")
        assert record is not None
        assert record.run_id == "run-w0"
        assert set(record.input_payload) == {"run_id", "git_sha", "git_branch", "created_at"}
        assert record.input_payload["run_id"] == "run-w0"
        assert len(sink) == 0

    def test_each_writer_has_a_record_that_precedes_its_shards(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        run_ids = [f"run-w{i}" for i in range(4)]
        sinks = [_sink(store, rid) for rid in run_ids]
        for i, sink in enumerate(sinks):
            for j in range(3):
                sink.stage(("shards", f"s{i}-{j}"), input_digest="d", x=np.full(3, j))
            sink.drain()

        writer_records = {rid: zc.read_record(store / WRITERS / rid) for rid in run_ids}
        assert all(r is not None for r in writer_records.values())
        assert zc.committed_keys(store, (WRITERS,)) == [(WRITERS, rid) for rid in sorted(run_ids)]

        shards = _shard_records(store)
        assert len(shards) == 12
        for key, shard in shards.items():
            writer = writer_records[shard.run_id]
            assert writer is not None, key
            assert datetime.fromisoformat(writer.committed_at) <= datetime.fromisoformat(
                shard.committed_at
            ), f"shard {key} committed before its writer record"

    def test_same_run_id_joining_twice_raises(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _sink(store, "run-dup")
        with pytest.raises(ValueError, match="already joined"):
            _sink(store, "run-dup")
        # The failed join left no trace: one writer record, and no leaked staging dir.
        assert zc.committed_keys(store, (WRITERS,)) == [(WRITERS, "run-dup")]
        staging = zc.staging_root(store)
        own_dirs = [p.name for p in staging.iterdir() if p.name.startswith("run-dup-")]
        assert len(own_dirs) == 1, own_dirs  # the first sink's; the failed joiner cleaned up

    def test_new_run_id_resume_is_a_plain_join(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        first = _sink(store)
        second = _sink(store)  # "sequential resume" = join by a new run_id
        assert second.store_record["creator_run_id"] == first._spec.run_id
        assert len(zc.committed_keys(store, (WRITERS,))) == 2


# --------------------------------------------------------------------------------------
# 6. Store identity / prefixes on join
# --------------------------------------------------------------------------------------


class TestJoinValidation:
    def test_identity_mismatch_carries_both_payloads(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _sink(store)
        other = {"kind": "durable-sink-test", "version": 2}
        with pytest.raises(zc.StoreIdentityMismatch) as exc:
            _sink(store, store_identity=other)
        assert exc.value.stored_payload == IDENTITY
        assert exc.value.current_payload == other
        # The message names the differing field with both values.
        assert "version" in str(exc.value)
        assert "stored=1" in str(exc.value) and "current=2" in str(exc.value)
        # The rejected joiner left nothing behind.
        assert len(zc.committed_keys(store, (WRITERS,))) == 1

    def test_missing_prefix_on_join_raises(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        _sink(store)
        with pytest.raises(zc.UnknownPrefixError):
            _sink(store, prefixes=(("shards",), ("missing",)))

    def test_non_durable_directory_raises(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        zarr.open_group(str(plain), mode="w")
        with pytest.raises(zc.NotADurableStoreError):
            _sink(plain)

    def test_extra_prefixes_created_with_store_and_writers_prefix_always_present(
        self, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        _sink(store, prefixes=(("a",), ("b", "c"), ("a",)))
        root = zarr.open_group(str(store), mode="r")
        for path in ("a", "b/c", WRITERS):
            assert path in root, path


# --------------------------------------------------------------------------------------
# 7. close / context manager / finalize / stamp_reserved
# --------------------------------------------------------------------------------------


class TestLifecycle:
    def test_close_removes_only_own_staging_dir(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store)
        sink.stage(("shards", "k"), input_digest="d", x=np.ones(1))
        sink.drain()
        foreign = zc.staging_root(store) / "some-other-writer"
        (foreign / "inflight").mkdir(parents=True)
        own = zc.staging_root(store) / sink._writer_id
        assert own.exists()

        sink.close()
        assert not own.exists()
        assert (foreign / "inflight").is_dir(), "close() touched another writer's staging"
        sink.close()  # idempotent
        assert not own.exists()

    def test_context_manager_exit_closes(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        with _sink(store) as sink:
            sink.stage(("shards", "k"), input_digest="d", x=np.ones(1))
            sink.drain()
            own = zc.staging_root(store) / sink._writer_id
            assert own.exists()
        assert not own.exists()

    def test_exit_closes_even_on_exception(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        own = None
        with pytest.raises(RuntimeError, match="boom"), _sink(store) as sink:
            own = zc.staging_root(store) / sink._writer_id
            assert own.exists()
            raise RuntimeError("boom")
        assert own is not None
        assert not own.exists()

    def test_context_manager_is_a_noop_in_exclusive_mode(self, tmp_path: Path) -> None:
        spec = SinkSpec(run_id="r", output_dir=tmp_path / "e", format="zarr", flush_every=10)
        with ZarrStagingSink(spec) as sink:
            sink.stage((0,), v=np.ones(1))
            sink.drain()
        sink.close()
        assert float(zarr.open_group(str(tmp_path / "e"), mode="r")["0"]["v"][0]) == 1.0

    def test_gc_staging_spares_own_dir_and_removes_stale_foreign(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store)
        own = zc.staging_root(store) / sink._writer_id
        own.mkdir(parents=True, exist_ok=True)
        (own / "marker").write_text("mine")
        foreign = zc.staging_root(store) / "dead-writer"
        foreign.mkdir(parents=True)
        (foreign / "marker").write_text("theirs")
        time.sleep(0.05)
        removed = sink.gc_staging(timedelta(0))
        assert foreign in removed
        assert not foreign.exists()
        assert (own / "marker").read_text() == "mine"

    def test_finalize_raises_in_durable_mode(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store)
        with pytest.raises(RuntimeError, match="never consolidated"):
            sink.finalize()
        # Nothing was consolidated: the writer can still add groups.
        sink.stage(("shards", "late"), input_digest="d", x=np.ones(1))
        assert isinstance(sink.drain()[("shards", "late")], zc.Committed)

    def test_stamp_reserved_on_root_raises_in_durable_mode(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        sink = _sink(store)
        before = _root_bytes(store)
        with pytest.raises(ValueError, match="never rewrite root attrs"):
            sink.stamp_reserved((), "report", {"a": 1})
        assert _root_bytes(store) == before

    def test_stamp_reserved_on_committed_key_works_and_keeps_content_digest(
        self, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        sink = _sink(store)
        key = ("shards", "k")
        sink.stage(key, input_digest="d", x=np.arange(4))
        sink.drain()
        sink.stamp_reserved(key, "run_report", {"n": 3, "ok": True})
        attrs = dict(zarr.open_group(str(store / "shards" / "k"), mode="r").attrs)
        assert attrs["xtrax.run_report"] == {"n": 3, "ok": True}
        assert zc.COMMIT_ATTR in attrs
        # xtrax.* attrs are excluded from the content digest, so verification still passes.
        assert isinstance(sink.lookup(key, "d", verify=True), zc.Reuse)

    def test_stamp_reserved_on_uncommitted_key_raises_and_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        sink = _sink(store)
        with pytest.raises(ValueError, match="no committed group"):
            sink.stamp_reserved(("shards", "ghost"), "note", {"a": 1})
        assert not (store / "shards" / "ghost").exists()

    def test_durable_only_methods_raise_in_exclusive_mode(self, tmp_path: Path) -> None:
        sink = ZarrStagingSink(SinkSpec(run_id="r", output_dir=tmp_path / "e", format="zarr"))
        with pytest.raises(RuntimeError, match="durable"):
            sink.lookup(("a",), "d")
        with pytest.raises(RuntimeError, match="durable"):
            sink.committed_keys()
        with pytest.raises(RuntimeError, match="durable"):
            sink.gc_staging(timedelta(0))


# --------------------------------------------------------------------------------------
# 10. Fault injection: die right after the writer record is in place
# --------------------------------------------------------------------------------------

_CHILD = f"""
import sys
from pathlib import Path

import numpy as np

from xtrax.run.sink import SinkSpec
from xtrax.run.zarr_sink import ZarrStagingSink

store, run_id = Path(sys.argv[1]), sys.argv[2]
sink = ZarrStagingSink(
    SinkSpec(
        run_id=run_id,
        output_dir=store,
        format="zarr",
        open_mode="create_or_join",
        store_identity={IDENTITY!r},
        prefixes={PREFIXES!r},
    )
)
print("joined", flush=True)
sink.stage(("shards", "child-" + run_id), input_digest="d", x=np.arange(4))
sink.drain()
print("drained", flush=True)
"""


def _run_child(script: Path, store: Path, run_id: str, fault: str | None) -> tuple[int, str]:
    env = {k: v for k, v in os.environ.items() if k != zc.FAULT_ENV}
    if fault is not None:
        env[zc.FAULT_ENV] = fault
    proc = subprocess.run(
        [sys.executable, str(script), str(store), run_id],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


class TestFaultAfterWriterRecord:
    def test_death_after_writer_record_leaves_record_but_no_shard(self, tmp_path: Path) -> None:
        script = tmp_path / "child.py"
        script.write_text(_CHILD)
        store = tmp_path / "store"
        # The parent creates the store first, so the child's first rename is its writer record.
        _sink(store, "run-parent")

        # Control (no fault): the child joins, drains, and its shard exists.
        rc, out = _run_child(script, store, "run-control", None)
        assert rc == 0, out
        assert (store / "shards" / "child-run-control").is_dir()

        rc, out = _run_child(script, store, "run-fault", "renamed")
        assert rc == 137, out
        assert "joined" not in out  # died inside __init__, before returning
        record = zc.read_record(store / WRITERS / "run-fault")
        assert record is not None and record.run_id == "run-fault"
        assert not (store / "shards" / "child-run-fault").exists()
        assert [r for r in _shard_records(store).values() if r.run_id == "run-fault"] == []
        # The store is still healthy and the dead writer's staging is gc-able.
        assert isinstance(
            zc.lookup(store, ("shards", "child-run-control"), "d", verify=True), zc.Reuse
        )


# --------------------------------------------------------------------------------------
# drain() reports outcomes of auto-flushed keys too
# --------------------------------------------------------------------------------------


class TestDrainReportsAutoFlushedOutcomes:
    def test_default_flush_every_stage_then_drain_reports_the_key(self, tmp_path: Path) -> None:
        sink = ZarrStagingSink(
            SinkSpec(
                run_id=new_run_id(),
                output_dir=tmp_path / "store",
                format="zarr",  # flush_every left at its default of 1
                open_mode="create_or_join",
                store_identity=IDENTITY,
                prefixes=PREFIXES,
            )
        )
        key = ("shards", "k")
        sink.stage(key, input_digest="d", x=np.arange(3))
        assert len(sink) == 0  # already auto-flushed
        outcomes = sink.drain()
        assert isinstance(outcomes[key], zc.Committed)
        assert sink.drain() == {}

    def test_flush_every_one_three_stages_then_one_drain_reports_all(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store", flush_every=1)
        keys = [("shards", f"k{i}") for i in range(3)]
        for key in keys:
            sink.stage(key, input_digest="d", x=np.arange(3))
        outcomes = sink.drain()
        assert list(outcomes) == keys
        assert all(isinstance(o, zc.Committed) for o in outcomes.values())
        assert sink.drain() == {}

    def test_flush_every_two_mixes_autoflushed_and_buffered_keys(self, tmp_path: Path) -> None:
        sink = _sink(tmp_path / "store", flush_every=2)
        k1, k2, k3 = (("shards", f"k{i}") for i in (1, 2, 3))
        sink.stage(k1, input_digest="d", x=np.ones(1))
        assert len(sink) == 1
        sink.stage(k2, input_digest="d", x=np.ones(1))  # auto-flush commits k1 and k2
        assert len(sink) == 0
        sink.stage(k3, input_digest="d", x=np.ones(1))
        assert len(sink) == 1
        outcomes = sink.drain()
        assert list(outcomes) == [k1, k2, k3]
        assert all(isinstance(o, zc.Committed) for o in outcomes.values())
        assert sink.drain() == {}

    def test_duplicate_outcome_from_autoflush_is_reported(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        key = ("shards", "k")
        first = _sink(store, flush_every=1)
        first.stage(key, input_digest="d", x=np.ones(1))
        assert isinstance(first.drain()[key], zc.Committed)
        second = _sink(store, flush_every=1)
        second.stage(key, input_digest="d", x=np.ones(1))
        assert isinstance(second.drain()[key], zc.Duplicate)

    def test_autoflush_conflict_still_reports_earlier_autoflushed_key(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        taken = ("shards", "taken")
        _sink(store, flush_every=1).stage(taken, input_digest="orig", x=np.ones(1))
        k1 = ("shards", "k1")

        sink = _sink(store, flush_every=1)
        sink.stage(k1, input_digest="d", x=np.ones(1))  # commits on its own auto-flush
        with pytest.raises(zc.CommitConflictError):
            sink.stage(taken, input_digest="changed", x=np.zeros(1))  # a later flush conflicts
        sink.take(taken)
        outcomes = sink.drain()
        assert list(outcomes) == [k1]
        assert isinstance(outcomes[k1], zc.Committed)

    def test_conflict_midway_through_one_autoflush_keeps_earlier_outcomes(
        self, tmp_path: Path
    ) -> None:
        store = tmp_path / "store"
        taken = ("shards", "taken")
        _sink(store, flush_every=1).stage(taken, input_digest="orig", x=np.ones(1))
        ok = ("shards", "ok")

        sink = _sink(store, flush_every=2)
        sink.stage(ok, input_digest="d", x=np.ones(1))
        with pytest.raises(zc.CommitConflictError):  # one flush: ok commits, taken conflicts
            sink.stage(taken, input_digest="changed", x=np.zeros(1))
        sink.take(taken)
        outcomes = sink.drain()
        assert list(outcomes) == [ok]
        assert isinstance(outcomes[ok], zc.Committed)

    def test_failed_explicit_drain_does_not_lose_earlier_outcomes(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        taken = ("shards", "taken")
        _sink(store, flush_every=1).stage(taken, input_digest="orig", x=np.ones(1))
        sink = _sink(store, flush_every=1000)
        sink.stage(("shards", "ok"), input_digest="d", x=np.ones(1))
        sink.stage(taken, input_digest="changed", x=np.zeros(1))
        with pytest.raises(zc.CommitConflictError):
            sink.drain()
        sink.take(taken)
        assert list(sink.drain()) == [("shards", "ok")]
