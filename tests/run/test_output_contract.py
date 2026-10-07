"""Debt #2523: the run layer owns the output contract."""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest
import zarr

import xtrax
from xtrax.run import (
    DIGEST_ALGO_VERSION,
    MEMORY_DIGEST_ALGO_VERSION,
    GitProvenance,
    MemorySink,
    atomic_write_bytes,
    atomic_write_text,
    make_sink,
)
from xtrax.run.sink import SinkSpec, derive_sink_spec
from xtrax.run.spec import RunSpec
from xtrax.run.zarr_integrity import zarr_content_digest
from xtrax.run.zarr_sink import ZarrStagingSink


def _run(**kwargs: object) -> RunSpec:
    base: dict[str, object] = {"seed": 0, "axes": [], "carry_specs": [], "boundaries": None}
    base.update(kwargs)
    return RunSpec(**base)  # type: ignore[arg-type]


def test_memory_sink_write_readback_and_receipt() -> None:
    """make_sink('memory') stages, drains, and reads back in order."""
    sink = make_sink(SinkSpec(run_id="mem-run", format="memory", seed=3, flush_every=100))
    assert isinstance(sink, MemorySink)
    values = np.array([1, 2, 3], dtype=np.int32)
    sink.stage((0,), values=values, attrs={"note": "a"})
    assert len(sink) == 1
    popped = sink.take((0,))
    np.testing.assert_array_equal(popped["values"], values)
    sink.stage((0,), values=values, attrs={"note": "a"})
    sink.drain()
    assert len(sink) == 0
    np.testing.assert_array_equal(sink.read((0,))["values"], values)
    assert sink.read_attrs((0,))["note"] == "a"
    assert sink.provenance["producer"] == "xtrax"
    assert sink.provenance["xtrax_version"] == xtrax.__version__
    assert sink.provenance["run_id"] == "mem-run"
    receipt = sink.finalize()
    assert receipt.path is None
    assert receipt.run_id == "mem-run"
    assert receipt.seed == 3
    assert receipt.digest_algo_version == MEMORY_DIGEST_ALGO_VERSION
    assert MEMORY_DIGEST_ALGO_VERSION != DIGEST_ALGO_VERSION
    other = make_sink(SinkSpec(run_id="other", format="memory", seed=9, flush_every=100))
    assert isinstance(other, MemorySink)
    other.stage((0,), values=values, attrs={"note": "a"})
    other.drain()
    assert other.finalize().digest == receipt.digest


def _memory_sink(**kwargs: object) -> MemorySink:
    spec_kwargs: dict[str, object] = {"run_id": "mem", "format": "memory", "flush_every": 100}
    spec_kwargs.update(kwargs)
    sink = make_sink(SinkSpec(**spec_kwargs))  # type: ignore[arg-type]
    assert isinstance(sink, MemorySink)
    return sink


def test_memory_append_concatenates_and_rejects_mismatch() -> None:
    """append extends axis 0; a dtype or trailing-shape mismatch leaves the store."""
    sink = _memory_sink(append=True)
    sink.stage((0,), value=np.array([[1, 2], [3, 4]], dtype=np.int32))
    sink.drain()
    sink.stage((0,), value=np.array([[5, 6]], dtype=np.int32))
    sink.drain()
    np.testing.assert_array_equal(
        sink.read((0,))["value"], np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    )

    bad_dtype = _memory_sink(append=True)
    bad_dtype.stage((0,), value=np.array([[1, 2]], dtype=np.int32))
    bad_dtype.drain()
    bad_dtype.stage((0,), value=np.array([[3.0, 4.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="dtype"):
        bad_dtype.drain()
    np.testing.assert_array_equal(bad_dtype.read((0,))["value"], np.array([[1, 2]], dtype=np.int32))

    bad_shape = _memory_sink(append=True)
    bad_shape.stage((0,), value=np.array([[1, 2]], dtype=np.int32))
    bad_shape.drain()
    bad_shape.stage((0,), value=np.array([9, 9, 9], dtype=np.int32))
    with pytest.raises(ValueError, match="trailing shape"):
        bad_shape.drain()
    np.testing.assert_array_equal(bad_shape.read((0,))["value"], np.array([[1, 2]], dtype=np.int32))


def test_memory_digest_includes_names_and_ignores_attrs() -> None:
    """Renaming an array changes the digest; changing a caller attr does not.

    Memory digests are not zarr content digests (different version, different inputs).
    """

    def digest(name: str, note: str) -> str:
        sink = _memory_sink()
        array = np.array([1, 2], dtype=np.int32)
        sink.stage((0,), attrs={"note": note}, **{name: array})
        sink.drain()
        receipt = sink.finalize()
        assert receipt.digest_algo_version == MEMORY_DIGEST_ALGO_VERSION
        return receipt.digest

    assert digest("values", "a") == digest("values", "b")
    assert digest("values", "a") != digest("other", "a")


def test_memory_copies_on_stage_read_and_drain() -> None:
    """Caller mutations and read()/take() results do not alias stored bytes."""
    src = np.array([1, 2, 3], dtype=np.int32)
    sink = _memory_sink()
    sink.stage((0,), values=src)
    staged = sink._pending[(0,)]["values"]
    assert staged is not src
    taken = sink.take((0,))["values"]
    assert taken is not src
    src[0] = 9
    assert int(taken[0]) == 1

    src2 = np.array([4, 5, 6], dtype=np.int32)
    sink.stage((0,), values=src2)
    pending = sink._pending[(0,)]["values"]
    sink.drain()
    stored = sink._committed[(0,)]["values"]
    assert stored is not src2
    assert stored is not pending
    src2[0] = 0
    pending[0] = 0
    read_back = sink.read((0,))["values"]
    assert read_back is not stored
    np.testing.assert_array_equal(read_back, np.array([4, 5, 6], dtype=np.int32))
    read_back[0] = 7
    np.testing.assert_array_equal(sink.read((0,))["values"], np.array([4, 5, 6], dtype=np.int32))


def test_finalize_fsyncs_once_after_consolidate_before_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """finalize calls fsync_tree once, after consolidate_metadata and before the digest."""
    import xtrax.run.zarr_sink as zarr_sink_mod

    sink = ZarrStagingSink(
        SinkSpec(run_id="fsync", output_dir=tmp_path / "out.zarr", format="zarr", flush_every=100)
    )
    sink.stage((0,), value=np.array([1], dtype=np.int32))
    sink.drain()
    order: list[str] = []
    real_consolidate = zarr.consolidate_metadata
    real_fsync = zarr_sink_mod.fsync_tree
    real_digest = zarr_sink_mod.zarr_content_digest

    def consolidate(store: str, *args: object, **kwargs: object) -> None:
        order.append("consolidate")
        real_consolidate(store, *args, **kwargs)

    def fsync(path: Path) -> None:
        order.append("fsync")
        real_fsync(path)

    def digest(path: Path, *args: object, **kwargs: object) -> str:
        order.append("digest")
        return real_digest(path, *args, **kwargs)

    monkeypatch.setattr(zarr, "consolidate_metadata", consolidate)
    monkeypatch.setattr(zarr_sink_mod, "fsync_tree", fsync)
    monkeypatch.setattr(zarr_sink_mod, "zarr_content_digest", digest)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not part in the Zarr format 3 spec.*")
        sink.finalize()
    assert order == ["consolidate", "fsync", "digest"]


def test_append_with_create_or_join_raises() -> None:
    with pytest.raises(ValueError, match="append=True is incompatible"):
        SinkSpec(
            run_id="r",
            format="zarr",
            open_mode="create_or_join",
            store_identity={"schema": "v1"},
            append=True,
        )


def test_zarr_appends_along_leading_axis_across_drains(tmp_path: Path) -> None:
    spec = SinkSpec(
        run_id="append-run",
        output_dir=tmp_path / "out.zarr",
        format="zarr",
        flush_every=100,
        append=True,
    )
    sink = ZarrStagingSink(spec)
    sink.stage((0,), value=np.array([[1, 2], [3, 4]], dtype=np.int32))
    sink.drain()
    sink.stage((0,), value=np.array([[5, 6]], dtype=np.int32))
    sink.drain()
    stored = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["0"]["value"]
    np.testing.assert_array_equal(stored[:], np.array([[1, 2], [3, 4], [5, 6]]))


def test_zarr_append_rejects_trailing_shape_mismatch_and_keeps_buffer(tmp_path: Path) -> None:
    sink = ZarrStagingSink(
        SinkSpec(
            run_id="append-run",
            output_dir=tmp_path / "out.zarr",
            format="zarr",
            flush_every=100,
            append=True,
        )
    )
    sink.stage((0,), value=np.array([[1, 2]], dtype=np.int32))
    sink.drain()
    sink.stage((0,), value=np.array([9, 9, 9], dtype=np.int32))
    with pytest.raises(ValueError, match="append"):
        sink.drain()
    assert len(sink) == 1
    stored = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")["0"]["value"]
    np.testing.assert_array_equal(stored[:], np.array([[1, 2]]))


def test_finalize_receipt_is_reproducible_across_run_identity(tmp_path: Path) -> None:
    def build(run_id: str, seed: int, dest: Path, sha: str):
        sink = ZarrStagingSink(
            SinkSpec(
                run_id=run_id,
                output_dir=dest,
                format="zarr",
                flush_every=100,
                seed=seed,
                provenance=GitProvenance(git_sha=sha, git_branch="main", git_dirty=False),
            )
        )
        sink.stage((0,), value=np.array([1, 2, 3], dtype=np.int32))
        sink.drain()
        return sink.finalize()

    first = build("run-a", 1, tmp_path / "a.zarr", "a" * 40)
    second = build("run-b", 2, tmp_path / "b.zarr", "b" * 40)
    assert first.digest == second.digest
    assert first.digest == zarr_content_digest(tmp_path / "a.zarr")
    assert first.digest_algo_version == DIGEST_ALGO_VERSION == second.digest_algo_version
    assert first.path == tmp_path / "a.zarr"
    assert first.run_id == "run-a"
    assert first.seed == 1
    assert second.run_id == "run-b"
    assert second.seed == 2


def test_run_spec_run_id_and_seed_reach_the_store(tmp_path: Path) -> None:
    run = _run(seed=11, run_id="run-fixed", output_root=tmp_path / "from-spec")
    derived = derive_sink_spec(run, output_dir=None)
    assert derived.run_id == "run-fixed"
    assert derived.seed == 11
    assert derived.output_dir == tmp_path / "from-spec"
    explicit = derive_sink_spec(run, output_dir=tmp_path / "explicit")
    assert explicit.output_dir == tmp_path / "explicit"
    sink = make_sink(derived)
    assert isinstance(sink, ZarrStagingSink)
    sink.stage((0,), value=np.array([4], dtype=np.int32))
    sink.drain()
    root = zarr.open_group(str(tmp_path / "from-spec"), mode="r")
    assert root.attrs["run_id"] == "run-fixed"
    assert root.attrs["seed"] == 11
    receipt = sink.finalize()
    assert receipt.run_id == "run-fixed"
    assert receipt.seed == 11


def test_default_provenance_does_not_shell_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("git subprocess")

    monkeypatch.setattr(subprocess, "check_output", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sink = ZarrStagingSink(
            SinkSpec(run_id="r", output_dir=tmp_path / "out.zarr", format="zarr", flush_every=100)
        )
    sink.drain()
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert root.attrs["git_sha"] == "unknown"
    assert root.attrs["producer"] == "xtrax"
    assert root.attrs["xtrax_version"] == xtrax.__version__


def test_injected_provenance_and_path_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repo / "seed.txt").write_text("seed\n")
    git("add", "-A")
    git("commit", "-m", "seed")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    injected = ZarrStagingSink(
        SinkSpec(
            run_id="inj",
            output_dir=tmp_path / "inj.zarr",
            format="zarr",
            flush_every=100,
            provenance=GitProvenance(git_sha="abc", git_branch="dev", git_dirty=True),
        )
    )
    injected.drain()
    inj_root = zarr.open_group(str(tmp_path / "inj.zarr"), mode="r")
    assert inj_root.attrs["git_sha"] == "abc"
    assert inj_root.attrs["git_branch"] == "dev"
    assert inj_root.attrs["git_dirty"] is True

    captured = ZarrStagingSink(
        SinkSpec(
            run_id="cap",
            output_dir=tmp_path / "cap.zarr",
            format="zarr",
            flush_every=100,
            provenance=repo,
        )
    )
    captured.drain()
    cap_root = zarr.open_group(str(tmp_path / "cap.zarr"), mode="r")
    assert cap_root.attrs["git_sha"] == sha
    assert cap_root.attrs["git_dirty"] is False


def test_level_schemas_apply_per_key_depth(tmp_path: Path) -> None:
    level_schemas = {
        1: {
            "type": "object",
            "properties": {"kind": {"type": "string"}},
            "required": ["kind"],
        },
        2: {
            "type": "object",
            "properties": {"depth": {"type": "integer"}},
            "required": ["depth"],
        },
    }
    sink = ZarrStagingSink(
        SinkSpec(
            run_id="levels",
            output_dir=tmp_path / "out.zarr",
            format="zarr",
            flush_every=100,
            extension_schema={
                "type": "object",
                "properties": {"fallback": {"type": "string"}},
                "required": ["fallback"],
            },
            level_schemas=level_schemas,
        )
    )
    with pytest.raises(ValueError, match="level_schemas"):
        sink.stage((0, 1), value=np.array([1]), attrs={"depth": "nope"})
    sink.stage((0,), value=np.array([1]), attrs={"kind": "row"})
    sink.drain()
    sink.stage(("batch", "item"), value=np.array([2]), attrs={"depth": 3})
    sink.drain()
    # A depth with no level schema still uses extension_schema.
    with pytest.raises(ValueError, match="extension_schema"):
        sink.stage(("a", "b", "c"), value=np.array([3]), attrs={"fallback": 1})


def test_run_spec_execution_fields_are_static_and_default_unset() -> None:
    spec = _run()
    assert spec.output_root is None
    assert spec.device_count is None
    assert spec.precision is None
    assert spec.shard_lineage is None
    root = Path("/tmp/xtrax-out")
    filled = _run(
        output_root=root,
        device_count=4,
        precision="float32",
        shard_lineage=("root", "shard-0"),
    )
    assert filled.output_root == root
    assert filled.device_count == 4
    assert filled.precision == "float32"
    assert filled.shard_lineage == ("root", "shard-0")
    import jax

    leaves, _ = jax.tree_util.tree_flatten(filled)
    assert root not in leaves
    assert 4 not in leaves
    assert "float32" not in leaves
    assert ("root", "shard-0") not in leaves
    with pytest.raises(ValueError, match="device_count"):
        _run(device_count=0)


def test_interrupted_atomic_write_leaves_old_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"old")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_bytes(path, b"new-bytes")
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["data.bin"]

    fresh = tmp_path / "missing.bin"
    with pytest.raises(OSError, match="interrupted"):
        atomic_write_text(fresh, "nope")
    assert not fresh.exists()


def test_atomic_write_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def replace(src: str, dst: str) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    path = tmp_path / "note.txt"
    atomic_write_text(path, "hello")
    assert path.read_text(encoding="utf-8") == "hello"
    assert events == ["fsync", "replace", "fsync"]
