"""Tests for xtrax.run.zarr_sink.ZarrStagingSink."""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import zarr
from beartype.roar import BeartypeCallHintParamViolation

from xtrax.run.sink import SinkSpec, derive_sink_spec, make_sink
from xtrax.run.spec import RunSpec
from xtrax.run.zarr_sink import ZarrStagingSink


def _sink(tmp_path: Path, flush_every: int = 1, run_id: str = "test-run") -> ZarrStagingSink:
    spec = SinkSpec(
        run_id=run_id, output_dir=tmp_path / "out.zarr", format="zarr", flush_every=flush_every
    )
    return ZarrStagingSink(spec)


def test_requires_zarr_format() -> None:
    with pytest.raises(ValueError, match="zarr"):
        ZarrStagingSink(SinkSpec(run_id="r", format="jsonl"))


def test_rejects_empty_run_id(tmp_path: Path) -> None:
    """Fail loud per #96: an empty run_id would poison store provenance attrs."""
    spec = SinkSpec(run_id="", output_dir=tmp_path / "out.zarr", format="zarr")
    with pytest.raises(ValueError, match="run_id"):
        ZarrStagingSink(spec)


def test_rejects_none_run_id_at_spec_construction() -> None:
    """None never reaches ZarrStagingSink: typing enforces str at SinkSpec."""
    with pytest.raises((BeartypeCallHintParamViolation, TypeError)):
        SinkSpec(run_id=None)  # type: ignore[arg-type]


def test_requires_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        ZarrStagingSink(SinkSpec(run_id="r", format="zarr", output_dir=None))


def test_stage_buffers_without_writing_to_disk(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage((0, 0, 4), sequences=np.arange(4), logits=np.ones((4, 21)))
    assert len(sink) == 1
    # Not flushed yet -- store should have no arrays under this key.
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert "0/0/4" not in root


def test_take_pops_pending_entry_without_draining(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sequences = np.arange(4)
    sink.stage((1,), sequences=sequences)
    popped = sink.take((1,))
    assert np.array_equal(popped["sequences"], sequences)
    assert len(sink) == 0
    with pytest.raises(KeyError):
        sink.take((1,))


def test_drain_writes_pending_payloads_to_zarr(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sequences = np.arange(4)
    logits = np.arange(4 * 21, dtype=np.float32).reshape(4, 21)
    sink.stage((0, 0, 4), sequences=sequences, logits=logits)
    sink.drain()
    assert len(sink) == 0

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    group = root["0/0/4"]
    assert np.array_equal(group["sequences"][:], sequences)
    assert np.array_equal(group["logits"][:], logits)


def test_auto_flush_at_flush_every(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=2)
    sink.stage((0,), value=np.array([1.0]))
    assert len(sink) == 1
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert "0" not in root  # not flushed after 1st stage

    sink.stage((1,), value=np.array([2.0]))
    assert len(sink) == 0  # flushed after 2nd stage (flush_every=2)

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert np.array_equal(root["0"]["value"][:], [1.0])
    assert np.array_equal(root["1"]["value"][:], [2.0])


def test_repeated_stage_same_key_merges_arrays(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage((0,), sequences=np.arange(4))
    sink.stage((0,), logits=np.ones((4, 21)))
    payload = sink.take((0,))
    assert set(payload) == {"sequences", "logits"}


def test_drain_is_safe_to_call_when_nothing_pending(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.drain()  # should not raise
    assert len(sink) == 0


def test_redraining_same_key_overwrites(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage((0,), value=np.array([1, 2, 3]))
    sink.drain()
    sink.stage((0,), value=np.array([4, 5]))
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert np.array_equal(root["0"]["value"][:], [4, 5])


def test_attrs_written_to_group_on_drain(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage(
        (0,),
        value=np.array([1]),
        attrs={"pool_type": "BackboneOnly", "structure_ids": ["a", "b"], "parent_structure_idx": 3},
    )
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    group = root["0"]
    assert group.attrs["pool_type"] == "BackboneOnly"
    assert list(group.attrs["structure_ids"]) == ["a", "b"]
    assert group.attrs["parent_structure_idx"] == 3


def test_attrs_merge_across_repeated_stage_calls(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage((0,), attrs={"a": 1})
    sink.stage((0,), value=np.array([1]), attrs={"b": 2})
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert root["0"].attrs["a"] == 1
    assert root["0"].attrs["b"] == 2


def test_attrs_only_stage_with_no_arrays_still_creates_group(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage((0,), attrs={"note": "no arrays here"})
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert root["0"].attrs["note"] == "no arrays here"


def test_take_discards_pending_attrs(tmp_path: Path) -> None:
    sink = _sink(tmp_path, flush_every=100)
    sink.stage((0,), value=np.array([1]), attrs={"a": 1})
    sink.take((0,))
    # Re-stage the same key with no attrs -- should NOT resurrect the old attrs.
    sink.stage((0,), value=np.array([2]))
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert "a" not in root["0"].attrs


def test_multiple_sinks_reopen_same_store(tmp_path: Path) -> None:
    """A new sink instance against the same output_dir sees prior writes (mode='a' semantics)."""
    sink1 = _sink(tmp_path, flush_every=100)
    sink1.stage((0,), value=np.array([9]))
    sink1.drain()

    sink2 = _sink(tmp_path, flush_every=100)
    sink2.stage((1,), value=np.array([10]))
    sink2.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert np.array_equal(root["0"]["value"][:], [9])
    assert np.array_equal(root["1"]["value"][:], [10])


# --- Provenance tracking (task 260824_default-sink-provenance-tracking) ---


@pytest.fixture()
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real committed git repo; the process cwd is moved inside it."""
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
    monkeypatch.chdir(repo)
    return repo


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def test_run_id_is_required_on_spec() -> None:
    with pytest.raises(TypeError, match="run_id"):
        SinkSpec(format="zarr")  # type: ignore[call-arg]


def test_core_provenance_record_on_root(git_repo: Path) -> None:
    sink = _sink(git_repo.parent)
    sink.stage((0,), value=np.array([1]))
    sink.drain()

    root = zarr.open_group(str(git_repo.parent / "out.zarr"), mode="r")
    attrs = root.attrs
    assert attrs["git_sha"] == _git(git_repo, "rev-parse", "HEAD")
    assert attrs["git_branch"] == _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD")
    assert attrs["git_dirty"] is False
    assert attrs["run_id"] == "test-run"
    created = datetime.fromisoformat(attrs["created_at"])
    assert created.tzinfo is not None and created.utcoffset() == UTC.utcoffset(None)


def test_git_dirty_flag_true_when_worktree_dirty(git_repo: Path) -> None:
    (git_repo / "wip.txt").write_text("wip\n")  # untracked file before sink construction
    sink = _sink(git_repo.parent)
    sink.drain()

    root = zarr.open_group(str(git_repo.parent / "out.zarr"), mode="r")
    assert root.attrs["git_dirty"] is True


def test_provenance_stable_across_drains(git_repo: Path) -> None:
    sink = _sink(git_repo.parent)
    sink.stage((0,), value=np.array([1]))
    sink.drain()
    first = zarr.open_group(str(git_repo.parent / "out.zarr"), mode="r").attrs
    snapshot = {k: first[k] for k in ("git_sha", "git_branch", "git_dirty", "run_id", "created_at")}

    sink.stage((1,), value=np.array([2]))
    sink.drain()
    second = zarr.open_group(str(git_repo.parent / "out.zarr"), mode="r").attrs
    for k, v in snapshot.items():
        assert second[k] == v


def test_git_unknown_outside_repo_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # plain dir -- no repository anywhere above it
    with pytest.warns(UserWarning, match="not inside a git repository"):
        _sink(tmp_path)
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert root.attrs["git_sha"] == "unknown"


def test_git_unknown_missing_binary_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    monkeypatch.setenv("PATH", str(fake_bin))  # no git executable resolvable
    with pytest.warns(UserWarning, match="'git' executable was not found"):
        _sink(tmp_path)
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert root.attrs["git_sha"] == "unknown"


def test_git_unknown_failing_shellout_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    script = "#!/bin/sh\necho 'fatal: bogus ref' >&2\nexit 128\n"
    (fake_bin / "git").write_text(script)
    (fake_bin / "git").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin))
    # Trailing "(" pins this to the narrow CalledProcessError branch's wording
    # ("a git shellout failed (...)"); the broad-catch fallback says
    # "failed unexpectedly: ..." and must NOT satisfy this match.
    with pytest.warns(UserWarning, match=r"shellout failed \("):
        _sink(tmp_path)
    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert root.attrs["git_sha"] == "unknown"


def test_per_key_group_gets_minimal_pointer(tmp_path: Path) -> None:
    sink = _sink(tmp_path)
    sink.stage((0, 1), value=np.array([1]), attrs={"note": "n"})
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    group = root["0/1"]
    assert group.attrs["run_id"] == "test-run"
    assert group.attrs["git_sha"]
    assert group.attrs["note"] == "n"
    # Minimal pointer only -- the full record stays on the root group.
    assert "created_at" not in dict(group.attrs)
    assert "git_branch" not in dict(group.attrs)
    assert "git_dirty" not in dict(group.attrs)


@pytest.mark.parametrize("field", ["git_sha", "git_branch", "git_dirty", "run_id", "created_at"])
def test_stage_rejects_reserved_core_field_names(tmp_path: Path, field: str) -> None:
    sink = _sink(tmp_path)
    with pytest.raises(ValueError, match="reserved core provenance"):
        sink.stage((0,), value=np.array([1]), attrs={field: "caller-overwrite"})
    # Failed stage must not have buffered anything.
    assert len(sink) == 0


_EXTENSION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"level": {"type": "string"}, "depth": {"type": "integer"}},
    "required": ["level"],
}


def _schema_sink(tmp_path: Path) -> ZarrStagingSink:
    spec = SinkSpec(  # type: ignore[assignment]
        run_id="test-run",
        output_dir=tmp_path / "out.zarr",
        format="zarr",
        flush_every=100,
        extension_schema=_EXTENSION_SCHEMA,
    )
    return ZarrStagingSink(spec)


def test_schema_type_violation_raises_at_stage_time(tmp_path: Path) -> None:
    sink = _schema_sink(tmp_path)
    with pytest.raises(ValueError, match="extension_schema.*level"):
        sink.stage((0,), value=np.array([1]), attrs={"level": 5})  # wrong JSON type
    assert len(sink) == 0  # nothing buffered -- validation precedes buffering


def test_schema_required_checked_when_attrs_staged(tmp_path: Path) -> None:
    sink = _schema_sink(tmp_path)
    with pytest.raises(ValueError, match="required field 'level'"):
        sink.stage((0,), value=np.array([1]), attrs={"depth": 2})
    assert len(sink) == 0


def test_schema_violation_on_merge_raises_immediately(tmp_path: Path) -> None:
    sink = _schema_sink(tmp_path)
    sink.stage((0,), attrs={"level": "backbone"})
    with pytest.raises(ValueError, match="extension_schema"):
        sink.stage((0,), value=np.array([1]), attrs={"level": 5})  # overwrite masks nothing
    sink.take((0,))


def test_split_stage_calls_satisfy_required_across_merge(tmp_path: Path) -> None:
    sink = _schema_sink(tmp_path)
    sink.stage((0,), attrs={"level": "backbone"})
    sink.stage((0,), value=np.array([1]), attrs={"depth": 2})
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    group = root["0"]
    assert group.attrs["level"] == "backbone"
    assert group.attrs["depth"] == 2
    assert group.attrs["run_id"] == "test-run"


def test_undeclared_keys_pass_through_with_schema_declared(tmp_path: Path) -> None:
    sink = _schema_sink(tmp_path)
    payload = {"a": [1, "two"]}
    sink.stage((0,), value=np.array([1]), attrs={"level": "x", "mystery": payload})
    sink.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert dict(root["0"].attrs["mystery"]) == payload


def test_finalize_consolidates_exactly_once_then_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real = zarr.consolidate_metadata

    def spy(store: object, **kwargs: object) -> object:
        calls.append(str(store))
        return real(store)  # type: ignore[arg-type]

    monkeypatch.setattr(zarr, "consolidate_metadata", spy)
    sink = _sink(tmp_path)
    sink.stage((0,), value=np.array([1]))
    sink.drain()
    sink.finalize()
    assert len(calls) == 1

    with pytest.raises(RuntimeError, match="only once"):
        sink.finalize()
    with pytest.raises(RuntimeError, match="finalize"):
        sink.drain()
    with pytest.raises(RuntimeError, match="finalize"):
        sink.stage((1,), value=np.array([2]))


def test_finalize_refuses_with_pending_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    real = zarr.consolidate_metadata

    def spy(store: object, **kwargs: object) -> object:
        calls.append(str(store))
        return real(store)  # type: ignore[arg-type]

    monkeypatch.setattr(zarr, "consolidate_metadata", spy)
    sink = _sink(tmp_path, flush_every=10)  # >1 so stage() does NOT auto-drain
    sink.stage((0,), value=np.array([1]))
    assert len(sink) == 1

    # Refusal must happen before any consolidation or locking.
    with pytest.raises(RuntimeError, match="drain\\(\\)"):
        sink.finalize()
    assert calls == []  # nothing was consolidated behind the caller's back
    assert len(sink) == 1  # payload neither stranded nor lost

    # After an explicit drain, finalize proceeds normally, exactly once.
    sink.drain()
    sink.finalize()
    assert len(calls) == 1


def test_second_sink_with_different_run_id_raises(tmp_path: Path) -> None:
    s1 = _sink(tmp_path, run_id="run-a")
    s1.stage((0,), value=np.array([1]))
    s1.drain()

    with pytest.raises(ValueError, match="run-a.*run-b|run-b.*run-a"):
        _sink(tmp_path, run_id="run-b")


def test_second_sink_with_same_run_id_allowed(tmp_path: Path) -> None:
    s1 = _sink(tmp_path, run_id="same")
    s1.stage((0,), value=np.array([1]))
    s1.drain()
    s2 = _sink(tmp_path, run_id="same")  # legitimate reopen: identical run identity
    s2.stage((1,), value=np.array([2]))
    s2.drain()

    root = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert np.array_equal(root["0"]["value"][:], [1])
    assert np.array_equal(root["1"]["value"][:], [2])


def test_derive_sink_spec_end_to_end(git_repo: Path) -> None:
    """Acceptance path (#4397): driver flow RunSpec -> derive_sink_spec -> store.

    Exercises the real public seam -- generated id must land in the actual
    zarr store's root provenance record, observed via zarr.open_group.
    """
    run_spec = RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None)
    spec = derive_sink_spec(run_spec, output_dir=git_repo.parent / "out.zarr")
    assert re.match(r"^run-[0-9a-f]{12}$", spec.run_id)  # type: ignore[arg-type]

    sink = make_sink(spec)
    assert isinstance(sink, ZarrStagingSink)
    sink.stage((0,), value=np.array([1]))
    sink.drain()

    root = zarr.open_group(str(git_repo.parent / "out.zarr"), mode="r")
    assert root.attrs["run_id"] == spec.run_id


def test_derive_sink_spec_explicit_override_reaches_store(git_repo: Path) -> None:
    """Explicit override wins precedence AND reaches the store verbatim."""
    run_spec = RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None, run_id="run-driver")
    spec = derive_sink_spec(
        run_spec, run_id="run-override", output_dir=git_repo.parent / "out.zarr"
    )
    sink = make_sink(spec)
    sink.stage((0,), value=np.array([2]))
    sink.drain()
    root = zarr.open_group(str(git_repo.parent / "out.zarr"), mode="r")
    assert root.attrs["run_id"] == "run-override"
