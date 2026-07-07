"""Tests for xtrax.run.zarr_sink.ZarrStagingSink."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.run.sink import SinkSpec
from xtrax.run.zarr_sink import ZarrStagingSink


def _sink(tmp_path: Path, flush_every: int = 1) -> ZarrStagingSink:
    spec = SinkSpec(output_dir=tmp_path / "out.zarr", format="zarr", flush_every=flush_every)
    return ZarrStagingSink(spec)


def test_requires_zarr_format() -> None:
    with pytest.raises(ValueError, match="zarr"):
        ZarrStagingSink(SinkSpec(format="jsonl"))


def test_requires_output_dir() -> None:
    with pytest.raises(ValueError, match="output_dir"):
        ZarrStagingSink(SinkSpec(format="zarr", output_dir=None))


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
