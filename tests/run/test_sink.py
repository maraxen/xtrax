"""Tests for xtrax.run.sink: SinkSpec and the make_sink factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from xtrax.run.sink import SinkSpec, make_sink
from xtrax.run.zarr_sink import ZarrStagingSink


def test_sink_spec_defaults() -> None:
    spec = SinkSpec()
    assert spec.output_dir is None
    assert spec.format == "jsonl"
    assert spec.flush_every == 1


def test_make_sink_none_format_returns_none() -> None:
    assert make_sink(SinkSpec(format="none")) is None


def test_make_sink_zarr_format_returns_zarr_staging_sink(tmp_path: Path) -> None:
    sink = make_sink(SinkSpec(output_dir=tmp_path / "out.zarr", format="zarr"))
    assert isinstance(sink, ZarrStagingSink)


@pytest.mark.parametrize("fmt", ["jsonl", "h5"])
def test_make_sink_unimplemented_formats_raise(fmt: str, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match=fmt):
        make_sink(SinkSpec(output_dir=tmp_path, format=fmt))  # type: ignore[arg-type]
