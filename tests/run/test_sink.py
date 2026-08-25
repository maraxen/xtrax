"""Tests for xtrax.run.sink: SinkSpec and the make_sink factory."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from xtrax.run.sink import SinkSpec, derive_sink_spec, make_sink
from xtrax.run.zarr_sink import ZarrStagingSink


def test_sink_spec_defaults() -> None:
    spec = SinkSpec(run_id="r")
    assert spec.run_id == "r"
    assert spec.output_dir is None
    assert spec.format == "jsonl"
    assert spec.flush_every == 1
    assert spec.extension_schema is None


def test_make_sink_none_format_returns_none() -> None:
    assert make_sink(SinkSpec(run_id="r", format="none")) is None


def test_make_sink_zarr_format_returns_zarr_staging_sink(tmp_path: Path) -> None:
    sink = make_sink(SinkSpec(run_id="r", output_dir=tmp_path / "out.zarr", format="zarr"))
    assert isinstance(sink, ZarrStagingSink)


@pytest.mark.parametrize("fmt", ["jsonl", "h5"])
def test_make_sink_unimplemented_formats_raise(fmt: str, tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match=fmt):
        make_sink(SinkSpec(run_id="r", output_dir=tmp_path, format=fmt))  # type: ignore[arg-type]


class TestDeriveSinkSpec:
    """derive_sink_spec: canonical seam for driver-side sink construction (#4397)."""

    def make_run_spec(self, **kwargs):
        from xtrax.run.spec import RunSpec

        return RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None, **kwargs)

    def test_explicit_override_wins(self) -> None:
        spec = self.make_run_spec(run_id="run-fromspec")
        derived = derive_sink_spec(spec, run_id="run-override", output_dir=None)
        assert derived.run_id == "run-override"

    def test_falls_back_to_run_spec_run_id(self) -> None:
        spec = self.make_run_spec(run_id="run-fromspec")
        derived = derive_sink_spec(spec, output_dir=None)
        assert derived.run_id == "run-fromspec"

    def test_generates_when_no_source(self) -> None:
        spec = self.make_run_spec()
        derived = derive_sink_spec(spec, output_dir=None)
        assert re.match(r"^run-[0-9a-f]{12}$", derived.run_id)

    def test_kwargs_forwarded_verbatim(self, tmp_path: Path) -> None:
        spec = self.make_run_spec()
        schema = {"trial": "int64"}
        derived = derive_sink_spec(
            spec,
            output_dir=tmp_path / "out.zarr",
            format="jsonl",
            flush_every=7,
            extension_schema=schema,
        )
        assert derived.output_dir == tmp_path / "out.zarr"
        assert derived.format == "jsonl"
        assert derived.flush_every == 7
        assert derived.extension_schema == schema

    def test_default_format_is_zarr(self) -> None:
        """Helper pins format='zarr' (deliberate divergence from bare SinkSpec)."""
        spec = self.make_run_spec()
        assert derive_sink_spec(spec, output_dir=None).format == "zarr"

    def test_output_dir_required(self) -> None:
        """output_dir is keyword-required: forces drivers to state intent."""
        spec = self.make_run_spec()
        with pytest.raises(TypeError):
            derive_sink_spec(spec)  # type: ignore[call-arg]


def test_sink_spec_run_id_type_enforced_outside_test_hook() -> None:
    """Production-path acceptance: without the conftest beartype hook, a plain
    TypeError still guards the provenance join key (jury B finding 1)."""
    import subprocess
    import sys

    code = (
        "from xtrax.run import SinkSpec\n"
        "for bad in (None, 4242):\n"
        "    try:\n"
        "        SinkSpec(run_id=bad)\n"
        "    except TypeError as e:\n"
        "        assert 'run_id' in str(e), e\n"
        "    else:\n"
        "        raise AssertionError(f'no TypeError for run_id={bad!r}')\n"
        "print('prod-guard-ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "prod-guard-ok"


def test_sink_spec_run_id_type_error_in_process() -> None:
    """In-process: non-str run_id raises TypeError naming the field."""
    with pytest.raises(TypeError, match="run_id"):
        SinkSpec(run_id=4242)  # type: ignore[arg-type]
