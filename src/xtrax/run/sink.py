"""Output sink routing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from xtrax.run.zarr_sink import ZarrStagingSink


@dataclass
class SinkSpec:
    """Routing config for output sinks.

    ``run_id`` is required: it is the join key linking everything a sink writes
    to the run that produced it (see ZarrStagingSink provenance tracking).
    """

    run_id: str
    output_dir: Path | None = None
    format: Literal["jsonl", "h5", "zarr", "none"] = "jsonl"
    flush_every: int = 1
    extension_schema: dict[str, Any] | None = None


def make_sink(spec: SinkSpec) -> ZarrStagingSink | None:
    """Construct the sink implementation named by ``spec.format``.

    Only ``"zarr"`` and ``"none"`` are backed by a real implementation today;
    ``"jsonl"``/``"h5"`` remain routing-only stub values pending their own
    writers.

    Raises:
        NotImplementedError: If ``spec.format`` has no writer yet.
    """
    if spec.format == "none":
        return None
    if spec.format == "zarr":
        from xtrax.run.zarr_sink import ZarrStagingSink

        return ZarrStagingSink(spec)
    msg = f"make_sink: format {spec.format!r} has no writer implementation yet"
    raise NotImplementedError(msg)
