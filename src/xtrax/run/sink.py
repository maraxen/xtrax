"""Output sink routing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xtrax.run.ident import new_run_id
from xtrax.run.spec import RunSpec

if TYPE_CHECKING:
    from xtrax.run.zarr_sink import ZarrStagingSink

Format = Literal["jsonl", "h5", "zarr", "none"]


@dataclass
class SinkSpec:
    """Routing config for output sinks.

    ``run_id`` is required: it is the join key linking everything a sink writes
    to the run that produced it (see ZarrStagingSink provenance tracking).
    """

    run_id: str
    output_dir: Path | None = None
    format: Format = "jsonl"
    flush_every: int = 1
    extension_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Plain-Python backstop: beartype/jaxtyping wrapping is test-env-only
        # (conftest import hook), but the provenance contract must hold for
        # production drivers too.
        if not isinstance(self.run_id, str):
            msg = f"SinkSpec.run_id must be str, got {type(self.run_id).__name__}"
            raise TypeError(msg)


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


def derive_sink_spec(
    run_spec: RunSpec,
    *,
    run_id: str | None = None,
    output_dir: Path | None,
    format: Format = "zarr",
    flush_every: int = 1,
    extension_schema: dict[str, Any] | None = None,
) -> SinkSpec:
    """Derive a :class:`SinkSpec` from a :class:`RunSpec` -- the canonical seam.

    Drivers (and the future ``xtrax run`` CLI) call this instead of
    hand-building ``SinkSpec``, so provenance run ids follow one precedence:

    1. explicit ``run_id=`` override
    2. ``run_spec.run_id``
    3. a freshly generated id (:func:`xtrax.run.ident.new_run_id`)

    Note: precedence uses truthiness, so an explicitly passed empty string
    falls through to lower-precedence sources rather than raising; the
    fail-loud backstop for empty ids remains sink construction (#96).

    Note on defaults: this helper pins ``format="zarr"`` (the provenance seam
    it serves) while bare ``SinkSpec`` defaults to ``"jsonl"``. That divergence
    is deliberate (spec 260824).

    Args:
        run_spec: The execution config carrying optional static ``run_id``.
        run_id: Explicit override; wins over ``run_spec.run_id`` when given.
        output_dir: Sink output directory (keyword-required).
        format: Routing format; defaults to ``"zarr"``.
        flush_every: Flush cadence in writes; forwarded verbatim.
        extension_schema: Extension schema mapping; forwarded verbatim.

    Returns:
        A fully-resolved ``SinkSpec`` whose ``run_id`` is never empty --
        falsy values are additionally rejected at sink construction (#96).
    """
    return SinkSpec(
        run_id=run_id or run_spec.run_id or new_run_id(),
        output_dir=output_dir,
        format=format,
        flush_every=flush_every,
        extension_schema=extension_schema,
    )
