"""Output sink routing configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xtrax.run.ident import new_run_id
from xtrax.run.spec import RunSpec

if TYPE_CHECKING:
    from xtrax.run.zarr_sink import ZarrStagingSink

Format = Literal["jsonl", "h5", "zarr", "none"]
OpenMode = Literal["exclusive", "create_or_join"]

_OPEN_MODES = ("exclusive", "create_or_join")


@dataclass
class SinkSpec:
    """Routing config for output sinks.

    ``run_id`` is required: it is the join key linking everything a sink writes
    to the run that produced it (see ZarrStagingSink provenance tracking).

    ``open_mode="exclusive"`` (default) is the original single-writer mode.
    ``open_mode="create_or_join"`` opens a durable, multi-writer store: it
    requires ``format == "zarr"`` and a ``store_identity`` payload (the
    canonical identity every joiner must match). ``prefixes`` lists the group
    prefixes that must exist in the store (created atomically with the store
    root); it is normalized to a tuple of tuples of ``str``.
    """

    run_id: str
    output_dir: Path | None = None
    format: Format = "jsonl"
    flush_every: int = 1
    extension_schema: dict[str, Any] | None = None
    open_mode: OpenMode = "exclusive"
    store_identity: Mapping[str, Any] | None = None
    prefixes: Sequence[tuple[str, ...]] = ()

    def __post_init__(self) -> None:
        # Plain-Python backstop: beartype/jaxtyping wrapping is test-env-only
        # (conftest import hook), but the provenance contract must hold for
        # production drivers too.
        if not isinstance(self.run_id, str):
            msg = f"SinkSpec.run_id must be str, got {type(self.run_id).__name__}"
            raise TypeError(msg)
        if self.open_mode not in _OPEN_MODES:
            msg = f"SinkSpec.open_mode must be one of {_OPEN_MODES}, got {self.open_mode!r}"
            raise ValueError(msg)
        if self.open_mode == "create_or_join":
            if self.store_identity is None:
                msg = "SinkSpec(open_mode='create_or_join') requires store_identity"
                raise ValueError(msg)
            if self.format != "zarr":
                msg = (
                    "SinkSpec(open_mode='create_or_join') requires format='zarr', "
                    f"got {self.format!r}"
                )
                raise ValueError(msg)
        normalized: list[tuple[str, ...]] = []
        for prefix in self.prefixes:
            if isinstance(prefix, str):
                msg = (
                    f"SinkSpec.prefixes entries must be tuples of str, got bare str {prefix!r} "
                    f"(did you mean ({prefix!r},)?)"
                )
                raise TypeError(msg)
            normalized.append(tuple(str(part) for part in prefix))
        self.prefixes = tuple(normalized)


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
    open_mode: OpenMode = "exclusive",
    store_identity: Mapping[str, Any] | None = None,
    prefixes: Sequence[tuple[str, ...]] = (),
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
        open_mode: ``"exclusive"`` (default) or ``"create_or_join"``; forwarded.
        store_identity: Canonical store identity payload; forwarded (required
            for ``"create_or_join"``).
        prefixes: Group prefixes that must exist in a durable store; forwarded.

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
        open_mode=open_mode,
        store_identity=store_identity,
        prefixes=prefixes,
    )
