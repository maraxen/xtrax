"""Output sink routing configuration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xtrax.run.ident import new_run_id
from xtrax.run.spec import RunSpec

if TYPE_CHECKING:
    from xtrax.run.memory_sink import MemorySink
    from xtrax.run.zarr_sink import ZarrStagingSink

Format = Literal["jsonl", "h5", "zarr", "none", "memory"]
OpenMode = Literal["exclusive", "create_or_join"]

_OPEN_MODES = ("exclusive", "create_or_join")


@dataclass(frozen=True)
class GitProvenance:
    """Precomputed git state for a sink.

    Inject this (or a mapping with the same fields, or a :class:`~pathlib.Path`
    package/repo root) via :attr:`SinkSpec.provenance`. A ``Path`` is the only
    form that shells out, and it shells out against that path -- never
    :meth:`pathlib.Path.cwd`.
    """

    git_sha: str
    git_branch: str
    git_dirty: bool = False


@dataclass(frozen=True)
class SinkReceipt:
    """Completion record returned by a sink's ``finalize()``.

    ``digest`` is reproducible for identical logical content (provenance such
    as run id, seed, git state, and producer version is not part of it).
    ``digest_algo_version`` is :data:`xtrax.run.zarr_integrity.DIGEST_ALGO_VERSION`.
    ``path`` is the store directory for a Zarr sink and ``None`` for an
    in-memory sink.
    """

    path: Path | None
    digest: str
    digest_algo_version: int
    run_id: str
    seed: int | None = None


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
    provenance: GitProvenance | Path | Mapping[str, Any] | None = None
    seed: int | None = None
    append: bool = False
    level_schemas: Mapping[int, Mapping[str, Any]] | None = None

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
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            msg = f"SinkSpec.seed must be int or None, got {type(self.seed).__name__}"
            raise TypeError(msg)
        if not isinstance(self.append, bool):
            msg = f"SinkSpec.append must be bool, got {type(self.append).__name__}"
            raise TypeError(msg)
        if self.provenance is not None and not isinstance(
            self.provenance, (GitProvenance, Path, Mapping)
        ):
            msg = (
                "SinkSpec.provenance must be a GitProvenance, pathlib.Path, mapping, or None, "
                f"got {type(self.provenance).__name__}"
            )
            raise TypeError(msg)
        if isinstance(self.provenance, Mapping):
            for field in ("git_sha", "git_branch"):
                value = self.provenance.get(field)
                if not isinstance(value, str):
                    msg = f"SinkSpec.provenance[{field!r}] must be str, got {type(value).__name__}"
                    raise TypeError(msg)
            dirty = self.provenance.get("git_dirty", False)
            if not isinstance(dirty, bool):
                msg = f"SinkSpec.provenance['git_dirty'] must be bool, got {type(dirty).__name__}"
                raise TypeError(msg)
        if self.level_schemas is not None:
            normalized_levels: dict[int, dict[str, Any]] = {}
            for depth, schema in self.level_schemas.items():
                if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
                    msg = f"SinkSpec.level_schemas keys must be non-negative ints, got {depth!r}"
                    raise ValueError(msg)
                if not isinstance(schema, Mapping):
                    msg = (
                        f"SinkSpec.level_schemas[{depth}] must be a mapping, "
                        f"got {type(schema).__name__}"
                    )
                    raise TypeError(msg)
                normalized_levels[depth] = dict(schema)
            self.level_schemas = normalized_levels


def make_sink(spec: SinkSpec) -> ZarrStagingSink | MemorySink | None:
    """Construct the sink implementation named by ``spec.format``.

    ``"zarr"`` returns a :class:`~xtrax.run.zarr_sink.ZarrStagingSink`,
    ``"memory"`` returns a :class:`~xtrax.run.memory_sink.MemorySink`, and
    ``"none"`` returns ``None``. ``"jsonl"``/``"h5"`` remain routing-only stub
    values pending their own writers.

    Raises:
        NotImplementedError: If ``spec.format`` has no writer yet.
    """
    if spec.format == "none":
        return None
    if spec.format == "memory":
        from xtrax.run.memory_sink import MemorySink

        return MemorySink(spec)
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
    provenance: GitProvenance | Path | Mapping[str, Any] | None = None,
    append: bool = False,
    level_schemas: Mapping[int, Mapping[str, Any]] | None = None,
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
        provenance: Precomputed git state, or a path to capture git from.
            Forwarded. ``None`` (the default) does not shell out.
        append: When true, repeated drains extend arrays along the leading axis.
        level_schemas: Per-key-depth extension schemas, keyed by ``len(key)``.
            A depth present here overrides ``extension_schema`` for that depth.

    The derived spec's ``seed`` is ``run_spec.seed``. ``output_dir=None`` falls
    back to ``run_spec.output_root`` (itself ``None`` by default).

    Returns:
        A fully-resolved ``SinkSpec`` whose ``run_id`` is never empty --
        falsy values are additionally rejected at sink construction (#96).
    """
    resolved_dir = run_spec.output_root if output_dir is None else output_dir
    return SinkSpec(
        run_id=run_id or run_spec.run_id or new_run_id(),
        output_dir=resolved_dir,
        format=format,
        flush_every=flush_every,
        extension_schema=extension_schema,
        open_mode=open_mode,
        store_identity=store_identity,
        prefixes=prefixes,
        provenance=provenance,
        seed=run_spec.seed,
        append=append,
        level_schemas=level_schemas,
    )
