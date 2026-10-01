"""Zarr-backed staging sink for JAX io_callback-driven streaming output.

Generic host-side sink: callers stage arbitrary named numpy-array payloads
under an opaque key (e.g. batch/chunk indices), then drain them into a
chunked, on-disk Zarr store. Domain-specific dispatch -- deciding WHICH
payload maps to which JAX op, and firing the actual
``jax.experimental.io_callback`` -- is the caller's responsibility; this
module only owns staging and Zarr storage.

Provenance tracking: the sink auto-captures static run provenance at
construction time (git SHA/branch/dirty status, ``SinkSpec.run_id``, and a
UTC creation timestamp) and stamps it onto the store's root group, plus a
minimal ``run_id``/``git_sha`` pointer on each drained key's own group. See
task ``260824_default-sink-provenance-tracking``.

Requires the optional ``zarr`` dependency: ``pip install xtrax[io]``. Zarr
itself is imported lazily inside :meth:`ZarrStagingSink.__init__`, so
importing this module (or ``xtrax.run``) never requires zarr to be
installed -- only constructing a sink does.
"""

import subprocess
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xtrax.run.sink import SinkSpec

#: Core provenance field names written by the sink itself. Caller-staged
#: attrs may not use these names (collision raises at :meth:`ZarrStagingSink.stage`).
_CORE_PROVENANCE_FIELDS = frozenset({"git_sha", "git_branch", "git_dirty", "run_id", "created_at"})

#: Prefix reserved for xtrax-written attrs. Attrs whose key starts with this
#: prefix are excluded from ``zarr_content_digest`` by default (unless
#: ``include_provenance=True``), ensuring that internal sink bookkeeping
#: (stamped via :meth:`ZarrStagingSink.stamp_reserved`) does not affect
#: content-based reproducibility.
RESERVED_ATTR_PREFIX = "xtrax."

_GIT_UNKNOWN = "unknown"

_JSON_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


class _GitCaptureFailed(Exception):
    """Git state could not be determined; ``cause`` names which of the causes applied."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause


def _capture_git_state(cwd: Path) -> tuple[str, str, bool]:
    """Capture ``(sha, branch, dirty)`` via git shellout (bathos GitState-style).

    Raises:
        _GitCaptureFailed: With a human-readable cause naming which failure
            applied: missing git binary, not a repository, or a failing shellout.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.PIPE
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.PIPE
            ).strip()
        )
    except FileNotFoundError as e:
        raise _GitCaptureFailed("the 'git' executable was not found on PATH") from e
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        if "not a git repository" in stderr.lower():
            raise _GitCaptureFailed("the working directory is not inside a git repository") from e
        raise _GitCaptureFailed(f"a git shellout failed ({e.cmd})") from e
    return sha, branch, dirty


def _prefix_message(e: Exception, context: str) -> None:
    """Put `context` into `e`'s own rendered message, in place (#5552).

    JAX's io_callback re-renders only an exception's message line, so a note or a
    chained cause is invisible there. `OSError(errno, strerror)` renders from
    `strerror`, not `args[0]`; most others render from a str `args[0]`. Anything
    whose `str()` still lacks the context afterwards gets a note as a last resort.
    """
    if isinstance(e, OSError) and isinstance(e.strerror, str):
        e.strerror = f"{context}: {e.strerror}"
    elif e.args and isinstance(e.args[0], str):
        e.args = (f"{context}: {e.args[0]}", *e.args[1:])
    if context not in str(e):
        e.add_note(context)


def _is_json_type(value: Any, pytypes: tuple[type, ...]) -> bool:  # noqa: ANN401
    # bool subclasses int, so JSON-Schema-wise they must be treated as disjoint types.
    if isinstance(value, bool):
        return bool in pytypes
    return isinstance(value, pytypes)


def _missing_required_fields(attrs: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Describe each schema ``required`` field absent from ``attrs`` (empty means complete)."""
    return [
        f"missing required field {name!r}"
        for name in schema.get("required", [])
        if name not in attrs
    ]


def _validate_attrs_against_schema(
    attrs: dict[str, Any], schema: dict[str, Any], *, check_required: bool = True
) -> list[str]:
    """Minimal stdlib-only validator: checks ``type``/``required``/``properties``.

    Follows JSON-Schema ``additionalProperties``-permitted semantics: only
    schema-declared keys are checked; any other key passes through untouched.
    ``check_required=False`` checks value types only -- the sink's stage()-time
    mode, since required fields may still arrive in a later stage() call.
    Returns a list of violation descriptions (empty means valid).
    """
    errors = _missing_required_fields(attrs, schema) if check_required else []
    properties = schema.get("properties", {})
    for name, value in attrs.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        declared = prop.get("type")
        if declared is None:
            continue
        allowed = declared if isinstance(declared, list) else [declared]
        pytypes = tuple(t for a in allowed if isinstance(a, str) for t in _JSON_TYPE_MAP.get(a, ()))
        if pytypes and not _is_json_type(value, pytypes):
            expected = "/".join(a for a in allowed if isinstance(a, str))
            got = type(value).__name__
            errors.append(f"field {name!r}: expected type {expected}, got {got}")
    return errors


class ZarrStagingSink:
    """Stages keyed numpy-array payloads for incremental drain into a chunked Zarr store.

    Each staged key maps to a nested Zarr group -- the key's components,
    stringified and joined by ``/``, become the group path -- and named
    arrays staged under that key become sibling arrays within the group.
    Writes are batched: :meth:`stage` buffers in memory and only touches
    disk once ``spec.flush_every`` stage calls have accumulated (or
    :meth:`drain` is called explicitly).

    Provenance: construction captures git SHA/branch/dirty status (never
    raising; falls back to ``git_sha="unknown"`` with a ``UserWarning``),
    plus ``spec.run_id`` and a UTC ``created_at`` timestamp, captured once.
    The full record lands on the store's root group; each drained key's own
    group gets a minimal ``run_id``/``git_sha`` pointer. Call
    :meth:`finalize` once at run end to consolidate store metadata; it
    refuses to run while staged payloads are undrained, so nothing is ever
    stranded silently.
    """

    def __init__(self, spec: SinkSpec) -> None:
        if spec.format != "zarr":
            msg = f"ZarrStagingSink requires SinkSpec.format == 'zarr', got {spec.format!r}"
            raise ValueError(msg)
        if spec.output_dir is None:
            msg = "ZarrStagingSink requires SinkSpec.output_dir"
            raise ValueError(msg)
        if not spec.run_id or not spec.run_id.strip():
            msg = (
                "ZarrStagingSink requires a non-blank SinkSpec.run_id -- it is the "
                "provenance join key stamped into the store. Derive one via "
                "xtrax.run ident helpers or pass an explicit id."
            )
            raise ValueError(msg)

        try:
            import zarr
        except ImportError as e:
            msg = (
                "ZarrStagingSink requires the optional 'zarr' dependency. "
                "Install with: pip install xtrax[io]"
            )
            raise ImportError(msg) from e

        self._spec = spec
        self._root: zarr.Group = zarr.open_group(str(spec.output_dir), mode="a")
        self._pending: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._pending_attrs: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._staged_since_drain = 0
        self._finalized = False

        # Core provenance record: captured once, here; re-written idempotently
        # (same values) on every drain().
        self._provenance: dict[str, Any] = {
            "run_id": spec.run_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        # Git capture must never raise, whatever the cause (broad outer wrapper
        # around the narrow per-command catches inside _capture_git_state).
        try:
            git_sha, git_branch, git_dirty = _capture_git_state(Path.cwd())
        except _GitCaptureFailed as e:
            self._provenance["git_sha"] = _GIT_UNKNOWN
            self._provenance["git_branch"] = _GIT_UNKNOWN
            self._provenance["git_dirty"] = False
            warnings.warn(
                f"ZarrStagingSink: could not determine git state ({e.cause}); "
                f"recording git_sha={_GIT_UNKNOWN!r} in the store's provenance record.",
                UserWarning,
                stacklevel=2,
            )
        except Exception as e:  # provenance capture alone must never raise
            self._provenance["git_sha"] = _GIT_UNKNOWN
            self._provenance["git_branch"] = _GIT_UNKNOWN
            self._provenance["git_dirty"] = False
            warnings.warn(
                f"ZarrStagingSink: could not determine git state (a git shellout failed "
                f"unexpectedly: {e!r}); recording git_sha={_GIT_UNKNOWN!r} in the store's "
                "provenance record.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._provenance["git_sha"] = git_sha
            self._provenance["git_branch"] = git_branch
            self._provenance["git_dirty"] = git_dirty

        # Multi-run reuse of one output_dir is legitimate (mode='a'), but a
        # silent root-record overwrite would orphan the earlier run's per-key
        # pointers -- refuse instead.
        existing_run_id = self._root.attrs.get("run_id")
        if existing_run_id is not None and existing_run_id != spec.run_id:
            msg = (
                f"ZarrStagingSink: output_dir {str(spec.output_dir)!r} already holds "
                f"provenance for run_id {existing_run_id!r}; refusing to open it for "
                f"run_id {spec.run_id!r} (per-key pointers from the earlier run would "
                "be orphaned). Use a fresh output_dir per run_id."
            )
            raise ValueError(msg)
        self._write_root_provenance()

    def _write_root_provenance(self) -> None:
        """Stamp the full core provenance record onto the store's root group."""
        self._root.attrs.update(dict(self._provenance))

    def _validate_stage_attrs(self, key: tuple[Any, ...], attrs: dict[str, Any]) -> None:
        """Fail loud at stage()-time: reserved-name collisions + extension-schema value types.

        Required fields are deliberately NOT checked here -- see
        :meth:`_validate_required_before_drain`.
        """
        collisions = sorted(_CORE_PROVENANCE_FIELDS.intersection(attrs))
        if collisions:
            msg = (
                f"ZarrStagingSink: staged attrs for key={key!r} use reserved core provenance "
                f"field name(s) {collisions}; these are managed by the sink and may not be "
                "overwritten by caller attrs."
            )
            raise ValueError(msg)
        reserved = sorted(k for k in attrs if str(k).startswith(RESERVED_ATTR_PREFIX))
        if reserved:
            msg = (
                f"ZarrStagingSink: staged attrs for key={key!r} use reserved namespace "
                f"{RESERVED_ATTR_PREFIX!r} (attr name(s) {reserved}); this namespace is "
                "reserved for the sink's own use. Use stamp_reserved() to write reserved attrs."
            )
            raise ValueError(msg)
        if self._spec.extension_schema is None:
            return
        # Type-check the post-merge view for this key: any invalid value fails
        # immediately (a later overwrite cannot mask it). Required fields may
        # still arrive in a later stage() call, so completeness is drain's job.
        merged = dict(self._pending_attrs.get(key, {}))
        merged.update(attrs)
        errors = _validate_attrs_against_schema(
            merged, self._spec.extension_schema, check_required=False
        )
        if errors:
            msg = (
                f"ZarrStagingSink: staged attrs for key={key!r} violate the SinkSpec "
                f"extension_schema: {'; '.join(errors)}"
            )
            raise ValueError(msg)

    def _validate_required_before_drain(self) -> None:
        """Fail loud before drain() writes anything: every key with staged attrs is schema-complete.

        Checks the view drain() will leave on disk -- the key group's existing
        attrs merged with its pending attrs -- so required fields may arrive
        across stage() calls and across earlier drains. Raises before any
        write, leaving the buffer intact for the caller to complete and retry.
        """
        schema = self._spec.extension_schema
        if schema is None or not schema.get("required"):
            return
        failures: list[str] = []
        for key, pending in self._pending_attrs.items():
            group_path = "/".join(str(part) for part in key)
            existing = self._root.get(group_path) if group_path else self._root
            merged = dict(existing.attrs) if existing is not None else {}
            merged.update(pending)
            missing = _missing_required_fields(merged, schema)
            if missing:
                failures.append(f"key={key!r}: {'; '.join(missing)}")
        if failures:
            msg = (
                "ZarrStagingSink: drain() refuses to persist attrs that are incomplete "
                f"under the SinkSpec extension_schema ({' | '.join(failures)}); nothing "
                "was written and the buffer is intact -- stage() the missing fields, "
                "then drain() again."
            )
            raise ValueError(msg)

    def stage(
        self,
        key: tuple[Any, ...],
        attrs: dict[str, Any] | None = None,
        **arrays: Any,  # noqa: ANN401
    ) -> None:
        """Buffer one or more named arrays (and optional metadata) under ``key``.

        Args:
            key: Opaque hashable tuple identifying this payload, e.g.
                ``(batch_idx, chunk_start, chunk_count)``.
            attrs: Optional JSON-safe metadata (scalars, strings, lists of
                either) written to the Zarr group's ``.attrs`` on drain --
                e.g. provenance fields that aren't themselves arrays.
                Repeated ``stage`` calls for the same key merge attrs the
                same way arrays merge (later keys overwrite earlier ones).
                Attrs keys colliding with core provenance field names raise;
                attrs keys starting with the reserved namespace (``"xtrax."``)
                also raise; when ``spec.extension_schema`` is declared, attr
                value types are validated immediately (before buffering) against
                the merged view, while ``required`` fields are enforced at
                :meth:`drain` -- so they may be split across stage() calls.
                An auto-flush (``spec.flush_every``) is a drain: split
                required fields across fewer calls than ``flush_every``.
            **arrays: Named numpy-convertible arrays to stage under ``key``.
                Repeated ``stage`` calls for the same key merge: later names
                overwrite earlier ones with the same name, new names
                accumulate alongside existing ones.

        Raises:
            ValueError: If ``attrs`` uses a reserved core provenance field
                name, the reserved ``"xtrax."`` namespace, or a value violates
                a ``spec.extension_schema`` type (nothing is buffered); or if
                this call triggers an auto-flush whose drain finds a key missing
                ``required`` fields (this call's payload stays buffered -- see
                :meth:`drain`).
            RuntimeError: If the sink has already been finalized.
        """
        if self._finalized:
            msg = (
                "ZarrStagingSink: stage() after finalize() is not legitimate -- "
                "finalize() ends the run; use a fresh sink."
            )
            raise RuntimeError(msg)
        if attrs:
            self._validate_stage_attrs(key, attrs)
        entry = self._pending.setdefault(key, {})
        entry.update({name: np.asarray(value) for name, value in arrays.items()})
        if attrs:
            self._pending_attrs.setdefault(key, {}).update(attrs)
        self._staged_since_drain += 1
        if self._staged_since_drain >= self._spec.flush_every:
            self.drain()

    def take(self, key: tuple[Any, ...]) -> dict[str, np.ndarray]:
        """Pop and return a still-buffered (not yet drained) payload for ``key``.

        Discards any pending ``attrs`` staged for ``key`` -- ``take`` is for
        in-memory access without persisting; use ``drain`` to persist.

        Raises:
            KeyError: If ``key`` has no pending (undrained) entry.
        """
        self._pending_attrs.pop(key, None)
        try:
            return self._pending.pop(key)
        except KeyError as e:
            msg = f"ZarrStagingSink: no pending entry for key={key!r}"
            raise KeyError(msg) from e

    def drain(self) -> None:
        """Write all pending payloads (and attrs) into the Zarr store, then clear the buffer.

        Also re-writes the root provenance record idempotently and stamps a
        minimal ``run_id``/``git_sha`` pointer onto each drained key's group.

        Raises:
            ValueError: If ``spec.extension_schema`` declares ``required``
                fields and any key with staged attrs would be persisted
                without them (existing group attrs merged with pending ones).
                Raised before any write; the buffer is left intact.
            RuntimeError: If the sink has already been finalized.
        """
        if self._finalized:
            msg = (
                "ZarrStagingSink: drain() after finalize() is not legitimate -- "
                "metadata was already consolidated for this run."
            )
            raise RuntimeError(msg)
        self._validate_required_before_drain()
        for key, arrays in self._pending.items():
            group_path = "/".join(str(part) for part in key)
            group = self._root.require_group(group_path) if group_path else self._root
            for name, array in arrays.items():
                try:
                    arr = group.create_array(
                        name=name,
                        shape=array.shape,
                        dtype=array.dtype,
                        # One chunk per array, rank-matched to its shape: 0-d gets
                        # chunks=() and zero-length dims get edge 1 (zarr rejects 0).
                        chunks=tuple(max(d, 1) for d in array.shape),
                        overwrite=True,
                    )
                    arr[...] = array
                except Exception as e:
                    # #5552: zarr's own message names only zarr internals, and from an
                    # io_callback JAX re-renders just the original exception's message
                    # line (no notes, no chained cause). So the context goes INTO the
                    # message, on the same exception object: type and traceback are
                    # unchanged for callers that catch it.
                    context = (
                        f"ZarrStagingSink.drain: failed writing array {name!r} "
                        f"(shape={tuple(array.shape)}, dtype={array.dtype}) "
                        f"for staged key {key!r} at group {group_path or '/'!r}; "
                        "the pending buffer was NOT cleared"
                    )
                    _prefix_message(e, context)
                    raise
            key_attrs = self._pending_attrs.get(key)
            if key_attrs:
                group.attrs.update(key_attrs)
            # Minimal provenance pointer on the key's own group, independent of
            # the root record -- survives the group being copied/exported alone.
            group.attrs.update(
                {
                    "run_id": self._provenance["run_id"],
                    "git_sha": self._provenance["git_sha"],
                }
            )
        self._pending.clear()
        self._pending_attrs.clear()
        self._staged_since_drain = 0
        self._write_root_provenance()

    def stamp_reserved(self, key: tuple[Any, ...], name: str, payload: Mapping[str, Any]) -> None:
        """Stamp a reserved-namespace attr onto a key's group.

        The one sanctioned writer of ``xtrax.``-prefixed attrs. Such attrs
        are excluded from ``zarr_content_digest`` by default, ensuring that
        internal sink bookkeeping does not affect content-based reproducibility.

        Args:
            key: The group address (tuple of path components; empty tuple for root).
            name: A non-empty, non-slash, non-dot name. Reserved names "commit"
                and "store" raise (owned by the sink). The full attr key written
                to the group is ``f"{RESERVED_ATTR_PREFIX}{name}"``.
            payload: A mapping to normalize, canonicalize, and write. Must be
                JSON-safe and pass ``canonical_json_bytes`` without error.

        Raises:
            RuntimeError: If the sink has already been finalized.
            ValueError: If ``name`` is empty, contains "/" or ".", or is one of
                the reserved values ("commit", "store"); or if ``payload`` cannot
                be normalized to JSON (e.g. contains a set, object, or other
                non-JSON-serializable value).
        """
        if self._finalized:
            msg = (
                "ZarrStagingSink: stamp_reserved() after finalize() is not legitimate -- "
                "metadata was already consolidated for this run."
            )
            raise RuntimeError(msg)
        # Validate name.
        if not name:
            msg = "ZarrStagingSink.stamp_reserved: name must be non-empty"
            raise ValueError(msg)
        if "/" in name or "." in name:
            msg = f"ZarrStagingSink.stamp_reserved: name={name!r} may not contain '/' or '.'"
            raise ValueError(msg)
        if name in {"commit", "store"}:
            msg = (
                f"ZarrStagingSink.stamp_reserved: name={name!r} is reserved by the sink itself "
                f"(owned by zarr consolidation or internal bookkeeping)"
            )
            raise ValueError(msg)
        # Normalize and validate payload is JSON-safe.
        try:
            from xtrax.run.zarr_integrity import canonical_json_bytes
            from xtrax.run.zarr_integrity import normalize_json_value as nv

            normalized = dict(nv(dict(payload)))
            canonical_json_bytes(normalized)
        except TypeError as e:
            msg = (
                f"ZarrStagingSink.stamp_reserved: key={key!r}, name={name!r}, "
                f"payload cannot be normalized to JSON-safe form: {e}"
            )
            raise ValueError(msg) from e
        except ValueError as e:
            msg = (
                f"ZarrStagingSink.stamp_reserved: key={key!r}, name={name!r}, "
                f"payload is not JSON-serializable: {e}"
            )
            raise ValueError(msg) from e
        # Write immediately to the group (not buffered).
        group_path = "/".join(str(p) for p in key) if key else ""
        group = self._root.require_group(group_path) if group_path else self._root
        group.attrs[RESERVED_ATTR_PREFIX + name] = normalized

    def finalize(self) -> None:
        """Signal run completion: consolidate store metadata exactly once.

        Calls ``zarr.consolidate_metadata()`` on the store. After this, no
        further ``stage()``/``drain()`` calls are legitimate on this instance.

        Raises:
            RuntimeError: If staged payloads are still pending (``drain()``
                them first -- finalize never strands buffered payloads), or if
                called more than once on the same instance.
        """
        if self._finalized:
            msg = "ZarrStagingSink: finalize() already ran for this sink; it may run only once."
            raise RuntimeError(msg)
        if self._pending:
            msg = (
                f"ZarrStagingSink: finalize() refuses to consolidate with "
                f"{len(self._pending)} staged key(s) undrained -- call drain() "
                "(to persist) or take() (to discard) first; finalize() will not "
                "strand buffered payloads silently."
            )
            raise RuntimeError(msg)
        import zarr

        zarr.consolidate_metadata(str(self._spec.output_dir))
        self._finalized = True

    def __len__(self) -> int:
        """Number of keys currently buffered (not yet drained)."""
        return len(self._pending)
