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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from xtrax.run.sink import SinkSpec

#: Core provenance field names written by the sink itself. Caller-staged
#: attrs may not use these names (collision raises at :meth:`ZarrStagingSink.stage`).
_CORE_PROVENANCE_FIELDS = frozenset({"git_sha", "git_branch", "git_dirty", "run_id", "created_at"})

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


def _is_json_type(value: Any, pytypes: tuple[type, ...]) -> bool:  # noqa: ANN401
    # bool subclasses int, so JSON-Schema-wise they must be treated as disjoint types.
    if isinstance(value, bool):
        return bool in pytypes
    return isinstance(value, pytypes)


def _validate_attrs_against_schema(attrs: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Minimal stdlib-only validator: checks ``type``/``required``/``properties``.

    Follows JSON-Schema ``additionalProperties``-permitted semantics: only
    schema-declared keys are checked; any other key passes through untouched.
    Returns a list of violation descriptions (empty means valid).
    """
    errors: list[str] = []
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in attrs:
            errors.append(f"missing required field {name!r}")
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
    :meth:`finalize` once at run end to consolidate store metadata.
    """

    def __init__(self, spec: SinkSpec) -> None:
        if spec.format != "zarr":
            msg = f"ZarrStagingSink requires SinkSpec.format == 'zarr', got {spec.format!r}"
            raise ValueError(msg)
        if spec.output_dir is None:
            msg = "ZarrStagingSink requires SinkSpec.output_dir"
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
        """Fail loud at stage()-time: reserved-name collisions + extension-schema validity."""
        collisions = sorted(_CORE_PROVENANCE_FIELDS.intersection(attrs))
        if collisions:
            msg = (
                f"ZarrStagingSink: staged attrs for key={key!r} use reserved core provenance "
                f"field name(s) {collisions}; these are managed by the sink and may not be "
                "overwritten by caller attrs."
            )
            raise ValueError(msg)
        if self._spec.extension_schema is None:
            return
        # Validate the post-merge view for this key: repeated stage() calls
        # merge attrs, so required fields may arrive across calls, while any
        # invalid value still fails immediately (masking is impossible).
        merged = dict(self._pending_attrs.get(key, {}))
        merged.update(attrs)
        errors = _validate_attrs_against_schema(merged, self._spec.extension_schema)
        if errors:
            msg = (
                f"ZarrStagingSink: staged attrs for key={key!r} violate the SinkSpec "
                f"extension_schema: {'; '.join(errors)}"
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
                when ``spec.extension_schema`` is declared, attrs are
                validated against it immediately (before buffering).
            **arrays: Named numpy-convertible arrays to stage under ``key``.
                Repeated ``stage`` calls for the same key merge: later names
                overwrite earlier ones with the same name, new names
                accumulate alongside existing ones.

        Raises:
            ValueError: If ``attrs`` uses a reserved core provenance field
                name, or violates ``spec.extension_schema``.
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
            RuntimeError: If the sink has already been finalized.
        """
        if self._finalized:
            msg = (
                "ZarrStagingSink: drain() after finalize() is not legitimate -- "
                "metadata was already consolidated for this run."
            )
            raise RuntimeError(msg)
        for key, arrays in self._pending.items():
            group_path = "/".join(str(part) for part in key)
            group = self._root.require_group(group_path) if group_path else self._root
            for name, array in arrays.items():
                arr = group.create_array(
                    name=name,
                    shape=array.shape,
                    dtype=array.dtype,
                    chunks=array.shape if array.shape else (1,),
                    overwrite=True,
                )
                arr[...] = array
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

    def finalize(self) -> None:
        """Signal run completion: consolidate store metadata exactly once.

        Calls ``zarr.consolidate_metadata()`` on the store. After this, no
        further ``stage()``/``drain()`` calls are legitimate on this instance.

        Raises:
            RuntimeError: If called more than once on the same instance.
        """
        if self._finalized:
            msg = "ZarrStagingSink: finalize() already ran for this sink; it may run only once."
            raise RuntimeError(msg)
        import zarr

        zarr.consolidate_metadata(str(self._spec.output_dir))
        self._finalized = True

    def __len__(self) -> int:
        """Number of keys currently buffered (not yet drained)."""
        return len(self._pending)
