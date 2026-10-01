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

import copy
import os
import shutil
import subprocess
import uuid
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from xtrax.run import zarr_commit as zc
from xtrax.run._sink_names import CORE_PROVENANCE_FIELDS, RESERVED_ATTR_PREFIX
from xtrax.run.digest import canonical_digest, numerics_env
from xtrax.run.sink import SinkSpec
from xtrax.run.zarr_integrity import normalize_json_value

#: Core provenance field names written by the sink itself (defined in
#: ``_sink_names`` to avoid an import cycle). Caller-staged attrs may not use
#: these names (collision raises at :meth:`ZarrStagingSink.stage`).
_CORE_PROVENANCE_FIELDS = CORE_PROVENANCE_FIELDS

#: Reserved top-level prefix holding one committed group per durable writer
#: (``("_xtrax_writers", run_id)``): the append-only lineage of a durable store.
WRITERS_PREFIX = "_xtrax_writers"

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

    Durable mode: ``SinkSpec(open_mode="create_or_join", store_identity=...)``
    opens a multi-writer, crash-safe store instead. The sink never opens the
    target with ``mode="a"`` and never rewrites root attrs; the root is either
    created atomically (renamed into place from staging, carrying the store
    identity and every ``SinkSpec.prefixes`` group) or joined after an identity
    and prefix check. Each sink then commits a writer record at
    ``("_xtrax_writers", run_id)`` before ``__init__`` returns, so a writer
    record always precedes that writer's shards. ``stage`` requires
    ``input_digest``; ``drain`` writes every pending key into
    ``<store>.staging/<writer_id>/`` and atomically renames it into the store,
    returning ``{key: Committed | Duplicate}`` -- nothing is ever written in
    place. A different ``input_digest`` already committed at a key raises
    :class:`~xtrax.run.zarr_commit.CommitConflictError`. ``lookup``,
    ``committed_keys`` and ``gc_staging`` expose resume-by-inspection;
    ``close()`` (or ``with``) removes this writer's staging directory and ends
    the sink's use (undrained keys are discarded with a warning); ``finalize()``
    raises because consolidated metadata would go stale as writers add groups.
    Every key's parent must equal or start with a declared prefix. Exclusive mode
    refuses to open a directory that is already a durable store. ``run_id`` must
    be unique per process (use :func:`xtrax.run.ident.new_run_id`).
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
        if spec.open_mode == "create_or_join":
            # run_id becomes a key part (the writer record) and a staging directory
            # name: reject anything that is not a valid single key part BEFORE any
            # filesystem work, so a bad id never leaves a half-created store behind.
            try:
                zc.key_path((spec.run_id,))
            except ValueError as e:
                msg = (
                    f"ZarrStagingSink: run_id {spec.run_id!r} is not usable in durable "
                    f"mode (it becomes a key part and a directory name): {e}"
                )
                raise ValueError(msg) from e

        try:
            import zarr
        except ImportError as e:
            msg = (
                "ZarrStagingSink requires the optional 'zarr' dependency. "
                "Install with: pip install xtrax[io]"
            )
            raise ImportError(msg) from e

        self._spec = spec
        self._durable = spec.open_mode == "create_or_join"
        self._closed = False
        # Durable mode resolves symlinks up front so staging lands next to the REAL
        # target (the rename into the store must stay on one filesystem).
        self._output_dir: Path = (
            Path(spec.output_dir).resolve() if self._durable else spec.output_dir
        )
        self._root: zarr.Group
        if not self._durable:
            self._root = zarr.open_group(str(spec.output_dir), mode="a")
            if zc.STORE_ATTR in self._root.attrs:
                msg = (
                    f"ZarrStagingSink: output_dir {str(spec.output_dir)!r} is a durable store "
                    f"(its root carries {zc.STORE_ATTR!r}); exclusive mode would rewrite the "
                    "root attrs. Open it with SinkSpec(open_mode='create_or_join')."
                )
                raise ValueError(msg)
        self._pending: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._pending_attrs: dict[tuple[Any, ...], dict[str, Any]] = {}
        # Durable-only per-key commit inputs: input_digest/input_payload/commit_meta/env_extra.
        self._pending_commit: dict[tuple[Any, ...], dict[str, Any]] = {}
        # Durable-only: outcomes of committed keys not yet returned by an explicit drain().
        self._unreported_outcomes: dict[tuple[Any, ...], zc.Committed | zc.Duplicate] = {}
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

        if self._durable:
            self._init_durable()
            return

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

    def _init_durable(self) -> None:
        """Create-or-join the durable store, then commit this writer's record.

        Never opens the target with ``mode="a"`` and never writes root attrs.

        Raises:
            StoreIdentityMismatch: If an existing store has a different identity.
            UnknownPrefixError: If a required prefix group is missing.
            NotADurableStoreError: If the target is not a durable store.
            ValueError: If this ``run_id`` already joined the store, or the store and its
                staging directory are on different filesystems.
        """
        import zarr

        spec = self._spec
        output_dir = self._output_dir
        identity = spec.store_identity
        assert identity is not None  # enforced by SinkSpec.__post_init__
        self._writer_id = f"{spec.run_id}-{uuid.uuid4().hex[:8]}"
        prefixes: list[tuple[str, ...]] = []
        for prefix in (*spec.prefixes, (WRITERS_PREFIX,)):
            if prefix not in prefixes:
                prefixes.append(prefix)
        writer_dir = zc.staging_root(output_dir) / self._writer_id
        try:
            if not output_dir.exists() or self._is_empty_dir(output_dir):
                # An existing EMPTY directory counts as absent: rename over an empty
                # directory is atomic on POSIX (a racing loser still gets ENOTEMPTY).
                # A False return means another process won the creation race:
                # fall through to join.
                zc.create_store(
                    output_dir,
                    identity_payload=identity,
                    creator_run_id=spec.run_id,
                    prefixes=prefixes,
                    writer_id=self._writer_id,
                )
            self.store_record: dict[str, Any] = zc.open_store(
                output_dir, identity_payload=identity, prefixes=prefixes
            )
            self._check_same_filesystem()
            self._root = zarr.open_group(str(output_dir), mode="r+", use_consolidated=False)
            self._process_env: dict[str, Any] = numerics_env()
            self._commit_writer_record()
        except BaseException:
            shutil.rmtree(writer_dir, ignore_errors=True)
            raise

    @staticmethod
    def _is_empty_dir(path: Path) -> bool:
        """Whether ``path`` is an existing directory with no entries."""
        return path.is_dir() and not any(path.iterdir())

    def _check_same_filesystem(self) -> None:
        """Staging lives beside the store; a rename across devices would not be atomic.

        Raises:
            ValueError: If the staging root's parent and the store are on different devices.
        """
        staging_parent = zc.staging_root(self._output_dir).parent
        staging_dev = os.stat(staging_parent).st_dev
        store_dev = os.stat(self._output_dir).st_dev
        if staging_dev != store_dev:
            msg = (
                f"ZarrStagingSink: staging directory parent {str(staging_parent)!r} (device "
                f"{staging_dev}) and the store {str(self._output_dir)!r} (device {store_dev}) are "
                "on different filesystems; the atomic rename into the store would not be atomic "
                "(or would fail with EXDEV). Use an output_dir that is not itself a mount point."
            )
            raise ValueError(msg)

    def _commit_writer_record(self) -> None:
        """Commit ``("_xtrax_writers", run_id)`` -- before any shard of this writer."""
        run_id = self._spec.run_id
        key = (WRITERS_PREFIX, run_id)
        staged_dir = zc.staging_root(self._output_dir) / self._writer_id / zc.key_path(key)
        zc.write_staged_group(staged_dir, {}, {})
        try:
            outcome = zc.commit_key(
                self._output_dir,
                key,
                staged_dir,
                input_digest=canonical_digest({"run_id": run_id, "env": self._process_env}),
                input_payload={
                    "run_id": run_id,
                    "git_sha": self._provenance["git_sha"],
                    "git_branch": self._provenance["git_branch"],
                    "created_at": self._provenance["created_at"],
                },
                meta={},
                run_id=run_id,
                env=self._process_env,
            )
        except zc.CommitConflictError as e:
            raise ValueError(self._duplicate_run_id_message(run_id)) from e
        if isinstance(outcome, zc.Duplicate):
            raise ValueError(self._duplicate_run_id_message(run_id))

    @staticmethod
    def _duplicate_run_id_message(run_id: str) -> str:
        return (
            f"ZarrStagingSink: run_id {run_id!r} already joined store; run ids must be "
            "unique per process (use new_run_id())"
        )

    def _require_open(self, what: str) -> None:
        if self._closed:
            msg = (
                f"ZarrStagingSink: {what}() after close() is not legitimate -- "
                "close() ends this sink's use; open a fresh sink."
            )
            raise RuntimeError(msg)

    def _require_durable(self, what: str) -> None:
        if not self._durable:
            msg = (
                f"ZarrStagingSink.{what}() is only available in durable mode "
                "(SinkSpec(open_mode='create_or_join'))"
            )
            raise RuntimeError(msg)

    def _prepare_commit_info(
        self,
        key: tuple[Any, ...],
        input_digest: str | None,
        input_payload: Mapping[str, Any] | None,
        commit_meta: Mapping[str, Any] | None,
        env_extra: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Validate the durable-only stage() arguments; ``None`` in exclusive mode."""
        supplied = any(v is not None for v in (input_digest, input_payload, commit_meta, env_extra))
        if not self._durable:
            if supplied:
                msg = (
                    "ZarrStagingSink: input_digest/input_payload/commit_meta/env_extra are "
                    "durable-only arguments; open the sink with open_mode='create_or_join'"
                )
                raise ValueError(msg)
            return None
        if input_digest is None:
            msg = f"ZarrStagingSink: stage() in durable mode requires input_digest (key={key!r})"
            raise ValueError(msg)
        try:
            zc.key_path(key)
        except ValueError as e:
            msg = f"ZarrStagingSink: invalid durable key {key!r}: {e}"
            raise ValueError(msg) from e
        if str(key[0]) == WRITERS_PREFIX:
            msg = (
                f"ZarrStagingSink: key={key!r} is under the reserved prefix "
                f"{WRITERS_PREFIX!r}, which holds the sink's own writer records"
            )
            raise ValueError(msg)
        allowed = [p for p in self._spec.prefixes if p and p[0] != WRITERS_PREFIX]
        parent = tuple(str(part) for part in key[:-1])
        if not any(parent[: len(p)] == p for p in allowed):
            declared = [list(p) for p in allowed]
            msg = (
                f"ZarrStagingSink: key={key!r} is not under a declared prefix; its parent "
                f"{parent!r} must equal or start with one of SinkSpec.prefixes {declared} "
                "(a committed key cannot be nested inside another committed key, and "
                "keys outside the declared prefixes are not part of this store's layout)"
            )
            raise ValueError(msg)
        extra = dict(normalize_json_value(dict(env_extra))) if env_extra else {}
        collisions = sorted(set(extra).intersection(self._process_env["numerics"]))
        if collisions:
            msg = (
                f"ZarrStagingSink: env_extra for key={key!r} collides with process "
                f"numerics field(s) {collisions}; per-key fields may not shadow them"
            )
            raise ValueError(msg)
        return {
            "input_digest": input_digest,
            "input_payload": dict(input_payload) if input_payload else {},
            "commit_meta": dict(commit_meta) if commit_meta else {},
            "env_extra": extra,
        }

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
            merged = dict(existing.attrs) if hasattr(existing, "attrs") else {}
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
        *,
        input_digest: str | None = None,
        input_payload: Mapping[str, Any] | None = None,
        commit_meta: Mapping[str, Any] | None = None,
        env_extra: Mapping[str, Any] | None = None,
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
                In durable mode an auto-flush commits durably but does NOT
                report: its ``Committed``/``Duplicate`` outcomes are kept and
                returned by the next explicit :meth:`drain` (so the pattern
                ``stage(k, ...); outcome = drain()[k]`` works at any
                ``flush_every``, including the default of 1).
            **arrays: Named numpy-convertible arrays to stage under ``key``.
                Repeated ``stage`` calls for the same key merge: later names
                overwrite earlier ones with the same name, new names
                accumulate alongside existing ones.
            input_digest: Durable mode only; REQUIRED there. The digest of the
                inputs that determine this key's content; a later writer staging
                the same key with the same digest gets ``Duplicate``, a
                different one ``CommitConflictError``.
            input_payload: Durable mode only. Human-readable inputs stored in
                the commit record (for diffing a ``Stale`` lookup).
            commit_meta: Durable mode only. Extra metadata stored in the
                commit record's ``meta``.
            env_extra: Durable mode only. Per-key numerics fields merged into
                the commit record's ``env["numerics"]``; a name colliding with a
                process-level field raises. Repeated ``stage`` calls for the same
                key overwrite the earlier ``input_digest``/``input_payload``/
                ``commit_meta``/``env_extra``.

        Raises:
            ValueError: If ``attrs`` uses a reserved core provenance field
                name, the reserved ``"xtrax."`` namespace, or a value violates
                a ``spec.extension_schema`` type (nothing is buffered); or if
                this call triggers an auto-flush whose drain finds a key missing
                ``required`` fields (this call's payload stays buffered -- see
                :meth:`drain`). In durable mode, also if ``input_digest`` is
                missing, ``key`` is empty/invalid, under the reserved writer
                prefix, or its parent is not equal to / nested under one of the
                declared ``spec.prefixes``, or ``env_extra`` collides with a process
                field. In exclusive mode, if any durable-only argument is passed.
            RuntimeError: If the sink has already been finalized or closed.
            CommitConflictError: Durable mode only; if this call triggers an
                auto-flush (``spec.flush_every``) whose drain finds a different
                ``input_digest`` already committed at a pending key. The call's
                payload stays buffered (see :meth:`drain`).
        """
        self._require_open("stage")
        if self._finalized:
            msg = (
                "ZarrStagingSink: stage() after finalize() is not legitimate -- "
                "finalize() ends the run; use a fresh sink."
            )
            raise RuntimeError(msg)
        commit_info = self._prepare_commit_info(
            key, input_digest, input_payload, commit_meta, env_extra
        )
        if attrs:
            self._validate_stage_attrs(key, attrs)
        # Convert BEFORE touching the buffer: a conversion error (e.g. a ragged list)
        # on a new key must not leave an orphan empty entry in ``_pending``.
        converted = {name: np.asarray(value) for name, value in arrays.items()}
        entry = self._pending.setdefault(key, {})
        entry.update(converted)
        if attrs:
            self._pending_attrs.setdefault(key, {}).update(attrs)
        if commit_info is not None:
            self._pending_commit[key] = commit_info
        self._staged_since_drain += 1
        if self._staged_since_drain >= self._spec.flush_every:
            if self._durable:
                # Auto-flush: commit, but leave the outcomes unreported for the next
                # explicit drain() to return.
                self._validate_required_before_drain()
                self._drain_durable()
            else:
                self.drain()

    def take(self, key: tuple[Any, ...]) -> dict[str, np.ndarray]:
        """Pop and return a still-buffered (not yet drained) payload for ``key``.

        Discards any pending ``attrs`` staged for ``key`` -- ``take`` is for
        in-memory access without persisting; use ``drain`` to persist.

        Raises:
            KeyError: If ``key`` has no pending (undrained) entry.
        """
        self._pending_attrs.pop(key, None)
        self._pending_commit.pop(key, None)
        try:
            return self._pending.pop(key)
        except KeyError as e:
            msg = f"ZarrStagingSink: no pending entry for key={key!r}"
            raise KeyError(msg) from e

    def _drain_durable(self) -> None:
        """Commit every pending key atomically, recording each outcome as unreported.

        Outcomes accumulate in ``self._unreported_outcomes`` the moment each key
        commits (so they survive a later key's exception); only an explicit
        :meth:`drain` returns and clears them.
        """
        for key in list(self._pending):
            arrays = self._pending[key]
            info = self._pending_commit.get(key)
            if info is None:
                msg = (
                    f"ZarrStagingSink.drain: durable key {key!r} is buffered without commit "
                    "inputs (input_digest etc.); this is an internal inconsistency, not a "
                    "caller error. The pending buffer was NOT cleared."
                )
                raise RuntimeError(msg)
            key_rel = zc.key_path(key)
            staged_dir = zc.staging_root(self._output_dir) / self._writer_id / key_rel
            if staged_dir.exists():
                shutil.rmtree(staged_dir)
            attrs = dict(self._pending_attrs.get(key, {}))
            # Minimal provenance pointer on the key's own group (as in exclusive mode).
            attrs.update(
                {"run_id": self._provenance["run_id"], "git_sha": self._provenance["git_sha"]}
            )
            try:
                zc.write_staged_group(staged_dir, arrays, attrs)
            except Exception as e:
                context = (
                    f"ZarrStagingSink.drain: failed staging arrays {sorted(arrays)} "
                    f"for staged key {key!r}; the pending buffer was NOT cleared"
                )
                _prefix_message(e, context)
                raise
            env = copy.deepcopy(self._process_env)
            env["numerics"].update(info["env_extra"])
            self._unreported_outcomes[key] = zc.commit_key(
                self._output_dir,
                key,
                staged_dir,
                input_digest=info["input_digest"],
                input_payload=info["input_payload"],
                meta=info["commit_meta"],
                run_id=self._spec.run_id,
                env=env,
            )
            # Drop this key from the buffer as soon as it has its own outcome, so a
            # later key's CommitConflictError leaves already-committed keys out of
            # the buffer and only the failing + later keys buffered.
            del self._pending[key]
            self._pending_attrs.pop(key, None)
            del self._pending_commit[key]
        self._staged_since_drain = 0

    def drain(self) -> dict[tuple[Any, ...], zc.Committed | zc.Duplicate]:
        """Write all pending payloads (and attrs) into the Zarr store, then clear the buffer.

        Exclusive mode: also re-writes the root provenance record idempotently
        and stamps a minimal ``run_id``/``git_sha`` pointer onto each drained
        key's group; returns ``{}``.

        Durable mode: each pending key (in insertion order) is written to
        ``<store>.staging/<writer_id>/<keypath>`` and atomically renamed into
        the store with a commit record; nothing is written in place and the root
        is never touched. Each key is removed from the buffer immediately after
        its own outcome, so if a later key raises
        :class:`~xtrax.run.zarr_commit.CommitConflictError` (or any other error)
        the keys committed before it are already out of the buffer and only the
        failing key and the keys after it remain buffered.

        Returns (durable mode): ``{key: Committed | Duplicate}`` for EVERY key
        committed since the previous explicit ``drain()`` -- including keys that
        an auto-flush (``spec.flush_every``, triggered from :meth:`stage`)
        committed in the meantime, plus this call's own. The returned outcomes
        are then cleared, so an immediately repeated ``drain()`` returns ``{}``.
        Auto-flushes never clear them. Outcomes recorded before an exception
        (from this drain or an earlier auto-flush) stay unreported and are
        returned by the next successful explicit ``drain()``. If the same key is
        committed twice in that window, the later outcome wins.

        Raises:
            ValueError: If ``spec.extension_schema`` declares ``required``
                fields and any key with staged attrs would be persisted
                without them (existing group attrs merged with pending ones).
                Raised before any write; the buffer is left intact.
            RuntimeError: If the sink has already been finalized or closed.
            CommitConflictError: Durable mode only; a different ``input_digest``
                is already committed at a key (see the buffer note above). The same
                error can also surface from :meth:`stage` when its auto-flush
                (``spec.flush_every``) hits the conflict.
        """
        self._require_open("drain")
        if self._finalized:
            msg = (
                "ZarrStagingSink: drain() after finalize() is not legitimate -- "
                "metadata was already consolidated for this run."
            )
            raise RuntimeError(msg)
        self._validate_required_before_drain()
        if self._durable:
            self._drain_durable()
            reported = self._unreported_outcomes
            self._unreported_outcomes = {}
            return reported
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
        return {}

    def stamp_reserved(self, key: tuple[Any, ...], name: str, payload: Mapping[str, Any]) -> None:
        """Stamp a reserved-namespace attr onto a key's group.

        The one sanctioned writer of ``xtrax.``-prefixed attrs. Such attrs
        are excluded from ``zarr_content_digest`` by default, ensuring that
        internal sink bookkeeping does not affect content-based reproducibility.

        Durable mode: only an already-committed key can be stamped (the root never
        is). The stamp is an in-place attr write on an otherwise immutable shard, not
        an atomic commit, so concurrent stamps of the same ``name`` on the same key
        by different writers are last-writer-wins.

        Args:
            key: The group address (tuple of path components; empty tuple for root).
            name: A non-empty, non-slash, non-dot name. Reserved names "commit"
                and "store" raise (owned by the sink). The full attr key written
                to the group is ``f"{RESERVED_ATTR_PREFIX}{name}"``.
            payload: A mapping to normalize, canonicalize, and write. Must be
                JSON-safe and pass ``canonical_json_bytes`` without error.

        Raises:
            RuntimeError: If the sink has already been finalized or closed.
            ValueError: If ``name`` is empty, contains "/" or ".", or is one of
                the reserved values ("commit", "store"); or if ``payload`` cannot
                be normalized to JSON (e.g. contains a set, object, or other
                non-JSON-serializable value).
        """
        self._require_open("stamp_reserved")
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
        if self._durable:
            if not key:
                msg = (
                    "ZarrStagingSink.stamp_reserved: durable stores never rewrite root attrs; "
                    "stamp a committed key instead"
                )
                raise ValueError(msg)
            import zarr

            existing = self._root.get(group_path)
            if not isinstance(existing, zarr.Group):
                msg = (
                    f"ZarrStagingSink.stamp_reserved: key={key!r} has no committed group in "
                    "this durable store; drain() it first (durable stores never create "
                    "groups in place)"
                )
                raise ValueError(msg)
            existing.attrs[RESERVED_ATTR_PREFIX + name] = normalized
            return
        group = self._root.require_group(group_path) if group_path else self._root
        group.attrs[RESERVED_ATTR_PREFIX + name] = normalized

    def finalize(self) -> None:
        """Signal run completion: consolidate store metadata exactly once.

        Calls ``zarr.consolidate_metadata()`` on the store. After this, no
        further ``stage()``/``drain()`` calls are legitimate on this instance.

        Raises:
            RuntimeError: If staged payloads are still pending (``drain()``
                them first -- finalize never strands buffered payloads), if
                called more than once on the same instance, or always in
                durable mode (consolidated metadata would go stale as writers
                add groups).
        """
        if self._durable:
            msg = "durable stores are never consolidated: writers keep adding groups"
            raise RuntimeError(msg)
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

    def lookup(
        self, key: tuple[str, ...], input_digest: str, *, verify: bool = True
    ) -> zc.Reuse | zc.Missing | zc.Stale | zc.Corrupt:
        """Classify ``key`` in the durable store (durable mode only).

        Delegates to :func:`xtrax.run.zarr_commit.lookup`.

        Raises:
            RuntimeError: In exclusive mode.
        """
        self._require_durable("lookup")
        return zc.lookup(self._output_dir, key, input_digest, verify=verify)

    def committed_keys(self, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
        """List committed keys under ``prefix`` (durable mode only).

        Raises:
            RuntimeError: In exclusive mode.
        """
        self._require_durable("committed_keys")
        return zc.committed_keys(self._output_dir, prefix)

    def gc_staging(self, older_than: timedelta) -> list[Path]:
        """Remove other writers' stale staging dirs (durable mode only).

        This writer's own staging directory is never removed here.

        Raises:
            RuntimeError: In exclusive mode.
        """
        self._require_durable("gc_staging")
        return zc.gc_staging(self._output_dir, older_than, exclude=[self._writer_id])

    def close(self) -> None:
        """Close the sink; later ``stage``/``drain``/``stamp_reserved`` raise ``RuntimeError``.

        Durable mode also removes this writer's own staging directory. Idempotent
        (a second call does nothing). Staged keys that were never drained are
        discarded, with a ``UserWarning`` naming how many.
        """
        if self._closed:
            return
        self._closed = True
        if self._pending:
            warnings.warn(
                f"ZarrStagingSink.close(): discarding {len(self._pending)} staged key(s) that "
                "were never drained (call drain() before close() to persist them).",
                UserWarning,
                stacklevel=2,
            )
            self._pending.clear()
            self._pending_attrs.clear()
            self._pending_commit.clear()
        if not self._durable:
            return
        shutil.rmtree(zc.staging_root(self._output_dir) / self._writer_id, ignore_errors=True)

    def __enter__(self) -> "ZarrStagingSink":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        """Number of keys currently buffered (not yet drained)."""
        return len(self._pending)
