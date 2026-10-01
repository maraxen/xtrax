"""Durable atomic-commit primitives for zarr v3 directory stores.

Provides crash-safe, verifiable staging + rename patterns for incremental
writes to zarr stores. A caller stages arrays into a temporary directory,
then commits that directory as a named key in the store only after content
verification and fsync durability. Crash recovery: a committed key carries
its input digest and record digest, enabling resume-by-inspection (lookup
returns Reuse if the key already exists with matching input digest; Stale
if the digest differs; Corrupt if checksums fail; Missing if not yet
present).
"""

import errno
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from xtrax.run.digest import canonical_digest
from xtrax.run.zarr_integrity import (
    fsync_directory,
    fsync_tree,
    normalize_json_value,
    zarr_content_digest,
)

COMMIT_ATTR = "xtrax.commit"
STORE_ATTR = "xtrax.store"
FAULT_ENV = "XTRAX_FAULT_INJECT"


class DurableStoreError(Exception):
    """Base exception for durable store errors."""

    pass


class CommitConflictError(DurableStoreError):
    """Raised when commit_key detects a conflict at the target key."""

    def __init__(
        self,
        key: tuple[str, ...],
        stored_record: "CommitRecord | None",
        attempted_input_digest: str,
        reason: str,
    ) -> None:
        self.key = key
        self.stored_record = stored_record
        self.attempted_input_digest = attempted_input_digest
        self.reason = reason
        msg = (
            f"CommitConflictError: key={key}, reason={reason}, "
            f"attempted_input_digest={attempted_input_digest}"
        )
        if stored_record:
            msg += f", stored_input_digest={stored_record.input_digest}"
        super().__init__(msg)


class UnknownPrefixError(DurableStoreError):
    """Raised when a key's prefix group does not exist, or is not a legal parent.

    A prefix is not a legal parent when it is itself a committed key, or lies
    inside one: committed keys are immutable, so nothing may be nested in them.
    """

    def __init__(
        self, key: tuple[str, ...], prefix: tuple[str, ...], reason: str | None = None
    ) -> None:
        self.key = key
        self.prefix = prefix
        self.reason = reason or "does not exist"
        msg = f"UnknownPrefixError: prefix {prefix} {self.reason} (for key {key})"
        super().__init__(msg)


class StoreIdentityMismatch(DurableStoreError):
    """Raised when store identity payload does not match."""

    def __init__(self, stored_payload: dict, current_payload: dict) -> None:
        self.stored_payload = stored_payload
        self.current_payload = current_payload
        stored_sorted = sorted(stored_payload.items())
        current_sorted = sorted(current_payload.items())
        diff_keys = sorted(set(k for k, _ in stored_sorted) | set(k for k, _ in current_sorted))
        diff_msg = ", ".join(
            f"{k}: stored={stored_payload.get(k)!r} vs current={current_payload.get(k)!r}"
            for k in diff_keys
            if stored_payload.get(k) != current_payload.get(k)
        )
        msg = f"StoreIdentityMismatch: {diff_msg}"
        super().__init__(msg)


class NotADurableStoreError(DurableStoreError):
    """Raised when a directory is not a durable store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        msg = f"NotADurableStoreError: {path} is not a durable store"
        super().__init__(msg)


@dataclass(frozen=True)
class CommitRecord:
    """A committed entry record (frozen; its dict fields are not hashable)."""

    input_digest: str
    input_payload: dict[str, Any]
    meta: dict[str, Any]
    content_digest: str
    run_id: str
    env: dict[str, Any]
    committed_at: str
    record_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CommitRecord":
        """Construct from a plain dict."""
        return cls(
            input_digest=data["input_digest"],
            input_payload=data.get("input_payload", {}),
            meta=data.get("meta", {}),
            content_digest=data["content_digest"],
            run_id=data["run_id"],
            env=data.get("env", {}),
            committed_at=data["committed_at"],
            record_digest=data["record_digest"],
        )


@dataclass(frozen=True)
class Committed:
    """Outcome: the key was newly committed."""

    record: CommitRecord


@dataclass(frozen=True)
class Duplicate:
    """Outcome: the key was already committed with matching input digest."""

    record: CommitRecord


@dataclass(frozen=True)
class Missing:
    """Lookup result: the key does not exist."""

    pass


@dataclass(frozen=True)
class Stale:
    """Lookup result: the key exists but with a different input digest."""

    record: CommitRecord


@dataclass(frozen=True)
class Corrupt:
    """Lookup result: the key exists but verification failed."""

    reason: str


@dataclass(frozen=True)
class Reuse:
    """Lookup result: the key exists with matching input digest and can be reused."""

    record: CommitRecord
    path: Path

    def open(self):
        """Open the reuse group read-only."""
        try:
            import zarr
        except ImportError as e:
            msg = "Reuse.open requires the optional 'zarr' dependency"
            raise ImportError(msg) from e
        return zarr.open_group(str(self.path), mode="r", use_consolidated=False)


def key_path(key: tuple[str, ...]) -> str:
    """Convert a key tuple to a forward-slash path.

    Raises:
        ValueError: If key is empty, any part is empty, ".", "..", or contains "/".
    """
    if not key:
        raise ValueError("key_path: key must be non-empty")
    parts = []
    for part in key:
        part_str = str(part)
        if not part_str or part_str in (".", "..") or "/" in part_str:
            raise ValueError(
                f"key_path: invalid part {part_str!r} (empty, '.', '..', or contains '/')"
            )
        parts.append(part_str)
    return "/".join(parts)


def staging_root(store: Path) -> Path:
    """Return the staging directory root (sibling of store with .staging suffix)."""
    return store.with_name(store.name + ".staging")


def _fault(step: str) -> None:
    """Inject a crash at the named step if FAULT_ENV is set."""
    if os.environ.get(FAULT_ENV) == step:
        os._exit(137)


def _has_commit_attr(group_dir: Path) -> bool:
    """Whether ``group_dir``'s zarr metadata carries ``COMMIT_ATTR`` (valid or tampered).

    Reads ``zarr.json`` directly (no zarr dependency, no array reads) so it is cheap
    enough to run on every ancestor of a key at commit time.
    """
    try:
        meta = json.loads((group_dir / "zarr.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    attrs = meta.get("attributes") if isinstance(meta, dict) else None
    return isinstance(attrs, dict) and COMMIT_ATTR in attrs


def _compute_record_digest(d: dict[str, Any]) -> str:
    """Compute the record digest without the record_digest field itself."""
    filtered = {k: v for k, v in d.items() if k != "record_digest"}
    return canonical_digest(filtered)


def write_staged_group(
    staged_dir: Path, arrays: Mapping[str, Any], attrs: Mapping[str, Any] | None = None
) -> None:
    """Write arrays and attrs to a staged zarr group.

    Args:
        staged_dir: Path to the staging directory (will be created as a new zarr root).
        arrays: Named arrays to write.
        attrs: Optional attrs dict to stamp on the group.
    """
    try:
        import zarr
    except ImportError as e:
        msg = "write_staged_group requires the optional 'zarr' dependency"
        raise ImportError(msg) from e

    import numpy as np

    staged_dir.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(staged_dir), mode="w")

    for name, array in arrays.items():
        arr = np.asarray(array)
        group.create_array(
            name=name,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=tuple(max(d, 1) for d in arr.shape),
        )
        # Ellipsis indexing works for 0-d, multi-d, and everything else.
        zarr_array = group[name]
        assert isinstance(zarr_array, zarr.Array)
        zarr_array[...] = arr

    if attrs:
        group.attrs.update(dict(attrs))

    _fault("staged_written")


def commit_key(
    store: Path,
    key: tuple[str, ...],
    staged_dir: Path,
    *,
    input_digest: str,
    input_payload: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    run_id: str,
    env: Mapping[str, Any],
) -> Committed | Duplicate:
    """Atomically commit a staged group into the store under the given key.

    Stages at staged_dir must have been created via write_staged_group or
    equivalent. This function computes the content digest, writes a commit
    record, fsyncs, and atomically renames the staged dir to the target key
    location inside the store. Recovery: if the rename fails with ENOTEMPTY/
    EEXIST, checks whether the existing group is a committed duplicate
    (matching input_digest) or a conflict.

    Args:
        store: The durable store root directory.
        key: A tuple of path components identifying this key.
        staged_dir: The staging directory (created but not yet in the store).
        input_digest: The input digest (passed through to the record).
        input_payload: Optional input payload (normalized and stored).
        meta: Optional metadata dict (stored as-is).
        run_id: The run ID (included in the record).
        env: Environment dict (included in the record).

    Returns:
        Committed if the key was newly committed.
        Duplicate if the key was already committed with matching input_digest.

    The post-rename parent-directory fsync is the final durability barrier: if it
    fails, the exception propagates for a key that is ALREADY visible in the store
    (the rename cannot be undone). Retrying the commit is safe and returns
    ``Duplicate``.

    Raises:
        UnknownPrefixError: If the target prefix does not exist as a zarr group, or
            the prefix is itself a committed key or lies inside one (committed keys
            are immutable; nothing is renamed and the staged dir is removed).
        CommitConflictError: If the target exists with a different input_digest.
    """
    try:
        import zarr
    except ImportError as e:
        msg = "commit_key requires the optional 'zarr' dependency"
        raise ImportError(msg) from e

    # Fail fast: the target prefix must already exist as a zarr group, and neither it
    # nor any ancestor below the store root may itself be a committed key (committed
    # keys are immutable: nesting a key inside one would invalidate its content digest).
    target = store / key_path(key)
    parent = target.parent
    parent_parts = tuple(parent.relative_to(store).parts)
    if not (parent / "zarr.json").is_file():
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise UnknownPrefixError(key, parent_parts)
    for depth in range(1, len(parent_parts) + 1):
        if _has_commit_attr(store.joinpath(*parent_parts[:depth])):
            shutil.rmtree(staged_dir, ignore_errors=True)
            raise UnknownPrefixError(
                key,
                parent_parts[:depth],
                "is a committed key; keys cannot be nested inside committed keys",
            )

    content_digest = zarr_content_digest(staged_dir)
    _fault("digested")

    # Normalize payloads.
    normalized_input_payload = dict(
        normalize_json_value(dict(input_payload)) if input_payload else {}
    )
    normalized_meta = dict(normalize_json_value(dict(meta)) if meta else {})

    record_dict = {
        "input_digest": input_digest,
        "input_payload": normalized_input_payload,
        "meta": normalized_meta,
        "content_digest": content_digest,
        "run_id": run_id,
        "env": dict(env),
        "committed_at": datetime.now(UTC).isoformat(),
    }
    record_dict["record_digest"] = _compute_record_digest(record_dict)

    # Write record to the staged dir.
    group = zarr.open_group(str(staged_dir), mode="r+")
    group.attrs[COMMIT_ATTR] = record_dict
    _fault("record_written")

    fsync_tree(staged_dir)
    fsync_directory(staged_dir.parent)
    _fault("fsynced")

    # Atomic rename.
    try:
        os.rename(str(staged_dir), str(target))
    except OSError as e:
        if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
            existing = read_record(target)
            shutil.rmtree(staged_dir)
            if existing is None:
                raise CommitConflictError(
                    key, None, input_digest, "uncommitted or tampered group already at target"
                ) from e
            if existing.record_digest != _compute_record_digest(existing.to_dict()):
                raise CommitConflictError(
                    key, existing, input_digest, "uncommitted or tampered group already at target"
                ) from e
            if existing.input_digest == input_digest:
                return Duplicate(existing)
            raise CommitConflictError(
                key, existing, input_digest, "different input_digest already committed"
            ) from e
        raise

    _fault("renamed")
    fsync_directory(parent)
    return Committed(CommitRecord.from_dict(record_dict))


def _read_record_status(group_dir: Path) -> tuple[CommitRecord | None, str | None]:
    """Read and verify the commit record of a group directory.

    Returns:
        ``(record, None)`` for a valid committed group, otherwise
        ``(None, reason)`` where reason distinguishes an uncommitted group
        (no ``xtrax.commit`` attr) from a tampered/malformed record.
    """
    try:
        import zarr
    except ImportError as e:
        msg = "read_record requires the optional 'zarr' dependency"
        raise ImportError(msg) from e

    if not (group_dir / "zarr.json").is_file():
        return None, "not a zarr group"

    try:
        group = zarr.open_group(str(group_dir), mode="r")
        record_data = group.attrs.get(COMMIT_ATTR)
    except (ValueError, KeyError, OSError) as e:
        return None, f"unreadable zarr group: {e}"

    if record_data is None:
        return None, "uncommitted group (no commit record)"
    if not isinstance(record_data, dict):
        return None, "commit record tampered or malformed (not a mapping)"

    data = cast(dict[str, Any], record_data)
    try:
        record = CommitRecord.from_dict(data)
    except (KeyError, TypeError):
        return None, "commit record tampered or malformed (missing fields)"

    if record.record_digest != _compute_record_digest(data):
        return None, "commit record tampered (record_digest mismatch)"
    return record, None


def read_record(group_dir: Path) -> CommitRecord | None:
    """Read and verify the commit record from a group directory.

    Returns None on any failure (not a group, no ``xtrax.commit`` attr, or
    ``record_digest`` verification failure).
    """
    record, _ = _read_record_status(group_dir)
    return record


def lookup(
    store: Path, key: tuple[str, ...], input_digest: str, *, verify: bool = True
) -> Reuse | Missing | Stale | Corrupt:
    """Look up a key in the store and classify its state.

    Args:
        store: The store root directory.
        key: The key tuple.
        input_digest: The input digest to match against.
        verify: If True, recompute content_digest on disk and compare.

    Returns:
        Reuse: Key exists with matching input_digest and optionally verified content.
        Missing: Key does not exist.
        Stale: Key exists but input_digest differs.
        Corrupt: Key exists but verification failed or record is invalid.

    Raises:
        OSError: If recomputing the content digest hits an environmental I/O
            error (it is propagated, never classified as ``Corrupt``).
        MemoryError: Likewise propagated.
    """
    target = store / key_path(key)
    if not target.exists():
        return Missing()
    record, reason = _read_record_status(target)
    if record is None:
        return Corrupt(reason or "unreadable commit record")

    if record.input_digest != input_digest:
        return Stale(record)

    if verify:
        from zarr.errors import BaseZarrError

        try:
            actual_digest = zarr_content_digest(target)
        except MemoryError:
            raise
        except BaseZarrError as e:
            # Zarr's missing-node errors are FileNotFoundErrors too; a node whose
            # metadata is gone is corruption, not an environmental I/O failure.
            return Corrupt(f"failed to compute content digest ({type(e).__name__}): {e}")
        except OSError:
            # Environmental I/O failure (EIO, EMFILE, ...): says nothing about the
            # shard's integrity, so it must not be reported as Corrupt.
            raise
        except Exception as e:
            return Corrupt(f"failed to compute content digest ({type(e).__name__}): {e}")
        if actual_digest != record.content_digest:
            return Corrupt("content digest mismatch")

    return Reuse(record, target)


def committed_keys(store: Path, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """List all committed keys under the given prefix in the store.

    Returns a sorted list of key tuples (relative to store root).
    """
    if prefix:
        search_root = store / key_path(prefix)
    else:
        search_root = store

    if not search_root.exists():
        return []

    committed = []

    def walk(current_dir: Path, path_parts: list[str]) -> None:
        """Recursively walk and collect committed keys."""
        if not current_dir.is_dir():
            return
        # Check if this dir is a committed key.
        record = read_record(current_dir)
        if record is not None:
            # This is a committed key; don't descend.
            committed.append(tuple(path_parts))
            return
        # Not a committed key; try descending into children.
        try:
            children = sorted(current_dir.iterdir())
        except (OSError, PermissionError):
            return
        for child in children:
            if child.is_dir():
                walk(child, path_parts + [child.name])

    walk(search_root, list(prefix))
    return sorted(committed)


_GC_SUFFIX_RE = re.compile(r"\.gc-[0-9a-f]{8}$")


def gc_staging(store: Path, older_than: timedelta, *, exclude: Iterable[str] = ()) -> list[Path]:
    """Remove staging directories older than the given age.

    To narrow the stat-then-delete race with a writer that has just touched its
    directory, each candidate is first renamed to ``<name>.gc-<uuid8>`` (atomic;
    it vanishes from its writer-visible name at once) and only then deleted. A
    candidate that disappears in between is skipped. Directories already carrying
    a ``.gc-`` suffix belong to another collector (or to one that crashed mid-delete):
    they are never renamed again and never reported, but an old one is swept with a
    best-effort delete so a crashed collector cannot leak it forever.

    Args:
        store: The store root directory.
        older_than: Timedelta; directories older than now - older_than are removed.
        exclude: Directory names to skip.

    Returns:
        Sorted list of removed paths (the original, pre-rename names).
    """
    staging = staging_root(store)
    if not staging.exists():
        return []

    now = time.time()
    exclude_set = set(exclude)
    removed = []

    try:
        children = sorted(staging.iterdir())
    except (OSError, PermissionError):
        return []

    for child in children:
        if child.name in exclude_set:
            continue
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except (OSError, PermissionError):
            continue
        if now - mtime <= older_than.total_seconds():
            continue
        if _GC_SUFFIX_RE.search(child.name):
            shutil.rmtree(child, ignore_errors=True)
            continue
        condemned = child.with_name(f"{child.name}.gc-{uuid.uuid4().hex[:8]}")
        try:
            os.rename(str(child), str(condemned))
        except OSError:
            # Gone (another collector won) or not renameable: leave it alone.
            continue
        try:
            shutil.rmtree(condemned)
        except OSError:
            continue
        removed.append(child)

    return sorted(removed)


def create_store(
    store: Path,
    *,
    identity_payload: Mapping[str, Any],
    creator_run_id: str,
    prefixes: Sequence[tuple[str, ...]],
    writer_id: str,
) -> bool:
    """Create a new durable store with prefixes and identity.

    Stages a root zarr group in a temporary directory, creates all prefix
    groups, then atomically renames into place. Recovery: if the rename
    fails with ENOTEMPTY/EEXIST (someone else created it), removes the
    staging directory and returns False.

    Args:
        store: The desired store location.
        identity_payload: Identity payload (normalized and stored on root).
        creator_run_id: Run ID of the creator.
        prefixes: List of prefix tuples to create (including parents).
        writer_id: Writer ID (used for staging directory isolation).

    Returns:
        True if the store was created.
        False if the store already existed (created by another process).
    """
    try:
        import zarr
    except ImportError as e:
        msg = "create_store requires the optional 'zarr' dependency"
        raise ImportError(msg) from e

    staging = staging_root(store)
    writer_staging = staging / writer_id
    root_staged = writer_staging / "__root__"

    root_staged.mkdir(parents=True, exist_ok=True)

    # Create root zarr group.
    root = zarr.open_group(str(root_staged), mode="w")

    normalized_identity = dict(normalize_json_value(dict(identity_payload)))
    identity_digest = canonical_digest(normalized_identity)
    root.attrs[STORE_ATTR] = {
        "identity_payload": normalized_identity,
        "identity_digest": identity_digest,
        "creator_run_id": creator_run_id,
        "created_at": datetime.now(UTC).isoformat(),
    }

    # Create all prefix groups.
    for prefix in prefixes:
        prefix_path = key_path(prefix)
        root.require_group(prefix_path)

    _fault("root_staged")

    fsync_tree(root_staged)
    fsync_directory(writer_staging)
    fsync_directory(staging)

    try:
        os.rename(str(root_staged), str(store))
    except OSError as e:
        if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
            shutil.rmtree(root_staged, ignore_errors=True)
            return False
        raise

    _fault("root_renamed")
    fsync_directory(store.parent)
    # root_staged has been renamed away; drop only this writer's now-empty dir.
    # Never touch the shared staging root: other writers' groups live there.
    try:
        os.rmdir(writer_staging)
    except OSError:
        pass
    return True


def open_store(
    store: Path,
    *,
    identity_payload: Mapping[str, Any],
    prefixes: Sequence[tuple[str, ...]],
) -> dict[str, Any]:
    """Open an existing durable store and verify identity.

    Never writes; only reads and validates.

    Args:
        store: The store root directory.
        identity_payload: Expected identity payload (must match).
        prefixes: List of prefix tuples that must exist.

    Returns:
        The store record dict (STORE_ATTR).

    Raises:
        NotADurableStoreError: If the store has no STORE_ATTR.
        StoreIdentityMismatch: If identity_payload doesn't match stored identity.
        UnknownPrefixError: If any prefix group is missing.
    """
    try:
        import zarr
    except ImportError as e:
        msg = "open_store requires the optional 'zarr' dependency"
        raise ImportError(msg) from e

    if not (store / "zarr.json").exists():
        raise NotADurableStoreError(store)

    try:
        root = zarr.open_group(str(store), mode="r")
    except (ValueError, KeyError):
        raise NotADurableStoreError(store)

    store_record = root.attrs.get(STORE_ATTR)
    if store_record is None:
        raise NotADurableStoreError(store)

    normalized_identity = dict(normalize_json_value(dict(identity_payload)))
    expected_digest = canonical_digest(normalized_identity)
    if not isinstance(store_record, dict):
        raise NotADurableStoreError(store)
    stored_payload = dict(normalize_json_value(store_record.get("identity_payload", {})))

    if store_record.get("identity_digest") != expected_digest:
        raise StoreIdentityMismatch(stored_payload, normalized_identity)

    for prefix in prefixes:
        prefix_path = key_path(prefix)
        try:
            child = root[prefix_path]
            if not hasattr(child, "attrs"):
                raise UnknownPrefixError(prefix, prefix)
        except (KeyError, ValueError):
            raise UnknownPrefixError(prefix, prefix)

    return dict(store_record)
