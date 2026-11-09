"""Local execution harnesses: sequential and process pool executors.

Covering [T3-02], [T3-03], GATE-07, and Sections 6.3.1, 6.3.2 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

import gc
import json
import multiprocessing
import os
import signal
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from xtrax.campaign.executors.protocol import CampaignSummary
from xtrax.campaign.lease import (
    FileLock,
    LeaseLostError,
    LockUnavailableError,
    StolenAttemptDiscarded,
)
from xtrax.campaign.marker import (
    CampaignDoneMarker,
    compute_content_digest,
    create_done_marker,
    quarantine_output,
    validate_unit_done,
)

#: ``_execute_single_unit`` outcomes.
UNIT_COMMITTED = "committed"
UNIT_LOCK_HELD = "lock_held"
UNIT_DISCARDED = "discarded"

#: How resume decides a completed unit is still valid. ``deep`` recomputes the store's content
#: digest against the done marker (reads every chunk); ``tier1`` checks only the marker and the
#: store's root metadata (the fast path, which trusts chunk bytes).
ResumeVerify = Literal["deep", "tier1"]
RESUME_VERIFY_MODES: tuple[str, ...] = ("deep", "tier1")

#: Log of units skipped because they were already done, one JSON object per line.
REUSE_LOG_NAME = "reuse_log.jsonl"


def _validate_resume_verify(resume_verify: str) -> ResumeVerify:
    """Return ``resume_verify`` narrowed to a supported mode, or raise ``ValueError``."""
    if resume_verify == "deep":
        return "deep"
    if resume_verify == "tier1":
        return "tier1"
    msg = f"resume_verify must be one of {', '.join(RESUME_VERIFY_MODES)}; got {resume_verify!r}"
    raise ValueError(msg)


class UnitProducedNoOutputError(RuntimeError):
    """Raised when a unit function returns without writing its output store."""


class UnitOutputNotAGroupError(ValueError):
    """Raised when a unit function writes a root zarr array instead of a group."""


class UnitInterruptedError(BaseException):
    """Raised inside a unit run when the process receives SIGTERM.

    A ``BaseException`` so that ``except Exception`` in unit code does not swallow it. It
    unwinds through the ``finally`` blocks of ``_execute_single_unit``, which release the
    lease and discard the partial output before the process exits.
    """


#: Sentinel: this unit run did not install a SIGTERM handler (not the main thread).
_NO_SIGTERM_HANDLER: Any = object()


if TYPE_CHECKING:
    from xtrax.campaign.manifest import CampaignManifest, ManifestRow


def _resolve_compilation_cache_dir(cache_dir: str | None) -> str | None:
    """Resolve the JAX persistent compilation cache directory, or None for off.

    Order: explicit ``cache_dir`` > ``JAX_COMPILATION_CACHE_DIR`` (``none`` or empty
    disables) > off. There is no default path: a cache is derived data and its location
    is configuration, never a directory invented under the current working directory.
    """
    if cache_dir:
        return cache_dir
    env_value = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if env_value is None or env_value.strip().lower() in ("", "none"):
        return None
    return env_value


def _initialize_worker_process(cache_dir: str | None = None) -> None:
    """Initialize worker process configuration for JAX persistent disk compilation cache.

    Per Section 6.3.2 of SPEC-261105-CAMPAIGN-RUNNER, each spawned worker configures
    jax_compilation_cache_dir to amortize compilation across recycled workers, but only
    when a cache directory is configured (see ``_resolve_compilation_cache_dir``).
    """
    resolved_cache = _resolve_compilation_cache_dir(cache_dir)
    if resolved_cache is None:
        return
    os.makedirs(resolved_cache, exist_ok=True)
    try:
        import jax
    except ImportError:
        return
    jax.config.update("jax_compilation_cache_dir", resolved_cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def _clean_memory(state: Any) -> None:
    """Execute device tensor blocking, delete state reference, and collect garbage.

    Per Section 6.3.1 of SPEC-261105-CAMPAIGN-RUNNER:
    jax.tree_util.tree_map(lambda x: getattr(x, "block_until_ready", lambda: x)(), state)
    del state
    gc.collect()
    """
    try:
        import jax

        jax.tree_util.tree_map(
            lambda x: getattr(x, "block_until_ready", lambda: x)(),
            state,
        )
    except (ImportError, Exception):
        if hasattr(state, "block_until_ready"):
            try:
                state.block_until_ready()
            except Exception:
                pass
    del state
    gc.collect()


def _resolve_output_path(output_dir: Path, row: "ManifestRow") -> Path:
    """Resolve the target output path for a manifest row."""
    resolved_dir = Path(output_dir).resolve()
    if row.output_locator:
        loc_path = Path(row.output_locator)
        if loc_path.is_absolute():
            return loc_path
        if len(loc_path.parts) > 1 and resolved_dir.name == loc_path.parts[0]:
            target = resolved_dir / Path(*loc_path.parts[1:])
        else:
            target = resolved_dir / loc_path
        # An absolute locator is allowed (the row hash covers it, #3308); a relative one
        # must stay under output_dir.
        if not Path(os.path.normpath(target)).is_relative_to(resolved_dir):
            raise ValueError(
                f"output_locator {row.output_locator!r} of job_id={row.job_id!r} escapes "
                f"output_dir {resolved_dir}; use an absolute locator for a store elsewhere"
            )
        return target
    return resolved_dir / f"{row.manifest_row_hash}.zarr"


def _check_resume_done(
    output_dir: Path,
    row: "ManifestRow",
    resume_verify: ResumeVerify = "deep",
) -> bool:
    """Return True when the unit's completed output may be reused instead of recomputed.

    A unit is done only when ``validate_unit_done`` accepts its resolved output path: marker
    present and matching, and the store present with valid root metadata. With
    ``resume_verify="deep"`` (the default) the store's content digest must also match the
    marker's ``content_digest_sha256``; ``"tier1"`` skips that byte-level check. A bare marker
    file is never enough.

    An I/O error while verifying returns False: the unit is not known to be done, so it
    goes to the leased run path, where the same error is raised and the unit is reported
    failed. Nothing is quarantined on an I/O error.
    """
    mode = _validate_resume_verify(resume_verify)
    output_path = _resolve_output_path(output_dir, row)
    try:
        val = validate_unit_done(
            output_path,
            row,
            deep_verify=mode == "deep",
            quarantine_on_failure=False,
        )
    except OSError:
        return False
    return val.is_valid


def _record_reuse(
    output_dir: Path,
    row: "ManifestRow",
    store_path: Path,
    verified: ResumeVerify,
    reused_by: str,
) -> None:
    """Append one JSON line to ``<output_dir>/reuse_log.jsonl`` for a skipped, done unit.

    The content digest is read from the unit's done marker. The line is written with one
    ``os.write`` on an ``O_APPEND`` descriptor so concurrent writers do not interleave.
    """
    marker_path = store_path.with_name(f"{store_path.name}.done.json")
    marker = CampaignDoneMarker.read(marker_path)
    record = {
        "manifest_row_hash": row.manifest_row_hash,
        "input_hash": row.input_hash,
        "store_path": str(store_path),
        "content_digest_sha256": marker.content_digest_sha256,
        "verified": verified,
        "reused_at_utc": datetime.now(UTC).isoformat(),
        "reused_by": reused_by,
    }
    data = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(
        output_dir / REUSE_LOG_NAME,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o644,
    )
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _set_aside_orphaned_output(
    output_path: Path,
    row: "ManifestRow",
    resume_verify: ResumeVerify = "deep",
    lock: FileLock | None = None,
) -> None:
    """Quarantine a store or done marker that does not form a valid completed unit.

    Called while holding the unit's lease, before the unit function runs. A store
    without a valid marker is left over from an attempt that crashed before stamping
    it; a marker without a valid store (for example, the store was deleted to force a
    rerun) would make the commit protocol discard this attempt as already done. A store
    that fails the configured resume verification (a corrupted chunk under ``deep``) is
    set aside too, so the recomputed output replaces it rather than being promoted over
    it. After this call neither exists, so whatever appears at ``output_path`` is this
    attempt's.

    With ``lock``, the lease is re-checked right before the quarantine (#3312): a holder
    that stalled after acquiring must not set aside the store of the worker that stole it.
    """
    marker_path = output_path.with_name(f"{output_path.name}.done.json")
    if not output_path.exists() and not marker_path.exists():
        return
    val = validate_unit_done(
        output_path,
        row,
        deep_verify=_validate_resume_verify(resume_verify) == "deep",
        quarantine_on_failure=False,
    )
    if not val.is_valid:
        if lock is not None:
            lock.assert_holder()
        quarantine_output(output_path, reason="orphaned_uncommitted_output")


def _write_state_store(state: Mapping[Any, Any], partial_path: Path) -> None:
    """Persist a unit function's returned mapping as a zarr v3 group at ``partial_path``.

    Values with one or more dimensions become arrays; scalars (including 0-d arrays)
    and JSON-serializable values become group attributes. Anything else is an error,
    never silently dropped.
    """
    import numpy as np
    import zarr

    from xtrax.campaign.manifest import normalize_json_value

    if not state:
        msg = "Unit function returned an empty mapping; there is no output to persist"
        raise UnitProducedNoOutputError(msg)

    arrays: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        if not key or "/" in key or key.startswith("."):
            msg = f"Unit output key {key!r} is not a valid zarr node name"
            raise ValueError(msg)
        if hasattr(value, "__array__") and not isinstance(value, (str, bytes)):
            arr = np.asarray(value)
            if arr.ndim > 0:
                arrays[key] = arr
                continue
            value = arr.item()
        try:
            normalized = normalize_json_value(value)
            json.dumps(normalized, allow_nan=False)
        except (TypeError, ValueError) as exc:
            msg = (
                f"Unit output {key!r} of type {type(value).__name__} is neither an array "
                f"nor a JSON-serializable value: {exc}"
            )
            raise TypeError(msg) from exc
        attrs[key] = normalized

    group = zarr.open_group(str(partial_path), mode="w", zarr_format=3)
    for key, arr in arrays.items():
        group.create_array(key, data=arr)
    if attrs:
        group.attrs.update(attrs)


def _require_group_store(output_path: Path) -> None:
    """Raise ``UnitOutputNotAGroupError`` if the unit wrote a root zarr array.

    The campaign content digest walks a zarr group, so a store whose root is an array
    could never be stamped done. Failing here names the problem; the abort path then sets
    the store aside like any other failed attempt.
    """
    if not output_path.is_dir():
        return
    if (output_path / ".zarray").exists():
        node_type = "array"
    else:
        meta = output_path / "zarr.json"
        try:
            node_type = json.loads(meta.read_text(encoding="utf-8")).get("node_type")
        except (OSError, ValueError, AttributeError):
            return
    if node_type == "array":
        msg = (
            f"Unit function wrote a root zarr array at {output_path}; campaign outputs must "
            "be zarr groups. Write the array inside a group (e.g. group.create_array(...)) "
            "or return a mapping of arrays"
        )
        raise UnitOutputNotAGroupError(msg)


def _stage_unit_output(
    state: Any, output_path: Path, partial_path: Path, lock: FileLock | None = None
) -> None:
    """Put this attempt's output at ``partial_path`` for the commit protocol.

    In order of precedence:

    1. The unit function wrote its own store at ``output_path``: move it to the partial.
       With ``lock``, the lease is re-checked first (#3312), so a stale holder never moves
       a stealer's store into its own partial, where abort cleanup would delete it.
    2. The unit function returned a ``Mapping``: write it as a zarr v3 group.
    3. Otherwise the unit fails. Stamping a done marker on missing output would record
       the unit as complete and resume would skip it forever.
    """
    if partial_path.exists():
        return
    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    if output_path.exists() and not done_marker_path.exists():
        _require_group_store(output_path)
        if lock is not None:
            lock.assert_holder()
        os.replace(output_path, partial_path)
        return
    if isinstance(state, Mapping):
        try:
            _write_state_store(state, partial_path)
        except BaseException:
            _discard_partial(partial_path)
            raise
        return
    msg = (
        f"Unit function neither wrote a store at {output_path} nor returned a mapping "
        f"(got {type(state).__name__}); refusing to stamp a done marker on missing output"
    )
    raise UnitProducedNoOutputError(msg)


def _still_holds(lock: FileLock) -> bool:
    """True when ``lock`` still holds its lease (see ``FileLock.assert_holder``)."""
    try:
        lock.assert_holder()
    except LeaseLostError:
        return False
    return True


def _discard_partial(partial_path: Path) -> None:
    """Remove a partially written attempt directory or file."""
    import shutil

    if partial_path.is_dir():
        shutil.rmtree(partial_path, ignore_errors=True)
    elif partial_path.exists():
        partial_path.unlink(missing_ok=True)


def _raise_unit_interrupted(signum: int, frame: Any) -> None:
    """SIGTERM handler for a unit run: unwind so cleanup in ``finally`` runs."""
    msg = f"unit run interrupted by signal {signum}"
    raise UnitInterruptedError(msg)


def _install_sigterm_interrupt() -> Any:
    """Route SIGTERM to ``UnitInterruptedError`` for the duration of one unit run.

    ``signal.signal`` only works in the main thread, so any other thread gets no handler
    and the sentinel ``_NO_SIGTERM_HANDLER`` is returned. Returns the previous handler
    otherwise, for ``_restore_sigterm``.
    """
    if threading.current_thread() is not threading.main_thread():
        return _NO_SIGTERM_HANDLER
    return signal.signal(signal.SIGTERM, _raise_unit_interrupted)


def _restore_sigterm(previous: Any) -> None:
    """Put back the handler returned by ``_install_sigterm_interrupt``."""
    if previous is _NO_SIGTERM_HANDLER:
        return
    # A handler installed from C returns None here, which signal.signal cannot restore.
    signal.signal(signal.SIGTERM, signal.SIG_DFL if previous is None else previous)


def _execute_single_unit(
    row: "ManifestRow",
    unit_fn: "Callable[[ManifestRow], Any]",
    output_dir: Path,
    lease_seconds: int = 1800,
    heartbeat_interval: float = 60.0,
    owner_token_prefix: str = "local",
    resume_verify: ResumeVerify = "deep",
) -> str:
    """Execute a single unit with atomic leasing, output promotion, and done marker stamping.

    Returns one of ``UNIT_COMMITTED`` (this attempt stamped the done marker),
    ``UNIT_LOCK_HELD`` (another worker holds the lease; nothing ran) or
    ``UNIT_DISCARDED`` (another worker finished the unit first). Raises on failure.

    Any exception, including ``UnitInterruptedError`` from SIGTERM, discards this attempt's
    ``.partial.<uuid>`` path and releases the lease, so a killed worker neither keeps a
    fresh lock nor leaves partial stores behind.
    """
    mode = _validate_resume_verify(resume_verify)
    output_path = _resolve_output_path(output_dir, row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    owner_token = f"{owner_token_prefix}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

    lock = FileLock(
        output_path=output_path,
        owner_token=owner_token,
        manifest_row_hash=row.manifest_row_hash,
        lease_seconds=lease_seconds,
        heartbeat_interval=heartbeat_interval,
    )

    try:
        lock.acquire()
    except LockUnavailableError:
        return UNIT_LOCK_HELD

    previous_sigterm = _install_sigterm_interrupt()
    partial_path: Path | None = None
    staged = False
    try:
        # Another worker may have committed between the resume check and our acquire,
        # or between _check_resume_done and _set_aside_orphaned_output.
        if _check_resume_done(output_dir, row, resume_verify=mode):
            _record_reuse(output_dir, row, output_path, mode, reused_by="lease_recheck")
            return UNIT_DISCARDED
        _set_aside_orphaned_output(output_path, row, resume_verify=mode, lock=lock)
        if validate_unit_done(
            output_path,
            row,
            deep_verify=mode == "deep",
            quarantine_on_failure=False,
        ).is_valid:
            _record_reuse(output_dir, row, output_path, mode, reused_by="lease_recheck")
            return UNIT_DISCARDED

        start_time = time.time()
        attempt_uuid = uuid.uuid4().hex[:8]
        partial_path = output_path.with_name(f"{output_path.name}.partial.{attempt_uuid}")

        state = unit_fn(row)
        duration_s = max(0.001, time.time() - start_time)

        try:
            if validate_unit_done(
                output_path,
                row,
                deep_verify=mode == "deep",
                quarantine_on_failure=False,
            ).is_valid:
                return UNIT_DISCARDED
            _stage_unit_output(state, output_path, partial_path, lock=lock)
            staged = True
        finally:
            # Memory hygiene: device tensor blocking & garbage collection
            _clean_memory(state)
            del state

        digest = compute_content_digest(partial_path)
        marker = create_done_marker(
            manifest_row_hash=row.manifest_row_hash,
            input_hash=row.input_hash,
            output_locator=str(output_path),
            content_digest_sha256=digest,
            duration_s=duration_s,
        )
        commit_res = lock.commit_output(
            partial_path=partial_path,
            output_path=output_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
        )
        if isinstance(commit_res, StolenAttemptDiscarded):
            return UNIT_DISCARDED
        return UNIT_COMMITTED
    finally:
        try:
            # After a successful commit the partial has been promoted, so this is a no-op.
            # On any failure it removes this attempt's output while the lease is still held.
            if partial_path is not None:
                _discard_partial(partial_path)
            if partial_path is not None and not staged and output_path.exists():
                val = validate_unit_done(
                    output_path,
                    row,
                    deep_verify=mode == "deep",
                    quarantine_on_failure=False,
                )
                # Only a current holder may set the store aside (#3312): after a steal it
                # is the stealer's in-progress output.
                if not val.is_valid and _still_holds(lock):
                    quarantine_output(output_path, reason="aborted_unit_output")
        finally:
            if lock.is_locked:
                lock.release()
            _restore_sigterm(previous_sigterm)


def _pool_worker_task(
    row: "ManifestRow",
    unit_fn: "Callable[[ManifestRow], Any]",
    output_dir: str,
    lease_seconds: int,
    heartbeat_interval: float,
    cache_dir: str | None,
    resume_verify: ResumeVerify = "deep",
    owner_token_prefix: str | None = None,
) -> dict[str, Any]:
    """Module-level task function executed in spawned worker processes.

    ``owner_token_prefix`` names the pool run that submitted the task, so the parent can
    recognise, and reclaim, a lease left behind by one of its own workers that died.
    """
    _initialize_worker_process(cache_dir)
    out_dir = Path(output_dir)
    try:
        outcome = _execute_single_unit(
            row=row,
            unit_fn=unit_fn,
            output_dir=out_dir,
            lease_seconds=lease_seconds,
            heartbeat_interval=heartbeat_interval,
            owner_token_prefix=owner_token_prefix or "pool",
            resume_verify=resume_verify,
        )
        return {
            "manifest_row_hash": row.manifest_row_hash,
            "success": True,
            "outcome": outcome,
            "error": None,
        }
    except (Exception, UnitInterruptedError) as exc:
        return {
            "manifest_row_hash": row.manifest_row_hash,
            "success": False,
            "outcome": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


#: Positional arguments of one pool task. A ``type`` alias is evaluated lazily, so it may
#: name ``ManifestRow``, which is imported only for type checking.
type _PoolTaskArgs = tuple[
    int,
    ManifestRow,
    Callable[[ManifestRow], Any],
    str,
    int,
    float,
    str | None,
    ResumeVerify,
]


def _pool_worker_task_wrapper(
    args: _PoolTaskArgs,
) -> "tuple[int, ManifestRow, dict[str, Any]]":
    """Unpack task arguments for pool.imap_unordered and return indexed result."""
    idx, row, unit_fn, output_dir, lease_seconds, heartbeat_interval, cache_dir, mode = args
    res = _pool_worker_task(
        row=row,
        unit_fn=unit_fn,
        output_dir=output_dir,
        lease_seconds=lease_seconds,
        heartbeat_interval=heartbeat_interval,
        cache_dir=cache_dir,
        resume_verify=mode,
    )
    return idx, row, res


class LocalSequentialExecutor:
    """Single-process sequential executor with memory and cache hygiene.

    Per Section 6.3.1 of SPEC-261105-CAMPAIGN-RUNNER:
    - Iterates through manifest rows sequentially in the host process.
    - Integrates FileLock with Tier-1 fast resume skipping.
    - Device tensor blocking (block_until_ready), del state, gc.collect().
    - Prohibits per-unit jax.clear_caches(); invokes cache clearing only upon
      arm backend or padding bucket transitions.
    """

    def __init__(
        self,
        *,
        lease_seconds: int = 1800,
        heartbeat_interval: float = 60.0,
        resume_verify: ResumeVerify = "deep",
    ) -> None:
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self.resume_verify = _validate_resume_verify(resume_verify)
        self._cancelled_tasks: set[Any] = set()
        self._last_backend: str | None = None
        self._last_bucket: Any | None = None

    def cancel_tasks(self, task_ids: list[Any]) -> None:
        """Record task cancellation for straggler control."""
        self._cancelled_tasks.update(task_ids)

    def run(
        self,
        manifest: "CampaignManifest",
        unit_fn: "Callable[[ManifestRow], Any]",
        *,
        output_dir: str,
        row_indices: list[int] | None = None,
        max_failures: int | None = None,
    ) -> CampaignSummary:
        start_time = time.time()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_indices = (
            list(row_indices) if row_indices is not None else list(range(len(manifest.rows)))
        )
        total_units = len(target_indices)
        completed_units = 0
        skipped_units = 0
        failed_units = 0
        locked_units = 0
        failures: list[dict[str, Any]] = []

        for idx in target_indices:
            row = manifest.rows[idx]

            if idx in self._cancelled_tasks or row.manifest_row_hash in self._cancelled_tasks:
                skipped_units += 1
                continue

            # Resume check: reuse a completed unit only if it still verifies.
            if _check_resume_done(out_dir, row, resume_verify=self.resume_verify):
                _record_reuse(
                    out_dir,
                    row,
                    _resolve_output_path(out_dir, row),
                    self.resume_verify,
                    reused_by="LocalSequentialExecutor",
                )
                skipped_units += 1
                continue

            # Arm backend & padding bucket transition check for jax.clear_caches()
            current_backend = row.arm_backend
            current_bucket = row.resource_requirements.padding_bucket_id

            if self._last_backend is not None and (
                current_backend != self._last_backend or current_bucket != self._last_bucket
            ):
                try:
                    import jax

                    jax.clear_caches()
                except (ImportError, Exception):
                    pass

            self._last_backend = current_backend
            self._last_bucket = current_bucket

            # Execute unit
            try:
                outcome = _execute_single_unit(
                    row=row,
                    unit_fn=unit_fn,
                    output_dir=out_dir,
                    lease_seconds=self.lease_seconds,
                    heartbeat_interval=self.heartbeat_interval,
                    owner_token_prefix=f"local_seq:{os.getpid()}",
                    resume_verify=self.resume_verify,
                )
            except Exception as exc:
                failed_units += 1
                failures.append(
                    {
                        "manifest_row_hash": row.manifest_row_hash,
                        "row_index": idx,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                if max_failures is not None and failed_units >= max_failures:
                    break
                continue
            if outcome == UNIT_COMMITTED:
                completed_units += 1
            elif outcome == UNIT_LOCK_HELD:
                locked_units += 1
            else:
                skipped_units += 1

        elapsed = time.time() - start_time
        return CampaignSummary(
            campaign_id=manifest.metadata.campaign_id,
            total_units=total_units,
            completed_units=completed_units,
            skipped_units=skipped_units,
            failed_units=failed_units,
            elapsed_seconds=elapsed,
            failures=failures,
            locked_units=locked_units,
        )


def _reclaim_dead_worker_leases(
    output_dir: Path, rows: "list[ManifestRow]", owner_prefix: str
) -> list[Path]:
    """Remove leases, and their partial outputs, left by this pool run's dead workers.

    Call only after the pool has shut down, when none of the run's workers is alive. A
    lock is removed only when its ``owner_token`` starts with ``owner_prefix``, which is
    unique to this run, so another run's or another host's live lease is never touched.
    The lock bytes are re-read just before unlinking, so a lock rewritten in between is
    left alone. Returns the lock paths removed.
    """
    import shutil

    reclaimed: list[Path] = []
    token_prefix = f"{owner_prefix}:"
    for row in rows:
        output_path = _resolve_output_path(output_dir, row)
        lock_path = output_path.with_name(f"{output_path.name}.lock")
        try:
            raw = lock_path.read_bytes()
            owner = json.loads(raw.decode("utf-8")).get("owner_token", "")
        except (OSError, ValueError, AttributeError):
            continue
        if not isinstance(owner, str) or not owner.startswith(token_prefix):
            continue

        def lock_unchanged(lock_path: Path = lock_path, raw: bytes = raw) -> bool:
            """True while the lock still has the bytes read above."""
            try:
                return lock_path.read_bytes() == raw
            except OSError:
                return False

        # Re-check the lock before each removal and before the unlink (#3312): if another
        # run steals the unit mid-way, its partials and lock are left alone.
        intact = True
        for partial in sorted(output_path.parent.glob(f"{output_path.name}.partial.*")):
            if not lock_unchanged():
                intact = False
                break
            if partial.is_dir():
                shutil.rmtree(partial, ignore_errors=True)
            else:
                partial.unlink(missing_ok=True)
        if not intact or not lock_unchanged():
            continue
        try:
            lock_path.unlink()
        except OSError:
            continue
        reclaimed.append(lock_path)
    return reclaimed


def _terminate_pool_workers(pool: ProcessPoolExecutor, grace_seconds: float = 3.0) -> None:
    """Stop every worker of ``pool``: SIGTERM, a short grace period, then SIGKILL.

    ``ProcessPoolExecutor.shutdown`` waits for running tasks; an early exit on
    ``max_failures`` must not. SIGTERM lets a unit in progress unwind (its handler
    releases the lease and discards the partial), but that worker then survives and
    waits for more work. Once the pool is broken, its internal cleanup joins every
    worker without a timeout, so a survivor hangs ``shutdown`` forever. Any worker still
    alive after the grace period is therefore killed; a lease it left behind is removed
    by ``_reclaim_dead_worker_leases`` after shutdown.

    Python 3.14's ``terminate_workers`` sends only SIGTERM, so the worker table
    (``_processes``) is used whenever it exists.
    """
    table = getattr(pool, "_processes", None)
    if table is None:
        terminate = getattr(pool, "terminate_workers", None)
        if callable(terminate):
            terminate()
        return
    processes = list(table.values())
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    deadline = time.time() + grace_seconds
    for proc in processes:
        proc.join(max(0.0, deadline - time.time()))
    for proc in processes:
        if proc.is_alive():
            proc.kill()
            proc.join()


class LocalProcessPoolExecutor:
    """Multiprocessing pool executor with chunked worker recycling and persistent JAX cache.

    Per Section 6.3.2 of SPEC-261105-CAMPAIGN-RUNNER:
    - Spawns isolated worker processes using multiprocessing.get_context("spawn").
    - Configurable chunked recycling (recycle_interval=50) to prevent allocator fragmentation.
    - Mandatory JAX persistent disk compilation cache initialization.
    """

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        recycle_interval: int | None = 50,
        cache_dir: str | None = None,
        lease_seconds: int = 1800,
        heartbeat_interval: float = 60.0,
        resume_verify: ResumeVerify = "deep",
    ) -> None:
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
        self.recycle_interval = recycle_interval
        self.cache_dir = cache_dir
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self.resume_verify = _validate_resume_verify(resume_verify)
        self._cancelled_tasks: set[Any] = set()

    def cancel_tasks(self, task_ids: list[Any]) -> None:
        """Record task cancellation for straggler control."""
        self._cancelled_tasks.update(task_ids)

    def run(
        self,
        manifest: "CampaignManifest",
        unit_fn: "Callable[[ManifestRow], Any]",
        *,
        output_dir: str,
        row_indices: list[int] | None = None,
        max_failures: int | None = None,
    ) -> CampaignSummary:
        start_time = time.time()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        target_indices = (
            list(row_indices) if row_indices is not None else list(range(len(manifest.rows)))
        )
        total_units = len(target_indices)
        completed_units = 0
        skipped_units = 0
        failed_units = 0
        locked_units = 0
        failures: list[dict[str, Any]] = []

        pending_rows: list[tuple[int, ManifestRow]] = []
        for idx in target_indices:
            row = manifest.rows[idx]
            if idx in self._cancelled_tasks or row.manifest_row_hash in self._cancelled_tasks:
                skipped_units += 1
                continue
            if _check_resume_done(out_dir, row, resume_verify=self.resume_verify):
                _record_reuse(
                    out_dir,
                    row,
                    _resolve_output_path(out_dir, row),
                    self.resume_verify,
                    reused_by="LocalProcessPoolExecutor",
                )
                skipped_units += 1
                continue
            pending_rows.append((idx, row))

        if not pending_rows:
            return CampaignSummary(
                campaign_id=manifest.metadata.campaign_id,
                total_units=total_units,
                completed_units=0,
                skipped_units=skipped_units,
                failed_units=0,
                elapsed_seconds=time.time() - start_time,
                failures=[],
            )

        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=ctx,
            initializer=_initialize_worker_process,
            initargs=(self.cache_dir,),
            max_tasks_per_child=self.recycle_interval,
        )
        unfinished: dict[int, ManifestRow] = dict(pending_rows)
        future_to_row: dict[Future[dict[str, Any]], tuple[int, ManifestRow]] = {}
        # Unique to this run, so after shutdown the parent can tell its own workers'
        # leftover leases from anyone else's.
        run_owner_prefix = f"pool:{os.getpid()}.{uuid.uuid4().hex[:8]}"
        broken_error: str | None = None
        try:
            try:
                for idx, row in pending_rows:
                    fut = pool.submit(
                        _pool_worker_task,
                        row,
                        unit_fn,
                        str(out_dir),
                        self.lease_seconds,
                        self.heartbeat_interval,
                        self.cache_dir,
                        self.resume_verify,
                        run_owner_prefix,
                    )
                    future_to_row[fut] = (idx, row)

                for fut in as_completed(future_to_row):
                    idx, row = future_to_row[fut]
                    res = fut.result()
                    unfinished.pop(idx, None)
                    if res.get("success"):
                        outcome = res.get("outcome")
                        if outcome == UNIT_COMMITTED:
                            completed_units += 1
                        elif outcome == UNIT_LOCK_HELD:
                            locked_units += 1
                        else:
                            skipped_units += 1
                    else:
                        failed_units += 1
                        failures.append(
                            {
                                "manifest_row_hash": row.manifest_row_hash,
                                "row_index": idx,
                                "error": res.get("error", "Unknown worker error"),
                            }
                        )
                        if max_failures is not None and failed_units >= max_failures:
                            for pending_fut in future_to_row:
                                pending_fut.cancel()
                            # Stop in-flight units now rather than waiting for them; each
                            # worker's SIGTERM handler releases its lease and partial store.
                            _terminate_pool_workers(pool)
                            break
            except BrokenProcessPool as exc:
                broken_error = f"{type(exc).__name__}: {exc}"
                for pending_fut in future_to_row:
                    pending_fut.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            # Every worker has exited. A worker killed outright (SIGKILL, the OOM killer)
            # never ran its cleanup, so its lease would block the unit for lease + 300 s.
            _reclaim_dead_worker_leases(out_dir, [row for _, row in pending_rows], run_owner_prefix)

        # Rows whose result never reached the parent (the pool broke, or a max_failures
        # exit stopped reading results). A unit may still have committed before its
        # worker died or was stopped, so count from what is on disk: a valid store is
        # completed; otherwise a broken pool is that unit's failure, and after an early
        # exit the unit simply did not run.
        for rem_idx, rem_row in unfinished.items():
            if validate_unit_done(
                _resolve_output_path(out_dir, rem_row), rem_row, deep_verify=False
            ).is_valid:
                completed_units += 1
            elif broken_error is not None:
                failed_units += 1
                failures.append(
                    {
                        "manifest_row_hash": rem_row.manifest_row_hash,
                        "row_index": rem_idx,
                        "error": broken_error,
                    }
                )

        elapsed = time.time() - start_time
        return CampaignSummary(
            campaign_id=manifest.metadata.campaign_id,
            total_units=total_units,
            completed_units=completed_units,
            skipped_units=skipped_units,
            failed_units=failed_units,
            elapsed_seconds=elapsed,
            failures=failures,
            locked_units=locked_units,
        )
