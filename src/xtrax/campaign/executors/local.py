"""Local execution harnesses: sequential and process pool executors.

Covering [T3-02], [T3-03], GATE-07, and Sections 6.3.1, 6.3.2 of SPEC-261105-CAMPAIGN-RUNNER.
Zero imports of controller/ or bathos.
"""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xtrax.campaign.executors.protocol import CampaignSummary
from xtrax.campaign.lease import FileLock, LockUnavailableError, StolenAttemptDiscarded
from xtrax.campaign.marker import (
    compute_content_digest,
    create_done_marker,
    quarantine_output,
    validate_unit_done,
)

#: ``_execute_single_unit`` outcomes.
UNIT_COMMITTED = "committed"
UNIT_LOCK_HELD = "lock_held"
UNIT_DISCARDED = "discarded"


class UnitProducedNoOutputError(RuntimeError):
    """Raised when a unit function returns without writing its output store."""


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


def _resolve_output_path(output_dir: Path, row: ManifestRow) -> Path:
    """Resolve the target output path for a manifest row."""
    if row.output_locator:
        loc_path = Path(row.output_locator)
        if loc_path.is_absolute():
            return loc_path
        if len(loc_path.parts) > 1 and output_dir.name == loc_path.parts[0]:
            return output_dir / Path(*loc_path.parts[1:])
        return output_dir / loc_path
    return output_dir / f"{row.manifest_row_hash}.zarr"


def _check_tier1_resume(output_dir: Path, row: ManifestRow) -> bool:
    """Check Tier-1 fast resume status (< 5 ms per unit).

    A unit is done only when ``validate_unit_done`` accepts its resolved output path:
    marker present and matching, and the store present with valid root metadata. A bare
    marker file is never enough.
    """
    output_path = _resolve_output_path(output_dir, row)
    val = validate_unit_done(output_path, row, deep_verify=False, quarantine_on_failure=False)
    return val.is_valid


def _set_aside_orphaned_output(output_path: Path, row: ManifestRow) -> None:
    """Quarantine a store or done marker that does not form a valid completed unit.

    Called while holding the unit's lease, before the unit function runs. A store
    without a valid marker is left over from an attempt that crashed before stamping
    it; a marker without a valid store (for example, the store was deleted to force a
    rerun) would make the commit protocol discard this attempt as already done. After
    this call neither exists, so whatever appears at ``output_path`` is this attempt's.
    """
    marker_path = output_path.with_name(f"{output_path.name}.done.json")
    if not output_path.exists() and not marker_path.exists():
        return
    val = validate_unit_done(output_path, row, deep_verify=False, quarantine_on_failure=False)
    if not val.is_valid:
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


def _stage_unit_output(state: Any, output_path: Path, partial_path: Path) -> None:
    """Put this attempt's output at ``partial_path`` for the commit protocol.

    In order of precedence:

    1. The unit function wrote its own store at ``output_path``: move it to the partial.
    2. The unit function returned a ``Mapping``: write it as a zarr v3 group.
    3. Otherwise the unit fails. Stamping a done marker on missing output would record
       the unit as complete and resume would skip it forever.
    """
    if partial_path.exists():
        return
    if output_path.exists():
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


def _discard_partial(partial_path: Path) -> None:
    """Remove a partially written attempt directory or file."""
    import shutil

    if partial_path.is_dir():
        shutil.rmtree(partial_path, ignore_errors=True)
    elif partial_path.exists():
        partial_path.unlink(missing_ok=True)


def _execute_single_unit(
    row: ManifestRow,
    unit_fn: Callable[[ManifestRow], Any],
    output_dir: Path,
    lease_seconds: int = 1800,
    heartbeat_interval: float = 60.0,
    owner_token_prefix: str = "local",
) -> str:
    """Execute a single unit with atomic leasing, output promotion, and done marker stamping.

    Returns one of ``UNIT_COMMITTED`` (this attempt stamped the done marker),
    ``UNIT_LOCK_HELD`` (another worker holds the lease; nothing ran) or
    ``UNIT_DISCARDED`` (another worker finished the unit first). Raises on failure.
    """
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

    try:
        # Another worker may have committed between the resume check and our acquire.
        if _check_tier1_resume(output_dir, row):
            return UNIT_DISCARDED
        _set_aside_orphaned_output(output_path, row)

        start_time = time.time()
        attempt_uuid = uuid.uuid4().hex[:8]
        partial_path = output_path.with_name(f"{output_path.name}.partial.{attempt_uuid}")

        state = unit_fn(row)
        duration_s = max(0.001, time.time() - start_time)

        try:
            _stage_unit_output(state, output_path, partial_path)
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
        if lock.is_locked:
            lock.release()


def _pool_worker_task(
    row: ManifestRow,
    unit_fn: Callable[[ManifestRow], Any],
    output_dir: str,
    lease_seconds: int,
    heartbeat_interval: float,
    cache_dir: str | None,
) -> dict[str, Any]:
    """Module-level task function executed in spawned worker processes."""
    _initialize_worker_process(cache_dir)
    out_dir = Path(output_dir)
    try:
        outcome = _execute_single_unit(
            row=row,
            unit_fn=unit_fn,
            output_dir=out_dir,
            lease_seconds=lease_seconds,
            heartbeat_interval=heartbeat_interval,
            owner_token_prefix=f"pool:{os.getpid()}",
        )
        return {
            "manifest_row_hash": row.manifest_row_hash,
            "success": True,
            "outcome": outcome,
            "error": None,
        }
    except Exception as exc:
        return {
            "manifest_row_hash": row.manifest_row_hash,
            "success": False,
            "outcome": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _pool_worker_task_wrapper(
    args: tuple[
        int,
        ManifestRow,
        Callable[[ManifestRow], Any],
        str,
        int,
        float,
        str | None,
    ],
) -> tuple[int, ManifestRow, dict[str, Any]]:
    """Unpack task arguments for pool.imap_unordered and return indexed result."""
    idx, row, unit_fn, output_dir, lease_seconds, heartbeat_interval, cache_dir = args
    res = _pool_worker_task(
        row=row,
        unit_fn=unit_fn,
        output_dir=output_dir,
        lease_seconds=lease_seconds,
        heartbeat_interval=heartbeat_interval,
        cache_dir=cache_dir,
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
    ) -> None:
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self._cancelled_tasks: set[Any] = set()
        self._last_backend: str | None = None
        self._last_bucket: Any | None = None

    def cancel_tasks(self, task_ids: list[Any]) -> None:
        """Record task cancellation for straggler control."""
        self._cancelled_tasks.update(task_ids)

    def run(
        self,
        manifest: CampaignManifest,
        unit_fn: Callable[[ManifestRow], Any],
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

            # Tier-1 Fast Resume Check (< 5 ms)
            if _check_tier1_resume(out_dir, row):
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
    ) -> None:
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1) - 1)
        self.recycle_interval = recycle_interval
        self.cache_dir = cache_dir
        self.lease_seconds = lease_seconds
        self.heartbeat_interval = heartbeat_interval
        self._cancelled_tasks: set[Any] = set()

    def cancel_tasks(self, task_ids: list[Any]) -> None:
        """Record task cancellation for straggler control."""
        self._cancelled_tasks.update(task_ids)

    def run(
        self,
        manifest: CampaignManifest,
        unit_fn: Callable[[ManifestRow], Any],
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
            if _check_tier1_resume(out_dir, row):
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
        pool_kwargs: dict[str, Any] = {
            "processes": self.max_workers,
            "initializer": _initialize_worker_process,
            "initargs": (self.cache_dir,),
        }
        if self.recycle_interval is not None:
            pool_kwargs["maxtasksperchild"] = self.recycle_interval

        with ctx.Pool(**pool_kwargs) as pool:
            task_args = [
                (
                    idx,
                    row,
                    unit_fn,
                    str(out_dir),
                    self.lease_seconds,
                    self.heartbeat_interval,
                    self.cache_dir,
                )
                for idx, row in pending_rows
            ]

            try:
                for idx, row, res in pool.imap_unordered(_pool_worker_task_wrapper, task_args):
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
                            pool.terminate()
                            pool.join()
                            break
            except Exception as exc:
                pool.terminate()
                pool.join()
                raise exc

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
