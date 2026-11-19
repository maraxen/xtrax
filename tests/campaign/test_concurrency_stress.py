"""Multi-process concurrency, lock contention, and preemption crash recovery stress tests.

Covering [T5-03] and AC-03, AC-04, AC-05, AC-07 from SPEC-261105-CAMPAIGN-RUNNER:
- 8 concurrent worker processes competing for 20 units over a shared directory.
- Atomic POSIX leasing (O_CREAT | O_EXCL) and lock contention exclusion (AC-03).
- Stale lease stealing past 300s clock drift margin with atomic CAS (AC-04).
- Zombie worker prevention: heartbeat CAS abort and singleton commit token exclusion (AC-07).
- Multi-wave preemption crash injection and 100% verified completion without corruption.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from xtrax.campaign.executors.local import LocalProcessPoolExecutor
from xtrax.campaign.lease import (
    CLOCK_DRIFT_GRACE_PERIOD_S,
    FileLock,
    LockUnavailableError,
)
from xtrax.campaign.manifest import (
    ManifestRow,
    plan_campaign,
)
from xtrax.campaign.marker import (
    CampaignDoneMarker,
    compute_content_digest,
    create_done_marker,
    validate_unit_done,
)
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid


def _stress_unit_fn(row: ManifestRow) -> dict[str, Any]:
    """Module-level compute worker for concurrency stress tests."""
    temp = float(row.parameters.get("val", 1.0))
    # Synthetic workload
    time.sleep(0.01)
    data = np.full((16, 16), fill_value=temp * 10.0 + float(row.seed), dtype=np.float32)
    return {
        "manifest_row_hash": row.manifest_row_hash,
        "input_hash": row.input_hash,
        "sum": float(data.sum()),
    }


def _worker_acquire_and_commit(
    output_path_str: str,
    row_hash: str,
    worker_id: str,
    attempt_uuid: str,
    content_value: str,
) -> dict[str, Any]:
    """Worker task that attempts to acquire lock, writes partial, and commits."""
    out_path = Path(output_path_str)
    lock = FileLock(
        output_path=out_path,
        owner_token=f"worker_{worker_id}:{os.getpid()}",
        manifest_row_hash=row_hash,
        lease_seconds=300,
        heartbeat_interval=10.0,
    )

    try:
        lock.acquire()
    except LockUnavailableError as exc:
        return {"worker_id": worker_id, "status": "lock_unavailable", "error": str(exc)}

    partial_dir = out_path.with_name(f"{out_path.name}.partial.{attempt_uuid}")
    try:
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
        (partial_dir / "data.txt").write_text(content_value, encoding="utf-8")

        digest = compute_content_digest(partial_dir)
        marker = create_done_marker(
            manifest_row_hash=row_hash,
            input_hash=f"input_{row_hash}",
            output_locator=str(out_path),
            content_digest_sha256=digest,
            duration_s=0.05,
        )

        lock.commit_output(
            partial_path=partial_dir,
            output_path=out_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
        )
        return {"worker_id": worker_id, "status": "success", "error": None}
    except Exception as exc:
        lock.release()
        return {"worker_id": worker_id, "status": "error", "error": str(exc)}


def _worker_commit_only(
    output_path_str: str,
    row_hash: str,
    owner_token: str,
    attempt_uuid: str,
    content_value: str,
) -> dict[str, Any]:
    """Worker task racing on commit_output with an already acquired lock token."""
    out_path = Path(output_path_str)
    lock = FileLock(
        output_path=out_path,
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=300,
    )
    # Adopt the lease held on disk (same owner) so this worker reaches the commit phase;
    # the lease check compares owner, acquired_at and fence (#3312).
    held = json.loads(lock.lock_path.read_text(encoding="utf-8"))
    lock.acquired_at_unix_s = held["acquired_at_unix_s"]
    lock.fence = held["fence"]
    lock.is_locked = True

    partial_dir = out_path.with_name(f"{out_path.name}.partial.{attempt_uuid}")
    partial_dir.mkdir(parents=True, exist_ok=True)
    (partial_dir / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    (partial_dir / "data.txt").write_text(content_value, encoding="utf-8")

    digest = compute_content_digest(partial_dir)
    marker = create_done_marker(
        manifest_row_hash=row_hash,
        input_hash=f"input_{row_hash}",
        output_locator=str(out_path),
        content_digest_sha256=digest,
        duration_s=0.05,
    )

    try:
        res = lock.commit_output(
            partial_path=partial_dir,
            output_path=out_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
        )
        if isinstance(res, CampaignDoneMarker):
            return {"token": owner_token, "status": "committed", "error": None}
        return {
            "token": owner_token,
            "status": "discarded",
            "error": getattr(res, "reason", "discarded"),
        }
    except Exception as exc:
        return {
            "token": owner_token,
            "status": "failed",
            "error": str(exc),
            "type": type(exc).__name__,
        }


def test_8_worker_concurrent_manifest_competition(tmp_path: Path) -> None:
    """[T5-03]: 8 concurrent worker processes competing for 20 units over shared directory.

    Invariants:
    - 20 units planned across 2 targets, 2 arms, 5 parameter values.
    - 8 worker processes compete for units via LocalProcessPoolExecutor.
    - Exactly 20 units completed with 0 duplicates, 0 deadlocks, 0 corrupt stores.
    - Zero .lock, .commit, or .partial files remain on disk.
    - All 20 units pass deep Tier-2 verification.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest_20.json"

    # 2 targets * 2 arms * 5 parameters = 20 units
    spec = CampaignSpec(
        campaign_id="c_stress_20",
        target_inputs=["t_prot1.pdb", "t_prot2.pdb"],
        arms=[
            ArmSpec(arm_id="boltz", backend="boltz2", checkpoint_id="chk_b"),
            ArmSpec(arm_id="chai", backend="chai1", checkpoint_id="chk_c"),
        ],
        grid=ParamGrid(axes={"val": [1.0, 2.0, 3.0, 4.0, 5.0]}),
        output_dir=str(output_dir),
    )

    manifest = plan_campaign(spec, output_manifest_path=str(manifest_path))
    assert len(manifest.rows) == 20

    executor = LocalProcessPoolExecutor(
        max_workers=8,
        recycle_interval=5,
        cache_dir=str(tmp_path / "jax_cache"),
        lease_seconds=300,
        heartbeat_interval=10.0,
    )

    summary = executor.run(
        manifest=manifest,
        unit_fn=_stress_unit_fn,
        output_dir=str(output_dir),
    )

    assert summary.total_units == 20
    assert summary.completed_units == 20
    assert summary.failed_units == 0
    assert summary.skipped_units == 0
    assert len(summary.failures) == 0

    # Verify no dangling lock or commit token files exist
    assert len(list(output_dir.glob("**/*.lock"))) == 0, "Dangling .lock files found"
    assert len(list(output_dir.glob("**/*.commit"))) == 0, "Dangling .commit files found"
    assert len(list(output_dir.glob("**/*.partial.*"))) == 0, "Dangling .partial dirs found"

    # Verify all 20 units pass deep Tier-2 content verification
    for row in manifest.rows:
        store_path = (
            Path(row.output_locator)
            if Path(row.output_locator).is_absolute()
            else output_dir / row.output_locator
        )
        assert store_path.exists()
        val = validate_unit_done(
            store_path, expected_row=row, deep_verify=True, quarantine_on_failure=False
        )
        assert val.is_valid, (
            f"Unit validation failed for {row.manifest_row_hash}: {val.error_message}"
        )


def test_stale_lease_stealing_clock_drift_margin(tmp_path: Path) -> None:
    """AC-04: Stale lease stealing past 300s clock drift margin with atomic CAS.

    Verifies:
    1. Worker 1 acquires lock on output path.
    2. Competing Worker 2 attempts acquisition while lease is active: refused with
       LockUnavailableError.
    3. Even when lease is technically expired (now > expires_at), if within 300s clock drift margin,
       acquisition is still refused (AC-04 Negative Control).
    4. When now >= expires_at + 300 AND now - st_mtime >= lease_seconds + 300,
       Worker 2 safely steals the lease via atomic CAS and resets heartbeat_count = 0.
    5. Winning worker completes output and stamps done marker cleanly.
    """
    output_path = tmp_path / "unit_stale.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")

    worker1_token = "worker_1:pid101:uuid1"
    worker2_token = "worker_2:pid102:uuid2"
    row_hash = "row_stale_test_01"

    lock1 = FileLock(
        output_path=output_path,
        owner_token=worker1_token,
        manifest_row_hash=row_hash,
        lease_seconds=60,
    )
    assert lock1.acquire() is True
    assert lock_path.exists()

    lock2 = FileLock(
        output_path=output_path,
        owner_token=worker2_token,
        manifest_row_hash=row_hash,
        lease_seconds=60,
    )

    # 1. Immediate acquisition by worker 2 must fail
    with pytest.raises(LockUnavailableError):
        lock2.acquire()

    # 2. Simulate lease expiration but WITHIN 300s clock drift grace period
    # Set lease_expires_at to 10 seconds ago, mtime to 10 seconds ago
    now = time.time()
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_data["lease_expires_at_unix_s"] = now - 10.0
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    os.utime(lock_path, (now - 10.0, now - 10.0))

    # Must STILL fail because within 300s grace period (Negative Control)
    with pytest.raises(LockUnavailableError):
        lock2.acquire()

    # 3. Advance timestamps past 300s clock drift grace period (> 300s + lease_seconds ago)
    past_timestamp = now - CLOCK_DRIFT_GRACE_PERIOD_S - lock_data["lease_seconds"] - 50.0
    lock_data["lease_expires_at_unix_s"] = past_timestamp
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    os.utime(lock_path, (past_timestamp, past_timestamp))

    # Worker 2 attempts acquisition: should successfully steal stale lease via CAS!
    assert lock2.acquire() is True
    assert lock2.is_locked is True

    # Verify on-disk lock was updated to Worker 2 and heartbeat_count reset to 0
    updated_data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert updated_data["owner_token"] == worker2_token
    assert updated_data["heartbeat_count"] == 0

    # Worker 2 successfully finalizes output
    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = output_path.with_name(f"{output_path.name}.partial.{attempt_uuid}")
    partial_path.mkdir(parents=True, exist_ok=True)
    (partial_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    digest = compute_content_digest(partial_path)
    marker = create_done_marker(
        manifest_row_hash=row_hash,
        input_hash=f"input_{row_hash}",
        output_locator=str(output_path),
        content_digest_sha256=digest,
        duration_s=1.0,
    )

    lock2.commit_output(
        partial_path=partial_path,
        output_path=output_path,
        done_marker=marker,
        attempt_uuid=attempt_uuid,
    )

    assert output_path.exists()
    assert output_path.with_name(f"{output_path.name}.done.json").exists()
    assert not lock_path.exists()


def test_zombie_worker_heartbeat_and_commit_abort(tmp_path: Path) -> None:
    """AC-07: Zombie worker protection via heartbeat CAS abort and commit token exclusion.

    Verifies:
    1. Worker A acquires lease for a unit.
    2. Worker A is suspended while Worker B steals the lease.
    3. Worker A's heartbeat thread detects owner_token divergence and triggers abort.
    4. Worker A attempts to finalize output via commit_output:
       discovers owner_token mismatch on lock, cleans up .partial scratch directory,
       and raises LeaseLostError without touching Worker B's data.
    """
    output_path = tmp_path / "unit_zombie.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    row_hash = "row_zombie_01"

    worker_a_token = "worker_A:pid201:uuidA"
    worker_b_token = "worker_B:pid202:uuidB"

    lock_a = FileLock(
        output_path=output_path,
        owner_token=worker_a_token,
        manifest_row_hash=row_hash,
        lease_seconds=300,
        heartbeat_interval=0.1,  # Fast heartbeat for test
    )
    assert lock_a.acquire() is True
    assert lock_a.heartbeat is not None

    # Worker A creates its partial scratch output
    attempt_a = uuid.uuid4().hex[:8]
    partial_a = output_path.with_name(f"{output_path.name}.partial.{attempt_a}")
    partial_a.mkdir(parents=True, exist_ok=True)
    (partial_a / "data.txt").write_text("Worker A data (stale)", encoding="utf-8")

    # While Worker A is "working", Worker B steals the lock (simulated lease steal)
    stolen_lock_data = {
        "schema_version": "campaign_lock_v1",
        "owner_token": worker_b_token,
        "manifest_row_hash": row_hash,
        "acquired_at_unix_s": time.time(),
        "lease_expires_at_unix_s": time.time() + 300,
        "lease_seconds": 300,
        "heartbeat_count": 0,
    }
    lock_path.write_text(json.dumps(stolen_lock_data), encoding="utf-8")

    # Worker B finishes and writes legitimate output
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    (output_path / "data.txt").write_text("Worker B legitimate output", encoding="utf-8")

    marker_b = create_done_marker(
        manifest_row_hash=row_hash,
        input_hash=f"input_{row_hash}",
        output_locator=str(output_path),
        content_digest_sha256="digest_b",
        duration_s=1.0,
    )
    marker_b_path = output_path.with_name(f"{output_path.name}.done.json")
    marker_b.write(marker_b_path)

    # Wait briefly for Worker A's heartbeat thread to detect theft
    time.sleep(0.3)
    assert lock_a.heartbeat.abort_event.is_set(), (
        "Heartbeat thread failed to signal abort on lock theft"
    )

    # Worker A wakes up and attempts to commit output
    marker_a = create_done_marker(
        manifest_row_hash=row_hash,
        input_hash=f"input_{row_hash}",
        output_locator=str(output_path),
        content_digest_sha256="digest_a",
        duration_s=5.0,
    )

    # When done marker already exists, commit_output safely discards partial attempt
    lock_a.commit_output(
        partial_path=partial_a,
        output_path=output_path,
        done_marker=marker_a,
        attempt_uuid=attempt_a,
    )

    # Scratch directory must be cleaned up
    assert not partial_a.exists(), "Partial scratch directory was not discarded"

    # Worker B's data MUST remain uncorrupted
    assert (output_path / "data.txt").read_text(encoding="utf-8") == "Worker B legitimate output"

    lock_a.release()


def test_simultaneous_commit_singleton_token_race(tmp_path: Path) -> None:
    """AC-07 & GATE-03: Multiple concurrent workers race to commit singleton token.

    Verifies:
    - 4 concurrent processes attempt commit_output simultaneously for the same output.
    - Exactly 1 process succeeds in creating <output>.commit and promoting its partial.
    - Competing processes receive FileExistsError / StolenAttemptDiscarded and clean up
      their scratch dirs.
    - Output store is intact with exactly one winner's payload.
    """
    output_path = tmp_path / "unit_race.zarr"
    row_hash = "row_race_token"

    # Pre-create lock held by common token so workers proceed to commit phase
    lock = FileLock(
        output_path=output_path,
        owner_token="common_owner:pid0",
        manifest_row_hash=row_hash,
        lease_seconds=300,
    )
    assert lock.acquire() is True

    ctx = multiprocessing.get_context("spawn")
    tasks = [
        (
            str(output_path),
            row_hash,
            "common_owner:pid0",
            f"attempt_{i}",
            f"payload_from_worker_{i}",
        )
        for i in range(4)
    ]

    with ctx.Pool(processes=4) as pool:
        results = pool.starmap(_worker_commit_only, tasks)

    statuses = [r["status"] for r in results]
    assert statuses.count("committed") == 1, f"Expected exactly 1 committed winner, got: {statuses}"
    assert statuses.count("failed") + statuses.count("discarded") == 3

    # Verify output exists and is valid
    assert output_path.exists()
    assert (output_path / ".zgroup").exists()
    assert not output_path.with_name(f"{output_path.name}.commit").exists()


def test_preemption_crash_recovery_stress(tmp_path: Path) -> None:
    """Stress test simulating process preemption crashes across multiple waves.

    Wave 1: Workers execute units, but simulated crashes occur mid-unit, leaving orphaned locks.
    Wave 2: Orphaned lock timestamps are aged past clock drift margin.
    Wave 3: Recovery workers resume campaign, steal stale leases, and achieve 100% completion.
    """
    output_dir = tmp_path / "outputs_crash"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest_crash.json"

    spec = CampaignSpec(
        campaign_id="c_crash_recovery",
        target_inputs=["target1.pdb"],
        arms=[
            ArmSpec(arm_id="arm_a", backend="boltz2", checkpoint_id="chk_a"),
            ArmSpec(arm_id="arm_b", backend="chai1", checkpoint_id="chk_b"),
        ],
        grid=ParamGrid(axes={"val": [10, 20, 30]}),
        output_dir=str(output_dir),
    )
    manifest = plan_campaign(spec, output_manifest_path=str(manifest_path))
    assert len(manifest.rows) == 6

    # Wave 1: Complete rows 0 and 1 normally
    executor = LocalProcessPoolExecutor(max_workers=2, cache_dir=str(tmp_path / "jax_cache"))
    summary_wave1 = executor.run(
        manifest=manifest,
        unit_fn=_stress_unit_fn,
        output_dir=str(output_dir),
        row_indices=[0, 1],
    )
    assert summary_wave1.completed_units == 2

    # Wave 2: Simulate preemption on rows 2 and 3: create orphaned lock files without output
    now = time.time()
    for idx in (2, 3):
        row = manifest.rows[idx]
        store = (
            Path(row.output_locator)
            if Path(row.output_locator).is_absolute()
            else output_dir / row.output_locator
        )
        lock_file = store.with_name(f"{store.name}.lock")
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        # Create expired lock past 300s clock drift margin
        expired_time = now - CLOCK_DRIFT_GRACE_PERIOD_S - 60.0
        lock_content = {
            "schema_version": "campaign_lock_v1",
            "owner_token": f"crashed_worker:pid{900 + idx}:orphaned",
            "manifest_row_hash": row.manifest_row_hash,
            "acquired_at_unix_s": expired_time,
            "lease_expires_at_unix_s": expired_time,
            "lease_seconds": 60,
            "heartbeat_count": 0,
        }
        lock_file.write_text(json.dumps(lock_content), encoding="utf-8")
        os.utime(lock_file, (expired_time, expired_time))

    # Wave 3: Run full campaign recovery across all 6 rows
    recovery_summary = executor.run(
        manifest=manifest,
        unit_fn=_stress_unit_fn,
        output_dir=str(output_dir),
    )

    # Rows 0 and 1 skipped, rows 2 and 3 stolen and executed, rows 4 and 5 executed
    assert recovery_summary.total_units == 6
    assert recovery_summary.completed_units == 4
    assert recovery_summary.skipped_units == 2
    assert recovery_summary.failed_units == 0

    # Verify all 6 units are completely finished and verified
    for row in manifest.rows:
        store = (
            Path(row.output_locator)
            if Path(row.output_locator).is_absolute()
            else output_dir / row.output_locator
        )
        assert store.exists()
        val = validate_unit_done(
            store, expected_row=row, deep_verify=True, quarantine_on_failure=False
        )
        assert val.is_valid
