"""Tests for atomic file leasing, stale lease stealing, and fsync durabilization.

Covering [T2-01] and AC-03, AC-04 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import errno
import json
import os
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.campaign import lease as lease_module
from xtrax.campaign.lease import (
    CLOCK_DRIFT_GRACE_PERIOD_S,
    CampaignLock,
    FileLock,
    LockUnavailableError,
    fsync_directory,
)
from xtrax.campaign.marker import CampaignDoneMarker, ProducerInfo


def _pause_first_heartbeat_fsync(paused: threading.Event, resume: threading.Event):
    """Wrap fsync_file so the first renewal parks between its CAS read and os.replace."""
    real_fsync_file = lease_module.fsync_file
    armed = [True]

    def _wrapper(path: Path) -> None:
        if armed[0] and threading.current_thread().name.startswith("Heartbeat"):
            armed[0] = False
            paused.set()
            resume.wait(10.0)
        real_fsync_file(path)

    return _wrapper


def _run_in_thread(fn):
    """Run fn in a daemon thread; return (thread, outcome dict with 'value' or 'error')."""
    outcome: dict[str, object] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to the test body
            outcome["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, outcome


def test_commit_output_lock_not_recreated_by_inflight_renewal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A renewal paused mid-CAS must not recreate the lock after commit_output unlinks it."""
    output_path = tmp_path / "unit_commit_race.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    paused = threading.Event()
    resume = threading.Event()
    monkeypatch.setattr(lease_module, "fsync_file", _pause_first_heartbeat_fsync(paused, resume))

    lock = FileLock(
        output_path=output_path,
        owner_token="worker_commit_race:pid401",
        manifest_row_hash="row_commit_race_001",
        lease_seconds=600,
        heartbeat_interval=0.02,
    )
    assert lock.acquire() is True
    assert paused.wait(5.0), "renewal never reached the fsync point"

    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = tmp_path / f"unit_commit_race.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()
    (partial_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    marker = CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash="row_commit_race_001",
        input_hash="input_commit_race_001",
        producer=ProducerInfo(version="xtrax@0.5.0", git_sha="clean", hostname="localhost"),
        output_locator=str(output_path),
        content_digest_sha256="digest_sha256_dummy",
        completed_at_utc="2026-11-05T19:00:00Z",
        completed_at_unix_s=time.time(),
        duration_s=1.0,
    )

    committer, outcome = _run_in_thread(
        lambda: lock.commit_output(
            partial_path=partial_path,
            output_path=output_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
        )
    )
    time.sleep(0.2)
    resume.set()
    committer.join(timeout=10.0)

    assert not committer.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert output_path.exists()
    assert not lock_path.exists(), "orphan lock recreated by renewal after commit"


def test_release_lock_not_recreated_by_inflight_renewal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A renewal paused mid-CAS must not recreate the lock after release() unlinks it."""
    output_path = tmp_path / "unit_release_race.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    paused = threading.Event()
    resume = threading.Event()
    monkeypatch.setattr(lease_module, "fsync_file", _pause_first_heartbeat_fsync(paused, resume))

    lock = FileLock(
        output_path=output_path,
        owner_token="worker_release_race:pid402",
        manifest_row_hash="row_release_race_001",
        lease_seconds=600,
        heartbeat_interval=0.02,
    )
    assert lock.acquire() is True
    assert paused.wait(5.0), "renewal never reached the fsync point"
    heartbeat = lock.heartbeat
    assert heartbeat is not None

    releaser, outcome = _run_in_thread(lock.release)
    # Hold the renewal past release()'s 1.0 s join timeout, so the unlink has a chance to
    # run before the renewal resumes.
    releaser.join(timeout=1.5)
    resume.set()
    releaser.join(timeout=10.0)
    heartbeat.join(timeout=10.0)

    assert not releaser.is_alive()
    assert not heartbeat.is_alive()
    assert "error" not in outcome, outcome.get("error")
    assert not lock_path.exists(), "orphan lock recreated by renewal after release"


def test_campaign_lock_schema_and_immutability():
    """Verify CampaignLock dataclass schema version, fields, and immutability."""
    lock = CampaignLock(
        schema_version="campaign_lock_v1",
        owner_token="node1.cluster:pid100:abc12345",
        manifest_row_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        acquired_at_unix_s=1730830000.0,
        lease_expires_at_unix_s=1730831800.0,
        lease_seconds=1800,
        heartbeat_count=0,
    )
    assert lock.schema_version == "campaign_lock_v1"
    assert lock.owner_token == "node1.cluster:pid100:abc12345"
    assert lock.acquired_at_unix_s == 1730830000.0
    assert lock.lease_seconds == 1800
    assert lock.heartbeat_count == 0

    # Frozen dataclass check
    with pytest.raises((AttributeError, TypeError)):
        lock.heartbeat_count = 1  # type: ignore

    # Serialization roundtrip
    d = lock.to_dict()
    assert d["schema_version"] == "campaign_lock_v1"
    restored = CampaignLock.from_dict(d)
    assert restored == lock


def test_fsync_directory_normal_and_nfs_suppression(tmp_path: Path):
    """Verify fsync_directory flushes dir entries and suppresses NFS-specific errors."""
    d = tmp_path / "test_dir"
    d.mkdir()

    # Normal fsync should pass cleanly
    fsync_directory(d)

    # Missing directory or access error should be handled gracefully
    fsync_directory(tmp_path / "non_existent_dir")

    # NFS errors (EINVAL, ENOTSUP, EOPNOTSUPP, EBADF) must be caught and suppressed
    for err in (errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EBADF):
        with patch("os.fsync", side_effect=OSError(err, os.strerror(err))):
            # Must not raise
            fsync_directory(d)

    # Unexpected errors like EIO must be propagated
    with patch("os.fsync", side_effect=OSError(errno.EIO, "I/O error")):
        with pytest.raises(OSError) as exc_info:
            fsync_directory(d)
        assert exc_info.value.errno == errno.EIO


def test_file_lock_atomic_acquisition_and_release(tmp_path: Path):
    """AC-03: Verify atomic acquisition creates lock file and release removes it."""
    output_path = tmp_path / "unit_001.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")

    lock = FileLock(
        output_path=output_path,
        owner_token="node1:pid101:uuid1",
        manifest_row_hash="abc1234",
        lease_seconds=1800,
        heartbeat_interval=60.0,
    )

    assert not lock_path.exists()
    assert lock.acquire() is True
    assert lock_path.exists()

    # Verify lock file content is valid CampaignLock JSON
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "campaign_lock_v1"
    assert data["owner_token"] == "node1:pid101:uuid1"
    assert data["manifest_row_hash"] == "abc1234"
    assert data["heartbeat_count"] == 0

    # Release unlinks the lock
    lock.release()
    assert not lock_path.exists()


def test_file_lock_mutual_exclusion_contention(tmp_path: Path):
    """AC-03 Negative Control: Simultaneous acquisition fails with LockUnavailableError."""
    output_path = tmp_path / "unit_002.zarr"

    lock1 = FileLock(
        output_path=output_path,
        owner_token="worker_A:pid101",
        manifest_row_hash="row_hash_002",
        lease_seconds=1800,
    )
    lock2 = FileLock(
        output_path=output_path,
        owner_token="worker_B:pid102",
        manifest_row_hash="row_hash_002",
        lease_seconds=1800,
    )

    assert lock1.acquire() is True

    # Worker B cannot acquire active unexpired lock
    with pytest.raises(LockUnavailableError):
        lock2.acquire()

    lock1.release()


def test_file_lock_context_manager(tmp_path: Path):
    """Verify FileLock works cleanly as a context manager."""
    output_path = tmp_path / "unit_003.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")

    with FileLock(
        output_path=output_path,
        owner_token="worker_ctx:pid103",
        manifest_row_hash="row_hash_003",
    ) as lock:
        assert lock.is_locked
        assert lock_path.exists()

    assert not lock_path.exists()


def test_stale_lease_stealing_with_clock_drift_safety_margin(tmp_path: Path):
    """AC-04: Test stale lease stealing past 300s clock drift margin."""
    output_path = tmp_path / "unit_004.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    assert CLOCK_DRIFT_GRACE_PERIOD_S == 300

    now = time.time()
    stale_epoch = now - 3600  # 1 hour ago
    lease_seconds = 1800  # 30 mins

    # Case 1: Active lock (expires in future) -> Cannot steal
    active_payload = {
        "schema_version": "campaign_lock_v1",
        "owner_token": "dead_worker:pid999",
        "manifest_row_hash": "row_hash_004",
        "acquired_at_unix_s": stale_epoch,
        "lease_expires_at_unix_s": now + 600,
        "lease_seconds": lease_seconds,
        "heartbeat_count": 5,
    }
    lock_path.write_text(json.dumps(active_payload), encoding="utf-8")
    os.utime(lock_path, (now - 100, now - 100))

    stealer = FileLock(
        output_path=output_path,
        owner_token="live_worker:pid111",
        manifest_row_hash="row_hash_004",
        lease_seconds=lease_seconds,
    )

    with pytest.raises(LockUnavailableError):
        stealer.acquire()

    # Case 2: Lease expired, but within 300s clock drift grace period -> Cannot steal
    # expired 100s ago (< 300s grace period)
    near_expired_payload = {
        "schema_version": "campaign_lock_v1",
        "owner_token": "dead_worker:pid999",
        "manifest_row_hash": "row_hash_004",
        "acquired_at_unix_s": stale_epoch,
        "lease_expires_at_unix_s": now - 100,
        "lease_seconds": lease_seconds,
        "heartbeat_count": 5,
    }
    lock_path.write_text(json.dumps(near_expired_payload), encoding="utf-8")
    os.utime(lock_path, (now - 1900, now - 1900))

    with pytest.raises(LockUnavailableError):
        stealer.acquire()

    # Case 3: Lease expired, but file mtime is fresh (< lease_seconds + 300) -> Cannot steal
    lock_path.write_text(
        json.dumps(
            {
                **near_expired_payload,
                "lease_expires_at_unix_s": now - 400,  # past grace period on nominal expiry
            }
        ),
        encoding="utf-8",
    )
    # But mtime is fresh (e.g. touched 10 seconds ago)
    os.utime(lock_path, (now - 10, now - 10))

    with pytest.raises(LockUnavailableError):
        stealer.acquire()

    # Case 4: Truly stale: expired > 300s ago AND mtime > lease_seconds + 300s -> Steal succeeds!
    stale_payload = {
        "schema_version": "campaign_lock_v1",
        "owner_token": "dead_worker:pid999",
        "manifest_row_hash": "row_hash_004",
        "acquired_at_unix_s": stale_epoch,
        "lease_expires_at_unix_s": now - 350,  # > 300s expired
        "lease_seconds": lease_seconds,
        "heartbeat_count": 5,
    }
    lock_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    stale_mtime = now - (lease_seconds + 350)
    os.utime(lock_path, (stale_mtime, stale_mtime))

    assert stealer.acquire() is True
    assert stealer.is_locked

    # Verify stolen lock has newly minted acquired_at_unix_s and new owner
    stolen_data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert stolen_data["owner_token"] == "live_worker:pid111"
    assert stolen_data["heartbeat_count"] == 0
    assert stolen_data["acquired_at_unix_s"] >= now - 5.0

    stealer.release()
