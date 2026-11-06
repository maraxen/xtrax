"""Tests for atomic file leasing, stale lease stealing, and fsync durabilization.

Covering [T2-01] and AC-03, AC-04 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.campaign.lease import (
    CLOCK_DRIFT_GRACE_PERIOD_S,
    CampaignLock,
    FileLock,
    LockUnavailableError,
    fsync_directory,
)


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
