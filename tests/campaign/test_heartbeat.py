"""Tests for the HeartbeatThread with CAS expiration guard.

Covering [T2-02] and AC-05, F-05 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from xtrax.campaign.lease import CampaignLock, HeartbeatThread


def test_heartbeat_thread_lifecycle_and_tick(tmp_path: Path):
    """Verify heartbeat thread periodically updates lock file and advances count."""
    lock_path = tmp_path / "unit_hb.zarr.lock"
    now = time.time()
    owner_token = "worker1:pid200:abc"
    row_hash = "row_hash_hb_001"

    initial_lock = CampaignLock(
        schema_version="campaign_lock_v1",
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        acquired_at_unix_s=now,
        lease_expires_at_unix_s=now + 10.0,
        lease_seconds=10,
        heartbeat_count=0,
    )
    lock_path.write_text(json.dumps(initial_lock.to_dict()), encoding="utf-8")

    # Start heartbeat thread with 0.1s interval and 10s lease
    hb = HeartbeatThread(
        lock_path=lock_path,
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=10,
        interval_seconds=0.05,
        initial_acquired_at=now,
    )
    assert not hb.abort_event.is_set()

    hb.start()
    time.sleep(0.15)  # Allow at least 2 ticks
    hb.stop()
    hb.join(timeout=1.0)

    assert not hb.is_alive()
    assert not hb.abort_event.is_set()

    # Read back lock file: heartbeat_count must have incremented, initial_acquired_at unchanged
    data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert data["heartbeat_count"] >= 1
    assert data["acquired_at_unix_s"] == pytest.approx(now, abs=1e-3)
    assert data["owner_token"] == owner_token


def test_heartbeat_thread_pre_renewal_expiry_guard(tmp_path: Path):
    """F-05 Negative Control: Thread sets abort_event if time is within 60s of lease expiry."""
    lock_path = tmp_path / "unit_guard.zarr.lock"
    now = time.time()
    owner_token = "worker_guard:pid201"
    row_hash = "row_hash_hb_002"

    initial_lock = CampaignLock(
        schema_version="campaign_lock_v1",
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        acquired_at_unix_s=now - 1750,
        lease_expires_at_unix_s=now + 30.0,  # Only 30s remaining (< 60s safety margin!)
        lease_seconds=1800,
        heartbeat_count=0,
    )
    lock_path.write_text(json.dumps(initial_lock.to_dict()), encoding="utf-8")

    hb = HeartbeatThread(
        lock_path=lock_path,
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=1800,
        interval_seconds=0.02,
        initial_acquired_at=now - 1750,
    )
    # Explicitly set current_lease_expires_at to match
    hb.current_lease_expires_at = now + 30.0

    hb.start()
    time.sleep(0.08)
    hb.stop()
    hb.join(timeout=1.0)

    # Expiry guard should have tripped and set abort_event
    assert hb.abort_event.is_set()


def test_heartbeat_thread_cas_stolen_lock_abort(tmp_path: Path):
    """AC-05 Negative Control: Thread sets abort_event if lock is stolen by another worker."""
    lock_path = tmp_path / "unit_stolen.zarr.lock"
    now = time.time()
    my_owner_token = "worker_original:pid202"
    row_hash = "row_hash_hb_003"

    initial_lock = CampaignLock(
        schema_version="campaign_lock_v1",
        owner_token=my_owner_token,
        manifest_row_hash=row_hash,
        acquired_at_unix_s=now,
        lease_expires_at_unix_s=now + 600,
        lease_seconds=600,
        heartbeat_count=0,
    )
    lock_path.write_text(json.dumps(initial_lock.to_dict()), encoding="utf-8")

    hb = HeartbeatThread(
        lock_path=lock_path,
        owner_token=my_owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=600,
        interval_seconds=0.05,
        initial_acquired_at=now,
    )
    hb.start()
    time.sleep(0.08)  # Let it tick at least once
    assert not hb.abort_event.is_set()

    # Simulate another worker stealing the lock
    stolen_lock = {
        "schema_version": "campaign_lock_v1",
        "owner_token": "worker_thief:pid999",
        "manifest_row_hash": row_hash,
        "acquired_at_unix_s": time.time(),
        "lease_expires_at_unix_s": time.time() + 600,
        "lease_seconds": 600,
        "heartbeat_count": 0,
    }
    lock_path.write_text(json.dumps(stolen_lock), encoding="utf-8")

    # Give heartbeat thread time to detect ownership mismatch on next tick
    time.sleep(0.12)
    hb.stop()
    hb.join(timeout=1.0)

    assert hb.abort_event.is_set()


def test_heartbeat_thread_deleted_lock_abort(tmp_path: Path):
    """Verify thread sets abort_event if lock file vanishes on disk."""
    lock_path = tmp_path / "unit_vanish.zarr.lock"
    now = time.time()
    owner_token = "worker_vanish:pid203"
    row_hash = "row_hash_hb_004"

    initial_lock = CampaignLock(
        schema_version="campaign_lock_v1",
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        acquired_at_unix_s=now,
        lease_expires_at_unix_s=now + 600,
        lease_seconds=600,
        heartbeat_count=0,
    )
    lock_path.write_text(json.dumps(initial_lock.to_dict()), encoding="utf-8")

    hb = HeartbeatThread(
        lock_path=lock_path,
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=600,
        interval_seconds=0.05,
        initial_acquired_at=now,
    )
    hb.start()
    time.sleep(0.08)
    assert not hb.abort_event.is_set()

    # Delete lock file
    lock_path.unlink()

    time.sleep(0.12)
    hb.stop()
    hb.join(timeout=1.0)

    assert hb.abort_event.is_set()
