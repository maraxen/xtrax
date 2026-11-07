"""Regression tests for R3 lease renewal interval and guard margin scaling."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

from xtrax.campaign.lease import FileLock, HeartbeatThread
from xtrax.campaign.marker import CampaignDoneMarker, ProducerInfo


def _make_marker(output_path: Path, row_hash: str) -> CampaignDoneMarker:
    return CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash=row_hash,
        input_hash=f"input_{row_hash}",
        producer=ProducerInfo(version="xtrax@0.5.0", git_sha="clean", hostname="localhost"),
        output_locator=str(output_path),
        content_digest_sha256="digest_sha256_r3",
        completed_at_utc="2026-11-06T00:00:00Z",
        completed_at_unix_s=time.time(),
        duration_s=10.0,
    )


def test_file_lock_12s_lease_held_10s_renews_and_commits(tmp_path: Path) -> None:
    """A FileLock with lease_seconds=12 and default heartbeat_interval held ~10s keeps its lease."""
    output_path = tmp_path / "unit_r3_12s.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    owner_token = "worker_r3:pid100"
    row_hash = "row_r3_12s_001"

    lock = FileLock(
        output_path=output_path,
        owner_token=owner_token,
        manifest_row_hash=row_hash,
        lease_seconds=12,
    )
    assert lock.acquire() is True
    assert lock.heartbeat is not None
    assert lock.acquired_at_unix_s is not None
    initial_expires_at = lock.acquired_at_unix_s + 12

    time.sleep(10.0)

    assert not lock.heartbeat.abort_event.is_set()
    assert lock.heartbeat.heartbeat_count >= 2
    assert lock_path.exists()
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_data["owner_token"] == owner_token
    assert lock_data["heartbeat_count"] >= 2
    assert lock_data["lease_expires_at_unix_s"] > initial_expires_at

    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = tmp_path / f"unit_r3_12s.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()
    (partial_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    marker = _make_marker(output_path, row_hash)

    result = lock.commit_output(
        partial_path=partial_path,
        output_path=output_path,
        done_marker=marker,
        attempt_uuid=attempt_uuid,
    )
    assert result == marker
    assert output_path.exists()
    assert not lock_path.exists()


def test_lease_120s_computes_effective_interval_below_guard_window(tmp_path: Path) -> None:
    """HeartbeatThread and FileLock with lease_seconds=120 derive interval below guard window."""
    lock_path = tmp_path / "unit_r3_120s.zarr.lock"
    hb = HeartbeatThread(
        lock_path=lock_path,
        owner_token="worker_r3:pid101",
        manifest_row_hash="row_r3_120s",
        lease_seconds=120,
    )
    assert hb.interval_seconds == pytest.approx(40.0)
    assert hb.expiry_guard_margin == pytest.approx(20.0)
    assert hb.interval_seconds < hb.lease_seconds - hb.expiry_guard_margin
    assert hb.expiry_guard_margin < hb.lease_seconds - hb.interval_seconds

    output_path = tmp_path / "unit_r3_120s_lock.zarr"
    lock = FileLock(
        output_path=output_path,
        owner_token="worker_r3:pid102",
        manifest_row_hash="row_r3_120s_lock",
        lease_seconds=120,
    )
    assert lock.heartbeat_interval == pytest.approx(40.0)
    assert lock.acquire() is True
    try:
        assert lock.heartbeat is not None
        assert lock.heartbeat.interval_seconds == pytest.approx(40.0)
        assert lock.heartbeat.expiry_guard_margin == pytest.approx(20.0)
        assert (
            lock.heartbeat.interval_seconds
            < lock.heartbeat.lease_seconds - lock.heartbeat.expiry_guard_margin
        )
    finally:
        lock.release()

    # Verify defaults for 1800s lease remain unchanged (interval 60, guard 60)
    hb_default = HeartbeatThread(
        lock_path=lock_path,
        owner_token="worker_r3:pid103",
        manifest_row_hash="row_r3_1800s",
        lease_seconds=1800,
    )
    assert hb_default.interval_seconds == pytest.approx(60.0)
    assert hb_default.expiry_guard_margin == pytest.approx(60.0)


def test_lease_seconds_under_10_raises_value_error(tmp_path: Path) -> None:
    """lease_seconds=5 raises ValueError in both FileLock and HeartbeatThread."""
    with pytest.raises(ValueError, match="lease_seconds"):
        FileLock(
            output_path=tmp_path / "unit_r3_5s.zarr",
            owner_token="worker_r3:pid104",
            manifest_row_hash="row_r3_5s",
            lease_seconds=5,
        )

    with pytest.raises(ValueError, match="lease_seconds"):
        HeartbeatThread(
            lock_path=tmp_path / "unit_r3_5s.zarr.lock",
            owner_token="worker_r3:pid104",
            manifest_row_hash="row_r3_5s",
            lease_seconds=5,
        )

    # Boundary check: lease_seconds=10 is allowed
    hb_10 = HeartbeatThread(
        lock_path=tmp_path / "unit_r3_10s.zarr.lock",
        owner_token="worker_r3:pid105",
        manifest_row_hash="row_r3_10s",
        lease_seconds=10,
    )
    assert hb_10.interval_seconds == pytest.approx(10.0 / 3.0)
    assert hb_10.expiry_guard_margin == pytest.approx(10.0 / 6.0)
    assert hb_10.interval_seconds < hb_10.lease_seconds - hb_10.expiry_guard_margin
