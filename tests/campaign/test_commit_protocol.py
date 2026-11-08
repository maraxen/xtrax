"""Tests for singleton commit token protocol, generation sealing, and zombie worker protection.

Covering [T2-03], GATE-03, and AC-07 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import pytest

from xtrax.campaign.lease import (
    FileLock,
    GenerationSealedError,
    LeaseLostError,
    StolenAttemptDiscarded,
)
from xtrax.campaign.marker import CampaignDoneMarker, ProducerInfo


def _make_dummy_marker(output_path: Path, row_hash: str = "row_hash_c001") -> CampaignDoneMarker:
    now = time.time()
    return CampaignDoneMarker(
        schema_version="campaign_done_marker_v3",
        manifest_row_hash=row_hash,
        input_hash="input_hash_c001",
        producer=ProducerInfo(version="xtrax@0.5.0", git_sha="clean", hostname="localhost"),
        output_locator=str(output_path),
        content_digest_sha256="digest_sha256_dummy",
        completed_at_utc="2026-11-05T19:00:00Z",
        completed_at_unix_s=now,
        duration_s=10.0,
    )


def test_commit_protocol_successful_promotion_and_cleanup(tmp_path: Path):
    """Verify normal commit moves partial to final, writes done marker, and cleans up."""
    output_path = tmp_path / "unit_c01.zarr"
    lock = FileLock(
        output_path=output_path,
        owner_token="worker_commit:pid301",
        manifest_row_hash="row_hash_c001",
        lease_seconds=600,
    )
    assert lock.acquire() is True

    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = tmp_path / f"unit_c01.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()
    (partial_path / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    marker = _make_dummy_marker(output_path, row_hash="row_hash_c001")

    commit_res = lock.commit_output(
        partial_path=partial_path,
        output_path=output_path,
        done_marker=marker,
        attempt_uuid=attempt_uuid,
    )
    assert commit_res == marker

    # 1. Partial directory should no longer exist
    assert not partial_path.exists()
    # 2. Final output path should exist
    assert output_path.exists()
    assert (output_path / ".zgroup").exists()
    # 3. Done marker should exist
    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    assert done_marker_path.exists()
    marker_data = json.loads(done_marker_path.read_text(encoding="utf-8"))
    assert marker_data["manifest_row_hash"] == "row_hash_c001"
    # 4. Commit token and lock file should be unlinked
    commit_path = output_path.with_name(f"{output_path.name}.commit")
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    assert not commit_path.exists()
    assert not lock_path.exists()


def test_gate_03_simultaneous_commit_singleton_exclusion(tmp_path: Path):
    """GATE-03 Negative Control: Simultaneous commit attempts fail with FileExistsError."""
    output_path = tmp_path / "unit_gate03.zarr"
    commit_token_path = output_path.with_name(f"{output_path.name}.commit")

    lock1 = FileLock(
        output_path=output_path,
        owner_token="worker_A:pid302",
        manifest_row_hash="row_hash_gate03",
        lease_seconds=600,
    )
    assert lock1.acquire() is True

    # Pre-create the singleton commit token to simulate a simultaneous commit in progress
    # Notice: the token path MUST NOT have attempt_uuid in it! It must be strictly <output>.commit
    fd = os.open(commit_token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.write(fd, b'{"attempt_uuid": "competing_worker", "owner_token": "worker_B:pid303"}')
    os.close(fd)
    assert commit_token_path.exists()

    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = tmp_path / f"unit_gate03.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()

    marker = _make_dummy_marker(output_path, row_hash="row_hash_gate03")

    # Invoking commit_output while singleton commit token exists MUST fail with FileExistsError
    with pytest.raises(FileExistsError):
        lock1.commit_output(
            partial_path=partial_path,
            output_path=output_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
        )

    # Partial path should be cleaned up; competing worker's commit token must still exist
    assert commit_token_path.exists()


def test_commit_protocol_discard_if_already_done(tmp_path: Path):
    """AC-07: Discard attempt and return StolenAttemptDiscarded if .done.json already exists."""
    output_path = tmp_path / "unit_already_done.zarr"
    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    done_marker_path.write_text('{"schema_version": "campaign_done_marker_v3"}', encoding="utf-8")

    lock = FileLock(
        output_path=output_path,
        owner_token="worker_late:pid304",
        manifest_row_hash="row_hash_c002",
        lease_seconds=600,
    )
    # Attempt uuid and partial directory
    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = tmp_path / f"unit_already_done.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()

    marker = _make_dummy_marker(output_path, row_hash="row_hash_c002")

    result = lock.commit_output(
        partial_path=partial_path,
        output_path=output_path,
        done_marker=marker,
        attempt_uuid=attempt_uuid,
    )

    # Must return StolenAttemptDiscarded and clean up partial directory
    assert isinstance(result, StolenAttemptDiscarded)
    assert not partial_path.exists()


def test_commit_protocol_generation_sealed_rejection(tmp_path: Path):
    """F-12 & AC-07: Reject with GenerationSealedError if generation is sealed."""
    gen_dir = tmp_path / "gen_0"
    gen_dir.mkdir()
    (gen_dir / ".generation_sealed").touch()

    output_path = gen_dir / "unit_sealed.zarr"
    lock = FileLock(
        output_path=output_path,
        owner_token="worker_zombie:pid305",
        manifest_row_hash="row_hash_c003",
        lease_seconds=600,
    )
    assert lock.acquire() is True

    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = gen_dir / f"unit_sealed.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()

    marker = _make_dummy_marker(output_path, row_hash="row_hash_c003")

    with pytest.raises(GenerationSealedError):
        lock.commit_output(
            partial_path=partial_path,
            output_path=output_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
            generation_dir=gen_dir,
        )

    # Partial path must be cleaned up
    assert not partial_path.exists()


def test_commit_protocol_zombie_worker_lost_lease_rejection(tmp_path: Path):
    """AC-07: Zombie worker whose lease was stolen raises LeaseLostError during commit."""
    output_path = tmp_path / "unit_zombie.zarr"
    lock_path = output_path.with_name(f"{output_path.name}.lock")

    lock = FileLock(
        output_path=output_path,
        owner_token="worker_zombie:pid306",
        manifest_row_hash="row_hash_c004",
        lease_seconds=600,
    )
    assert lock.acquire() is True

    attempt_uuid = uuid.uuid4().hex[:8]
    partial_path = tmp_path / f"unit_zombie.zarr.partial.{attempt_uuid}"
    partial_path.mkdir()

    # Simulate another worker stealing the lock while zombie was computing
    stolen_lock_payload = {
        "schema_version": "campaign_lock_v1",
        "owner_token": "worker_new:pid999",
        "manifest_row_hash": "row_hash_c004",
        "acquired_at_unix_s": time.time(),
        "lease_expires_at_unix_s": time.time() + 600,
        "lease_seconds": 600,
        "heartbeat_count": 0,
    }
    lock_path.write_text(json.dumps(stolen_lock_payload), encoding="utf-8")

    marker = _make_dummy_marker(output_path, row_hash="row_hash_c004")

    with pytest.raises(LeaseLostError):
        lock.commit_output(
            partial_path=partial_path,
            output_path=output_path,
            done_marker=marker,
            attempt_uuid=attempt_uuid,
        )

    # Partial path must be deleted and commit token unlinked
    assert not partial_path.exists()
    commit_path = output_path.with_name(f"{output_path.name}.commit")
    assert not commit_path.exists()
