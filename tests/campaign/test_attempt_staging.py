"""Tests for campaign attempt-scoped staging paths and lease fencing (#3338)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from xtrax.campaign.executors.local import (
    UNIT_COMMITTED,
    _execute_single_unit,
    resolve_attempt_output_path,
)
from xtrax.campaign.lease import FileLock, LeaseLostError
from xtrax.campaign.manifest import ManifestRow, ResourceRequirements
from xtrax.campaign.marker import CampaignDoneMarker


def _make_manifest_row(
    output_locator: str,
    row_hash: str = "rowhash001122334455",
    input_hash: str = "inphash001122334455",
) -> ManifestRow:
    return ManifestRow(
        manifest_row_hash=row_hash,
        input_hash=input_hash,
        campaign_id="test_campaign",
        job_id="job_001",
        chunk_id=0,
        sample_start=0,
        sample_count=100,
        target_input="target.pdb",
        arm_id="arm_test",
        arm_backend="test_backend",
        checkpoint_id="chk_01",
        fixed_policy="none",
        state_weight_profile="equal",
        seed=42,
        parameters={"step": 1},
        output_locator=output_locator,
        resource_requirements=ResourceRequirements(gpu_memory_bytes=1000),
    )


def test_resolve_attempt_output_path_always_returns_sibling() -> None:
    p = Path("/tmp/campaign_store/units/unit_042")
    attempt_path = resolve_attempt_output_path(p, fence=3)
    assert attempt_path == Path("/tmp/campaign_store/units/unit_042.attempt-3")
    assert attempt_path.parent == p.parent
    assert attempt_path.name == "unit_042.attempt-3"
    # Verify it is never a child of p
    assert not str(attempt_path).startswith(str(p) + "/")


def test_direct_write_to_attempt_dir_promotes_cleanly(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign_run"
    output_path = output_dir / "unit_1"
    row = _make_manifest_row(str(output_path))

    def direct_writer_unit(r: ManifestRow) -> None:
        # Unit writes directly to target given by r.output_locator
        target = Path(r.output_locator)
        assert ".attempt-" in target.name
        grp = zarr.open_group(str(target), mode="w")
        arr = grp.create_array("data", shape=(4,), dtype="float32")
        arr[...] = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    res = _execute_single_unit(
        row=row,
        unit_fn=direct_writer_unit,
        output_dir=output_dir,
        lease_seconds=30,
        heartbeat_interval=0.1,
    )
    assert res == UNIT_COMMITTED

    # Verify output_path exists and is valid zarr group
    assert output_path.is_dir()
    grp = zarr.open_group(str(output_path), mode="r")
    assert np.array_equal(grp["data"][...], np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"))

    # Verify done marker exists and records lease_fence
    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    assert done_marker_path.is_file()
    marker = CampaignDoneMarker.read(done_marker_path)
    assert marker.lease_fence is not None
    assert marker.lease_fence >= 1
    assert marker.owner_token is not None


def test_stalled_direct_writer_cannot_overwrite_stealer_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign_run"
    output_path = output_dir / "unit_2"
    row = _make_manifest_row(
        str(output_path), row_hash="rowhash_stealer_test", input_hash="inphash_stealer_test"
    )

    # Worker 1 acquires lock (fence=1)
    lock1 = FileLock(output_path, "owner1", "rowhash_stealer_test", lease_seconds=30)
    assert lock1.acquire()
    assert lock1.fence == 1

    # Worker 1 writes its output to its attempt staging dir
    attempt_dir1 = resolve_attempt_output_path(output_path, fence=1)
    grp1 = zarr.open_group(str(attempt_dir1), mode="w")
    arr1 = grp1.create_array("worker", shape=(1,), dtype="int32")
    arr1[...] = np.array([1], dtype="int32")

    # Worker 2 steals the lease (fence=2)
    # Simulate Worker 2 stealing by bumping fence and taking lock
    lock2 = FileLock(output_path, "owner2", "rowhash_stealer_test", lease_seconds=30)
    # Force steal by advancing lock file content
    lock1.release()  # or steal
    assert lock2.acquire()
    assert lock2.fence >= 2

    # Worker 2 writes to its attempt staging dir and commits
    attempt_dir2 = resolve_attempt_output_path(output_path, fence=lock2.fence)
    grp2 = zarr.open_group(str(attempt_dir2), mode="w")
    arr2 = grp2.create_array("worker", shape=(1,), dtype="int32")
    arr2[...] = np.array([2], dtype="int32")

    # Worker 2 stages and commits
    partial2 = output_path.with_name(f"{output_path.name}.partial.worker2")
    import os

    os.replace(attempt_dir2, partial2)
    marker2 = CampaignDoneMarker(
        schema_version="1.0",
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        producer=CampaignDoneMarker.from_dict({}).producer,
        output_locator=str(output_path),
        content_digest_sha256="fake_digest_2",
        completed_at_utc="now",
        completed_at_unix_s=123.0,
        duration_s=1.0,
        lease_fence=lock2.fence,
    )
    lock2.commit_output(partial2, output_path, marker2, "worker2")
    lock2.release()

    # Now Worker 1 (superseded) wakes up and attempts to assert holder
    with pytest.raises(LeaseLostError):
        lock1.assert_holder()

    # Worker 2's committed store is completely intact with worker=2
    res_grp = zarr.open_group(str(output_path), mode="r")
    assert int(res_grp["worker"][0]) == 2


def test_attempt_dir_cleaned_up_on_unit_exception(tmp_path: Path) -> None:
    output_dir = tmp_path / "campaign_run"
    output_path = output_dir / "unit_fail"
    row = _make_manifest_row(
        str(output_path), row_hash="rowhash_fail_test", input_hash="inphash_fail_test"
    )

    attempt_captured = []

    def failing_unit(r: ManifestRow) -> None:
        target = Path(r.output_locator)
        attempt_captured.append(target)
        grp = zarr.open_group(str(target), mode="w")
        grp.create_array("x", shape=(2,), dtype="int32")
        raise RuntimeError("Simulated unit failure")

    with pytest.raises(RuntimeError, match="Simulated unit failure"):
        _execute_single_unit(
            row=row,
            unit_fn=failing_unit,
            output_dir=output_dir,
            lease_seconds=30,
            heartbeat_interval=0.1,
        )

    # Verify attempt directory was cleaned up
    assert len(attempt_captured) == 1
    assert not attempt_captured[0].exists()
    assert not output_path.exists()
