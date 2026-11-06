"""P7 Code Review Remediation Regression Tests.

Verifies:
1. Zombie worker whose lock was stolen is denied commitment and raises LeaseLostError.
2. In-memory unit_fn creates output via lock.commit_output with valid done marker.
3. CLI 'xtrax campaign run --manifest <path>' without '--output-dir' places outputs
   in manifest parent without path doubling.
4. CLI 'xtrax campaign status' and 'verify' without '--output-dir' inspect manifest parent.
5. LocalProcessPoolExecutor exits early when max_failures is reached without blocking.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from tests.campaign.unit_fns import ARRAY_UNIT_FN
from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    _execute_single_unit,
    _resolve_output_path,
)
from xtrax.campaign.lease import FileLock, LeaseLostError
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
    build_manifest_row,
)
from xtrax.campaign.marker import (
    validate_unit_done,
)
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid
from xtrax.cli.campaign_verb import CampaignArgs, run_campaign


def _make_dummy_row(
    campaign_id: str = "camp_remed_01",
    index: int = 0,
    output_locator: str | None = None,
) -> ManifestRow:
    return build_manifest_row(
        campaign_id=campaign_id,
        job_id=f"job_{index:03d}",
        chunk_id=index,
        sample_start=index * 10,
        sample_count=10,
        target_input="target.pdb",
        arm_id="arm_test",
        arm_backend="boltz2",
        checkpoint_id="chk_01",
        output_locator=output_locator or f"{campaign_id}/unit_{index:03d}.zarr",
        resource_requirements=ResourceRequirements(
            gpu_memory_bytes=0,
            cpu_cores=1,
            timeout_seconds=60,
            padding_bucket_id=None,
        ),
    )


# ==============================================================================
# 1. Zombie Worker Lease Steal Rejection in _execute_single_unit
# ==============================================================================


def test_zombie_worker_lock_stolen_denied_commitment(tmp_path: Path) -> None:
    """Zombie worker whose lock lease was stolen during computation must be denied commit."""
    output_dir = tmp_path / "camp_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _make_dummy_row(campaign_id="camp_out", index=0)
    output_path = _resolve_output_path(output_dir, row)
    lock_path = output_path.with_name(f"{output_path.name}.lock")

    def unit_fn_stolen_while_running(r: ManifestRow) -> dict[str, Any]:
        # Simulate lock being stolen by another worker process while unit_fn is running
        assert lock_path.exists(), "Lock should have been acquired by primary worker"
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_data["owner_token"] = "stolen_by_thief_worker:99999"
        lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
        return {"data": "zombie_result"}

    with pytest.raises(LeaseLostError, match="lost, stolen, or expired"):
        _execute_single_unit(
            row=row,
            unit_fn=unit_fn_stolen_while_running,
            output_dir=output_dir,
            lease_seconds=300,
            heartbeat_interval=10.0,
        )

    # Verify that zombie output was NOT committed to output_path
    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    assert not done_marker_path.exists(), (
        "Done marker should NOT have been created by zombie worker"
    )
    assert not output_path.exists(), "Output store should NOT have been committed by zombie worker"

    # Verify partial directory was cleaned up
    partials = list(output_path.parent.glob(f"{output_path.name}.partial.*"))
    assert len(partials) == 0, f"Dangling partial directories found: {partials}"


# ==============================================================================
# 2. In-Memory unit_fn Commits via lock.commit_output Protocol
# ==============================================================================


def test_in_memory_unit_fn_commits_via_protocol(tmp_path: Path) -> None:
    """Standard in-memory unit_fn must stage and commit output via lock.commit_output."""
    output_dir = tmp_path / "in_memory_camp"
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _make_dummy_row(campaign_id="in_memory_camp", index=1)
    output_path = _resolve_output_path(output_dir, row)

    def pure_in_memory_fn(r: ManifestRow) -> dict[str, Any]:
        # Returns in-memory state without touching disk
        return {"tensor": np.ones((10, 10)), "row_hash": r.manifest_row_hash}

    orig_commit = FileLock.commit_output
    calls = []

    def spy_commit(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((self, args, kwargs))
        return orig_commit(self, *args, **kwargs)

    with patch.object(FileLock, "commit_output", side_effect=spy_commit, autospec=True):
        _execute_single_unit(
            row=row,
            unit_fn=pure_in_memory_fn,
            output_dir=output_dir,
            lease_seconds=300,
        )
        assert len(calls) == 1, "lock.commit_output was not called!"

    # Output store and done marker should exist and pass deep validation
    assert output_path.exists(), f"Output store {output_path} does not exist"
    # The returned mapping is persisted as a zarr v3 group (user decision 261106:
    # persist a returned Mapping; an empty synthetic store is no longer written).
    assert (output_path / "zarr.json").exists()
    import zarr

    np.testing.assert_array_equal(
        np.asarray(zarr.open_group(str(output_path), mode="r")["tensor"]), np.ones((10, 10))
    )

    done_marker_path = output_path.with_name(f"{output_path.name}.done.json")
    assert done_marker_path.exists(), f"Done marker {done_marker_path} does not exist"

    val = validate_unit_done(output_path, expected_row=row, deep_verify=True)
    assert val.is_valid is True, f"Validation failed: {val.error_message}"

    # Verify no dangling lock or commit token
    assert not output_path.with_name(f"{output_path.name}.lock").exists()
    assert not output_path.with_name(f"{output_path.name}.commit").exists()


# ==============================================================================
# 3. CLI 'run' Without --output-dir Resolves to Manifest Parent (No Doubling)
# ==============================================================================


def test_cli_run_without_output_dir_no_path_doubling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """xtrax campaign run without --output-dir must use manifest_path.parent without doubling."""
    monkeypatch.chdir(tmp_path)
    camp_dir = tmp_path / "screen_target"
    spec = CampaignSpec(
        campaign_id="screen_target",
        target_inputs=["protA.pdb"],
        arms=[ArmSpec(arm_id="boltz", backend="boltz2", checkpoint_id="chk1")],
        grid=ParamGrid(axes={"temp": [0.1, 0.5]}),
        output_dir=str(camp_dir),
    )
    spec_path = tmp_path / "spec.toml"
    spec.to_toml(spec_path)

    # 1. Plan campaign
    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path)))
    manifest_path = camp_dir / "manifest.json"
    assert manifest_path.exists()

    # 2. Run campaign without passing output_dir
    summary = run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            executor="local-seq",
        )
    )
    assert summary.completed_units == 2

    # 3. Verify outputs are placed directly in camp_dir without doubling
    doubled_dir = camp_dir / "screen_target"
    assert not doubled_dir.exists(), f"Path doubling detected! Found directory: {doubled_dir}"

    zarr_stores = list(camp_dir.glob("*.zarr"))
    assert len(zarr_stores) == 2, (
        f"Expected 2 zarr stores directly in {camp_dir}, found {zarr_stores}"
    )

    # 4. Verify no output files leaked to current working directory
    cwd_doubled = tmp_path / "screen_target" / "screen_target"
    assert not cwd_doubled.exists()


# ==============================================================================
# 4. CLI 'status' and 'verify' Without --output-dir Inspect Manifest Parent
# ==============================================================================


def test_cli_status_and_verify_without_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """xtrax campaign status and verify without --output-dir must inspect manifest_path.parent."""
    monkeypatch.chdir(tmp_path)
    camp_dir = tmp_path / "audit_target"
    spec = CampaignSpec(
        campaign_id="audit_target",
        target_inputs=["protA.pdb"],
        arms=[ArmSpec(arm_id="boltz", backend="boltz2", checkpoint_id="chk1")],
        output_dir=str(camp_dir),
    )
    spec_path = tmp_path / "spec.toml"
    spec.to_toml(spec_path)

    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path)))
    manifest_path = camp_dir / "manifest.json"

    # Run the campaign
    run_campaign(
        CampaignArgs(
            action="run", fn=ARRAY_UNIT_FN, manifest=str(manifest_path), executor="local-seq"
        )
    )

    # Status without output_dir
    status_res = run_campaign(CampaignArgs(action="status", manifest=str(manifest_path)))
    assert status_res["completed"] == 1
    assert status_res["failed"] == 0

    # Verify without output_dir
    verify_res = run_campaign(CampaignArgs(action="verify", manifest=str(manifest_path), deep=True))
    assert verify_res["passed"] == 1
    assert verify_res["failed"] == 0


# ==============================================================================
# 5. LocalProcessPoolExecutor Early Exit on max_failures
# ==============================================================================


def _failing_worker_fn(row: ManifestRow) -> dict[str, Any]:
    """Worker function that fails immediately on row index 0 and sleeps on others."""
    idx = int(row.parameters.get("index", 0))
    if idx == 0:
        raise RuntimeError("Immediate worker failure on row 0")
    # Simulate a slow task that would take long if not terminated
    time.sleep(2.0)
    return {"status": "ok"}


def test_process_pool_max_failures_early_exit(tmp_path: Path) -> None:
    """LocalProcessPoolExecutor must terminate workers early when max_failures is reached."""
    output_dir = tmp_path / "pool_early_exit"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(10):
        row = build_manifest_row(
            campaign_id="pool_early_exit",
            job_id=f"job_{i:03d}",
            chunk_id=i,
            sample_start=i * 10,
            sample_count=10,
            target_input="target.pdb",
            arm_id="arm_test",
            arm_backend="boltz2",
            checkpoint_id="chk_01",
            parameters={"index": i},
            output_locator=f"pool_early_exit/unit_{i:03d}.zarr",
        )
        rows.append(row)

    manifest = CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=ManifestMetadata(
            campaign_id="pool_early_exit",
            created_at_utc="2026-11-05T20:00:00Z",
            planner_version="xtrax@0.5.0",
            spec_sha256="0" * 64,
            total_rows=len(rows),
        ),
        rows=rows,
    )

    executor = LocalProcessPoolExecutor(max_workers=2)

    t0 = time.time()
    summary = executor.run(
        manifest=manifest,
        unit_fn=_failing_worker_fn,
        output_dir=str(output_dir),
        max_failures=1,
    )
    elapsed = time.time() - t0

    assert summary.failed_units >= 1
    # 10 units with 2s each across 2 workers would take ~10s without early termination.
    # With early exit on max_failures=1, it should terminate well before all 10 complete.
    assert summary.completed_units < 10
    assert elapsed < 5.0, f"Early exit took too long: {elapsed:.2f}s"
