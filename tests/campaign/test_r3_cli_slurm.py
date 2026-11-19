"""Regression tests for Round-3 review findings on SLURM executor and campaign CLI."""

from __future__ import annotations

import errno
import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zarr

from tests.campaign.unit_fns import ARRAY_UNIT_FN, array_unit
from xtrax.campaign.executors.local import LocalSequentialExecutor
from xtrax.campaign.executors.slurm import SlurmArrayExecutor
from xtrax.campaign.lease import CLOCK_DRIFT_GRACE_PERIOD_S, CampaignLock
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ResourceRequirements,
    build_manifest_row,
    canonical_json_bytes,
)
from xtrax.campaign.marker import compute_content_digest, create_done_marker, quarantine_output
from xtrax.cli.campaign_verb import CampaignArgs, run_campaign
from xtrax.cli.errors import CLIError


def _make_manifest(num_rows: int = 4, campaign_id: str = "r3_test") -> CampaignManifest:
    rows = [
        build_manifest_row(
            campaign_id=campaign_id,
            job_id=f"job_{i:03d}",
            chunk_id=i,
            sample_start=i,
            sample_count=1,
            target_input="target.pdb",
            arm_id="arm_a",
            arm_backend="boltz2",
            checkpoint_id="chk_v1",
            fixed_policy="none",
            state_weight_profile="equal",
            seed=100 + i,
            parameters={"index": i},
            output_locator=f"{campaign_id}/unit_{i:03d}.zarr",
            resource_requirements=ResourceRequirements(0, 1, 60, None),
        )
        for i in range(num_rows)
    ]
    meta = ManifestMetadata(
        campaign_id=campaign_id,
        created_at_utc="2026-11-06T00:00:00Z",
        planner_version="xtrax@test",
        spec_sha256="0" * 64,
        total_rows=len(rows),
    )
    return CampaignManifest(schema_version="campaign_manifest_v1", metadata=meta, rows=rows)


def _write_manifest(path: Path, manifest: CampaignManifest) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest.to_dict()))
    return path


# ==============================================================================
# Finding 1: SlurmArrayExecutor row_indices bounds check, pre-filter before
# resume verification, and --resume-verify in generated script
# ==============================================================================


def test_slurm_bounds_checks_row_indices_and_cli_surfaces_cli_error(tmp_path: Path) -> None:
    """Out-of-range row_indices must raise ValueError in SlurmArrayExecutor and CLIError in CLI."""
    manifest = _make_manifest(num_rows=3)
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    out_dir = tmp_path / "out"

    executor = SlurmArrayExecutor(manifest_path=manifest_path, fn_spec=ARRAY_UNIT_FN)

    for bad_idx in (-1, 3, 99):
        with pytest.raises(ValueError, match="out of range"):
            executor.generate_script(manifest, out_dir, row_indices=[bad_idx])
        with pytest.raises(ValueError, match="out of range"):
            executor.run(manifest, array_unit, output_dir=str(out_dir), row_indices=[bad_idx])
        with pytest.raises(CLIError, match="out of range"):
            run_campaign(
                CampaignArgs(
                    action="run",
                    manifest=str(manifest_path),
                    output_dir=str(out_dir),
                    executor="slurm",
                    fn=ARRAY_UNIT_FN,
                    row_index=bad_idx,
                )
            )


def test_slurm_filters_row_indices_before_resume_verification(tmp_path: Path) -> None:
    """Only rows in row_indices may be verified and logged to reuse_log.jsonl."""
    manifest = _make_manifest(num_rows=4)
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    out_dir = tmp_path / "out"

    # Complete rows 0, 1, 2; leave row 3 pending.
    LocalSequentialExecutor().run(
        manifest, array_unit, output_dir=str(out_dir), row_indices=[0, 1, 2]
    )
    reuse_log = out_dir / "reuse_log.jsonl"
    assert not reuse_log.exists()

    executor = SlurmArrayExecutor(
        manifest_path=manifest_path,
        fn_spec=ARRAY_UNIT_FN,
        resume_verify="tier1",
    )

    # Select only rows [1, 3]: row 1 is done (reused), row 3 is pending.
    # Rows 0 and 2 must NOT be verified or logged to reuse_log.jsonl.
    summary = executor.run(
        manifest,
        array_unit,
        output_dir=str(out_dir),
        row_indices=[1, 3],
    )
    assert summary.total_units == 2
    assert summary.skipped_units == 1
    assert summary.submitted_units == 1

    records = [json.loads(line) for line in reuse_log.read_text(encoding="utf-8").splitlines()]
    assert [r["manifest_row_hash"] for r in records] == [manifest.rows[1].manifest_row_hash]
    assert records[0]["verified"] == "tier1"

    # Also verify generate_script directly filters before checking resume:
    reuse_log.unlink()
    script = executor.generate_script(
        manifest,
        out_dir,
        row_indices=[2, 3],
        resume=True,
    )
    assert script is not None
    records_gen = [json.loads(line) for line in reuse_log.read_text(encoding="utf-8").splitlines()]
    assert [r["manifest_row_hash"] for r in records_gen] == [manifest.rows[2].manifest_row_hash]


def test_slurm_generated_script_emits_resume_verify_mode(tmp_path: Path) -> None:
    """Generated sbatch script must pass --resume-verify <mode> to xtrax campaign run."""
    manifest = _make_manifest(num_rows=2)
    manifest_path = _write_manifest(tmp_path / "manifest.json", manifest)
    out_dir = tmp_path / "out"

    for mode in ("deep", "tier1"):
        executor = SlurmArrayExecutor(
            manifest_path=manifest_path,
            fn_spec=ARRAY_UNIT_FN,
            resume_verify=mode,
        )
        script = executor.generate_script(manifest, out_dir)
        assert script is not None
        assert f"--resume-verify {mode}" in script
        res = subprocess.run(["bash", "-n", "-c", script], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


# ==============================================================================
# Finding 2: _run_verify positional file target as manifest, and empty hashes in
# manifest-less mode fail verification
# ==============================================================================


def test_verify_positional_manifest_file_verifies_planned_rows(tmp_path: Path) -> None:
    """'xtrax campaign verify <path/to/manifest.json>' must verify manifest rows, not exit 0."""
    manifest = _make_manifest(num_rows=2, campaign_id="verify_pos")
    out_dir = tmp_path / "outputs"
    manifest_path = _write_manifest(out_dir / "manifest.json", manifest)

    # No units have been run yet: verifying the manifest file via positional target must fail.
    with pytest.raises(CLIError, match="verification failed"):
        run_campaign(CampaignArgs(action="verify", target=str(manifest_path)))

    # After running all units, verifying via positional manifest file passes all 2 units.
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out_dir))
    res = run_campaign(CampaignArgs(action="verify", target=str(manifest_path), deep=True))
    assert res == {"passed": 2, "failed": 0}


@pytest.mark.parametrize(
    ("row_hash", "input_hash"),
    [
        ("", "valid_input_hash"),
        ("valid_row_hash", ""),
        ("", ""),
    ],
)
def test_verify_manifestless_rejects_empty_hashes(
    tmp_path: Path, row_hash: str, input_hash: str
) -> None:
    """In manifest-less mode, a done marker with empty manifest_row_hash or input_hash must fail."""
    out_dir = tmp_path / "outputs"
    store_path = out_dir / "unit_000.zarr"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_group(str(store_path), mode="w").create_array("x", data=np.ones(4, dtype=np.float32))
    digest = compute_content_digest(store_path)

    marker = create_done_marker(
        manifest_row_hash=row_hash,
        input_hash=input_hash,
        output_locator=str(store_path),
        content_digest_sha256=digest,
        duration_s=0.1,
    )
    marker.write(store_path.with_name(f"{store_path.name}.done.json"))

    with pytest.raises(CLIError, match="verification failed"):
        run_campaign(CampaignArgs(action="verify", output_dir=str(out_dir)))


# ==============================================================================
# Finding 3: _run_status requires sibling store for Completed, ignores expired
# locks for completed units, and deduplicates ENOSPC .corrupt.<id> pairs
# ==============================================================================


def test_status_requires_sibling_store_ignores_completed_expired_lock_and_dedupes_corrupt_pair(
    tmp_path: Path,
) -> None:
    """Verify all three _run_status fixes:
    1. *.done.json without its sibling store is not counted as Completed.
    2. An expired lock for a completed unit is not counted as Failed.
    3. An ENOSPC quarantine pair (<store>.corrupt.<id> and <store>.done.json.corrupt.<id>)
       is counted once (keyed on <id>).
    """
    out_dir = tmp_path / "status_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Orphaned done marker whose sibling store is missing -> NOT Completed.
    orphan_store = out_dir / "orphan_unit.zarr"
    create_done_marker(
        manifest_row_hash="hash_orphan",
        input_hash="inp_orphan",
        output_locator=str(orphan_store),
        content_digest_sha256="0" * 64,
        duration_s=0.5,
    ).write(orphan_store.with_name(f"{orphan_store.name}.done.json"))

    # 2. Completed unit (store + done marker exist) with an expired lock -> Completed=1, Failed=0.
    done_store = out_dir / "done_unit.zarr"
    zarr.open_group(str(done_store), mode="w").create_array("x", data=np.zeros(2))
    create_done_marker(
        manifest_row_hash="hash_completed",
        input_hash="inp_completed",
        output_locator=str(done_store),
        content_digest_sha256=compute_content_digest(done_store),
        duration_s=0.5,
    ).write(done_store.with_name(f"{done_store.name}.done.json"))

    expired_lock = CampaignLock(
        schema_version="campaign_lock_v1",
        manifest_row_hash="hash_completed",
        owner_token="worker:old",
        acquired_at_unix_s=100.0,
        lease_expires_at_unix_s=time.time() - CLOCK_DRIFT_GRACE_PERIOD_S - 600.0,
        lease_seconds=60,
        heartbeat_count=0,
    )
    done_store.with_name(f"{done_store.name}.lock").write_text(
        expired_lock.to_json(), encoding="utf-8"
    )

    # 3. ENOSPC quarantine pair (<store>.corrupt.<id> + <marker>.corrupt.<id>) -> Quarantined=1.
    enospc_store = out_dir / "enospc_unit.zarr"
    zarr.open_group(str(enospc_store), mode="w").create_array("x", data=np.ones(2))
    create_done_marker(
        manifest_row_hash="hash_enospc",
        input_hash="inp_enospc",
        output_locator=str(enospc_store),
        content_digest_sha256=compute_content_digest(enospc_store),
        duration_s=0.5,
    ).write(enospc_store.with_name(f"{enospc_store.name}.done.json"))

    with patch.object(Path, "mkdir", side_effect=OSError(errno.ENOSPC, "No space left on device")):
        quarantine_output(enospc_store, reason="corrupt_chunk")

    corrupt_paths = list(out_dir.glob("*.corrupt.*"))
    assert len(corrupt_paths) == 2, f"Expected ENOSPC quarantine pair, got {corrupt_paths}"

    status = run_campaign(CampaignArgs(action="status", output_dir=str(out_dir)))
    assert status == {
        "completed": 1,
        "running": 0,
        "failed": 0,
        "quarantined": 1,
    }
