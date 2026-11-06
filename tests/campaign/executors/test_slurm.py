"""Tests for SlurmArrayExecutor and SLURM array script generator.

Covering [T3-04], GATE-08, GATE-09, and AC-09 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from xtrax.campaign.executors.slurm import (
    SlurmArrayExecutor,
    format_slurm_array_indices,
    get_pending_row_indices,
)
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
    build_manifest_row,
    canonical_json_bytes,
)


def _create_slurm_manifest(
    num_rows: int = 100,
    campaign_id: str = "camp_slurm_test",
) -> CampaignManifest:
    rows = []
    for i in range(num_rows):
        row = build_manifest_row(
            campaign_id=campaign_id,
            job_id=f"job_{i:03d}",
            chunk_id=i,
            sample_start=i * 10,
            sample_count=10,
            target_input="target.pdb",
            arm_id="arm_slurm",
            arm_backend="boltz2",
            checkpoint_id="chk_v1",
            fixed_policy="none",
            state_weight_profile="equal",
            seed=1000 + i,
            parameters={"idx": i},
            output_locator=f"{campaign_id}/unit_{i:03d}.zarr",
            resource_requirements=ResourceRequirements(
                gpu_memory_bytes=8000000000,
                cpu_cores=4,
                timeout_seconds=7200,
            ),
        )
        rows.append(row)

    metadata = ManifestMetadata(
        campaign_id=campaign_id,
        created_at_utc="2026-11-05T19:00:00Z",
        planner_version="xtrax@0.5.0",
        spec_sha256="0" * 64,
        total_rows=len(rows),
    )
    return CampaignManifest(
        schema_version="campaign_manifest_v1",
        metadata=metadata,
        rows=rows,
    )


def _mark_done(output_dir: Path, row: ManifestRow) -> None:
    """Commit a real unit for ``row``: a zarr store plus a matching done marker."""
    import numpy as np
    import zarr

    from xtrax.campaign.executors.local import _resolve_output_path
    from xtrax.campaign.marker import compute_content_digest, create_done_marker

    store = _resolve_output_path(output_dir, row)
    store.parent.mkdir(parents=True, exist_ok=True)
    zarr.open_group(str(store), mode="w").create_array("x", data=np.zeros(2))
    create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(store),
        content_digest_sha256=compute_content_digest(store),
        duration_s=0.1,
    ).write(store.with_name(f"{store.name}.done.json"))


def test_format_slurm_array_indices_ranges():
    """Verify conversion of sparse indices into compact SLURM range strings."""
    # Empty
    assert format_slurm_array_indices([]) == ""

    # Single index
    assert format_slurm_array_indices([5]) == "5"
    assert format_slurm_array_indices([5], max_concurrent=10) == "5%10"

    # Discrete discontinuous indices
    assert format_slurm_array_indices([1, 3, 5]) == "1,3,5"

    # Single contiguous range
    assert format_slurm_array_indices(list(range(10))) == "0-9"

    # Mixed ranges and isolated indices with concurrency limit
    indices = [12, 15, 44, 45, 46, 47, 48, 49, 50]
    assert format_slurm_array_indices(indices, max_concurrent=20) == "12,15,44-50%20"
    assert format_slurm_array_indices(indices) == "12,15,44-50"

    # Duplicate and unsorted input handling
    unsorted_indices = [50, 44, 12, 45, 15, 46, 47, 48, 49, 12, 50]
    assert format_slurm_array_indices(unsorted_indices, max_concurrent=20) == "12,15,44-50%20"


def test_get_pending_row_indices_with_done_markers(tmp_path: Path):
    """Verify get_pending_row_indices correctly detects completed vs uncompleted rows."""
    manifest = _create_slurm_manifest(num_rows=10)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mark rows 0, 1, 2, 5 as completed
    completed_indices = {0, 1, 2, 5}
    for idx in completed_indices:
        row = manifest.rows[idx]
        _mark_done(output_dir, row)

    pending = get_pending_row_indices(manifest, output_dir)
    assert pending == [3, 4, 6, 7, 8, 9]


def test_gate_08_positive_control_sparse_range_on_resume(tmp_path: Path):
    """GATE-08: Positive Control: sparse range on resume.

    SlurmArrayExecutor emits sparse range #SBATCH --array=12,15,44-50%20 on resume.
    Per AC-09 and GATE-08 from SPEC-261105-CAMPAIGN-RUNNER:
    - 100-unit manifest where units 12, 15, and 44-50 are pending
    - Script generates '#SBATCH --array=12,15,44-50%20'
    - Script passes bash -n syntax verification
    """
    manifest = _create_slurm_manifest(num_rows=100, campaign_id="gate08_campaign")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pending_targets = {12, 15, 44, 45, 46, 47, 48, 49, 50}
    # Mark all rows EXCEPT pending_targets as completed
    for idx, row in enumerate(manifest.rows):
        if idx not in pending_targets:
            _mark_done(output_dir, row)

    # Verify pending detected correctly
    detected_pending = get_pending_row_indices(manifest, output_dir)
    assert detected_pending == sorted(pending_targets)

    executor = SlurmArrayExecutor(
        job_name="gate08_run",
        cpus_per_task=4,
        gpus=1,
        time_limit="02:00:00",
        max_concurrent=20,
    )

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_bytes(canonical_json_bytes(manifest.to_dict()))
    script_path = tmp_path / "submit_gate08.sh"

    script_text = executor.generate_script(
        manifest=manifest,
        output_dir=output_dir,
        manifest_path=manifest_file,
        fn_spec="mypkg.units:predict",
        resume=True,
        script_path=script_path,
    )

    assert script_text is not None
    # Each array task runs exactly one row with the real manifest, output dir and fn.
    assert f"MANIFEST_PATH={manifest_file.resolve()}" in script_text
    assert f"OUTPUT_DIR={output_dir.resolve()}" in script_text
    assert "UNIT_FN=mypkg.units:predict" in script_text
    assert '--fn "${UNIT_FN}"' in script_text
    assert '--output-dir "${OUTPUT_DIR}"' in script_text
    assert "--executor local-seq" in script_text
    assert "#SBATCH --array=12,15,44-50%20" in script_text
    assert "#SBATCH --job-name=gate08_run" in script_text
    assert "#SBATCH --cpus-per-task=4" in script_text
    assert "#SBATCH --gres=gpu:1" in script_text
    assert script_path.exists()

    # Positive control: script passes bash -n syntax verification
    res = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"Script syntax error: {res.stderr}"


def test_gate_09_boundary_control_empty_pending_indices(tmp_path: Path, capsys):
    """GATE-09: Boundary Control: early return on empty pending_indices.

    SlurmArrayExecutor returns early with completion message when pending_indices == [].
    Per AC-09 and GATE-09 from SPEC-261105-CAMPAIGN-RUNNER:
    - When all units are complete, generate_script returns None and writes no script
    - Emits clean completion message
    - run() returns CampaignSummary with all units skipped/completed and 0 failures
    """
    manifest = _create_slurm_manifest(num_rows=5, campaign_id="gate09_campaign")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mark ALL 5 rows as completed
    for row in manifest.rows:
        _mark_done(output_dir, row)

    executor = SlurmArrayExecutor(job_name="gate09_run")
    script_path = tmp_path / "submit_gate09.sh"

    # 1. generate_script returns None and writes NO script
    result = executor.generate_script(
        manifest=manifest,
        output_dir=output_dir,
        resume=True,
        script_path=script_path,
    )
    assert result is None
    assert not script_path.exists(), "GATE-09 FAILED: Script was written despite 0 pending units"

    captured = capsys.readouterr()
    assert "already completed" in captured.out or "No tasks to schedule" in captured.out

    # 2. run() returns CampaignSummary cleanly without running sbatch
    summary = executor.run(
        manifest=manifest,
        unit_fn=lambda r: None,
        output_dir=str(output_dir),
    )
    assert summary.total_units == 5
    assert summary.completed_units == 0
    assert summary.skipped_units == 5
    assert summary.failed_units == 0
    assert summary.submitted_units == 0
    assert summary.accounted_units() == summary.total_units
    assert len(summary.failures) == 0


def test_slurm_array_executor_cancel_tasks():
    """Verify cancel_tasks issues scancel <job_id>_<task_id>."""
    executor = SlurmArrayExecutor(job_id="123456")

    mock_run = MagicMock()
    with patch("subprocess.run", mock_run):
        executor.cancel_tasks([12, 15, 44])

    assert mock_run.call_count == 3
    mock_run.assert_any_call(["scancel", "123456_12"], check=True)
    mock_run.assert_any_call(["scancel", "123456_15"], check=True)
    mock_run.assert_any_call(["scancel", "123456_44"], check=True)


def test_slurm_array_executor_cancel_tasks_explicit_job_id():
    """Verify cancel_tasks accepts explicit job_id override."""
    executor = SlurmArrayExecutor()

    mock_run = MagicMock()
    with patch("subprocess.run", mock_run):
        executor.cancel_tasks([99], job_id="778899")

    mock_run.assert_called_once_with(["scancel", "778899_99"], check=True)
