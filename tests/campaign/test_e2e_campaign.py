"""End-to-end multi-arm campaign execution integration tests.

Covering [T5-02] and AC-01 through AC-12 from SPEC-261105-CAMPAIGN-RUNNER:
- Full campaign lifecycle: spec -> plan -> status -> execute (LocalProcessPoolExecutor)
  -> verify (Tier-1 and Tier-2) -> fast resume -> corruption quarantine recovery.
- Multi-target, multi-arm campaign execution with real Zarr output stores and .done.json markers.
- CLI verb integration: plan, run, status, verify.
- Single-unit SLURM array task row_index execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from tests.campaign.unit_fns import ARRAY_UNIT_FN
from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
)
from xtrax.campaign.manifest import (
    ManifestRow,
    plan_campaign,
)
from xtrax.campaign.marker import (
    validate_unit_done,
)
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid
from xtrax.cli.campaign_verb import CampaignArgs, run_campaign


def _custom_zarr_unit_fn(row: ManifestRow) -> dict[str, Any]:
    """Simulated prediction unit function returning compute state with tensors."""
    temp = float(row.parameters.get("temperature", 1.0))
    chunk_data = np.arange(100, dtype=np.float32).reshape((10, 10)) * temp + float(row.seed)

    return {
        "manifest_row_hash": row.manifest_row_hash,
        "predictions": chunk_data,
        "mean_val": float(chunk_data.mean()),
    }


def _resolve_row_store(base_dir: Path, row: ManifestRow) -> Path:
    loc = Path(row.output_locator)
    return loc if loc.is_absolute() else base_dir / loc


def test_e2e_multi_arm_lifecycle_process_pool(tmp_path: Path) -> None:
    """[T5-02]: End-to-end multi-arm campaign execution under LocalProcessPoolExecutor.

    Verifies:
    1. Spec declaration: 2 targets * 2 arms * 2 parameter points = 8 units.
    2. Host planning: produces deterministic manifest (campaign_manifest_v1).
    3. Status inspection: all 8 units initially pending.
    4. Process pool execution: 8 units execute across worker processes.
    5. Real Zarr stores and .done.json markers produced with matching digests.
    6. Status inspection: 8 units completed.
    7. Tier-1 and Tier-2 verification pass on all outputs.
    8. Fast resume: re-executing skips all 8 units in Tier-1 (< 5 ms per unit).
    9. Quarantine recovery: tampering 1 unit quarantines it, and resume re-runs only that unit.
    """
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"

    # 1. Spec declaration: 2 targets * 2 arms * 2 temperatures = 8 units
    spec = CampaignSpec(
        campaign_id="c_e2e_multiarm",
        target_inputs=["targets/target_alpha.pdb", "targets/target_beta.pdb"],
        arms=[
            ArmSpec(
                arm_id="arm_boltz",
                backend="boltz2",
                checkpoint_id="boltz2_v1",
                parameters={"recycling_steps": 3},
            ),
            ArmSpec(
                arm_id="arm_chai",
                backend="chai1",
                checkpoint_id="chai1_v2",
                parameters={"use_esm": False},
            ),
        ],
        grid=ParamGrid(axes={"temperature": [0.2, 0.8]}),
        output_dir=str(output_dir),
    )

    # 2. Plan manifest (AC-01)
    manifest = plan_campaign(spec, output_manifest_path=str(manifest_path))
    assert manifest.schema_version == "campaign_manifest_v1"
    assert manifest.metadata.campaign_id == "c_e2e_multiarm"
    assert len(manifest.rows) == 8

    # All row hashes and input hashes must be unique
    row_hashes = [r.manifest_row_hash for r in manifest.rows]
    input_hashes = [r.input_hash for r in manifest.rows]
    assert len(set(row_hashes)) == 8
    assert len(set(input_hashes)) == 8

    # 3. Status inspection before run: 0 completed
    status_args = CampaignArgs(action="status", output_dir=str(output_dir))
    initial_status = run_campaign(status_args)
    assert initial_status["completed"] == 0
    assert initial_status["running"] == 0
    assert initial_status["quarantined"] == 0

    # 4. Multi-process execution via LocalProcessPoolExecutor
    cache_dir = tmp_path / "jax_cache"
    executor = LocalProcessPoolExecutor(
        max_workers=2,
        recycle_interval=4,
        cache_dir=str(cache_dir),
        lease_seconds=300,
        heartbeat_interval=10.0,
    )

    summary = executor.run(
        manifest=manifest,
        unit_fn=_custom_zarr_unit_fn,
        output_dir=str(output_dir),
    )

    assert summary.total_units == 8
    assert summary.completed_units == 8
    assert summary.failed_units == 0
    assert summary.skipped_units == 0
    assert len(summary.failures) == 0

    # 5. Output inspection: verify all 8 Zarr stores and .done.json markers
    for row in manifest.rows:
        store_path = _resolve_row_store(output_dir, row)
        assert store_path.exists(), f"Store missing: {store_path}"
        assert (store_path / "zarr.json").exists()
        # The unit function's returned arrays are what the store holds.
        group = zarr.open_group(str(store_path), mode="r")
        temp = float(row.parameters.get("temperature", 1.0))
        expected = np.arange(100, dtype=np.float32).reshape((10, 10)) * temp + float(row.seed)
        np.testing.assert_array_equal(np.asarray(group["predictions"]), expected)
        assert group.attrs["mean_val"] == float(expected.mean())

        done_marker_path = store_path.with_name(f"{store_path.name}.done.json")
        assert done_marker_path.exists(), f"Done marker missing: {done_marker_path}"

        marker_data = json.loads(done_marker_path.read_text(encoding="utf-8"))
        assert marker_data["schema_version"] == "campaign_done_marker_v3"
        assert marker_data["manifest_row_hash"] == row.manifest_row_hash
        assert marker_data["input_hash"] == row.input_hash
        assert len(marker_data["content_digest_sha256"]) == 64

    # 6. Status inspection after run: 8 completed
    post_status = run_campaign(status_args)
    assert post_status["completed"] == 8
    assert post_status["running"] == 0
    assert post_status["failed"] == 0

    # 7. Verification: Tier-1 (< 5 ms) and Tier-2 deep verification
    for row in manifest.rows:
        store_path = _resolve_row_store(output_dir, row)
        # Tier-1 fast verification
        res_t1 = validate_unit_done(
            store_path, expected_row=row, deep_verify=False, quarantine_on_failure=False
        )
        assert res_t1.is_valid
        assert res_t1.tier == 1

        # Tier-2 deep verification
        res_t2 = validate_unit_done(
            store_path, expected_row=row, deep_verify=True, quarantine_on_failure=False
        )
        assert res_t2.is_valid

    # 8. Fast resume test: re-executing skips all completed units
    resume_summary = executor.run(
        manifest=manifest,
        unit_fn=_custom_zarr_unit_fn,
        output_dir=str(output_dir),
    )
    assert resume_summary.total_units == 8
    assert resume_summary.completed_units == 0
    assert resume_summary.skipped_units == 8
    assert resume_summary.accounted_units() == resume_summary.total_units
    assert resume_summary.failed_units == 0

    # 9. Corruption & quarantine recovery:
    # Tamper with metadata in store 0
    row_0 = manifest.rows[0]
    store_0 = _resolve_row_store(output_dir, row_0)
    root_meta_file = store_0 / "zarr.json"
    root_meta_file.write_text("corrupted_invalid_json_metadata", encoding="utf-8")

    # Tier-2 verification detects tampering and triggers defensive quarantine
    val_tampered = validate_unit_done(
        store_0, expected_row=row_0, deep_verify=True, quarantine_on_failure=True
    )
    assert not val_tampered.is_valid
    assert val_tampered.quarantine_path is not None
    assert val_tampered.quarantine_path.exists()
    assert not store_0.exists(), "Corrupt store should have been quarantined"

    # Status reflects quarantine
    status_quarantined = run_campaign(status_args)
    assert status_quarantined["completed"] == 7
    assert status_quarantined["quarantined"] >= 1

    # Resume re-executes only the missing unit (row 0): 1 completed, 7 skipped
    recovery_summary = executor.run(
        manifest=manifest,
        unit_fn=_custom_zarr_unit_fn,
        output_dir=str(output_dir),
    )
    assert recovery_summary.total_units == 8
    assert recovery_summary.completed_units == 1
    assert recovery_summary.skipped_units == 7
    assert recovery_summary.failed_units == 0
    assert store_0.exists()

    # Re-verify row 0 is valid again under Tier-2
    val_recovered = validate_unit_done(
        store_0, expected_row=row_0, deep_verify=True, quarantine_on_failure=False
    )
    assert val_recovered.is_valid


def test_e2e_campaign_cli_full_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify full end-to-end campaign lifecycle through CLI verbs (plan, run, status, verify)."""
    monkeypatch.chdir(tmp_path)
    spec_file = tmp_path / "campaign_spec.toml"
    out_dir = tmp_path / "cli_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = CampaignSpec(
        campaign_id="cli_camp_e2e",
        target_inputs=["targets/p1.pdb"],
        arms=[
            ArmSpec(arm_id="arm_seq", backend="boltz2", checkpoint_id="v1"),
            ArmSpec(arm_id="arm_diff", backend="chai1", checkpoint_id="v2"),
        ],
        grid=ParamGrid(axes={"steps": [10, 20]}),
        output_dir=str(out_dir),
    )
    spec.to_toml(spec_file)

    manifest_dest = tmp_path / "planned_manifest.json"

    # 1. Action: plan
    args_plan = CampaignArgs(
        action="plan",
        spec_path=str(spec_file),
        output=str(manifest_dest),
    )
    run_campaign(args_plan)
    assert manifest_dest.exists()

    manifest_data = json.loads(manifest_dest.read_text(encoding="utf-8"))
    assert len(manifest_data["rows"]) == 4

    # 2. Action: status (0 completed)
    args_status = CampaignArgs(action="status", output_dir=str(out_dir))
    s1 = run_campaign(args_status)
    assert s1["completed"] == 0

    # 3. Action: run with LocalSequentialExecutor
    args_run = CampaignArgs(
        action="run",
        fn=ARRAY_UNIT_FN,
        manifest=str(manifest_dest),
        output_dir=str(out_dir),
        executor="local-seq",
    )
    summary = run_campaign(args_run)
    assert summary.completed_units == 4
    assert summary.failed_units == 0

    # 4. Action: status (4 completed)
    s2 = run_campaign(args_status)
    assert s2["completed"] == 4
    assert s2["failed"] == 0

    # 5. Action: verify (Tier 1)
    args_verify = CampaignArgs(
        action="verify",
        manifest=str(manifest_dest),
        output_dir=str(out_dir),
        deep=False,
    )
    v_t1 = run_campaign(args_verify)
    assert v_t1["passed"] == 4
    assert v_t1["failed"] == 0

    # 6. Action: verify (Tier 2 Deep)
    args_verify_deep = CampaignArgs(
        action="verify",
        manifest=str(manifest_dest),
        output_dir=str(out_dir),
        deep=True,
    )
    v_t2 = run_campaign(args_verify_deep)
    assert v_t2["passed"] == 4
    assert v_t2["failed"] == 0


def test_e2e_single_unit_execution_and_array_task(tmp_path: Path) -> None:
    """Verify single-unit targeted execution (row_index) as used by SLURM array tasks."""
    out_dir = tmp_path / "single_unit_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tmp_path / "manifest.json"

    spec = CampaignSpec(
        campaign_id="c_single_unit",
        target_inputs=["t1.pdb", "t2.pdb"],
        arms=[ArmSpec(arm_id="arm1", backend="b1", checkpoint_id="c1")],
        grid=ParamGrid(axes={"p": [1, 2]}),
        output_dir=str(out_dir),
    )
    manifest = plan_campaign(spec, output_manifest_path=str(manifest_path))
    assert len(manifest.rows) == 4

    executor = LocalSequentialExecutor()

    # Execute only row_index=2
    summary_single = executor.run(
        manifest=manifest,
        unit_fn=_custom_zarr_unit_fn,
        output_dir=str(out_dir),
        row_indices=[2],
    )
    assert summary_single.total_units == 1
    assert summary_single.completed_units == 1
    assert summary_single.failed_units == 0

    # Check only row 2 was written
    target_row = manifest.rows[2]
    target_store = _resolve_row_store(out_dir, target_row)
    assert target_store.exists()
    assert target_store.with_name(f"{target_store.name}.done.json").exists()

    # The other 3 rows must not exist yet
    for idx in (0, 1, 3):
        other_row = manifest.rows[idx]
        assert not _resolve_row_store(out_dir, other_row).exists()

    # Now run the entire manifest: row 2 is skipped (fast resume), and rows 0, 1, 3 execute
    summary_all = executor.run(
        manifest=manifest,
        unit_fn=_custom_zarr_unit_fn,
        output_dir=str(out_dir),
    )
    assert summary_all.total_units == 4
    assert summary_all.completed_units == 3
    assert summary_all.skipped_units == 1  # Row 2 skipped!
    assert summary_all.failed_units == 0

    # All 4 rows now complete
    for row in manifest.rows:
        s = _resolve_row_store(out_dir, row)
        assert s.exists()
        assert s.with_name(f"{s.name}.done.json").exists()
