"""Regression tests for the 261105 code-review findings on the campaign runner.

One section per finding. Each test failed on the pre-fix tree (4220514) and passes after.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from tests.campaign.unit_fns import ARRAY_UNIT_FN, array_unit
from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    _check_resume_done,
    _initialize_worker_process,
    _resolve_output_path,
)
from xtrax.campaign.executors.slurm import SlurmArrayExecutor, get_pending_row_indices
from xtrax.campaign.lease import (
    CLOCK_DRIFT_GRACE_PERIOD_S,
    FileLock,
    LockUnavailableError,
)
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
    build_manifest_row,
    canonical_json_bytes,
)
from xtrax.campaign.marker import (
    compute_content_digest,
    create_done_marker,
    quarantine_output,
    validate_unit_done,
)
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid
from xtrax.cli.errors import CLIError

CAMPAIGN_ID = "review261105"


def _manifest(num_rows: int = 3, campaign_id: str = CAMPAIGN_ID) -> CampaignManifest:
    rows = [
        build_manifest_row(
            campaign_id=campaign_id,
            job_id=f"job_{i:03d}",
            chunk_id=i,
            sample_start=i,
            sample_count=1,
            target_input="t.pdb",
            arm_id="arm",
            arm_backend="boltz2",
            checkpoint_id="v1",
            fixed_policy="none",
            state_weight_profile="equal",
            seed=10 + i,
            parameters={"index": i},
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


def _marker_path(store: Path) -> Path:
    return store.with_name(f"{store.name}.done.json")


def _backdate(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


# ==============================================================================
# 1. A unit that produces no output must fail, not be stamped done
# ==============================================================================


def test_unit_writing_nothing_fails_and_is_not_marked_done(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=2)
    out = tmp_path / "out"

    summary = LocalSequentialExecutor().run(manifest, lambda row: None, output_dir=str(out))

    assert summary.failed_units == 2
    assert summary.completed_units == 0
    assert "UnitProducedNoOutputError" in summary.failures[0]["error"]
    for row in manifest.rows:
        store = _resolve_output_path(out, row)
        assert not store.exists()
        assert not _marker_path(store).exists()
    # A later run with a real unit function still runs every unit.
    rerun = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    assert rerun.completed_units == 2


def test_unit_returning_empty_mapping_fails(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=1)
    summary = LocalSequentialExecutor().run(manifest, lambda row: {}, output_dir=str(tmp_path))
    assert summary.failed_units == 1
    assert summary.completed_units == 0


def test_returned_mapping_is_persisted(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=1)
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    row = manifest.rows[0]
    group = zarr.open_group(str(_resolve_output_path(tmp_path, row)), mode="r")
    np.testing.assert_array_equal(np.asarray(group["values"]), array_unit(row)["values"])
    assert group.attrs["seed"] == row.seed


def test_unit_writing_its_own_store_is_used(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=1)

    def writes_store(row: ManifestRow) -> None:
        store = _resolve_output_path(tmp_path, row)
        zarr.open_group(str(store), mode="w").create_array("own", data=np.ones(3))

    summary = LocalSequentialExecutor().run(manifest, writes_store, output_dir=str(tmp_path))
    assert summary.completed_units == 1
    store = _resolve_output_path(tmp_path, manifest.rows[0])
    assert "own" in zarr.open_group(str(store), mode="r")


def test_cli_run_without_fn_is_refused(tmp_path: Path) -> None:
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    manifest_path = _write_manifest(tmp_path / "m.json", _manifest(num_rows=1))
    with pytest.raises(CLIError, match="--fn"):
        run_campaign(CampaignArgs(action="run", manifest=str(manifest_path)))


_SWEEP_CONFIG = """
schema_version = 1
seed = 0
num_epochs = 1

[model]
path = "tests.cli._run_fixtures:make_model"
kwargs = {}

[optimizer]
path = "xtrax.training.optim:adamw_with_schedule"
kwargs = {peak_lr = 1e-3}

[loss]
path = "tests.cli._run_fixtures:make_loss"
kwargs = {}

[data]
factory = "tests.cli._run_fixtures:make_dataset"
kwargs = {}
batch_size = 2

[sweep.axes]
seed = [1, 2]
"""


def test_failed_sweep_point_is_not_marked_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xtrax.cli.sweep_verb import SweepArgs, run_sweep

    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.toml"
    cfg.write_text(_SWEEP_CONFIG)

    def fake_run(cfg_obj: Any, run_id: str | None = None) -> Any:
        if cfg_obj.seed == 2:
            raise RuntimeError("diverged")
        return MagicMock()

    with (
        patch("xtrax.cli.sweep_verb.run_from_config", fake_run),
        patch("xtrax.cli.sweep_verb._record_sweep_failure"),
        pytest.warns(FutureWarning),
    ):
        run_sweep(SweepArgs(config_path=str(cfg)))

    markers = [
        m for m in Path(".xtrax/sweeps").rglob("*.done.json") if ".quarantine" not in m.parts
    ]
    assert len(markers) == 1, markers
    store = markers[0].with_name(markers[0].name.removesuffix(".done.json"))
    assert "run_id" in zarr.open_group(str(store), mode="r").attrs


# ==============================================================================
# 2. The SLURM script must name the real manifest, --fn and --output-dir
# ==============================================================================


def test_slurm_script_runs_real_manifest_and_fn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    manifest_path = _write_manifest(tmp_path / "plans" / "m.json", _manifest(num_rows=8))
    results = tmp_path / "results"

    summary = run_campaign(
        CampaignArgs(
            action="run",
            manifest=str(manifest_path),
            output_dir=str(results),
            executor="slurm",
            fn=ARRAY_UNIT_FN,
            row_index=5,
        )
    )
    script = (results / "submit_campaign.sh").read_text(encoding="utf-8")

    assert f"MANIFEST_PATH={manifest_path.resolve()}" in script
    assert f"OUTPUT_DIR={results.resolve()}" in script
    assert f"UNIT_FN={ARRAY_UNIT_FN}" in script
    assert "--executor local-seq" in script
    assert "#SBATCH --array=5\n" in script  # --row-index 5 schedules only row 5
    assert summary.total_units == 1
    assert summary.submitted_units == 1
    assert summary.completed_units == 0
    res = subprocess.run(["bash", "-n", "-c", script], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_slurm_rejects_row_index_beyond_max_array_size(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=6)
    executor = SlurmArrayExecutor(
        manifest_path=tmp_path / "m.json", fn_spec=ARRAY_UNIT_FN, max_array_size=4
    )
    with pytest.raises(ValueError, match="MaxArraySize"):
        executor.generate_script(manifest, tmp_path / "out")


def test_slurm_run_requires_manifest_path_and_fn(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest_path"):
        SlurmArrayExecutor().run(_manifest(1), array_unit, output_dir=str(tmp_path))


# ==============================================================================
# 3. A unit whose lease is held elsewhere is not counted as completed
# ==============================================================================


def _hold_lease(out: Path, row: ManifestRow) -> FileLock:
    lock = FileLock(
        output_path=_resolve_output_path(out, row),
        owner_token="other-worker",
        manifest_row_hash=row.manifest_row_hash,
        lease_seconds=3600,
        heartbeat_interval=3600,
    )
    assert lock.acquire()
    return lock


def test_lock_held_unit_counted_as_locked_sequential(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=2)
    lock = _hold_lease(tmp_path, manifest.rows[0])
    try:
        summary = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    finally:
        lock.release()
    assert summary.completed_units == 1
    assert summary.locked_units == 1
    assert summary.accounted_units() == summary.total_units


def test_lock_held_unit_counted_as_locked_pool(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=2)
    lock = _hold_lease(tmp_path, manifest.rows[0])
    try:
        summary = LocalProcessPoolExecutor(max_workers=1).run(
            manifest, array_unit, output_dir=str(tmp_path)
        )
    finally:
        lock.release()
    assert summary.completed_units == 1
    assert summary.locked_units == 1
    assert summary.accounted_units() == summary.total_units


def test_pool_all_done_reports_zero_completed(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=2)
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    summary = LocalProcessPoolExecutor(max_workers=1).run(
        manifest, array_unit, output_dir=str(tmp_path)
    )
    assert summary.completed_units == 0
    assert summary.skipped_units == 2
    assert summary.accounted_units() == summary.total_units


# ==============================================================================
# 4. Stale commit tokens and orphaned promoted outputs must not wedge a unit
# ==============================================================================


def test_stale_commit_token_from_dead_holder_is_cleared(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=1)
    store = _resolve_output_path(tmp_path, manifest.rows[0])
    store.parent.mkdir(parents=True, exist_ok=True)
    store.with_name(f"{store.name}.commit").write_text(
        json.dumps({"attempt_uuid": "dead", "owner_token": "preempted-worker"})
    )

    summary = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))

    assert summary.completed_units == 1, summary.failures
    assert not store.with_name(f"{store.name}.commit").exists()


def test_crash_after_promotion_does_not_wedge_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(num_rows=1)
    row = manifest.rows[0]
    store = _resolve_output_path(tmp_path, row)

    from xtrax.campaign import lease as lease_mod

    real_fsync = lease_mod.fsync_file

    def crash_on_done_tmp(path: Path) -> None:
        if ".done.json.tmp." in path.name:
            raise OSError(28, "No space left on device")
        real_fsync(path)

    monkeypatch.setattr(lease_mod, "fsync_file", crash_on_done_tmp)
    first = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    assert first.failed_units == 1
    assert store.exists() and not _marker_path(store).exists()  # promoted, unstamped
    assert not store.with_name(f"{store.name}.commit").exists()

    monkeypatch.setattr(lease_mod, "fsync_file", real_fsync)
    second = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    assert second.completed_units == 1, second.failures
    assert validate_unit_done(store, row, deep_verify=True, quarantine_on_failure=False).is_valid


# ==============================================================================
# 5. zarr v3 stores pass the Tier-1 check
# ==============================================================================


def test_zarr_v3_store_passes_tier1(tmp_path: Path) -> None:
    row = _manifest(num_rows=1).rows[0]
    store = tmp_path / "v3.zarr"
    zarr.open_group(str(store), mode="w", zarr_format=3).create_array("x", data=np.ones(4))
    create_done_marker(
        manifest_row_hash=row.manifest_row_hash,
        input_hash=row.input_hash,
        output_locator=str(store),
        content_digest_sha256=compute_content_digest(store),
        duration_s=0.1,
    ).write(_marker_path(store))

    val = validate_unit_done(store, row, deep_verify=False, quarantine_on_failure=True)
    assert val.is_valid, val.error_message
    assert store.exists()  # not quarantined


# ==============================================================================
# 6. Resume trusts only a validated store, never a bare marker
# ==============================================================================


def test_deleted_store_is_rescheduled_despite_marker(tmp_path: Path) -> None:
    import shutil

    manifest = _manifest(num_rows=2)
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    row = manifest.rows[0]
    store = _resolve_output_path(tmp_path, row)
    shutil.rmtree(store)  # user deletes a bad store to force a rerun; marker remains

    assert not _check_resume_done(tmp_path, row)
    assert get_pending_row_indices(manifest, tmp_path) == [0]
    rerun = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    assert rerun.completed_units == 1
    assert rerun.skipped_units == 1


def test_no_top_level_hash_marker_written(tmp_path: Path) -> None:
    manifest = _manifest(num_rows=1)
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(tmp_path))
    row = manifest.rows[0]
    assert not (tmp_path / f"{row.manifest_row_hash}.done.json").exists()


# ==============================================================================
# 7. Unparseable locks and orphaned steal tokens are reclaimable
# ==============================================================================


def test_old_unparseable_lock_is_stolen(tmp_path: Path) -> None:
    output = tmp_path / "u.zarr"
    lock_path = output.with_name("u.zarr.lock")
    lock_path.write_text("")  # writer died between O_EXCL and write
    _backdate(lock_path, 60 + CLOCK_DRIFT_GRACE_PERIOD_S + 1)

    lock = FileLock(output, owner_token="me", manifest_row_hash="h", lease_seconds=60)
    try:
        assert lock.acquire() is True
        assert json.loads(lock_path.read_text())["owner_token"] == "me"
    finally:
        lock.release()


def test_orphaned_steal_token_is_reclaimed(tmp_path: Path) -> None:
    output = tmp_path / "u.zarr"
    lock_path = output.with_name("u.zarr.lock")
    lock_path.write_text(
        json.dumps(
            {
                "owner_token": "dead",
                "manifest_row_hash": "h",
                "acquired_at_unix_s": 1.0,
                "lease_expires_at_unix_s": 2.0,
                "lease_seconds": 60,
            }
        )
    )
    _backdate(lock_path, 60 + CLOCK_DRIFT_GRACE_PERIOD_S + 1)
    steal = lock_path.with_name("u.zarr.lock.steal")
    steal.write_text("")
    _backdate(steal, CLOCK_DRIFT_GRACE_PERIOD_S + 1)

    lock = FileLock(output, owner_token="me", manifest_row_hash="h", lease_seconds=60)
    try:
        assert lock.acquire() is True
    finally:
        lock.release()
    assert not steal.exists()


def test_fresh_steal_token_still_blocks(tmp_path: Path) -> None:
    output = tmp_path / "u.zarr"
    lock_path = output.with_name("u.zarr.lock")
    lock_path.write_text("")
    _backdate(lock_path, 60 + CLOCK_DRIFT_GRACE_PERIOD_S + 1)
    lock_path.with_name("u.zarr.lock.steal").write_text("")  # a live stealer

    lock = FileLock(output, owner_token="me", manifest_row_hash="h", lease_seconds=60)
    with pytest.raises(LockUnavailableError, match="Lost CAS race"):
        lock.acquire()


# ==============================================================================
# 8. The shared canonical hash keeps its version-1 byte layout
# ==============================================================================


def test_shared_digest_keeps_negative_zero(tmp_path: Path) -> None:
    import math

    from xtrax.run.zarr_integrity import (
        DIGEST_ALGO_VERSION,
        normalize_json_value,
        zarr_content_digest,
    )

    # Digest version 1 hashes an attr of -0.0 distinctly from 0.0. Changing that changes
    # every recorded digest for such stores and needs a version bump; the campaign planner
    # normalizes -0.0 in its own helper (xtrax.campaign.manifest) instead.
    assert DIGEST_ALGO_VERSION == 1
    assert math.copysign(1.0, normalize_json_value(-0.0)) == -1.0
    digests = []
    for name, bias in (("neg", -0.0), ("pos", 0.0)):
        group = zarr.open_group(str(tmp_path / f"{name}.zarr"), mode="w")
        group.attrs["bias"] = bias
        digests.append(zarr_content_digest(tmp_path / f"{name}.zarr"))
    assert digests[0] != digests[1]


# ==============================================================================
# 9. A directory zarr cannot read is an error, not a different hash
# ==============================================================================


def test_digest_of_non_zarr_directory_raises(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "a.txt").write_text("x")
    with pytest.raises(Exception, match="(?i)group|zarr|node"):
        compute_content_digest(plain)


# ==============================================================================
# 10. Spec TOML round-trips keys that are not bare
# ==============================================================================


def test_spec_toml_roundtrips_dotted_and_spaced_keys() -> None:
    spec = CampaignSpec(
        campaign_id="c",
        target_inputs=["t.pdb"],
        arms=[ArmSpec(arm_id="a", backend="b", checkpoint_id="v", parameters={"x.y": 1})],
        grid=ParamGrid(axes={"optimizer.lr": [0.1, 0.01], "has space": [1, 2]}),
        metadata={"owner.name": "m"},
    )
    back = CampaignSpec.from_toml(spec.to_toml())
    assert back.grid.axes == spec.grid.axes
    assert back.arms[0].parameters == {"x.y": 1}
    assert back.metadata == {"owner.name": "m"}


# ==============================================================================
# Smaller findings
# ==============================================================================


def test_quarantine_same_unit_twice_within_a_second(tmp_path: Path) -> None:
    store = tmp_path / "s.zarr"
    for _ in range(2):
        store.mkdir()
        (store / "zarr.json").write_text("{}")
        quarantine_output(store, reason="corrupt")
    assert len(list((tmp_path / ".quarantine").iterdir())) == 2


def test_worker_init_has_no_default_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    with patch("jax.config.update") as update:
        _initialize_worker_process(None)
    update.assert_not_called()
    assert not (tmp_path / ".jax_compilation_cache").exists()


def test_verify_and_status_find_nested_units_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    manifest = _manifest(num_rows=3)
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    assert _resolve_output_path(out, manifest.rows[0]).parent != out  # nested locator

    verified = run_campaign(CampaignArgs(action="verify", output_dir=str(out), deep=True))
    assert verified == {"passed": 3, "failed": 0}
    status = run_campaign(CampaignArgs(action="status", output_dir=str(out)))
    assert status["completed"] == 3
