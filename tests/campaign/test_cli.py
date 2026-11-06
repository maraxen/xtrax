"""Tests for CLI campaign verbs (Track 4: T4-01).

Covering xtrax campaign plan, run, status, verify, approval gate (T2-32),
and registry registration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.campaign.unit_fns import ARRAY_UNIT_FN
from xtrax.campaign.lease import CampaignLock
from xtrax.campaign.marker import create_done_marker
from xtrax.campaign.spec import ArmSpec, CampaignSpec, ParamGrid
from xtrax.cli.errors import CLIError


def test_tyro_isolation() -> None:
    """Importing campaign_verb must not pull in tyro."""
    sys.modules.pop("tyro", None)
    import xtrax.cli.campaign_verb  # noqa: F401

    assert "tyro" not in sys.modules, "tyro was imported at module level in campaign_verb"


def test_campaign_registry_wiring() -> None:
    """Verify that 'campaign' verb is registered in REGISTRY."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign
    from xtrax.cli.registry import REGISTRY

    assert "campaign" in REGISTRY
    args_cls, run_fn = REGISTRY["campaign"]
    assert args_cls is CampaignArgs
    assert run_fn is run_campaign


def _write_spec_file(
    path: Path,
    campaign_id: str = "test_camp",
    output_dir: str = "outputs",
) -> Path:
    spec = CampaignSpec(
        campaign_id=campaign_id,
        target_inputs=["protA.pdb", "protB.pdb"],
        arms=[
            ArmSpec(arm_id="boltz", backend="boltz2", checkpoint_id="chk1"),
            ArmSpec(arm_id="chai", backend="chai1", checkpoint_id="chk2"),
        ],
        grid=ParamGrid(axes={"temp": [0.1, 0.5]}),
        output_dir=output_dir,
    )
    spec.to_toml(path)
    return path


def test_campaign_plan_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign plan': takes campaign_spec.toml and emits manifest.json."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    spec_path = _write_spec_file(tmp_path / "spec.toml", output_dir=str(tmp_path / "out"))
    manifest_path = tmp_path / "custom_manifest.json"

    args = CampaignArgs(
        action="plan",
        spec_path=str(spec_path),
        output=str(manifest_path),
    )
    run_campaign(args)
    assert manifest_path.exists()

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == "campaign_manifest_v1"
    assert manifest_data["metadata"]["campaign_id"] == "test_camp"
    # 2 targets * 2 arms * 2 temp = 8 rows
    assert len(manifest_data["rows"]) == 8


def test_campaign_plan_default_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign plan' emits to <output_dir>/manifest.json by default."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "planned_out"
    spec_path = _write_spec_file(tmp_path / "spec.toml", output_dir=str(out_dir))

    args = CampaignArgs(
        action="plan",
        spec_path=str(spec_path),
    )
    run_campaign(args)

    default_manifest = out_dir / "manifest.json"
    assert default_manifest.exists()


def test_campaign_plan_invalid_spec_raises(tmp_path: Path) -> None:
    """Test 'xtrax campaign plan' raises CLIError on invalid or missing spec."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    # Missing file
    args = CampaignArgs(action="plan", spec_path=str(tmp_path / "nonexistent.toml"))
    with pytest.raises(CLIError):
        run_campaign(args)

    # Malformed spec
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("invalid = [toml", encoding="utf-8")
    args_bad = CampaignArgs(action="plan", spec_path=str(bad_toml))
    with pytest.raises(CLIError):
        run_campaign(args_bad)


def test_campaign_run_local_seq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign run' with local-seq executor."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "run_out"
    spec_path = _write_spec_file(tmp_path / "spec.toml", output_dir=str(out_dir))
    manifest_path = tmp_path / "manifest.json"

    # Plan
    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path), output=str(manifest_path)))

    # Run
    summary = run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            executor="local-seq",
        )
    )

    assert summary.total_units == 8
    assert summary.completed_units == 8
    assert summary.failed_units == 0


def test_campaign_run_single_row_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign run' with --row-index executing a single unit."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "single_out"
    spec_path = _write_spec_file(tmp_path / "spec.toml", output_dir=str(out_dir))
    manifest_path = tmp_path / "manifest.json"

    # Plan
    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path), output=str(manifest_path)))

    # Run only row index 2
    summary = run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            row_index=2,
            executor="local-seq",
        )
    )

    assert summary.total_units == 1
    assert summary.completed_units == 1


def test_campaign_run_local_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign run' with local-pool executor."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "pool_out"
    spec_path = _write_spec_file(tmp_path / "spec.toml", output_dir=str(out_dir))
    manifest_path = tmp_path / "manifest.json"

    # Plan
    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path), output=str(manifest_path)))

    # Run via local-pool
    summary = run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            executor="local-pool",
        )
    )

    assert summary.total_units == 8
    assert summary.completed_units == 8


def test_campaign_run_slurm_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign run' with slurm executor generates submission script."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "slurm_out"
    spec_path = _write_spec_file(tmp_path / "spec.toml", output_dir=str(out_dir))
    manifest_path = tmp_path / "manifest.json"

    # Plan
    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path), output=str(manifest_path)))

    # Run slurm
    run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            output_dir=str(out_dir),
            executor="slurm",
        )
    )

    submit_script = out_dir / "submit_campaign.sh"
    assert submit_script.exists()
    assert "#SBATCH --array=" in submit_script.read_text(encoding="utf-8")


def test_campaign_run_approval_gate_enforcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-12: Test approval gate enforcement when --require-approval is set."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign
    from xtrax.loop.campaign_approval_gate import CampaignNotApprovedError

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "gate_out"
    spec_path = _write_spec_file(
        tmp_path / "spec.toml",
        campaign_id="unapproved_camp",
        output_dir=str(out_dir),
    )
    manifest_path = tmp_path / "manifest.json"

    run_campaign(CampaignArgs(action="plan", spec_path=str(spec_path), output=str(manifest_path)))

    # Negative control: missing attestation raises CampaignNotApprovedError or CLIError
    with pytest.raises((CampaignNotApprovedError, CLIError)):
        run_campaign(
            CampaignArgs(
                action="run",
                fn=ARRAY_UNIT_FN,
                manifest=str(manifest_path),
                require_approval=True,
            )
        )

    # Positive control: provide valid fresh attestation in gates file
    gates_file = tmp_path / ".praxia" / "loop_human_gates.toml"
    gates_file.parent.mkdir(parents=True, exist_ok=True)
    gates_file.write_text(
        """
[[gates]]
id = "T2-32"
event_ref = "unapproved_camp"
attested_at = "2026-11-05T00:00:00Z"
ttl_days = 365.0
attested_by = "marielle"
note = "approved for test"
""",
        encoding="utf-8",
    )

    summary = run_campaign(
        CampaignArgs(
            action="run",
            fn=ARRAY_UNIT_FN,
            manifest=str(manifest_path),
            require_approval=True,
        )
    )
    assert summary.completed_units == 8


def test_campaign_status_reporting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """Test 'xtrax campaign status' reports completed, running, failed, quarantined units."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "status_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Completed unit: create .done.json
    marker = create_done_marker(
        manifest_row_hash="hash_done_1",
        input_hash="input_done_1",
        output_locator=str(out_dir / "hash_done_1.zarr"),
        content_digest_sha256="abc",
        duration_s=1.0,
    )
    marker.write(out_dir / "hash_done_1.done.json")

    # 2. Running unit: active lock file
    import time

    lock = CampaignLock(
        schema_version="campaign_lock_v1",
        manifest_row_hash="hash_running_1",
        owner_token="worker:123",
        acquired_at_unix_s=time.time(),
        lease_expires_at_unix_s=time.time() + 1800,
        lease_seconds=1800,
        heartbeat_count=0,
    )
    (out_dir / "hash_running_1.lock").write_text(lock.to_json(), encoding="utf-8")

    # 3. Quarantined unit in .quarantine/
    quarantine_dir = out_dir / ".quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    (quarantine_dir / "corrupt_unit.zarr").mkdir(parents=True, exist_ok=True)

    status = run_campaign(CampaignArgs(action="status", output_dir=str(out_dir)))

    assert status["completed"] >= 1
    assert status["running"] >= 1
    assert status["quarantined"] >= 1

    captured = capsys.readouterr().out
    assert "Completed:" in captured and "1" in captured
    assert "Running:" in captured and "1" in captured
    assert "Quarantined:" in captured and "1" in captured


def test_campaign_verify_tier1_and_tier2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'xtrax campaign verify' running Tier-1 and Tier-2 deep audits."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "verify_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup valid unit
    unit_path = out_dir / "unit_001.zarr"
    import numpy as np
    import zarr

    zarr.open_group(str(unit_path), mode="w").create_array(
        "x", data=np.arange(16, dtype=np.float32), chunks=(16,)
    )
    from xtrax.campaign.marker import compute_content_digest

    digest = compute_content_digest(unit_path)

    marker = create_done_marker(
        manifest_row_hash="row_hash_001",
        input_hash="input_hash_001",
        output_locator=str(unit_path),
        content_digest_sha256=digest,
        duration_s=1.5,
    )
    marker.write(out_dir / "unit_001.zarr.done.json")

    # Tier-1 verify should pass
    res_tier1 = run_campaign(CampaignArgs(action="verify", output_dir=str(out_dir), deep=False))
    assert res_tier1["passed"] == 1
    assert res_tier1["failed"] == 0

    # Tier-2 deep verify should pass
    res_tier2 = run_campaign(CampaignArgs(action="verify", output_dir=str(out_dir), deep=True))
    assert res_tier2["passed"] == 1
    assert res_tier2["failed"] == 0

    # Tamper with the array's data. (A stray non-zarr file is not array content, so the
    # logical digest ignores it by design; the corruption must change what zarr reads.)
    zarr.open_group(str(unit_path), mode="r+")["x"][0] = 999.0

    # Tier-2 deep verify should now detect failure
    with pytest.raises(CLIError):
        run_campaign(CampaignArgs(action="verify", output_dir=str(out_dir), deep=True))


def test_campaign_unknown_action_raises() -> None:
    """Test unknown action raises CLIError."""
    from xtrax.cli.campaign_verb import CampaignArgs, run_campaign

    args = CampaignArgs(action="unknown_verb")
    with pytest.raises(CLIError):
        run_campaign(args)
