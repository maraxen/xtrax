"""Regression tests for the PR #267 review (Grok, praxia surface_dispatch, 261106).

Each test failed on 88b5aad, before the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.campaign.executors.test_local import _create_mock_manifest
from tests.campaign.test_r3_local import _sigkill_unit_fn
from tests.campaign.unit_fns import array_unit
from xtrax.campaign.executors import local as local_mod
from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    _check_resume_done,
    _reclaim_dead_worker_leases,
    _resolve_output_path,
)
from xtrax.campaign.marker import validate_unit_done
from xtrax.cli.campaign_verb import CampaignArgs, _run_status


def _eio(_path: Path) -> str:
    raise OSError(5, "Input/output error (simulated)")


# ==============================================================================
# 1. An I/O error during the deep digest must not quarantine a good store
# ==============================================================================


def test_deep_verify_io_error_raises_instead_of_reporting_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_mock_manifest(num_rows=1, campaign_id="camp_pr267_eio")
    row = manifest.rows[0]
    out = tmp_path / "camp_pr267_eio"
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    path = _resolve_output_path(out, row)

    import xtrax.campaign.marker as marker_mod

    monkeypatch.setattr(marker_mod, "compute_content_digest", _eio)
    with pytest.raises(OSError, match="simulated"):
        validate_unit_done(path, row, deep_verify=True, quarantine_on_failure=True)
    assert path.exists(), "store moved on an I/O error"
    assert not (out / ".quarantine").exists()

    # The resume pre-check does not reuse a unit it could not verify
    assert _check_resume_done(out, row, resume_verify="deep") is False
    with pytest.raises(OSError, match="simulated"):
        local_mod._set_aside_orphaned_output(path, row, resume_verify="deep")
    assert path.exists()


def test_run_with_io_error_fails_the_unit_and_keeps_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _create_mock_manifest(num_rows=1, campaign_id="camp_pr267_eio_run")
    row = manifest.rows[0]
    out = tmp_path / "camp_pr267_eio_run"
    LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    path = _resolve_output_path(out, row)

    import xtrax.campaign.marker as marker_mod

    monkeypatch.setattr(marker_mod, "compute_content_digest", _eio)
    summary = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(out))
    assert summary.failed_units == 1
    assert "OSError" in summary.failures[0]["error"]
    assert path.exists()
    assert not (out / ".quarantine").exists()
    assert not path.with_name(f"{path.name}.lock").exists()

    monkeypatch.undo()
    assert validate_unit_done(path, row, deep_verify=True).is_valid


# ==============================================================================
# 2. status counts each store once and does not hide an empty-hash marker
# ==============================================================================


def test_status_counts_corrupt_marker_with_expired_lock_once(tmp_path: Path) -> None:
    out = tmp_path / "camp_pr267_status"
    out.mkdir()
    (out / "u.zarr").mkdir()
    (out / "u.zarr.done.json").write_text("{not json", encoding="utf-8")
    (out / "u.zarr.lock").write_text("{not json either", encoding="utf-8")
    res = _run_status(CampaignArgs(action="status", output_dir=str(out)))
    assert res["failed"] == 1


def test_status_reports_empty_hash_marker_as_failed(tmp_path: Path) -> None:
    out = tmp_path / "camp_pr267_empty_hash"
    out.mkdir()
    (out / "e.zarr").mkdir()
    (out / "e.zarr.done.json").write_text(json.dumps({"manifest_row_hash": ""}), encoding="utf-8")
    res = _run_status(CampaignArgs(action="status", output_dir=str(out)))
    assert res["failed"] == 1
    assert res["completed"] == 0


# ==============================================================================
# 3. A SIGKILLed pool worker's lease is reclaimed by the parent
# ==============================================================================


def test_pool_reclaims_lease_of_sigkilled_worker(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=2, campaign_id="camp_pr267_sigkill")
    out = tmp_path / "camp_pr267_sigkill"
    first = LocalProcessPoolExecutor(max_workers=1).run(
        manifest, _sigkill_unit_fn, output_dir=str(out)
    )
    assert first.failed_units >= 1
    assert not list(out.rglob("*.lock")), "a killed worker's lease was left behind"
    assert not list(out.rglob("*.partial.*"))

    # A rerun redoes the unit at once instead of reporting it locked for lease + 300 s.
    second = LocalProcessPoolExecutor(max_workers=1).run(manifest, array_unit, output_dir=str(out))
    assert second.locked_units == 0
    assert second.failed_units == 0
    assert second.completed_units + second.skipped_units == 2


def test_reclaim_leaves_other_owners_leases_alone(tmp_path: Path) -> None:
    manifest = _create_mock_manifest(num_rows=2, campaign_id="camp_pr267_foreign")
    out = tmp_path / "camp_pr267_foreign"
    out.mkdir()
    ours, theirs = manifest.rows
    for row, owner in ((ours, "pool:111.abcd1234:5:x"), (theirs, "pool:111.ffff0000:5:y")):
        path = _resolve_output_path(out, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(f"{path.name}.lock").write_text(
            json.dumps({"owner_token": owner}), encoding="utf-8"
        )
    removed = _reclaim_dead_worker_leases(out, list(manifest.rows), "pool:111.abcd1234")
    ours_lock = _resolve_output_path(out, ours)
    theirs_lock = _resolve_output_path(out, theirs)
    assert removed == [ours_lock.with_name(f"{ours_lock.name}.lock")]
    assert theirs_lock.with_name(f"{theirs_lock.name}.lock").exists()
