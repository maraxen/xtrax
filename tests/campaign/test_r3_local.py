"""Regression tests for Round 3 review findings in xtrax.campaign.executors.local."""

from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import zarr

from tests.campaign.executors.test_local import _create_mock_manifest
from tests.campaign.unit_fns import array_unit
from xtrax.campaign.executors.local import (
    UNIT_DISCARDED,
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    UnitInterruptedError,
    _execute_single_unit,
    _resolve_output_path,
)
from xtrax.campaign.manifest import ManifestRow
from xtrax.campaign.marker import (
    compute_content_digest,
    create_done_marker,
    validate_unit_done,
)


def _sigkill_unit_fn(row: ManifestRow) -> dict[str, Any]:
    """Module-level unit function that kills its own worker process with SIGKILL on chunk 0."""
    if row.chunk_id == 0:
        os.kill(os.getpid(), signal.SIGKILL)
    return {"values": np.ones(4, dtype=np.float32) * float(row.seed)}


def _always_fail_unit_fn(row: ManifestRow) -> dict[str, Any]:
    """Module-level unit function that always raises RuntimeError."""
    raise RuntimeError(f"boom on chunk {row.chunk_id}")


# ==============================================================================
# Finding 1: _resolve_output_path resolves output_dir before comparing .name
# ==============================================================================


def test_r3_1_resolve_output_path_dot_matches_absolute_campaign_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running with output_dir='.' inside the campaign dir resolves the same store path."""
    campaign_id = "camp_r3_dot"
    campaign_dir = tmp_path / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    manifest = _create_mock_manifest(num_rows=2, campaign_id=campaign_id)
    row = manifest.rows[0]

    monkeypatch.chdir(campaign_dir)

    dot_resolved = _resolve_output_path(Path("."), row)
    abs_resolved = _resolve_output_path(campaign_dir, row)
    assert dot_resolved == abs_resolved
    assert dot_resolved == campaign_dir.resolve() / "unit_000.zarr"

    # Running with output_dir='.' then resuming with absolute campaign_dir must reuse stores
    s1 = LocalSequentialExecutor().run(manifest, array_unit, output_dir=".")
    assert s1.completed_units == 2
    assert (campaign_dir / "unit_000.zarr").exists()
    assert not (campaign_dir / campaign_id / "unit_000.zarr").exists()

    s2 = LocalSequentialExecutor().run(manifest, array_unit, output_dir=str(campaign_dir))
    assert s2.completed_units == 0
    assert s2.skipped_units == 2


# ==============================================================================
# Finding 2: Commit between _check_resume_done and _set_aside_orphaned_output
# ==============================================================================


def test_r3_2_concurrent_commit_before_orphan_check_is_reused_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If another worker commits between _check_resume_done and _set_aside_orphaned_output,
    _execute_single_unit must record reuse (reused_by='lease_recheck'), return UNIT_DISCARDED
    without calling unit_fn, and never move or delete the committed store.
    """
    import xtrax.campaign.executors.local as local_mod

    manifest = _create_mock_manifest(num_rows=1, campaign_id="camp_r3_race")
    row = manifest.rows[0]
    output_dir = tmp_path / "outputs"
    store_path = _resolve_output_path(output_dir, row)
    marker_path = store_path.with_name(f"{store_path.name}.done.json")

    def _simulate_other_worker_commit_after_first_check(
        out_dir: Path, r: ManifestRow, resume_verify: Any = "deep"
    ) -> bool:
        # Simulate that the initial _check_resume_done saw False, and immediately
        # afterward another worker finished committing the valid store + done marker.
        store_path.parent.mkdir(parents=True, exist_ok=True)
        grp = zarr.open_group(str(store_path), mode="w", zarr_format=3)
        grp.create_array("committed_by_peer", data=np.arange(6, dtype=np.float32))
        digest = compute_content_digest(store_path)
        create_done_marker(
            manifest_row_hash=r.manifest_row_hash,
            input_hash=r.input_hash,
            output_locator=str(store_path),
            content_digest_sha256=digest,
            duration_s=0.05,
        ).write(marker_path)
        return False

    monkeypatch.setattr(
        local_mod, "_check_resume_done", _simulate_other_worker_commit_after_first_check
    )

    unit_spy = MagicMock(side_effect=lambda r: {"values": np.ones(4, dtype=np.float32)})
    outcome = _execute_single_unit(row=row, unit_fn=unit_spy, output_dir=output_dir)

    assert outcome == UNIT_DISCARDED
    assert unit_spy.call_count == 0, "unit_fn must not run when unit validated as done"
    assert store_path.exists(), "Validly committed store was moved or deleted!"
    assert marker_path.exists()
    val = validate_unit_done(store_path, row, deep_verify=True, quarantine_on_failure=False)
    assert val.is_valid, val.error_message

    reuse_log = output_dir / "reuse_log.jsonl"
    assert reuse_log.exists()
    records = [json.loads(line) for line in reuse_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["manifest_row_hash"] == row.manifest_row_hash
    assert records[0]["reused_by"] == "lease_recheck"


# ==============================================================================
# Finding 3: Aborted unit_fn writing directly at output_path is quarantined
# ==============================================================================


def test_r3_3_aborted_direct_output_path_store_is_quarantined(tmp_path: Path) -> None:
    """If unit_fn writes directly to output_path and raises (or UnitInterruptedError arrives)
    before _stage_unit_output, the half-written store is quarantined with reason
    'aborted_unit_output' while the lease is held.
    """
    manifest = _create_mock_manifest(num_rows=2, campaign_id="camp_r3_abort")
    output_dir = tmp_path / "outputs"
    row0, row1 = manifest.rows[0], manifest.rows[1]
    store0 = _resolve_output_path(output_dir, row0)
    store1 = _resolve_output_path(output_dir, row1)

    def _direct_write_then_raise(r: ManifestRow) -> dict[str, Any]:
        target = _resolve_output_path(output_dir, r)
        grp = zarr.open_group(str(target), mode="w", zarr_format=3)
        grp.attrs["half_written"] = True
        raise RuntimeError("crashed after writing half a store at output_path")

    summary = LocalSequentialExecutor().run(
        manifest, _direct_write_then_raise, output_dir=str(output_dir), row_indices=[0]
    )
    assert summary.failed_units == 1
    assert not store0.exists(), "Half-written store was left at output_path"

    quarantine_root = store0.parent / ".quarantine"
    assert quarantine_root.is_dir(), "Expected .quarantine directory to be created"
    q_dirs = list(quarantine_root.iterdir())
    assert len(q_dirs) == 1
    assert "aborted_unit_output" in q_dirs[0].name
    q_manifest = json.loads((q_dirs[0] / "quarantine_manifest.json").read_text(encoding="utf-8"))
    assert q_manifest["reason"] == "aborted_unit_output"
    assert (q_dirs[0] / store0.name).exists()

    # Also verify UnitInterruptedError (BaseException from SIGTERM) quarantines output_path
    def _direct_write_then_sigterm(r: ManifestRow) -> dict[str, Any]:
        target = _resolve_output_path(output_dir, r)
        grp = zarr.open_group(str(target), mode="w", zarr_format=3)
        grp.attrs["interrupted"] = True
        raise UnitInterruptedError("unit run interrupted by signal 15")

    with pytest.raises(UnitInterruptedError):
        _execute_single_unit(row=row1, unit_fn=_direct_write_then_sigterm, output_dir=output_dir)

    assert not store1.exists(), "Interrupted store was left at output_path"
    assert not store1.with_name(f"{store1.name}.lock").exists()
    q_dirs_after = list(quarantine_root.iterdir())
    assert len(q_dirs_after) == 2


# ==============================================================================
# Finding 4: LocalProcessPoolExecutor survives SIGKILL (BrokenProcessPool)
# ==============================================================================


def test_r3_4_process_pool_sigkill_returns_broken_pool_failure_without_hanging(
    tmp_path: Path,
) -> None:
    """A worker killed by SIGKILL must not hang LocalProcessPoolExecutor.run(); unfinished
    rows are recorded as failed units naming BrokenProcessPool within ~30s.
    """
    manifest = _create_mock_manifest(num_rows=2, campaign_id="camp_r3_sigkill")
    output_dir = tmp_path / "outputs"
    executor = LocalProcessPoolExecutor(max_workers=1, recycle_interval=50)

    result_holder: dict[str, Any] = {}

    def _target() -> None:
        try:
            result_holder["summary"] = executor.run(
                manifest, _sigkill_unit_fn, output_dir=str(output_dir)
            )
        except BaseException as exc:  # pragma: no cover
            result_holder["exc"] = exc

    runner_thread = threading.Thread(target=_target, daemon=True)
    runner_thread.start()
    runner_thread.join(timeout=20.0)

    assert not runner_thread.is_alive(), "LocalProcessPoolExecutor.run() hung after worker SIGKILL"
    assert "exc" not in result_holder, f"run() raised unexpectedly: {result_holder.get('exc')!r}"

    summary = result_holder["summary"]
    assert summary.failed_units >= 1
    assert summary.accounted_units() == summary.total_units
    assert any("BrokenProcessPool" in f["error"] for f in summary.failures), summary.failures
    failed_indices = {f["row_index"] for f in summary.failures}
    assert 0 in failed_indices

    # Verify max_failures early-exit behavior cancels pending futures and shuts down
    mf_manifest = _create_mock_manifest(num_rows=4, campaign_id="camp_r3_max_failures")
    mf_summary = LocalProcessPoolExecutor(max_workers=1).run(
        mf_manifest,
        _always_fail_unit_fn,
        output_dir=str(tmp_path / "mf_outputs"),
        max_failures=1,
    )
    assert mf_summary.failed_units == 1
    assert len(mf_summary.failures) == 1
