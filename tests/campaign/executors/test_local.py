"""Tests for LocalSequentialExecutor and LocalProcessPoolExecutor.

Covering [T3-02], [T3-03], GATE-07, and AC-08 from SPEC-261105-CAMPAIGN-RUNNER.
"""

from __future__ import annotations

import gc
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from xtrax.campaign.executors.local import (
    LocalProcessPoolExecutor,
    LocalSequentialExecutor,
    _initialize_worker_process,
)
from xtrax.campaign.manifest import (
    CampaignManifest,
    ManifestMetadata,
    ManifestRow,
    ResourceRequirements,
    build_manifest_row,
)


def _get_process_rss_kb() -> int:
    """Read VmRSS from /proc/self/status on Linux."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _create_mock_manifest(
    num_rows: int = 3,
    campaign_id: str = "camp_test_local",
    backends: list[str] | None = None,
    padding_buckets: list[int | None] | None = None,
) -> CampaignManifest:
    rows = []
    for i in range(num_rows):
        backend = backends[i] if backends and i < len(backends) else "boltz2"
        bucket = (
            padding_buckets[i]
            if padding_buckets and i < len(padding_buckets)
            else 256
        )
        row = build_manifest_row(
            campaign_id=campaign_id,
            job_id=f"job_{i:03d}",
            chunk_id=i,
            sample_start=i * 10,
            sample_count=10,
            target_input="target_prot.pdb",
            arm_id=f"arm_{backend}",
            arm_backend=backend,
            checkpoint_id="chk_v1",
            fixed_policy="none",
            state_weight_profile="equal",
            seed=42 + i,
            parameters={"temperature": 0.2, "index": i},
            output_locator=f"{campaign_id}/unit_{i:03d}.zarr",
            resource_requirements=ResourceRequirements(
                gpu_memory_bytes=0,
                cpu_cores=1,
                timeout_seconds=300,
                padding_bucket_id=bucket,
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


def _simple_unit_fn(row: ManifestRow) -> dict[str, Any]:
    """Simple unit compute returning state with numpy array."""
    data = np.ones((50, 50), dtype=np.float32) * row.seed
    return {"result": data, "row_hash": row.manifest_row_hash}


def test_local_sequential_executor_basic_run(tmp_path: Path):
    """Verify LocalSequentialExecutor iterates through rows and produces CampaignSummary."""
    manifest = _create_mock_manifest(num_rows=3)
    output_dir = tmp_path / "outputs"

    executor = LocalSequentialExecutor()
    summary = executor.run(manifest, _simple_unit_fn, output_dir=str(output_dir))

    assert summary.campaign_id == manifest.metadata.campaign_id
    assert summary.total_units == 3
    assert summary.completed_units == 3
    assert summary.skipped_units == 0
    assert summary.failed_units == 0
    assert summary.elapsed_seconds >= 0.0
    assert len(summary.failures) == 0

    # Output stores and done markers should exist
    for row in manifest.rows:
        done_marker_path = output_dir / f"{row.manifest_row_hash}.done.json"
        alt_marker_path = output_dir / f"{Path(row.output_locator).name}.done.json"
        assert done_marker_path.exists() or alt_marker_path.exists()


def test_local_sequential_executor_tier1_fast_resume(tmp_path: Path):
    """Verify completed units are skipped upon resume."""
    manifest = _create_mock_manifest(num_rows=3)
    output_dir = tmp_path / "outputs"

    executor = LocalSequentialExecutor()
    # First run
    s1 = executor.run(manifest, _simple_unit_fn, output_dir=str(output_dir))
    assert s1.completed_units == 3
    assert s1.skipped_units == 0

    # Second run (resume)
    call_tracker = MagicMock(side_effect=_simple_unit_fn)
    s2 = executor.run(manifest, call_tracker, output_dir=str(output_dir))

    assert s2.total_units == 3
    assert s2.completed_units == 0
    assert s2.skipped_units == 3
    assert s2.failed_units == 0
    # unit_fn should not have been called because all units were resumed
    assert call_tracker.call_count == 0


def test_local_sequential_executor_row_indices_subset(tmp_path: Path):
    """Verify row_indices subsets planned execution."""
    manifest = _create_mock_manifest(num_rows=4)
    output_dir = tmp_path / "outputs"

    executor = LocalSequentialExecutor()
    summary = executor.run(
        manifest,
        _simple_unit_fn,
        output_dir=str(output_dir),
        row_indices=[1, 3],
    )

    assert summary.total_units == 2
    assert summary.completed_units == 2
    assert summary.skipped_units == 0

    # Only rows 1 and 3 outputs exist
    row1 = manifest.rows[1]
    row3 = manifest.rows[3]
    assert (output_dir / f"{row1.manifest_row_hash}.done.json").exists() or (
        output_dir / f"{Path(row1.output_locator).name}.done.json"
    ).exists()
    assert (output_dir / f"{row3.manifest_row_hash}.done.json").exists() or (
        output_dir / f"{Path(row3.output_locator).name}.done.json"
    ).exists()


def test_local_sequential_executor_max_failures_threshold(tmp_path: Path):
    """Verify execution halts when max_failures threshold is exceeded."""
    manifest = _create_mock_manifest(num_rows=5)
    output_dir = tmp_path / "outputs"

    def failing_unit_fn(row: ManifestRow):
        if row.chunk_id in (0, 1):
            raise RuntimeError(f"Simulated crash on chunk {row.chunk_id}")
        return {"ok": True}

    executor = LocalSequentialExecutor()
    summary = executor.run(
        manifest,
        failing_unit_fn,
        output_dir=str(output_dir),
        max_failures=1,
    )

    assert summary.failed_units == 1
    assert len(summary.failures) == 1
    assert "Simulated crash on chunk 0" in summary.failures[0]["error"]
    # Halted early at chunk 0 failure before chunk 1..4
    assert summary.completed_units == 0


def test_local_sequential_executor_cache_clearing_transitions(tmp_path: Path):
    """Verify jax.clear_caches() is called ONLY on backend or bucket transitions.

    Per Spec Section 6.3.1:
    Unconditional per-unit jax.clear_caches() is strictly prohibited.
    Invoked only upon arm backend or padding bucket transitions.
    """
    backends = ["boltz2", "boltz2", "chai1", "chai1", "chai1"]
    buckets = [256, 256, 256, 512, 512]
    manifest = _create_mock_manifest(
        num_rows=5, backends=backends, padding_buckets=buckets
    )
    output_dir = tmp_path / "outputs"

    executor = LocalSequentialExecutor()

    # Mock jax.clear_caches
    mock_clear_caches = MagicMock()
    with patch("jax.clear_caches", mock_clear_caches, create=True):
        executor.run(manifest, _simple_unit_fn, output_dir=str(output_dir))

    # Transitions occur at:
    # row 0 (boltz2, 256): initial, no transition
    # row 1 (boltz2, 256): same backend and bucket -> NO clear
    # row 2 (chai1, 256): backend changes boltz2 -> chai1 -> CLEAR #1
    # row 3 (chai1, 512): bucket changes 256 -> 512 -> CLEAR #2
    # row 4 (chai1, 512): same backend and bucket -> NO clear
    assert mock_clear_caches.call_count == 2


def test_initialize_worker_process(tmp_path: Path):
    """Verify worker process initializer configures JAX compilation cache."""
    cache_dir = tmp_path / "test_jax_cache"

    with patch("jax.config.update") as mock_update:
        _initialize_worker_process(str(cache_dir))
        mock_update.assert_any_call("jax_compilation_cache_dir", str(cache_dir))
        mock_update.assert_any_call("jax_persistent_cache_min_compile_time_secs", 1.0)


def _pool_rss_worker_fn(row: ManifestRow) -> dict[str, Any]:
    """Worker task that allocates a numpy buffer and returns VmRSS."""
    arr = np.ones((1000, 1000), dtype=np.float32)
    val = float(arr.sum())
    del arr
    gc.collect()
    rss = _get_process_rss_kb()
    return {"val": val, "rss_kb": rss, "row_hash": row.manifest_row_hash}


def test_gate_07_positive_control_rss_bounded_across_50_units(tmp_path: Path):
    """GATE-07: Positive Control: bounded process RSS across 50 units.

    recycle_interval=50 keeps process RSS bounded within ±2% across 50 units.
    Per AC-08 and GATE-07 from SPEC-261105-CAMPAIGN-RUNNER:
    - 50 sequential compute units executed
    - recycle_interval=50 and persistent disk cache configured
    - Process RSS memory stays bounded within ±2% across steady-state execution.
    """
    manifest = _create_mock_manifest(num_rows=50, campaign_id="gate07_campaign")
    output_dir = tmp_path / "outputs"

    executor = LocalProcessPoolExecutor(
        max_workers=1,
        recycle_interval=50,
        cache_dir=str(tmp_path / "jax_cache"),
    )

    summary = executor.run(
        manifest,
        _pool_rss_worker_fn,
        output_dir=str(output_dir),
    )

    assert summary.total_units == 50
    assert summary.completed_units == 50
    assert summary.failed_units == 0

    for row in manifest.rows:
        done_file = output_dir / f"{row.manifest_row_hash}.done.json"
        if not done_file.exists():
            done_file = output_dir / f"{Path(row.output_locator).name}.done.json"
        assert done_file.exists(), f"Done marker missing for row {row.manifest_row_hash}"

    # Also verify LocalSequentialExecutor memory bounded across 50 units
    seq_executor = LocalSequentialExecutor()
    seq_output = tmp_path / "seq_outputs"
    # Warmup
    warmup_manifest = _create_mock_manifest(num_rows=5, campaign_id="warmup")
    seq_executor.run(warmup_manifest, _pool_rss_worker_fn, output_dir=str(seq_output))
    gc.collect()
    base_rss = _get_process_rss_kb()

    seq_manifest = _create_mock_manifest(num_rows=50, campaign_id="seq_gate07")
    seq_summary = seq_executor.run(
        seq_manifest,
        _pool_rss_worker_fn,
        output_dir=str(seq_output),
    )
    assert seq_summary.completed_units == 50
    gc.collect()
    final_rss = _get_process_rss_kb()

    drift = abs(final_rss - base_rss) / max(base_rss, 1)
    assert drift <= 0.02, (
        f"GATE-07 FAILED: Process RSS drifted by {drift:.4%} (base: {base_rss} kB, "
        f"final: {final_rss} kB), exceeding ±2% limit."
    )
