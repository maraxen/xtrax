"""Smoke + unit pins for the xtrax-optimizing probe drivers (260825 scope).

The drivers are scripts, not library API; these tests pin what must stay true
for their ProbeRecords to remain trustworthy:

1. every driver imports and exposes main(argv);
2. the stage-0 metric normalizer never lets booleans/non-finite junk through;
3. the Tier-1 correctness gate fails loud on broken sink observations;
4. the cheap drivers run END TO END against tmp out-dirs at tiny sizes,
   emitting records that pass ProbeRecord.read (fail-closed validation).

End-to-end cases use tiny parameters to bound wall time; they exercise the
same code paths as the committed outputs/profiling records.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import pytest

from scripts.prof_stage0_onehot_cost import (
    _numeric_cost_metrics as stage0_numeric_cost_metrics,
)
from scripts.prof_stage0_onehot_cost import main as stage0_main
from scripts.prof_stage1_feed_overlap import main as feed_overlap_main
from scripts.prof_stage1_host_boundary import VARIANTS as HOST_VARIANTS
from scripts.prof_stage1_host_boundary import (
    _correctness_gate,
)
from scripts.prof_stage1_host_boundary import main as host_boundary_main
from scripts.prof_stage1_onehot_micro import PARITY_TOLERANCE
from scripts.prof_stage1_onehot_micro import main as stage1_onehot_main


class TestDriverImports:
    def test_every_driver_exposes_main(self) -> None:
        for main in (stage0_main, stage1_onehot_main, host_boundary_main, feed_overlap_main):
            assert callable(main)

    def test_host_boundary_variant_vocabulary(self) -> None:
        assert HOST_VARIANTS == ("none", "unordered", "ordered")


class TestStage0MetricNormalizer:
    def test_drops_bools_nonfinite_and_junk_keeps_finite_floats(self) -> None:
        metrics = stage0_numeric_cost_metrics(
            {
                "Flops": 12.0,
                "bytes accessed": 34,
                "bad flag": True,
                "nan value": float("nan"),
                "inf value": float("inf"),
                "junk": "not-a-number",
                "none": None,
            }
        )
        assert metrics == {"flops": 12.0, "bytes_accessed": 34.0}


class TestHostBoundaryCorrectnessGate:
    def test_accepts_ordered_and_unordered_valid_observations(self) -> None:
        _correctness_gate(
            {
                "unordered": [3, 1, 0, 2],  # reorder allowed
                "ordered": [0, 1, 2, 3],  # order required
            },
            steps=4,
        )

    def test_raises_when_a_step_is_missed_or_duplicated(self) -> None:
        with pytest.raises(SystemExit, match="unordered"):
            _correctness_gate({"unordered": [0, 0, 2], "ordered": [0, 1, 2]}, steps=3)

    def test_raises_when_ordered_breaks_order(self) -> None:
        with pytest.raises(SystemExit, match="ordering guarantee BROKEN"):
            _correctness_gate({"unordered": [0, 1, 2], "ordered": [1, 0, 2]}, steps=3)


class TestParityToleranceContract:
    def test_tolerance_is_finite_positive_small(self) -> None:
        assert math.isfinite(PARITY_TOLERANCE) and 0 < PARITY_TOLERANCE <= 1e-3


class TestEndToEndTinyRuns:
    """Full driver runs at reduced sizes; records must round-trip read()."""

    def test_stage0_onehot_writes_two_readable_records(self, tmp_path: Path) -> None:
        from xtrax.profiling.record import ProbeRecord

        assert (
            stage0_main(
                ["--out-dir", str(tmp_path), "--rows", "16", "--classes", "4", "--cols", "4"]
            )
            == 0
        )
        paths = sorted(tmp_path.glob("stage0_onehot_*.json"))
        assert len(paths) == 2
        for path in paths:
            record = ProbeRecord.read(path)
            assert record.stage == 0
            assert record.probe_id == path.stem

    def test_host_boundary_tiny_run_record_is_readable(self, tmp_path: Path) -> None:
        from xtrax.profiling.record import ProbeRecord

        rc = host_boundary_main(
            ["--out-dir", str(tmp_path), "--steps", "4", "--warmup", "1", "--trials", "2"]
        )
        assert rc == 0
        record = ProbeRecord.read(tmp_path / "stage1_host_boundary.json")
        assert record.metrics["ordered_over_none_ratio"] > 0
        # Every variant contributed dispatch counts.
        for variant in HOST_VARIANTS:
            assert f"{variant}_n_executions" in record.metrics

    def test_stage1_onehot_micro_tiny_run_record_is_readable(self, tmp_path: Path) -> None:
        from xtrax.profiling.record import ProbeRecord

        rc = stage1_onehot_main(
            [
                "--out-dir",
                str(tmp_path),
                "--rows",
                "16",
                "--classes",
                "4",
                "--cols",
                "4",
                "--warmup",
                "1",
                "--trials",
                "2",
            ]
        )
        assert rc == 0
        record = ProbeRecord.read(tmp_path / "stage1_onehot_micro.json")
        assert set(record.scopes or {}) >= {"onehot_materialized", "onehot_onthefly"}
        assert 0 <= record.metrics["parity_max_abs_diff"] <= PARITY_TOLERANCE

    def test_feed_overlap_tiny_run_record_is_readable(self, tmp_path: Path) -> None:
        from xtrax.profiling.record import ProbeRecord

        rc = feed_overlap_main(
            [
                "--out-dir",
                str(tmp_path),
                "--batches",
                "3",
                "--rows",
                "16",
                "--cols",
                "8",
                "--feed-sleep-ms",
                "0.1",
                "--trials",
                "1",
            ]
        )
        assert rc == 0
        record = ProbeRecord.read(tmp_path / "stage1_feed_overlap.json")
        assert record.metrics["sequential_seconds"] > 0
        assert record.metrics["overlapped_seconds"] > 0
        assert jnp.isfinite(record.metrics["speedup_ratio"])
