"""Tests for D4 performance gate (N2.5 / #1584)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates._trace_probe import ProbeResult, run_trace_probe
from xtrax.devtools.gates.performance import (
    METRIC_KEY,
    WALL_TIME_METRIC_KEY,
    load_performance_targets,
    run_performance_gate,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"
TARGETS_PATH = ROOT / "audit" / "performance_targets.toml"
PROBE_STABLE = "xtrax.devtools.gates._performance_probes:probe_stable_jnp_kernel"


def test_performance_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "performance.toml")
    assert table.dimension == "performance"
    assert len(table.anchors) == 5


def test_load_performance_targets_repo_config() -> None:
    targets = load_performance_targets(TARGETS_PATH)
    assert targets.schema == "performance-gate-v0"
    assert targets.max_traces_default == 1
    assert len(targets.probes) == 1
    assert targets.probes[0].qualname.endswith("sparse_filter_jit_kernel")
    assert targets.probes[0].trace_probe is not None


def test_run_trace_probe_skips_without_trace_probe() -> None:
    result = run_trace_probe(
        "xtrax.sparse.inference.sparse_filter_jit",
        max_traces=1,
        trace_probe=None,
    )
    assert result.skipped is True
    assert result.passed is False


def test_run_trace_probe_stable_jnp_kernel_passes(tmp_path: Path) -> None:
    probe_module = tmp_path / "trace_kernel_fixture.py"
    probe_module.write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def kernel(x):
                return x + 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))

    result = run_trace_probe(
        "trace_kernel_fixture.kernel",
        max_traces=1,
        trace_probe=PROBE_STABLE,
    )
    assert result.passed is True
    assert not result.skipped


def test_run_performance_gate_fails_on_mock_trace_failure(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "xtrax.fake.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    mock_failure = ProbeResult(
        qualname="xtrax.fake.kernel",
        passed=False,
        reason="trace count exceeded",
        trace_probe=PROBE_STABLE,
        max_traces=1,
    )

    with (
        patch(
            "xtrax.devtools.gates.performance.run_trace_probe",
            return_value=mock_failure,
        ),
        patch(
            "xtrax.devtools.gates.performance._measure_wall_time_median_ms",
            return_value=1.0,
        ),
    ):
        result = run_performance_gate(
            targets_path=targets_path,
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is False
    assert result.trace_violation_count == 1
    assert result.findings_emitted >= 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "performance"
    assert record["payload"]["violation_kind"] == "trace_count"
    assert record["severity"] == "major"


def test_run_performance_gate_passes_with_stable_probe(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    probe_module = tmp_path / "stable_kernel_fixture.py"
    probe_module.write_text(
        textwrap.dedent(
            """
            import jax.numpy as jnp

            def kernel(x):
                return x + 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))

    targets_path = tmp_path / "performance_targets.toml"
    targets_path.write_text(
        textwrap.dedent(
            f"""
            [gate]
            schema = "performance-gate-v0"
            version = "0.1.0"
            max_traces_default = 1

            [[probes]]
            qualname = "stable_kernel_fixture.kernel"
            max_traces = 1
            trace_probe = "{PROBE_STABLE}"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is True
    assert result.trace_violation_count == 0
    assert result.wall_time_median_ms is not None
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    wall_record = json.loads(lines[-1])
    assert wall_record["severity"] == "info"
    assert wall_record["payload"]["recorded_only"] is True
    assert "wall_time_median_ms" in wall_record["payload"]


def test_run_performance_gate_emits_recorded_wall_time(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    mock_pass = ProbeResult(
        qualname="xtrax.devtools.gates._performance_probes.sparse_filter_jit_kernel",
        passed=True,
        trace_probe="xtrax.devtools.gates._performance_probes:probe_sparse_filter_jit_kernel",
        max_traces=1,
    )

    with (
        patch(
            "xtrax.devtools.gates.performance.run_trace_probe",
            return_value=mock_pass,
        ),
        patch(
            "xtrax.devtools.gates.performance._measure_wall_time_median_ms",
            return_value=1.0,
        ),
    ):
        result = run_performance_gate(
            targets_path=TARGETS_PATH,
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is True
    assert result.wall_time_median_ms == 1.0
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[-1])
    assert record["payload"]["recorded_only"] is True
    assert record["payload"]["wall_time_median_ms"] == 1.0


def test_run_performance_gate_integration_sparse_filter_jit(tmp_path: Path) -> None:
    """End-to-end trace gate against repo performance_targets (no mocks)."""
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_performance_gate(
        targets_path=TARGETS_PATH,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )

    assert result.passed is True
    assert result.trace_violation_count == 0
    assert result.wall_time_median_ms is not None


def test_committed_baseline_has_performance_trace_metric() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert METRIC_KEY in loaded.metrics
    entry = loaded.metrics[METRIC_KEY]
    assert entry.comparator == "minimize"
    assert entry.value == 0.0


def test_audit_performance_gate_cli_exits_zero(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            METRIC_KEY: MetricEntry(
                key=METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_performance_gate.py",
            "--baseline-path",
            str(baseline_path),
            "--audits-path",
            str(audits_path),
            "--no-write-baseline",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout
    assert WALL_TIME_METRIC_KEY in result.stdout
