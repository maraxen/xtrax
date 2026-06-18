"""Tests for D3 type-hardening gate (N2.3 / #1583)."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates.type_hardening import (
    COVERAGE_METRIC_KEY,
    SHAPE_METRIC_KEY,
    run_type_hardening_gate,
    scan_package_annotations,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"


def _write_fixture_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")


def test_type_hardening_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "type_hardening.toml")
    assert table.dimension == "type_hardening"
    assert len(table.anchors) == 5


def test_scan_package_annotations_fixture_module(tmp_path: Path) -> None:
    pkg = tmp_path / "sample_pkg"
    _write_fixture_module(
        pkg / "typed.py",
        """
        import jax
        from jaxtyping import Float

        def public_fn(x: Float[jax.Array, "n"], y: int) -> Float[jax.Array, "n"]:
            return x

        def _private_fn(z):
            return z
        """,
    )
    stats = scan_package_annotations(pkg)
    assert stats.public_params == 3
    assert stats.annotated_params == 3
    assert stats.array_annotations == 2
    assert stats.shape_typed_arrays == 2
    assert stats.annotation_coverage_pct == 100.0
    assert stats.shape_specificity_pct == 100.0


def test_scan_package_annotations_partial_coverage(tmp_path: Path) -> None:
    pkg = tmp_path / "partial_pkg"
    _write_fixture_module(
        pkg / "mixed.py",
        """
        import jax
        from jaxtyping import Float

        def half_typed(x: Float[jax.Array, "n"], y) -> Float[jax.Array, "n"]:
            return x
        """,
    )
    stats = scan_package_annotations(pkg)
    assert stats.public_params == 3
    assert stats.annotated_params == 2
    assert stats.array_annotations == 2
    assert stats.shape_typed_arrays == 2
    assert stats.annotation_coverage_pct == pytest.approx(200 / 3)
    assert stats.shape_specificity_pct == 100.0


def test_scan_package_annotations_no_arrays_returns_full_shape_specificity(
    tmp_path: Path,
) -> None:
    pkg = tmp_path / "plain_pkg"
    _write_fixture_module(
        pkg / "plain.py",
        """
        def echo(x: int) -> int:
            return x
        """,
    )
    stats = scan_package_annotations(pkg)
    assert stats.array_annotations == 0
    assert stats.shape_specificity_pct == 100.0


def test_run_type_hardening_gate_passes_at_baseline(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write_fixture_module(
        pkg / "typed.py",
        """
        import jax
        from jaxtyping import Float

        def f(x: Float[jax.Array, "n"]) -> Float[jax.Array, "n"]:
            return x
        """,
    )
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COVERAGE_METRIC_KEY: MetricEntry(
                key=COVERAGE_METRIC_KEY,
                value=0.0,
                comparator="maximize",
            ),
            SHAPE_METRIC_KEY: MetricEntry(
                key=SHAPE_METRIC_KEY,
                value=0.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_type_hardening_gate(
        target=pkg,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )
    assert result.passed is True
    assert result.annotation_coverage_pct == 100.0
    assert result.shape_specificity_pct == 100.0


def test_run_type_hardening_gate_fails_on_regression(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write_fixture_module(
        pkg / "untyped.py",
        """
        def f(x, y):
            return x
        """,
    )
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COVERAGE_METRIC_KEY: MetricEntry(
                key=COVERAGE_METRIC_KEY,
                value=100.0,
                comparator="maximize",
            ),
            SHAPE_METRIC_KEY: MetricEntry(
                key=SHAPE_METRIC_KEY,
                value=100.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_type_hardening_gate(
        target=pkg,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=False,
    )
    assert result.passed is False
    assert result.findings_emitted >= 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "type_hardening"


def test_run_type_hardening_gate_tightens_baseline(
    tmp_path: Path,
) -> None:
    pkg = tmp_path / "pkg"
    _write_fixture_module(
        pkg / "typed.py",
        """
        import jax
        from jaxtyping import Float

        def f(x: Float[jax.Array, "n"]) -> Float[jax.Array, "n"]:
            return x
        """,
    )
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COVERAGE_METRIC_KEY: MetricEntry(
                key=COVERAGE_METRIC_KEY,
                value=0.0,
                comparator="maximize",
            ),
            SHAPE_METRIC_KEY: MetricEntry(
                key=SHAPE_METRIC_KEY,
                value=0.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = run_type_hardening_gate(
        target=pkg,
        audits_path=audits_path,
        baseline_path=baseline_path,
        write_baseline=True,
    )
    assert result.passed is True
    assert result.baseline_updated is True
    updated = load_baseline(path=baseline_path)
    assert updated.metrics[COVERAGE_METRIC_KEY].value == 100.0
    assert updated.metrics[SHAPE_METRIC_KEY].value == 100.0


def test_committed_baseline_has_type_hardening_metrics() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert COVERAGE_METRIC_KEY in loaded.metrics
    assert SHAPE_METRIC_KEY in loaded.metrics
    assert loaded.metrics[COVERAGE_METRIC_KEY].comparator == "maximize"
    assert loaded.metrics[SHAPE_METRIC_KEY].comparator == "maximize"


def test_audit_type_hardening_gate_cli_exits_zero_on_fixture(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    _write_fixture_module(
        pkg / "typed.py",
        """
        import jax
        from jaxtyping import Float

        def f(x: Float[jax.Array, "n"]) -> Float[jax.Array, "n"]:
            return x
        """,
    )
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COVERAGE_METRIC_KEY: MetricEntry(
                key=COVERAGE_METRIC_KEY,
                value=0.0,
                comparator="maximize",
            ),
            SHAPE_METRIC_KEY: MetricEntry(
                key=SHAPE_METRIC_KEY,
                value=0.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_type_hardening_gate.py",
            str(pkg),
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
