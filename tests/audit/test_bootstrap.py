"""Tests for N3.1 audit bootstrap orchestrator (#1592)."""

from __future__ import annotations

import tomllib
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    save_baseline,
)
from xtrax.devtools.bootstrap import (
    DimensionRun,
    run_audit_bootstrap,
    write_bootstrap_manifest,
)
from xtrax.devtools.gates import test_rigor as test_rigor_gate
from xtrax.devtools.gates.api_ergonomics import GateResult as ApiGateResult
from xtrax.devtools.gates.correctness import GateResult as CorrectnessGateResult
from xtrax.devtools.gates.documentation import GateResult as DocumentationGateResult
from xtrax.devtools.gates.jax_purity import GateResult as JaxPurityGateResult
from xtrax.devtools.gates.performance import GateResult as PerformanceGateResult
from xtrax.devtools.gates.structure_complexity import (
    GateResult as StructureGateResult,
)
from xtrax.devtools.gates.type_hardening import (
    AnnotationStats,
)
from xtrax.devtools.gates.type_hardening import (
    GateResult as TypeHardeningGateResult,
)

ROOT = Path(__file__).resolve().parents[2]


def _passing_gate_results() -> dict[str, object]:
    return {
        "correctness": CorrectnessGateResult(
            passed=True,
            violation_count=0,
            findings_emitted=0,
            baseline_updated=False,
        ),
        "jax_purity": JaxPurityGateResult(
            passed=True,
            violation_count=0,
            findings_emitted=0,
            baseline_updated=False,
        ),
        "type_hardening": TypeHardeningGateResult(
            passed=True,
            stats=AnnotationStats(
                public_params=10,
                annotated_params=10,
                array_annotations=5,
                shape_typed_arrays=5,
            ),
            annotation_coverage_pct=100.0,
            shape_specificity_pct=100.0,
            findings_emitted=0,
            baseline_updated=False,
        ),
        "performance": PerformanceGateResult(
            passed=True,
            trace_violation_count=0,
            wall_time_median_ms=1.5,
            findings_emitted=0,
            baseline_updated=False,
        ),
        "documentation": DocumentationGateResult(
            passed=True,
            interrogate_coverage_pct=90.0,
            jd_violation_count=0,
            findings_emitted=0,
            baseline_updated=False,
        ),
        "api_ergonomics": ApiGateResult(
            passed=True,
            violation_count=0,
            violations=(),
            findings_emitted=0,
            baseline_updated=False,
        ),
        "test_rigor": test_rigor_gate.GateResult(
            passed=True,
            stats=test_rigor_gate.CoverageStats(
                line_pct=95.0,
                branch_pct=90.0,
                tests_run=10,
                tests_failed=0,
            ),
            line_coverage_pct=95.0,
            branch_coverage_pct=90.0,
            findings_emitted=0,
            baseline_updated=False,
        ),
        "structure_complexity": StructureGateResult(
            passed=True,
            cognitive_complexity_max=10.0,
            ruff_violation_count=0,
            cognitive_hits=(),
            ruff_hits=(),
            findings_emitted=0,
            baseline_updated=False,
        ),
    }


def _seed_baseline(path: Path) -> None:
    baseline = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            "correctness.jl_violation_count": MetricEntry(
                key="correctness.jl_violation_count",
                value=0.0,
                comparator="minimize",
            ),
            "jax_purity.jl_violation_count": MetricEntry(
                key="jax_purity.jl_violation_count",
                value=0.0,
                comparator="minimize",
            ),
            "type_hardening.annotation_coverage_pct": MetricEntry(
                key="type_hardening.annotation_coverage_pct",
                value=0.0,
                comparator="maximize",
            ),
            "type_hardening.shape_specificity_pct": MetricEntry(
                key="type_hardening.shape_specificity_pct",
                value=0.0,
                comparator="maximize",
            ),
            "performance.trace_violation_count": MetricEntry(
                key="performance.trace_violation_count",
                value=0.0,
                comparator="minimize",
            ),
            "documentation.interrogate_coverage_pct": MetricEntry(
                key="documentation.interrogate_coverage_pct",
                value=0.0,
                comparator="maximize",
            ),
            "documentation.jd_violation_count": MetricEntry(
                key="documentation.jd_violation_count",
                value=0.0,
                comparator="minimize",
            ),
            "api_ergonomics.param_sprawl_violation_count": MetricEntry(
                key="api_ergonomics.param_sprawl_violation_count",
                value=0.0,
                comparator="minimize",
            ),
            "test_rigor.line_coverage_pct": MetricEntry(
                key="test_rigor.line_coverage_pct",
                value=0.0,
                comparator="maximize",
            ),
            "test_rigor.branch_coverage_pct": MetricEntry(
                key="test_rigor.branch_coverage_pct",
                value=0.0,
                comparator="maximize",
            ),
            "structure.cognitive_complexity_max": MetricEntry(
                key="structure.cognitive_complexity_max",
                value=100.0,
                comparator="minimize",
            ),
            "structure.ruff_complexity_violation_count": MetricEntry(
                key="structure.ruff_complexity_violation_count",
                value=100.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(baseline, path=path)


def _patch_gate_runners(gate_results: dict[str, object]):
    @contextmanager
    def _manager():
        with (
            patch(
                "xtrax.devtools.bootstrap.run_correctness_gate",
                return_value=gate_results["correctness"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_jax_purity_gate",
                return_value=gate_results["jax_purity"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_type_hardening_gate",
                return_value=gate_results["type_hardening"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_performance_gate",
                return_value=gate_results["performance"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_documentation_gate",
                return_value=gate_results["documentation"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_api_ergonomics_gate",
                return_value=gate_results["api_ergonomics"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_test_rigor_gate",
                return_value=gate_results["test_rigor"],
            ),
            patch(
                "xtrax.devtools.bootstrap.run_structure_complexity_gate",
                return_value=gate_results["structure_complexity"],
            ),
        ):
            yield

    return _manager()


@pytest.fixture
def mock_gate_runners() -> dict[str, object]:
    return _passing_gate_results()


def test_run_audit_bootstrap_all_passed_writes_manifest(
    tmp_path: Path,
    mock_gate_runners: dict[str, object],
) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    manifest_path = tmp_path / "audit_bootstrap_manifest.toml"
    _seed_baseline(baseline_path)

    with _patch_gate_runners(mock_gate_runners):
        result = run_audit_bootstrap(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=ROOT,
            write_baseline=False,
            test_rigor_quick=True,
            run_id="bootstrap-test-run",
            manifest_path=manifest_path,
        )

    assert result.passed is True
    assert len(result.runs) == 8
    assert manifest_path.is_file()

    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bootstrap"]["run_id"] == "bootstrap-test-run"
    assert manifest["bootstrap"]["all_passed"] is True
    assert len(manifest["dimensions"]) == 8
    dimensions = [entry["dimension"] for entry in manifest["dimensions"]]
    assert dimensions == [
        "correctness",
        "jax_purity",
        "type_hardening",
        "performance",
        "documentation",
        "api_ergonomics",
        "test_rigor",
        "structure_complexity",
    ]


def test_run_audit_bootstrap_fails_when_any_dimension_fails(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    manifest_path = tmp_path / "audit_bootstrap_manifest.toml"
    _seed_baseline(baseline_path)

    gate_results = _passing_gate_results()
    gate_results["jax_purity"] = JaxPurityGateResult(
        passed=False,
        violation_count=2,
        findings_emitted=2,
        baseline_updated=False,
    )

    with _patch_gate_runners(gate_results):
        result = run_audit_bootstrap(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=ROOT,
            write_baseline=False,
            manifest_path=manifest_path,
        )

    assert result.passed is False
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["bootstrap"]["all_passed"] is False
    jax_entry = next(
        entry for entry in manifest["dimensions"] if entry["dimension"] == "jax_purity"
    )
    assert jax_entry["passed"] is False
    assert jax_entry["backlog_seed"].startswith("Fix JL001")


def test_run_audit_bootstrap_uses_quick_test_rigor_path(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    _seed_baseline(baseline_path)
    gate_results = _passing_gate_results()

    with _patch_gate_runners(gate_results):
        with patch(
            "xtrax.devtools.bootstrap.run_test_rigor_gate",
            return_value=gate_results["test_rigor"],
        ) as test_rigor_mock:
            run_audit_bootstrap(
                audits_path=audits_path,
                baseline_path=baseline_path,
                root=ROOT,
                write_baseline=False,
                test_rigor_quick=True,
            )

    _, kwargs = test_rigor_mock.call_args
    assert kwargs["tests_path"] == (ROOT / "tests" / "audit").resolve()


def test_write_bootstrap_manifest_emits_debt_seed_for_maximize_floor(
    tmp_path: Path,
) -> None:
    baseline = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            "test_rigor.line_coverage_pct": MetricEntry(
                key="test_rigor.line_coverage_pct",
                value=50.0,
                comparator="maximize",
            ),
        },
    )
    runs = (
        DimensionRun(
            dimension="test_rigor",
            passed=True,
            metrics={"test_rigor.line_coverage_pct": 50.0},
            findings_emitted=1,
            baseline_updated=False,
        ),
    )
    manifest_path = tmp_path / "manifest.toml"
    write_bootstrap_manifest(
        manifest_path,
        run_id="debt-run",
        updated_at="2026-06-20T00:00:00+00:00",
        all_passed=True,
        runs=runs,
        baseline=baseline,
    )
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert "Ratchet floor debt" in manifest["dimensions"][0]["backlog_seed"]
