"""Tests for D5 documentation gate (N2.6 / #1585)."""

from __future__ import annotations

import json
import subprocess
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
from xtrax.devtools.gates.documentation import (
    COVERAGE_METRIC_KEY,
    JD_METRIC_KEY,
    filter_jd_jm_errors,
    run_documentation_gate,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"


def test_documentation_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "documentation.toml")
    assert table.dimension == "documentation"
    assert len(table.anchors) == 5


def test_filter_jd_jm_errors_keeps_jd_jm_errors_only() -> None:
    findings = [
        {"rule_id": "JD001", "severity": "error", "message": "missing doc"},
        {"rule_id": "JM002", "severity": "error", "message": "math doc"},
        {"rule_id": "JD001", "severity": "warning", "message": "warn"},
        {"rule_id": "JL001", "severity": "error", "message": "purity"},
        {"rule_id": "JM003", "severity": "info", "message": "info"},
    ]
    errors = filter_jd_jm_errors(findings)
    assert len(errors) == 2
    assert {e["rule_id"] for e in errors} == {"JD001", "JM002"}


def test_run_documentation_gate_passes_at_baseline(tmp_path: Path) -> None:
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
            JD_METRIC_KEY: MetricEntry(
                key=JD_METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with (
        patch(
            "xtrax.devtools.gates.documentation.run_interrogate_coverage",
            return_value=75.0,
        ),
        patch(
            "xtrax.devtools.gates.documentation._run_jaxlint_json",
            return_value=[],
        ),
    ):
        result = run_documentation_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is True
    assert result.interrogate_coverage_pct == 75.0
    assert result.jd_violation_count == 0
    assert result.findings_emitted == 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "documentation"
    assert record["payload"]["interrogate_coverage_pct"] == 75.0


def test_run_documentation_gate_fails_on_coverage_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COVERAGE_METRIC_KEY: MetricEntry(
                key=COVERAGE_METRIC_KEY,
                value=90.0,
                comparator="maximize",
            ),
            JD_METRIC_KEY: MetricEntry(
                key=JD_METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with (
        patch(
            "xtrax.devtools.gates.documentation.run_interrogate_coverage",
            return_value=80.0,
        ),
        patch(
            "xtrax.devtools.gates.documentation._run_jaxlint_json",
            return_value=[],
        ),
    ):
        result = run_documentation_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is False
    assert result.interrogate_coverage_pct == 80.0


def test_run_documentation_gate_fails_on_jd_regression(tmp_path: Path) -> None:
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
            JD_METRIC_KEY: MetricEntry(
                key=JD_METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    mock_findings = [
        {
            "rule_id": "JD001",
            "severity": "error",
            "message": "missing module docstring",
            "path": "src/xtrax/foo.py",
            "line": 1,
        },
        {
            "rule_id": "JL001",
            "severity": "error",
            "message": "excluded purity rule",
            "path": "src/xtrax/bar.py",
            "line": 3,
        },
    ]

    with (
        patch(
            "xtrax.devtools.gates.documentation.run_interrogate_coverage",
            return_value=85.0,
        ),
        patch(
            "xtrax.devtools.gates.documentation._run_jaxlint_json",
            return_value=mock_findings,
        ),
    ):
        result = run_documentation_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=False,
        )

    assert result.passed is False
    assert result.jd_violation_count == 1
    assert result.findings_emitted == 2
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    jd_record = json.loads(lines[1])
    assert jd_record["payload"]["rule_id"] == "JD001"
    assert jd_record["severity"] == "major"


def test_run_documentation_gate_tightens_baseline_on_improvement(
    tmp_path: Path,
) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            COVERAGE_METRIC_KEY: MetricEntry(
                key=COVERAGE_METRIC_KEY,
                value=50.0,
                comparator="maximize",
            ),
            JD_METRIC_KEY: MetricEntry(
                key=JD_METRIC_KEY,
                value=2.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with (
        patch(
            "xtrax.devtools.gates.documentation.run_interrogate_coverage",
            return_value=80.0,
        ),
        patch(
            "xtrax.devtools.gates.documentation._run_jaxlint_json",
            return_value=[],
        ),
    ):
        result = run_documentation_gate(
            target=ROOT / "src" / "xtrax",
            audits_path=audits_path,
            baseline_path=baseline_path,
            write_baseline=True,
        )

    assert result.passed is True
    assert result.baseline_updated is True
    updated = load_baseline(path=baseline_path)
    assert updated.metrics[COVERAGE_METRIC_KEY].value == 80.0
    assert updated.metrics[JD_METRIC_KEY].value == 0.0


def test_committed_baseline_has_documentation_metrics() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert COVERAGE_METRIC_KEY in loaded.metrics
    assert JD_METRIC_KEY in loaded.metrics
    assert loaded.metrics[COVERAGE_METRIC_KEY].comparator == "maximize"
    assert loaded.metrics[JD_METRIC_KEY].comparator == "minimize"
    assert loaded.metrics[JD_METRIC_KEY].value == 0.0


def test_audit_documentation_gate_cli_exits_zero_on_mocked_clean_tree(
    tmp_path: Path,
) -> None:
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
            JD_METRIC_KEY: MetricEntry(
                key=JD_METRIC_KEY,
                value=0.0,
                comparator="minimize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)

    with (
        patch(
            "xtrax.devtools.gates.documentation.run_interrogate_coverage",
            return_value=70.0,
        ),
        patch(
            "xtrax.devtools.gates.documentation._run_jaxlint_json",
            return_value=[],
        ),
    ):
        result = subprocess.run(
            [
                "uv",
                "run",
                "python",
                "scripts/audit_documentation_gate.py",
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
