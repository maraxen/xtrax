"""Tests for D7 test-rigor gate (N2.7 / #1587)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.audit_test_rigor_gate import main
from xtrax.devtools.baseline import (
    BASELINE_SCHEMA_VERSION,
    AuditBaseline,
    MetricEntry,
    load_baseline,
    save_baseline,
)
from xtrax.devtools.gates.test_rigor import (
    BRANCH_METRIC,
    LINE_METRIC,
    CoverageStats,
    GateResult,
    parse_coverage_json,
    parse_pytest_summary,
    run_test_rigor_gate,
)
from xtrax.devtools.rubrics import load_rubric

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"


def test_test_rigor_rubric_loads() -> None:
    table = load_rubric(RUBRICS_DIR / "test_rigor.toml")
    assert table.dimension == "test_rigor"
    assert len(table.anchors) == 5


def test_parse_coverage_json_fixture(tmp_path: Path) -> None:
    cov_path = tmp_path / "coverage.json"
    cov_path.write_text(
        json.dumps(
            {
                "totals": {
                    "percent_covered": 42.5,
                    "percent_branches_covered": 31.25,
                }
            }
        ),
        encoding="utf-8",
    )
    line_pct, branch_pct = parse_coverage_json(cov_path)
    assert line_pct == 42.5
    assert branch_pct == 31.25


@pytest.mark.parametrize(
    ("summary", "tests_run", "tests_failed"),
    [
        (
            "....... [100%]\n7 passed in 2.57s",
            7,
            0,
        ),
        ("5 passed, 2 failed in 1.2s", 7, 2),
        ("2 failed, 5 passed in 1.2s", 7, 2),
        ("1 failed, 2 errors in 0.5s", 3, 3),
    ],
)
def test_parse_pytest_summary(
    summary: str,
    tests_run: int,
    tests_failed: int,
) -> None:
    assert parse_pytest_summary(summary) == (tests_run, tests_failed)


def test_run_test_rigor_gate_passes_at_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            LINE_METRIC: MetricEntry(
                key=LINE_METRIC,
                value=0.0,
                comparator="maximize",
            ),
            BRANCH_METRIC: MetricEntry(
                key=BRANCH_METRIC,
                value=0.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)
    mock_stats = CoverageStats(
        line_pct=55.0,
        branch_pct=40.0,
        tests_run=10,
        tests_failed=0,
    )

    with patch(
        "xtrax.devtools.gates.test_rigor.run_pytest_coverage",
        return_value=mock_stats,
    ):
        result = run_test_rigor_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            write_baseline=False,
        )

    assert result.passed is True
    assert result.line_coverage_pct == 55.0
    assert result.branch_coverage_pct == 40.0
    assert result.findings_emitted == 1
    lines = audits_path.read_text(encoding="utf-8").strip().splitlines()
    record = json.loads(lines[0])
    assert record["dim"] == "test_rigor"
    assert record["payload"]["line_coverage_pct"] == 55.0


def test_run_test_rigor_gate_fails_on_regression(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            LINE_METRIC: MetricEntry(
                key=LINE_METRIC,
                value=90.0,
                comparator="maximize",
            ),
            BRANCH_METRIC: MetricEntry(
                key=BRANCH_METRIC,
                value=80.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)
    mock_stats = CoverageStats(
        line_pct=85.0,
        branch_pct=75.0,
        tests_run=20,
        tests_failed=0,
    )

    with patch(
        "xtrax.devtools.gates.test_rigor.run_pytest_coverage",
        return_value=mock_stats,
    ):
        result = run_test_rigor_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            write_baseline=False,
        )

    assert result.passed is False


def test_run_test_rigor_gate_tightens_baseline(tmp_path: Path) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            LINE_METRIC: MetricEntry(
                key=LINE_METRIC,
                value=0.0,
                comparator="maximize",
            ),
            BRANCH_METRIC: MetricEntry(
                key=BRANCH_METRIC,
                value=0.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)
    mock_stats = CoverageStats(
        line_pct=12.5,
        branch_pct=8.0,
        tests_run=5,
        tests_failed=0,
    )

    with patch(
        "xtrax.devtools.gates.test_rigor.run_pytest_coverage",
        return_value=mock_stats,
    ):
        result = run_test_rigor_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            write_baseline=True,
        )

    assert result.passed is True
    assert result.baseline_updated is True
    updated = load_baseline(path=baseline_path)
    assert updated.metrics[LINE_METRIC].value == 12.5
    assert updated.metrics[BRANCH_METRIC].value == 8.0


def test_committed_baseline_has_test_rigor_metrics() -> None:
    repo_baseline = ROOT / ".praxia" / "audit_baseline.json"
    if not repo_baseline.is_file():
        pytest.skip("seed baseline not present in checkout")
    loaded = load_baseline(path=repo_baseline)
    assert LINE_METRIC in loaded.metrics
    assert BRANCH_METRIC in loaded.metrics
    assert loaded.metrics[LINE_METRIC].comparator == "maximize"
    assert loaded.metrics[BRANCH_METRIC].comparator == "maximize"


def test_audit_test_rigor_gate_cli_exits_zero_with_mock(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "audit_baseline.json"
    audits_path = tmp_path / "audits.jsonl"
    seed = AuditBaseline(
        schema_version=BASELINE_SCHEMA_VERSION,
        updated_at="2026-06-19T00:00:00+00:00",
        metrics={
            LINE_METRIC: MetricEntry(
                key=LINE_METRIC,
                value=0.0,
                comparator="maximize",
            ),
            BRANCH_METRIC: MetricEntry(
                key=BRANCH_METRIC,
                value=0.0,
                comparator="maximize",
            ),
        },
    )
    save_baseline(seed, path=baseline_path)
    mock_stats = CoverageStats(
        line_pct=1.0,
        branch_pct=1.0,
        tests_run=1,
        tests_failed=0,
    )
    mock_result = GateResult(
        passed=True,
        stats=mock_stats,
        line_coverage_pct=1.0,
        branch_coverage_pct=1.0,
        findings_emitted=1,
        baseline_updated=False,
    )

    with patch(
        "scripts.audit_test_rigor_gate.run_test_rigor_gate",
        return_value=mock_result,
    ):
        exit_code = main(
            [
                "--baseline-path",
                str(baseline_path),
                "--audits-path",
                str(audits_path),
                "--no-write-baseline",
                "--tests-path",
                str(ROOT / "tests" / "audit"),
            ]
        )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out
