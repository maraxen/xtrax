"""Tests for D7 test-rigor gate (N2.7 / #1587)."""

from __future__ import annotations

import json
import subprocess
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
    run_pytest_coverage,
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


def _extract_cov_report_path(cmd: list[str]) -> Path:
    """Pull the ``--cov-report=json:<path>`` target out of a pytest cmd list."""
    for arg in cmd:
        if arg.startswith("--cov-report=json:"):
            return Path(arg.removeprefix("--cov-report=json:"))
    msg = f"no --cov-report=json: arg found in {cmd!r}"
    raise AssertionError(msg)


def test_run_pytest_coverage_raises_runtime_error_when_report_missing(
    tmp_path: Path,
) -> None:
    """C4 case 1: subprocess exits non-zero and never writes a report.

    Against current code this must NOT reach this RuntimeError at all --
    ``tempfile.NamedTemporaryFile`` already pre-created an (empty) file at
    ``cov_path`` before the subprocess ever ran, so ``cov_path.is_file()`` is
    always True and the "missing" branch is dead code. Control instead falls
    through to ``parse_coverage_json``, which raises a bare
    ``json.JSONDecodeError`` on the empty file. This test pins the FIXED
    behavior (a ``RuntimeError`` naming the exit code and captured output) and
    is expected to fail against current code with a JSONDecodeError instead.
    """

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        # Deliberately do NOT touch the report path -- simulate pytest dying
        # before pytest-cov ever wrote a report.
        _extract_cov_report_path(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=17,
            stdout="",
            stderr="boom-case1-no-report-stderr",
        )

    with patch(
        "xtrax.devtools.gates.test_rigor.subprocess.run",
        side_effect=fake_run,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            run_pytest_coverage(root=tmp_path)

    message = str(exc_info.value)
    assert "17" in message
    assert "boom-case1-no-report-stderr" in message


def test_run_pytest_coverage_raises_distinct_runtime_error_when_report_unparsable(
    tmp_path: Path,
) -> None:
    """C4 case 2: subprocess exits zero but writes a zero-byte report.

    Distinct from the missing-report case: the report file genuinely exists
    (pytest-cov started writing it) but is empty/truncated and cannot be
    parsed as JSON. Against current code this also raises a bare
    ``json.JSONDecodeError`` because the empty pre-created temp file is
    indistinguishable from a partially-written one under the current
    ``cov_path.is_file()`` guard.
    """

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        cov_path = _extract_cov_report_path(cmd)
        # Explicitly write a zero-byte report -- present-but-unparsable,
        # distinct in intent from "never written" above.
        cov_path.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="5 passed in 0.1s",
            stderr="",
        )

    with patch(
        "xtrax.devtools.gates.test_rigor.subprocess.run",
        side_effect=fake_run,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            run_pytest_coverage(root=tmp_path)

    message = str(exc_info.value)
    assert "0" in message
    assert "5 passed in 0.1s" in message


def test_run_test_rigor_gate_fails_on_red_suite_despite_good_coverage(
    tmp_path: Path,
) -> None:
    """C4 case 3: a failing suite must fail the gate even at good coverage.

    Drives the real gate end-to-end (``run_pytest_coverage`` is NOT mocked
    out) with a stubbed subprocess that reports excellent coverage
    percentages but a non-zero exit code and failed tests. Against current
    code ``passed = passes_line and passes_branch`` ignores the suite result
    entirely, so this fails -- either because ``result.passed`` is
    (wrongly) True, or because ``GateResult`` has no ``failure_detail``
    field yet (an AttributeError on a frozen/slots dataclass is a legitimate
    red here too).
    """
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

    def fake_run(
        cmd: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        cov_path = _extract_cov_report_path(cmd)
        cov_path.write_text(
            json.dumps(
                {
                    "totals": {
                        "percent_covered": 99.0,
                        "percent_branches_covered": 99.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="....... [100%]\n3 failed, 7 passed in 2.1s",
            stderr="",
        )

    with patch(
        "xtrax.devtools.gates.test_rigor.subprocess.run",
        side_effect=fake_run,
    ):
        result = run_test_rigor_gate(
            audits_path=audits_path,
            baseline_path=baseline_path,
            root=tmp_path,
            write_baseline=False,
        )

    assert result.passed is False
    assert "1" in result.failure_detail
    assert "3" in result.failure_detail
