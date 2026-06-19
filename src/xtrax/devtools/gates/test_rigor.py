"""D7 test-rigor gate: pytest-cov line + branch coverage ratchet (N2.7 / #1587)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from xtrax.devtools.baseline import (
    DEFAULT_BASELINE_PATH,
    evaluate_metric,
    load_baseline,
    save_baseline,
    update_metric,
)
from xtrax.devtools.emit import append_finding, emit_metric_finding

LINE_METRIC = "test_rigor.line_coverage_pct"
BRANCH_METRIC = "test_rigor.branch_coverage_pct"
DIMENSION = "test_rigor"


@dataclass(frozen=True, slots=True)
class CoverageStats:
    line_pct: float
    branch_pct: float
    tests_run: int
    tests_failed: int


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    stats: CoverageStats
    line_coverage_pct: float
    branch_coverage_pct: float
    findings_emitted: int
    baseline_updated: bool
    line_metric_key: str = LINE_METRIC
    branch_metric_key: str = BRANCH_METRIC


def parse_coverage_json(path: Path) -> tuple[float, float]:
    """Parse pytest-cov JSON report totals -> (line_pct, branch_pct)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    totals = data["totals"]
    return float(totals["percent_covered"]), float(totals["percent_branches_covered"])


def parse_pytest_summary(output: str) -> tuple[int, int]:
    """Return (tests_run, tests_failed) from pytest -q summary line."""
    tests_run = 0
    tests_failed = 0
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if (
            " passed" not in stripped
            and " failed" not in stripped
            and " error" not in stripped
        ):
            continue
        passed_m = re.search(r"(\d+) passed", stripped)
        failed_m = re.search(r"(\d+) failed", stripped)
        error_m = re.search(r"(\d+) error", stripped)
        if passed_m:
            tests_run += int(passed_m.group(1))
        if failed_m:
            count = int(failed_m.group(1))
            tests_run += count
            tests_failed += count
        if error_m:
            count = int(error_m.group(1))
            tests_run += count
            tests_failed += count
        if passed_m or failed_m or error_m:
            break
    return tests_run, tests_failed


def run_pytest_coverage(
    root: Path,
    tests_path: Path | None = None,
) -> CoverageStats:
    """Run pytest with branch coverage JSON report under ``root``."""
    resolved_root = root.resolve()
    resolved_tests = (tests_path or Path("tests")).resolve()
    if not resolved_tests.is_absolute():
        resolved_tests = (resolved_root / resolved_tests).resolve()

    with tempfile.NamedTemporaryFile(
        suffix=".json",
        prefix="coverage-",
        delete=False,
    ) as handle:
        cov_path = Path(handle.name)

    env = {**os.environ, "PYTEST_ADDOPTS": ""}
    cmd = [
        "uv",
        "run",
        "pytest",
        str(resolved_tests),
        "--cov=xtrax",
        "--cov-branch",
        f"--cov-report=json:{cov_path}",
        "-q",
        "-o",
        "addopts=",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=resolved_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = f"{result.stdout}\n{result.stderr}"
        tests_run, tests_failed = parse_pytest_summary(combined)
        if not cov_path.is_file():
            msg = (
                "pytest-cov JSON report missing; "
                f"pytest exit={result.returncode}: {combined.strip()}"
            )
            raise RuntimeError(msg)
        line_pct, branch_pct = parse_coverage_json(cov_path)
    finally:
        cov_path.unlink(missing_ok=True)

    return CoverageStats(
        line_pct=line_pct,
        branch_pct=branch_pct,
        tests_run=tests_run,
        tests_failed=tests_failed,
    )


def run_test_rigor_gate(
    audits_path: Path,
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    root: Path | None = None,
    *,
    tests_path: Path | None = None,
    run_id: str | None = None,
    write_baseline: bool = True,
) -> GateResult:
    """Run pytest coverage, emit findings, evaluate dual baseline ratchet."""
    resolved_root = root or Path.cwd()
    stats = run_pytest_coverage(resolved_root, tests_path=tests_path)
    line_pct = stats.line_pct
    branch_pct = stats.branch_pct

    emitted = 0
    record = emit_metric_finding(
        dim=DIMENSION,
        severity="info",
        file_line=str(resolved_root / "src" / "xtrax"),
        evidence=(
            f"line coverage {line_pct:.1f}%, branch coverage {branch_pct:.1f}% "
            f"({stats.tests_run} tests, {stats.tests_failed} failed)"
        ),
        rule_id="test_rigor.coverage",
        symbol_qualname="",
        payload={
            "violation_kind": "coverage",
            "line_coverage_pct": line_pct,
            "branch_coverage_pct": branch_pct,
            "tests_run": stats.tests_run,
            "tests_failed": stats.tests_failed,
        },
        run_id=run_id,
    )
    append_finding(record, audits_path=audits_path)
    emitted += 1

    baseline = load_baseline(path=baseline_path)
    passes_line, update_line = evaluate_metric(baseline, LINE_METRIC, line_pct)
    passes_branch, update_branch = evaluate_metric(baseline, BRANCH_METRIC, branch_pct)
    passed = passes_line and passes_branch

    baseline_updated = False
    if passed and write_baseline and (update_line or update_branch):
        tightened = baseline
        if update_line:
            tightened = update_metric(tightened, LINE_METRIC, line_pct, "maximize")
        if update_branch:
            tightened = update_metric(
                tightened,
                BRANCH_METRIC,
                branch_pct,
                "maximize",
            )
        save_baseline(tightened, path=baseline_path)
        baseline_updated = True

    return GateResult(
        passed=passed,
        stats=stats,
        line_coverage_pct=line_pct,
        branch_coverage_pct=branch_pct,
        findings_emitted=emitted,
        baseline_updated=baseline_updated,
    )
