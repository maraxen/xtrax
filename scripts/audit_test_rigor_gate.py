#!/usr/bin/env python3
"""D7 test-rigor gate CLI — pytest-cov coverage% + baseline ratchet (N2.7 / #1587)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.baseline import DEFAULT_BASELINE_PATH
from xtrax.devtools.gates.test_rigor import run_test_rigor_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS_PATH = ROOT / "tests"
QUICK_TESTS_PATH = ROOT / "tests" / "audit"
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run tests/audit only (fast CI recipe)",
    )
    parser.add_argument(
        "--tests-path",
        type=Path,
        default=None,
        help="Override pytest path (default: tests/ or tests/audit with --quick)",
    )
    parser.add_argument(
        "--audits-path",
        type=Path,
        default=DEFAULT_AUDITS_PATH,
        help="JSONL path for emitted findings",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Baseline JSON for test_rigor coverage metrics",
    )
    parser.add_argument(
        "--no-write-baseline",
        action="store_true",
        help="Evaluate only; do not tighten baseline on improvement",
    )
    args = parser.parse_args(argv)

    if args.tests_path is not None:
        tests_path = args.tests_path
    elif args.quick:
        tests_path = QUICK_TESTS_PATH
    else:
        tests_path = DEFAULT_TESTS_PATH

    if not tests_path.exists():
        print(f"tests path not found: {tests_path}", file=sys.stderr)
        return 2

    result = run_test_rigor_gate(
        audits_path=args.audits_path,
        baseline_path=args.baseline_path,
        root=ROOT,
        tests_path=tests_path,
        run_id=str(uuid.uuid4()),
        write_baseline=not args.no_write_baseline,
    )

    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status}: test_rigor.line_coverage_pct={result.line_coverage_pct:.1f} "
        f"test_rigor.branch_coverage_pct={result.branch_coverage_pct:.1f} "
        f"(tests_run={result.stats.tests_run}, "
        f"tests_failed={result.stats.tests_failed}, "
        f"findings_emitted={result.findings_emitted}, "
        f"baseline_updated={result.baseline_updated})"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
