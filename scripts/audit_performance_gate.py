#!/usr/bin/env python3
"""D4 performance gate CLI — trace count + recorded wall-time (N2.5 / #1584)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.baseline import DEFAULT_BASELINE_PATH
from xtrax.devtools.gates.performance import (
    DEFAULT_TARGETS_PATH,
    METRIC_KEY,
    WALL_TIME_METRIC_KEY,
    run_performance_gate,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets-path",
        type=Path,
        default=ROOT / DEFAULT_TARGETS_PATH,
        help="Path to performance_targets.toml",
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
        help="Baseline JSON for performance metrics",
    )
    parser.add_argument(
        "--no-write-baseline",
        action="store_true",
        help="Evaluate only; do not tighten baseline on improvement",
    )
    args = parser.parse_args(argv)

    targets_path = args.targets_path.resolve()
    if not targets_path.is_file():
        print(f"targets not found: {targets_path}", file=sys.stderr)
        return 2

    result = run_performance_gate(
        targets_path=targets_path,
        audits_path=args.audits_path,
        baseline_path=args.baseline_path,
        run_id=str(uuid.uuid4()),
        write_baseline=not args.no_write_baseline,
    )

    status = "PASS" if result.passed else "FAIL"
    wall = (
        f"{result.wall_time_median_ms:.3f}ms" if result.wall_time_median_ms is not None else "n/a"
    )
    print(
        f"{status}: {METRIC_KEY}={result.trace_violation_count} "
        f"{WALL_TIME_METRIC_KEY}={wall} "
        f"(findings_emitted={result.findings_emitted}, "
        f"baseline_updated={result.baseline_updated})"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
