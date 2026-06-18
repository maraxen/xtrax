#!/usr/bin/env python3
"""D1 correctness gate CLI — jaxlint JL counts + baseline ratchet (N2.1 / #1581)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.baseline import DEFAULT_BASELINE_PATH
from xtrax.devtools.gates.correctness import run_correctness_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "xtrax"
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Path to scan (default: src/xtrax)",
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
        help="Baseline JSON for correctness.jl_violation_count",
    )
    parser.add_argument(
        "--no-write-baseline",
        action="store_true",
        help="Evaluate only; do not tighten baseline on improvement",
    )
    args = parser.parse_args(argv)

    target = Path(args.target).resolve()
    if not target.exists():
        print(f"target not found: {target}", file=sys.stderr)
        return 2

    result = run_correctness_gate(
        target=target,
        audits_path=args.audits_path,
        baseline_path=args.baseline_path,
        root=ROOT,
        run_id=str(uuid.uuid4()),
        write_baseline=not args.no_write_baseline,
    )

    status = "PASS" if result.passed else "FAIL"
    print(
        f"{status}: correctness.jl_violation_count={result.violation_count} "
        f"(findings_emitted={result.findings_emitted}, "
        f"baseline_updated={result.baseline_updated})"
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
