#!/usr/bin/env python3
"""N3.1 audit bootstrap CLI — orchestrate dimension gates (#1592)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.baseline import DEFAULT_BASELINE_PATH
from xtrax.devtools.bootstrap import run_audit_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Baseline JSON for dimension ratchets",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--no-write-baseline",
        action="store_true",
        help="Evaluate only; do not tighten baseline on improvement",
    )
    parser.add_argument(
        "--full-test-rigor",
        action="store_true",
        help="Run full tests/ tree for test_rigor (default: tests/audit only)",
    )
    args = parser.parse_args(argv)

    result = run_audit_bootstrap(
        audits_path=args.audits_path,
        baseline_path=args.baseline_path,
        root=args.root,
        write_baseline=not args.no_write_baseline,
        test_rigor_quick=not args.full_test_rigor,
        run_id=str(uuid.uuid4()),
    )

    status = "PASS" if result.passed else "FAIL"
    failed = [run.dimension for run in result.runs if not run.passed]
    print(
        f"{status}: audit bootstrap all_passed={result.passed} "
        f"manifest={result.manifest_path}"
    )
    if failed:
        print(f"failed dimensions: {', '.join(failed)}", file=sys.stderr)
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
