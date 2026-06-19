#!/usr/bin/env python3
"""D3' added-types diff gate CLI — LibCST merge-base public annotation check (#1589)."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from xtrax.devtools.gates.added_types_diff import (
    DEFAULT_TARGET,
    run_added_types_diff_gate,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDITS_PATH = ROOT / ".praxia" / "audits.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="Package subtree to scan (default: src/xtrax)",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="Explicit merge-base SHA (default: auto-resolve)",
    )
    parser.add_argument(
        "--audits-path",
        type=Path,
        default=DEFAULT_AUDITS_PATH,
        help="JSONL path for emitted findings",
    )
    parser.add_argument(
        "--no-emit",
        action="store_true",
        help="Do not append findings to audits JSONL",
    )
    args = parser.parse_args(argv)

    result = run_added_types_diff_gate(
        args.repo_root.resolve(),
        target=args.target,
        merge_base=args.base,
        audits_path=None if args.no_emit else args.audits_path,
        run_id=str(uuid.uuid4()),
    )

    if result.status == "skip":
        print(
            f"SKIP: added-types diff gate — {result.skip_reason}",
            file=sys.stderr,
        )
        return 0

    print(
        f"merge-base={result.merge_base} "
        f"files_checked={result.files_checked} "
        f"callables_checked={result.callables_checked}"
    )
    if result.status == "fail":
        print("FAIL: added-types diff gate", file=sys.stderr)
        for violation in result.violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print("PASS: added-types diff gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
