#!/usr/bin/env python3
"""Public dataclass field-shape diff gate CLI (#5090).

Fails when a shipped public dataclass-like class loses, renames or retypes a field,
loses a default, or gains a required field since the merge-base, and CHANGELOG.md's
added lines do not name the class. See xtrax.devtools.gates.dataclass_shape_diff.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xtrax.devtools.gates.added_types_diff import DEFAULT_TARGET
from xtrax.devtools.gates.dataclass_shape_diff import run_dataclass_shape_gate

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="Repository root")
    parser.add_argument(
        "--target", type=Path, default=DEFAULT_TARGET, help="Package subtree (default: src/xtrax)"
    )
    parser.add_argument("--base", default=None, help="Explicit merge-base SHA (default: auto)")
    args = parser.parse_args(argv)

    result = run_dataclass_shape_gate(
        args.repo_root.resolve(), target=args.target, merge_base=args.base
    )
    if result.status == "skip":
        print(f"SKIP: dataclass shape diff gate -- {result.skip_reason}", file=sys.stderr)
        return 0

    print(
        f"merge-base={result.merge_base} files_checked={result.files_checked} "
        f"shape_changes={len(result.changes)}"
    )
    for change in result.changes:
        if change not in result.unacknowledged:
            print(f"  acknowledged in CHANGELOG: {change.render()}")
    if result.status == "fail":
        print(
            "FAIL: public dataclass shape changed without a CHANGELOG.md entry naming the "
            "class (consumers break on attribute access or construction):",
            file=sys.stderr,
        )
        for change in result.unacknowledged:
            print(f"  - {change.render()}", file=sys.stderr)
        return 1

    print("PASS: dataclass shape diff gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
