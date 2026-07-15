#!/usr/bin/env python3
"""Grep src/xtrax/ for mock/test-double EvaluateFn seam imports (T2-04, AC-E2).

Enforces the no-stub sealed-seam-lock invariant
(`.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md`, T2-04, PM-3): the
`EvaluateFn`/`SealedEvaluatorRegistry` seam (T1-07, `src/xtrax/stages/evaluate.py`) must always
resolve to its real implementation, never a mock/fake/stub/dummy stand-in that quietly never
became load-bearing. `scripts/audit_wave1_load_bearing.py` (T2-02) checked this once, at the
wave-1 gate; this script is the standing, per-PR tripwire against a stand-in creeping in later.

Reuses the `scan()` word-boundary engine from `scripts/audit_compiler_boundary.py` (T1-14) rather
than reimplementing it -- see that module for the boundary-matching rules. No mock/stub
`Evaluate*` implementation exists anywhere in the repo today (verified before writing this gate),
so `ALLOWLIST` starts empty and any future hit is a real, new violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_compiler_boundary import ROOT, scan  # noqa: E402

DEFAULT_TARGET = ROOT / "src" / "xtrax"

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mock-evaluator", re.compile(r"mock[_-]?evaluat(?:e|es|ed|ing|or|ors|ion)", re.IGNORECASE)),
    ("fake-evaluator", re.compile(r"fake[_-]?evaluat(?:e|es|ed|ing|or|ors|ion)", re.IGNORECASE)),
    ("stub-evaluator", re.compile(r"stub[_-]?evaluat(?:e|es|ed|ing|or|ors|ion)", re.IGNORECASE)),
    ("dummy-evaluator", re.compile(r"dummy[_-]?evaluat(?:e|es|ed|ing|or|ors|ion)", re.IGNORECASE)),
    ("unittest-mock-import", re.compile(r"unittest\.mock", re.IGNORECASE)),
)

# No deliberate exceptions exist yet -- see module docstring.
ALLOWLIST: frozenset[tuple[str, int]] = frozenset()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Directory to scan for mock/test-double seam identifiers (default: src/xtrax)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target)
    violations = scan(target, root=ROOT, allowlist=ALLOWLIST, patterns=FORBIDDEN_PATTERNS)

    if violations:
        for v in violations:
            print(
                f"{v.path}:{v.line_number}: [{v.pattern_label}] {v.line_text}",
                file=sys.stderr,
            )
        print(f"FAIL: {len(violations)} sealed-seam violation(s)", file=sys.stderr)
        return 1

    print("PASS: sealed-seam invariant holds (zero mock/test-double EvaluateFn hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
