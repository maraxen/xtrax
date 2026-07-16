#!/usr/bin/env python3
r"""Grep src/xtrax/ for bathos dependencies (LC-02, AC-1b).

Enforces the bathos-independence invariant: xtrax's pure run-layer + loop-layer modules must
have zero coupling to bathos, with ALL bathos interactions routed through a future loop
controller (the top-level orchestrator). This mirrors the dispatch-independence gate (T2-21,
AC-28) which enforces the same principle for praxia.

Scope: scans `src/xtrax` only (NOT `controller/`, which legitimately imports bathos as the
orchestration layer). The `controller/` directory, when it exists, is explicitly out of scope
for this gate by design -- it's where bathos coupling lives.

Verified zero existing hits in `src/xtrax` before writing this -- `ALLOWLIST` holds exactly one
entry (a backtick-quoted MCP tool-name mention that genuinely contains the `mcp__bathos` substring
in doc prose, not an import), not empty; see the pattern design note below for why.

Pattern design note: patterns are scoped to actual dependency signals (a real Python import
statement, an MCP tool-call identifier), not a bare `bathos` substring. The `bathos-import`
pattern is anchored to logical line start (`^\s*(?:import|from)\s+bathos\b`, `re.MULTILINE`) --
matching genuine Python import syntax, which is always the first token on its logical line --
rather than "import/from immediately before bathos anywhere in the line". An unanchored version
was tried first and found (via adversarial review) to false-positive on this repo's own
docstrings describing xtrax's independence from bathos (e.g. "does not import bathos, by design"
literally contains the substring "import bathos"), which would have required a 9-entry allowlist
purely to route around English prose, not real dependencies. Anchoring to line start eliminates
that entire false-positive class while still catching every real import form (bare, dotted,
indented, aliased -- see `tests/audit/test_bathos_independence_gate.py`'s regression tests).
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
    (
        "bathos-import",
        re.compile(r"^\s*(?:import|from)\s+bathos\b", re.IGNORECASE | re.MULTILINE),
    ),
    ("mcp-bathos-tool", re.compile(r"mcp__bathos", re.IGNORECASE)),
)

# Known, deliberate references to bathos in docstrings that genuinely contain a substring the
# mcp-bathos-tool pattern matches (a backtick-quoted tool-name mention), not an actual bathos
# dependency. Shrink-only: any NEW hit not in this set still fails the gate.
ALLOWLIST: frozenset[tuple[str, int]] = frozenset(
    {
        # Module docstring: backtick-quoted MCP tool reference `mcp__bathos__claim_validate`
        ("src/xtrax/loop/prereg_match.py", 10),
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Directory to scan for bathos identifiers (default: src/xtrax)",
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
        print(f"FAIL: {len(violations)} bathos-independence violation(s)", file=sys.stderr)
        return 1

    print("PASS: bathos-independence invariant holds (zero bathos dependencies in src/xtrax)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
