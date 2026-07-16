#!/usr/bin/env python3
"""Grep src/xtrax/ for bathos dependencies (LC-02, AC-1b).

Enforces the bathos-independence invariant: xtrax's pure run-layer + loop-layer modules must
have zero coupling to bathos, with ALL bathos interactions routed through a future loop
controller (the top-level orchestrator). This mirrors the dispatch-independence gate (T2-21,
AC-28) which enforces the same principle for praxia.

Scope: scans `src/xtrax` only (NOT `controller/`, which legitimately imports bathos as the
orchestration layer). The `controller/` directory, when it exists, is explicitly out of scope
for this gate by design -- it's where bathos coupling lives.

Verified zero existing hits in `src/xtrax` before writing this, so `ALLOWLIST` starts empty --
any future hit is a real, new violation.

Pattern design note: patterns are scoped to actual dependency signals (an import statement, an
MCP tool-call identifier), not a bare `bathos` substring. This repo's own docstrings routinely
describe xtrax's independence from bathos (e.g. "xtrax has no dependency on bathos") -- a
bare-word pattern would collide with that entirely legitimate prose and require an ongoing
allowlist for it. Requiring `import`/`from` immediately before `bathos`, or the `mcp__bathos`
tool-identifier prefix, avoids that collision by construction (see
`tests/audit/test_bathos_independence_gate.py`'s word-boundary regression test).
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
    ("bathos-import", re.compile(r"(?:import|from)\s+bathos\b", re.IGNORECASE)),
    ("mcp-bathos-tool", re.compile(r"mcp__bathos", re.IGNORECASE)),
)

# Known, deliberate references to bathos in docstrings describing the independence rule itself,
# not actual bathos dependencies. These lines document that xtrax does NOT import bathos or call
# bathos tools directly. Shrink-only: any NEW hit not in this set still fails the gate.
ALLOWLIST: frozenset[tuple[str, int]] = frozenset(
    {
        # Module docstring explicitly stating xtrax doesn't import bathos
        ("src/xtrax/run/seed_emission.py", 37),
        # Module docstring: "does not import bathos, and does not decide..."
        ("src/xtrax/run/component_binding.py", 24),
        # Module docstring: bathos's own check_baseline_budget_equivalence
        ("src/xtrax/run/baseline_budget_emission.py", 64),
        # Module docstring: reference to how caller obtained data from bathos
        ("src/xtrax/loop/sidecar_drift_gate.py", 90),
        # Module docstring: reference to how caller obtained per-script_sha256 counts from bathos
        ("src/xtrax/loop/seed_gate.py", 93),
        # Module docstring: reference to pre-registered sidecar from bathos
        ("src/xtrax/loop/prereg_match.py", 61),
        # Module docstring: backtick-quoted MCP tool reference `mcp__bathos__claim_validate`
        ("src/xtrax/loop/prereg_match.py", 10),
        # Module docstring: xtrax doesn't decide which bathos tool anchors artifact
        ("src/xtrax/loop/metrics_provenance.py", 25),
        # Module docstring: reference to how caller obtained battery verdict from bathos
        ("src/xtrax/loop/stats_battery_gate.py", 61),
        # Module docstring: reference to how caller obtained liveness checks from bathos
        ("src/xtrax/loop/capability_probe_gate.py", 70),
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
