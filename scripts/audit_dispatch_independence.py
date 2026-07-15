#!/usr/bin/env python3
"""Grep src/xtrax/ for praxia-plugin-dispatch dependencies (T2-21, AC-28).

Enforces the dispatch-independence invariant
(`.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md`, T2-21; spec Fork
15/16 decision log, `.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md:54`):
the loop's effectful actions must route through the pure-xtrax run-layer + bathos-MCP, with zero
coupling to `praxia rig-run`. Fork 15/16's own reasoning is that coupling to praxia today would
make #2181 depend on two upstream praxia gaps (G1: rig-run's hardcoded 9-arm dispatch table; G2:
strict-mode tools are compiled Rust, so YAML flows can't add new ones) -- both tracked as
praxia-repo T3 items, explicitly out of scope for this MVP. The mandate doc's locked decision #4
already settles "zero praxia coupling for the MVP" -- this script is the standing tripwire that
keeps it true, not a decision it's making itself.

Scope: this covers AC-28's *static* half only -- "any praxia-plugin-dispatch dependency detected
-> CI exit 1" is inherently a static-detectability claim, and this reuses T1-14's grep-gate engine
the same way T2-04's `audit_sealed_seam.py` did. AC-28's other clause ("success: loop runs
end-to-end with praxia absent") is a runtime/integration claim with no controller to exercise yet
-- only `xtrax.loop.admission` (T2-01) and `xtrax.loop.closure_lock` (T2-05) exist today, both
narrow primitives whose own docstrings explicitly decline to invent controller/escalation logic
ahead of a real consumer. Building git-commit/reset + bathos dispatch code now, solely to give
AC-28 a runtime surface to gate, would repeat the mistake those modules both declined to make.
That half is deferred until an actual #2181 loop controller lands.

Verified zero existing hits in `src/xtrax` before writing this (`rg -rn "praxia|rig.run|rig_run"
src/xtrax`), so `ALLOWLIST` starts empty -- any future hit is a real, new violation.

Pattern design note: patterns are scoped to actual dependency signals (an import statement, an
MCP tool-call identifier, a rig-run identifier), not a bare `praxia` substring. This repo's own
docstrings routinely cite `.praxia/docs/...` paths (praxia's own internal-docs convention) --
a bare-word pattern would collide with that entirely legitimate prose and require an ongoing
allowlist for it. Requiring `import`/`from` immediately before `praxia`, or the `mcp__praxia`
tool-identifier prefix, avoids that collision by construction (see
`tests/audit/test_dispatch_independence_gate.py`'s word-boundary regression test).
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
    ("praxia-import", re.compile(r"(?:import|from)\s+praxia\b", re.IGNORECASE)),
    ("mcp-praxia-tool", re.compile(r"mcp__praxia", re.IGNORECASE)),
    ("rig-run-dependency", re.compile(r"rig[_-]run", re.IGNORECASE)),
)

# No deliberate exceptions exist yet -- see module docstring.
ALLOWLIST: frozenset[tuple[str, int]] = frozenset()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Directory to scan for praxia-plugin-dispatch identifiers (default: src/xtrax)",
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
        print(f"FAIL: {len(violations)} dispatch-independence violation(s)", file=sys.stderr)
        return 1

    print("PASS: dispatch-independence invariant holds (zero praxia-plugin-dispatch hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
