#!/usr/bin/env python3
"""Grep src/xtrax/ for UI/chain-map/plugin-state identifiers (T1-14, AC6).

Enforces the compiler_boundary invariant (FIXED constraint #1,
`.praxia/docs/specs/260702_design-2174-next-slices-minimal-composit.md:13`): no UI/chain-map/
plugin-state identifiers may leak into `src/xtrax` core while the chain-map visual UI epic (E4,
deferred) remains undesigned. No canonical forbidden-identifier list exists in any doc, so the
FORBIDDEN_PATTERNS below are a grounded judgment call, sourced from the invariant's own vocabulary
and the research synthesis Exclusions list ("chain-map visual UI + MathJax rendering") — revisable
once E4 is actually designed.

Known limitation: the `chain_map`/`chain-map` pattern would also match a legitimate future
`from collections import ChainMap` (stdlib dict-chaining, unrelated to UI). No such usage exists
today. If one is ever needed, use a local import alias, or extend this script's ALLOWLIST.

Boundary semantics: word boundaries are checked manually (not via regex `\b`) so that `_`, `-`,
digits, and case transitions all count as valid identifier-component separators (matching real
Python/compound-identifier conventions -- `chain_map_view`, `ChainMapWidget`, and mid-identifier
camelCase compounds like `someChainMapValue`/`getPluginStateNow` all count as hits), while an
immediately-adjacent LOWERCASE letter continuing the SAME case run does not (rejecting same-word
extensions like `chain_mapper` or `plugin_statement`, which are different words, not the
forbidden term).

Scope: scans both `.py` and `.toml` files under the target directory. `src/xtrax/` currently has
exactly one non-`.py`, non-marker tracked file type worth scanning
(`node_metadata_schema.toml`), confirmed by `git ls-files src/xtrax`. This is a deliberate,
bounded scope choice, not an oversight -- other file types are not currently expected to define
identifiers relevant to this invariant.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "xtrax"

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("chain_map/chain-map", re.compile(r"chain[_-]?map", re.IGNORECASE)),
    ("mathjax/math_jax", re.compile(r"math[_-]?jax", re.IGNORECASE)),
    ("plugin_state/plugin-state", re.compile(r"plugin[_-]?state", re.IGNORECASE)),
)

# (repo-relative path, 1-indexed line number): known, deliberate prose references to the
# deferred chain-map/plugin-state epic in docstrings/comments describing the compiler_boundary
# invariant itself -- not an actual UI identifier leaking into src/xtrax core. Shrink-only:
# `test_allowlist_entries_are_still_real_matches` fails loud if an entry no longer corresponds
# to real text (stale allowlist), and any NEW hit not in this set still fails the gate.
ALLOWLIST: frozenset[tuple[str, int]] = frozenset(
    {
        ("src/xtrax/inference/ir_schema.py", 5),
        # `mathjax_label` is a real, already-shipped metadata slot NAME (T1-06, #3059,
        # node_metadata_schema.toml:16) -- a data field describing content type (the string
        # holds MathJax-flavored markup), not UI rendering code/logic in src/xtrax core. This
        # docstring line documents that data schema, it doesn't implement UI. Whether the slot
        # NAME itself should be UI-technology-agnostic (e.g. `formatted_label`) is a separate,
        # out-of-scope design question flagged for follow-up, not fixed here.
        ("src/xtrax/composition/node_metadata.py", 4),
        # The canonical definition of the same `mathjax_label` slot, in the schema TOML itself.
        # Same reasoning as above: a data-schema slot name, not UI code.
        ("src/xtrax/composition/node_metadata_schema.toml", 16),
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    path: str
    line_number: int
    pattern_label: str
    line_text: str


def _has_word_boundary(line: str, start: int, end: int) -> bool:
    # A lowercase char immediately before/after continues the SAME word (reject) unless it's a
    # camelCase transition -- lowercase followed by the match's own leading/trailing uppercase
    # letter counts as a valid component separator (e.g. `someChainMapValue`, `ChainMapValue`).
    before_ok = start == 0 or not line[start - 1].islower() or line[start].isupper()
    after_ok = end == len(line) or not line[end].islower()
    return before_ok and after_ok


def scan(
    target: Path, *, root: Path = ROOT, allowlist: frozenset[tuple[str, int]] = ALLOWLIST
) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    paths = sorted({*target.rglob("*.py"), *target.rglob("*.toml")})
    for path in paths:
        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if (rel, line_number) in allowlist:
                continue
            for label, pattern in FORBIDDEN_PATTERNS:
                for match in pattern.finditer(line):
                    if not _has_word_boundary(line, match.start(), match.end()):
                        continue
                    violations.append(
                        BoundaryViolation(
                            path=rel,
                            line_number=line_number,
                            pattern_label=label,
                            line_text=line.strip(),
                        )
                    )
                    break
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=str(DEFAULT_TARGET),
        help="Directory to scan for forbidden identifiers (default: src/xtrax)",
    )
    args = parser.parse_args(argv)

    target = Path(args.target)
    violations = scan(target, root=ROOT)

    if violations:
        for v in violations:
            print(
                f"{v.path}:{v.line_number}: [{v.pattern_label}] {v.line_text}",
                file=sys.stderr,
            )
        print(f"FAIL: {len(violations)} compiler_boundary violation(s)", file=sys.stderr)
        return 1

    print("PASS: compiler_boundary invariant holds (zero UI/chain-map/plugin-state hits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
