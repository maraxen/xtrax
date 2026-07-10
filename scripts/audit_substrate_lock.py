#!/usr/bin/env python3
"""T3-02 substrate-lock grep-gate (#3022): no `action_mode: strict` in phase 1.

Locks port_validation (and every other xtrax workflow template) on the working
Claude-PCW dispatch path. A premature `action_mode: strict` local-model variant
cannot merge until the phase-2 MCP-reachability probe (T3-07/AC-V2) passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS_DIR = ROOT / "agent_assets" / "workflows"
FORBIDDEN_MARKER = "action_mode: strict"


def find_forbidden_marker(workflows_dir: Path) -> list[str]:
    """Return "path:line" hits for FORBIDDEN_MARKER under workflows_dir."""
    hits: list[str] = []
    if not workflows_dir.is_dir():
        return hits
    for path in sorted(workflows_dir.rglob("*.yaml")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if FORBIDDEN_MARKER in line:
                hits.append(f"{path}:{lineno}")
    return hits


def audit_substrate_lock(workflows_dir: Path) -> tuple[bool, list[str]]:
    hits = find_forbidden_marker(workflows_dir)
    return len(hits) == 0, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=DEFAULT_WORKFLOWS_DIR,
        help="Directory to scan for workflow YAML templates",
    )
    args = parser.parse_args(argv)

    passed, hits = audit_substrate_lock(args.workflows_dir)
    if passed:
        print("PASS: substrate-lock gate (0 'action_mode: strict' hits)")
        return 0

    print("FAIL: substrate-lock gate", file=sys.stderr)
    print(
        "  premature 'action_mode: strict' variant found "
        "(blocked until T3-07/AC-V2 MCP-reachability probe passes):",
        file=sys.stderr,
    )
    for hit in hits:
        print(f"  - {hit}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
