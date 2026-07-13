#!/usr/bin/env python3
"""T1-09 IR-schema single-source grep-gate (#3064): no second hand-authored JSON Schema.

`xtrax.inference.ir_schema.emit_ir_schema` is the single source of truth for the composition
IR JSON Schema, derived live from the real E1 types. A hand-authored, parallel JSON Schema
definition anywhere else in `src/xtrax` would recreate the "two competing graph formats" risk
the design spec rejected a standalone shared schema package over -- this gate fails loud if the
`"$schema"` JSON-Schema marker shows up outside the one allowed emitter module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC_DIR = ROOT / "src" / "xtrax"
DEFAULT_ALLOWED_PATH = DEFAULT_SRC_DIR / "inference" / "ir_schema.py"
SCHEMA_MARKER = '"$schema"'


def find_schema_definitions(src_dir: Path) -> list[str]:
    """Return "path:line" hits for SCHEMA_MARKER under src_dir."""
    hits: list[str] = []
    if not src_dir.is_dir():
        return hits
    for path in sorted(src_dir.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if SCHEMA_MARKER in line:
                hits.append(f"{path}:{lineno}")
    return hits


def audit_ir_schema_single_source(src_dir: Path, allowed_path: Path) -> tuple[bool, list[str]]:
    hits = find_schema_definitions(src_dir)
    disallowed = [hit for hit in hits if not hit.startswith(f"{allowed_path}:")]
    return len(disallowed) == 0, disallowed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help="Package subtree to scan (default: src/xtrax)",
    )
    parser.add_argument(
        "--allowed-path",
        type=Path,
        default=DEFAULT_ALLOWED_PATH,
        help="The single file permitted to define '\"$schema\"' (default: ir_schema.py)",
    )
    args = parser.parse_args(argv)

    passed, hits = audit_ir_schema_single_source(args.src_dir, args.allowed_path)
    if passed:
        print("PASS: IR-schema single-source gate (0 disallowed '\"$schema\"' hits)")
        return 0

    print("FAIL: IR-schema single-source gate", file=sys.stderr)
    print(
        f"  '\"$schema\"' found outside the single allowed source ({args.allowed_path}):",
        file=sys.stderr,
    )
    for hit in hits:
        print(f"  - {hit}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
