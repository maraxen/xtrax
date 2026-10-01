#!/usr/bin/env python3
"""Rename SafeMap -> ChunkedMap in Markdown prose and code blocks (xtrax #3644).

ast-grep's rules.yml covers Python structurally; Markdown has no Python syntax tree,
so this does whole-word replacement instead:

    SafeMapIterator -> ChunkedMapIterator,  SafeMap -> ChunkedMap,  safe_map -> chunked_map

A line that mentions JAX's own, unrelated `safe_map` (`util.safe_map`, "JAX's safe_map")
is left untouched. Pass files or directories; directories are searched for *.md. Keep
CHANGELOG history out of the arguments: past entries describe what shipped then.

    python codemods/safemap-to-chunkedmap/rename_markdown.py docs README.md
    python codemods/safemap-to-chunkedmap/rename_markdown.py --check docs   # exit 1 if changes
"""

import argparse
import re
import sys
from pathlib import Path

_WORDS = [
    (re.compile(r"\bSafeMapIterator\b"), "ChunkedMapIterator"),
    (re.compile(r"\bSafeMap\b"), "ChunkedMap"),
    (re.compile(r"\bsafe_map\b"), "chunked_map"),
]
_JAX_OWN = re.compile(r"util\.safe_map|(?i:jax('s)?\s+(own\s+)?safe_map)")


def rename_text(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        if not _JAX_OWN.search(line):
            for pattern, new in _WORDS:
                line = pattern.sub(new, line)
        out.append(line)
    return "".join(out)


def _files(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*.md") if "_build" not in p.parts))
        elif path.suffix == ".md":
            found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="report, do not write")
    args = parser.parse_args(argv)
    changed = []
    for path in _files(args.paths):
        before = path.read_text(encoding="utf-8")
        after = rename_text(before)
        if after != before:
            changed.append(path)
            if not args.check:
                path.write_text(after, encoding="utf-8")
    for path in changed:
        print(("would rename in " if args.check else "renamed in ") + str(path))
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
