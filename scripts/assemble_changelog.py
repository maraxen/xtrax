#!/usr/bin/env python3
"""Assemble ``changelog.d/`` fragments into a ``CHANGELOG.md`` release section.

Feature PRs add one file per change under ``changelog.d/`` instead of editing
``## [Unreleased]`` (every PR that did conflicted with every other). The release PR
runs::

    python scripts/assemble_changelog.py --version 0.4.1b5 --date 2026-11-02

which merges every fragment plus any bullets already under ``## [Unreleased]`` into a
new ``## [0.4.1b5] - 2026-11-02`` section directly beneath an emptied
``## [Unreleased]``, grouped by Keep a Changelog category, then deletes the fragments.
``--check`` validates the fragments only (CI runs it on every PR).

Fragment file name: ``<slug>.<category>.md`` with category one of
``added changed deprecated removed fixed security``. Content: Markdown exactly as it
should appear under that category's heading.

Exit status: 0 on success, 1 on any invalid fragment, missing ``## [Unreleased]``,
duplicate version, or nothing to release.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
FRAGMENTS_DIR = ROOT / "changelog.d"

# Keep a Changelog order.
CATEGORIES = ("added", "changed", "deprecated", "removed", "fixed", "security")
# Files in the fragments directory that are not fragments.
_IGNORED = frozenset({"README.md", ".gitkeep"})
_SLUG = r"[A-Za-z0-9][A-Za-z0-9_-]*"
_FRAGMENT_RE = re.compile(rf"^(?P<slug>{_SLUG})\.(?P<category>[a-z]+)\.md$")
_UNRELEASED_RE = re.compile(r"^## \[Unreleased\][ \t]*$", re.MULTILINE)
_H2_RE = re.compile(r"^## ", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^#{1,3}\s")
_HTML_COMMENT_RE = re.compile(r"^\s*<!--.*-->\s*$")
_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+!_-]*$")
UNRELEASED_NOTE = "<!-- new entries: add a file under changelog.d/ -->"


class ChangelogError(Exception):
    """Raised for any condition that must stop the run with a message."""


@dataclass(frozen=True)
class Fragment:
    """One validated ``changelog.d`` file."""

    path: Path
    category: str
    text: str


def list_problems(fragments_dir: Path) -> list[str]:
    """Return one message per invalid file in *fragments_dir* (empty when all are fine)."""
    problems: list[str] = []
    if not fragments_dir.is_dir():
        return [f"{fragments_dir}: fragments directory does not exist"]
    for path in sorted(fragments_dir.iterdir()):
        if path.name in _IGNORED:
            continue
        if not path.is_file():
            problems.append(f"{path.name}: not a file")
            continue
        match = _FRAGMENT_RE.match(path.name)
        if match is None:
            problems.append(
                f"{path.name}: name must be <slug>.<category>.md "
                f"(slug of letters, digits, '-', '_'; category one of {', '.join(CATEGORIES)})"
            )
            continue
        if match["category"] not in CATEGORIES:
            problems.append(
                f"{path.name}: unknown category {match['category']!r}; "
                f"use one of {', '.join(CATEGORIES)}"
            )
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            problems.append(f"{path.name}: file is empty")
            continue
        for line in text.splitlines():
            if _HEADING_LINE_RE.match(line):
                problems.append(
                    f"{path.name}: contains a heading line {line!r}; "
                    "write bullet lines only, the category comes from the file name"
                )
                break
    return problems


def load_fragments(fragments_dir: Path) -> list[Fragment]:
    """Load valid fragments sorted by file name; raise :class:`ChangelogError` otherwise."""
    problems = list_problems(fragments_dir)
    if problems:
        raise ChangelogError("invalid changelog fragments:\n  " + "\n  ".join(problems))
    fragments = []
    for path in sorted(fragments_dir.iterdir()):
        match = _FRAGMENT_RE.match(path.name)
        if path.name in _IGNORED or match is None:
            continue
        fragments.append(
            Fragment(path, match["category"], path.read_text(encoding="utf-8").strip())
        )
    return fragments


def _split_unreleased(text: str) -> tuple[int, int, dict[str, str]]:
    """Locate the Unreleased section; return (start, end, {category: body}).

    ``start`` is the offset of the ``## [Unreleased]`` heading and ``end`` the offset of
    the next ``## `` heading (or end of file). Bodies are verbatim, stripped.
    """
    match = _UNRELEASED_RE.search(text)
    if match is None:
        raise ChangelogError("CHANGELOG.md has no '## [Unreleased]' heading")
    nxt = _H2_RE.search(text, match.end())
    end = nxt.start() if nxt else len(text)
    body = text[match.end() : end]
    parts = re.split(r"^### (.+?)[ \t]*$", body, flags=re.MULTILINE)
    preamble = "\n".join(
        line for line in parts[0].splitlines() if not _HTML_COMMENT_RE.match(line)
    ).strip()
    if preamble:
        raise ChangelogError(
            "text directly under '## [Unreleased]' belongs to no category; "
            "move it under a '### <Category>' heading first:\n" + preamble
        )
    existing: dict[str, str] = {}
    for title, section in zip(parts[1::2], parts[2::2], strict=True):
        category = title.strip().lower()
        if category not in CATEGORIES:
            raise ChangelogError(
                f"unknown category '### {title.strip()}' under '## [Unreleased]'; "
                f"expected one of {', '.join(c.capitalize() for c in CATEGORIES)}"
            )
        if category in existing:
            raise ChangelogError(f"'### {title.strip()}' appears twice under '## [Unreleased]'")
        existing[category] = section.strip()
    return match.start(), end, existing


def _join_blocks(blocks: list[str]) -> str:
    """Join non-empty blocks; consecutive bullet blocks stay one tight list."""
    out = ""
    for block in blocks:
        if not block:
            continue
        if not out:
            out = block
            continue
        both_bullets = out.splitlines()[-1].lstrip().startswith(
            ("- ", "* ")
        ) and block.lstrip().startswith(("- ", "* "))
        out += ("\n" if both_bullets else "\n\n") + block
    return out


def assemble(text: str, fragments: list[Fragment], version: str, date: str) -> str:
    """Return the new CHANGELOG text with a ``## [version] - date`` section cut."""
    if not _VERSION_RE.match(version):
        raise ChangelogError(f"--version {version!r} does not look like a version")
    try:
        datetime.date.fromisoformat(date)
    except ValueError as exc:
        raise ChangelogError(f"--date {date!r} is not YYYY-MM-DD") from exc
    if re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE):
        raise ChangelogError(f"CHANGELOG.md already has a '## [{version}]' section")
    start, end, existing = _split_unreleased(text)

    merged: dict[str, str] = {}
    for category in CATEGORIES:
        blocks = [existing.get(category, "")]
        blocks += [f.text for f in fragments if f.category == category]
        body = _join_blocks(blocks)
        if body:
            merged[category] = body
    if not merged:
        raise ChangelogError(
            "nothing to release: no changelog.d fragments and '## [Unreleased]' is empty"
        )

    section = [f"## [{version}] - {date}", ""]
    for category in CATEGORIES:
        if category in merged:
            section += [f"### {category.capitalize()}", "", merged[category], ""]
    new_unreleased = f"## [Unreleased]\n\n{UNRELEASED_NOTE}\n\n"
    return text[:start] + new_unreleased + "\n".join(section) + "\n" + text[end:]


def run_assemble(
    changelog: Path, fragments_dir: Path, version: str, date: str
) -> tuple[str, list[Fragment]]:
    """Validate, assemble and write ``changelog``; delete the consumed fragments."""
    fragments = load_fragments(fragments_dir)
    new_text = assemble(changelog.read_text(encoding="utf-8"), fragments, version, date)
    changelog.write_text(new_text, encoding="utf-8")
    for fragment in fragments:
        fragment.path.unlink()
    return new_text, fragments


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="validate fragments only")
    parser.add_argument("--version", help="release version for the new section")
    parser.add_argument("--date", help="release date, YYYY-MM-DD")
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    parser.add_argument("--fragments-dir", type=Path, default=FRAGMENTS_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    if args.check:
        if args.version or args.date:
            parser.error("--check does not take --version/--date")
        problems = list_problems(args.fragments_dir)
        for problem in problems:
            logger.error("%s", problem)
        if problems:
            return 1
        logger.info("changelog fragments OK")
        return 0
    if not (args.version and args.date):
        parser.error("give --check, or both --version and --date")
    try:
        _, fragments = run_assemble(args.changelog, args.fragments_dir, args.version, args.date)
    except ChangelogError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "wrote ## [%s] - %s from %d fragment(s); deleted them",
        args.version,
        args.date,
        len(fragments),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
