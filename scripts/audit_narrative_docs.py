#!/usr/bin/env python3
"""Distribution N4b narrative docs gate — quickstart + architecture prose (#1458)."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "narrative_docs.toml"


@dataclass(frozen=True)
class NarrativePage:
    path: str
    min_bytes: int
    required_markers: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeDocsConfig:
    version: str
    min_page_bytes: int
    pages: tuple[NarrativePage, ...]
    forbidden_phrases: tuple[str, ...]


def load_narrative_docs_config(config_path: Path) -> NarrativeDocsConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    narrative = data.get("narrative")
    if not isinstance(narrative, dict):
        raise ValueError(f"missing [narrative] section in {config_path}")

    version = narrative.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("narrative.version must be a non-empty string")

    min_page_bytes = narrative.get("min_page_bytes")
    if not isinstance(min_page_bytes, int) or min_page_bytes <= 0:
        raise ValueError("narrative.min_page_bytes must be a positive integer")

    raw_pages = data.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("narrative docs must define at least one [[pages]] entry")

    pages: list[NarrativePage] = []
    for raw in raw_pages:
        if not isinstance(raw, dict):
            raise ValueError("each [[pages]] entry must be a table")
        path = raw.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("pages[].path must be a non-empty string")
        min_bytes = raw.get("min_bytes", min_page_bytes)
        if not isinstance(min_bytes, int) or min_bytes <= 0:
            raise ValueError(f"pages[{path!r}].min_bytes must be a positive integer")
        markers = raw.get("required_markers")
        if not isinstance(markers, list) or not markers:
            raise ValueError(f"pages[{path!r}].required_markers must be a non-empty list")
        if not all(isinstance(item, str) and item for item in markers):
            raise ValueError(f"pages[{path!r}].required_markers must contain strings")
        pages.append(
            NarrativePage(
                path=path,
                min_bytes=min_bytes,
                required_markers=tuple(str(item) for item in markers),
            )
        )

    forbidden_section = data.get("forbidden", {})
    if not isinstance(forbidden_section, dict):
        raise ValueError("missing [forbidden] section")
    phrases = forbidden_section.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise ValueError("forbidden.phrases must be a non-empty list")

    return NarrativeDocsConfig(
        version=version,
        min_page_bytes=min_page_bytes,
        pages=tuple(pages),
        forbidden_phrases=tuple(str(item) for item in phrases),
    )


def audit_narrative_docs(
    root: Path,
    config_path: Path,
) -> tuple[bool, list[str]]:
    config = load_narrative_docs_config(config_path)
    failures: list[str] = []

    for page in config.pages:
        path = root / page.path
        if not path.is_file():
            failures.append(f"missing narrative page: {page.path}")
            continue
        text = path.read_text(encoding="utf-8")
        size = path.stat().st_size
        if size < page.min_bytes:
            failures.append(f"{page.path} too small ({size} bytes < {page.min_bytes})")
        for marker in page.required_markers:
            if marker not in text:
                failures.append(f"{page.path} missing marker: {marker!r}")

    for rel in (page.path for page in config.pages):
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in config.forbidden_phrases:
            if phrase in text:
                failures.append(f"{rel} contains forbidden phrase: {phrase!r}")

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to narrative_docs.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_narrative_docs(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
    )
    if passed:
        print("PASS: narrative docs gate")
        return 0

    print("FAIL: narrative docs gate", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
