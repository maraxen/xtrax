#!/usr/bin/env python3
"""Distribution N0 coverage hygiene gate — forbid committed coverage artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "coverage_hygiene.toml"


@dataclass(frozen=True)
class HygieneConfig:
    version: str
    required_gitignore_patterns: tuple[str, ...]
    forbidden_tracked_globs: tuple[str, ...]


def load_hygiene_config(config_path: Path) -> HygieneConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    hygiene = data.get("hygiene")
    if not isinstance(hygiene, dict):
        raise ValueError(f"missing [hygiene] section in {config_path}")

    version = hygiene.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("hygiene.version must be a non-empty string")

    required = hygiene.get("required_gitignore_patterns")
    if not isinstance(required, list) or not required:
        raise ValueError("hygiene.required_gitignore_patterns must be a non-empty list")
    if not all(isinstance(item, str) and item for item in required):
        raise ValueError("hygiene.required_gitignore_patterns must contain strings")

    forbidden = hygiene.get("forbidden_tracked_globs")
    if not isinstance(forbidden, list) or not forbidden:
        raise ValueError("hygiene.forbidden_tracked_globs must be a non-empty list")
    if not all(isinstance(item, str) and item for item in forbidden):
        raise ValueError("hygiene.forbidden_tracked_globs must contain strings")

    return HygieneConfig(
        version=version,
        required_gitignore_patterns=tuple(required),
        forbidden_tracked_globs=tuple(forbidden),
    )


def iter_gitignore_lines(gitignore_path: Path) -> list[str]:
    if not gitignore_path.is_file():
        return []
    return gitignore_path.read_text(encoding="utf-8").splitlines()


def gitignore_contains_pattern(lines: list[str], pattern: str) -> bool:
    """Return True when pattern appears as a line or path suffix in .gitignore."""
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == pattern or line.endswith(pattern):
            return True
    return False


def check_gitignore_patterns(
    gitignore_path: Path,
    required_patterns: tuple[str, ...],
) -> list[str]:
    lines = iter_gitignore_lines(gitignore_path)
    return [
        pattern
        for pattern in required_patterns
        if not gitignore_contains_pattern(lines, pattern)
    ]


def path_matches_forbidden_glob(path: str, glob: str) -> bool:
    if glob.endswith("/*"):
        prefix = glob[:-2]
        return path == prefix or path.startswith(f"{prefix}/")
    return path == glob or path.endswith(f"/{glob}")


def list_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "git ls-files failed"
        raise RuntimeError(stderr)
    return [line for line in result.stdout.splitlines() if line]


def find_forbidden_tracked_paths(
    tracked_paths: list[str],
    forbidden_globs: tuple[str, ...],
) -> list[str]:
    violations: list[str] = []
    for path in tracked_paths:
        if any(path_matches_forbidden_glob(path, glob) for glob in forbidden_globs):
            violations.append(path)
    return sorted(violations)


def audit_coverage_hygiene(
    *,
    root: Path,
    config_path: Path,
    gitignore_path: Path | None = None,
) -> tuple[bool, list[str]]:
    config = load_hygiene_config(config_path)
    resolved_gitignore = (
        gitignore_path if gitignore_path is not None else root / ".gitignore"
    )

    failures: list[str] = []
    missing_patterns = check_gitignore_patterns(
        resolved_gitignore,
        config.required_gitignore_patterns,
    )
    for pattern in missing_patterns:
        failures.append(
            f".gitignore missing required coverage pattern: {pattern!r}"
        )

    try:
        tracked = list_tracked_files(root)
    except RuntimeError as exc:
        failures.append(str(exc))
        tracked = []

    forbidden = find_forbidden_tracked_paths(tracked, config.forbidden_tracked_globs)
    for path in forbidden:
        failures.append(f"tracked coverage artifact: {path}")

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to coverage_hygiene.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--gitignore",
        type=Path,
        default=None,
        help="Optional .gitignore path (default: <root>/.gitignore)",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_coverage_hygiene(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
        gitignore_path=args.gitignore.resolve() if args.gitignore else None,
    )
    if passed:
        print("PASS: coverage hygiene — no tracked artifacts; .gitignore contract ok")
        return 0

    print("FAIL: coverage hygiene", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
