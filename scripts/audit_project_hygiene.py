#!/usr/bin/env python3
"""Distribution N8 project hygiene gate — README/CHANGELOG/CITATION (#1460)."""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "distribution" / "project_hygiene.toml"


def parse_init_version(init_path: Path, *, attribute: str = "__version__") -> str:
    source = init_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(init_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == attribute:
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                raise ValueError(
                    f"{init_path}:{node.lineno}: {attribute} must be a string literal"
                )
    raise ValueError(f"{attribute} assignment not found in {init_path}")


@dataclass(frozen=True)
class ProjectHygieneConfig:
    version: str
    version_source: str
    version_attribute: str
    forbidden_root_paths: tuple[str, ...]
    required_files: tuple[str, ...]
    min_readme_bytes: int
    readme_markers: tuple[str, ...]
    changelog_markers: tuple[str, ...]
    citation_keys: tuple[str, ...]
    pyproject_urls: tuple[str, ...]


def load_project_hygiene_config(config_path: Path) -> ProjectHygieneConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    hygiene = data.get("hygiene")
    if not isinstance(hygiene, dict):
        raise ValueError(f"missing [hygiene] section in {config_path}")

    version = hygiene.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError("hygiene.version must be a non-empty string")

    version_source = hygiene.get("version_source")
    if not isinstance(version_source, str) or not version_source:
        raise ValueError("hygiene.version_source must be a non-empty string")

    version_attribute = hygiene.get("version_attribute")
    if not isinstance(version_attribute, str) or not version_attribute:
        raise ValueError("hygiene.version_attribute must be a non-empty string")

    forbidden = hygiene.get("forbidden_root_paths", [])
    if not isinstance(forbidden, list):
        raise ValueError("hygiene.forbidden_root_paths must be a list")

    required = hygiene.get("required_files")
    if not isinstance(required, list) or not required:
        raise ValueError("hygiene.required_files must be a non-empty list")

    min_readme = hygiene.get("min_readme_bytes")
    if not isinstance(min_readme, int) or min_readme <= 0:
        raise ValueError("hygiene.min_readme_bytes must be a positive integer")

    def _markers(section: str) -> tuple[str, ...]:
        table = hygiene.get(section, {})
        if not isinstance(table, dict):
            raise ValueError(f"hygiene.{section} must be a table")
        values = table.get("markers") or table.get("keys") or table.get("required")
        if not isinstance(values, list) or not values:
            raise ValueError(f"hygiene.{section} list must be non-empty")
        return tuple(str(item) for item in values)

    readme_markers = _markers("readme_markers")
    changelog_markers = _markers("changelog_markers")

    citation_table = hygiene.get("citation_keys", {})
    if not isinstance(citation_table, dict):
        raise ValueError("hygiene.citation_keys must be a table")
    citation_keys = citation_table.get("keys")
    if not isinstance(citation_keys, list) or not citation_keys:
        raise ValueError("hygiene.citation_keys.keys must be a non-empty list")

    urls_table = hygiene.get("pyproject_urls", {})
    if not isinstance(urls_table, dict):
        raise ValueError("hygiene.pyproject_urls must be a table")
    pyproject_urls = urls_table.get("required")
    if not isinstance(pyproject_urls, list) or not pyproject_urls:
        raise ValueError("hygiene.pyproject_urls.required must be a non-empty list")

    return ProjectHygieneConfig(
        version=version,
        version_source=version_source,
        version_attribute=version_attribute,
        forbidden_root_paths=tuple(str(item) for item in forbidden),
        required_files=tuple(str(item) for item in required),
        min_readme_bytes=min_readme,
        readme_markers=readme_markers,
        changelog_markers=changelog_markers,
        citation_keys=tuple(str(item) for item in citation_keys),
        pyproject_urls=tuple(str(item) for item in pyproject_urls),
    )


def _parse_citation_version(citation_path: Path) -> str | None:
    text = citation_path.read_text(encoding="utf-8")
    match = re.search(r"^version:\s*['\"]?([^'\"\n]+)", text, flags=re.MULTILINE)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_citation_keys(citation_path: Path) -> set[str]:
    keys: set[str] = set()
    for line in citation_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith(" "):
            keys.add(line.split(":", 1)[0].strip())
    return keys


def audit_project_hygiene(
    root: Path,
    config_path: Path,
) -> tuple[bool, list[str]]:
    config = load_project_hygiene_config(config_path)
    failures: list[str] = []

    for rel in config.required_files:
        path = root / rel
        if not path.is_file():
            failures.append(f"missing required file: {rel}")

    for rel in config.forbidden_root_paths:
        if (root / rel).exists():
            failures.append(f"forbidden root path present: {rel}")

    readme_path = root / "README.md"
    if readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8")
        if readme_path.stat().st_size < config.min_readme_bytes:
            failures.append(
                f"README.md too small ({readme_path.stat().st_size} bytes)"
            )
        for marker in config.readme_markers:
            if marker not in readme_text:
                failures.append(f"README.md missing marker: {marker!r}")

    changelog_path = root / "CHANGELOG.md"
    if changelog_path.is_file():
        changelog_text = changelog_path.read_text(encoding="utf-8")
        for marker in config.changelog_markers:
            if marker not in changelog_text:
                failures.append(f"CHANGELOG.md missing marker: {marker!r}")

    citation_path = root / "CITATION.cff"
    init_path = root / config.version_source
    if citation_path.is_file() and init_path.is_file():
        present_keys = _parse_citation_keys(citation_path)
        missing_keys = [
            key for key in config.citation_keys if key not in present_keys
        ]
        if missing_keys:
            failures.append(
                "CITATION.cff missing keys: " + ", ".join(missing_keys)
            )
        package_version = parse_init_version(
            init_path,
            attribute=config.version_attribute,
        )
        citation_version = _parse_citation_version(citation_path)
        if citation_version is None:
            failures.append("CITATION.cff missing version field")
        elif citation_version != package_version:
            failures.append(
                "CITATION.cff version "
                f"{citation_version!r} != {config.version_attribute} "
                f"{package_version!r}"
            )

    pyproject_path = root / "pyproject.toml"
    if pyproject_path.is_file():
        data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        project = data.get("project", {})
        readme = project.get("readme")
        if readme != "README.md":
            failures.append("pyproject.toml project.readme must be README.md")
        urls = project.get("urls", {})
        if not isinstance(urls, dict):
            failures.append("pyproject.toml missing [project.urls]")
        else:
            for key in config.pyproject_urls:
                if key not in urls:
                    failures.append(f"pyproject.toml missing project.urls.{key}")

    return len(failures) == 0, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to project_hygiene.toml",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root",
    )
    args = parser.parse_args(argv)

    passed, failures = audit_project_hygiene(
        root=args.root.resolve(),
        config_path=args.config.resolve(),
    )
    if passed:
        print("PASS: project hygiene gate")
        return 0

    print("FAIL: project hygiene gate", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
