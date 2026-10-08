"""Praxia manifest lists every repo skill at the package version (#2597)."""

import re
import tomllib
from pathlib import Path

import pytest

import xtrax

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / ".praxia" / "manifest.toml"
SKILLS = ROOT / "agent_assets" / "skills"


def _skill_dirs() -> list[Path]:
    return sorted(
        path for path in SKILLS.iterdir() if path.is_dir() and (path / "SKILL.md").is_file()
    )


def _frontmatter(skill_md: Path) -> dict:
    text = skill_md.read_text(encoding="utf-8")
    end = text.find("\n---", 4)
    data = yaml.safe_load(text[4:end])
    assert isinstance(data, dict)
    return data


def manifest_drift(manifest_path: Path, repo_root: Path) -> list[str]:
    """Differences between a praxia manifest and this repo's skills.

    Empty when ``plugin.version`` equals ``xtrax.__version__`` and every
    ``agent_assets/skills/*/SKILL.md`` has a ``[[plugin.skills]]`` entry whose
    name, path, and triggers match the skill file.
    """
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    version = data["plugin"]["version"]
    if version != xtrax.__version__:
        problems.append(f"version {version} != {xtrax.__version__}")

    entries = data["plugin"].get("skills", [])
    by_name = {entry["name"]: entry for entry in entries}
    skills_root = repo_root / "agent_assets" / "skills"
    expected = {
        path.name
        for path in skills_root.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    }
    for name in sorted(expected - by_name.keys()):
        problems.append(f"missing skill {name}")
    for name in sorted(by_name.keys() - expected):
        problems.append(f"unexpected skill {name}")

    for name, entry in sorted(by_name.items()):
        if name not in expected:
            continue
        rel = f"agent_assets/skills/{name}/SKILL.md"
        if entry.get("path") != rel:
            problems.append(f"{name} path {entry.get('path')!r} != {rel!r}")
            continue
        triggers = list(_frontmatter(repo_root / rel).get("triggers") or [])
        if list(entry.get("triggers") or []) != triggers:
            problems.append(f"{name} triggers drift")
    return problems


def test_skills_are_discovered() -> None:
    assert _skill_dirs(), "no agent_assets/skills/*/SKILL.md found"


def test_repo_manifest_matches_package_and_skills() -> None:
    assert manifest_drift(MANIFEST, ROOT) == []


def test_tmp_copy_with_changed_version_drifts(tmp_path: Path) -> None:
    original = MANIFEST.read_text(encoding="utf-8")
    needle = f'version         = "{xtrax.__version__}"'
    assert needle in original
    mutated = original.replace(needle, 'version         = "9.9.9"', 1)
    path = tmp_path / "manifest.toml"
    path.write_text(mutated, encoding="utf-8")

    problems = manifest_drift(path, ROOT)
    assert any("version 9.9.9 !=" in item for item in problems), problems
    assert not any(item.startswith("missing skill") for item in problems), problems


def test_tmp_copy_missing_a_skill_drifts(tmp_path: Path) -> None:
    original = MANIFEST.read_text(encoding="utf-8")
    data = tomllib.loads(original)
    names = [entry["name"] for entry in data["plugin"].get("skills", [])]
    expected = {path.name for path in _skill_dirs()}
    assert set(names) == expected

    dropped = sorted(names)[0]
    parts = original.split("[[plugin.skills]]")
    head, sections = parts[0], parts[1:]
    kept = [
        section
        for section in sections
        if re.search(rf'^name\s*=\s*"{re.escape(dropped)}"', section, flags=re.MULTILINE) is None
    ]
    path = tmp_path / "manifest.toml"
    body = "".join("[[plugin.skills]]" + section for section in kept)
    path.write_text(head + body, encoding="utf-8")

    problems = manifest_drift(path, ROOT)
    assert f"missing skill {dropped}" in problems
    for name in expected - {dropped}:
        assert f"missing skill {name}" not in problems
