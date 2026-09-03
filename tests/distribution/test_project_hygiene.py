"""Tests for distribution N8 project hygiene gate (#1460)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from scripts.audit_project_hygiene import (
    audit_project_hygiene,
    load_project_hygiene_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "project_hygiene.toml"


def _write_config(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "project_hygiene.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_hygiene_files(repo_root: Path, *, version: str = "0.3.0") -> None:
    (repo_root / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (repo_root / "README.md").write_text(
        textwrap.dedent(
            """
            # xtrax

            pip install xtrax

            Docs: https://xtrax.readthedocs.io

            Apache License 2.0

            See CONTRIBUTING.md and CHANGELOG.md
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "CHANGELOG.md").write_text(
        "# Changelog\n\nKeep a Changelog\n\n## [Unreleased]\n",
        encoding="utf-8",
    )
    (repo_root / "CONTRIBUTING.md").write_text("# Contributing\n", encoding="utf-8")
    (repo_root / "CITATION.cff").write_text(
        textwrap.dedent(
            f"""
            cff-version: 1.2.0
            title: xtrax
            version: {version}
            repository-code: https://github.com/maraxen/xtrax
            license: Apache-2.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    init = repo_root / "src" / "xtrax"
    init.mkdir(parents=True)
    (init / "__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (repo_root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "xtrax"
            readme = "README.md"

            [project.urls]
            Documentation = "https://xtrax.readthedocs.io"
            Changelog = "https://github.com/maraxen/xtrax/blob/main/CHANGELOG.md"
            Repository = "https://github.com/maraxen/xtrax"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_project_hygiene_config_reads_committed_toml() -> None:
    config = load_project_hygiene_config(CONFIG_PATH)
    assert config.version == "0.1.0"
    assert "main.py" in config.forbidden_root_paths
    assert "README.md" in config.required_files
    assert "CONTRIBUTING.md" in config.readme_markers


def test_audit_project_hygiene_passes_on_repo() -> None:
    passed, failures = audit_project_hygiene(ROOT, CONFIG_PATH)
    assert passed is True, failures
    assert failures == []


def test_audit_project_hygiene_fails_on_missing_readme(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_hygiene_files(tmp_path)
    (tmp_path / "README.md").unlink()
    passed, failures = audit_project_hygiene(tmp_path, config_path)
    assert passed is False
    assert any("README.md" in item for item in failures)


def test_audit_project_hygiene_fails_on_root_main_py(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_hygiene_files(tmp_path)
    (tmp_path / "main.py").write_text("print('demo')\n", encoding="utf-8")
    passed, failures = audit_project_hygiene(tmp_path, config_path)
    assert passed is False
    assert any("main.py" in item for item in failures)


def test_audit_project_hygiene_fails_on_citation_version_mismatch(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    _write_hygiene_files(tmp_path, version="0.3.0")
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8").replace("0.3.0", "0.2.0"),
        encoding="utf-8",
    )
    passed, failures = audit_project_hygiene(tmp_path, config_path)
    assert passed is False
    assert any("CITATION.cff version" in item for item in failures)


def _write_skill(repo_root: Path, name: str, version: str | None) -> Path:
    skill_dir = repo_root / "agent_assets" / "skills" / name
    skill_dir.mkdir(parents=True)
    version_line = "" if version is None else f"xtrax_version: {version}\n"
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: a skill\n{version_line}---\n\nbody\n",
        encoding="utf-8",
    )
    return skill_path


def test_audit_project_hygiene_fails_on_skill_version_mismatch(tmp_path: Path) -> None:
    """A SKILL.md marker that lags __version__ is a failure.

    agent_assets/skills/*/SKILL.md is a third version site after __init__.py and
    CITATION.cff, and it went stale at 0.4.0a8 in exactly the way CITATION.cff would
    have without its own check -- all three markers still read 0.4.0a7 with nothing
    to say so.
    """
    config_path = _write_config(tmp_path)
    _write_hygiene_files(tmp_path, version="0.3.0")
    _write_skill(tmp_path, "using-xtrax", "0.2.0")

    passed, failures = audit_project_hygiene(tmp_path, config_path)

    assert passed is False
    assert any("using-xtrax/SKILL.md xtrax_version" in item for item in failures)


def test_audit_project_hygiene_accepts_matching_skill_version(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_hygiene_files(tmp_path, version="0.3.0")
    _write_skill(tmp_path, "using-xtrax", "0.3.0")

    _passed, failures = audit_project_hygiene(tmp_path, config_path)

    # Asserting the absence of THIS failure, not an overall pass: the synthetic fixture
    # trips unrelated rules (its README is under min_readme_bytes), so `passed is True`
    # would be testing the fixture rather than the check.
    assert not any("SKILL.md xtrax_version" in item for item in failures), failures


def test_audit_project_hygiene_ignores_skill_without_a_version_marker(tmp_path: Path) -> None:
    """The marker is optional -- a skill that declares no version cannot go stale.

    Without this, adding the check would have silently required a version stamp on
    every skill ever added, which is a different policy than the one intended.
    """
    config_path = _write_config(tmp_path)
    _write_hygiene_files(tmp_path, version="0.3.0")
    _write_skill(tmp_path, "unversioned-skill", None)

    _passed, failures = audit_project_hygiene(tmp_path, config_path)

    assert not any("SKILL.md xtrax_version" in item for item in failures), failures


def test_main_cli_passes_on_repo() -> None:
    # --no-sync: a bare `uv run` in a subprocess re-resolves the shared venv to whatever
    # extras it infers, stripping deps out from under whichever tier is running this and
    # producing failures elsewhere that look nothing like their cause.
    result = subprocess.run(
        ["uv", "run", "--no-sync", "python", "scripts/audit_project_hygiene.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout


def test_justfile_defines_audit_project_hygiene() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-project-hygiene:" in text
