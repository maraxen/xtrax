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


def test_main_cli_passes_on_repo() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_project_hygiene.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout


def test_justfile_defines_audit_project_hygiene() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-project-hygiene:" in text
