"""Tests for distribution N4a docs plumbing gate (#1457)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

from scripts.audit_docs_plumbing import (
    audit_docs_plumbing,
    check_sphinx_conf,
    check_wiring_files,
    load_docs_plumbing_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "docs_plumbing.toml"


def _write_config(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "docs_plumbing.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def test_load_docs_plumbing_config_reads_committed_toml() -> None:
    config = load_docs_plumbing_config(CONFIG_PATH)
    assert config.version == "0.1.0"
    assert config.sphinx_conf == "docs/conf.py"
    assert config.readthedocs_config == ".readthedocs.yaml"
    assert config.workflow == ".github/workflows/docs.yml"
    assert config.sphinx_warn_is_error is True
    assert config.sphinx_nitpicky is True
    assert "docs" in config.install_groups
    assert "eda" in config.install_extras
    assert "sphinx.ext.autosummary" in config.required_extensions


def test_check_wiring_files_passes_on_repo() -> None:
    config = load_docs_plumbing_config(CONFIG_PATH)
    assert check_wiring_files(config, ROOT) == []


def test_check_sphinx_conf_passes_on_repo() -> None:
    config = load_docs_plumbing_config(CONFIG_PATH)
    assert check_sphinx_conf(config, ROOT) == []


def test_check_sphinx_conf_flags_missing_extension(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = load_docs_plumbing_config(config_path)
    conf = tmp_path / "docs"
    conf.mkdir()
    (conf / "conf.py").write_text(
        textwrap.dedent(
            """
            extensions = ["sphinx.ext.autodoc"]
            autosummary_generate = True
            html_theme = "furo"
            intersphinx_mapping = {}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    failures = check_sphinx_conf(config, tmp_path)
    assert any("extensions missing" in item for item in failures)


def test_audit_docs_plumbing_check_only_passes_on_repo() -> None:
    passed, failures = audit_docs_plumbing(
        root=ROOT,
        config_path=CONFIG_PATH,
        skip_build=True,
    )
    assert passed is True
    assert failures == []


def test_audit_docs_plumbing_fails_when_workflow_missing(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    passed, failures = audit_docs_plumbing(
        root=tmp_path,
        config_path=config_path,
        skip_build=True,
    )
    assert passed is False
    assert any("docs.yml" in item for item in failures)


def test_main_check_only_exits_zero() -> None:
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/audit_docs_plumbing.py",
            "--check-only",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout


def test_justfile_defines_audit_docs_build() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-docs-build:" in text
    assert "audit_docs_plumbing.py" in text


def test_docs_workflow_uses_warn_is_error_and_eda_extra() -> None:
    text = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    assert "sphinx-build -W -n" in text
    assert "--extra eda" in text


def test_audit_docs_plumbing_builds_docs() -> None:
    passed, failures = audit_docs_plumbing(
        root=ROOT,
        config_path=CONFIG_PATH,
        skip_build=False,
    )
    assert passed is True, failures


def test_audit_docs_plumbing_mocks_build_failure(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    (tmp_path / ".readthedocs.yaml").write_text("version: 2\n", encoding="utf-8")
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "docs.yml").write_text("name: docs\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "conf.py").write_text(
        (ROOT / "docs" / "conf.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def fake_sync(root, config):  # noqa: ANN001
        return True, ""

    def fake_build(root, config):  # noqa: ANN001
        return False, "mock sphinx failure"

    with (
        patch("scripts.audit_docs_plumbing.run_uv_sync", side_effect=fake_sync),
        patch("scripts.audit_docs_plumbing.run_sphinx_build", side_effect=fake_build),
    ):
        passed, failures = audit_docs_plumbing(
            root=tmp_path,
            config_path=config_path,
            skip_build=False,
        )
    assert passed is False
    assert any("sphinx-build" in item for item in failures)
