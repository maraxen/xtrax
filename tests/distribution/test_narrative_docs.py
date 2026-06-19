"""Tests for distribution N4b narrative docs gate (#1458)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from scripts.audit_narrative_docs import (
    audit_narrative_docs,
    load_narrative_docs_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "narrative_docs.toml"


def _write_config(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "narrative_docs.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_narrative_pages(repo_root: Path) -> None:
    docs = repo_root / "docs"
    (docs / "advanced").mkdir(parents=True)
    quickstart = (
        textwrap.dedent(
            """
        # Quickstart

        pip install xtrax

        Trainer and ResumableState with Engine.fit_sync.
        """
        ).strip()
        + "\n" * 40
    )
    architecture = (
        textwrap.dedent(
            """
        # Architecture

        BatchPlanner chooses default_batch_size.
        ResumableState and Engine orchestrate training.
        """
        ).strip()
        + "\n" * 120
    )
    concepts = (
        textwrap.dedent(
            """
        # Concepts

        AxisSpec uses default_batch_size. ResumableState flows through Engine.
        """
        ).strip()
        + "\n" * 90
    )
    debugging = (
        textwrap.dedent(
            """
        # Debugging

        Watch for recompilation when static batch sizes change.
        """
        ).strip()
        + "\n"
    )
    (docs / "quickstart.md").write_text(quickstart, encoding="utf-8")
    (docs / "architecture.md").write_text(architecture, encoding="utf-8")
    (docs / "concepts.md").write_text(concepts, encoding="utf-8")
    (docs / "advanced" / "debugging.md").write_text(debugging, encoding="utf-8")


def test_load_narrative_docs_config_reads_committed_toml() -> None:
    config = load_narrative_docs_config(CONFIG_PATH)
    assert config.version == "0.1.0"
    assert len(config.pages) == 4
    assert "AxisSpec.batch_size" in config.forbidden_phrases


def test_audit_narrative_docs_passes_on_repo() -> None:
    passed, failures = audit_narrative_docs(ROOT, CONFIG_PATH)
    assert passed is True, failures
    assert failures == []


def test_audit_narrative_docs_fails_on_forbidden_phrase(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_narrative_pages(tmp_path)
    arch = tmp_path / "docs" / "architecture.md"
    arch.write_text(
        arch.read_text(encoding="utf-8") + "\nAxisSpec.batch_size\n",
        encoding="utf-8",
    )
    passed, failures = audit_narrative_docs(tmp_path, config_path)
    assert passed is False
    assert any("forbidden phrase" in item for item in failures)


def test_audit_narrative_docs_fails_on_missing_marker(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_narrative_pages(tmp_path)
    quickstart = tmp_path / "docs" / "quickstart.md"
    quickstart.write_text("# Quickstart\n\nshort\n", encoding="utf-8")
    passed, failures = audit_narrative_docs(tmp_path, config_path)
    assert passed is False
    assert any("quickstart.md" in item for item in failures)


def test_main_cli_passes_on_repo() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_narrative_docs.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout


def test_justfile_defines_audit_narrative_docs() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-narrative-docs:" in text
