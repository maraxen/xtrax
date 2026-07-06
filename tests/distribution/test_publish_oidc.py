"""Tests for distribution N7 publish OIDC gate (#1461)."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from scripts.audit_publish_oidc import (
    audit_publish_oidc,
    load_publish_oidc_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "distribution" / "publish_oidc.toml"


def _write_config(repo_root: Path) -> Path:
    config_dir = repo_root / "distribution"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "publish_oidc.toml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return config_path


def _write_publish_files(repo_root: Path) -> None:
    workflow_dir = repo_root / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "publish.yml").write_text(
        textwrap.dedent(
            """
            on:
              push:
                tags:
                  - 'v*'
              workflow_dispatch:

            jobs:
              build:
                steps:
                  - run: uv build
                  - run: uv run --with twine twine check dist/*
                  - name: Wheel smoke test
                    run: |
                      /tmp/wheel-smoke/bin/python -c \\
                        "import xtrax; print(xtrax.__version__)"
                  - uses: actions/upload-artifact@v4

              publish-pypi:
                if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
                environment:
                  name: pypi
                permissions:
                  id-token: write
                steps:
                  - uses: pypa/gh-action-pypi-publish@release/v1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (repo_root / "CONTRIBUTING.md").write_text(
        textwrap.dedent(
            """
            # Contributing

            TestPyPI and OIDC Trusted Publishing.

            Run `just audit-deterministic` and `just audit-publish-oidc` before release.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_publish_oidc_config_reads_committed_toml() -> None:
    config = load_publish_oidc_config(CONFIG_PATH)
    assert config.version == "0.2.0"
    assert config.human_prerequisite_backlog == 1454
    assert config.tag_pattern == "v*"


def test_audit_publish_oidc_passes_on_repo() -> None:
    passed, failures = audit_publish_oidc(ROOT, CONFIG_PATH)
    assert passed is True, failures
    assert failures == []


def test_audit_publish_oidc_fails_on_missing_workflow(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_publish_files(tmp_path)
    (tmp_path / ".github" / "workflows" / "publish.yml").unlink()
    passed, failures = audit_publish_oidc(tmp_path, config_path)
    assert passed is False
    assert any("missing publish workflow" in item for item in failures)


def test_audit_publish_oidc_fails_on_token_phrase(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _write_publish_files(tmp_path)
    workflow = tmp_path / ".github" / "workflows" / "publish.yml"
    workflow.write_text(
        workflow.read_text(encoding="utf-8") + "\npassword: ${{ secrets.PYPI }}\n",
        encoding="utf-8",
    )
    passed, failures = audit_publish_oidc(tmp_path, config_path)
    assert passed is False
    assert any("forbidden phrase" in item for item in failures)


def test_main_cli_passes_on_repo() -> None:
    result = subprocess.run(
        ["uv", "run", "python", "scripts/audit_publish_oidc.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "PASS" in result.stdout


def test_justfile_defines_audit_publish_oidc() -> None:
    text = (ROOT / "Justfile").read_text(encoding="utf-8")
    assert "audit-publish-oidc:" in text
