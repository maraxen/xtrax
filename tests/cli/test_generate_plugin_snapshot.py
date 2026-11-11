"""Tests for scripts/generate_plugin_snapshot.py (#3144)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_plugin_snapshot.py"
MANIFEST_PATH = REPO_ROOT / ".praxia" / "manifest.toml"


def test_script_exists() -> None:
    assert SCRIPT_PATH.is_file(), f"{SCRIPT_PATH} does not exist"


def test_generate_snapshot_custom_output(tmp_path: Path) -> None:
    pytest.importorskip("cisternal.cli")
    out_file = tmp_path / "custom_agent_plugin.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--out",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "bundle" in data
    assert data["bundle"]["metadata"]["name"] == "xtrax"


def test_generate_snapshot_check_mode(tmp_path: Path) -> None:
    pytest.importorskip("cisternal.cli")
    out_file = tmp_path / "checked_plugin.json"
    # First generate
    res1 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--out",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert res1.returncode == 0

    # Then check - should pass
    res2 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--out",
            str(out_file),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert res2.returncode == 0

    # Corrupt file - check should fail with returncode 1
    out_file.write_text('{"bundle": {"stale": true}}', encoding="utf-8")
    res3 = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(MANIFEST_PATH),
            "--out",
            str(out_file),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert res3.returncode == 1


def test_generate_snapshot_missing_manifest(tmp_path: Path) -> None:
    missing_manifest = tmp_path / "nonexistent.toml"
    out_file = tmp_path / "out.json"
    res = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--manifest",
            str(missing_manifest),
            "--out",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert res.returncode == 1
