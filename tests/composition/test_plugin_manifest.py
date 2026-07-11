"""Fresh-install invariant tests for the packaging manifest (T3-01, #3021, AC-X1)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / ".praxia/manifest.toml"


def _load_manifest() -> dict:
    return tomllib.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_exists_and_parses() -> None:
    assert MANIFEST_PATH.is_file()
    data = _load_manifest()
    assert data["plugin"]["name"] == "xtrax"
    workflows = data["plugin"]["workflows"]
    assert len(workflows) >= 1
    for entry in workflows:
        assert "template_path" in entry, entry
        assert "path" not in entry, (
            f"{entry}: use 'template_path', not 'path' -- WorkflowEntry has no 'path' field "
            "and a manifest using it fails TOML deserialization at install time (see cisterna's "
            ".praxia/manifest.toml for the real-world counter-example of this exact bug)"
        )


def test_every_workflow_entry_template_path_exists() -> None:
    data = _load_manifest()
    for entry in data["plugin"]["workflows"]:
        assert (ROOT / entry["template_path"]).is_file(), entry


@pytest.mark.skipif(shutil.which("praxia") is None, reason="praxia CLI not installed")
def test_fresh_install_workflow_count_matches_manifest() -> None:
    data = _load_manifest()
    expected_count = len(data["plugin"]["workflows"])

    with tempfile.TemporaryDirectory() as tmp_home:
        env = {**os.environ, "HOME": tmp_home}
        result = subprocess.run(
            ["praxia", "plugin", "install", str(ROOT)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        workflows_dir = Path(tmp_home) / ".praxia" / "workflows"
        exported = sorted(workflows_dir.glob("xtrax_*.yaml"))
        assert len(exported) == expected_count, exported
