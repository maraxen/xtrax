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
        # praxia's WorkflowEntry requires 'template_path' (a manifest with 'path'
        # INSTEAD of it fails deserialization at install time -- cisterna's
        # .praxia/manifest.toml hit exactly that). cisternal's snapshot reads 'path',
        # and praxia ignores unknown keys (praxia-config manifest.rs, no
        # deny_unknown_fields), so 'path' is allowed only as a copy of 'template_path'.
        assert "template_path" in entry, entry
        if "path" in entry:
            assert entry["path"] == entry["template_path"], (
                f"{entry}: 'path' (cisternal) must equal 'template_path' (praxia)"
            )


def test_every_workflow_entry_template_path_exists() -> None:
    data = _load_manifest()
    for entry in data["plugin"]["workflows"]:
        assert (ROOT / entry["template_path"]).is_file(), entry


@pytest.mark.skipif(shutil.which("praxia") is None, reason="praxia CLI not installed")
def test_fresh_install_leaves_no_workflow_copies() -> None:
    """praxia no longer copies plugin workflows into ``~/.praxia/workflows``.

    Since praxia's 260925 embedded-asset-resolution change (writer.rs: "Workflows
    are NOT copied"), a registered plugin's templates are read in place by the
    shared resolver, so a fresh install must succeed and leave no ``xtrax_*.yaml``
    shadow copy and no ``.yaml`` in ``exported_files``. That every manifest
    template exists in this checkout is covered by
    ``test_every_workflow_entry_template_path_exists``.
    """
    with tempfile.TemporaryDirectory() as tmp_home:
        env = {
            **os.environ,
            "HOME": tmp_home,
            "PRAXIA_HOME": str(Path(tmp_home) / ".praxia"),
        }
        result = subprocess.run(
            ["praxia", "plugin", "install", str(ROOT)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

        workflows_dir = Path(tmp_home) / ".praxia" / "workflows"
        assert sorted(workflows_dir.glob("xtrax_*.yaml")) == []

        registry = tomllib.loads(
            (Path(tmp_home) / ".praxia" / "plugins.toml").read_text(encoding="utf-8")
        )
        files = [f for fs in registry["plugins"]["xtrax"]["exported_files"].values() for f in fs]
        assert not [f for f in files if f.endswith(".yaml")], files
