"""The packaged agent_plugin.json must match the manifest it was generated from.

A skill edit without `cisternal assets snapshot --out src/xtrax/agent_plugin.json`
fails here. Skips cleanly when the `plugin` extra (cisternal) is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / ".praxia" / "manifest.toml"
SNAPSHOT = REPO_ROOT / "src" / "xtrax" / "agent_plugin.json"


def test_packaged_snapshot_is_fresh() -> None:
    cli = pytest.importorskip("cisternal.cli")
    try:
        cli.app(
            [
                "assets",
                "snapshot",
                "--check",
                "--manifest",
                str(MANIFEST),
                "--out",
                str(SNAPSHOT),
            ]
        )
    except SystemExit as exc:
        assert exc.code in (None, 0), (
            f"{SNAPSHOT.relative_to(REPO_ROOT)} is stale or missing; regenerate with "
            "`uv run --extra plugin cisternal assets snapshot --out src/xtrax/agent_plugin.json`"
        )
