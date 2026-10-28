"""The packaged agent_plugin.json is generated; the release PR regenerates it.

Feature PRs do not commit `src/xtrax/agent_plugin.json` (every skill edit used to
conflict on it). The release PR regenerates it with

    just plugin-snapshot

and CI enforces freshness only there: the drift check below runs when
`XTRAX_ENFORCE_PLUGIN_SNAPSHOT=1` (set by `.github/workflows/ci.yml` for
`chore/release-*` branches). The wheel must still ship the file, so the
existence and parse checks run on every PR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / ".praxia" / "manifest.toml"
SNAPSHOT = REPO_ROOT / "src" / "xtrax" / "agent_plugin.json"
ENFORCE_ENV = "XTRAX_ENFORCE_PLUGIN_SNAPSHOT"


def test_packaged_snapshot_exists_and_parses() -> None:
    assert SNAPSHOT.is_file(), f"{SNAPSHOT.relative_to(REPO_ROOT)} is missing from the tree"
    data = json.loads(SNAPSHOT.read_text())
    assert isinstance(data, dict)
    assert "bundle" in data


@pytest.mark.skipif(
    os.environ.get(ENFORCE_ENV) != "1",
    reason=(
        f"snapshot drift is enforced only in the release PR ({ENFORCE_ENV}=1); "
        "regenerate it there with `just plugin-snapshot`"
    ),
)
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
            "`just plugin-snapshot`"
        )
