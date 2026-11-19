"""Shared fixtures for campaign tests."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_run_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the run ledger at a per-test directory.

    `xtrax campaign run` writes a run-ledger row; without this, every test that runs a
    campaign would append to ``.xtrax/ledger`` under the repository.
    """
    root = tmp_path / "run_ledger"
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(root))
    return root
