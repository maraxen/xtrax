"""Smoke test for import-linter cycle-block foundation gate."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_import_linter_reports_no_cycles() -> None:
    result = subprocess.run(
        ["uv", "run", "lint-imports"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import-linter failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
