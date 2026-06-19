"""Verify ruff per-file-ignores documents jaxtyping F722/F821 pattern."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"


def test_ruff_per_file_ignores_has_jaxtyping_pattern() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    per_file = data.get("tool", {}).get("ruff", {}).get("lint", {}).get("per-file-ignores", {})
    assert per_file, "Expected [tool.ruff.lint.per-file-ignores] in pyproject.toml"
    combined = {rule for rules in per_file.values() for rule in rules}
    assert "F722" in combined or "F821" in combined, (
        "Expected F722 and/or F821 in per-file-ignores for jaxtyping shape strings"
    )
