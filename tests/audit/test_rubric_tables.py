"""Contract tests for judgment rubric anchor tables (N4.2 / #1591)."""

from pathlib import Path

import pytest

from xtrax.devtools.rubrics import (
    DEFAULT_RUBRICS_DIR,
    RubricTable,
    load_all_rubrics,
    load_rubric,
)

ROOT = Path(__file__).resolve().parents[2]
RUBRICS_DIR = ROOT / "audit" / "rubrics"

EXPECTED_DIMENSIONS = frozenset(
    {
        "correctness",
        "jax_purity",
        "type_hardening",
        "performance",
        "documentation",
        "api_ergonomics",
        "test_rigor",
        "structure_complexity",
    }
)


@pytest.mark.parametrize("dimension", sorted(EXPECTED_DIMENSIONS))
def test_each_rubric_file_parses_with_five_anchors(dimension: str) -> None:
    path = RUBRICS_DIR / f"{dimension}.toml"
    table = load_rubric(path)
    assert table.dimension == dimension
    assert table.version
    assert table.dim_label
    scores = {anchor.score for anchor in table.anchors}
    assert scores == {1, 2, 3, 4, 5}
    for anchor in table.anchors:
        assert anchor.criterion
        assert anchor.evidence_hint


def test_load_all_rubrics_returns_eight_tables() -> None:
    tables = load_all_rubrics(dir=RUBRICS_DIR)
    assert set(tables) == EXPECTED_DIMENSIONS
    assert len(tables) == 8
    for table in tables.values():
        assert isinstance(table, RubricTable)
        assert len(table.anchors) == 5


def test_default_rubrics_dir_is_repo_relative() -> None:
    assert DEFAULT_RUBRICS_DIR == Path("audit/rubrics")
