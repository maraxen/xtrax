"""Unit tests for Campaign EDA schemas and data types."""

import numpy as np
import pytest

from xtrax.eda.campaign.types import (
    CellStatus,
    DomainMetricResult,
    ResidueTable,
)


def test_domain_metric_result_validation():
    """Verify that validate_schema enforces finite numeric floats/ints (MAJ-02)."""
    # Valid metrics
    valid = DomainMetricResult(
        metrics={"diversity": 0.45, "rec": 0.62, "nll": 1.25},
        flags={"degenerate": False},
        tracks={},
    )
    valid.validate_schema()

    # Invalid: NaN metric
    invalid_nan = DomainMetricResult(
        metrics={"diversity": float("nan")},
        flags={},
        tracks={},
    )
    with pytest.raises(TypeError, match="finite"):
        invalid_nan.validate_schema()

    # Invalid: Infinity metric
    invalid_inf = DomainMetricResult(
        metrics={"nll": float("inf")},
        flags={},
        tracks={},
    )
    with pytest.raises(TypeError, match="finite"):
        invalid_inf.validate_schema()


def test_cell_status_values():
    """Verify CellStatus enum values."""
    assert CellStatus.COMMITTED.value == "COMMITTED"
    assert CellStatus.PARTIAL.value == "PARTIAL"
    assert CellStatus.FAILED.value == "FAILED"
    assert CellStatus.MISSING.value == "MISSING"


def test_residue_table_creation():
    """Verify ResidueTable creation and frozen semantics."""
    table = ResidueTable(
        native_tokens="MQIFVK",
        chain_ids=("A", "A", "A", "A", "A", "A"),
        res_numbers=(1, 2, 3, 4, 5, 6),
        insertion_codes=("", "", "", "", "", ""),
        real_mask=np.ones(6, dtype=bool),
        parser_name="proxide",
        parser_version="1.0",
    )
    assert table.native_tokens == "MQIFVK"
    assert len(table.res_numbers) == 6
    assert table.real_mask.shape == (6,)
