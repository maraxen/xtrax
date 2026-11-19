"""Tests for 2D Pallas grid batching with stride-0 shared topology (#3192)."""

import pytest
from jax.experimental import pallas as pl

from xtrax.tiling.grid import make_2d_grid_specs


class TestPallasGridBatching:
    """Verification controls for 2D Pallas grid and BlockSpec construction."""

    def test_make_2d_grid_specs_dimensions_and_types(self):
        """Constructs grid and BlockSpecs with correct shapes and callable index mappings."""
        batch_size = 4
        total_elements = 128
        block_size = 32

        grid, batched_spec, shared_spec = make_2d_grid_specs(
            batch_size=batch_size,
            total_elements=total_elements,
            block_size=block_size,
        )

        assert grid == (4, 4)
        assert isinstance(batched_spec, pl.BlockSpec)
        assert isinstance(shared_spec, pl.BlockSpec)

        # Verify block shapes
        assert batched_spec.block_shape == (1, 32)
        assert shared_spec.block_shape == (1, 32)

        # Verify index map signatures and results
        # Batched index map maps (b, i) -> (b, i * block_size)
        assert batched_spec.index_map(0, 0) == (0, 0)
        assert batched_spec.index_map(2, 3) == (2, 96)

        # Shared index map has stride-0 along batch axis: (b, i) -> (0, i * block_size)
        assert shared_spec.index_map(0, 0) == (0, 0)
        assert shared_spec.index_map(2, 3) == (0, 96)
        assert shared_spec.index_map(3, 1) == (0, 32)

    def test_non_divisible_total_elements_raises_value_error(self):
        """total_elements not divisible by block_size raises ValueError."""
        with pytest.raises(ValueError, match="must be divisible by block_size"):
            make_2d_grid_specs(batch_size=2, total_elements=100, block_size=32)

    def test_non_positive_dimensions_raise_value_error(self):
        """Non-positive batch_size, total_elements, or block_size raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            make_2d_grid_specs(batch_size=0, total_elements=32, block_size=32)

        with pytest.raises(ValueError, match="must be positive"):
            make_2d_grid_specs(batch_size=2, total_elements=0, block_size=32)

        with pytest.raises(ValueError, match="must be positive"):
            make_2d_grid_specs(batch_size=2, total_elements=32, block_size=0)
