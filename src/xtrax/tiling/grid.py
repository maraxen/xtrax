"""2D Pallas grid and BlockSpec construction with stride-0 shared topology (#3192)."""

from jax.experimental import pallas as pl


def make_2d_grid_specs(
    batch_size: int,
    total_elements: int,
    block_size: int,
) -> tuple[tuple[int, int], pl.BlockSpec, pl.BlockSpec]:
    """Construct 2D Pallas grid and BlockSpecs for batched inputs with shared topology.

    Grid shape: (batch_size, num_tiles)

    Returns:
        (grid, batched_block_spec, shared_block_spec)
        - batched_block_spec: maps (b, i) -> (b, i * block_size)
        - shared_block_spec: maps (b, i) -> (0, i * block_size) [stride-0 along batch]
    """
    if batch_size <= 0 or total_elements <= 0 or block_size <= 0:
        raise ValueError(
            f"batch_size ({batch_size}), total_elements ({total_elements}), "
            f"and block_size ({block_size}) must be positive integers."
        )
    if total_elements % block_size != 0:
        raise ValueError(
            f"total_elements ({total_elements}) must be divisible by block_size ({block_size})"
        )

    num_tiles = total_elements // block_size
    grid = (batch_size, num_tiles)

    # pl.BlockSpec(block_shape, index_map) with keyword arguments (Debt #3192 Finding C1)
    batched_block_spec = pl.BlockSpec(
        block_shape=(1, block_size),
        index_map=lambda b, i: (b, i * block_size),
    )

    shared_block_spec = pl.BlockSpec(
        block_shape=(1, block_size),
        index_map=lambda b, i: (0, i * block_size),
    )

    return grid, batched_block_spec, shared_block_spec
