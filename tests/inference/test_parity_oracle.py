"""Parity oracle tests: E1 synthesize_axes can REPRODUCE hand-written AxisSpec declarations.

Goal: Prove E1's synthesize_axes (driven by @axis_config / AxisOverride) can
REPRODUCE existing hand-written AxisSpec declarations field-for-field.

AC8 assertions:
1. Parity test 1: N_ATOMS (tile_granularity=128 override)
2. Parity test 2: N_CONFORMERS (heterogeneous=True override)
3. Parity test 3: N_MOLS with bucket_boundaries (non-default override)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import ShapeDtypeStruct

from xtrax.inference.axes import synthesize_axes
from xtrax.inference.config import AxisOverride
from xtrax.inference.errors import AxisRole


class TestParityOracle:
    """E1.8: Parity oracle — prove synthesize_axes can reproduce hand-written specs."""

    def test_parity_n_atoms_with_tile_granularity_128(self):
        """Parity: N_ATOMS with non-default tile_granularity=128.

        Hand-written site (from test_prolix_compat.py):
            N_ATOMS = AxisSpec(
                name="n_atoms",
                cardinality=60_000,
                default_batch_size=60_000,
                tile_granularity=128,
                heterogeneous=False,
            )
        """
        from xtrax.tiling import AxisSpec

        # Build matching AxisOverride
        override = AxisOverride(
            name="n_atoms",
            default_batch_size=60_000,
            cardinality=60_000,
            tile_granularity=128,
            heterogeneous=False,
            dedup_eligible=False,
            bucket_boundaries=None,
        )

        # Build dummy abstract input: one leaf with leading dim matching cardinality
        abstract_input = ShapeDtypeStruct((60_000,), jnp.float32)

        # Synthesize
        derived_specs = synthesize_axes([abstract_input], [override])
        assert len(derived_specs) == 1
        derived = derived_specs[0]

        # Construct expected hand-written spec
        expected = AxisSpec(
            name="n_atoms",
            cardinality=60_000,
            default_batch_size=60_000,
            tile_granularity=128,
            heterogeneous=False,
            dedup_eligible=False,
            bucket_boundaries=None,
        )

        # Assert ALL SEVEN fields match
        assert derived.name == expected.name
        assert derived.cardinality == expected.cardinality
        assert derived.default_batch_size == expected.default_batch_size
        assert derived.tile_granularity == expected.tile_granularity
        assert derived.heterogeneous == expected.heterogeneous
        assert derived.dedup_eligible == expected.dedup_eligible
        assert derived.bucket_boundaries == expected.bucket_boundaries
        # Role: derived should be KNOWN (override present)
        assert derived.role is AxisRole.KNOWN

    def test_parity_n_conformers_with_heterogeneous_true(self):
        """Parity: N_CONFORMERS with non-default heterogeneous=True.

        Hand-written site (from test_prolix_compat.py):
            N_CONFORMERS = AxisSpec(
                name="n_conformers",
                cardinality=2048,
                default_batch_size=1,
                tile_granularity=1,
                heterogeneous=True,
            )
        """
        from xtrax.tiling import AxisSpec

        # Build matching AxisOverride
        override = AxisOverride(
            name="n_conformers",
            default_batch_size=1,
            cardinality=2048,
            tile_granularity=1,
            heterogeneous=True,
            dedup_eligible=False,
            bucket_boundaries=None,
        )

        # Build dummy abstract input: one leaf with leading dim matching cardinality
        abstract_input = ShapeDtypeStruct((2048,), jnp.float32)

        # Synthesize
        derived_specs = synthesize_axes([abstract_input], [override])
        assert len(derived_specs) == 1
        derived = derived_specs[0]

        # Construct expected hand-written spec
        expected = AxisSpec(
            name="n_conformers",
            cardinality=2048,
            default_batch_size=1,
            tile_granularity=1,
            heterogeneous=True,
            dedup_eligible=False,
            bucket_boundaries=None,
        )

        # Assert ALL SEVEN fields match
        assert derived.name == expected.name
        assert derived.cardinality == expected.cardinality
        assert derived.default_batch_size == expected.default_batch_size
        assert derived.tile_granularity == expected.tile_granularity
        assert derived.heterogeneous == expected.heterogeneous
        assert derived.dedup_eligible == expected.dedup_eligible
        assert derived.bucket_boundaries == expected.bucket_boundaries
        # Role: derived should be KNOWN (override present)
        assert derived.role is AxisRole.KNOWN

    def test_parity_n_mols_with_bucket_boundaries(self):
        """Parity: N_MOLS with bucket_boundaries (non-default override).

        Hand-written base (from test_prolix_compat.py):
            N_MOLS = AxisSpec(
                name="n_mols", cardinality=64, default_batch_size=1,
                tile_granularity=1, heterogeneous=True
            )

        Extended with bucket_boundaries=(8, 16, 32) to test bucketing parity.
        """
        from xtrax.tiling import AxisSpec

        # Build matching AxisOverride with bucket_boundaries
        override = AxisOverride(
            name="n_mols",
            default_batch_size=1,
            cardinality=64,
            tile_granularity=1,
            heterogeneous=True,
            dedup_eligible=False,
            bucket_boundaries=(8, 16, 32),
        )

        # Build dummy abstract input: one leaf with leading dim matching cardinality
        abstract_input = ShapeDtypeStruct((64,), jnp.float32)

        # Synthesize
        derived_specs = synthesize_axes([abstract_input], [override])
        assert len(derived_specs) == 1
        derived = derived_specs[0]

        # Construct expected hand-written spec
        expected = AxisSpec(
            name="n_mols",
            cardinality=64,
            default_batch_size=1,
            tile_granularity=1,
            heterogeneous=True,
            dedup_eligible=False,
            bucket_boundaries=(8, 16, 32),
        )

        # Assert ALL SEVEN fields match
        assert derived.name == expected.name
        assert derived.cardinality == expected.cardinality
        assert derived.default_batch_size == expected.default_batch_size
        assert derived.tile_granularity == expected.tile_granularity
        assert derived.heterogeneous == expected.heterogeneous
        assert derived.dedup_eligible == expected.dedup_eligible
        assert derived.bucket_boundaries == expected.bucket_boundaries
        # Role: derived should be KNOWN (override present)
        assert derived.role is AxisRole.KNOWN
