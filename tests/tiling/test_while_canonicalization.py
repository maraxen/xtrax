"""Tests for carry-leaf canonicalization and transition validation (#3193)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.carry import (
    CarryCanonicalizationError,
    canonicalize_carry,
    validate_carry_transition,
)
from xtrax.tiling.iterator import WhileLoopIterator


class TestWhileCanonicalization:
    """Verification controls for canonicalize_carry and validate_carry_transition."""

    def test_canonicalize_converts_python_and_numpy_to_jax_arrays(self):
        """Python scalars and NumPy arrays in carry PyTree become immutable JAX arrays."""
        raw_carry = {
            "step": 0,
            "weight": np.array([1.0, 2.0], dtype=np.float32),
            "flag": True,
        }
        canonical = canonicalize_carry(raw_carry)
        assert isinstance(canonical["step"], jax.Array)
        assert isinstance(canonical["weight"], jax.Array)
        assert isinstance(canonical["flag"], jax.Array)
        assert canonical["weight"].shape == (2,)

    def test_validate_carry_transition_passes_identical_structure_and_types(self):
        """Valid transition with invariant structure, shapes, and dtypes succeeds."""
        init = (jnp.zeros((), dtype=jnp.int32), jnp.ones((4,), dtype=jnp.float32))
        next_c = (jnp.array(1, dtype=jnp.int32), jnp.ones((4,), dtype=jnp.float32) * 2.0)
        validate_carry_transition(init, next_c)  # Should not raise

    def test_structure_mismatch_raises_canonicalization_error(self):
        """Transition returning different PyTree structure raises CarryCanonicalizationError."""
        init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((4,)))
        # Body returns a 3-tuple instead of 2-tuple
        bad_next = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((4,)), 1.0)
        with pytest.raises(CarryCanonicalizationError, match="structure mismatch"):
            validate_carry_transition(init, bad_next)

    def test_shape_mutation_raises_canonicalization_error(self):
        """Transition returning modified array shape raises CarryCanonicalizationError."""
        init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((4,)))
        bad_shape = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((5,)))
        with pytest.raises(CarryCanonicalizationError, match="shape mismatch"):
            validate_carry_transition(init, bad_shape)

    def test_dtype_drift_raises_canonicalization_error(self):
        """Transition mutating dtype (promoting int to float) raises error."""
        init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((2,), dtype=jnp.float32))
        bad_dtype = (jnp.array(1.0, dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32))
        with pytest.raises(CarryCanonicalizationError, match="dtype mismatch"):
            validate_carry_transition(init, bad_dtype)

    def test_while_loop_iterator_enforces_validation(self):
        """WhileLoopIterator detects and rejects invalid carry transitions during execution."""

        def cond(c):
            return c[0] < 5

        # Buggy body: expands shape on each step
        def bad_body(c):
            return (c[0] + 1, jnp.concatenate([c[1], jnp.array([1.0])]))

        init = (0, np.array([0.0], dtype=np.float32))

        iterator = WhileLoopIterator()
        with pytest.raises(CarryCanonicalizationError):
            iterator(cond, bad_body, init)
