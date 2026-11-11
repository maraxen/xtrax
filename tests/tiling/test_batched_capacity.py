"""Tests for BatchedCapacityState (#3194)."""

import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.bucket import BatchedCapacityState
from xtrax.tiling.strategy import fixed_step_count_cond


class TestBatchedCapacityState:
    """Verification controls for BatchedCapacityState."""

    def test_headroom_and_needs_reallocation_device_safe(self):
        """headroom and needs_reallocation are pure JAX array operations safe inside JIT."""
        state = BatchedCapacityState(
            allocated_capacity=100,
            active_counts=jnp.array([80, 95, 60]),
            headroom_threshold=10,
        )

        # Headroom: 100 - active_counts = [20, 5, 40]
        np.testing.assert_array_equal(state.headroom, jnp.array([20, 5, 40]))
        # 5 <= 10, so needs_reallocation is True
        assert bool(state.needs_reallocation) is True

        # Pure JIT execution of properties
        @jax.jit
        def check_status(s):
            return s.headroom, s.needs_reallocation

        hr, nr = check_status(state)
        np.testing.assert_array_equal(hr, jnp.array([20, 5, 40]))
        assert bool(nr) is True

    def test_no_reallocation_when_headroom_sufficient(self):
        """needs_reallocation is False when all items exceed headroom_threshold."""
        state = BatchedCapacityState(
            allocated_capacity=100,
            active_counts=jnp.array([50, 60, 70]),
            headroom_threshold=10,
        )
        assert bool(state.needs_reallocation) is False

    def test_suggest_next_rung_host_coordinator(self):
        """suggest_next_rung selects next rung accommodating max occupancy (Finding C5)."""
        state = BatchedCapacityState(
            allocated_capacity=64,
            active_counts=jnp.array([45, 60, 58]),
            headroom_threshold=10,
        )
        # Max active is 60; next rung from (64, 128, 256) is 64 (or 128 if max_active+1 exceeds 64)
        ladder = (32, 64, 128, 256)
        # max_active = 60, 60 + 1 = 61 <= 64, rung is 64
        assert state.suggest_next_rung(ladder) == 64

        state_exceeded = BatchedCapacityState(
            allocated_capacity=64,
            active_counts=jnp.array([64, 70, 50]),
            headroom_threshold=5,
        )
        # max_active = 70, 70 + 1 = 71 -> next rung is 128
        assert state_exceeded.suggest_next_rung(ladder) == 128

    def test_negative_capacity_raises_value_error(self):
        """Negative allocated_capacity raises ValueError."""
        with pytest.raises(ValueError, match="allocated_capacity must be >= 0"):
            BatchedCapacityState(allocated_capacity=-10, active_counts=jnp.array([1, 2]))

    def test_jaxtyping_type_hints_resolve_at_runtime(self):
        """get_type_hints resolves jaxtyping annotations at runtime."""
        hints_state = typing.get_type_hints(BatchedCapacityState)
        assert "active_counts" in hints_state
        hints_cond = typing.get_type_hints(fixed_step_count_cond)
        assert "n_steps" in hints_cond
