"""Tests for CarrySpec.transition end-to-end execution contract (#2543)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.dispatch import axis_dispatch
from xtrax.tiling.plan import AxisSpec, BatchPlanner
from xtrax.tiling.strategy import Scan, WhileCarry, fixed_step_count_cond


class TestCarryExecutionContract:
    """End-to-end contract verification for CarrySpec.transition execution."""

    def test_carry_spec_scan_end_to_end_execution(self):
        """CarrySpec with collect_outputs=True plans to Scan and executes exact recurrence."""

        # Recurrence: carry_t = carry_{t-1} + x_t, y_t = carry_t * 2
        def transition_fn(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
            new_carry = carry + x
            y = new_carry * 2.0
            return new_carry, y

        xs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)

        spec = AxisSpec(name="time_steps", cardinality=4, default_batch_size=4)
        carry_spec = CarrySpec(
            axis_name="time_steps",
            init=jnp.array(0.0, dtype=jnp.float32),
            transition=transition_fn,
            collect_outputs=True,
        )

        plan = BatchPlanner(carry_specs=[carry_spec]).plan([spec])
        decision = plan.decision_for("time_steps")
        assert isinstance(decision.strategy, Scan)

        # Dispatch execution
        final_carry, ys = axis_dispatch(decision.strategy, None, xs=xs)

        # Mathematical verification:
        # Step 0: c = 1.0, y = 2.0
        # Step 1: c = 3.0, y = 6.0
        # Step 2: c = 6.0, y = 12.0
        # Step 3: c = 10.0, y = 20.0
        np.testing.assert_allclose(final_carry, 10.0)
        np.testing.assert_allclose(ys, jnp.array([2.0, 6.0, 12.0, 20.0]))

    def test_carry_spec_while_carry_end_to_end_execution(self):
        """CarrySpec with collect_outputs=False plans to WhileCarry and runs to completion."""

        # Body: increment step and accumulate sum: (step_i, total) -> (step_i + 1, total + step_i)
        def body_fn(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            step_i, total = carry
            return step_i + 1, total + step_i

        spec = AxisSpec(name="sim_steps", cardinality=10, default_batch_size=10)
        carry_spec = CarrySpec(
            axis_name="sim_steps",
            init=(jnp.array(0, dtype=jnp.int32), jnp.array(0.0, dtype=jnp.float32)),
            transition=body_fn,
            collect_outputs=False,
            cond=fixed_step_count_cond(5),
        )

        plan = BatchPlanner(carry_specs=[carry_spec]).plan([spec])
        decision = plan.decision_for("sim_steps")
        assert isinstance(decision.strategy, WhileCarry)

        final_carry = axis_dispatch(decision.strategy, None, xs=None)
        final_step, final_total = final_carry

        # 0 + 1 + 2 + 3 + 4 = 10.0
        assert int(final_step) == 5
        np.testing.assert_allclose(float(final_total), 10.0)

    def test_planning_on_heterogeneous_axis_raises_value_error(self):
        """Planning a CarrySpec on a heterogeneous axis raises ValueError."""
        spec = AxisSpec(name="ragged", cardinality=5, default_batch_size=5)
        carry_spec = CarrySpec(
            axis_name="ragged",
            init=0.0,
            transition=lambda c, x: (c, x),
        )

        planner = BatchPlanner(carry_specs=[carry_spec], heterogeneous_axes={"ragged"})
        with pytest.raises(ValueError, match="Cannot create a carry strategy for axis 'ragged'"):
            planner.plan([spec])
