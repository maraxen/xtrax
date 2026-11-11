"""Tests for dynamic scalar n_steps in WhileCarry continuation predicates (#3193)."""

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.tiling.iterator import WhileLoopIterator
from xtrax.tiling.strategy import fixed_step_count_cond


class TestWhileDynamicSteps:
    """Verification controls for dynamic scalar n_steps in fixed_step_count_cond."""

    def test_dynamic_jax_scalar_tracer_in_jit(self):
        """fixed_step_count_cond accepts a dynamic JAX scalar tracer inside @jax.jit."""

        @jax.jit
        def run_loop(n_steps: jax.Array) -> tuple[jax.Array, jax.Array]:
            cond = fixed_step_count_cond(n_steps)

            def body(carry):
                step_i, total = carry
                return step_i + 1, total + 2.0

            init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.float32))
            iterator = WhileLoopIterator()
            return iterator(cond, body, init)

        # Execute with dynamic scalar array
        steps_arg = jnp.array(5, dtype=jnp.int32)
        final_step, final_total = run_loop(steps_arg)

        assert int(final_step) == 5
        np.testing.assert_allclose(float(final_total), 10.0)

    def test_squeezes_1d_scalar_tensor_to_scalar_rank(self):
        """1-D array with shape (1,) is squeezed to scalar rank (Finding M3)."""

        @jax.jit
        def run_loop(n_steps: jax.Array):
            cond = fixed_step_count_cond(n_steps)

            def body(c):
                return (c[0] + 1, c[1] + 1.0)

            init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.float32))
            return WhileLoopIterator()(cond, body, init)

        # 1-D array: shape (1,)
        steps_1d = jnp.array([4], dtype=jnp.int32)
        final_step, _ = run_loop(steps_1d)
        assert int(final_step) == 4

    def test_zero_or_negative_steps_terminates_immediately(self):
        """Dynamic steps <= 0 terminates immediately without executing body."""

        @jax.jit
        def run_loop(n_steps: jax.Array):
            cond = fixed_step_count_cond(n_steps)

            def body(c):
                return (c[0] + 1, c[1] + 1.0)

            init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.float32))
            return WhileLoopIterator()(cond, body, init)

        final_step_0, _ = run_loop(jnp.array(0, dtype=jnp.int32))
        assert int(final_step_0) == 0

        final_step_neg, _ = run_loop(jnp.array(-3, dtype=jnp.int32))
        assert int(final_step_neg) == 0

    def test_static_python_int_compatibility(self):
        """fixed_step_count_cond still accepts a static Python int."""
        cond = fixed_step_count_cond(3)

        def body(c):
            return (c[0] + 1, c[1] + 1.0)

        init = (jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.float32))
        final_step, _ = WhileLoopIterator()(cond, body, init)
        assert int(final_step) == 3
