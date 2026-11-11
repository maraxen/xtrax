"""Tests for vmap_any_cond device-side amortized batch conditioning (#3194)."""

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.tiling.dispatch import vmap_any_cond


class TestVmapAnyCond:
    """Verification controls for vmap_any_cond."""

    def test_executes_work_fn_when_any_predicate_true(self):
        """When at least one batch item satisfies pred_fn, work_fn executes across all."""

        def pred_fn(x):
            return x > 5.0

        def work_fn(x):
            return x * 2.0

        def default_fn(x):
            return x + 1.0

        # Batch where second element satisfies predicate
        xs = jnp.array([2.0, 7.0, 1.0], dtype=jnp.float32)

        out = vmap_any_cond(pred_fn, work_fn, default_fn, xs)
        # Because 7.0 > 5.0 is True, work_fn runs on the whole batch
        expected = xs * 2.0
        np.testing.assert_allclose(out, expected)

    def test_executes_default_fn_when_all_predicates_false(self):
        """When no batch item satisfies pred_fn, default_fn executes across all."""

        def pred_fn(x):
            return x > 10.0

        def work_fn(x):
            return x * 2.0

        def default_fn(x):
            return x + 1.0

        xs = jnp.array([2.0, 7.0, 1.0], dtype=jnp.float32)

        out = vmap_any_cond(pred_fn, work_fn, default_fn, xs)
        # Because no element > 10.0, default_fn runs on the whole batch
        expected = xs + 1.0
        np.testing.assert_allclose(out, expected)

    def test_compatible_with_jit(self):
        """vmap_any_cond traces and executes inside @jax.jit."""

        @jax.jit
        def jitted_step(xs):
            return vmap_any_cond(
                pred_fn=lambda x: jnp.sum(x) > 0.0,
                work_fn=lambda x: x * 10.0,
                default_fn=lambda x: x - 10.0,
                xs=xs,
            )

        xs1 = jnp.array([[1.0, 0.0], [-1.0, -1.0]])
        np.testing.assert_allclose(jitted_step(xs1), xs1 * 10.0)

        xs2 = jnp.array([[-1.0, 0.0], [-2.0, -1.0]])
        np.testing.assert_allclose(jitted_step(xs2), xs2 - 10.0)
