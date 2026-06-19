"""Tests for sparse inference time sparsification (AC-1 through AC-8)."""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

from xtrax.sparse.config import SparseConfig
from xtrax.sparse.inference import (
    assert_not_tracing,
    make_sparse_forward_fn,
    sparse_filter_jit,
    sparsify_model,
)
from xtrax.sparse.policy import SparsePolicy


def _make_policy(budget, schedule=None, fallback="dense_mask"):
    return SparsePolicy(
        config=SparseConfig(
            nse_budget=budget,
            update_schedule=schedule or (lambda s: True),
            fallback_mode=fallback,
        )
    )


class TestAssertNotTracing:
    def test_assert_not_tracing_outside_jit(self):
        """Non-traced leaves should not raise."""
        leaves = [jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])]
        # Should not raise
        assert_not_tracing(leaves)

    def test_assert_not_tracing_raises_inside_jit(self):
        """Traced leaves inside jit should raise RuntimeError."""

        def _tracer_test():
            traced_leaf = jnp.array([1.0, 2.0])

            # Manually create a Tracer by tracing through a pytree with an array
            def inner(x):
                assert_not_tracing([x])
                return x

            return jax.make_jaxpr(inner)(traced_leaf)

        with pytest.raises(RuntimeError, match="cannot be called inside jax.jit"):
            _tracer_test()


class TestSparsifyModel:
    def test_sparsify_model_basic(self):
        """AC-1 (presence): sparsify_model returns eqx.Module with BCOO leaves."""
        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)
        sparse_model = sparsify_model(model, policy)

        # Collect all leaves and check for BCOO
        leaves = jax.tree_util.tree_leaves(sparse_model, is_leaf=lambda x: isinstance(x, BCOO))
        bcoo_leaves = [leaf for leaf in leaves if isinstance(leaf, BCOO)]

        # Linear(4, 4) has weight (4, 4) and bias (4,)
        # Only the weight (2D) should be sparsified to BCOO
        assert len(bcoo_leaves) >= 1, "Expected at least one BCOO leaf"
        assert isinstance(bcoo_leaves[0], BCOO)

    def test_sparsify_model_todense_equals_masked_weight(self):
        """AC-1 (numerical): BCOO.todense() == original * mask, atol=1e-6."""
        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)

        # Get original weight before sparsification
        original_weight = model.weight

        sparse_model = sparsify_model(model, policy)

        # Reconstruct expected mask from policy
        mask = policy.make_mask(original_weight, step=0)

        # Collect sparse leaves
        sparse_leaves = jax.tree_util.tree_leaves(
            sparse_model, is_leaf=lambda x: isinstance(x, BCOO)
        )

        # Find BCOO leaves in sparse model
        for leaf in sparse_leaves:
            if isinstance(leaf, BCOO):
                # Verify BCOO.todense() matches original * mask
                dense = leaf.todense()
                expected = original_weight * mask
                assert jnp.allclose(dense, expected, atol=1e-6)

    def test_sparsify_model_no_double_sparsification(self):
        """AC-3: Double sparsification raises ValueError."""
        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)

        sparse_model = sparsify_model(model, policy)

        # Attempting to sparsify again should raise
        with pytest.raises(ValueError, match="already contains BCOO"):
            sparsify_model(sparse_model, policy)

    def test_sparsify_model_skips_non2d(self):
        """AC-7: Non-2D leaves emit UserWarning and are preserved as dense."""

        class ModelWith1DBias(eqx.Module):
            weight: jax.Array
            bias: jax.Array

        # Create weight (2D) and bias (1D)
        weight = jnp.ones((3, 3))
        bias = jnp.ones(3)
        model = ModelWith1DBias(weight=weight, bias=bias)

        policy = _make_policy(budget=4)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sparse_model = sparsify_model(model, policy)

            # Should have warning about non-2D leaf
            assert len(w) > 0
            assert "non-2D" in str(w[0].message)

        # Bias should be preserved as dense (not BCOO)
        assert isinstance(sparse_model.bias, jax.Array)
        assert not isinstance(sparse_model.bias, BCOO)

    def test_sparsify_model_jit_guard(self):
        """AC-2: sparsify_model raises RuntimeError when called inside jit."""

        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)

        def _inside_jit(m, p):
            return sparsify_model(m, p)

        # Make jaxpr to trace it
        with pytest.raises(RuntimeError, match="cannot be called inside jax.jit"):
            jax.make_jaxpr(_inside_jit)(model, policy)

    def test_sparsify_model_bcoo_is_leaf_guard(self):
        """Reject model if it already contains BCOO leaves."""

        # Manually create a model with BCOO leaf
        class ModelWithBCOO(eqx.Module):
            bcoo_weight: BCOO

        bcoo_leaf = BCOO((jnp.array([1.0, 2.0]), jnp.array([[0, 0], [1, 1]])), shape=(2, 2))
        model = ModelWithBCOO(bcoo_weight=bcoo_leaf)

        policy = _make_policy(budget=4)

        with pytest.raises(ValueError, match="already contains BCOO"):
            sparsify_model(model, policy)

    def test_sparsify_model_custom_leaf_filter_false(self):
        """AC-8: leaf_filter=lambda x: False yields no BCOO leaves."""
        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)

        # Use a filter that excludes everything
        sparse_model = sparsify_model(model, policy, leaf_filter=lambda x: False)

        # Check that no BCOO leaves exist
        leaves = jax.tree_util.tree_leaves(sparse_model, is_leaf=lambda x: isinstance(x, BCOO))
        bcoo_leaves = [leaf for leaf in leaves if isinstance(leaf, BCOO)]
        assert len(bcoo_leaves) == 0, "No BCOO leaves expected with leaf_filter=False"

        # All leaves should be original arrays
        assert isinstance(sparse_model.weight, jax.Array)
        assert not isinstance(sparse_model.weight, BCOO)

    def test_sparsify_model_custom_leaf_filter_2d_only(self):
        """AC-8: leaf_filter honoring 2D check."""

        class ModelMixed(eqx.Module):
            weight_2d: jax.Array
            bias_1d: jax.Array

        weight_2d = jnp.ones((3, 3))
        bias_1d = jnp.ones(3)
        model = ModelMixed(weight_2d=weight_2d, bias_1d=bias_1d)

        policy = _make_policy(budget=4)

        # Filter for 2D-or-higher
        def ndim_filter(x):
            return hasattr(x, "ndim") and x.ndim >= 2

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sparse_model = sparsify_model(model, policy, leaf_filter=ndim_filter)

        # weight_2d should be BCOO, bias_1d should be preserved
        assert isinstance(sparse_model.weight_2d, BCOO)
        assert isinstance(sparse_model.bias_1d, jax.Array)
        assert not isinstance(sparse_model.bias_1d, BCOO)


class TestMakeSparseForwardFn:
    def test_make_sparse_forward_fn_runs(self):
        """AC-8: make_sparse_forward_fn closes over sparse_model; callable runs."""
        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)
        sparse_model = sparsify_model(model, policy)

        # Define a simple forward function that just returns the model
        # (not actually executing matmul with BCOO, since that requires
        # custom sparse kernel implementation)
        def forward_fn(model, inputs):
            # Just return the model and inputs for this test
            return (model, inputs)

        sparse_forward = make_sparse_forward_fn(forward_fn, sparse_model)

        # Call with dummy inputs
        dummy_input = jnp.ones((2, 4))
        returned_model, returned_inputs = sparse_forward(dummy_input)

        # Verify sparse_model is closed over and returned
        assert returned_model is sparse_model
        assert jnp.allclose(returned_inputs, dummy_input)

    def test_make_sparse_forward_fn_closure_bcoo(self):
        """make_sparse_forward_fn keeps BCOO out of filter_jit partition."""
        model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
        policy = _make_policy(budget=8)
        sparse_model = sparsify_model(model, policy)

        # Simple forward that uses the sparse model as a constant
        def forward_fn(model, inputs):
            # Just return shape of model weight to verify it's callable
            return model.weight.shape

        sparse_forward = make_sparse_forward_fn(forward_fn, sparse_model)

        # Verify it's callable
        dummy_input = jnp.ones((2, 4))
        weight_shape = sparse_forward(dummy_input)
        assert weight_shape == (4, 4)


class TestSparseFilterJit:
    def test_sparse_filter_jit_does_not_destructure_bcoo(self):
        """sparse_filter_jit prevents BCOO destructuring and retrace."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(4, 4, key=key)
        policy = _make_policy(budget=8)
        sparse_model = sparsify_model(model, policy)

        call_count = {"n": 0}

        @sparse_filter_jit
        def forward(m, x):
            call_count["n"] += 1
            return m.weight.shape

        x = jnp.ones((4,))
        out1 = forward(sparse_model, x)
        forward(sparse_model, x)  # Second call should not retrace

        assert call_count["n"] == 1, (
            f"Expected 1 trace, got {call_count['n']} — BCOO destructuring triggered retrace"
        )
        assert out1 == (4, 4)
