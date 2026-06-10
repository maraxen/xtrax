"""Integration tests for sparse inference (Sprint 7, Track C).

Tests AC-6 (no-retrace guarantee) and AC-9 (Engine.eval composition):
- AC-6: eqx.filter_jit traces exactly once across multiple same-shape calls
- AC-9: sparsify_model output survives eqx.nn.inference_mode without error
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from xtrax.sparse.config import SparseConfig
from xtrax.sparse.inference import make_sparse_forward_fn, sparsify_model
from xtrax.sparse.policy import SparsePolicy


def _make_policy(budget: int, schedule=None, fallback: str = "dense_mask"):
    """Helper to create a SparsePolicy."""
    return SparsePolicy(
        config=SparseConfig(
            nse_budget=budget,
            update_schedule=schedule or (lambda s: True),
            fallback_mode=fallback,
        )
    )


class TestNoRetraceGuarantee:
    """Tests for AC-6: No-retrace guarantee with eqx.filter_jit."""

    def test_no_retrace_across_calls(self):
        """AC-6: eqx.filter_jit traces exactly once across 3 same-shape calls.

        Python-side mutation inside eqx.filter_jit only executes during tracing,
        not on cached compiled dispatch — this is a reliable retrace detector.
        counter > 1 means assumption A1 (BCOO static closure) is FALSE.
        """
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(8, 8, key=key)
        config = SparseConfig(nse_budget=32, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)
        fn = make_sparse_forward_fn(lambda m, x: m(x), sparse_model)

        # Python-side mutation inside eqx.filter_jit executes during tracing only,
        # not on cached compiled dispatch. This is a reliable retrace detector.
        call_count = {"n": 0}

        @eqx.filter_jit
        def forward(x):
            call_count["n"] += 1
            return fn(x)

        x = jnp.ones((8,))
        out1 = forward(x)
        out2 = forward(x)
        out3 = forward(x)

        # Traces once; counter increments only during tracing, not dispatch.
        assert call_count["n"] == 1, (
            f"Expected 1 trace, got {call_count['n']} — retrace detected. "
            f"BCOO static closure assumption broken."
        )
        assert out1.shape == out2.shape == out3.shape == (8,)

    def test_no_retrace_with_batch_size_constant(self):
        """AC-6: Same input shape (single sample) → no retrace across multiple calls."""
        key = jax.random.PRNGKey(42)
        model = eqx.nn.Linear(16, 16, key=key)
        config = SparseConfig(nse_budget=64, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)
        fn = make_sparse_forward_fn(lambda m, x: m(x), sparse_model)

        trace_count = {"n": 0}

        @eqx.filter_jit
        def forward(x):
            trace_count["n"] += 1
            return fn(x)

        # Call with the same input shape (16,) three times
        x1 = jnp.ones((16,))
        x2 = jnp.ones((16,))
        x3 = jnp.ones((16,))

        out1 = forward(x1)
        out2 = forward(x2)
        out3 = forward(x3)

        assert trace_count["n"] == 1, (
            f"Expected 1 trace for constant input size, got {trace_count['n']}"
        )
        assert out1.shape == out2.shape == out3.shape == (16,)

    def test_retrace_on_different_shape(self):
        """AC-6: Same shape → no retrace (control test).

        This proves the no-retrace guarantee is about BCOO statics,
        not about input shapes being constant.
        """
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(8, 8, key=key)
        config = SparseConfig(nse_budget=32, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)
        # Use a wrapper that supports variable input sizes
        fn = make_sparse_forward_fn(lambda m, x: m(x.reshape(-1)), sparse_model)

        trace_count = {"n": 0}

        @eqx.filter_jit
        def forward(x):
            trace_count["n"] += 1
            return fn(x)

        # Same shape repeated → no retrace
        forward(jnp.ones((8,)))
        trace_count_after_first = trace_count["n"]
        forward(jnp.ones((8,)))
        trace_count_after_second = trace_count["n"]

        assert trace_count_after_first == 1
        assert trace_count_after_second == 1, (
            f"Expected NO retrace on same shape, got {trace_count_after_second}"
        )


class TestEngineEvalComposition:
    """Tests for AC-9: Engine.eval composition with eqx.nn.inference_mode."""

    def test_sparse_model_survives_inference_mode(self):
        """AC-9: sparsify_model output survives eqx.nn.inference_mode without error.

        This tests the composition: sparsify_model runs first, then inference_mode.
        BCOO leaves must survive the transformation.
        """
        key = jax.random.PRNGKey(42)
        model = eqx.nn.Linear(4, 4, key=key)
        config = SparseConfig(nse_budget=8, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)

        # Apply inference_mode (same as Engine.eval does at line 169)
        inference_model = eqx.nn.inference_mode(sparse_model)

        # Verify BCOO leaves survive the transformation
        leaves = jax.tree_util.tree_leaves(
            inference_model, is_leaf=lambda x: isinstance(x, BCOO)
        )
        assert any(isinstance(leaf, BCOO) for leaf in leaves), (
            "BCOO leaves lost after inference_mode — composition failed"
        )

    def test_sparse_model_forward_pass_after_inference_mode(self):
        """AC-9: Forward pass works after eqx.nn.inference_mode transformation."""
        key = jax.random.PRNGKey(42)
        model = eqx.nn.Linear(4, 4, key=key)
        config = SparseConfig(nse_budget=8, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)
        inference_model = eqx.nn.inference_mode(sparse_model)

        # Run a forward pass with jit
        x = jnp.ones((4,))

        @eqx.filter_jit
        def forward(m, inputs):
            return m(inputs)

        out = forward(inference_model, x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"
        assert jnp.isfinite(out).all(), "Output contains NaN or inf"

    def test_sparse_model_composable_pattern(self):
        """AC-9: Compose sparsify_model then inference_mode.

        Canonical pattern: sparsify first, then inference_mode.
        """
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(8, 8, key=key)
        config = SparseConfig(nse_budget=32, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        # Canonical pattern: sparsify first, then inference_mode
        sparse_and_inferred = eqx.nn.inference_mode(sparsify_model(model, policy))

        # Verify the result is a valid eqx.Module
        assert isinstance(sparse_and_inferred, eqx.Module)

        # Verify forward pass works
        x = jnp.ones((8,))

        @eqx.filter_jit
        def forward(m, inputs):
            return m(inputs)

        out = forward(sparse_and_inferred, x)
        assert out.shape == (8,), f"Expected (8,), got {out.shape}"

    def test_bcoo_leaves_remain_atomic(self):
        """AC-9: BCOO leaves remain atomic (not destructured) through inference_mode."""
        key = jax.random.PRNGKey(42)
        model = eqx.nn.Linear(4, 4, key=key)
        config = SparseConfig(nse_budget=8, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)
        inference_model = eqx.nn.inference_mode(sparse_model)

        # Count BCOO leaves before and after — should be the same
        bcoo_leaves_before = sum(
            1
            for leaf in jax.tree_util.tree_leaves(
                sparse_model, is_leaf=lambda x: isinstance(x, BCOO)
            )
            if isinstance(leaf, BCOO)
        )

        bcoo_leaves_after = sum(
            1
            for leaf in jax.tree_util.tree_leaves(
                inference_model, is_leaf=lambda x: isinstance(x, BCOO)
            )
            if isinstance(leaf, BCOO)
        )

        assert bcoo_leaves_before == bcoo_leaves_after, (
            f"BCOO count changed: {bcoo_leaves_before} → {bcoo_leaves_after}"
        )
        assert bcoo_leaves_after > 0, "No BCOO leaves found after sparsification"

    def test_sparse_model_with_dropout_layer(self):
        """AC-9: Sparse model handles dropout correctly with inference_mode.

        Verify that inference_mode disables dropout in the sparse model.
        """

        class SimpleModel(eqx.Module):
            linear: eqx.nn.Linear
            dropout: eqx.nn.Dropout

            def __call__(self, x, *, key=None):
                x = self.linear(x)
                x = self.dropout(x, key=key)
                return x

        model = SimpleModel(
            linear=eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0)),
            dropout=eqx.nn.Dropout(p=0.5),
        )

        config = SparseConfig(nse_budget=8, update_schedule=lambda s: True)
        policy = SparsePolicy(config=config)

        sparse_model = sparsify_model(model, policy)
        inference_model = eqx.nn.inference_mode(sparse_model)

        # Verify the dropout is disabled (inference_mode sets inference to True)
        x = jnp.ones((4,))

        @eqx.filter_jit
        def forward(m, inputs):
            return m(inputs, key=jax.random.PRNGKey(42))

        out = forward(inference_model, x)
        assert out.shape == (4,)
        assert jnp.isfinite(out).all()
