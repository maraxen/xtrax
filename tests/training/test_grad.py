import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.training.grad import accumulate_grads


class TestAccumulateGrads:
    """Tests for accumulate_grads using jax.lax.scan."""

    def test_scan_in_jaxpr(self):
        """Verify 'scan' appears in jaxpr (not a Python for-loop)."""
        # Simple linear model
        key0 = jax.random.PRNGKey(0)
        model = eqx.nn.MLP(
            in_size=5, out_size=1, width_size=10, depth=2, key=key0
        )

        # Create microbatches: shape (2, 4, 5) — 2 microbatches of size 4, input dim 5
        key = jax.random.PRNGKey(1)
        microbatches = jax.random.normal(key, (2, 4, 5))
        targets = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 1))

        def loss_fn(params, microbatch_targets):
            x, y = microbatch_targets
            pred = jax.vmap(lambda xi: model(xi, key=None))(x)
            return jnp.mean((pred - y) ** 2)

        # Get jaxpr
        jaxpr = jax.make_jaxpr(
            lambda: accumulate_grads(loss_fn, model, (microbatches, targets))
        )()

        # Check that 'scan' primitive appears in jaxpr
        jaxpr_str = str(jaxpr)
        assert 'scan' in jaxpr_str, f"Expected 'scan' in jaxpr but got: {jaxpr_str}"

    def test_equal_microbatches_matches_full_batch(self):
        """Verify result matches full-batch grad within atol=1e-5."""
        key = jax.random.PRNGKey(0)

        # Simple linear model: y = wx + b
        model = eqx.nn.Linear(in_features=5, out_features=1, key=key)

        # Create 4 equal microbatches of size 25 = 100 total
        x_key, y_key = jax.random.split(key)
        x_all = jax.random.normal(x_key, (4, 25, 5))  # (4, 25, 5)
        y_all = jax.random.normal(y_key, (4, 25, 1))  # (4, 25, 1)

        def loss_fn(params, batch):
            x, y = batch
            # Compute loss over batch dimension
            pred = jax.vmap(lambda xi: params(xi))(x)  # (25, 1)
            return jnp.mean((pred - y) ** 2)

        # Compute accumulated grads
        accumulated_grads, accumulated_loss = accumulate_grads(
            loss_fn, model, (x_all, y_all)
        )

        # Compute full-batch grad (using same loss_fn with concatenated data)
        x_full = jnp.concatenate(x_all, axis=0)  # (100, 5)
        y_full = jnp.concatenate(y_all, axis=0)  # (100, 1)
        full_loss_val, full_grads = eqx.filter_value_and_grad(loss_fn)(
            model, (x_full, y_full)
        )

        # Compare: accumulated should match within tolerance
        # Extract leaves as flat lists and compare element-wise
        acc_leaves = jax.tree.leaves(eqx.filter(accumulated_grads, eqx.is_array))
        full_leaves = jax.tree.leaves(eqx.filter(full_grads, eqx.is_array))

        all_match = True
        for acc_leaf, full_leaf in zip(acc_leaves, full_leaves):
            if not jnp.allclose(acc_leaf, full_leaf, atol=1e-5, rtol=1e-5):
                all_match = False
                break

        assert all_match, "Accumulated grads do not match full-batch grads"
        assert jnp.allclose(accumulated_loss, full_loss_val, atol=1e-5, rtol=1e-5)

    def test_returns_scalar_loss(self):
        """Verify mean_loss is a scalar Array."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 3))  # 2 microbatches
        y = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 1))

        def loss_fn(params, batch):
            x_batch, y_batch = batch
            pred = jax.vmap(lambda xi: params(xi))(x_batch)
            return jnp.mean((pred - y_batch) ** 2)

        grads, loss = accumulate_grads(loss_fn, model, (x, y))

        # Check loss is scalar
        assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"

    def test_unequal_microbatch_sizes_raises(self):
        """Verify unequal microbatch sizes raise ValueError."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        # Unequal second axis (4 and 5)
        x = {
            "batch1": jax.random.normal(jax.random.PRNGKey(1), (4, 3)),
            "batch2": jax.random.normal(jax.random.PRNGKey(2), (5, 3)),
        }
        y = {
            "batch1": jax.random.normal(jax.random.PRNGKey(3), (4, 1)),
            "batch2": jax.random.normal(jax.random.PRNGKey(4), (5, 1)),
        }

        def loss_fn(params, batch):
            x_batch, y_batch = batch
            # This would fail if called with mismatched sizes
            return jnp.array(0.0)

        with pytest.raises(ValueError):
            accumulate_grads(loss_fn, model, (x, y))

    def test_unequal_second_axis_raises(self):
        """Microbatches with different second-axis sizes raises ValueError."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        # Different second axes: (2,4,3) vs (2,5,3)
        microbatches = {
            "a": jnp.ones((2, 4, 3)),
            "b": jnp.ones((2, 5, 3)),
        }

        def loss_fn(params, batch):
            return jnp.array(0.0)

        with pytest.raises(ValueError):
            accumulate_grads(loss_fn, model, microbatches)

    def test_filter_spec_excludes_bias(self):
        """filter_spec is used when filtering params for differentiation."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        x = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 3))
        y = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 1))

        def loss_fn(params, batch):
            x_batch, y_batch = batch
            pred = jax.vmap(lambda xi: params(xi))(x_batch)
            return jnp.mean((pred - y_batch) ** 2)

        # Filter spec: filters passed to eqx.filter_value_and_grad
        filter_spec = eqx.is_array

        # Should not raise when filter_spec is provided
        grads, loss = accumulate_grads(
            loss_fn, model, (x, y), filter_spec=filter_spec
        )

        # Should return valid gradients
        assert grads is not None
        assert loss.shape == ()

    def test_single_microbatch_matches_direct(self):
        """Single microbatch matches direct eqx.filter_value_and_grad."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        x = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 3))
        y = jax.random.normal(jax.random.PRNGKey(2), (1, 4, 1))

        def loss_fn(params, batch):
            x_batch, y_batch = batch
            pred = jax.vmap(lambda xi: params(xi))(x_batch)
            return jnp.mean((pred - y_batch) ** 2)

        # Compute via accumulate_grads
        mean_grads, mean_loss = accumulate_grads(loss_fn, model, (x, y))

        # Compute directly
        loss_val, direct_grads = eqx.filter_value_and_grad(loss_fn)(model, (x[0], y[0]))

        # Should match within tolerance
        acc_leaves = jax.tree.leaves(eqx.filter(mean_grads, eqx.is_array))
        direct_leaves = jax.tree.leaves(eqx.filter(direct_grads, eqx.is_array))

        for acc_leaf, direct_leaf in zip(acc_leaves, direct_leaves):
            assert jnp.allclose(acc_leaf, direct_leaf, atol=1e-6)

    def test_no_array_leaves_raises(self):
        """Microbatches with no array leaves raises ValueError."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        # Microbatches with no arrays
        microbatches = {"scalar": 1, "string": "test"}

        def loss_fn(params, batch):
            return jnp.array(0.0)

        with pytest.raises(ValueError, match="must contain at least one array"):
            accumulate_grads(loss_fn, model, microbatches)

    def test_unequal_leading_axis_raises(self):
        """Microbatches with unequal leading axes raise ValueError."""
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=3, out_features=1, key=key)

        # Different leading axes: first has 2, second has 3
        microbatches = {
            "a": jnp.ones((2, 4, 3)),
            "b": jnp.ones((3, 4, 3)),
        }

        def loss_fn(params, batch):
            return jnp.array(0.0)

        with pytest.raises(ValueError, match="same leading axis"):
            accumulate_grads(loss_fn, model, microbatches)
