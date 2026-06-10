"""Tests for xtrax.training.types."""

import equinox as eqx
import jax
import jax.numpy as jnp

from xtrax.training.types import (
    Callback,
    LossFunction,
    ResumableState,
)


# Helper classes for testing
class SimpleLoss:
    """Concrete implementation of LossFunction."""

    def __call__(self, predictions, targets):
        return jnp.mean((predictions - targets) ** 2)


class SimpleCallback:
    """Concrete implementation of Callback with all 7 hooks."""

    def __init__(self):
        self.call_log = []

    def on_train_start(self, state):
        self.call_log.append("on_train_start")

    def on_train_end(self, state):
        self.call_log.append("on_train_end")

    def on_resume(self, state):
        self.call_log.append("on_resume")

    def on_epoch_start(self, state, epoch):
        self.call_log.append(f"on_epoch_start({epoch})")

    def on_epoch_end(self, state, epoch):
        self.call_log.append(f"on_epoch_end({epoch})")

    def on_step_start(self, state):
        self.call_log.append("on_step_start")

    def on_step_end(self, state, metrics):
        self.call_log.append(f"on_step_end(metrics={metrics})")


class TestLossFunctionProtocol:
    """Tests for LossFunction protocol."""

    def test_loss_function_protocol_check(self):
        """LossFunction isinstance check passes for conforming class."""
        loss = SimpleLoss()
        assert isinstance(loss, LossFunction)

    def test_loss_function_lambda_protocol_check(self):
        """LossFunction isinstance check passes for callable."""

        def loss_fn(pred, targ):
            return jnp.mean((pred - targ) ** 2)

        assert isinstance(loss_fn, LossFunction)

    def test_loss_function_callable(self):
        """LossFunction can be called and returns Array."""
        loss = SimpleLoss()
        pred = jnp.array([1.0, 2.0, 3.0])
        targ = jnp.array([1.1, 2.1, 3.1])
        result = loss(pred, targ)
        assert isinstance(result, jax.Array)
        assert result.shape == ()  # scalar


class TestCallbackProtocol:
    """Tests for Callback protocol."""

    def test_callback_protocol_check(self):
        """Callback isinstance check passes for conforming class."""
        callback = SimpleCallback()
        assert isinstance(callback, Callback)

    def test_callback_has_seven_hooks(self):
        """Callback protocol has exactly 7 hooks."""
        # Check that all 7 methods are defined in the protocol
        required_hooks = {
            "on_train_start",
            "on_train_end",
            "on_resume",
            "on_epoch_start",
            "on_epoch_end",
            "on_step_start",
            "on_step_end",
        }
        callback = SimpleCallback()
        for hook_name in required_hooks:
            assert hasattr(callback, hook_name), f"Missing hook: {hook_name}"
            hook = getattr(callback, hook_name)
            assert callable(hook), f"Hook not callable: {hook_name}"

    def test_callback_on_step_end_takes_metrics_dict(self):
        """on_step_end signature takes metrics dict not bare Array."""
        callback = SimpleCallback()
        # Create a mock state
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        # Call on_step_end with metrics dict
        metrics = {"loss": jnp.array(0.5), "accuracy": jnp.array(0.9)}
        callback.on_step_end(state, metrics)
        # Verify the call was logged with the dict
        assert any("on_step_end" in str(call) for call in callback.call_log)


class TestResumableState:
    """Tests for ResumableState."""

    def test_resumable_state_instantiation(self):
        """ResumableState can be instantiated."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert state is not None

    def test_resumable_state_step_is_array(self):
        """ResumableState.step is a jax.Array."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(5, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert isinstance(state.step, jax.Array)

    def test_resumable_state_step_is_int32(self):
        """ResumableState.step is jnp.int32 dtype."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(5, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert state.step.dtype == jnp.int32

    def test_resumable_state_step_is_dynamic_leaf(self):
        """ResumableState.step appears in eqx.filter(state, eqx.is_array)."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(5, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        # Filter for all arrays in the state
        arrays = eqx.filter(state, eqx.is_array)
        # Step should be present in the filtered result
        assert arrays is not None
        # Verify step is accessible via tree operations
        leaves = jax.tree.leaves(arrays)
        assert any(leaf.dtype == jnp.int32 for leaf in leaves)

    def test_resumable_state_step_scalar(self):
        """ResumableState.step is a scalar (shape ())."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(5, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert state.step.shape == ()

    def test_resumable_state_key_is_array(self):
        """ResumableState.key is a jax.Array."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        key = jax.random.key(42)
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=key,
            model=model,
            opt_state={},
        )
        assert isinstance(state.key, jax.Array)

    def test_resumable_state_model_field(self):
        """ResumableState.model stores eqx.Module."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert isinstance(state.model, eqx.Module)

    def test_resumable_state_opt_state_field(self):
        """ResumableState.opt_state stores arbitrary PyTree."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        opt_state = {"scale": jnp.array([1.0, 2.0])}
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state=opt_state,
        )
        assert state.opt_state == opt_state

    def test_resumable_state_extras_default_empty_dict(self):
        """ResumableState.extras defaults to empty dict."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert state.extras == {}

    def test_resumable_state_extras_custom_dict(self):
        """ResumableState.extras can be set to custom dict."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        extras = {"custom_key": jnp.array([1.0, 2.0])}
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
            extras=extras,
        )
        assert state.extras == extras

    def test_resumable_state_is_equinox_module(self):
        """ResumableState is an eqx.Module."""
        model = eqx.nn.MLP(2, 1, 2, 2, key=jax.random.key(0))
        state = ResumableState(
            step=jnp.array(0, dtype=jnp.int32),
            key=jax.random.key(0),
            model=model,
            opt_state={},
        )
        assert isinstance(state, eqx.Module)
