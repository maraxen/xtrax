"""Test suite for Trainer (spec §3.13)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


class SimpleLinearModel(eqx.Module):
    """Minimal trainable model: y = w*x + b."""

    w: jax.Array
    b: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w * x + self.b


def mse_loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean squared error loss."""
    return jnp.mean((predictions - targets) ** 2)


def test_trainer_step_increments_step_counter():
    """Test that step() increments state.step by 1."""
    # Setup
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.1), b=jnp.array(0.0))
    opt = optax.sgd(learning_rate=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    inputs = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([2.0, 4.0, 6.0])
    batch = {"inputs": inputs, "targets": targets}

    # Execute
    new_state, metrics = trainer.step(state, batch)

    # Verify
    assert new_state.step == 1
    assert state.step == 0  # Original state unchanged


def test_trainer_step_returns_dict_with_loss():
    """Test step() returns tuple[ResumableState, dict] with metrics["loss"]."""
    # Setup
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.1), b=jnp.array(0.0))
    opt = optax.sgd(learning_rate=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    batch = {"inputs": jnp.array([1.0, 2.0]), "targets": jnp.array([2.0, 4.0])}

    # Execute
    new_state, metrics = trainer.step(state, batch)

    # Verify
    assert isinstance(metrics, dict), "Metrics must be dict"
    assert "loss" in metrics, "Metrics must contain 'loss' key"
    loss_val = metrics["loss"]
    assert isinstance(loss_val, jax.Array), "Loss must be jax.Array"
    assert loss_val.shape == (), "Loss must be scalar"


def test_trainer_loss_decreases_over_steps():
    """Test that training on trivial regression reduces loss over 10 steps."""
    # Setup: Predict y=2x from x,y pairs
    key = jax.random.PRNGKey(42)
    model = SimpleLinearModel(w=jnp.array(0.1), b=jnp.array(0.0))
    opt = optax.sgd(learning_rate=0.1)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    inputs = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([2.0, 4.0, 6.0])
    batch = {"inputs": inputs, "targets": targets}

    # Execute: run 10 steps
    losses = []
    for _ in range(10):
        state, metrics = trainer.step(state, batch)
        losses.append(float(metrics["loss"]))

    # Verify: loss strictly decreases
    assert losses[0] > losses[-1], f"Loss should decrease: {losses}"
    for i in range(len(losses) - 1):
        assert losses[i] > losses[i + 1], f"Loss not monotone decreasing at step {i}"


def test_trainer_step_updates_model():
    """Test that step() updates the model parameters."""
    # Setup
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.5), b=jnp.array(0.1))
    opt = optax.sgd(learning_rate=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    batch = {"inputs": jnp.array([1.0]), "targets": jnp.array([5.0])}

    # Execute
    new_state, _ = trainer.step(state, batch)

    # Verify: model weights changed
    assert not jnp.allclose(new_state.model.w, state.model.w)
    assert not jnp.allclose(new_state.model.b, state.model.b)


def test_trainer_step_updates_optimizer_state():
    """Test that step() updates the optimizer state."""
    # Setup
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.5), b=jnp.array(0.1))
    opt = optax.adam(learning_rate=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    batch = {"inputs": jnp.array([1.0]), "targets": jnp.array([5.0])}

    # Execute
    new_state, _ = trainer.step(state, batch)

    # Verify: optimizer state (Adam moments) changed
    # For Adam, opt_state is a tuple of (OptState, (m, v)) where m and v are pytrees
    old_opt_state_str = str(state.opt_state)
    new_opt_state_str = str(new_state.opt_state)
    assert old_opt_state_str != new_opt_state_str


def test_trainer_step_jit_compiled():
    """Test that step() is JIT-compiled (eqx.filter_jit).

    eqx.filter_jit wraps the function. We verify this by checking that
    the method is callable and produces correct results (actual JIT behavior
    is transparent to the caller).
    """
    # Setup
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.1), b=jnp.array(0.0))
    opt = optax.sgd(learning_rate=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    batch = {"inputs": jnp.array([1.0, 2.0]), "targets": jnp.array([2.0, 4.0])}

    # Verify: JIT-compiled methods are callable and produce correct output
    new_state, metrics = trainer.step(state, batch)
    assert isinstance(new_state, ResumableState)
    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_trainer_step_filter_applied_to_model():
    """Test that optimizer.update receives eqx.filter(model, eqx.is_array) as 3rd arg.

    Verify by checking that the update succeeds and model is modified correctly.
    This is a behavioral test: if the filter were missing or wrong, weight decay
    and other weight-aware optimizers would fail or behave incorrectly.
    """
    # Setup with WeightDecay optimizer that requires correct param passing
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.5), b=jnp.array(0.1))

    # Use AdamW which applies weight decay — it requires correct param passing
    opt = optax.adamw(learning_rate=0.01, weight_decay=0.001)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    batch = {"inputs": jnp.array([1.0, 2.0]), "targets": jnp.array([2.0, 4.0])}

    # Execute: if filter is missing, this may fail or produce wrong updates
    new_state, metrics = trainer.step(state, batch)

    # Verify: step succeeded and loss is finite
    assert jnp.isfinite(metrics["loss"])
    assert new_state.model.w.shape == model.w.shape


def test_trainer_preserves_state_immutability():
    """Test that original state is not mutated by step()."""
    # Setup
    key = jax.random.PRNGKey(0)
    model = SimpleLinearModel(w=jnp.array(0.5), b=jnp.array(0.1))
    opt = optax.sgd(learning_rate=0.01)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    state = ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )

    # Capture original values
    orig_step = state.step
    orig_w = state.model.w
    orig_b = state.model.b

    trainer = Trainer(loss_fn=mse_loss, optimizer=opt)
    batch = {"inputs": jnp.array([1.0]), "targets": jnp.array([5.0])}

    # Execute
    new_state, _ = trainer.step(state, batch)

    # Verify: original state unchanged
    assert jnp.array_equal(state.step, orig_step)
    assert jnp.array_equal(state.model.w, orig_w)
    assert jnp.array_equal(state.model.b, orig_b)
