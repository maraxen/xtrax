"""Tests for SafetyTrainStep and create_train_step (spec §3.14)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.safety.manager import SafetyManager
from xtrax.training.step import SafetyTrainStep, create_train_step
from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


# Fixtures
@pytest.fixture
def simple_model():
    """A minimal trainable model."""

    class SimpleModel(eqx.Module):
        w: jax.Array

        def __call__(self, x):
            return x * self.w

    return SimpleModel(w=jnp.array([1.0]))


@pytest.fixture
def loss_fn():
    """Simple MSE loss function."""

    def mse(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

    return mse


@pytest.fixture
def optimizer():
    """Simple optimizer."""
    return optax.adam(learning_rate=0.01)


@pytest.fixture
def batch():
    """Simple batch of data."""
    return {
        "inputs": jnp.array([1.0, 2.0, 3.0]),
        "targets": jnp.array([2.0, 4.0, 6.0]),
    }


@pytest.fixture
def state(simple_model, optimizer):
    """Create a simple ResumableState."""
    key = jax.random.PRNGKey(0)
    opt_state = optimizer.init(eqx.filter(simple_model, eqx.is_array))
    return ResumableState(
        step=jnp.int32(0),
        key=key,
        model=simple_model,
        opt_state=opt_state,
        extras={},
    )


# Tests


def test_create_train_step_no_safety_returns_trainer(loss_fn, optimizer):
    """create_train_step(safety=False) returns Trainer."""
    trainer = create_train_step(loss_fn, optimizer, safety=False)
    assert isinstance(trainer, Trainer)
    assert not isinstance(trainer, SafetyTrainStep)


def test_create_train_step_with_safety_returns_safety_train_step(loss_fn, optimizer):
    """create_train_step(safety=True) returns SafetyTrainStep."""
    step = create_train_step(loss_fn, optimizer, safety=True)
    assert isinstance(step, SafetyTrainStep)


def test_safety_train_step_has_public_fields(loss_fn, optimizer):
    """SafetyTrainStep has PUBLIC .trainer and .safety_manager fields."""
    step = create_train_step(loss_fn, optimizer, safety=True)
    assert hasattr(step, "trainer")
    assert hasattr(step, "safety_manager")
    assert isinstance(step.trainer, Trainer)
    assert isinstance(step.safety_manager, SafetyManager)


def test_trainer_step_interface(loss_fn, optimizer, batch, state):
    """Trainer.step returns (new_state, dict) with metrics["loss"] as Array."""
    trainer = create_train_step(loss_fn, optimizer, safety=False)
    new_state, metrics = trainer.step(state, batch)

    assert isinstance(new_state, ResumableState)
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], jax.Array)
    assert metrics["loss"].shape == ()  # scalar
    assert new_state.step == state.step + 1


def test_safety_train_step_interface(loss_fn, optimizer, batch, state):
    """SafetyTrainStep.step returns (new_state, dict) with metrics["loss"]."""
    step = create_train_step(loss_fn, optimizer, safety=True)
    new_state, metrics = step.step(state, batch)

    assert isinstance(new_state, ResumableState)
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], jax.Array)
    assert metrics["loss"].shape == ()  # scalar
    assert new_state.step == state.step + 1


def test_safety_train_step_with_disabled_safety_delegates_to_trainer(
    loss_fn, optimizer, batch, state
):
    """SafetyTrainStep with disabled safety delegates to trainer.step."""
    disabled_manager = SafetyManager(enabled=False, check_nans=False, check_infs=False)
    step = create_train_step(
        loss_fn, optimizer, safety=True, safety_manager=disabled_manager
    )

    # With disabled safety, should call trainer.step directly
    new_state, metrics = step.step(state, batch)

    assert isinstance(new_state, ResumableState)
    assert isinstance(metrics, dict)
    assert "loss" in metrics


def test_safety_train_step_with_nan_loss_raises(simple_model, optimizer):
    """SafetyTrainStep with NaN-producing loss raises on host (err.throw() fires)."""

    # Create a loss function that returns NaN via log(-1)
    # (checkify only detects NaN from operations, not literals)
    def nan_loss(predictions, targets):
        return jnp.log(-1.0)  # produces NaN that checkify detects

    step = create_train_step(
        nan_loss,
        optimizer,
        safety=True,
        safety_manager=SafetyManager(enabled=True, check_nans=True, check_infs=False),
    )

    # Build a state and batch
    key = jax.random.PRNGKey(0)
    opt_state = optimizer.init(eqx.filter(simple_model, eqx.is_array))
    state = ResumableState(
        step=jnp.int32(0),
        key=key,
        model=simple_model,
        opt_state=opt_state,
        extras={},
    )
    batch = {
        "inputs": jnp.array([1.0, 2.0, 3.0]),
        "targets": jnp.array([2.0, 4.0, 6.0]),
    }

    # Should raise because checkify detects NaN
    with pytest.raises(Exception):  # checkify.Error or subclass
        step.step(state, batch)


def test_create_train_step_with_custom_safety_manager(loss_fn, optimizer):
    """create_train_step(safety=True, safety_manager=...) uses custom manager."""
    custom_manager = SafetyManager(enabled=True, check_nans=True, check_infs=True)
    step = create_train_step(
        loss_fn, optimizer, safety=True, safety_manager=custom_manager
    )

    assert step.safety_manager is custom_manager


def test_trainer_and_safety_step_produce_same_loss_for_normal_batch(
    loss_fn, optimizer, batch, state
):
    """Trainer and SafetyTrainStep produce the same loss for normal (non-NaN) batch."""
    trainer = create_train_step(loss_fn, optimizer, safety=False)
    safe_step = create_train_step(loss_fn, optimizer, safety=True)

    # Run both with the same state and batch
    _, trainer_metrics = trainer.step(state, batch)
    _, safe_metrics = safe_step.step(state, batch)

    # Losses should match (within floating point tolerance)
    assert jnp.allclose(
        trainer_metrics["loss"], safe_metrics["loss"], atol=1e-5, rtol=1e-5
    )
