"""Comprehensive tests for Engine (spec §3.18)."""

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.engine.engine import Engine
from xtrax.training.step import SafetyTrainStep
from xtrax.training.trainer import Trainer
from xtrax.training.types import Callback, ResumableState

# ============================================================================
# Test Helpers
# ============================================================================


class DummyLoss:
    """Simple MSE loss for testing."""

    def __call__(self, predictions: Any, targets: Any) -> jax.Array:
        return jnp.mean((predictions - targets) ** 2)


class DummyModel(eqx.Module):
    """Minimal trainable model for testing."""

    weight: jax.Array = eqx.field()

    def __init__(self, key):
        self.weight = jax.random.normal(key, (2,))

    def __call__(self, x):
        return x @ self.weight


class TrackingCallback:
    """Callback that records all hook calls for verification."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.call_log: list[tuple[str, ...]] = []

    def on_train_start(self, state: ResumableState) -> None:
        self.call_log.append(("on_train_start",))

    def on_train_end(self, state: ResumableState) -> None:
        self.call_log.append(("on_train_end",))

    def on_resume(self, state: ResumableState) -> None:
        # Not called per DEVIATION NOTE; included for protocol completeness
        self.call_log.append(("on_resume",))

    def on_epoch_start(self, state: ResumableState, epoch: int) -> None:
        self.call_log.append(("on_epoch_start", epoch))

    def on_epoch_end(self, state: ResumableState, epoch: int) -> None:
        self.call_log.append(("on_epoch_end", epoch))

    def on_step_start(self, state: ResumableState) -> None:
        self.call_log.append(("on_step_start",))

    def on_step_end(self, state: ResumableState, metrics: dict[str, jax.Array]) -> None:
        # Verify metrics is a dict
        assert isinstance(metrics, dict), f"metrics must be dict, got {type(metrics)}"
        self.call_log.append(("on_step_end", tuple(sorted(metrics.keys()))))


class DummyDataModule:
    """Simple data provider (not an eqx.Module to avoid init issues)."""

    def __init__(self, batch_count: int = 3, batch_size: int = 2):
        self.batch_count = batch_count
        self.batch_size = batch_size

    def train_iter(self) -> Iterator[dict[str, Any]]:
        """Generate dummy batches for training."""
        for i in range(self.batch_count):
            inputs = jax.random.normal(jax.random.PRNGKey(i), (self.batch_size, 2))
            targets = jax.random.normal(jax.random.PRNGKey(i + 100), (self.batch_size,))
            yield {"inputs": inputs, "targets": targets}

    def eval_iter(self) -> Iterator[dict[str, Any]]:
        """Generate dummy batches for evaluation."""
        for i in range(self.batch_count):
            inputs = jax.random.normal(
                jax.random.PRNGKey(i + 200), (self.batch_size, 2)
            )
            targets = jax.random.normal(jax.random.PRNGKey(i + 300), (self.batch_size,))
            yield {"inputs": inputs, "targets": targets}


def create_test_engine(
    callbacks: tuple[Callback, ...] = (),
    validation_callbacks: tuple[Callback, ...] = (),
) -> Engine:
    """Create a test Engine with Trainer."""
    loss_fn = DummyLoss()
    optimizer = optax.adam(learning_rate=0.001)
    trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
    return Engine(
        trainer=trainer,
        callbacks=callbacks,
        validation_callbacks=validation_callbacks,
    )


def create_test_state() -> ResumableState:
    """Create a minimal test ResumableState."""
    key = jax.random.PRNGKey(0)
    model = DummyModel(key)
    opt_state = optax.adam(0.001).init(eqx.filter(model, eqx.is_array))
    return ResumableState(
        step=jnp.array(0, dtype=jnp.int32),
        key=key,
        model=model,
        opt_state=opt_state,
    )


class MultiMetricLoss:
    """Loss function that also tracks accuracy for multi-metric testing."""

    def __call__(self, predictions: Any, targets: Any) -> jax.Array:
        return jnp.mean((predictions - targets) ** 2)


class MultiMetricTrainer(eqx.Module):
    """Mock trainer that returns multiple metrics (loss + accuracy)."""

    loss_fn: Any = eqx.field(static=True)
    optimizer: Any = eqx.field(static=True)

    def __init__(self, loss_fn: Any, optimizer: Any):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def step(
        self, state: ResumableState, batch: dict[str, Any]
    ) -> tuple[ResumableState, dict[str, jax.Array]]:
        """Return metrics dict with both loss and accuracy."""
        # Compute loss (simplified for testing)
        predictions = state.model(batch["inputs"])
        loss = self.loss_fn(predictions, batch["targets"])

        # Compute accuracy (simplified: proportion of predictions within 0.5 of targets)
        errors = jnp.abs(predictions - batch["targets"])
        accuracy = jnp.mean(errors < 0.5).astype(jnp.float32)

        # Return state with incremented step and multi-key metrics dict
        new_state = eqx.tree_at(
            lambda s: s.step, state, state.step + jnp.array(1, dtype=jnp.int32)
        )
        return new_state, {"loss": loss, "accuracy": accuracy}


def create_test_engine_with_multi_metrics(
    callbacks: tuple[Callback, ...] = (),
    validation_callbacks: tuple[Callback, ...] = (),
) -> Engine:
    """Create a test Engine with a trainer that returns multi-key metrics."""
    loss_fn = MultiMetricLoss()
    optimizer = optax.adam(learning_rate=0.001)
    trainer = MultiMetricTrainer(loss_fn=loss_fn, optimizer=optimizer)
    return Engine(
        trainer=trainer,
        callbacks=callbacks,
        validation_callbacks=validation_callbacks,
    )


# ============================================================================
# Tests
# ============================================================================


class TestEngineStructure:
    """Tests for Engine as eqx.Module."""

    def test_engine_is_eqx_module(self):
        """Engine must be an eqx.Module."""
        engine = create_test_engine()
        assert isinstance(engine, eqx.Module)

    def test_engine_fields_static(self):
        """trainer, callbacks, validation_callbacks must be static fields."""
        engine = create_test_engine()
        # Check that fields are present
        assert hasattr(engine, "trainer")
        assert hasattr(engine, "callbacks")
        assert hasattr(engine, "validation_callbacks")
        # Verify they are indeed static by checking they're tuples/objects, not arrays
        assert isinstance(engine.callbacks, tuple)
        assert isinstance(engine.validation_callbacks, tuple)

    def test_engine_accepts_trainer(self):
        """Engine must accept Trainer as trainer field."""
        loss_fn = DummyLoss()
        optimizer = optax.adam(0.001)
        trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
        engine = Engine(trainer=trainer, callbacks=(), validation_callbacks=())
        assert isinstance(engine.trainer, Trainer)

    def test_engine_accepts_safetytrainstep(self):
        """Engine must accept SafetyTrainStep as trainer field."""
        from xtrax.safety.manager import SafetyManager

        loss_fn = DummyLoss()
        optimizer = optax.adam(0.001)
        trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
        safety_mgr = SafetyManager(enabled=False, check_nans=False, check_infs=False)
        safe_step = SafetyTrainStep(trainer=trainer, safety_manager=safety_mgr)
        engine = Engine(trainer=safe_step, callbacks=(), validation_callbacks=())
        assert isinstance(engine.trainer, SafetyTrainStep)

    def test_engine_default_validation_callbacks(self):
        """validation_callbacks should default to empty tuple."""
        loss_fn = DummyLoss()
        optimizer = optax.adam(0.001)
        trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
        engine = Engine(trainer=trainer, callbacks=())
        assert engine.validation_callbacks == ()
        assert isinstance(engine.validation_callbacks, tuple)


@pytest.mark.asyncio
class TestEngineFit:
    """Tests for Engine.fit() method."""

    async def test_fit_increments_step(self):
        """fit should increment state.step by 1 per batch."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=3, batch_size=2)

        initial_step = int(state.step)
        final_state = await engine.fit(state, data, num_epochs=2)

        expected_steps = 3 * 2  # 3 batches per epoch × 2 epochs
        final_step = int(final_state.step)
        assert final_step == initial_step + expected_steps

    async def test_fit_fires_on_train_start_once(self):
        """on_train_start should fire exactly once at start."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=2, batch_size=1)

        await engine.fit(state, data, num_epochs=1)

        on_train_start_calls = [x for x in cb.call_log if x[0] == "on_train_start"]
        assert len(on_train_start_calls) == 1

    async def test_fit_fires_on_train_end_once(self):
        """on_train_end should fire exactly once at end."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=2, batch_size=1)

        await engine.fit(state, data, num_epochs=1)

        on_train_end_calls = [x for x in cb.call_log if x[0] == "on_train_end"]
        assert len(on_train_end_calls) == 1

    async def test_fit_fires_on_epoch_start_per_epoch(self):
        """on_epoch_start should fire exactly num_epochs times."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=2, batch_size=1)
        num_epochs = 3

        await engine.fit(state, data, num_epochs=num_epochs)

        on_epoch_start_calls = [x for x in cb.call_log if x[0] == "on_epoch_start"]
        assert len(on_epoch_start_calls) == num_epochs

    async def test_fit_fires_on_epoch_end_per_epoch(self):
        """on_epoch_end should fire exactly num_epochs times."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=2, batch_size=1)
        num_epochs = 3

        await engine.fit(state, data, num_epochs=num_epochs)

        on_epoch_end_calls = [x for x in cb.call_log if x[0] == "on_epoch_end"]
        assert len(on_epoch_end_calls) == num_epochs

    async def test_fit_fires_on_step_start_per_step(self):
        """on_step_start should fire once per batch."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        batch_count = 3
        num_epochs = 2
        data = DummyDataModule(batch_count=batch_count, batch_size=1)

        await engine.fit(state, data, num_epochs=num_epochs)

        on_step_start_calls = [x for x in cb.call_log if x[0] == "on_step_start"]
        assert len(on_step_start_calls) == batch_count * num_epochs

    async def test_fit_fires_on_step_end_per_step(self):
        """on_step_end should fire once per batch."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        batch_count = 3
        num_epochs = 2
        data = DummyDataModule(batch_count=batch_count, batch_size=1)

        await engine.fit(state, data, num_epochs=num_epochs)

        on_step_end_calls = [x for x in cb.call_log if x[0] == "on_step_end"]
        assert len(on_step_end_calls) == batch_count * num_epochs

    async def test_fit_on_step_end_receives_metrics_dict(self):
        """on_step_end should receive a dict[str, Array]."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        await engine.fit(state, data, num_epochs=1)

        on_step_end_calls = [x for x in cb.call_log if x[0] == "on_step_end"]
        assert len(on_step_end_calls) > 0
        # TrackingCallback verifies metrics is a dict in on_step_end
        # If metrics wasn't a dict, we'd have gotten an assertion error

    async def test_fit_callback_hook_order(self):
        """Callback hooks should fire in correct order."""
        cb = TrackingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        await engine.fit(state, data, num_epochs=1)

        # Verify order: train_start, epoch_start, step_start, step_end,
        # epoch_end, train_end
        call_names = [call[0] for call in cb.call_log]
        assert call_names[0] == "on_train_start"
        assert call_names[1] == "on_epoch_start"
        assert call_names[2] == "on_step_start"
        assert call_names[3] == "on_step_end"
        assert call_names[4] == "on_epoch_end"
        assert call_names[-1] == "on_train_end"

    async def test_fit_with_multiple_callbacks(self):
        """Multiple callbacks should all receive all hooks."""
        cb1 = TrackingCallback("cb1")
        cb2 = TrackingCallback("cb2")
        engine = create_test_engine(callbacks=(cb1, cb2))
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        await engine.fit(state, data, num_epochs=1)

        # Both callbacks should have been called
        assert len(cb1.call_log) > 0
        assert len(cb2.call_log) > 0
        # Both should have same structure
        assert len(cb1.call_log) == len(cb2.call_log)

    async def test_fit_returns_final_state(self):
        """fit should return the final ResumableState."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        final_state = await engine.fit(state, data, num_epochs=1)

        assert isinstance(final_state, ResumableState)
        assert int(final_state.step) > int(state.step)

    async def test_fit_updates_model(self):
        """fit should update the model weights."""
        engine = create_test_engine()
        state = create_test_state()
        initial_weight = state.model.weight.copy()
        data = DummyDataModule(batch_count=1, batch_size=1)

        final_state = await engine.fit(state, data, num_epochs=1)

        # Weights should have changed after training
        final_weight = final_state.model.weight
        assert not jnp.allclose(initial_weight, final_weight)

    async def test_fit_on_train_end_fires_on_exception(self):
        """on_train_end should fire even if exception occurs."""

        class FailingCallback:
            def __init__(self):
                self.on_train_end_called = False

            def on_train_start(self, state):
                raise RuntimeError("Intentional failure in on_train_start")

            def on_train_end(self, state):
                self.on_train_end_called = True

            def on_resume(self, state):
                pass

            def on_epoch_start(self, state, epoch):
                pass

            def on_epoch_end(self, state, epoch):
                pass

            def on_step_start(self, state):
                pass

            def on_step_end(self, state, metrics):
                pass

        cb = FailingCallback()
        engine = create_test_engine(callbacks=(cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        with pytest.raises(RuntimeError, match="Intentional failure"):
            await engine.fit(state, data, num_epochs=1)

        assert cb.on_train_end_called


@pytest.mark.asyncio
class TestEngineEval:
    """Tests for Engine.eval() method."""

    async def test_eval_returns_metrics_dict(self):
        """eval should return dict[str, Array]."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=2, batch_size=2)

        metrics = await engine.eval(state, data)

        assert isinstance(metrics, dict)
        assert all(isinstance(k, str) for k in metrics.keys())

    async def test_eval_does_not_modify_state_step(self):
        """eval should not modify state.step."""
        engine = create_test_engine()
        state = create_test_state()
        initial_step = int(state.step)
        data = DummyDataModule(batch_count=2, batch_size=2)

        await engine.eval(state, data)

        assert int(state.step) == initial_step

    async def test_eval_inference_mode_wrapping(self):
        """eval should wrap model in inference_mode."""
        # This is verified implicitly by the fact that eval works
        # and doesn't error. Explicit verification would require
        # a model with dropout/stochastic layers.
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        metrics = await engine.eval(state, data)

        assert isinstance(metrics, dict)

    async def test_eval_aggregates_metrics(self):
        """eval should aggregate metrics across batches via jnp.mean."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=3, batch_size=2)

        metrics = await engine.eval(state, data)

        # Should have "loss" key from trainer.step
        assert "loss" in metrics
        # Loss should be a scalar Array
        assert metrics["loss"].shape == ()

    async def test_eval_fires_validation_callbacks(self):
        """eval should fire validation_callbacks."""
        # Create a simple validation callback tracker
        val_cb = TrackingCallback()
        engine = create_test_engine(validation_callbacks=(val_cb,))
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        # Note: eval fires validation_callbacks at start and end
        metrics = await engine.eval(state, data)
        assert isinstance(metrics, dict)
        # Verify that at least one validation callback hook was invoked
        assert len(val_cb.call_log) > 0

    async def test_eval_with_custom_loss_fn(self):
        """eval should accept optional loss_fn."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        def custom_loss(predictions, targets):
            return jnp.sum((predictions - targets) ** 2)

        metrics = await engine.eval(state, data, loss_fn=custom_loss)

        assert "loss" in metrics

    async def test_eval_empty_data_returns_empty_dict(self):
        """eval with empty dataset should return empty dict."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=0, batch_size=1)

        metrics = await engine.eval(state, data)

        assert metrics == {}

    async def test_eval_aggregates_multi_key_metrics(self):
        """eval should aggregate all keys in multi-key metrics dict across batches."""
        # Use custom trainer that returns {"loss": ..., "accuracy": ...}
        engine = create_test_engine_with_multi_metrics()
        state = create_test_state()
        # Use batch_count >= 2 to test aggregation across multiple batches
        data = DummyDataModule(batch_count=2, batch_size=2)

        metrics = await engine.eval(state, data)

        # Both keys must be present
        assert "loss" in metrics
        assert "accuracy" in metrics
        # Both must be scalar shape-() Arrays
        loss_shape = metrics["loss"].shape
        accuracy_shape = metrics["accuracy"].shape
        assert loss_shape == (), f"Expected loss shape (), got {loss_shape}"
        assert accuracy_shape == (), f"Expected accuracy shape (), got {accuracy_shape}"
        # Both must be finite (no NaN/Inf)
        assert jnp.isfinite(metrics["loss"])
        assert jnp.isfinite(metrics["accuracy"])


class TestEngineFitSync:
    """Tests for Engine.fit_sync() wrapper."""

    def test_fit_sync_delegates_to_fit(self):
        """fit_sync should delegate to asyncio.run(fit(...))."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=2, batch_size=1)

        final_state = engine.fit_sync(state, data, num_epochs=1)

        assert isinstance(final_state, ResumableState)
        assert int(final_state.step) > int(state.step)

    def test_fit_sync_returns_final_state(self):
        """fit_sync should return the same result as asyncio.run(fit(...))."""
        engine = create_test_engine()
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        final_state = engine.fit_sync(state, data, num_epochs=1)

        assert final_state.step.shape == ()  # Scalar Array


class TestEnginePolymorphism:
    """Tests for Engine polymorphism with Trainer variants."""

    @pytest.mark.asyncio
    async def test_fit_with_safetytrainstep(self):
        """Engine.fit should work with SafetyTrainStep trainer."""
        from xtrax.safety.manager import SafetyManager

        loss_fn = DummyLoss()
        optimizer = optax.adam(0.001)
        trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer)
        safety_mgr = SafetyManager(enabled=False, check_nans=False, check_infs=False)
        safe_step = SafetyTrainStep(trainer=trainer, safety_manager=safety_mgr)

        engine = Engine(trainer=safe_step, callbacks=(), validation_callbacks=())
        state = create_test_state()
        data = DummyDataModule(batch_count=1, batch_size=1)

        final_state = await engine.fit(state, data, num_epochs=1)

        assert int(final_state.step) > int(state.step)
