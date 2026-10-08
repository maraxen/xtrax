"""Engine validation, early stopping, restore, and step-cadence checkpoints."""

from collections.abc import Iterator
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from xtrax.checkpoint.orbax import get_checkpoint_manager, save_checkpoint
from xtrax.engine.engine import EarlyStopping, Engine
from xtrax.telemetry.ledger import iter_rows
from xtrax.telemetry.record import KIND_TRAIN, STATUS_COMPLETE
from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState


class _Loss:
    def __call__(self, predictions, targets):
        return jnp.mean((predictions - targets) ** 2)


class _Model(eqx.Module):
    weight: jax.Array

    def __init__(self, key):
        self.weight = jax.random.normal(key, (2,))

    def __call__(self, x):
        return x @ self.weight


class _Data:
    def __init__(self, batch_count: int):
        self.batch_count = batch_count

    def train_iter(self) -> Iterator[dict[str, Any]]:
        for i in range(self.batch_count):
            yield {
                "inputs": jnp.ones((2, 2)) + i,
                "targets": jnp.ones((2,)),
            }

    def eval_iter(self) -> Iterator[dict[str, Any]]:
        return iter(())


class _EpochCounter:
    def __init__(self):
        self.epochs: list[int] = []
        self.train_end = 0

    def on_train_start(self, state):
        return None

    def on_train_end(self, state):
        self.train_end += 1

    def on_resume(self, state):
        return None

    def on_epoch_start(self, state, epoch):
        return None

    def on_epoch_end(self, state, epoch):
        self.epochs.append(epoch)

    def on_step_start(self, state):
        return None

    def on_step_end(self, state, metrics):
        return None


def _engine(**kwargs) -> Engine:
    trainer = Trainer(loss_fn=_Loss(), optimizer=optax.sgd(1e-3))
    return Engine(trainer=trainer, callbacks=kwargs.get("callbacks", ()), validation_callbacks=())


def _state(**overrides) -> ResumableState:
    key = overrides.get("key", jax.random.PRNGKey(0))
    model = overrides.get("model", _Model(key))
    return ResumableState(
        step=overrides.get("step", jnp.array(0, dtype=jnp.int32)),
        key=key,
        model=model,
        opt_state=optax.sgd(1e-3).init(eqx.filter(model, eqx.is_array)),
        extras=overrides.get("extras", {}),
    )


def test_validate_every_steps_records_step_cadence():
    calls: list[int] = []

    def validate_fn(state):
        calls.append(int(state.step))
        return {"val_loss": jnp.array(1.0)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=6),
        num_epochs=1,
        validate_fn=validate_fn,
        validate_every_steps=2,
    )

    assert calls == [2, 4, 6]
    assert int(final.step) == 6


def test_validate_every_epochs_records_epoch_cadence():
    calls: list[int] = []

    def validate_fn(state):
        calls.append(int(state.step))
        return {"val_loss": jnp.array(1.0)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=4,
        validate_fn=validate_fn,
        validate_every_epochs=2,
    )

    assert calls == [2, 4]
    assert int(final.step) == 4


def test_validate_fn_defaults_to_every_epoch():
    """With neither cadence set, validate_fn runs once per completed epoch."""
    calls: list[int] = []

    def validate_fn(state):
        calls.append(int(state.step))
        return {"val_loss": jnp.array(1.0)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=2),
        num_epochs=3,
        validate_fn=validate_fn,
    )

    assert calls == [2, 4, 6]
    assert int(final.step) == 6


def test_early_stopping_min_mode_patience_and_min_delta():
    """mode=min, patience, and min_delta all participate in the stop decision."""
    schedule = [10.0, 5.0, 4.0, 4.0]
    seen: list[float] = []
    callback = _EpochCounter()

    def validate_fn(state):
        value = schedule[len(seen)]
        seen.append(value)
        return {"val_loss": jnp.asarray(value)}

    final = _engine(callbacks=(callback,)).fit_sync(
        _state(),
        _Data(batch_count=2),
        num_epochs=8,
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=2, min_delta=2.0),
    )

    # 10 sets the best. 5 improves (10 - 2). 4 and 4 do not (5 - 2 = 3). Stop.
    assert seen == [10.0, 5.0, 4.0, 4.0]
    assert int(final.step) == 8
    assert callback.epochs == [0, 1, 2, 3]
    assert callback.train_end == 1


def test_early_stopping_max_mode():
    schedule = [1.0, 3.0, 3.0, 3.0]
    seen: list[float] = []

    def validate_fn(state):
        value = schedule[len(seen)]
        seen.append(value)
        return {"score": jnp.asarray(value)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=2),
        num_epochs=6,
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="score", mode="max", patience=2, min_delta=0.5),
    )

    assert seen == schedule
    assert int(final.step) == 8


def test_early_stop_resets_wait_when_the_metric_improves():
    """A stall before an improvement must not carry into the next stall.

    Schedule [10, 10, 5, 5, 5], mode=min, patience=2:
    10 sets the best, the second 10 waits once, 5 improves and clears the
    wait, then two non-improving 5s exhaust patience. Stopping on the fourth
    check means the improvement did not reset ``wait``.
    """
    schedule = [10.0, 10.0, 5.0, 5.0, 5.0]
    seen: list[float] = []

    def validate_fn(state):
        value = schedule[len(seen)]
        seen.append(value)
        return {"val_loss": jnp.asarray(value)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=8,
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=2, min_delta=0.0),
    )

    assert seen == schedule
    assert int(final.step) == 5


def test_early_stop_improve_stall_improve_stall_does_not_stop():
    """Improve, stall, improve, stall stays under patience=2 and runs to the end."""
    schedule = [10.0, 7.0, 7.0, 3.0, 3.0, 1.0]
    seen: list[float] = []
    callback = _EpochCounter()

    def validate_fn(state):
        value = schedule[len(seen)]
        seen.append(value)
        return {"val_loss": jnp.asarray(value)}

    final = _engine(callbacks=(callback,)).fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=len(schedule),
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=2, min_delta=0.0),
    )

    assert seen == schedule
    assert int(final.step) == len(schedule)
    assert callback.epochs == list(range(len(schedule)))
    assert callback.train_end == 1


def test_step_cadence_early_stop_breaks_inside_the_epoch():
    """A step-cadence stop returns at the stopping step, not the epoch length."""
    calls: list[int] = []

    def validate_fn(state):
        calls.append(int(state.step))
        return {"val_loss": jnp.array(1.0)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=6),
        num_epochs=2,
        validate_fn=validate_fn,
        validate_every_steps=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=1, min_delta=0.0),
    )

    # Step 1 sets the best. The constant metric at step 2 exhausts patience.
    assert calls == [1, 2]
    assert int(final.step) == 2


def test_checkpoint_every_steps_plus_epoch_end(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=5),
        num_epochs=1,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_steps=2,
    )

    assert int(final.step) == 5
    assert sorted(get_checkpoint_manager(checkpoint_dir).all_steps()) == [2, 4, 5]


def test_restore_brings_back_key_and_extras(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    saved_key = jax.random.PRNGKey(7)
    saved_model = _Model(jax.random.PRNGKey(1))
    saved_model = eqx.tree_at(lambda m: m.weight, saved_model, jnp.array([3.0, 4.0]))
    saved = _state(
        key=saved_key,
        model=saved_model,
        step=jnp.array(4, dtype=jnp.int32),
        extras={"flag": jnp.array(9.0), "note": "kept"},
    )
    manager = get_checkpoint_manager(checkpoint_dir)
    save_checkpoint(manager, saved, step=4)

    template_model = _Model(jax.random.PRNGKey(2))
    template_model = eqx.tree_at(lambda m: m.weight, template_model, jnp.array([0.0, 0.0]))
    template = _state(
        key=jax.random.PRNGKey(0),
        model=template_model,
        extras={"flag": jnp.array(0.0), "note": "template"},
    )

    loaded = _engine().restore(checkpoint_dir, template)

    assert jnp.array_equal(jax.random.key_data(loaded.key), jax.random.key_data(saved_key))
    assert not jnp.array_equal(jax.random.key_data(loaded.key), jax.random.key_data(template.key))
    assert jnp.allclose(loaded.extras["flag"], jnp.array(9.0))
    assert loaded.extras["note"] == "kept"
    assert jnp.allclose(loaded.model.weight, jnp.array([3.0, 4.0]))
    assert int(loaded.step) == 4


def test_early_stop_still_writes_a_complete_ledger(tmp_path, monkeypatch):
    ledger_root = tmp_path / "ledger"
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(ledger_root))
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    callback = _EpochCounter()

    def validate_fn(state):
        return {"val_loss": jnp.array(1.0)}

    final = _engine(callbacks=(callback,)).fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=5,
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=1, min_delta=0.0),
    )

    # Constant metric, patience=1: epoch 0 sets the best, epoch 1 stops.
    assert int(final.step) == 2
    assert callback.epochs == [0, 1]
    rows = list(iter_rows(ledger_root))
    assert len(rows) == 1
    assert rows[0].kind == KIND_TRAIN
    assert rows[0].telemetry_status == STATUS_COMPLETE


def test_early_stop_epoch_end_checkpoint_includes_stopping_step(tmp_path):
    """The epoch-end save still runs on the epoch where early stopping fires."""
    checkpoint_dir = tmp_path / "ckpt"

    def validate_fn(state):
        return {"val_loss": jnp.array(1.0)}

    final = _engine().fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=5,
        checkpoint_dir=checkpoint_dir,
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=1, min_delta=0.0),
    )

    assert int(final.step) == 2
    assert int(final.step) in get_checkpoint_manager(checkpoint_dir).all_steps()


def test_checkpoint_every_steps_requires_checkpoint_dir():
    with pytest.raises(ValueError, match="checkpoint_every_steps requires checkpoint_dir"):
        _engine().fit_sync(
            _state(),
            _Data(batch_count=1),
            num_epochs=1,
            checkpoint_every_steps=1,
        )


def test_early_stop_unknown_metric_raises_key_error():
    def validate_fn(state):
        return {"val_loss": jnp.array(1.0)}

    with pytest.raises(KeyError, match="not in metrics"):
        _engine().fit_sync(
            _state(),
            _Data(batch_count=1),
            num_epochs=1,
            validate_fn=validate_fn,
            validate_every_epochs=1,
            early_stop=EarlyStopping(metric="missing", mode="min", patience=1),
        )


def test_fit_advances_state_key_each_step():
    """takes_key training replaces state.key on every step, including across a fit."""
    seen: list[jax.Array] = []

    class _KeyLog:
        def on_train_start(self, state):
            return None

        def on_train_end(self, state):
            return None

        def on_resume(self, state):
            return None

        def on_epoch_start(self, state, epoch):
            return None

        def on_epoch_end(self, state, epoch):
            return None

        def on_step_start(self, state):
            seen.append(jax.random.key_data(state.key))

        def on_step_end(self, state, metrics):
            return None

    def keyed_loss(model, batch, key):
        predictions = model(batch["inputs"])
        return jnp.mean((predictions - batch["targets"]) ** 2)

    trainer = Trainer(loss_fn=keyed_loss, optimizer=optax.sgd(1e-3), takes_key=True)
    engine = Engine(trainer=trainer, callbacks=(_KeyLog(),), validation_callbacks=())
    initial = _state()
    final = engine.fit_sync(initial, _Data(batch_count=3), num_epochs=1)

    initial_data = jax.random.key_data(initial.key)
    observed = [*seen, jax.random.key_data(final.key)]
    assert len(seen) == 3
    assert jnp.array_equal(observed[0], initial_data)
    for key_data in observed[1:]:
        assert not jnp.array_equal(key_data, initial_data)
    for earlier, later in zip(observed, observed[1:]):
        assert not jnp.array_equal(earlier, later)


def test_early_stopping_on_training_metrics_when_validation_is_unset():
    """Without validate_fn, patience counts epoch-end training metrics."""

    def constant_loss(predictions, targets):
        return jnp.array(1.0) + 0.0 * jnp.mean((predictions - targets) ** 2)

    engine = Engine(
        trainer=Trainer(loss_fn=constant_loss, optimizer=optax.sgd(0.0)),
        callbacks=(),
        validation_callbacks=(),
    )
    final = engine.fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=6,
        early_stop=EarlyStopping(metric="loss", mode="min", patience=1, min_delta=0.0),
    )

    # First epoch sets the best. The second equal loss is not an improvement.
    assert int(final.step) == 2


def test_early_stopping_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(metric="loss", mode="none", patience=1)
