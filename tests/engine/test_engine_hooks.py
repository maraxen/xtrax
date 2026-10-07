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

    def validate_fn(state):
        return {"val_loss": jnp.array(1.0)}

    _engine().fit_sync(
        _state(),
        _Data(batch_count=1),
        num_epochs=5,
        validate_fn=validate_fn,
        validate_every_epochs=1,
        early_stop=EarlyStopping(metric="val_loss", mode="min", patience=1, min_delta=0.0),
    )

    rows = list(iter_rows(ledger_root))
    assert len(rows) == 1
    assert rows[0].kind == KIND_TRAIN
    assert rows[0].telemetry_status == STATUS_COMPLETE


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
