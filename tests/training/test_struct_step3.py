"""Training path on ``xtrax.Struct`` (equinox decoupling, step 3).

Spec ``.praxia/docs/specs/261023_equinox-decoupling.md`` section 6. The classes moved here are
``Trainer``, ``SafetyTrainStep``, ``ResumableState``, ``ComposedLoss`` (plus the deprecated
``WeightedLoss``, ``MultiTaskLoss``, ``SafetyManager``, ``ShardingPolicy``) and ``Engine``.
Equinox appears only as a fixture: the models a user hands to xtrax.
"""

import asyncio
import dataclasses
import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import xtrax.engine.engine as engine_module
from xtrax._tree import Struct, combine, is_inexact_array, partition, value_and_grad
from xtrax.checkpoint.orbax import get_checkpoint_manager, load_checkpoint, save_checkpoint
from xtrax.engine.engine import Engine
from xtrax.training.loss import ComposedLoss
from xtrax.training.state import init_state
from xtrax.training.step import create_train_step
from xtrax.training.trainer import Trainer
from xtrax.training.types import ResumableState

with pytest.deprecated_call(match="xtrax #6107"):
    from xtrax.distributed.sharding import ShardingPolicy
    from xtrax.safety.manager import SafetyManager
    from xtrax.training.loss import MultiTaskLoss, WeightedLoss
    from xtrax.training.step import SafetyTrainStep

LR = 0.1


def _tree_structure(obj):
    return jax.tree_util.tree_structure(obj)


def _roundtrip(obj):
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    return jax.tree_util.tree_unflatten(treedef, leaves)


# --- fixtures: a plain-dict model, and equinox models as user input ---------------------------


def _dict_model(seed=0, act="tanh"):
    w = jax.random.normal(jax.random.key(seed), (3, 2))
    model = {"w": w, "b": jnp.zeros(2)}
    return model if act is None else {**model, "act": act}


def _batch(i=0):
    kx, ky = jax.random.split(jax.random.key(100 + i))
    return {"x": jax.random.normal(kx, (4, 3)), "y": jax.random.normal(ky, (4, 2))}


def _dict_loss(model, batch, key):
    pred = batch["x"] @ model["w"] + model["b"]
    if model.get("act") == "tanh":
        pred = jnp.tanh(pred)
    return jnp.mean((pred - batch["y"]) ** 2)


def _dict_trainer(loss=_dict_loss, lr=LR):
    return Trainer(loss_fn=loss, optimizer=optax.sgd(lr), takes_key=True)


def _mlp(seed=0):
    return eqx.nn.MLP(3, 2, 8, 2, key=jax.random.key(seed))


def _mlp_loss(model, batch, key):
    return jnp.mean((jax.vmap(model)(batch["x"]) - batch["y"]) ** 2)


# --- leaves and treedef: static fields are not leaves -------------------------------------------


def test_resumable_state_leaves_and_treedef():
    model = _dict_model()
    state = init_state(model, optax.sgd(LR), seed=0)
    assert isinstance(state, Struct) and not isinstance(state, eqx.Module)
    fields = {f.name: f.metadata.get("static", False) for f in dataclasses.fields(ResumableState)}
    assert fields == {
        "step": False,
        "key": False,
        "model": False,
        "opt_state": False,
        "extras": False,
    }
    leaves = jax.tree_util.tree_leaves(state)
    # step, key, w, b; "tanh" is a non-array leaf of the model; sgd's state holds no arrays
    assert sum(isinstance(x, jax.Array) for x in leaves) == 4
    assert "tanh" in leaves
    again = _roundtrip(state)
    assert type(again) is ResumableState and again.model["act"] == "tanh"
    assert again.extras == {}


def test_trainer_static_fields_are_in_the_treedef_not_leaves():
    plain = Trainer(loss_fn=_dict_loss, optimizer=optax.sgd(LR))
    assert [f.name for f in dataclasses.fields(Trainer) if f.metadata.get("static")] == [
        "has_aux",
        "takes_key",
    ]
    leaves = jax.tree_util.tree_leaves(plain)
    assert False not in leaves and True not in leaves  # the flags are metadata
    assert _dict_loss in leaves
    assert _tree_structure(plain) != _tree_structure(dataclasses.replace(plain, has_aux=True))
    assert _tree_structure(plain) != _tree_structure(dataclasses.replace(plain, takes_key=True))
    assert _tree_structure(plain) == _tree_structure(dataclasses.replace(plain, loss_fn=print))
    assert type(_roundtrip(plain)) is Trainer


def test_safety_train_step_and_manager_leaves():
    with pytest.deprecated_call(match="safety=True is deprecated|xtrax #6107"):
        step = create_train_step(_dict_loss, optax.sgd(LR), safety=True, takes_key=True)
    assert isinstance(step, SafetyTrainStep) and isinstance(step, Struct)
    assert jax.tree_util.tree_leaves(step.safety_manager) == []
    assert _dict_loss in jax.tree_util.tree_leaves(step)
    off = dataclasses.replace(
        step, safety_manager=SafetyManager(enabled=False, check_nans=False, check_infs=False)
    )
    assert _tree_structure(step) != _tree_structure(off)
    assert type(_roundtrip(step)) is SafetyTrainStep


def _term(predictions, targets, **flags):
    return jnp.mean((predictions - targets) ** 2)


def _other_term(predictions, targets, **flags):
    return jnp.mean(jnp.abs(predictions - targets))


def test_composed_loss_weights_are_leaves_terms_are_static():
    loss = ComposedLoss([("a", _term), ("b", _other_term)], [1.0, 0.5])
    assert isinstance(loss, Struct)
    leaves = jax.tree_util.tree_leaves(loss)
    assert len(leaves) == 1 and leaves[0].shape == (2,)
    static = {f.name for f in dataclasses.fields(ComposedLoss) if f.metadata.get("static")}
    assert static == {"terms", "names"}
    other = ComposedLoss([("a", _term), ("b", _other_term)], [3.0, 4.0])
    assert _tree_structure(loss) == _tree_structure(other)  # weight values are not structure
    swapped = ComposedLoss([("a", _other_term), ("b", _term)], [1.0, 0.5])
    assert _tree_structure(loss) != _tree_structure(swapped)  # terms are
    renamed = ComposedLoss([("a", _term), ("c", _other_term)], [1.0, 0.5])
    assert _tree_structure(loss) != _tree_structure(renamed)
    again = _roundtrip(loss)
    assert type(again) is ComposedLoss and again.names == ("a", "b")
    assert again.terms == (_term, _other_term)


def test_composed_loss_init_runs_check_init_but_unflatten_does_not():
    with pytest.raises(ValueError, match="at least one term"):
        ComposedLoss([], [])
    with pytest.raises(ValueError, match="must be unique"):
        ComposedLoss([("a", _term), ("a", _other_term)], [1.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        ComposedLoss([("a", _term)], [1.0, 2.0])
    with pytest.raises(ValueError, match="one entry per term"):
        ComposedLoss([("a", _term)], [1.0], placement_weights=[None, None])
    loss = ComposedLoss([("a", _term)], [1.0], placement_weights=[jnp.ones(2)])
    # a tree_map that turns every leaf into None must not trip the weights-shape check
    nulled = jax.tree_util.tree_map(lambda x: None, loss)
    assert type(nulled) is ComposedLoss and nulled.weights is None


def test_composed_loss_call_unchanged_through_the_struct_init():
    loss = ComposedLoss([("a", _term), ("b", _other_term)], jnp.asarray([1.0, 0.5]))
    pred, tgt = jnp.asarray([1.0, 3.0]), jnp.asarray([0.0, 1.0])
    total, aux = loss(pred, tgt)
    assert float(aux["a"]) == pytest.approx(2.5) and float(aux["b"]) == pytest.approx(1.5)
    assert float(total) == pytest.approx(2.5 + 0.75)


def test_deprecated_losses_manager_and_policy_are_structs():
    weighted = WeightedLoss(loss_fn=_term, weight=2.0)
    multi = MultiTaskLoss(losses=(weighted,), weight_schedule=None)
    assert jax.tree_util.tree_leaves(weighted) == [_term]
    assert _tree_structure(weighted) != _tree_structure(WeightedLoss(loss_fn=_term, weight=3.0))
    assert type(_roundtrip(multi)) is MultiTaskLoss
    policy = ShardingPolicy(rules=(("w", jax.sharding.PartitionSpec("data")),))
    assert jax.tree_util.tree_leaves(policy) == []
    for obj in (weighted, multi, policy, SafetyManager(True, True, True)):
        assert isinstance(obj, Struct) and not isinstance(obj, eqx.Module)


class _NoopCallback:
    def on_train_start(self, state): ...
    def on_train_end(self, state): ...
    def on_resume(self, state): ...
    def on_epoch_start(self, state, epoch): ...
    def on_epoch_end(self, state, epoch): ...
    def on_step_start(self, state): ...
    def on_step_end(self, state, metrics): ...


def test_engine_is_static_only():
    trainer = _dict_trainer()
    engine = Engine(trainer=trainer, callbacks=())
    assert isinstance(engine, Struct) and not isinstance(engine, eqx.Module)
    assert jax.tree_util.tree_leaves(engine) == []
    assert all(f.metadata.get("static") for f in dataclasses.fields(Engine))
    assert engine.validation_callbacks == ()
    assert type(_roundtrip(engine)) is Engine and _roundtrip(engine).trainer is trainer
    with_cb = dataclasses.replace(engine, callbacks=(_NoopCallback(),))
    assert _tree_structure(engine) != _tree_structure(with_cb)


# --- retrace counts -----------------------------------------------------------------------------


def _counting_loss(counter):
    def loss(model, batch, key):
        counter.append(1)  # runs at trace time only
        return _dict_loss(model, batch, key)

    return loss


def _run_steps(trainer, state, n=5):
    for i in range(n):
        state, metrics = trainer.step(state, _batch(i))
    return state, metrics


def test_trainer_step_traces_once_over_many_steps():
    counter = []
    trainer = _dict_trainer(_counting_loss(counter))
    state = init_state(_dict_model(), optax.sgd(LR), seed=0)
    state, _ = _run_steps(trainer, state, n=6)
    assert int(state.step) == 6
    assert len(counter) == 1


def test_trainer_step_array_change_does_not_retrace_static_change_does():
    counter = []
    trainer = _dict_trainer(_counting_loss(counter))
    state = init_state(_dict_model(), optax.sgd(LR), seed=0)
    trainer.step(state, _batch(0))
    assert len(counter) == 1
    # new array values in model and batch: same trace
    other = dataclasses.replace(state, model={**state.model, "w": state.model["w"] + 1.0})
    trainer.step(other, _batch(1))
    assert len(counter) == 1
    # a non-array model leaf is a compile key
    flipped = dataclasses.replace(state, model={**state.model, "act": "linear"})
    trainer.step(flipped, _batch(0))
    assert len(counter) == 2
    # and so is a new batch shape
    wide = {"x": jnp.ones((6, 3)), "y": jnp.ones((6, 2))}
    trainer.step(state, wide)
    assert len(counter) == 3
    # a static field of the trainer is a compile key too
    aux_trainer = dataclasses.replace(trainer, has_aux=True, loss_fn=_aux_loss(counter))
    aux_trainer.step(state, _batch(0))
    assert len(counter) == 4


def _aux_loss(counter):
    def loss(model, batch, key):
        counter.append(1)
        return _dict_loss(model, batch, key), {"extra": jnp.asarray(1.0)}

    return loss


def test_retrace_counter_control_a_fresh_loss_each_call_retraces():
    # prove the counter can go red: a new function object is a new static leaf every call
    counter = []
    state = init_state(_dict_model(), optax.sgd(LR), seed=0)
    for i in range(3):
        _dict_trainer(_counting_loss(counter)).step(state, _batch(i))
    assert len(counter) == 3


class _ComposedAdapter(Struct):
    """Hold a ComposedLoss as a pytree field and call it with the (model, batch, key) form."""

    composed: Any

    def __call__(self, model, batch, key):
        pred = batch["x"] @ model["w"] + model["b"]
        return self.composed(pred, batch["y"])


def test_composed_loss_weight_change_does_not_retrace_term_change_does():
    counter = []

    def counted_a(predictions, targets, **flags):
        counter.append("a")
        return _term(predictions, targets)

    def counted_b(predictions, targets, **flags):
        counter.append("b")
        return _other_term(predictions, targets)

    optimizer = optax.sgd(LR)  # one instance: a new optax transformation is a new static leaf

    def make(name, term, weight):
        adapter = _ComposedAdapter(ComposedLoss([(name, term)], [weight]))
        return Trainer(loss_fn=adapter, optimizer=optimizer, has_aux=True, takes_key=True)

    state = init_state(_dict_model(), optax.sgd(LR), seed=0)
    make("t", counted_a, 1.0).step(state, _batch(0))
    assert counter == ["a"]
    make("t", counted_a, 5.0).step(state, _batch(0))
    assert counter == ["a"]  # a weight value is a traced leaf, not a compile key
    make("t", counted_b, 1.0).step(state, _batch(0))
    assert counter == ["a", "b"]  # another term function is a new static value
    make("u", counted_a, 1.0).step(state, _batch(0))
    assert counter == ["a", "b", "a"]  # another term name changes the treedef


def test_safety_train_step_traces_once():
    counter = []
    with pytest.deprecated_call(match="safety=True is deprecated|xtrax #6107"):
        step = create_train_step(
            _counting_loss(counter), optax.sgd(LR), safety=True, takes_key=True
        )
    state = init_state(_dict_model(act=None), optax.sgd(LR), seed=0)
    for i in range(4):
        state, _ = step.step(state, _batch(i))
    assert int(state.step) == 4
    assert len(counter) == 1


# --- gradients ------------------------------------------------------------------------------------


def test_trainer_matches_a_hand_unrolled_jax_grad_loop_on_a_dict_model():
    trainer = _dict_trainer()
    state = init_state(_dict_model(), optax.sgd(LR), seed=0)
    n = 5
    manual = {k: v for k, v in _dict_model().items() if k != "act"}
    for i in range(n):
        batch = _batch(i)

        def loss(m, batch=batch):
            return _dict_loss({**m, "act": "tanh"}, batch, None)

        loss_before, grads = jax.value_and_grad(loss)(manual)
        manual = {k: manual[k] - LR * grads[k] for k in manual}
        state, metrics = trainer.step(state, batch)
        np.testing.assert_allclose(metrics["loss"], loss_before, rtol=1e-6)
    for k in manual:
        np.testing.assert_allclose(state.model[k], manual[k], rtol=1e-6, atol=1e-6)
    assert state.model["act"] == "tanh"


def test_trainer_hand_unrolled_control_detects_a_wrong_update():
    # prove red: a hand loop with a different learning rate must not match
    trainer = _dict_trainer()
    state = init_state(_dict_model(), optax.sgd(LR), seed=0)
    state, _ = trainer.step(state, _batch(0))
    manual = {k: v for k, v in _dict_model().items() if k != "act"}
    grads = jax.grad(lambda m: _dict_loss({**m, "act": "tanh"}, _batch(0), None))(manual)
    wrong = {k: manual[k] - 2 * LR * grads[k] for k in manual}
    assert not np.allclose(state.model["w"], wrong["w"], rtol=1e-6, atol=1e-6)


def test_trainer_loss_decreases_on_a_dict_model():
    trainer = _dict_trainer(lr=0.3)
    state = init_state(_dict_model(), optax.sgd(0.3), seed=0)
    batch = _batch(0)
    first = float(trainer.step(state, batch)[1]["loss"])
    for _ in range(40):
        state, metrics = trainer.step(state, batch)
    assert float(metrics["loss"]) < 0.7 * first


def test_trainer_step_matches_eqx_filter_grad_on_an_mlp_once():
    """Compat check against equinox (not a parity suite): one sgd(1.0) step == model - grad."""
    model = _mlp()
    batch = _batch(0)
    trainer = Trainer(loss_fn=_mlp_loss, optimizer=optax.sgd(1.0), takes_key=True)
    state = init_state(model, optax.sgd(1.0), seed=0)
    new_state, _ = trainer.step(state, batch)
    want = eqx.filter_grad(_mlp_loss)(model, batch, None)
    got = jax.tree_util.tree_map(
        lambda old, new: old - new,
        eqx.filter(model, eqx.is_array),
        eqx.filter(new_state.model, eqx.is_array),
    )
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def test_value_and_grad_holds_integer_arrays_and_python_values_constant():
    model = {"w": jnp.ones(2), "idx": jnp.asarray([0, 1]), "scale": 3.0, "name": "m"}

    def loss(m):
        return jnp.sum(m["w"][m["idx"]]) * m["scale"]

    value, grads = value_and_grad(loss)(model)
    assert float(value) == 6.0
    assert grads["idx"] is None and grads["scale"] is None and grads["name"] is None
    np.testing.assert_array_equal(grads["w"], [3.0, 3.0])


def test_is_inexact_array_and_partition_with_a_predicate():
    assert is_inexact_array(jnp.ones(2)) and is_inexact_array(np.ones(2, np.complex64))
    assert not any(is_inexact_array(x) for x in (jnp.asarray([1, 2]), 1.0, "s", None, jnp.tanh))
    tree = {"f": jnp.ones(2), "i": jnp.asarray([1]), "s": "x"}
    floats, rest = partition(tree, is_inexact_array)
    assert floats["i"] is None and floats["s"] is None and floats["f"] is tree["f"]
    assert rest["f"] is None and rest["i"] is tree["i"] and rest["s"] == "x"
    assert combine(floats, rest)["i"] is tree["i"]
    # default predicate keeps integer arrays on the array side
    assert partition(tree)[0]["i"] is tree["i"]


# --- round trip: an eqx model comes back an eqx model --------------------------------------------


def test_trained_eqx_mlp_comes_back_as_an_equinox_module():
    model = _mlp()
    trainer = Trainer(loss_fn=_mlp_loss, optimizer=optax.adam(1e-2), takes_key=True)
    state = init_state(model, optax.adam(1e-2), seed=0)
    batch = _batch(0)
    first = float(_mlp_loss(model, batch, None))
    for _ in range(30):
        state, metrics = trainer.step(state, batch)
    assert isinstance(state.model, eqx.nn.MLP) and isinstance(state.model, eqx.Module)
    assert state.model.activation is model.activation  # function leaf survived as static
    assert state.model.layers[0].in_features == model.layers[0].in_features
    assert float(metrics["loss"]) < first
    # the trained model is a live model, not a placeholder tree
    assert state.model(jnp.ones(3)).shape == (2,)
    assert not np.array_equal(state.model.layers[0].weight, model.layers[0].weight)


def test_eqx_tree_at_works_on_resumable_state():
    state = init_state(_mlp(), optax.sgd(LR), seed=0)
    other = _mlp(seed=9)
    swapped = eqx.tree_at(lambda s: s.model, state, other)
    assert type(swapped) is ResumableState and eqx.tree_equal(swapped.model, other)
    bumped = eqx.tree_at(lambda s: (s.step, s.key), state, (jnp.asarray(5, jnp.int32), state.key))
    assert type(bumped) is ResumableState and int(bumped.step) == 5
    leaf_edit = eqx.tree_at(lambda s: s.model.layers[0].weight, state, jnp.zeros((8, 3)))
    assert isinstance(leaf_edit.model, eqx.nn.MLP)
    assert float(jnp.abs(leaf_edit.model.layers[0].weight).sum()) == 0.0
    # control: the original state is untouched (frozen)
    assert float(jnp.abs(state.model.layers[0].weight).sum()) > 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.step = 1


# --- Engine.eval keeps switching equinox models to inference mode -----------------------------


class _DropoutModel(eqx.Module):
    linear: eqx.nn.Linear
    drop: eqx.nn.Dropout

    def __call__(self, x):
        return self.drop(self.linear(x))  # no key: only valid in inference mode


class _Data:
    def train_iter(self):
        return iter([])

    def eval_iter(self):
        yield {"inputs": jnp.ones(3), "targets": jnp.ones(2)}


def _mse(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)


def _eval_engine():
    trainer = Trainer(loss_fn=_mse, optimizer=optax.sgd(LR))
    model = _DropoutModel(eqx.nn.Linear(3, 2, key=jax.random.key(0)), eqx.nn.Dropout(0.5))
    return Engine(trainer=trainer, callbacks=()), init_state(model, optax.sgd(LR), seed=0)


def test_engine_eval_puts_an_equinox_model_in_inference_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(tmp_path / "ledger"))
    engine, state = _eval_engine()
    metrics = asyncio.run(engine.eval(state, _Data()))
    assert metrics["loss"].shape == ()
    assert state.model.drop.inference is False  # the caller's state is not mutated


def test_engine_eval_control_without_inference_mode_the_dropout_model_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(tmp_path / "ledger"))
    monkeypatch.setattr(engine_module, "inference_mode", lambda model: model)
    engine, state = _eval_engine()
    with pytest.raises(Exception, match="key"):
        asyncio.run(engine.eval(state, _Data()))


# --- checkpoints: a state saved by the pre-migration class restores into the new one --------------


class _LegacyResumableState(eqx.Module):
    """The pre-migration ``ResumableState``: same fields, an ``eqx.Module``."""

    step: jax.Array
    key: jax.Array
    model: Any
    opt_state: Any
    extras: dict = eqx.field(default_factory=dict)


def _build(cls, seed):
    model = eqx.nn.Linear(3, 2, key=jax.random.key(seed))
    opt = optax.adam(1e-3)
    return cls(
        step=jnp.asarray(seed, jnp.int32),
        key=jax.random.PRNGKey(seed),
        model=model,
        opt_state=opt.init(eqx.filter(model, eqx.is_array)),
        extras={"e": jnp.arange(3.0) + seed},
    )


def _saved_leaf_paths(directory):
    (meta,) = (directory / "7" / "state").glob("_METADATA")
    return json.loads(meta.read_text())["tree_metadata"]


def test_checkpoint_saved_from_the_equinox_class_restores_into_resumable_state(tmp_path):
    old = _build(_LegacyResumableState, 7)
    save_checkpoint(get_checkpoint_manager(tmp_path / "old"), old)
    manager = get_checkpoint_manager(tmp_path / "old")
    restored = load_checkpoint(manager, _build(ResumableState, 0))
    assert type(restored) is ResumableState and isinstance(restored.model, eqx.nn.Linear)
    got = jax.tree_util.tree_leaves(restored)
    want = jax.tree_util.tree_leaves(old)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a, b)
    assert int(restored.step) == 7  # control: not the template's 0


def test_checkpoint_layout_is_identical_for_the_old_and_new_class(tmp_path):
    save_checkpoint(get_checkpoint_manager(tmp_path / "old"), _build(_LegacyResumableState, 7))
    save_checkpoint(get_checkpoint_manager(tmp_path / "new"), _build(ResumableState, 7))
    old_paths = _saved_leaf_paths(tmp_path / "old")
    new_paths = _saved_leaf_paths(tmp_path / "new")
    assert set(old_paths) == set(new_paths) and "('model', 'weight')" in new_paths
    # control: a state whose layout differs is detected
    other = dataclasses.replace(_build(ResumableState, 7), extras={"other": jnp.zeros(3)})
    save_checkpoint(get_checkpoint_manager(tmp_path / "other"), other)
    assert set(_saved_leaf_paths(tmp_path / "other")) != set(old_paths)


def test_engine_restore_reads_a_checkpoint_written_by_the_equinox_class(tmp_path):
    save_checkpoint(get_checkpoint_manager(tmp_path / "ckpt"), _build(_LegacyResumableState, 7))
    engine = Engine(trainer=_dict_trainer(), callbacks=())
    restored = engine.restore(tmp_path / "ckpt", _build(ResumableState, 0))
    assert int(restored.step) == 7
    np.testing.assert_array_equal(restored.extras["e"], jnp.arange(3.0) + 7)
