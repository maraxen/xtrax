"""Pytree behaviour of the classes moved onto ``xtrax.Struct`` in step 2a.

Equinox decoupling, step 2a (spec ``.praxia/docs/specs/261023_equinox-decoupling.md`` §5).
Every assertion here was run green on ``eqx.Module`` before the classes moved and again
on ``Struct`` after, so the file pins the behaviour the move must keep: leaves, which
fields live in the treedef, equality/hash, ``dataclasses.replace``, the ``DataModule``
epoch counter, and the downstream ``class X(StageBundle)`` shape.
"""

import ast
import dataclasses
import itertools
import pathlib
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import xtrax.data.module as data_module
import xtrax.stages.boundaries as boundaries_module
import xtrax.stages.bundle as bundle_module
import xtrax.tiling.iterator as iterator_module
from xtrax._tree import field as xfield
from xtrax._tree import model_jit
from xtrax.data.module import DataModule
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.bundle import StageBundle
from xtrax.tiling import _BatchPlanWrapper
from xtrax.tiling.iterator import (
    ChunkedMapIterator,
    JaxScanIterator,
    VmapIterator,
    WhileLoopIterator,
    WhileLoopWithYsIterator,
)


def _double(x):
    return x * 2


def _make_dm(**kwargs):
    base = dict(dataset=[1, 2, 3], batch_size=2, num_epochs=1, seed=7, distributed=False)
    base.update(kwargs)
    return DataModule(**base)


def _roundtrip(obj):
    leaves, treedef = jax.tree_util.tree_flatten(obj)
    return jax.tree_util.tree_unflatten(treedef, leaves)


# --- iterators ---------------------------------------------------------------------------


_ITERATOR_FACTORIES = [
    lambda: VmapIterator(),
    lambda: ChunkedMapIterator(tile=4),
    lambda: JaxScanIterator(),
    lambda: WhileLoopIterator(),
    lambda: WhileLoopWithYsIterator(max_steps=3),
]


@pytest.mark.parametrize("make", _ITERATOR_FACTORIES)
def test_iterators_have_no_leaves_and_roundtrip(make):
    it = make()
    assert jax.tree_util.tree_leaves(it) == []
    again = _roundtrip(it)
    assert type(again) is type(it)
    assert again == it
    assert hash(again) == hash(it)
    assert jax.tree_util.tree_map(lambda x: x, it) == it


def test_iterator_static_fields_are_in_the_treedef():
    tree = jax.tree_util.tree_structure
    assert tree(ChunkedMapIterator(tile=4)) == tree(ChunkedMapIterator(tile=4))
    assert tree(ChunkedMapIterator(tile=4)) != tree(ChunkedMapIterator(tile=5))
    assert tree(WhileLoopWithYsIterator(max_steps=3)) == tree(WhileLoopWithYsIterator(max_steps=3))
    assert tree(WhileLoopWithYsIterator(max_steps=3)) != tree(WhileLoopWithYsIterator(max_steps=4))
    assert tree(VmapIterator()) != tree(JaxScanIterator())
    assert ChunkedMapIterator(tile=4) == ChunkedMapIterator(tile=4)
    assert ChunkedMapIterator(tile=4) != ChunkedMapIterator(tile=5)
    assert VmapIterator() != JaxScanIterator()
    for cls, name in [(ChunkedMapIterator, "tile"), (WhileLoopWithYsIterator, "max_steps")]:
        (f,) = dataclasses.fields(cls)
        assert f.name == name
        assert f.metadata["static"] is True


def test_iterator_as_pytree_field_keeps_treedef_semantics():
    """aminx stores iterators as pytree fields of its own modules (autoregressive.py:350)."""
    holder = {"it": WhileLoopIterator(), "w": jnp.ones(2)}
    leaves, treedef = jax.tree_util.tree_flatten(holder)
    assert len(leaves) == 1
    out = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(out["it"]) is WhileLoopIterator
    again = jax.tree_util.tree_structure({"it": WhileLoopIterator(), "w": jnp.zeros(2)})
    assert again == treedef


def test_iterator_passes_through_jax_jit_and_retraces_only_on_static_change():
    traces = []

    @jax.jit
    def run(it, xs):
        traces.append(type(it).__name__)
        return it(_double, xs)

    xs = jnp.arange(6.0)
    out = run(ChunkedMapIterator(tile=2), xs)
    assert out.tolist() == (xs * 2).tolist()
    run(ChunkedMapIterator(tile=2), xs + 1)
    assert len(traces) == 1
    run(ChunkedMapIterator(tile=3), xs)
    assert len(traces) == 2


def test_scan_and_while_iterators_work_under_jit_as_arguments():
    scan_out = jax.jit(lambda it, xs: it(lambda c, x: (c + x, c), 0.0, xs))(
        JaxScanIterator(), jnp.arange(4.0)
    )
    assert scan_out[0] == 6.0
    final = jax.jit(lambda it, init: it(lambda c: c < 5, lambda c: c + 1, init))(
        WhileLoopIterator(), jnp.asarray(0)
    )
    assert final == 5
    carry, ys, n = jax.jit(
        lambda it, init: it(lambda c: c < 10, lambda c: (c + 1, c * 2), init, jnp.zeros(()))
    )(WhileLoopWithYsIterator(max_steps=4), jnp.asarray(0))
    assert (carry, n) == (4, 4)
    assert ys.shape == (4,)


# --- _BatchPlanWrapper ---------------------------------------------------------------------


class _Plan:
    batch = 8


def test_batch_plan_wrapper_hashes_by_plan_identity():
    plan = _Plan()
    a, b = _BatchPlanWrapper(plan), _BatchPlanWrapper(plan)
    assert a == b
    assert hash(a) == hash(b)
    assert a != _BatchPlanWrapper(_Plan())
    assert a != plan
    assert a.batch == 8
    assert a._plan is plan
    assert repr(a).startswith("_BatchPlanWrapper(")
    assert _BatchPlanWrapper(None)._id is None


def test_batch_plan_wrapper_is_a_static_jit_argument():
    traces = []
    plan = _Plan()

    @jax.jit
    def run(x, wrapped):
        traces.append(1)
        return x * wrapped.batch

    run_static = jax.jit(run.__wrapped__, static_argnums=1)
    assert run_static(jnp.ones(2), _BatchPlanWrapper(plan)).tolist() == [8.0, 8.0]
    run_static(jnp.ones(2), _BatchPlanWrapper(plan))
    assert len(traces) == 1
    run_static(jnp.ones(2), _BatchPlanWrapper(_Plan()))
    assert len(traces) == 2


# --- AxisBoundary --------------------------------------------------------------------------


def _fuse(stacked):
    return stacked


def test_axis_boundary_is_all_static():
    ab = AxisBoundary(fuse=_fuse, tap=_fuse, sink=_fuse, materialize=True)
    assert jax.tree_util.tree_leaves(ab) == []
    assert all(f.metadata["static"] is True for f in dataclasses.fields(AxisBoundary))
    assert [f.name for f in dataclasses.fields(AxisBoundary)] == [
        "fuse",
        "tap",
        "sink",
        "materialize",
        "sink_receives_index",
    ]
    again = _roundtrip(ab)
    assert type(again) is AxisBoundary
    assert again == ab
    assert hash(again) == hash(ab)
    assert again.fuse is _fuse


def test_axis_boundary_treedef_and_equality_follow_static_values():
    tree = jax.tree_util.tree_structure
    assert tree(AxisBoundary(fuse=_fuse)) == tree(AxisBoundary(fuse=_fuse))
    assert tree(AxisBoundary(fuse=_fuse)) != tree(AxisBoundary())
    assert tree(AxisBoundary(materialize=True)) != tree(AxisBoundary())
    assert tree(AxisBoundary(sink_receives_index=True)) != tree(AxisBoundary())
    assert AxisBoundary(fuse=_fuse) == AxisBoundary(fuse=_fuse)
    assert AxisBoundary(fuse=_fuse) != AxisBoundary(tap=_fuse)
    assert hash(AxisBoundary(tap=_fuse)) == hash(AxisBoundary(tap=_fuse))


def test_axis_boundary_defaults_and_replace():
    ab = AxisBoundary()
    assert (ab.fuse, ab.tap, ab.sink, ab.materialize, ab.sink_receives_index) == (
        None,
        None,
        None,
        False,
        False,
    )
    # export/pipeline.py:139 rebuilds boundaries this way.
    new = dataclasses.replace(AxisBoundary(sink=_fuse, materialize=True), sink=_double)
    assert new.sink is _double
    assert new.materialize is True
    assert type(new) is AxisBoundary


def test_axis_boundary_is_frozen():
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        AxisBoundary().materialize = True


def test_axis_boundary_rides_in_a_dict_through_jit():
    traces = []

    @jax.jit
    def run(bs, x):
        traces.append(1)
        return bs["a"].fuse(x)

    run({"a": AxisBoundary(fuse=_double)}, jnp.ones(2))
    run({"a": AxisBoundary(fuse=_double)}, jnp.zeros(2))
    assert len(traces) == 1
    run({"a": AxisBoundary(fuse=_fuse)}, jnp.zeros(2))
    assert len(traces) == 2


# --- StageBundle (downstream shape) ----------------------------------------------------------


class _Downstream(StageBundle):
    """Undecorated downstream subclass mixing static and traced callable fields."""

    encode: Callable | None = None
    decode: Callable | None = eqx.field(static=True, default=None)
    sink: Callable | None = None


class _DownstreamXtraxField(StageBundle):
    encode: Callable | None = None
    decode: Callable | None = xfield(static=True, default=None)


def test_downstream_stage_bundle_subclass_static_vs_traced_fields():
    b = _Downstream(encode=_double, decode=_fuse, sink=None)
    assert jax.tree_util.tree_leaves(b) == [_double]
    fields = {f.name: f.metadata.get("static", False) for f in dataclasses.fields(_Downstream)}
    assert fields == {"encode": False, "decode": True, "sink": False}
    again = _roundtrip(b)
    assert type(again) is _Downstream
    assert (again.encode, again.decode, again.sink) == (_double, _fuse, None)
    tree = jax.tree_util.tree_structure
    assert tree(b) == tree(_Downstream(encode=_fuse, decode=_fuse, sink=None))
    assert tree(b) != tree(_Downstream(encode=_double, decode=_double, sink=None))
    assert tree(b) != tree(_Downstream(encode=_double, decode=_fuse, sink=_fuse))


def test_downstream_stage_bundle_active_stages_survive_a_tree_map():
    b = _Downstream(encode=_double, decode=_fuse, sink=_double)
    assert b.active_stages() == ["encode", "decode", "sink"]
    mapped = jax.tree_util.tree_map(lambda f: f, b)
    # Declaration order, not unflatten order (which sets static fields first).
    assert mapped.active_stages() == ["encode", "decode", "sink"]
    assert mapped.has_stage("decode") and mapped.has_stage("encode")
    assert not _Downstream().has_stage("encode")


def test_downstream_stage_bundle_with_xtrax_field_and_replace():
    b = _DownstreamXtraxField(encode=_double, decode=_fuse)
    assert jax.tree_util.tree_leaves(b) == [_double]
    new = dataclasses.replace(b, encode=_fuse)
    assert type(new) is _DownstreamXtraxField
    assert (new.encode, new.decode) == (_fuse, _fuse)


def test_downstream_stage_bundle_check_init_runs_and_is_inherited():
    class Checked(StageBundle):
        encode: Callable | None = None

        def __check_init__(self):
            if self.encode is None:
                raise ValueError("encode required")

    class Child(Checked):
        decode: Callable | None = None

    with pytest.raises(ValueError, match="encode required"):
        Checked()
    with pytest.raises(ValueError, match="encode required"):
        Child(decode=_double)
    assert Child(encode=_double).has_stage("encode")
    # Unflatten does not rerun the check (a placeholder leaf must not raise).
    jax.tree_util.tree_map(lambda f: None, Child(encode=_double))


def test_stage_bundle_still_rejects_non_callable_annotations():
    with pytest.raises(TypeError, match="must be Optional"):

        class Bad(StageBundle):
            x: int | None = None


@model_jit
def _run_bundle(bundle, x):
    _BUNDLE_TRACES.append(1)
    return bundle.encode(x)


_BUNDLE_TRACES = []


def test_stage_bundle_under_model_jit_as_argument():
    """Function leaves are not arrays, so a bundle goes through ``model_jit``, not ``jax.jit``."""
    _BUNDLE_TRACES.clear()
    first = _run_bundle(_Downstream(encode=_double), jnp.ones(2))
    _run_bundle(_Downstream(encode=_double), jnp.zeros(2))
    assert first.tolist() == [2.0, 2.0]
    assert len(_BUNDLE_TRACES) == 1
    _run_bundle(_Downstream(encode=_fuse), jnp.zeros(2))
    assert len(_BUNDLE_TRACES) == 2


# --- DataModule --------------------------------------------------------------------------------


def test_data_module_leaves_and_static_fields():
    dm = _make_dm(dataset=(jnp.ones(2), jnp.zeros(3)))
    assert len(jax.tree_util.tree_leaves(dm)) == 2
    static = {f.name for f in dataclasses.fields(DataModule) if f.metadata.get("static")}
    assert static == {
        "batch_size",
        "num_epochs",
        "seed",
        "distributed",
        "collate_fn",
        "use_grain_pipeline",
        "pad_fn",
        "_epochs",
    }
    epochs_field = next(f for f in dataclasses.fields(DataModule) if f.name == "_epochs")
    assert epochs_field.init is False


def test_data_module_epoch_counter_survives_flatten_unflatten_exactly():
    dm = _make_dm()
    assert list(dm.train_iter()) == [1, 2, 3]
    assert list(dm.train_iter()) == [1, 2, 3]
    leaves, treedef = jax.tree_util.tree_flatten(dm)
    same = jax.tree_util.tree_unflatten(treedef, leaves)
    mapped = jax.tree_util.tree_map(lambda x: x, dm)
    # The counter object is carried by identity, so every copy shares one sequence.
    assert same._epochs is dm._epochs
    assert mapped._epochs is dm._epochs
    assert next(dm._epochs) == 2
    assert next(same._epochs) == 3
    assert next(mapped._epochs) == 4


def test_data_module_epoch_counter_not_reset_by_jit_paths():
    dm = _make_dm(dataset=(jnp.ones(2),))
    list(dm.train_iter())
    traces = []

    @jax.jit
    def run(module, x):
        traces.append(1)
        return x + module.batch_size

    run(dm, jnp.ones(2))
    list(dm.train_iter())
    run(dm, jnp.ones(2))
    assert len(traces) == 1  # same instance, advancing counter: treedef unchanged
    assert next(dm._epochs) == 2  # two train_iter calls so far; no reset by jit
    other = _make_dm(dataset=(jnp.ones(2),))
    run(other, jnp.ones(2))
    assert len(traces) == 2  # a distinct counter object is a distinct static value


def test_data_module_fresh_instances_start_at_zero_and_do_not_share():
    a, b = _make_dm(), _make_dm()
    assert a._epochs is not b._epochs
    list(a.train_iter())
    list(a.train_iter())
    assert next(a._epochs) == 2
    assert next(b._epochs) == 0
    assert isinstance(_make_dm()._epochs, itertools.count)


def test_data_module_replace_gets_a_fresh_counter():
    dm = _make_dm()
    list(dm.train_iter())
    new = dataclasses.replace(dm, seed=9)
    assert new._epochs is not dm._epochs
    assert next(new._epochs) == 0
    assert new.seed == 9


def test_data_module_seed_offsets_follow_the_counter(monkeypatch):
    seeds = []

    def fake(*args, **kwargs):
        seeds.append(kwargs["seed"])
        return iter([])

    monkeypatch.setattr("xtrax.data.pipeline.build_input_pipeline", fake)
    dm = _make_dm(use_grain_pipeline=True)
    for _ in range(3):
        list(dm.train_iter())
    jax.tree_util.tree_map(lambda x: x, dm)
    list(_roundtrip(dm).train_iter())
    assert seeds == [7, 8, 9, 10]


def test_data_module_is_frozen():
    with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
        _make_dm().seed = 1


# --- migrated files no longer import equinox ----------------------------------------------------


@pytest.mark.parametrize("module", [boundaries_module, bundle_module, data_module, iterator_module])
def test_migrated_module_does_not_import_equinox(module):
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        elif isinstance(node, ast.Name) and node.id == "eqx":
            names.append("eqx")
    assert not [n for n in names if n.split(".")[0] in {"equinox", "eqx"}]


@pytest.mark.parametrize(
    "cls",
    [
        VmapIterator,
        ChunkedMapIterator,
        JaxScanIterator,
        WhileLoopIterator,
        WhileLoopWithYsIterator,
        AxisBoundary,
        StageBundle,
        DataModule,
    ],
)
def test_migrated_class_is_not_an_equinox_module(cls):
    assert not issubclass(cls, eqx.Module)
