"""Tests for xtrax.intent.map."""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import pytest

from xtrax.intent import (
    IntentError,
    Session,
    SiteDecision,
    SiteKey,
    activate,
    current_session,
    loop,
)
from xtrax.intent import map as xmap
from xtrax.intent._trace import decision_for
from xtrax.tiling.strategy import ChunkedMap, Scan, Vmap

_W = jnp.float32(0.7)
_XS = jnp.linspace(-1.0, 1.0, 15, dtype=jnp.float32).reshape(5, 3)
_T = jnp.ones_like(_XS)


def ok(a, b) -> bool:
    leaves_a, def_a = jax.tree_util.tree_flatten(a)
    leaves_b, def_b = jax.tree_util.tree_flatten(b)
    if def_a != def_b or len(leaves_a) != len(leaves_b):
        return False
    return all(
        bool(jnp.allclose(left, right, atol=1e-5))
        for left, right in zip(leaves_a, leaves_b, strict=True)
    )


def same(a, b) -> None:
    leaves_a, def_a = jax.tree_util.tree_flatten(a)
    leaves_b, def_b = jax.tree_util.tree_flatten(b)
    assert def_a == def_b
    assert len(leaves_a) == len(leaves_b)
    for left, right in zip(leaves_a, leaves_b, strict=True):
        assert left.shape == right.shape
        assert left.dtype == right.dtype
    assert ok(a, b)


def layer(x):
    return jnp.tanh(x * _W + 0.1)


def _decision(key: SiteKey, primal, diff, kind: str) -> SiteDecision:
    return SiteDecision(
        key=key,
        kind=kind,
        primal=primal,
        diff=diff,
        primal_reason="test",
        diff_reason="test",
    )


def _exec(name: str, primal, diff, xs, fun):
    key = SiteKey(parent=None, path="primal", label=name, ordinal=0)
    decision = _decision(key, primal, diff, "map")
    decisions = {key: decision}
    plan = Session(mode="plan", decisions=decisions)
    with activate(plan):
        fun(xs)
    return Session(mode="exec", records=plan.records, decisions=decisions)


def test_map_unresolved_equals_vmap():
    assert current_session() is None
    same(xmap(layer, _XS), jax.vmap(layer)(_XS))

    session = Session(mode="plan")

    def run(xs):
        return xmap(layer, xs, name="plain")

    with activate(session):
        text = str(jax.make_jaxpr(run)(_XS))
        got = run(_XS)
    assert "scan[" not in text
    same(got, jax.vmap(layer)(_XS))
    assert session.records
    assert all(record.kind == "map" and record.length is None for record in session.records)

    def f(xs):
        return xmap(layer, xs, name="plain").sum()

    def f_ref(xs):
        return jax.vmap(layer)(xs).sum()

    with activate(Session(mode="plan")):
        same(jax.grad(f)(_XS), jax.grad(f_ref)(_XS))


def test_map_chunked_decision_uses_lax_map():
    xs = jnp.linspace(-1.0, 1.0, 400, dtype=jnp.float32).reshape(100, 4)
    key = SiteKey(parent=None, path="primal", label="layer", ordinal=0)
    decisions = {key: _decision(key, ChunkedMap(32), ChunkedMap(32), "map")}

    def run(x):
        return xmap(layer, x, name="layer")

    plan = Session(mode="plan", decisions=decisions)
    with activate(plan):
        planned = run(xs)
    exe = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(exe):
        got = run(xs)
    traced = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(traced):
        text = str(jax.make_jaxpr(run)(xs))
    ref = jax.vmap(layer)(xs)
    assert "scan[" in text
    same(planned, ref)
    same(got, ref)

    def f(x):
        return xmap(layer, x, name="layer").sum()

    def f_ref(x):
        return jax.vmap(layer)(x).sum()

    tangent = jnp.ones_like(xs)

    def _fresh() -> Session:
        return Session(mode="exec", records=plan.records, decisions=decisions)

    with activate(_fresh()):
        same(jax.grad(f)(xs), jax.grad(f_ref)(xs))
    with activate(_fresh()):
        same(jax.jvp(f, (xs,), (tangent,))[1], jax.jvp(f_ref, (xs,), (tangent,))[1])
    with activate(_fresh()):
        assert "scan[" in str(jax.make_jaxpr(jax.grad(f))(xs))


def test_map_grad_and_jvp_match_vmap():
    def f(xs):
        return xmap(layer, xs).sum()

    def f_ref(xs):
        return jax.vmap(layer)(xs).sum()

    same(jax.grad(f)(_XS), jax.grad(f_ref)(_XS))
    same(jax.jvp(f, (_XS,), (_T,))[1], jax.jvp(f_ref, (_XS,), (_T,))[1])
    same(
        jax.vmap(jax.grad(f))(jnp.stack([_XS, 0.5 * _XS])),
        jax.vmap(jax.grad(f_ref))(jnp.stack([_XS, 0.5 * _XS])),
    )


def test_map_leading_axis_one():
    def f_ref(xs):
        return jax.vmap(layer)(xs).sum()

    for n in (0, 1):
        xs = jnp.arange(n * 4, dtype=jnp.float32).reshape(n, 4) + jnp.float32(0.5)
        ref = jax.vmap(layer)(xs)
        same(xmap(layer, xs), ref)

        def f(xs, n=n):
            return xmap(layer, xs, name=f"lead{n}").sum()

        same(jax.grad(f)(xs), jax.grad(f_ref)(xs))

        for strategy in (Vmap(), ChunkedMap(8)):
            label = f"{type(strategy).__name__}{n}"

            def run(x, label=label):
                return xmap(layer, x, name=label)

            exe = _exec(label, strategy, strategy, xs, run)
            with activate(exe):
                same(run(xs), ref)
            graded = Session(mode="exec", records=exe.records, decisions=exe.decisions)

            def f_dec(x, label=label):
                return xmap(layer, x, name=label).sum()

            with activate(graded):
                same(jax.grad(f_dec)(xs), jax.grad(f_ref)(xs))


def test_map_closure_over_params_grad():
    xs = _XS

    def f(w):
        def body(x):
            return jnp.tanh(x * w + 0.1)

        return xmap(body, xs).sum()

    def f_ref(w):
        def body(x):
            return jnp.tanh(x * w + 0.1)

        return jax.vmap(body)(xs).sum()

    w = jnp.float32(0.7)
    same(f(w), f_ref(w))
    same(jax.grad(f)(w), jax.grad(f_ref)(w))


def test_map_mismatched_leading_sizes_before_call():
    traced = {"n": 0}

    def body(tree):
        traced["n"] += 1
        left, right = tree
        return left + jnp.float32(1.0), right

    xs = (jnp.ones((3, 2), dtype=jnp.float32), jnp.ones((4, 2), dtype=jnp.float32))
    with pytest.raises(ValueError, match="inconsistent sizes") as exc:
        xmap(body, xs)
    assert traced["n"] == 0
    message = str(exc.value)
    assert "3" in message and "4" in message

    scalar = jnp.float32(1.0)
    with pytest.raises(ValueError, match="rank should be at least 1"):
        xmap(body, scalar)
    assert traced["n"] == 0

    matched = (jnp.ones((3, 2), dtype=jnp.float32), jnp.ones((3,), dtype=jnp.float32))

    def both(tree):
        left, right = tree
        return left + jnp.float32(1.0), right * jnp.float32(2.0)

    same(xmap(both, matched), jax.vmap(both)(matched))


def test_map_scan_decision_raises():
    xs = _XS
    primal_key = SiteKey(parent=None, path="primal", label="decoder", ordinal=0)
    primal = Session(
        mode="plan",
        decisions={primal_key: _decision(primal_key, Scan(), Vmap(), "map")},
    )
    with activate(primal), pytest.raises(IntentError) as exc:
        xmap(layer, xs, name="decoder")
    err = exc.value
    assert err.path == "primal"
    assert "decoder" in err.site
    assert "Scan" in err.constraint
    assert "Scan" in str(err)

    diff_key = SiteKey(parent=None, path="primal", label="decoder", ordinal=0)
    diff = Session(
        mode="plan",
        decisions={diff_key: _decision(diff_key, Vmap(), Scan(), "map")},
    )
    with activate(diff), pytest.raises(IntentError) as exc_diff:
        xmap(layer, xs, name="decoder")
    err_diff = exc_diff.value
    assert err_diff.path == "diff"
    assert "decoder" in err_diff.site
    assert "Scan" in err_diff.constraint


def test_map_grad_after_activate_records_diff_child():
    session = Session(mode="plan")
    xs = jnp.linspace(-1.0, 1.0, 24, dtype=jnp.float32).reshape(4, 3, 2)

    def f(xs):
        def layer_row(row):
            y = xmap(lambda z: jnp.tanh(z + jnp.float32(0.1)), row, name="inner")
            return jnp.sum(y)

        return xmap(layer_row, xs, name="outer").sum()

    def f_ref(xs):
        def layer_row(row):
            y = jax.vmap(lambda z: jnp.tanh(z + jnp.float32(0.1)))(row)
            return jnp.sum(y)

        return jax.vmap(layer_row)(xs).sum()

    compiled = jax.jit(f)
    with activate(session):
        compiled(xs)
    assert current_session() is None
    outer = next(record.key for record in session.records if record.key.label == "outer")
    before = len(session.records)
    same(jax.grad(compiled)(xs), jax.grad(f_ref)(xs))
    assert current_session() is None
    added = session.records[before:]
    assert len(added) == 1
    child = added[0]
    assert child.key.path == "diff"
    assert child.key.parent == outer
    assert child.key.ordinal == 0
    assert child.key.label == "inner"
    assert child.kind == "map"


def test_map_nested_loop_exec_honors_scan():
    xs = jnp.linspace(-1.0, 1.0, 30, dtype=jnp.float32).reshape(6, 5)
    outer = SiteKey(parent=None, path="primal", label="outer", ordinal=0)
    inner = SiteKey(parent=outer, path="primal", label="inner", ordinal=0)
    decisions = {
        outer: _decision(outer, Vmap(), Vmap(), "map"),
        inner: _decision(inner, Scan(), Scan(), "loop"),
    }

    def f(values):
        def layer_row(row):
            def body(carry, _):
                return carry + jnp.float32(0.1), jnp.sum(carry)

            carry, ys = loop(body, row, length=2, name="inner")
            return carry + jnp.sum(ys)

        return xmap(layer_row, values, name="outer")

    def f_ref(values):
        def layer_row(row):
            def body(carry, _):
                return carry + jnp.float32(0.1), jnp.sum(carry)

            carry, ys = jax.lax.scan(body, row, None, length=2)
            return carry + jnp.sum(ys)

        return jax.vmap(layer_row)(values)

    plan = Session(mode="plan", decisions=decisions)
    with activate(plan):
        planned = f(xs)
    same(planned, f_ref(xs))
    exe = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(exe):
        got = f(xs)
    same(got, f_ref(xs))
    stored = next(record for record in exe.records if record.key == outer)
    assert stored.kind == "map"
    text = str(jax.make_jaxpr(lambda row: stored.fn(row))(xs[0]))
    assert "scan[" in text


def test_map_nested_map_under_chunked():
    xs = jnp.linspace(-1.0, 1.0, 240, dtype=jnp.float32).reshape(4, 20, 3)
    outer = SiteKey(parent=None, path="primal", label="outer", ordinal=0)
    inner = SiteKey(parent=outer, path="primal", label="inner", ordinal=0)
    decisions = {
        outer: _decision(outer, Vmap(), Vmap(), "map"),
        inner: _decision(inner, ChunkedMap(8), ChunkedMap(8), "map"),
    }

    def f(values):
        def layer_row(row):
            return xmap(lambda z: jnp.tanh(z * jnp.float32(0.5)), row, name="inner")

        return xmap(layer_row, values, name="outer")

    def f_ref(values):
        def layer_row(row):
            return jax.vmap(lambda z: jnp.tanh(z * jnp.float32(0.5)))(row)

        return jax.vmap(layer_row)(values)

    plan = Session(mode="plan", decisions=decisions)
    with activate(plan):
        planned = f(xs)
    same(planned, f_ref(xs))
    exe = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(exe):
        got = f(xs)
    traced = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(traced):
        text = str(jax.make_jaxpr(f)(xs))
    same(got, f_ref(xs))
    assert "scan[" in text
    stored = next(record for record in exe.records if record.key == outer)
    stored_text = str(jax.make_jaxpr(lambda row: stored.fn(row))(xs[0]))
    assert "scan[" in stored_text

    def loss(values):
        return jnp.sum(f(values))

    def loss_ref(values):
        return jnp.sum(f_ref(values))

    grad_plan = Session(mode="plan", decisions=decisions)
    with activate(grad_plan):

        def grad_loss(values):
            return jax.grad(loss)(values)

        grad_text = str(jax.make_jaxpr(grad_loss)(xs))
    assert "scan[" in grad_text

    value_plan = Session(mode="plan", decisions=decisions)
    with activate(value_plan):

        def grad_loss_values(values):
            return jax.grad(loss)(values)

        got_grad = grad_loss_values(xs)
    same(got_grad, jax.grad(loss_ref)(xs))


def test_decision_for_diff_key_prefers_explicit():
    parent = SiteKey(parent=None, path="primal", label="outer", ordinal=0)
    primal = SiteKey(parent=parent, path="primal", label="inner", ordinal=0)
    diff = dataclasses.replace(primal, path="diff")
    primal_decision = _decision(primal, ChunkedMap(8), ChunkedMap(8), "map")
    diff_decision = _decision(diff, Vmap(), Vmap(), "map")
    both = Session(
        mode="plan",
        decisions={primal: primal_decision, diff: diff_decision},
    )
    assert decision_for(both, diff) is diff_decision
    assert decision_for(both, primal) is primal_decision
    only_primal = Session(mode="plan", decisions={primal: primal_decision})
    assert decision_for(only_primal, diff) is primal_decision


def test_map_dict_output():
    def body(x):
        return {"s": jnp.sum(x), "y": x + jnp.float32(1.0)}

    got = xmap(body, _XS)
    ref = jax.vmap(body)(_XS)
    assert isinstance(got, dict)
    same(got, ref)

    def f(xs):
        out = xmap(body, xs)
        return out["s"].sum() + out["y"].sum()

    def f_ref(xs):
        out = jax.vmap(body)(xs)
        return out["s"].sum() + out["y"].sum()

    same(jax.grad(f)(_XS), jax.grad(f_ref)(_XS))


def test_map_label_and_avals():
    session = Session(mode="plan")
    xs = (
        jnp.ones((3, 2), dtype=jnp.float32),
        jnp.ones((3,), dtype=jnp.float32),
    )
    with activate(session):
        call_line = inspect.currentframe().f_lineno + 1
        xmap(lambda tree: tree, xs)
    record = session.records[0]
    assert record.key.label == f"test_map.py:{call_line}"
    assert record.kind == "map"
    assert record.length is None
    assert record.avals == (
        jax.ShapeDtypeStruct((3, 2), jnp.float32),
        jax.ShapeDtypeStruct((3,), jnp.float32),
    )

    with activate(session):
        xmap(layer, _XS, name="decoder")
    assert session.records[-1].key.label == "decoder"
