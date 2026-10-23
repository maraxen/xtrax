"""Tests for xtrax.intent.loop."""

import inspect

import jax
import jax.numpy as jnp
import pytest

from xtrax.intent import IntentError, Session, SiteDecision, SiteKey, activate, loop
from xtrax.tiling import Vmap
from xtrax.tiling.strategy import Scan, StaticWhile

_LENGTH = 32
_W = jnp.float32(0.7)
_X = jnp.linspace(-1.0, 1.0, 5)
_T = jnp.ones_like(_X)
_XB = jnp.stack([_X, 0.5 * _X])


def ok(a, b) -> bool:
    leaves_a, def_a = jax.tree_util.tree_flatten(a)
    leaves_b, def_b = jax.tree_util.tree_flatten(b)
    if def_a != def_b or len(leaves_a) != len(leaves_b):
        return False
    return all(
        bool(jnp.allclose(left, right, atol=1e-5))
        for left, right in zip(leaves_a, leaves_b, strict=True)
    )


def _step(c, _):
    return jnp.tanh(c * _W + 0.1), jnp.sum(c)


def f_loop(x):
    carry, ys = loop(_step, x, length=_LENGTH)
    return carry.sum() + ys.sum()


def f_scan_ref(x):
    carry, ys = jax.lax.scan(_step, x, None, length=_LENGTH)
    return carry.sum() + ys.sum()


def test_loop_rejects_cond_mixed_with_length_and_empty_call():
    init = jnp.zeros((2,), dtype=jnp.float32)

    def body(c, _):
        return c, jnp.sum(c)

    with pytest.raises(ValueError, match="cond form"):
        loop(body, init, length=2, cond=lambda c: jnp.array(True))
    with pytest.raises(ValueError, match="requires"):
        loop(body, init)


def test_loop_forward_is_while_not_scan():
    jx = str(jax.make_jaxpr(f_loop)(_X))
    assert "while[" in jx and "scan[" not in jx
    assert ok(jax.jit(f_loop)(_X), f_scan_ref(_X))


def test_loop_transforms_match_scan_reference():
    assert ok(jax.grad(f_loop)(_X), jax.grad(f_scan_ref)(_X))
    assert ok(jax.jvp(f_loop, (_X,), (_T,))[1], jax.jvp(f_scan_ref, (_X,), (_T,))[1])
    assert ok(jax.jacfwd(jax.jacrev(f_loop))(_X), jax.hessian(f_scan_ref)(_X))
    assert ok(jax.vmap(jax.grad(f_loop))(_XB), jax.vmap(jax.grad(f_scan_ref))(_XB))
    gj = str(jax.make_jaxpr(jax.grad(f_loop))(_X))
    assert "scan[" in gj and "while[" not in gj


def test_loop_closure_over_params_grad():
    def f(w):
        def body(c, _):
            return jnp.tanh(c * w + 0.1), jnp.sum(c)

        carry, ys = loop(body, _X, length=_LENGTH)
        return carry.sum() + ys.sum()

    def f_ref(w):
        def body(c, _):
            return jnp.tanh(c * w + 0.1), jnp.sum(c)

        carry, ys = jax.lax.scan(body, _X, None, length=_LENGTH)
        return carry.sum() + ys.sum()

    w = jnp.float32(0.7)
    assert ok(jax.grad(f)(w), jax.grad(f_ref)(w))


def test_loop_collects_ys_like_scan():
    init = _X

    def body_xs(c, x):
        c2 = jnp.tanh(c + x)
        return c2, jnp.sum(c)

    xs = jnp.linspace(0.0, 1.0, 8)
    carry, ys = loop(body_xs, init, xs=xs)
    carry_r, ys_r = jax.lax.scan(body_xs, init, xs)
    assert ys.shape == ys_r.shape and ys.dtype == ys_r.dtype
    assert ok((carry, ys), (carry_r, ys_r))

    def f_xs(init):
        carry_i, ys_i = loop(body_xs, init, xs=xs)
        return carry_i.sum() + ys_i.sum()

    def f_xs_ref(init):
        carry_i, ys_i = jax.lax.scan(body_xs, init, xs)
        return carry_i.sum() + ys_i.sum()

    assert ok(jax.grad(f_xs)(init), jax.grad(f_xs_ref)(init))

    def body_len(c, _):
        c2 = jnp.tanh(c + 0.1)
        return c2, jnp.sum(c)

    carry_l, ys_l = loop(body_len, init, length=8)
    carry_lr, ys_lr = jax.lax.scan(body_len, init, None, length=8)
    assert ys_l.shape == ys_lr.shape and ys_l.dtype == ys_lr.dtype
    assert ok((carry_l, ys_l), (carry_lr, ys_lr))


def test_loop_pytree_carry_and_ys():
    def body(carry, x):
        a, b = carry
        a2 = a + x
        b2 = {"u": b["u"] + x, "v": b["v"] + jnp.float32(1.0)}
        y = {"s": jnp.sum(a2), "u": b2["u"]}
        return (a2, b2), y

    init = (jnp.zeros((5,)), {"u": jnp.ones((5,)), "v": jnp.float32(0.0)})
    xs = jnp.linspace(-0.5, 0.5, 6)
    got = loop(body, init, xs=xs)
    ref = jax.lax.scan(body, init, xs)
    assert ok(got, ref)
    _, ys = got
    _, ys_r = ref
    assert isinstance(ys, dict)
    assert ys["s"].shape == ys_r["s"].shape
    assert ys["u"].dtype == ys_r["u"].dtype


def test_loop_grad_with_int_counter():
    def body(carry, _):
        step, state = carry
        state2 = jnp.tanh(state + 0.1)
        return (step + jnp.int32(1), state2), state2

    def f(state0):
        (_, state), ys = loop(body, (jnp.int32(0), state0), length=8)
        return jnp.sum(state) + jnp.sum(ys)

    def f_ref(state0):
        (_, state), ys = jax.lax.scan(body, (jnp.int32(0), state0), None, length=8)
        return jnp.sum(state) + jnp.sum(ys)

    assert ok(f(_X), f_ref(_X))
    assert ok(jax.grad(f)(_X), jax.grad(f_ref)(_X))


def test_loop_degenerate_lengths():
    def body(c, _):
        return jnp.tanh(c + 0.1), jnp.sum(c)

    for length in (0, 1):

        def f(x, length=length):
            carry, ys = loop(body, x, length=length)
            return jnp.sum(carry) + jnp.sum(ys)

        def f_ref(x, length=length):
            carry, ys = jax.lax.scan(body, x, None, length=length)
            return jnp.sum(carry) + jnp.sum(ys)

        carry, ys = loop(body, _X, length=length)
        carry_r, ys_r = jax.lax.scan(body, _X, None, length=length)
        assert ys.shape == ys_r.shape and ys.dtype == ys_r.dtype
        assert ok((carry, ys), (carry_r, ys_r))
        if length == 0:
            assert jnp.array_equal(carry, _X)
            assert ys.shape == (0,)
        assert ok(jax.grad(f)(_X), jax.grad(f_ref)(_X))


def test_loop_mixed_closures():
    py = 0.25
    frozen = jax.lax.stop_gradient(jnp.float32(0.3))

    def f(w):
        def body(c, _):
            c2 = jnp.tanh(c * w + py + frozen)
            return c2, jnp.sum(c)

        carry, ys = loop(body, _X, length=16)
        return jnp.sum(carry) + jnp.sum(ys)

    def f_ref(w):
        def body(c, _):
            c2 = jnp.tanh(c * w + py + frozen)
            return c2, jnp.sum(c)

        carry, ys = jax.lax.scan(body, _X, None, length=16)
        return jnp.sum(carry) + jnp.sum(ys)

    w = jnp.float32(0.7)
    assert ok(f(w), f_ref(w))
    assert ok(jax.grad(f)(w), jax.grad(f_ref)(w))


def test_loop_while_form_is_plain_while_loop():
    def cond(carry):
        return jnp.sum(carry) < 3.0

    def body(carry):
        return carry + 0.5

    init = jnp.zeros((5,), dtype=jnp.float32)
    assert ok(loop(body, init, cond=cond), jax.lax.while_loop(cond, body, init))

    session = Session(mode="plan")
    with activate(session):
        got = loop(body, init, cond=cond)
    assert ok(got, jax.lax.while_loop(cond, body, init))
    record = session.records[0]
    assert record.kind == "loop"
    assert record.length is None
    assert record.avals == (jax.ShapeDtypeStruct((5,), jnp.float32),)
    _converted, consts = jax.closure_convert(body, init)
    assert ok(record.fn(init, *tuple(consts)), body(init))


def test_loop_outside_compile_exports():
    exported = jax.export.export(jax.jit(f_loop), platforms=["cpu"])(_X)
    assert ok(exported.call(_X), f_scan_ref(_X))
    assert "custom_call" not in exported.mlir_module()


def test_loop_label_points_at_user_call():
    session = Session(mode="plan")
    init = jnp.zeros((2,), dtype=jnp.float32)
    weight = jnp.float32(0.5)

    def body(c, x):
        return c * weight + x, jnp.sum(c)

    xs = jnp.ones((3, 2), dtype=jnp.float32)
    with activate(session):
        call_line = inspect.currentframe().f_lineno + 1
        loop(body, init, xs=xs)
    record = session.records[0]
    assert record.key.label == f"test_loop.py:{call_line}"
    assert record.kind == "loop"
    assert record.length == 3
    assert record.avals == (
        jax.ShapeDtypeStruct((2,), jnp.float32),
        jax.ShapeDtypeStruct((3, 2), jnp.float32),
    )
    _converted, consts = jax.closure_convert(body, init, xs[0])
    assert record.fn is not body
    assert ok(record.fn(init, xs[0], *tuple(consts)), body(init, xs[0]))

    def body_named(c, _):
        return c + 1, jnp.sum(c)

    with activate(session):
        loop(body_named, init, length=2, name="decoder")
    assert session.records[-1].key.label == "decoder"


def test_loop_plan_grad_records_inner_on_diff():
    session = Session(mode="plan")
    x = jnp.linspace(-1.0, 1.0, 5)

    def f(x):
        def body(c, _):
            def inner(c2, _):
                return c2 + 0.1, jnp.sum(c2)

            c3, _ys = loop(inner, c, length=2)
            return jnp.tanh(c3), jnp.sum(c)

        carry, ys = loop(body, x, length=4)
        return carry.sum() + ys.sum()

    with activate(session):
        jax.make_jaxpr(jax.grad(f))(x)

    primal = [r for r in session.records if r.kind == "loop" and r.key.path == "primal"]
    assert len(primal) == 1
    outer = primal[0].key
    assert outer.parent is None
    nested = [
        r
        for r in session.records
        if r.kind == "loop" and r.key.parent == outer and r.key.path == "diff"
    ]
    assert len(nested) == 1


def _decoder_key() -> SiteKey:
    return SiteKey(parent=None, path="primal", label="decoder", ordinal=0)


def _decoder(x):
    carry, ys = loop(
        lambda c, _: (jnp.tanh(c + 0.1), jnp.sum(c)),
        x,
        length=4,
        name="decoder",
    )
    return carry.sum() + ys.sum()


def _decoder_while(x):
    return _decoder(x)


def _decoder_scan(x):
    return _decoder(x)


def _decoder_ref(x):
    carry, ys = jax.lax.scan(
        lambda c, _: (jnp.tanh(c + 0.1), jnp.sum(c)),
        x,
        None,
        length=4,
    )
    return carry.sum() + ys.sum()


def _session_for(strategy) -> Session:
    key = _decoder_key()
    decision = SiteDecision(
        key=key,
        kind="loop",
        primal=strategy,
        diff=Scan(),
        primal_reason="test",
        diff_reason="test",
    )
    return Session(mode="plan", decisions={key: decision})


def test_loop_primal_decision_selects_strategy():
    x = jnp.linspace(-1.0, 1.0, 4)
    with activate(_session_for(StaticWhile(4))):
        jx = str(jax.make_jaxpr(_decoder_while)(x))
        assert "while[" in jx and "scan[" not in jx
    with activate(_session_for(StaticWhile(4))):
        assert ok(_decoder(x), _decoder_ref(x))

    with activate(_session_for(Scan())):
        jx = str(jax.make_jaxpr(_decoder_scan)(x))
        assert "scan[" in jx and "while[" not in jx
    with activate(_session_for(Scan())):
        assert ok(_decoder(x), _decoder_ref(x))

    with activate(_session_for(Vmap())), pytest.raises(IntentError) as exc:
        _decoder(x)
    err = exc.value
    assert err.path == "primal"
    assert "decoder" in err.site
    assert "Vmap" in err.constraint
    assert "Vmap" in str(err)
