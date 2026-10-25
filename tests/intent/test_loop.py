"""Tests for xtrax.intent.loop."""

import inspect

import jax
import jax.numpy as jnp
import pytest

from xtrax.intent import (
    Context,
    IntentError,
    Session,
    SiteDecision,
    SiteKey,
    activate,
    compile,
    current_session,
    loop,
    resolve,
)
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

    def f_of_xs(xs_arg):
        carry_i, ys_i = loop(body_xs, init, xs=xs_arg)
        return carry_i.sum() + ys_i.sum()

    def f_of_xs_ref(xs_arg):
        carry_i, ys_i = jax.lax.scan(body_xs, init, xs_arg)
        return carry_i.sum() + ys_i.sum()

    assert ok(jax.grad(f_of_xs)(xs), jax.grad(f_of_xs_ref)(xs))

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
        step, flag, state = carry
        state2 = jnp.tanh(state + 0.1)
        return (step + jnp.int32(1), jnp.logical_not(flag), state2), state2

    init = (jnp.int32(0), jnp.bool_(False), _X)

    def f(state0):
        (_, _, state), ys = loop(body, (jnp.int32(0), jnp.bool_(False), state0), length=8)
        return jnp.sum(state) + jnp.sum(ys)

    def f_ref(state0):
        (_, _, state), ys = jax.lax.scan(
            body, (jnp.int32(0), jnp.bool_(False), state0), None, length=8
        )
        return jnp.sum(state) + jnp.sum(ys)

    assert ok(loop(body, init, length=8), jax.lax.scan(body, init, None, length=8))
    assert ok(loop(body, init, length=7), jax.lax.scan(body, init, None, length=7))
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


def _cond_loop_run():
    """One named cond loop and the lax.while_loop it must match."""
    init = jnp.zeros((5,), dtype=jnp.float32)

    def cond(carry):
        return jnp.sum(carry) < 3.0

    def body(carry):
        return carry + 0.5

    def run():
        return loop(body, init, cond=cond, name="until")

    return run, lambda: jax.lax.while_loop(cond, body, init)


def test_resolved_cond_loop_executes_while_loop():
    """An exec session using resolve's plan runs the cond loop as lax.while_loop.

    Fails while resolve's WhileCarry primal is rejected before the loop runs.
    """
    run, reference = _cond_loop_run()
    plan_session = Session(mode="plan")
    with activate(plan_session):
        run()
    resolved = resolve(plan_session.records, Context())
    exe = Session(mode="exec", records=plan_session.records, decisions=resolved.by_key())
    with activate(exe):
        got = run()
    assert ok(got, reference())


def test_cond_loop_scan_primal_raises():
    """A cond loop whose decision primal is Scan is not a plain while."""
    run, _reference = _cond_loop_run()
    plan_session = Session(mode="plan")
    with activate(plan_session):
        run()
    key = plan_session.records[0].key
    decision = SiteDecision(
        key=key,
        kind="loop",
        primal=Scan(),
        diff=Scan(),
        primal_reason="test",
        diff_reason="test",
    )
    exe = Session(mode="exec", records=plan_session.records, decisions={key: decision})
    with activate(exe), pytest.raises(IntentError) as exc_info:
        run()
    err = exc_info.value
    assert err.path == "primal"
    assert err.site == str(key)
    assert err.constraint == repr(Scan())


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


def _unwrap_jaxpr(jaxpr):
    return jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr


def _inner_scan_in_outer_while(jaxpr) -> bool:
    """True when the outer loop is a while and the nested loop's primal is a scan.

    A static outer loop is a ``custom_jvp`` whose primal jaxpr is the while.
    A conditional outer loop is the while itself. Either way the nested static
    loop is the ``custom_jvp`` inside that while body.
    """
    custom = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "custom_jvp_call"]
    whiles = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "while"]
    if len(custom) == 1 and not whiles:
        jaxpr = _unwrap_jaxpr(custom[0].params["call_jaxpr"])
        whiles = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "while"]
    if len(whiles) != 1:
        return False
    body = _unwrap_jaxpr(whiles[0].params["body_jaxpr"])
    inner = [eqn for eqn in body.eqns if eqn.primitive.name == "custom_jvp_call"]
    if len(inner) != 1:
        return False
    inner_names = [eqn.primitive.name for eqn in _unwrap_jaxpr(inner[0].params["call_jaxpr"]).eqns]
    return "scan" in inner_names and "while" not in inner_names


def _nested_static_pair():
    """Fresh outer ``StaticWhile`` loop whose body calls a static ``inner`` loop."""

    def f(x):
        def body(c, _):
            def inner(c2, _):
                return c2 + jnp.float32(0.1), jnp.sum(c2)

            c3, ys = loop(inner, c, length=2, name="inner")
            return jnp.tanh(c3), jnp.sum(c) + jnp.sum(ys)

        carry, ys = loop(body, x, length=4, name="outer")
        return carry.sum() + ys.sum()

    return f


def _nested_static_ref(x):
    def body(c, _):
        def inner(c2, _):
            return c2 + jnp.float32(0.1), jnp.sum(c2)

        c3, ys = jax.lax.scan(inner, c, None, length=2)
        return jnp.tanh(c3), jnp.sum(c) + jnp.sum(ys)

    carry, ys = jax.lax.scan(body, x, None, length=4)
    return carry.sum() + ys.sum()


def _outer_inner_decisions(inner_primal):
    outer = SiteKey(parent=None, path="primal", label="outer", ordinal=0)
    inner = SiteKey(parent=outer, path="primal", label="inner", ordinal=0)
    decisions = {
        outer: SiteDecision(
            key=outer,
            kind="loop",
            primal=StaticWhile(4),
            diff=Scan(),
            primal_reason="test",
            diff_reason="test",
        ),
        inner: SiteDecision(
            key=inner,
            kind="loop",
            primal=inner_primal,
            diff=Scan(),
            primal_reason="test",
            diff_reason="test",
        ),
    }
    return outer, inner, decisions


def test_loop_nested_scan_executes_inner_decision():
    x = jnp.linspace(-1.0, 1.0, 5)
    outer_key, inner_key, decisions = _outer_inner_decisions(Scan())
    plan = Session(mode="plan", decisions=decisions)
    with activate(plan):
        planned = _nested_static_pair()(x)
    assert ok(planned, _nested_static_ref(x))
    assert [record.key for record in plan.records] == [outer_key, inner_key]
    stored = next(record for record in plan.records if record.key == outer_key)
    stored_jx = str(jax.make_jaxpr(lambda c: stored.fn(c, None))(x))
    assert "scan[" in stored_jx

    exe = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(exe):
        closed = jax.make_jaxpr(_nested_static_pair())(x)
    text = str(closed)
    assert "scan[" in text and "while[" in text
    assert _inner_scan_in_outer_while(closed.jaxpr)

    values = Session(mode="exec", records=plan.records, decisions=decisions)
    with activate(values):
        got = _nested_static_pair()(x)
    assert ok(got, _nested_static_ref(x))


def test_loop_cond_nested_site_honors_decision():
    state0 = jnp.linspace(-1.0, 1.0, 5)

    def f():
        def cond(carry):
            return carry[0] < 3

        def body(carry):
            index, state = carry

            def step(c, _):
                return c + jnp.float32(0.25), jnp.sum(c)

            state2, ys = loop(step, state, length=2, name="inner")
            return index + jnp.int32(1), state2 + jnp.sum(ys)

        return loop(body, (jnp.int32(0), state0), cond=cond, name="outer")

    def ref():
        def cond(carry):
            return carry[0] < 3

        def body(carry):
            index, state = carry

            def step(c, _):
                return c + jnp.float32(0.25), jnp.sum(c)

            state2, ys = jax.lax.scan(step, state, None, length=2)
            return index + jnp.int32(1), state2 + jnp.sum(ys)

        return jax.lax.while_loop(cond, body, (jnp.int32(0), state0))

    plan = Session(mode="plan")
    with activate(plan):
        assert ok(f(), ref())
    outer = next(record.key for record in plan.records if record.key.label == "outer")
    inner = next(record.key for record in plan.records if record.key.label == "inner")
    assert inner.parent == outer
    assert inner.path == "primal"
    assert inner.ordinal == 0
    assert len(plan.records) == 2

    decision = SiteDecision(
        key=inner,
        kind="loop",
        primal=Scan(),
        diff=Scan(),
        primal_reason="test",
        diff_reason="test",
    )
    exe = Session(mode="exec", records=plan.records, decisions={inner: decision})
    with activate(exe):
        closed = jax.make_jaxpr(f)()
    text = str(closed)
    assert "scan[" in text and "while[" in text
    assert _inner_scan_in_outer_while(closed.jaxpr)
    values = Session(mode="exec", records=plan.records, decisions={inner: decision})
    with activate(values):
        assert ok(f(), ref())


def test_loop_session_primal_and_grad_one_conversion():
    def run(kind: str) -> tuple[jax.Array, int]:
        counter = {"n": 0}

        def body(c, _):
            counter["n"] += 1
            return c + jnp.float32(counter["n"]), c

        def f(c):
            carry, _ys = loop(body, c, length=50, name="impure")
            return carry

        z = jnp.float32(0.0)
        with activate(Session(mode="plan")):
            if kind == "forward":
                out = f(z)
            elif kind == "jvp":
                out = jax.jvp(f, (z,), (jnp.float32(1.0),))[0]
            else:
                out, _grad = jax.value_and_grad(f)(z)
        return out, counter["n"]

    forward, forward_traces = run("forward")
    primal, jvp_traces = run("jvp")
    grad_value, grad_traces = run("grad")
    assert forward_traces == 2
    assert jvp_traces == 2
    assert grad_traces == 2
    assert ok(forward, jnp.float32(100.0))
    assert ok(primal, forward)
    assert ok(grad_value, forward)


def test_loop_empty_xs_matches_scan():
    init = jnp.linspace(-1.0, 1.0, 5)
    xs = jnp.zeros((0,) + init.shape)

    def body(c, x):
        return c + x, jnp.sum(c)

    carry, ys = loop(body, init, xs=xs)
    carry_r, ys_r = jax.lax.scan(body, init, xs)
    assert jnp.array_equal(carry, init)
    assert ys.shape == ys_r.shape and ys.dtype == ys_r.dtype
    assert ys.shape == (0,)
    assert ok((carry, ys), (carry_r, ys_r))

    def f(carry0):
        carry_i, ys_i = loop(body, carry0, xs=xs)
        return jnp.sum(carry_i) + jnp.sum(ys_i)

    def f_ref(carry0):
        carry_i, ys_i = jax.lax.scan(body, carry0, xs)
        return jnp.sum(carry_i) + jnp.sum(ys_i)

    assert ok(jax.grad(f)(init), jax.grad(f_ref)(init))


def test_loop_length_disagreement_raises_before_trace():
    init = jnp.linspace(-1.0, 1.0, 5)
    xs = jnp.zeros((3, 5), dtype=init.dtype)
    traced = {"n": 0}

    def body(c, a):
        traced["n"] += 1
        return c + a, jnp.sum(c)

    with pytest.raises(
        ValueError,
        match=r"length` argument of 5 which disagrees with leading axis sizes \[3\]",
    ):
        loop(body, init, xs=xs, length=5)
    assert traced["n"] == 0

    xs_tree = (jnp.ones((3, 5), dtype=init.dtype), jnp.ones((4,), dtype=init.dtype))
    with pytest.raises(
        ValueError,
        match=r"length` argument of 3 which disagrees with leading axis sizes \[3, 4\]",
    ):
        loop(body, init, xs=xs_tree, length=3)
    assert traced["n"] == 0

    carry, ys = loop(body, init, xs=xs, length=3)
    carry_r, ys_r = jax.lax.scan(body, init, xs, length=3)
    assert ok((carry, ys), (carry_r, ys_r))


def test_loop_grad_after_activate_records_diff_child():
    session = Session(mode="plan")
    x = jnp.linspace(-1.0, 1.0, 5)

    def f(x):
        def body(c, _):
            def inner(c2, _):
                return c2 + jnp.float32(0.1), jnp.sum(c2)

            c3, _ys = loop(inner, c, length=2, name="inner")
            return jnp.tanh(c3), jnp.sum(c)

        carry, ys = loop(body, x, length=4, name="outer")
        return carry.sum() + ys.sum()

    def f_ref(x):
        def body(c, _):
            def inner(c2, _):
                return c2 + jnp.float32(0.1), jnp.sum(c2)

            c3, _ys = jax.lax.scan(inner, c, None, length=2)
            return jnp.tanh(c3), jnp.sum(c)

        carry, ys = jax.lax.scan(body, x, None, length=4)
        return carry.sum() + ys.sum()

    compiled = jax.jit(f)
    with activate(session):
        compiled(x)
    assert current_session() is None
    outer = next(record.key for record in session.records if record.key.label == "outer")
    before = len(session.records)
    assert ok(jax.grad(compiled)(x), jax.grad(f_ref)(x))
    assert current_session() is None
    added = session.records[before:]
    assert len(added) == 1
    child = added[0]
    assert child.key.path == "diff"
    assert child.key.parent == outer
    assert child.key.ordinal == 0
    assert child.key.label == "inner"


def test_loop_undecided_session_primal_is_while():
    session = Session(mode="plan")
    x = jnp.linspace(-1.0, 1.0, 4)

    def f(x):
        carry, ys = loop(
            lambda c, _: (jnp.tanh(c + 0.1), jnp.sum(c)),
            x,
            length=4,
            name="plain",
        )
        return carry.sum() + ys.sum()

    with activate(session):
        jx = str(jax.make_jaxpr(f)(x))
    assert "while[" in jx and "scan[" not in jx
    assert len(session.records) == 1
    assert session.records[0].kind == "loop"


def test_compiled_loop_python_scalar_init_matches_scan():
    """``loop(..., 0.0, xs=xs)`` inside ``compile`` matches ``jax.lax.scan``."""

    def scalar_init(values):
        return loop(lambda c, x: (c + jnp.sum(x), x), 0.0, xs=values, name="l")

    xs = jnp.ones((4, 3), jnp.float32)
    got_carry, got_ys = compile(scalar_init)(xs)
    exp_carry, exp_ys = jax.lax.scan(lambda c, x: (c + jnp.sum(x), x), 0.0, xs)
    assert jnp.allclose(got_carry, exp_carry, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(got_ys, exp_ys, rtol=1e-5, atol=1e-5)


def test_loop_empty_xs_raises_scan_value_error():
    """Empty ``xs`` with no length raises the same ``ValueError`` as ``jax.lax.scan``."""
    with pytest.raises(
        ValueError,
        match=r"scan got no values to scan over and `length` not provided\.",
    ):
        loop(lambda c, x: (c, x), jnp.float32(0.0), xs=())
