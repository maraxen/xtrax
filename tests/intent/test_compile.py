"""Tests for xtrax.intent.compile."""

import logging
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.intent import (
    Context,
    IntentError,
    Plan,
    SiteDecision,
    SiteKey,
    SiteMismatchError,
    UnsatisfiableContextError,
    compile,
    loop,
    map,
)
from xtrax.intent._trace import decision_for
from xtrax.intent.sites import Session
from xtrax.tiling.budget import MemoryBudget
from xtrax.tiling.strategy import ChunkedMap, Scan, Vmap


def _same(got, expected) -> None:
    """Assert two pytrees match leaf for leaf."""
    got_leaves, got_def = jax.tree_util.tree_flatten(got)
    expected_leaves, expected_def = jax.tree_util.tree_flatten(expected)
    assert got_def == expected_def
    for left, right in zip(got_leaves, expected_leaves, strict=True):
        assert jnp.allclose(left, right, rtol=1e-5, atol=1e-5)


def test_compile_unconstrained_equals_plain_jax():
    """No budget: a loop and a map match scan and vmap."""

    def fn(xs):
        def body(carry, x):
            return carry + x, jnp.sum(carry)

        init = jnp.zeros(xs.shape[1:], dtype=xs.dtype)
        carry, ys = loop(body, init, xs=xs, name="steps")
        scaled = map(lambda y: y * jnp.float32(2), ys, name="rows")
        return carry.sum() + scaled.sum()

    def reference(xs):
        def body(carry, x):
            return carry + x, jnp.sum(carry)

        init = jnp.zeros(xs.shape[1:], dtype=xs.dtype)
        carry, ys = jax.lax.scan(body, init, xs)
        scaled = jax.vmap(lambda y: y * jnp.float32(2))(ys)
        return carry.sum() + scaled.sum()

    xs = jnp.linspace(-1.0, 1.0, 12, dtype=jnp.float32).reshape(4, 3)
    compiled = compile(fn)
    _same(compiled(xs), reference(xs))


def test_nested_sites_no_false_mismatch():
    """Loop-in-map, map-in-loop, and both cond branches stay consistent.

    Plain, grad, and checkpoint runs match the scan/vmap reference and do not
    raise SiteMismatchError.
    """

    def fn(xs, pred):
        def loop_in_map(values):
            def row(x):
                def body(carry, _):
                    return carry + x, jnp.sum(carry)

                carry, ys = loop(body, jnp.zeros_like(x), length=2, name="true_loop")
                return carry + jnp.sum(ys)

            return map(row, values, name="true_map")

        def map_in_loop(values):
            def body(carry, x):
                y = map(lambda z: z + carry, x, name="false_map")
                return carry + jnp.sum(y), jnp.sum(y)

            init = jnp.zeros(values.shape[-1:], dtype=values.dtype)
            carry, ys = loop(body, init, xs=values, name="false_loop")
            return jnp.sum(carry) + jnp.sum(ys)

        return jax.lax.cond(
            pred,
            lambda values: jnp.sum(loop_in_map(values)),
            lambda values: map_in_loop(values),
            xs,
        )

    def reference(xs, pred):
        def loop_in_map(values):
            def row(x):
                def body(carry, _):
                    return carry + x, jnp.sum(carry)

                carry, ys = jax.lax.scan(body, jnp.zeros_like(x), None, length=2)
                return carry + jnp.sum(ys)

            return jax.vmap(row)(values)

        def map_in_loop(values):
            def body(carry, x):
                y = jax.vmap(lambda z: z + carry)(x)
                return carry + jnp.sum(y), jnp.sum(y)

            init = jnp.zeros(values.shape[-1:], dtype=values.dtype)
            carry, ys = jax.lax.scan(body, init, values)
            return jnp.sum(carry) + jnp.sum(ys)

        return jax.lax.cond(
            pred,
            lambda values: jnp.sum(loop_in_map(values)),
            lambda values: map_in_loop(values),
            xs,
        )

    xs = jnp.linspace(-0.5, 0.5, 12, dtype=jnp.float32).reshape(4, 3)
    pred = jnp.bool_(True)
    compiled = compile(fn)
    _same(compiled(xs, pred), reference(xs, pred))
    _same(jax.grad(compiled)(xs, pred), jax.grad(reference)(xs, pred))
    _same(jax.checkpoint(compiled)(xs, pred), jax.checkpoint(reference)(xs, pred))


def test_nested_site_under_grad_uses_diff_decision():
    """A nested map's diff strategy comes from the plan, not the Vmap default."""

    def fn(xs):
        def row(values):
            return map(lambda z: jnp.sin(z), values, name="inner")

        return jnp.sum(map(row, xs, name="outer"))

    def reference(xs):
        def row(values):
            return jax.vmap(jnp.sin)(values)

        return jnp.sum(jax.vmap(row)(xs))

    xs = jnp.linspace(-1.0, 1.0, 24, dtype=jnp.float32).reshape(4, 6)
    compiled = compile(fn, overrides={"inner": (Vmap(), ChunkedMap(batch_size=2))})
    plan = compiled.plan_for(xs)
    inner = next(
        decision
        for decision in plan.decisions
        if decision.key.label == "inner" and decision.key.path == "primal"
    )
    assert isinstance(inner.primal, Vmap)
    assert isinstance(inner.diff, ChunkedMap)
    forward = str(jax.make_jaxpr(compiled)(xs))
    assert "scan[" not in forward
    backward = str(jax.make_jaxpr(jax.grad(compiled))(xs))
    assert "scan[" in backward
    _same(jax.grad(compiled)(xs), jax.grad(reference)(xs))


def test_nonreproducible_trace_raises_before_execution():
    """A trace-time shape change names the site and the planned avals."""
    state = {"n": 0}

    def fn(xs):
        state["n"] += 1
        rows = xs if state["n"] == 1 else xs[:-1]
        return map(lambda z: jnp.sum(z), rows, name="rows").sum()

    compiled = compile(fn)
    with pytest.raises(SiteMismatchError) as exc_info:
        compiled(jnp.ones((6, 3), dtype=jnp.float32))
    err = exc_info.value
    assert "rows" in err.site
    assert "planned" in err.constraint


def test_unsatisfiable_context_raises_at_compile_call():
    """An export cond-loop is refused when the compiled function is called."""

    def fn(x):
        def cond(carry):
            return carry[0] < jnp.float32(3)

        def body(carry):
            return carry + jnp.float32(1)

        return loop(body, x, cond=cond, name="stop")

    compiled = compile(fn, Context(target="export", export_format="stablehlo"))
    with pytest.raises(UnsatisfiableContextError) as exc_info:
        compiled(jnp.zeros(2, dtype=jnp.float32))
    err = exc_info.value
    assert "stop" in err.site
    assert err.path == "primal"
    assert err.constraint == "export forbids unbounded while_loop"


def test_plans_once_zero_recompiles(caplog):
    """One plan and no further XLA compiles across three calls of one shape."""

    def fn(xs):
        return map(lambda z: z + jnp.float32(1), xs, name="batch").sum()

    compiled = compile(fn)
    xs = jnp.ones((4, 3), dtype=jnp.float32)
    with jax.log_compiles(True), caplog.at_level(logging.WARNING, logger="jax"):
        compiled(xs)
        after_first = len(caplog.records)
        compiled(xs)
        compiled(xs)
    assert compiled.plans == 1
    later = [record.getMessage() for record in caplog.records[after_first:]]
    compiled_again = [
        message
        for message in later
        if "Finished XLA compilation" in message or "Compiling" in message
    ]
    assert compiled_again == []


def test_plan_table_and_json_from_compiled():
    """plan_for returns a Plan whose table and JSON list the site."""

    def fn(xs):
        return map(lambda z: z + jnp.float32(1), xs, name="batch").sum()

    ctx = Context()
    compiled = compile(fn, ctx)
    xs = jnp.ones((4, 3), dtype=jnp.float32)
    plan = compiled.plan_for(xs)
    assert isinstance(plan, Plan)
    assert "batch" in plan.table()
    assert "Vmap" in plan.table()
    restored = Plan.from_json(plan.to_json(), context=ctx)
    assert restored.decisions == plan.decisions
    assert compiled.plans == 1
    compiled(xs)
    assert compiled.plans == 1


def test_compile_new_shape_replans():
    """A new leading shape plans again; repeating the first shape does not."""

    def fn(xs):
        return map(lambda z: jnp.sum(z), xs, name="batch")

    compiled = compile(fn)
    compiled(jnp.ones((4, 3), dtype=jnp.float32))
    assert compiled.plans == 1
    compiled(jnp.ones((8, 3), dtype=jnp.float32))
    assert compiled.plans == 2
    compiled(jnp.ones((4, 3), dtype=jnp.float32))
    assert compiled.plans == 2


def test_compile_under_outer_vmap():
    """compile plans from tracer shapes under an outer vmap."""

    def fn(xs):
        return map(lambda z: z * jnp.float32(2) + jnp.float32(1), xs, name="batch").sum()

    def reference(xs):
        return jax.vmap(lambda z: z * jnp.float32(2) + jnp.float32(1))(xs).sum()

    compiled = compile(fn)
    xs = jnp.ones((5, 4, 3), dtype=jnp.float32)
    _same(jax.vmap(compiled)(xs), jax.vmap(reference)(xs))


def test_no_equinox_import():
    """import xtrax.intent and a tiny compile do not load equinox."""
    code = """
import sys

class _BlockEquinox:
    def find_spec(self, fullname, path, target=None):
        if fullname == "equinox" or fullname.startswith("equinox."):
            raise ImportError("equinox import blocked")
        return None

sys.meta_path.insert(0, _BlockEquinox())
import jax.numpy as jnp
from xtrax.intent import compile, map

def f(x):
    return map(lambda z: z + 1, x, name="batch")

compile(f)(jnp.ones((4, 3)))
import xtrax.tiling as tiling
names = dir(tiling)
expected = (
    "BucketIterator",
    "ChunkedMapIterator",
    "JaxScanIterator",
    "MapIterator",
    "ScanIterator",
    "VmapIterator",
    "WhileLoopIterator",
    "WhileLoopWithYsIterator",
)
missing = [name for name in expected if name not in names]
assert not missing, missing
assert "TYPE_CHECKING" not in names
assert "equinox" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_compile_exact_inputs_skip_diff_pass():
    """An all-exact map skips the diff plan pass."""
    traces = {"n": 0}

    def fn(xs):
        traces["n"] += 1
        return map(lambda z: z + jnp.int32(1), xs, name="batch")

    def reference(xs):
        return jax.vmap(lambda z: z + jnp.int32(1))(xs)

    xs = jnp.arange(12, dtype=jnp.int32).reshape(4, 3)
    compiled = compile(fn)
    _same(compiled(xs), reference(xs))
    assert traces["n"] == 2
    assert compiled.plans == 1
    plan = compiled.plan_for(xs)
    assert compiled.plans == 1
    assert plan.decisions
    for decision in plan.decisions:
        assert decision.diff is not None
        assert decision.diff_reason == "same strategy as primal in the MVP"


def _vmap_is_expensive(decisions):
    """Count Vmap at 100 bytes and every other strategy at 10."""
    return sum(100 if isinstance(decision.strategy, Vmap) else 10 for decision in decisions)


def test_posthoc_scan_around_while_check(monkeypatch):
    """ChunkedMap around a loop is rejected unless an override bypassed the hazard."""
    resolve_module = sys.modules["xtrax.intent.resolve"]
    compile_module = sys.modules["xtrax.intent.compile"]
    # Only the post-hoc set includes "cpu": this simulates a resolver miss, so the
    # compiled jaxpr check is the only line of defence.
    monkeypatch.setattr(compile_module, "SCAN_AROUND_WHILE_BACKENDS", frozenset({"cpu"}))
    budget = MemoryBudget(bytes=50, estimate=_vmap_is_expensive)

    def fn(xs):
        def row(x):
            def body(carry, _):
                return carry + x, jnp.sum(carry)

            carry, ys = loop(body, jnp.zeros_like(x), length=3, name="inner")
            return carry + jnp.sum(ys)

        return map(row, xs, name="outer")

    def reference(xs):
        def row(x):
            def body(carry, _):
                return carry + x, jnp.sum(carry)

            carry, ys = jax.lax.scan(body, jnp.zeros_like(x), None, length=3)
            return carry + jnp.sum(ys)

        return jax.vmap(row)(xs)

    xs = jnp.linspace(-0.2, 0.2, 32, dtype=jnp.float32).reshape(8, 4)
    blocked = compile(fn, Context(memory_budget=budget, chunk_size=4))
    blocked_plan = blocked.plan_for(xs)
    assert not any(
        "bypassed scan_around_while" in warning
        for decision in blocked_plan.decisions
        for warning in decision.warnings
    )
    with pytest.raises(IntentError) as exc_info:
        blocked(xs)
    assert exc_info.value.constraint == "scan_around_while"
    assert not isinstance(exc_info.value, UnsatisfiableContextError)

    # With the resolver seeing the same backend set, the hazard is refused at plan time.
    monkeypatch.setattr(resolve_module, "SCAN_AROUND_WHILE_BACKENDS", frozenset({"cpu"}))
    refused = compile(fn, Context(memory_budget=budget, chunk_size=4))
    with pytest.raises(UnsatisfiableContextError) as refused_info:
        refused.plan_for(xs)
    assert refused_info.value.constraint == "scan_around_while"
    assert "outer" in refused_info.value.site

    allowed = compile(
        fn,
        Context(backend="cpu", memory_budget=budget, chunk_size=4),
        overrides={"outer": (ChunkedMap(batch_size=4), ChunkedMap(batch_size=4))},
    )
    allowed_plan = allowed.plan_for(xs)
    assert any(
        warning == "override:outer bypassed scan_around_while"
        for decision in allowed_plan.decisions
        for warning in decision.warnings
    )
    _same(allowed(xs), reference(xs))


def _sum_leaves(tree) -> jax.Array:
    """Scalar reduction so ``jax.grad`` is defined on a pytree result."""
    return sum(jnp.sum(leaf) for leaf in jax.tree_util.tree_leaves(tree))


def test_budgeted_loop_in_map_grad_and_jvp():
    """ChunkedMap on both paths around a loop matches vmap(scan) under grad and jvp.

    ``xs`` is ``(8, 5, 3)`` and the inner loop length is 5. The budget demotes
    the outer map to ``ChunkedMap(batch_size=2)`` on the primal and the diff
    path. Fails while the JVP rule's primal shapes disagree with the
    custom_jvp primal, or while a second inner-loop visit raises
    SiteMismatchError.
    """

    def fn(xs, w):
        def one(row):
            def step(carry, x):
                y = x * w + carry
                return carry + y, y

            init = jnp.zeros(row.shape[-1:], dtype=row.dtype)
            return loop(step, init, xs=row, name="inner_loop")

        return map(one, xs, name="outer_map")

    def reference(xs, w):
        def one(row):
            def step(carry, x):
                y = x * w + carry
                return carry + y, y

            init = jnp.zeros(row.shape[-1:], dtype=row.dtype)
            return jax.lax.scan(step, init, row)

        return jax.vmap(one)(xs)

    xs = jnp.linspace(-0.4, 0.6, 8 * 5 * 3, dtype=jnp.float32).reshape(8, 5, 3)
    w = jnp.array([0.2, 0.3, 0.4], dtype=jnp.float32)
    compiled = compile(
        fn,
        Context(
            memory_budget=MemoryBudget(bytes=50, estimate=_vmap_is_expensive),
            chunk_size=2,
            backend="cpu",
        ),
    )
    plan = compiled.plan_for(xs, w)
    outer = next(
        decision
        for decision in plan.decisions
        if decision.key.label == "outer_map" and decision.key.path == "primal"
    )
    assert isinstance(outer.primal, ChunkedMap)
    assert outer.primal.batch_size == 2
    assert isinstance(outer.diff, ChunkedMap)
    assert outer.diff.batch_size == 2

    def loss_compiled(values, weight):
        return _sum_leaves(compiled(values, weight))

    def loss_reference(values, weight):
        return _sum_leaves(reference(values, weight))

    _same(jax.grad(loss_compiled)(xs, w), jax.grad(loss_reference)(xs, w))
    tangents = (jnp.ones_like(xs), jnp.ones_like(w))
    got_primal, got_tangent = jax.jvp(compiled, (xs, w), tangents)
    exp_primal, exp_tangent = jax.jvp(reference, (xs, w), tangents)
    _same(got_primal, exp_primal)
    _same(got_tangent, exp_tangent)


def test_budgeted_map_in_loop_closes_over_carry_grad():
    """A chunked diff map that closes over the loop carry matches scan/vmap under grad.

    The body is ``sin(z * w + carry)``. The primal path stays ``Vmap`` and the
    diff path is ``ChunkedMap(batch_size=2)``. Gradients with respect to the
    inputs and ``w`` must match. The carry update is scaled by 0.1: at scale 1
    the carry feedback amplifies float32 rounding to about 1e-3, and vmap, lax.map
    and the chunked path each sit that far from the float64 gradient.
    """

    def fn(xs, w):
        def step(carry, row):
            y = map(lambda z: jnp.sin(z * w + carry), row, name="inner_map")
            return carry + 0.1 * jnp.sum(y), y

        return loop(step, jnp.float32(0.0), xs=xs, name="outer_loop")

    def reference(xs, w):
        def step(carry, row):
            y = jax.vmap(lambda z: jnp.sin(z * w + carry))(row)
            return carry + 0.1 * jnp.sum(y), y

        return jax.lax.scan(step, jnp.float32(0.0), xs)

    xs = jnp.linspace(-0.3, 0.3, 16, dtype=jnp.float32).reshape(4, 4)
    w = jnp.linspace(0.2, 0.5, 4, dtype=jnp.float32)
    compiled = compile(
        fn,
        Context(backend="cpu"),
        overrides={"inner_map": (Vmap(), ChunkedMap(batch_size=2))},
    )
    plan = compiled.plan_for(xs, w)
    inner = next(
        decision
        for decision in plan.decisions
        if decision.key.label == "inner_map" and decision.key.path == "primal"
    )
    assert isinstance(inner.primal, Vmap)
    assert isinstance(inner.diff, ChunkedMap)
    assert inner.diff.batch_size == 2

    def loss_compiled(values, weight):
        carry, _ys = compiled(values, weight)
        return jnp.sum(carry)

    def loss_reference(values, weight):
        carry, _ys = reference(values, weight)
        return jnp.sum(carry)

    for arg in (0, 1):
        _same(
            jax.grad(loss_compiled, argnums=arg)(xs, w),
            jax.grad(loss_reference, argnums=arg)(xs, w),
        )


def test_int_budget_map_closed_over_const_matches_vmap():
    """An int memory budget plans a map that closes over a traced value.

    ``plan_for`` succeeds and the call matches ``jax.vmap``, including a
    one-leaf pytree input.
    """

    def fn(xs, w):
        return map(lambda z: jnp.sin(z * w), xs, name="m")

    xs = jnp.ones((8, 3), jnp.float32)
    w = jnp.ones((3,), jnp.float32)
    compiled = compile(fn, Context(memory_budget=10**9))
    compiled.plan_for(xs, w)
    _same(compiled(xs, w), jax.vmap(lambda z: jnp.sin(z * w))(xs))

    def fn_tree(tree, weight):
        return map(lambda z: jnp.sin(z["pos"] * weight), tree, name="m")

    tree = {"pos": xs}
    compiled_tree = compile(fn_tree, Context(memory_budget=10**9))
    compiled_tree.plan_for(tree, w)
    _same(compiled_tree(tree, w), jax.vmap(lambda z: jnp.sin(z["pos"] * w))(tree))


def test_compile_python_scalar_argument_matches_vmap():
    """``compile(f)(xs, 2.0)`` matches ``jax.vmap``."""

    def fn(xs, scale):
        return map(lambda z: z * scale, xs, name="m")

    xs = jnp.ones((4, 3), jnp.float32)
    _same(compile(fn)(xs, 2.0), jax.vmap(lambda z: z * 2.0)(xs))


def test_compile_weak_scalar_promotes_like_plain_jax():
    """A weak float32 scalar promotes like JAX and does not share a strong plan."""

    def fn(xs, scale):
        ys = map(lambda z: z * scale, xs, name="scale")
        return map(jnp.sin, ys, name="sin")

    xs = jnp.ones((4, 3), jnp.bfloat16)
    weak = jnp.asarray(2.0)
    strong = jnp.array(2.0, dtype=jnp.float32)

    def reference(scale):
        return jax.vmap(jnp.sin)(jax.vmap(lambda z: z * scale)(xs))

    compiled = compile(fn)
    got_weak = compiled(xs, weak)
    got_strong = compiled(xs, strong)
    assert got_weak.dtype == reference(weak).dtype
    assert got_strong.dtype == reference(strong).dtype
    _same(got_weak.astype(jnp.float32), reference(weak).astype(jnp.float32))
    _same(got_strong.astype(jnp.float32), reference(strong).astype(jnp.float32))


def _host_scale(value):
    """Scale ``value`` on the host. ``pure_callback`` has no JVP rule."""
    return jax.pure_callback(
        lambda raw: np.asarray(raw) * 2,
        jax.ShapeDtypeStruct(value.shape, value.dtype),
        value,
    )


def test_nondiff_argument_keeps_diff_sites_for_other_arguments():
    """One non-differentiable input does not drop diff sites for the others.

    ``aux`` goes through ``pure_callback``. Gradients with respect to ``w``
    still match plain JAX, and the plan names ``aux`` and ``ValueError``
    instead of the callback's message.
    """

    def fn(xs, w, aux):
        scale = jnp.sum(_host_scale(aux))

        def step(carry, row):
            y = map(lambda z: jnp.sin(z * w + carry), row, name="inner_map")
            return carry + 0.1 * jnp.sum(y), y

        carry, _ys = loop(step, jnp.float32(0.0), xs=xs, name="outer_loop")
        return carry * scale

    def reference(xs, w, aux):
        scale = jnp.sum(_host_scale(aux))

        def step(carry, row):
            y = jax.vmap(lambda z: jnp.sin(z * w + carry))(row)
            return carry + 0.1 * jnp.sum(y), y

        carry, _ys = jax.lax.scan(step, jnp.float32(0.0), xs)
        return carry * scale

    xs = jnp.linspace(-0.3, 0.3, 16, dtype=jnp.float32).reshape(4, 4)
    w = jnp.linspace(0.2, 0.5, 4, dtype=jnp.float32)
    aux = jnp.ones((3,), jnp.float32)
    compiled = compile(fn, Context(backend="cpu"))
    _same(jax.grad(compiled, argnums=1)(xs, w, aux), jax.grad(reference, argnums=1)(xs, w, aux))
    plan = compiled.plan_for(xs, w, aux)
    assert any(
        decision.key.label == "inner_map" and decision.key.path == "diff"
        for decision in plan.decisions
    )
    outer = next(
        decision
        for decision in plan.decisions
        if decision.key.label == "outer_loop" and decision.key.path == "primal"
    )
    assert isinstance(outer.diff, Scan)
    assert outer.diff_reason == (
        "diff path resolves to Scan; not differentiable w.r.t. aux (ValueError)"
    )
    assert "Pure callbacks" not in outer.diff_reason
    assert "custom_jvp" not in outer.diff_reason


def test_primal_diff_sites_kept_for_integer_input():
    """Diff sites recorded while planning an internal grad survive an integer input.

    The diff pass does not run for an integer ``scale``. The primal pass still
    records the differentiated map, and the call matches ``scan`` / ``vmap``.
    """

    def fn(scale):
        xs = jnp.ones((4, 4), jnp.float32) * scale.astype(jnp.float32)

        def energy(weight):
            def step(carry, row):
                y = map(lambda z: jnp.sin(z * weight + carry), row, name="m")
                return carry + 0.1 * jnp.sum(y), y

            return loop(step, jnp.float32(0.0), xs=xs, name="l")[0]

        return jax.grad(energy)(jnp.float32(0.3))

    def reference(scale):
        xs = jnp.ones((4, 4), jnp.float32) * scale.astype(jnp.float32)

        def energy(weight):
            def step(carry, row):
                y = jax.vmap(lambda z: jnp.sin(z * weight + carry))(row)
                return carry + 0.1 * jnp.sum(y), y

            return jax.lax.scan(step, jnp.float32(0.0), xs)[0]

        return jax.grad(energy)(jnp.float32(0.3))

    scale = jnp.int32(1)
    compiled = compile(fn)
    _same(compiled(scale), reference(scale))
    assert any(decision.key.path == "diff" for decision in compiled.plan_for(scale).decisions)


def test_three_level_diff_fallback_uses_canonical_chunked_map(monkeypatch):
    """loop, map, loop under ``jax.grad`` keeps a budget ``ChunkedMap``.

    A key whose parent is already on the diff path is not fixed by flipping
    only that key's own path. The canonical twin holds the ``ChunkedMap``.
    An explicit decision stored on the exec key still wins. The grad jaxpr
    scans chunks.
    """
    outer = SiteKey(parent=None, path="primal", label="outer_loop", ordinal=0)
    canonical_map = SiteKey(parent=outer, path="primal", label="inner_map", ordinal=0)
    canonical_loop = SiteKey(parent=canonical_map, path="primal", label="inner_loop", ordinal=0)
    diff_map = SiteKey(parent=outer, path="diff", label="inner_map", ordinal=0)
    deep_primal = SiteKey(parent=diff_map, path="primal", label="inner_loop", ordinal=0)
    deep_diff = SiteKey(parent=diff_map, path="diff", label="inner_loop", ordinal=0)
    map_decision = SiteDecision(
        key=canonical_map,
        kind="map",
        primal=ChunkedMap(batch_size=2),
        diff=ChunkedMap(batch_size=2),
        primal_reason="budget",
        diff_reason="same strategy as primal in the MVP",
    )
    loop_decision = SiteDecision(
        key=canonical_loop,
        kind="loop",
        primal=Scan(),
        diff=Scan(),
        primal_reason="fixed length resolves to StaticWhile",
        diff_reason="diff path resolves to Scan",
    )
    session = Session(
        "exec", decisions={canonical_map: map_decision, canonical_loop: loop_decision}
    )
    assert decision_for(session, diff_map) is map_decision
    assert decision_for(session, deep_primal) is loop_decision
    assert decision_for(session, deep_diff) is loop_decision
    explicit = SiteDecision(
        key=deep_diff,
        kind="loop",
        primal=Scan(),
        diff=None,
        primal_reason="explicit",
        diff_reason="explicit",
    )
    explicit_session = Session(
        "exec",
        decisions={canonical_loop: loop_decision, deep_diff: explicit},
    )
    assert decision_for(explicit_session, deep_diff) is explicit

    calls = {"chunked": 0}
    map_module = sys.modules["xtrax.intent.map"]
    real_chunked = map_module.chunked_map

    def _spy_chunked(*args, **kwargs):
        calls["chunked"] += 1
        return real_chunked(*args, **kwargs)

    monkeypatch.setattr(map_module, "chunked_map", _spy_chunked)

    def fn(xs, weight):
        def outer_step(carry, row):
            def mid(z):
                def inner_step(inner, _):
                    y = jnp.sin(z * weight + inner)
                    return inner + 0.1 * y, y

                init = jnp.zeros_like(z)
                carry_n, ys = loop(inner_step, init, length=3, name="inner_loop")
                return carry_n + jnp.sum(ys)

            y = map(mid, row, name="inner_map")
            return carry + 0.1 * jnp.sum(y), y

        return loop(outer_step, jnp.float32(0.0), xs=xs, name="outer_loop")[0]

    def reference(xs, weight):
        def outer_step(carry, row):
            def mid(z):
                def inner_step(inner, _):
                    y = jnp.sin(z * weight + inner)
                    return inner + 0.1 * y, y

                init = jnp.zeros_like(z)
                carry_n, ys = jax.lax.scan(inner_step, init, None, length=3)
                return carry_n + jnp.sum(ys)

            y = jax.vmap(mid)(row)
            return carry + 0.1 * jnp.sum(y), y

        return jax.lax.scan(outer_step, jnp.float32(0.0), xs)[0]

    def _vmap_is_costly(decisions):
        return sum(100 if isinstance(decision.strategy, Vmap) else 10 for decision in decisions)

    xs = jnp.linspace(-0.2, 0.2, 32, dtype=jnp.float32).reshape(4, 8)
    weight = jnp.float32(0.3)
    compiled = compile(
        fn,
        Context(
            memory_budget=MemoryBudget(bytes=50, estimate=_vmap_is_costly),
            chunk_size=2,
            backend="cpu",
        ),
    )
    plan = compiled.plan_for(xs, weight)
    inner = next(
        decision
        for decision in plan.decisions
        if decision.key.label == "inner_map" and decision.key.path == "primal"
    )
    assert isinstance(inner.primal, ChunkedMap)
    assert inner.primal.batch_size == 2
    calls["chunked"] = 0
    _same(jax.grad(compiled)(xs, weight), jax.grad(reference)(xs, weight))
    assert calls["chunked"] >= 1
    backward = str(jax.make_jaxpr(jax.grad(compiled))(xs, weight))
    assert "scan[" in backward


def test_partial_diff_names_failed_argument_and_keeps_loop_scan():
    """One non-differentiable argument does not drop the loop's diff strategy.

    ``aux`` goes through ``pure_callback``. The loop diff stays Scan and names
    ``aux``. Gradients with respect to ``w`` match the scan reference. When
    every inexact argument fails, the loop diff stays ``None``. Fails while
    any failed argument sets ``diff_available`` false and the loop diff is None.
    """

    def fn(xs, w, aux):
        scale = jnp.sum(_host_scale(aux))

        def step(carry, row):
            y = map(lambda z: jnp.sin(z * w + carry), row, name="inner_map")
            return carry + 0.1 * jnp.sum(y), y

        carry, _ys = loop(step, jnp.float32(0.0), xs=xs, name="outer_loop")
        return carry * scale

    def reference(xs, w, aux):
        scale = jnp.sum(_host_scale(aux))

        def step(carry, row):
            y = jax.vmap(lambda z: jnp.sin(z * w + carry))(row)
            return carry + 0.1 * jnp.sum(y), y

        carry, _ys = jax.lax.scan(step, jnp.float32(0.0), xs)
        return carry * scale

    xs = jnp.linspace(-0.3, 0.3, 16, dtype=jnp.float32).reshape(4, 4)
    w = jnp.linspace(0.2, 0.5, 4, dtype=jnp.float32)
    aux = jnp.ones((3,), jnp.float32)
    compiled = compile(fn, Context(backend="cpu"))
    outer = next(
        decision
        for decision in compiled.plan_for(xs, w, aux).decisions
        if decision.key.label == "outer_loop" and decision.key.path == "primal"
    )
    assert isinstance(outer.diff, Scan)
    assert outer.diff_reason == (
        "diff path resolves to Scan; not differentiable w.r.t. aux (ValueError)"
    )
    assert "Pure callbacks" not in outer.diff_reason
    _same(jax.grad(compiled, argnums=1)(xs, w, aux), jax.grad(reference, argnums=1)(xs, w, aux))

    def only_aux(aux):
        scale = jnp.sum(_host_scale(aux))
        rows = jnp.ones((4, 4), jnp.float32)

        def step(carry, row):
            return carry + scale * jnp.sum(row), row

        return loop(step, jnp.float32(0.0), xs=rows, name="outer_loop")[0]

    only = compile(only_aux, Context(backend="cpu"))
    only_outer = next(
        decision
        for decision in only.plan_for(aux).decisions
        if decision.key.label == "outer_loop" and decision.key.path == "primal"
    )
    assert only_outer.diff is None
    assert "aux" in only_outer.diff_reason
    assert "ValueError" in only_outer.diff_reason


def test_posthoc_rejected_program_does_not_execute(monkeypatch):
    """A scan around a while is refused before the jitted program runs.

    The verdict is cached, so a second call raises the same error without
    running the body. Fails while the program runs and the callback fires
    before the check.
    """
    compile_module = sys.modules["xtrax.intent.compile"]
    monkeypatch.setattr(compile_module, "SCAN_AROUND_WHILE_BACKENDS", frozenset({"cpu"}))
    budget = MemoryBudget(bytes=50, estimate=_vmap_is_expensive)
    calls = {"n": 0}

    def fn(xs):
        def row(x):
            def body(carry, _):
                jax.debug.callback(lambda: calls.__setitem__("n", calls["n"] + 1))
                return carry + x, jnp.sum(carry)

            carry, ys = loop(body, jnp.zeros_like(x), length=3, name="inner")
            return carry + jnp.sum(ys)

        return map(row, xs, name="outer")

    xs = jnp.linspace(-0.2, 0.2, 32, dtype=jnp.float32).reshape(8, 4)
    blocked = compile(fn, Context(backend="cpu", memory_budget=budget, chunk_size=4))
    with pytest.raises(IntentError) as first:
        blocked(xs)
    assert first.value.constraint == "scan_around_while"
    assert not isinstance(first.value, UnsatisfiableContextError)
    assert calls["n"] == 0
    with pytest.raises(IntentError) as second:
        blocked(xs)
    assert second.value is first.value
    assert calls["n"] == 0


def test_posthoc_bypass_does_not_exempt_other_maps(monkeypatch):
    """An override bypass exempts that map only.

    The other map is still a scan around a while and is rejected. Fails while
    one bypass warning turns the check off for the whole program.
    """
    resolve_module = sys.modules["xtrax.intent.resolve"]
    compile_module = sys.modules["xtrax.intent.compile"]
    monkeypatch.setattr(compile_module, "SCAN_AROUND_WHILE_BACKENDS", frozenset({"cpu"}))
    monkeypatch.setattr(resolve_module, "SCAN_AROUND_WHILE_BACKENDS", frozenset({"cpu"}))
    budget = MemoryBudget(bytes=50, estimate=_vmap_is_expensive)

    def fn(xs):
        def static_row(x):
            def body(carry, _):
                return carry + x, jnp.sum(carry)

            carry, ys = loop(body, jnp.zeros_like(x), length=3, name="static_inner")
            return carry + jnp.sum(ys)

        def cond_row(x):
            def body(carry):
                return carry + jnp.float32(0.5)

            def cond(carry):
                return jnp.sum(carry) < jnp.float32(1)

            return loop(body, x, cond=cond, name="cond_inner")

        left = map(static_row, xs, name="left")
        right = map(cond_row, xs, name="right")
        return left + right

    xs = jnp.linspace(-0.2, 0.2, 32, dtype=jnp.float32).reshape(8, 4)
    compiled = compile(
        fn,
        Context(backend="cpu", memory_budget=budget, chunk_size=4),
        overrides={"left": (ChunkedMap(batch_size=4), ChunkedMap(batch_size=4))},
    )
    plan = compiled.plan_for(xs)
    assert any(
        warning == "override:left bypassed scan_around_while"
        for decision in plan.decisions
        for warning in decision.warnings
    )
    right = next(
        decision
        for decision in plan.decisions
        if decision.key.label == "right" and decision.key.path == "primal"
    )
    assert not any("bypassed scan_around_while" in warning for warning in right.warnings)
    with pytest.raises(IntentError) as exc_info:
        compiled(xs)
    assert exc_info.value.constraint == "scan_around_while"
    assert not isinstance(exc_info.value, UnsatisfiableContextError)
