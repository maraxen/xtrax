"""#5234 primitive-list audit and #5233 execute_screened for memoize_jaxpr."""

from __future__ import annotations

import gc

import jax
import jax.experimental
import jax.extend.random
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.inference import MemoPolicy, memo, memoize_jaxpr
from xtrax.inference.errors import MemoImpurityError, MemoStalenessError


def _registered_primitive_names() -> set[str]:
    # Primitives register on module import; exercise the families the lists cover so
    # every lazily created one exists before the scan.
    jax.make_jaxpr(lambda k: jax.random.normal(k, (2,)))(jax.random.key(0))
    jax.make_jaxpr(lambda k: jax.random.gamma(k, 1.0))(jax.random.key(0))
    from jax._src.core import Primitive

    return {o.name for o in gc.get_objects() if isinstance(o, Primitive)}


# --- #5234 ----------------------------------------------------------------------------


def test_every_listed_primitive_is_registered_by_this_jax() -> None:
    """A banned or admitted name JAX no longer registers screens nothing: fail loudly.

    On 2026-09-22 five of eleven listed names (e.g. `threefry2x32_p`) were registered by
    neither supported JAX, so the stateful list banned nothing at all.
    """
    registered = _registered_primitive_names()
    listed = memo._BANNED_PRIMITIVES | memo._ADMITTED_KEY_PRIMITIVES
    missing = sorted(listed - registered)
    assert missing == [], f"not registered by jax {jax.__version__}: {missing}"


def test_banned_and_admitted_do_not_overlap() -> None:
    assert not (memo._BANNED_PRIMITIVES & memo._ADMITTED_KEY_PRIMITIVES)


def test_every_registered_draw_or_callback_primitive_is_classified() -> None:
    """Completeness, the direction the existence test cannot see: a draw-like or
    callback-like primitive this JAX registers must be banned or admitted by decision,
    never silently unclassified (philox2x32/4x32 were, on first review)."""
    import re

    family = re.compile(r"random|rng|threefry|philox|callback|debug")
    registered = {n for n in _registered_primitive_names() if family.search(n)}
    unclassified = sorted(registered - memo._BANNED_PRIMITIVES - memo._ADMITTED_KEY_PRIMITIVES)
    assert unclassified == [], f"classify these (jax {jax.__version__}): {unclassified}"


def test_registry_scan_can_fail() -> None:
    """Negative control: the scan distinguishes a registered name from a dead one."""
    registered = _registered_primitive_names()
    assert "random_bits" in registered
    assert "threefry2x32_p" not in registered


@pytest.mark.parametrize(
    ("label", "fn"),
    [
        ("debug_print", lambda x: (jax.debug.print("{}", x), x * 2)[1]),
        ("debug_callback", lambda x: (jax.debug.callback(lambda _: None, x), x * 2)[1]),
    ],
)
def test_debug_side_effects_are_rejected(label: str, fn) -> None:  # noqa: ANN001
    with pytest.raises(MemoImpurityError, match=label):
        memoize_jaxpr(fn)(jnp.ones(3))


@pytest.mark.parametrize(
    "fn",
    [
        lambda k: jax.random.split(k),
        lambda k: jax.random.fold_in(k, 3),
    ],
    ids=["split", "fold_in"],
)
def test_key_plumbing_on_a_key_argument_is_admitted(fn) -> None:  # noqa: ANN001
    # Raw uint32 key; typed keys are covered by the #5679 tests below.
    key = jax.random.PRNGKey(0)
    m = memoize_jaxpr(fn)
    a = m(key)
    b = m(key)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(fn(key)))
    assert b is a  # second call is a hit


def test_a_draw_on_a_key_argument_is_still_rejected() -> None:
    """Control for the admission above: drawing, not key plumbing, is what is banned."""
    with pytest.raises(MemoImpurityError, match="random_bits"):
        memoize_jaxpr(lambda k: jax.random.normal(k, (2,)))(jax.random.PRNGKey(0))


# --- #5679: typed PRNG keys (jax.random.key) ------------------------------------------

# Every impl the installed JAX ships (jax.extend.random has no public registry).
_KEY_IMPLS = [
    getattr(jax.extend.random, name).name
    for name in ("threefry_prng_impl", "rbg_prng_impl", "unsafe_rbg_prng_impl")
]


@pytest.mark.parametrize("impl", _KEY_IMPLS)
def test_typed_key_argument_is_memoized(impl: str) -> None:
    """Used to raise `TypeError: Cannot interpret 'key<fry>' as a data type`."""
    split = memoize_jaxpr(lambda k: jax.random.split(k))
    key = jax.random.key(0, impl=impl)
    a = split(key)
    assert split(key) is a  # hit
    np.testing.assert_array_equal(
        jax.random.key_data(a), jax.random.key_data(jax.random.split(key))
    )
    other = split(jax.random.key(1, impl=impl))  # different bits: a miss, not an alias
    assert split.memo_get_stats()["misses"] == 2
    assert not np.array_equal(jax.random.key_data(other), jax.random.key_data(a))


def test_same_key_bits_under_two_impls_do_not_share_an_entry() -> None:
    bits = jnp.arange(4, dtype=jnp.uint32)  # rbg and unsafe_rbg both hold 4 x uint32
    rbg = jax.random.wrap_key_data(bits, impl="rbg")
    urbg = jax.random.wrap_key_data(bits, impl="unsafe_rbg")
    split = memoize_jaxpr(lambda k: jax.random.split(k))
    a, b = split(rbg), split(urbg)
    assert split.memo_get_stats()["misses"] == 2
    assert jax.random.key_impl(a) != jax.random.key_impl(b)


def test_typed_key_output_passes_its_own_spot_check() -> None:
    split = memoize_jaxpr(lambda k: jax.random.split(k), policy=MemoPolicy(spot_check_every=1))
    key = jax.random.key(0)
    split(key)
    split(key)  # a spot-checked hit: equal keys must not read as staleness
    assert split.memo_get_stats()["spot_check_mismatches"] == 0


def test_closed_over_typed_key_is_digested() -> None:
    base = jax.random.key(7)
    m = memoize_jaxpr(lambda i: jax.random.fold_in(base, i))
    np.testing.assert_array_equal(
        jax.random.key_data(m(jnp.int32(3))), jax.random.key_data(jax.random.fold_in(base, 3))
    )


def test_a_draw_on_a_typed_key_argument_is_rejected_by_the_screen() -> None:
    """Control: the typed key now reaches the screen, which still bans the draw."""
    with pytest.raises(MemoImpurityError, match="random_bits"):
        memoize_jaxpr(lambda k: jax.random.normal(k, (2,)))(jax.random.key(0))


def test_captured_mutable_ref_is_rejected_by_the_screen() -> None:
    """Previously crashed in _program_digest with 'Out of bound indexer'."""
    ref = jax.new_ref(jnp.zeros(3))

    def bump(x):  # noqa: ANN001, ANN202
        ref[...] = ref[...] + x
        return ref[...]

    with pytest.raises(MemoImpurityError, match=r"mutable jax\.Ref"):
        memoize_jaxpr(bump)(jnp.ones(3))


def test_locally_allocated_ref_is_admitted() -> None:
    """Control: a ref allocated inside the function is pure and must not be rejected."""

    def double(x):  # noqa: ANN001, ANN202
        r = jax.new_ref(x)
        r[...] = r[...] * 2
        return r[...]

    np.testing.assert_array_equal(np.asarray(memoize_jaxpr(double)(jnp.ones(3))), 2.0)


# --- #5233 ----------------------------------------------------------------------------


def _divergent(x):  # noqa: ANN001, ANN202
    """Traces one path, runs another eagerly: the N5 idiom."""
    if isinstance(x, jax.core.Tracer):
        return x * 0.0
    return x + 1.0


def test_default_policy_still_runs_fn_eagerly_on_a_miss() -> None:
    """Pins the documented default (blind spot N5): the eager path is what gets cached."""
    m = memoize_jaxpr(_divergent)
    np.testing.assert_array_equal(np.asarray(m(jnp.ones(2))), 2.0)


def test_execute_screened_runs_the_screened_program() -> None:
    m = memoize_jaxpr(_divergent, policy=MemoPolicy(execute_screened=True))
    first = m(jnp.ones(2))
    second = m(jnp.ones(2))
    np.testing.assert_array_equal(np.asarray(first), 0.0)  # the traced path
    assert second is first
    assert m.memo_get_stats()["misses"] == 1


def test_execute_screened_preserves_output_pytree_structure() -> None:
    def f(x, *, scale):  # noqa: ANN001, ANN202
        return {"y": x * scale, "pair": (x, x + 1)}

    m = memoize_jaxpr(f, policy=MemoPolicy(execute_screened=True))
    out = m(jnp.arange(3.0), scale=2.0)
    ref = f(jnp.arange(3.0), scale=2.0)
    assert jax.tree.structure(out) == jax.tree.structure(ref)
    for got, want in zip(jax.tree.leaves(out), jax.tree.leaves(ref)):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_execute_screened_static_mode_signature(monkeypatch) -> None:  # noqa: ANN001
    """A scalar that forces the STATIC retrace resolves to a runner stored under the
    token call() looks up: repeats of a signature never re-trace or grow the table."""

    def f(x, n):  # noqa: ANN001, ANN202
        return x[:n] * 2  # slicing needs a concrete n -> ABSTRACT fails, STATIC succeeds

    m = memoize_jaxpr(f, policy=MemoPolicy(execute_screened=True))
    core = m._memo_core
    np.testing.assert_array_equal(np.asarray(m(jnp.arange(5.0), 2)), [0.0, 2.0])
    np.testing.assert_array_equal(np.asarray(m(jnp.arange(5.0), 3)), [0.0, 2.0, 4.0])
    assert set(core._runners) <= set(core._screened)
    n_runners = len(core._runners)

    traces = {"n": 0}
    real = memo._trace_closed

    def counting(*a, **k):  # noqa: ANN002, ANN003, ANN202
        traces["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(memo, "_trace_closed", counting)
    # Same static n, new array values: a cache MISS on an already-screened signature.
    np.testing.assert_array_equal(np.asarray(m(jnp.arange(5.0) + 1, 2)), [2.0, 4.0])
    assert traces["n"] == 0
    assert len(core._runners) == n_runners


def test_execute_screened_miss_does_not_alias_the_argument() -> None:
    x = jnp.ones(3)
    m = memoize_jaxpr(lambda a: a, policy=MemoPolicy(execute_screened=True))
    y = m(x)
    assert y.unsafe_buffer_pointer() != x.unsafe_buffer_pointer()


def test_spot_check_under_execute_screened_recomputes_the_screened_program() -> None:
    """Spot-checking against eager fn would poison the wrapper on the very divergence
    the caller opted out of; it re-runs the screened program instead."""
    m = memoize_jaxpr(_divergent, policy=MemoPolicy(execute_screened=True, spot_check_every=2))
    x = _f32(1.0, 1.0)
    for _ in range(6):  # includes spot-checked hits
        np.testing.assert_array_equal(np.asarray(m(x)), 0.0)
    assert m.memo_get_stats()["spot_check_mismatches"] == 0


def test_stored_runner_for_another_digest_is_not_used() -> None:
    """A runner under this token whose digest differs from the resolved one is refused
    and rebuilt through the digest-checked path."""
    m = memoize_jaxpr(_divergent, policy=MemoPolicy(execute_screened=True))
    x = _f32(1.0, 1.0)
    m(x)
    core = m._memo_core
    (token,) = core._runners
    stale = core._runners[token]
    core._runners[token] = memo._ScreenedRunner(
        digest="not-the-screened-digest",
        traced_positions=stale.traced_positions,
        out_tree=stale.out_tree,
        compiled=lambda *a: [a[0] + 100.0],  # would be visibly wrong if used
    )
    np.testing.assert_array_equal(np.asarray(m(_f32(2.0, 2.0))), 0.0)
    assert core._runners[token].digest == stale.digest


def _f32(*values: float) -> jax.Array:
    # Identical construction for every call, so all share ONE signature token (a
    # weak-typed array would be a different signature and bypass the rebuild path).
    return jnp.asarray(np.asarray(values, dtype=np.float32))


def test_runner_is_rebuilt_after_it_is_dropped() -> None:
    m = memoize_jaxpr(_divergent, policy=MemoPolicy(execute_screened=True))
    m(_f32(1.0, 1.0))
    core = m._memo_core
    core._runners.clear()  # simulate a concurrent eviction of the runner only
    np.testing.assert_array_equal(np.asarray(m(_f32(5.0, 5.0))), 0.0)
    assert len(core._screened) == 1  # same signature: the rebuild path, not a re-screen
    assert len(core._runners) == 1


def test_rebuild_refuses_a_program_that_changed_since_screening() -> None:
    calls = {"n": 0}

    def drifting(x):  # noqa: ANN001, ANN202
        calls["n"] += 1
        return x * float(calls["n"])  # a different constant on every trace

    m = memoize_jaxpr(drifting, policy=MemoPolicy(execute_screened=True))
    m(_f32(1.0, 1.0))
    m._memo_core._runners.clear()
    with pytest.raises(MemoStalenessError, match="different program"):
        m(_f32(3.0, 3.0))
    assert len(m._memo_core._screened) == 1
