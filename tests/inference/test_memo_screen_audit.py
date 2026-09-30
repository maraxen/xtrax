"""#5234 primitive-list audit and #5233 execute_screened for memoize_jaxpr."""

from __future__ import annotations

import gc

import jax
import jax.experimental
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
    # Raw uint32 key: a typed `jax.random.key` ARGUMENT currently fails leaf
    # classification with a TypeError (separate item); both emit the same primitives.
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


def test_execute_screened_static_mode_signature() -> None:
    """A scalar that forces the STATIC retrace still resolves to a runner for that token."""

    def f(x, n):  # noqa: ANN001, ANN202
        return x[:n] * 2  # slicing needs a concrete n -> ABSTRACT fails, STATIC succeeds

    m = memoize_jaxpr(f, policy=MemoPolicy(execute_screened=True))
    np.testing.assert_array_equal(np.asarray(m(jnp.arange(5.0), 2)), [0.0, 2.0])
    np.testing.assert_array_equal(np.asarray(m(jnp.arange(5.0), 3)), [0.0, 2.0, 4.0])


def test_execute_screened_miss_does_not_alias_the_argument() -> None:
    x = jnp.ones(3)
    m = memoize_jaxpr(lambda a: a, policy=MemoPolicy(execute_screened=True))
    assert m(x) is not x


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
