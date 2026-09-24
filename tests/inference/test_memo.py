"""Tests for xtrax.inference.memo — spec 260825 §4.2 (AC3-AC6, AC9, AC12-AC20)
and spec 260922 §3 donation, both directions (AC-1 to AC-17)."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.inference import memo
from xtrax.inference.memo import (
    MemoDonationError,
    MemoImpurityError,
    MemoKeyUnsupportedLeafError,
    MemoMultiDeviceError,
    MemoPolicy,
    MemoStalenessError,
    memoize_jaxpr,
)


def _tracer_aware_spy():
    """Execution counter that ignores tracing passes (AC4 semantics)."""
    calls = {"concrete": 0}

    def fn(x):
        if not isinstance(x, jax.core.Tracer):
            calls["concrete"] += 1
        return jnp.sin(x) * 2.0

    return fn, calls


class TestAdmission:
    def test_ac3_unkeyed_randomness_rejected(self):
        def impure():
            return jax.random.uniform(jax.random.key(0), (4,))

        # Zero-arg callables screen at wrap time (AC3: never concretely run).
        with pytest.raises(MemoImpurityError):
            memoize_jaxpr(impure)

    def test_ac3_deferred_screen_for_arg_fns(self):
        """Arg-taking impure fns screen at FIRST CALL (deferred path)."""

        def impure(x):
            key = jax.random.key(0)
            return jax.random.uniform(key, x.shape) + x

        wrapped = memoize_jaxpr(impure)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoImpurityError):
            wrapped(x)

    def test_screen_latches_after_first_error(self):
        """AC20: zero-arg impure fn raises at wrap; deferred path latches."""

        def make_impure():
            def impure(x):
                return jax.random.uniform(jax.random.key(0), (4,))

            return impure

        wrapped = memoize_jaxpr(make_impure())
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoImpurityError):
            wrapped(x)  # first call: screen runs, raises, latches
        assert wrapped._memo_core.screen_latched_error is not None
        with pytest.raises(MemoImpurityError):
            wrapped(x)  # immediate, latched — no re-trace
        wrapped.memo_rewrap()
        assert wrapped._memo_core.screen_latched_error is None

    def test_multi_device_rejected(self, monkeypatch):
        """AC15/N5: wrap raises MemoMultiDeviceError when >1 local device."""
        import xtrax.inference.memo as m

        def fake_local_devices():
            return [
                type("D", (), {"id": 0, "device_kind": "cpu"})(),
                type("D", (), {"id": 1, "device_kind": "cpu"})(),
            ]

        monkeypatch.setattr(m.jax, "local_devices", fake_local_devices)
        with pytest.raises(MemoMultiDeviceError):
            memoize_jaxpr(lambda x: x + 1)

    def test_stamp_override_requires_env(self, monkeypatch):
        monkeypatch.delenv("XTRAX_MEMO_STAMP_OVERRIDE", raising=False)
        with pytest.raises(ValueError):
            MemoPolicy(_stamp_override="fake-stamp")

    def test_stamp_override_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("XTRAX_MEMO_STAMP_OVERRIDE", "1")
        p = MemoPolicy(_stamp_override="test-stamp")
        assert p._stamp_override == "test-stamp"


class TestPurityWalk:
    """AC-9, AC-10, AC-11: purity screen walks every sub-jaxpr with no depth cap."""

    def test_ac9_cond_branch_random_rejected(self):
        """AC-9: random draw inside lax.cond branch raises MemoImpurityError."""

        def f(p, x):
            return lax.cond(
                p,
                lambda y: y + jax.random.uniform(jax.random.key(0), y.shape),
                lambda y: y,
                x,
            )

        wrapped = memoize_jaxpr(f)
        with pytest.raises(MemoImpurityError, match=r"branches\["):
            wrapped(jnp.array(True), jnp.ones((4,), jnp.float32))

    def test_ac10_deeply_nested_jit_random_rejected(self):
        """AC-10: random draw nested inside 10 levels of jax.jit raises MemoImpurityError.

        Verifies that the screen traversal has no depth cap, and that the nesting
        is real (each level is a distinct wrapper).
        """

        def base(y):
            return y + jax.random.uniform(jax.random.key(0), y.shape)

        # Build 10 levels of jit, each explicitly wrapping the previous
        g = base
        for _ in range(10):

            def wrap(inner):
                return jax.jit(lambda y: inner(y) * 1.0)

            g = wrap(g)

        # Verify nesting is real: check that jax.make_jaxpr shows depth >= 9
        x = jnp.ones((4,), jnp.float32)
        jaxpr = jax.make_jaxpr(g)(x)

        # Count nesting depth by walking the params
        def count_nesting_depth(jaxpr_obj, depth=0):
            max_depth = depth
            for eqn in jaxpr_obj.eqns:
                for param_val in eqn.params.values():
                    if hasattr(param_val, "eqns"):
                        max_depth = max(max_depth, count_nesting_depth(param_val, depth + 1))
            return max_depth

        nesting = count_nesting_depth(jaxpr.jaxpr)
        assert nesting >= 9, f"Expected nesting >= 9, got {nesting}"

        wrapped = memoize_jaxpr(g)
        with pytest.raises(MemoImpurityError):
            wrapped(x)

    def test_ac11_while_loop_random_rejected(self):
        """AC-11: random draw inside lax.while_loop body raises MemoImpurityError."""

        def f(x):
            def cond_fn(carry):
                return carry < 5

            def body_fn(carry):
                return carry + 1 + jax.random.uniform(jax.random.key(0), ())

            return lax.while_loop(cond_fn, body_fn, x)

        wrapped = memoize_jaxpr(f)
        with pytest.raises(MemoImpurityError):
            wrapped(jnp.array(0.0))

    def test_ac11_scan_random_rejected(self):
        """AC-11: random draw inside lax.scan body raises MemoImpurityError."""

        def f(x):
            def body_fn(carry, inp):
                return carry + inp + jax.random.uniform(jax.random.key(0), ()), None

            return lax.scan(body_fn, jnp.array(0.0), x)[0]

        wrapped = memoize_jaxpr(f)
        with pytest.raises(MemoImpurityError):
            wrapped(jnp.ones((4,), jnp.float32))


class TestCaching:
    def test_ac4_cache_hit_no_reexecution(self):
        fn, calls = _tracer_aware_spy()
        f = memoize_jaxpr(fn)
        x = jnp.ones((4,), jnp.float32)
        f(x)
        f(x)
        s = f.memo_get_stats()
        assert calls["concrete"] == 1 and s["hits"] == 2 - 1 or True
        # Strict:
        assert calls["concrete"] == 1
        assert s["hits"] == 1 and s["misses"] == 1

    def test_ac5_salt_isolates_entries(self):
        fn_a, calls_a = _tracer_aware_spy()
        fn_b, calls_b = _tracer_aware_spy()
        fa = memoize_jaxpr(fn_a, policy=MemoPolicy(salt="a"))
        fb = memoize_jaxpr(fn_b, policy=MemoPolicy(salt="b"))
        x = jnp.ones((4,), jnp.float32)
        fa(x)
        fa(x)
        fb(x)
        fb(x)
        assert calls_a["concrete"] == 1 and calls_b["concrete"] == 1
        assert fa.memo_get_stats()["hits"] == 1
        assert fb.memo_get_stats()["hits"] == 1

    def test_ac13_cross_stamp_isolation(self, monkeypatch):
        monkeypatch.setenv("XTRAX_MEMO_STAMP_OVERRIDE", "1")
        fn_a, calls_a = _tracer_aware_spy()
        f1 = memoize_jaxpr(fn_a, policy=MemoPolicy(_stamp_override="stamp-1"))
        x = jnp.ones((4,), jnp.float32)
        f1(x)
        f1(x)
        assert calls_a["concrete"] == 1

        # Same underlying behavior but different injected stamp: separate wrapper
        # instance => separate cache anyway. The isolation guarantee under test is
        # that two wrappers with different stamps never share entries even when
        # pointed at one shared cache — approximated here by key comparison.
        f1(x)  # ensures program digest + stamp materialized
        f2 = memoize_jaxpr(fn_a, policy=MemoPolicy(_stamp_override="stamp-2"))
        f2(x)
        # Get digest by calling _ensure_screened
        digest1 = f1._memo_core._ensure_screened((x,), {})
        digest2 = f2._memo_core._ensure_screened((x,), {})
        k1 = f1._memo_core.build_key(digest1, (x,), {})
        k2 = f2._memo_core.build_key(digest2, (x,), {})
        assert k1 != k2

    def test_ac17_python_float_value_discriminates(self):
        calls = {"n": 0}

        def fn(x, scale):
            if not isinstance(x, jax.core.Tracer):
                calls["n"] += 1
            return x * scale

        f = memoize_jaxpr(fn)
        x = jnp.ones((4,), jnp.float32)
        f(x, 2.0)
        f(x, 3.0)
        assert calls["n"] == 2  # different float args must miss

    def test_unsupported_leaf_type_rejected(self):
        def fn(x, weird):
            return x * 2.0

        f = memoize_jaxpr(fn)
        with pytest.raises(MemoKeyUnsupportedLeafError):
            f(jnp.ones((2,), jnp.float32), {"set": {1, 2}})

    def test_eviction_respects_max_entries(self):
        fn, calls = _tracer_aware_spy()
        f = memoize_jaxpr(fn, policy=MemoPolicy(max_entries=2))
        for i in range(5):
            f(jnp.full((4,), float(i)))
        s = f.memo_get_stats()
        assert s["evictions"] >= 3 and len(f._memo_core.cache) <= 2


class TestSpotCheck:
    def test_ac6_staleness_detected_via_closure_mutation(self):
        config = {"scale": 2.0}

        concrete_calls = {"n": 0}

        def score(x):
            if not isinstance(x, jax.core.Tracer):
                concrete_calls["n"] += 1
            return x * config["scale"]

        f = memoize_jaxpr(score, policy=MemoPolicy(spot_check_every=2))
        x = jnp.ones((4,), jnp.float32)

        f(x)  # call 1: miss, caches result with scale=2
        config["scale"] = 9.0  # invisible-to-tracing state mutation
        with pytest.raises(MemoStalenessError):
            f(x)  # call 2: hit + spot-check recomputes -> mismatch
        assert f.memo_get_stats()["spot_check_mismatches"] == 1

    def test_poisoned_counter_until_reset(self):
        config = {"scale": 2.0}

        def score(x):
            return x * config["scale"]

        f = memoize_jaxpr(score, policy=MemoPolicy(spot_check_every=1))
        x = jnp.ones((4,), jnp.float32)
        f(x)
        config["scale"] = 5.0
        with pytest.raises(MemoStalenessError):
            f(x)
        # Poisoned: every subsequent call raises immediately.
        with pytest.raises(MemoStalenessError):
            f(x)
        f.memo_reset()
        # After reset the counter clears; entry was evicted so this recomputes.
        f(x)  # no raise

    def test_ac14_forced_corruption_triggers_error(self):
        def score(x):
            return x * 2.0

        f = memoize_jaxpr(score, policy=MemoPolicy(spot_check_every=1))
        x = jnp.ones((4,), jnp.float32)
        f(x)
        core = f._memo_core
        # Operational definition (spec AC14): swap a private LRU entry value.
        key = next(iter(core.cache))
        entry = core.cache[key]
        entry.value = jnp.zeros_like(entry.value)  # corrupt
        with pytest.raises(MemoStalenessError):
            f(x)


class TestAsyncAndDonation:
    def test_ac9_blocking_mode_stores_ready_buffers(self):
        def score(x):
            return jnp.sin(x) * 2.0

        f = memoize_jaxpr(score)
        x = jnp.ones((4,), jnp.float32)
        f(x)
        core = f._memo_core
        entry = next(iter(core.cache.values()))
        assert entry.ready is True
        y = f(x)  # synchronous second call returns allclose-equal values
        assert bool(jnp.allclose(y, jnp.sin(x) * 2.0))


class TestDonation:
    """T1: memoize_jaxpr donation, both directions (spec §3, AC-1 to AC-14)."""

    # -- AC-1 -----------------------------------------------------------

    def test_ac1_top_level_input_donation_rejected(self):
        from xtrax.inference.memo import MemoDonationError

        def f(x, y):
            return x * 2 + y

        wrapped = memoize_jaxpr(jax.jit(f, donate_argnums=0))
        x = jnp.ones((4,), jnp.float32)
        y = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoDonationError) as exc_info:
            wrapped(x, y)
        assert isinstance(exc_info.value, MemoImpurityError)
        assert wrapped.memo_get_stats()["misses"] == 0
        assert x.is_deleted() is False

        # Negative control: no donation -> admitted normally.
        def g(x, y):
            return x * 2 + y

        wrapped_ok = memoize_jaxpr(jax.jit(g))
        x2 = jnp.ones((4,), jnp.float32)
        y2 = jnp.ones((4,), jnp.float32)
        result = wrapped_ok(x2, y2)
        assert wrapped_ok.memo_get_stats()["misses"] == 1
        assert bool(jnp.allclose(result, x2 * 2 + y2))

    # -- AC-2 -----------------------------------------------------------

    def test_ac2_wrapped_input_leaf_indices(self):
        from xtrax.inference.memo import MemoDonationError

        def f(p, y):
            a, b = p
            return a + b + y

        wrapped_a = memoize_jaxpr(jax.jit(f, donate_argnums=1))
        with pytest.raises(MemoDonationError) as exc_a:
            wrapped_a((jnp.ones((4,)), jnp.ones((4,))), jnp.ones((4,)))
        assert exc_a.value.sites[0][3] == (2,)

        wrapped_b = memoize_jaxpr(jax.jit(f, donate_argnames="y"))
        with pytest.raises(MemoDonationError) as exc_b:
            wrapped_b((jnp.ones((4,)), jnp.ones((4,))), jnp.ones((4,)))
        assert exc_b.value.sites[0][3] == (2,)

        def g(d):
            return d["a"] + d["b"]

        wrapped_c = memoize_jaxpr(jax.jit(g, donate_argnums=0))
        with pytest.raises(MemoDonationError) as exc_c:
            wrapped_c({"a": jnp.ones((4,)), "b": jnp.ones((4,))})
        assert exc_c.value.sites[0][3] == (0, 1)

    # -- AC-3 -----------------------------------------------------------

    def test_ac3_eager_wrapper_around_donating_jit(self):
        """The spy lives on `outer` (the memoized function) itself, not on
        `f` (wrapped by the inner jit): a jit-wrapped body only ever traces
        with abstract tracers — it never sees a concrete arg, whether or not
        admission succeeds — so it cannot distinguish rejection from
        admission. `outer`'s own body does: it is traced (tracer args) during
        screening, but called EAGERLY (concrete args) on a real miss.
        """
        from xtrax.inference.memo import MemoDonationError

        calls = {"concrete": 0}

        def f(x, y):
            return x * 2 + y

        def outer(x, y):
            if not isinstance(x, jax.core.Tracer):
                calls["concrete"] += 1
            return jax.jit(f, donate_argnums=1)(x, y) + 1.0

        wrapped = memoize_jaxpr(outer)
        x = jnp.ones((4,), jnp.float32)
        y = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoDonationError) as exc_info:
            wrapped(x, y)
        assert calls["concrete"] == 0
        assert y.is_deleted() is False
        assert "jit" in exc_info.value.sites[0][0]

        # Negative control: no donation -> admitted; outer runs concretely once.
        def outer_ok(x, y):
            if not isinstance(x, jax.core.Tracer):
                calls["concrete"] += 1
            return jax.jit(f)(x, y) + 1.0

        wrapped_ok = memoize_jaxpr(outer_ok)
        wrapped_ok(x, y)
        assert calls["concrete"] == 1

    # -- AC-4 -----------------------------------------------------------

    def test_ac4_donation_inside_cond_branch_rejected(self):
        from xtrax.inference.memo import MemoDonationError

        def branch_true(x):
            return jax.jit(lambda y: y * 2, donate_argnums=0)(x)

        def branch_false(x):
            return x + 1.0

        def f(pred, x):
            return jax.lax.cond(pred, branch_true, branch_false, x)

        wrapped = memoize_jaxpr(f)
        with pytest.raises(MemoDonationError) as exc_info:
            wrapped(True, jnp.ones((4,), jnp.float32))
        assert any("branches[" in site[0] for site in exc_info.value.sites)

        # Negative control: no donation in either branch -> admitted.
        def branch_true_ok(x):
            return x * 2.0

        def f_ok(pred, x):
            return jax.lax.cond(pred, branch_true_ok, branch_false, x)

        wrapped_ok = memoize_jaxpr(f_ok)
        result = wrapped_ok(True, jnp.ones((4,), jnp.float32))
        assert bool(jnp.allclose(result, jnp.ones((4,)) * 2.0))

    def test_ac4_naive_getattr_walker_misses_cond_branches(self, monkeypatch):
        """Red control (#5216): a walker using getattr(param, "eqns") alone
        (the OLD _screen_jaxpr strategy) never descends into the tuple-valued
        `branches` param, so it misses the donation entirely. Demonstrates that
        recursing into tuple/list values (_iter_subjaxprs) is load-bearing for AC-4.
        """
        import xtrax.inference.memo as m

        def naive_iter_subjaxprs(value, path=""):
            sub_eqns = getattr(value, "eqns", None)
            if sub_eqns:
                yield path, value

        monkeypatch.setattr(m, "_iter_subjaxprs", naive_iter_subjaxprs)

        def branch_true(x):
            return jax.jit(lambda y: y * 2, donate_argnums=0)(x)

        def branch_false(x):
            return x + 1.0

        def f(pred, x):
            return jax.lax.cond(pred, branch_true, branch_false, x)

        wrapped = memoize_jaxpr(f)
        # Under the naive walker, donation inside branches[1] is missed:
        # admission wrongly succeeds (this is the bug the real walker fixes).
        result = wrapped(True, jnp.ones((4,), jnp.float32))
        assert result is not None

    # -- AC-5 -----------------------------------------------------------

    def test_ac5_donation_in_scan_body_and_deep_nesting_rejected(self):
        from xtrax.inference.memo import MemoDonationError

        def body(carry, x):
            y = jax.jit(lambda z: z * 2, donate_argnums=0)(x)
            return carry + y, y

        def f_scan(init, xs):
            return jax.lax.scan(body, init, xs)

        wrapped_scan = memoize_jaxpr(f_scan)
        with pytest.raises(MemoDonationError):
            wrapped_scan(jnp.float32(0.0), jnp.ones((4,), jnp.float32))

        # 10 nested jit levels deep; the innermost donates.
        def make_nested(n):
            if n == 0:
                return lambda x: jax.jit(lambda z: z * 2, donate_argnums=0)(x)
            inner = make_nested(n - 1)
            return lambda x: jax.jit(inner)(x)

        deep_fn = make_nested(9)  # 10 jit levels total
        wrapped_deep = memoize_jaxpr(deep_fn)
        with pytest.raises(MemoDonationError):
            wrapped_deep(jnp.ones((4,), jnp.float32))

        # Negative control: no donation anywhere -> admitted.
        def body_ok(carry, x):
            y = x * 2
            return carry + y, y

        def f_scan_ok(init, xs):
            return jax.lax.scan(body_ok, init, xs)

        wrapped_scan_ok = memoize_jaxpr(f_scan_ok)
        wrapped_scan_ok(jnp.float32(0.0), jnp.ones((4,), jnp.float32))
        assert wrapped_scan_ok.memo_get_stats()["misses"] == 1

    def test_ac5_capped_walker_misses_deep_nesting(self, monkeypatch):
        """Red control: capping how many nested-jaxpr levels the walker will
        still explore (<= 9) fails to reach the 10th-level donation."""
        import xtrax.inference.memo as m

        real_iter = m._iter_subjaxprs
        state = {"calls": 0}

        def capped_iter(value, path=""):
            state["calls"] += 1
            if state["calls"] > 9:
                return
            yield from real_iter(value, path)

        monkeypatch.setattr(m, "_iter_subjaxprs", capped_iter)

        def make_nested(n):
            if n == 0:
                return lambda x: jax.jit(lambda z: z * 2, donate_argnums=0)(x)
            inner = make_nested(n - 1)
            return lambda x: jax.jit(inner)(x)

        deep_fn = make_nested(9)
        wrapped_deep = memoize_jaxpr(deep_fn)
        # Depth-capped walker fails to see the deepest donation -> wrongly
        # admits (this is the bug the uncapped real walker fixes).
        result = wrapped_deep(jnp.ones((4,), jnp.float32))
        assert result is not None

    # -- AC-6 -----------------------------------------------------------

    def test_ac6_device_put_donate_rejected(self):
        from xtrax.inference.memo import MemoDonationError

        def outer(x):
            return jax.device_put(x, donate=True) * 2.0

        wrapped = memoize_jaxpr(outer)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoDonationError) as exc_info:
            wrapped(x)
        assert exc_info.value.sites[0][1] == "copy_semantics"

        # Negative control: device_put without donate -> admitted.
        def outer_ok(x):
            return jax.device_put(x) * 2.0

        wrapped_ok = memoize_jaxpr(outer_ok)
        x2 = jnp.ones((4,), jnp.float32)
        result = wrapped_ok(x2)
        assert bool(jnp.allclose(result, x2 * 2.0))

    # -- AC-7 -----------------------------------------------------------

    def test_intermediate_donation_rejected_by_design(self):
        """§3.3(b): intermediate donation is rejected (over-rejection, by
        design) even though the donated buffer is invisible to the caller."""
        from xtrax.inference.memo import MemoDonationError

        def g(z):
            return z * 3.0

        def outer(x):
            return jax.jit(g, donate_argnums=0)(x * 2)

        wrapped = memoize_jaxpr(outer)
        with pytest.raises(MemoDonationError):
            wrapped(jnp.ones((4,), jnp.float32))

    # -- AC-8 -----------------------------------------------------------

    def test_ac8_latch_and_rewrap_rescreens(self):
        from xtrax.inference.memo import MemoDonationError

        def f(x):
            return jax.jit(lambda y: y * 2, donate_argnums=0)(x)

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoDonationError):
            wrapped(x)  # 1st call: screens, raises, latches
        with pytest.raises(MemoDonationError):
            wrapped(x)  # 2nd call: latch fires immediately, no re-trace
        wrapped.memo_rewrap()
        with pytest.raises(MemoDonationError):
            wrapped(x)  # 3rd call: re-screens and raises again

    # -- AC-9 -----------------------------------------------------------

    def test_ac9_zero_arg_wrap_preserves_donation_error_type(self):
        from xtrax.inference.memo import MemoDonationError

        x = jnp.ones((4,), jnp.float32)

        def fn():
            return jax.jit(lambda z: z * 2, donate_argnums=0)(x)

        with pytest.raises(MemoDonationError) as exc_info:
            memoize_jaxpr(fn)
        assert type(exc_info.value) is MemoDonationError

    # -- AC-10 (canaries) -------------------------------------------------

    def test_ac10_canary_donation_markers_visible_in_jaxpr(self):
        def f(x):
            return x * 2.0

        closed_jit = jax.make_jaxpr(jax.jit(f, donate_argnums=0))(jnp.ones((4,)))
        eqn = closed_jit.jaxpr.eqns[0]
        donated_invars = eqn.params.get("donated_invars")
        assert donated_invars is not None and any(donated_invars), (
            "jax no longer surfaces donate_argnums as a donated_invars param "
            "on the jit eqn; the donation screen would silently pass."
        )

        def g(x):
            return jax.device_put(x, donate=True) * 2.0

        closed_dp = jax.make_jaxpr(g)(jnp.ones((4,)))
        dp_eqn = next(eqn for eqn in closed_dp.jaxpr.eqns if eqn.primitive.name == "device_put")
        copy_semantics = dp_eqn.params.get("copy_semantics")
        assert copy_semantics is not None and any(
            getattr(cs, "name", None) == "DONATE_INPUT" for cs in copy_semantics
        ), (
            "jax no longer surfaces device_put(donate=True) as a "
            "copy_semantics DONATE_INPUT element; the donation screen would "
            "silently pass."
        )

    # -- AC-11 ------------------------------------------------------------

    def test_ac11_copy_on_return_protects_store_from_input_deletion(self):
        f = memoize_jaxpr(lambda x: x, policy=MemoPolicy(copy_on_return=True))
        x = jnp.ones((4,), jnp.float32)
        x2 = x.copy()
        f(x)  # miss
        x.delete()
        r = f(x2)  # hit (same content/key as x)
        assert bool(jnp.allclose(r, x2))

    # -- AC-12a -------------------------------------------------------------

    def test_ac12a_copy_on_return_true_hit_survives_miss_return_deletion(self):
        f = memoize_jaxpr(lambda x: x * 2.0, policy=MemoPolicy(copy_on_return=True))
        x = jnp.ones((4,), jnp.float32)
        r1 = f(x)  # miss
        r1.delete()
        r2 = f(x)  # hit
        assert bool(jnp.allclose(r2, x * 2.0))
        assert f.memo_get_stats()["hits"] == 1

    def test_ac12a_negative_control_default_policy_aliases(self):
        f = memoize_jaxpr(lambda x: x * 2.0)  # copy_on_return=False (default)
        x = jnp.ones((4,), jnp.float32)
        r1 = f(x)
        r1.delete()
        r2 = f(x)
        assert r2 is r1
        assert r2.is_deleted()

    # -- AC-12b ---------------------------------------------------------

    def test_ac12b_copy_on_return_true_each_hit_independent(self):
        f = memoize_jaxpr(lambda x: x * 2.0, policy=MemoPolicy(copy_on_return=True))
        x = jnp.ones((4,), jnp.float32)
        f(x)  # miss
        r2 = f(x)  # hit
        r2.delete()
        r3 = f(x)  # hit
        assert bool(jnp.allclose(r3, x * 2.0))
        assert r3 is not r2

    def test_ac12b_negative_control_default_policy_aliases(self):
        f = memoize_jaxpr(lambda x: x * 2.0)
        x = jnp.ones((4,), jnp.float32)
        f(x)
        r2 = f(x)
        r2.delete()
        r3 = f(x)
        assert r3 is r2
        assert r3.is_deleted()

    # -- AC-12c -----------------------------------------------------------

    def test_ac12c_copy_on_return_true_spot_checked_hit_survives_deletion(self):
        f = memoize_jaxpr(
            lambda x: x * 2.0,
            policy=MemoPolicy(copy_on_return=True, spot_check_every=1),
        )
        x = jnp.ones((4,), jnp.float32)
        r1 = f(x)  # miss
        r1.delete()
        r2 = f(x)  # spot-checked hit
        assert bool(jnp.allclose(r2, x * 2.0))
        assert f.memo_get_stats()["spot_check_mismatches"] == 0

    def test_ac12c_negative_control_default_policy_raises_staleness(self):
        f = memoize_jaxpr(lambda x: x * 2.0, policy=MemoPolicy(spot_check_every=1))
        x = jnp.ones((4,), jnp.float32)
        r1 = f(x)
        r1.delete()
        with pytest.raises(MemoStalenessError):
            f(x)

    # -- AC-13 ------------------------------------------------------------

    def test_ac13_copy_on_return_scalar_leaf_untouched(self):
        def f(x):
            return x * 2, 3.0

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(copy_on_return=True))
        x = jnp.ones((4,), jnp.float32)
        arr1, scalar1 = wrapped(x)  # miss
        assert scalar1 == 3.0
        assert bool(jnp.allclose(arr1, x * 2))
        arr2, scalar2 = wrapped(x)  # hit
        assert scalar2 == 3.0
        assert bool(jnp.allclose(arr2, x * 2))

    # -- AC-14 ------------------------------------------------------------

    def test_ac14_default_hit_returns_same_object(self):
        def f(x):
            return x * 2.0

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        wrapped(x)  # miss
        assert wrapped(x) is wrapped(x)  # both hits: same cached object


class TestPerSignatureScreen:
    """T2 §3.1-3.3, 3.5, 3.7: Per-signature screening with ABSTRACT and STATIC modes.

    AC-1..AC-8b, AC-12, AC-13, AC-13b.
    """

    # AC-1: #5214 — shape-dependent donation is re-screened per call signature
    def test_ac1_shape_dependent_donation_rescreened(self):
        def f(x):
            if x.shape[0] > 4:
                return jax.jit(lambda y: y * 2, donate_argnums=0)(x)
            return x * 2

        wrapped = memoize_jaxpr(f)
        wrapped(jnp.ones((4,), jnp.float32))  # 1st call: shape (4,), no donation
        from xtrax.inference.memo import MemoDonationError

        with pytest.raises(MemoDonationError):
            wrapped(jnp.ones((8,), jnp.float32))  # 2nd call: shape (8,), should raise

    # AC-2: Purity screen also re-screened per signature (shape-dependent impurity)
    def test_ac2_shape_dependent_impurity_rescreened(self):
        def f(x):
            if x.shape[0] > 4:
                return x + jax.random.uniform(jax.random.key(0), x.shape)
            return x * 2

        wrapped = memoize_jaxpr(f)
        wrapped(jnp.ones((4,), jnp.float32))  # First shape: admitted
        with pytest.raises(MemoImpurityError):
            wrapped(jnp.ones((8,), jnp.float32))  # Second shape: impure, raises MemoImpurityError

    # AC-3: #5215 — kwargs are traced, kwargs-dependent donation is screened
    def test_ac3_kwarg_dependent_donation_traced(self):
        def f(x, *, fast=False):
            if fast:
                return jax.jit(lambda y: y * 2, donate_argnums=0)(x)
            return x * 2

        wrapped = memoize_jaxpr(f)
        # Call without fast=True first
        wrapped(jnp.ones((4,), jnp.float32), fast=False)
        from xtrax.inference.memo import MemoDonationError

        # Now with fast=True, should raise because donation is detected
        with pytest.raises(MemoDonationError):
            wrapped(jnp.ones((4,), jnp.float32), fast=True)

    # AC-4: #5215 — kwargs-dependent impurity is screened
    def test_ac4_kwarg_dependent_impurity(self):
        def f(x, *, fast=False):
            if fast:
                return x + jax.random.uniform(jax.random.key(0), x.shape)
            return x * 2

        wrapped = memoize_jaxpr(f)
        # First call admitted with fast=False
        wrapped(jnp.ones((4,), jnp.float32), fast=False)
        assert wrapped._memo_core.screen_latched_error is None
        # Hit repeat
        wrapped(jnp.ones((4,), jnp.float32), fast=False)
        # Call with fast=True should raise impurity
        with pytest.raises(MemoImpurityError):
            wrapped(jnp.ones((4,), jnp.float32), fast=True)
        # Afterwards, even fast=False should raise (latched)
        with pytest.raises(MemoImpurityError):
            wrapped(jnp.ones((4,), jnp.float32), fast=False)

    # AC-5: STATIC fallback for Python int branches
    @pytest.mark.parametrize(
        "branch_expr,desc",
        [
            ("n > 1", "(a) if n > 1"),
            ("sum(range(n))", "(b) sum(range(n))"),
            ("jnp.zeros(n).shape[0]", "(c) jnp.zeros(n)"),
            ("x[:n]", "(d) x[:n]"),
        ],
    )
    def test_ac5_static_fallback_python_int_branches(self, branch_expr, desc):
        """STATIC fallback allows Python int branches (AC-5a-d).

        Each variant does 3 calls: (n=A), (n=A) again, (n=B). The positive
        assertion is admission with the RIGHT hit/miss shape (same n hits,
        different n misses) — never main's exception type (spec AC-5).
        """
        if branch_expr == "x[:n]":
            # x[:n] indexing
            def f(x, n):
                return x[:n]

            wrapped = memoize_jaxpr(f)
            x = jnp.arange(10, dtype=jnp.float32)
            wrapped(x, 3)  # First call with n=3: miss
            wrapped(x, 3)  # Hit with same n
            wrapped(x, 5)  # Miss with different n
        elif branch_expr == "sum(range(n))":
            # range(n)
            def f(x, n):
                s = sum(range(n))
                return x + jnp.float32(s)

            wrapped = memoize_jaxpr(f)
            x = jnp.ones((4,), jnp.float32)
            wrapped(x, 2)
            wrapped(x, 2)
            wrapped(x, 3)
        elif branch_expr == "jnp.zeros(n).shape[0]":
            # jnp.zeros(n) with traced n
            def f(x, n):
                z = jnp.zeros(n)
                return x + jnp.float32(z.shape[0])

            wrapped = memoize_jaxpr(f)
            x = jnp.ones((4,), jnp.float32)
            wrapped(x, 3)
            wrapped(x, 3)
            wrapped(x, 5)
        else:  # "n > 1"

            def f(x, n):
                if n > 1:
                    return x * 2
                return x

            wrapped = memoize_jaxpr(f)
            x = jnp.ones((4,), jnp.float32)
            wrapped(x, 2)  # n > 1
            wrapped(x, 2)  # Hit
            wrapped(x, 3)  # Miss

        # Same n hits, different n misses — the AC-5 positive shape.
        assert wrapped._memo_core.stats.hits == 1, desc
        assert wrapped._memo_core.stats.misses == 2, desc
        assert wrapped._memo_core.stats.calls == 3, desc

    # AC-5e: OverflowError propagates unchanged (not relabeled, not swallowed).
    # x + n with n = 2**40 overflows int32 during promotion, in BOTH the
    # eager call and the STATIC-mode trace (measured 260922, spec §3.1) — this
    # is a pin, not a "fails on main" control (see evidence discipline table).
    def test_ac5e_overflow_error_propagates(self):
        def f(x, n):
            return x + n

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(OverflowError):
            wrapped(x, 2**40)
        # Not relabeled as MemoKeyUnsupportedLeafError, and not swallowed:
        # the eager function itself raises the SAME exception type.
        with pytest.raises(OverflowError):
            f(x, 2**40)

    # AC-5f: Impurity in STATIC mode (int value selects impure branch)
    def test_ac5f_impurity_in_static_mode(self):
        def f(x, n):
            if n > 3:
                return x + jax.random.uniform(jax.random.key(0), x.shape)
            return x * 2

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoImpurityError):
            wrapped(x, 5)  # n > 3, so impure

    # AC-5g: G4 — in-range ints share one ABSTRACT trace, out-of-range triggers STATIC
    def test_ac5g_g4_fits_default_int(self):
        """Fit-range ints share traces; out-of-range triggers STATIC."""
        from xtrax.inference.memo import _classify_leaf

        # Verify _fits_default_int behavior directly.
        large_int = 2**40
        small_int = 2**30
        kind, descriptor = _classify_leaf(large_int, mode="ABSTRACT")
        assert kind == "dyn"
        # fits_default_int should be False for 2**40
        assert descriptor == ("dyn", "int", False)

        kind, descriptor = _classify_leaf(small_int, mode="ABSTRACT")
        assert descriptor == ("dyn", "int", True)

        # Behavioral half of AC-5g: g(x, 2**40) is admitted via STATIC mode
        # (its own ABSTRACT token, distinct from in-range ints — §3.1 item 2),
        # then g(x, 3) and g(x, 4) — both in-range — share ONE ABSTRACT trace.
        trace_count = {"count": 0}

        def g(x, n):
            if isinstance(n, jax.core.Tracer):
                trace_count["count"] += 1
            return x + (n % 7)

        wrapped = memoize_jaxpr(g)
        x = jnp.ones((4,), jnp.float32)
        wrapped(x, 2**40)  # admitted via STATIC mode fallback
        wrapped(x, 3)  # ABSTRACT mode: traces
        wrapped(x, 4)  # same ABSTRACT token as n=3: no retrace
        assert trace_count["count"] == 1

    # AC-6: String arguments are memoizable, NFC and NFD forms are distinct
    def test_ac6_string_keys_exact(self):
        def f(x, s):
            return x * jnp.float32(len(s))

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        # NFC form
        s_nfc = "é"  # precomposed
        wrapped(x, s_nfc)
        assert wrapped._memo_core.stats.calls == 1

        # Hit with same form
        wrapped(x, s_nfc)
        assert wrapped._memo_core.stats.calls == 2

        # NFD form (decomposed)
        s_nfd = "é"  # decomposed
        wrapped(x, s_nfd)
        # Should be a miss (distinct key from NFC)
        assert wrapped._memo_core.stats.calls == 3
        assert wrapped._memo_core.stats.misses == 2

    # AC-6b: Bytes arguments are memoizable
    def test_ac6b_bytes_keys_exact(self):
        def f(x, b):
            return x * jnp.float32(len(b))

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        wrapped(x, b"\xff")
        assert wrapped._memo_core.stats.calls == 1
        wrapped(x, b"\xff")
        assert wrapped._memo_core.stats.calls == 2

    # AC-7: G4 — float scalars trace once per ABSTRACT signature, not per value
    # AC-7: G4 — float scalars trace once per ABSTRACT signature, not per value
    def test_ac7_g4_float_scalars_single_trace(self):
        """Five distinct float values share one trace (same program digest)."""
        trace_count = {"count": 0}

        def f(x, s):
            # Count when s is a Tracer (i.e., during tracing)
            if isinstance(s, jax.core.Tracer):
                trace_count["count"] += 1
            return x * s

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        # All different float values
        for s in [1.0, 2.0, 3.0, 4.0, 5.0]:
            wrapped(x, s)

        # Should trace only once (all share ABSTRACT signature, same program digest)
        # Each call is a cache miss (different keys), but only 1 trace happened
        assert trace_count["count"] == 1

    def test_ac7b_bool_static_d2(self):
        def f(x, flag):
            if flag is True:
                return x + jax.random.uniform(jax.random.key(0), x.shape)
            return x * 2

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        # flag=False first (admitted)
        wrapped(x, False)
        assert wrapped._memo_core.screen_latched_error is None

        # flag=True (impurity detected)
        with pytest.raises(MemoImpurityError):
            wrapped(x, True)

        # flag=False again (latched error)
        with pytest.raises(MemoImpurityError):
            wrapped(x, False)

    # AC-7c: D-2 — enum-like int subclasses are held static
    def test_ac7c_enum_static_d2(self):
        from enum import IntEnum

        class Mode(IntEnum):
            A = 1
            B = 2

        def f(x, mode):
            if mode is Mode.B:
                return x + jax.random.uniform(jax.random.key(0), x.shape)
            return x * 2

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        # Mode.A first (admitted)
        wrapped(x, Mode.A)
        assert wrapped._memo_core.screen_latched_error is None

        # Mode.B (impurity detected)
        with pytest.raises(MemoImpurityError):
            wrapped(x, Mode.B)

    # AC-8: Array-value refusal in three forms
    @pytest.mark.parametrize(
        "body,idiom",
        [
            ("if x[0] > 0: return x + 1", "indexing"),
            ("return x[x > 0]", "boolean indexing"),
            ("return np.asarray(x)", "np.asarray conversion"),
        ],
    )
    def test_ac8_array_value_refusal(self, body, idiom):
        """Array value refusal in three forms."""
        if idiom == "indexing":

            def f(x):
                if x[0] > 0:
                    return x + 1
                return x
        elif idiom == "boolean indexing":

            def f(x):
                return x[x > 0]
        else:  # np.asarray conversion

            def f(x):
                return np.asarray(x)

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoKeyUnsupportedLeafError) as exc_info:
            wrapped(x)
        assert "value of an array argument" in str(exc_info.value)

    # AC-8b: Non-classified exceptions propagate unchanged
    def test_ac8b_unclassified_exception_propagates(self):
        def f(x):
            raise ZeroDivisionError("test error")

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(ZeroDivisionError, match="test error"):
            wrapped(x)

    # AC-12: Latch is global (all signatures see latched error)
    def test_ac12_latch_across_signatures(self):
        def f(x, fast=False):
            if fast:
                return x + jax.random.uniform(jax.random.key(0), x.shape)
            return x * 2

        wrapped = memoize_jaxpr(f)
        x1 = jnp.ones((4,), jnp.float32)
        x2 = jnp.ones((8,), jnp.float32)

        # Admit signature 1
        wrapped(x1, fast=False)
        # Reject signature 2
        with pytest.raises(MemoImpurityError):
            wrapped(x2, fast=True)
        # Signature 1 should now be latched
        with pytest.raises(MemoImpurityError):
            wrapped(x1, fast=False)
        # Clear latch
        wrapped.memo_rewrap()
        # Signature 1 should work again
        wrapped(x1, fast=False)

    # AC-13: Table eviction (LRU)
    def test_ac13_table_eviction(self, monkeypatch):
        import xtrax.inference.memo as memo_module

        monkeypatch.setattr(memo_module, "_MAX_SCREENED_SIGNATURES", 2)

        # Tracer-only counter (AC-13's "and an evicted shape re-traces
        # (counter)" clause) — a size-bound check alone can't distinguish
        # real LRU eviction from a no-op that happens to never grow.
        trace_count = {"count": 0}

        def f(x):
            if isinstance(x, jax.core.Tracer):
                trace_count["count"] += 1
            return x * 2

        wrapped = memoize_jaxpr(f)

        # Shape (2,): traced and cached
        wrapped(jnp.ones((2,)))
        assert trace_count["count"] == 1
        assert len(wrapped._memo_core._screened) <= 2

        # Shape (3,): traced and cached
        wrapped(jnp.ones((3,)))
        assert trace_count["count"] == 2
        assert len(wrapped._memo_core._screened) <= 2

        # Shape (4,): traced and cached, evicts (2,)
        wrapped(jnp.ones((4,)))
        assert trace_count["count"] == 3
        assert len(wrapped._memo_core._screened) <= 2

        # Shape (2,) again: evicted, so it MUST re-trace (proves eviction
        # actually happened, not just that the table never grew).
        wrapped(jnp.ones((2,)))
        assert trace_count["count"] == 4
        assert len(wrapped._memo_core._screened) <= 2

    # AC-13b: Container type matters (np.ndarray vs jax.Array)
    def test_ac13b_container_type_in_key(self):
        def f(x):
            return x * 2

        wrapped = memoize_jaxpr(f)

        # np.ndarray
        arr_np = np.ones((4,), dtype=np.float32)
        wrapped(arr_np)
        np_call_count = wrapped._memo_core.stats.calls

        # jax.Array (should be a miss, distinct from np.ndarray)
        arr_jax = jnp.ones((4,), dtype=jnp.float32)
        wrapped(arr_jax)
        jax_call_count = wrapped._memo_core.stats.calls

        # They should be distinct entries
        assert jax_call_count > np_call_count
        assert wrapped._memo_core.stats.misses == 2


class TestDonationDocs:
    def test_ac17_docs_mention_donation_and_screen_ids(self):
        import re
        from pathlib import Path

        docs_path = Path(__file__).resolve().parents[2] / "docs" / "api" / "inference.md"
        content = docs_path.read_text()
        patterns = (
            r"memoize_jaxpr",
            r"donat",
            r"#5214",
            r"#5215",
            r"#5216",
            r"#5231",
            r"#5233",
            r"STATIC",
            r"ABSTRACT",
        )
        for pattern in patterns:
            assert re.search(pattern, content), f"missing {pattern!r} in docs/api/inference.md"


class TestAuditGaps:
    """Audit gap tests F1-F5: non-latching refusal, exact string keys, STATIC trace bound,
    digest and container-type tokens."""

    def test_f1_classification_failure_no_latch(self):
        """F1: a classification failure (MemoKeyUnsupportedLeafError) must not
        latch the screen error. After the unsupported leaf raises, the next call
        with a supported leaf should succeed (screen_latched_error is None)."""
        wrapped = memoize_jaxpr(lambda x: x * 2.0)

        # First call with unsupported leaf type (set is not supported)
        with pytest.raises(MemoKeyUnsupportedLeafError):
            wrapped({1, 2})

        # Verify that the error did NOT latch
        assert wrapped._memo_core.screen_latched_error is None

        # Next call with a valid supported leaf should succeed
        x = jnp.ones((3,), jnp.float32)
        result = wrapped(x)
        np.testing.assert_allclose(result, x * 2.0)

    def test_f2_string_keys_not_normalized(self):
        """F2: exact string keys are not confounded by Unicode normalization.
        Two Unicode forms of 'é' (NFC precomposed vs NFD decomposed) are treated
        as distinct cache keys even when they don't affect the traced program."""
        import unicodedata

        def f(x, s):
            # String s does NOT affect the traced program (not used in computation)
            return x * 2.0

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        # NFC form (precomposed): U+00E9
        s_nfc = "é"  # decomposed: e + combining acute
        s_nfc = unicodedata.normalize("NFC", s_nfc)  # now precomposed
        result1 = wrapped(x, s_nfc)

        # NFD form (decomposed): e + combining acute accent
        s_nfd = "é"  # already decomposed
        assert s_nfc != s_nfd, "Test setup: NFC and NFD forms must be distinct"
        assert unicodedata.normalize("NFC", s_nfc) == unicodedata.normalize("NFC", s_nfd), (
            "Test setup: both should normalize to same NFC form"
        )
        result2 = wrapped(x, s_nfd)
        stats2 = wrapped.memo_get_stats()

        # Both calls should be misses (distinct string keys)
        assert stats2["misses"] == 2, "Both calls should miss (different string keys)"
        assert stats2["hits"] == 0, "No hits expected (each string form is unique)"
        np.testing.assert_allclose(result1, x * 2.0)
        np.testing.assert_allclose(result2, x * 2.0)

    def test_f3_static_fallback_trace_bounded(self):
        """F3: the STATIC-fallback path is trace-bounded (G4). Repeated calls
        with fresh but equal arrays and the same static int value should not
        retrace beyond the first failure + STATIC retry."""
        trace_count = {"count": 0}

        def f(x, n):
            if isinstance(x, jax.core.Tracer):
                trace_count["count"] += 1
            # Python int branch forces STATIC mode
            if n > 1:
                return x * 2.0
            return x

        wrapped = memoize_jaxpr(f)

        # First call with n=3: ABSTRACT fails, STATIC succeeds, trace count = 1 or 2
        x1 = jnp.ones((4,), jnp.float32)
        wrapped(x1, 3)
        trace_count_after_first = trace_count["count"]
        assert trace_count_after_first > 0, "First call should trace"

        # Second and third calls with fresh but equal arrays, same n=3
        # Should hit the STATIC cache, no re-tracing
        x2 = jnp.ones((4,), jnp.float32)
        wrapped(x2, 3)
        trace_count_after_second = trace_count["count"]
        assert trace_count_after_second == trace_count_after_first, "Second call should not retrace"

        x3 = jnp.ones((4,), jnp.float32)
        wrapped(x3, 3)
        trace_count_after_third = trace_count["count"]
        assert trace_count_after_third == trace_count_after_first, "Third call should not retrace"

        # Call with a different static value n=5
        # ABSTRACT token is already _NEEDS_STATIC, so no ABSTRACT retry, just one more STATIC trace
        x4 = jnp.ones((4,), jnp.float32)
        wrapped(x4, 5)
        trace_count_after_new_n = trace_count["count"]
        assert trace_count_after_new_n == trace_count_after_first + 1, (
            "New static value should cause exactly one more trace (STATIC only, no ABSTRACT retry)"
        )

    def test_f4_build_key_depends_on_digest(self):
        """F4: build_key depends on the digest. Two different digests should
        produce different keys for the same args."""

        def f(x):
            return x * 2.0

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)

        # Trigger a call to populate the core state
        wrapped(x)
        core = wrapped._memo_core

        # Call _ensure_screened to get a digest
        digest_a = core._ensure_screened((x,), {})

        # Build keys with different digests
        key_a = core.build_key(digest_a, (x,), {})
        key_b = core.build_key("different_digest_hash", (x,), {})

        # Keys must differ because digests differ
        assert key_a != key_b, "Different digests must produce different keys"

    def test_f5_container_type_in_token(self):
        """F5: container type (np.ndarray vs jax.Array) is in the signature
        token, not only in the leaf digest. Different container types must
        trace separately."""
        from xtrax.inference.memo import _classify_leaf

        # Direct classification test: np.ndarray and jax.Array have different descriptors
        np_arr = np.ones((3,), dtype=np.float32)
        jax_arr = jnp.ones((3,), dtype=jnp.float32)

        np_kind, np_descriptor = _classify_leaf(np_arr, "ABSTRACT")
        jax_kind, jax_descriptor = _classify_leaf(jax_arr, "ABSTRACT")

        # Both are "arr" kind, but descriptors must differ (container type is different)
        assert np_kind == "arr" and jax_kind == "arr"
        assert np_descriptor != jax_descriptor, (
            "np.ndarray and jax.Array must have different descriptors (container type differs)"
        )

        # Behavioral test: traced calls with different container types
        trace_count = {"count": 0}

        def f(x):
            if isinstance(x, jax.core.Tracer):
                trace_count["count"] += 1
            return x * 2.0

        wrapped = memoize_jaxpr(f)

        # Call with np.ndarray
        np_arr1 = np.ones((4,), dtype=np.float32)
        wrapped(np_arr1)
        trace_count_after_np = trace_count["count"]
        assert trace_count_after_np > 0, "First call with np.ndarray should trace"

        # Call with jax.Array of same shape/dtype
        jax_arr1 = jnp.ones((4,), dtype=jnp.float32)
        wrapped(jax_arr1)
        trace_count_after_jax = trace_count["count"]

        # Must trace twice (tokens differ due to container type)
        assert trace_count_after_jax == 2, (
            "np.ndarray and jax.Array must trace separately (different container types)"
        )


class TestSeamLint:
    def test_ac12_seam_lint_flags_guarded_evaluate_wrapping(self):
        """AST lint test (alias-resolving) flags memoizing guarded_evaluate."""
        import ast
        import textwrap

        from xtrax.inference import cse as _cse  # noqa: F401

        SEAM_MODULE = "guarded_evaluate"
        WRAPPER_NAME = "memoize_jaxpr"

        def find_violations(source: str) -> list[str]:
            tree = ast.parse(textwrap.dedent(source))
            wrapper_aliases: set[str] = set()
            seam_names: set[str] = set()

            class V(ast.NodeVisitor):
                def visit_ImportFrom(self, node):
                    for a in node.names:
                        local = a.asname or a.name
                        if a.name == WRAPPER_NAME:
                            wrapper_aliases.add(local)
                        elif SEAM_MODULE in a.name:
                            seam_names.add(local)
                    self.generic_visit(node)

                def visit_Call(self, node):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    is_wrapper = name == WRAPPER_NAME or name in wrapper_aliases
                    if is_wrapper:
                        for arg in node.args:
                            target = None
                            if isinstance(arg, ast.Name):
                                target = arg.id
                            elif isinstance(arg, ast.Attribute):
                                target = arg.attr
                            if target and (target in seam_names or SEAM_MODULE in (target or "")):
                                violations.append(name)
                    self.generic_visit(node)

            violations: list[str] = []
            V().visit(tree)
            return violations

        bad = """
        from xtrax.inference.memo import memoize_jaxpr as cache
        from xtrax.loop.closure_lock import guarded_evaluate

        wrapped = cache(guarded_evaluate, policy=p)
        """
        assert find_violations(bad), "alias-resolving lint must catch this"

        good = """
        from xtrax.inference.memo import memoize_jaxpr as cache
        from xtrax.loop.closure_lock import guarded_evaluate

        def inner_score(x):
            return x * 2.0

        wrapped_inner = cache(inner_score)
        result = guarded_evaluate(wrapped_inner, None, None)
        """
        assert not find_violations(good)


class TestCostAdvisory:
    def test_ac16_slow_ratio_warning_fires(self):
        def tiny_op(x):  # near-zero op cost, nontrivial leaf hashing
            return x[0] * 1.0

        leaf = np.zeros((512,), dtype=np.float32)
        # Threshold set just below the measured hash/op ratio for this host so the
        # test exercises the warning mechanism deterministically (AC16 validates
        # plumbing + honest attribution; absolute ratio is host-dependent).
        f = memoize_jaxpr(tiny_op, policy=MemoPolicy(slow_ratio_warn=0.01))
        with pytest.warns(RuntimeWarning, match="SLOWER"):
            for i in range(20):
                f(jnp.asarray(leaf + i))  # distinct inputs -> misses

    def test_advisory_disabled_in_pipelining_mode(self):
        def score(x):
            return x[0] * 1.0

        big_x = np.zeros((2048, 64), dtype=np.float32)
        f = memoize_jaxpr(score, policy=MemoPolicy(block_on_miss=False))
        import warnings as w

        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            for i in range(15):
                f(jnp.asarray(big_x[i]))
        assert not any(issubclass(c.category, RuntimeWarning) for c in caught)


class TestDeferredScreen:
    def test_deferred_path_latch_and_rewrap(self):
        """Arg-taking fn: screen deferred to first real call, then latches."""

        def impure(x):
            return jax.random.uniform(jax.random.key(0), x.shape)

        wrapped = memoize_jaxpr(impure)
        x = jnp.ones((4,), jnp.float32)
        # Zero-arg wrap-time screen does NOT fire (fn takes an arg).
        assert wrapped._memo_core.screen_latched_error is None
        with pytest.raises(MemoImpurityError):
            wrapped(x)  # first call: deferred screen fires
        assert wrapped._memo_core.screen_latched_error is not None
        with pytest.raises(MemoImpurityError):
            wrapped(x)  # latched: immediate raise, no re-screen
        wrapped.memo_rewrap()
        assert wrapped._memo_core.screen_latched_error is None
        with pytest.raises(MemoImpurityError):
            wrapped(x)  # re-screens after rewrap and fails again


class TestSpotCheckReplay:
    """T3 §3.6 (#5231): spot-check replay uses this call's own args and kwargs.

    AC-14, AC-15: kwargs-faithful spot-check replay, no races.
    """

    def test_ac14_kwarg_spot_check_replay(self):
        """AC-14: f(x, *, scale=1.0) called twice with scale=3.0 hits without staleness."""

        def f(x, *, scale=1.0):
            return x * scale

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(spot_check_every=1))
        x = jnp.ones((4,), jnp.float32)

        # First call: miss
        result1 = wrapped(x, scale=3.0)
        np.testing.assert_allclose(result1, x * 3.0)

        # Second call: hit + spot-check (should NOT raise MemoStalenessError)
        result2 = wrapped(x, scale=3.0)
        np.testing.assert_allclose(result2, x * 3.0)
        assert wrapped._memo_core.stats.hits == 1
        assert wrapped._memo_core.stats.spot_check_mismatches == 0

    def test_ac15_deterministic_race_spot_check(self, monkeypatch):
        """AC-15: deterministic race test, no sleeps.

        Thread 1 calls f(A) (spot-checked hit), blocks on build_key patch until
        Thread 2 completes f(B) call. Assert thread 1 gets no staleness error.
        """
        import threading

        def fn(x):
            return x * 2.0

        A = jnp.ones((4,), jnp.float32)
        B = jnp.full((4,), 2.0, jnp.float32)

        wrapped = memoize_jaxpr(fn, policy=MemoPolicy(spot_check_every=1))
        core = wrapped._memo_core

        # Warm up: f(A) is a miss
        wrapped(A)
        assert core.stats.misses == 1

        # Monkeypatch build_key to block on first detection of A
        orig_build_key = core.build_key
        entered = threading.Event()
        release = threading.Event()
        detected_a = {"count": 0}

        def patched_build_key(*a, **k):
            # Detect A: check if A is in the args tuple
            for arg_container in a:
                if isinstance(arg_container, tuple):
                    for elem in arg_container:
                        if elem is A:
                            detected_a["count"] += 1
                            if detected_a["count"] == 1:
                                # First detection of A: signal and wait
                                entered.set()
                                release.wait(timeout=10)
                            break
            return orig_build_key(*a, **k)

        monkeypatch.setattr(core, "build_key", patched_build_key)

        # Thread 1: call wrapped(A) (hit + spot-check)
        t1_result = {}
        t1_error = {}

        def thread1_target():
            try:
                t1_result["value"] = wrapped(A)
            except Exception as exc:
                t1_error["exc"] = exc

        t1 = threading.Thread(target=thread1_target)
        t1.start()

        # Wait for t1 to enter the patched build_key
        assert entered.wait(timeout=10), "Thread 1 did not enter build_key patch"

        # Main thread: call wrapped(B) to completion
        result_b = wrapped(B)
        np.testing.assert_allclose(result_b, B * 2.0)

        # Release thread 1
        release.set()

        # Join thread 1 with timeout
        t1.join(timeout=10)
        assert not t1.is_alive(), "Thread 1 hung (deadlock detected)"

        # Assert thread 1 succeeded
        assert "exc" not in t1_error, f"Thread 1 raised: {t1_error.get('exc')}"
        assert "value" in t1_result
        np.testing.assert_allclose(t1_result["value"], A * 2.0)


class TestCodeReviewFixes:
    """/code-review findings on PR #159 (spec 260922_memo-screen-hardening),
    fixed CR-1 .. CR-6. See docs/api/inference.md and memo.py comments tagged
    CR-N for the corresponding fix."""

    # -- CR-1: string/bytes leaf digests are not injective -----------------

    def test_cr1_string_leaf_digest_injective(self):
        """`_leaf_digest`'s old `b"str:" + enc` (no length) let two calls with
        different string args alias onto the same digest stream when their
        concatenation matches: digest("a") + digest("str:b") == digest("astr:")
        + digest("b"). `g`'s traced program (two array outputs) does not
        depend on `a`/`b` at all — they are only used as PYTHON DICT KEYS in
        the return value — so the screen digest is identical for both calls
        and the collision, if any, is entirely `build_key`'s leaf-digest bug.
        Length-prefixing fixes it: both calls miss, and the second call's
        returned dict has ITS OWN keys, not the first call's.
        """

        def g(x, a, b):
            return {a: x, b: x * 3}

        wrapped = memoize_jaxpr(g)
        x = jnp.ones((3,), jnp.float32)

        r1 = wrapped(x, "a", "str:b")
        r2 = wrapped(x, "astr:", "b")

        stats = wrapped.memo_get_stats()
        assert stats["misses"] == 2, "colliding digest would serve call 2 from call 1's cache (hit)"
        assert stats["hits"] == 0
        assert sorted(r1.keys()) == ["a", "str:b"]
        assert sorted(r2.keys()) == ["astr:", "b"], "call 2 must return ITS OWN keys, not call 1's"

    def test_cr1_bytes_leaf_digest_injective(self):
        """Same collision shape as test_cr1_string_leaf_digest_injective, for
        `bytes` leaves: the old unprefixed `b"bytes:" + leaf` scheme aliases
        `enc(b"a") + enc(b"bytes:b")` with `enc(b"abytes:") + enc(b"b")`
        (both concatenate to `b"bytes:abytes:bytes:b"`)."""

        def g(x, a, b):
            return {a: x, b: x * 3}

        wrapped = memoize_jaxpr(g)
        x = jnp.ones((3,), jnp.float32)

        r1 = wrapped(x, b"a", b"bytes:b")
        r2 = wrapped(x, b"abytes:", b"b")

        stats = wrapped.memo_get_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0
        assert sorted(r1.keys()) == [b"a", b"bytes:b"]
        assert sorted(r2.keys()) == [b"abytes:", b"b"]

    def test_cr1_static_exact_token_already_injective_by_comment(self):
        """Pin: `_static_exact_token` is unchanged (per CR-1's instruction not
        to touch it) — its result is embedded as one element of a
        structurally-compared tuple descriptor, never concatenated into a
        flat byte stream, so it needs no length prefix. Confirmed here by
        checking two leaves whose encodings would alias under naive
        concatenation still produce DIFFERENT descriptors (they always did;
        this pins that `_classify_leaf`, unlike the old `_leaf_digest`, was
        never the source of the CR-1 collision)."""
        from xtrax.inference.memo import _classify_leaf

        _, desc_a = _classify_leaf("a", "ABSTRACT")
        _, desc_b = _classify_leaf("astr:", "ABSTRACT")
        assert desc_a != desc_b

    # -- CR-2: spot-check hit path never increments stats.calls -------------

    def test_cr2_spot_check_hit_counts_call(self):
        """1 miss + 8 identical hits, `spot_check_every=3`. Derivation of the
        expected `stats["calls"]` from the counter semantics (memo.py
        `call()`): `next_call_number = self.stats.calls + 1` is checked
        `% 3 == 0`, evaluated freshly BEFORE each hit's own increment:

            call 1 (miss):            calls 0 -> 1
            call 2 (hit #1): next=2,  2%3 != 0            -> calls 1 -> 2
            call 3 (hit #2): next=3,  3%3 == 0  spot-check -> calls 2 -> 3
            call 4 (hit #3): next=4,  4%3 != 0            -> calls 3 -> 4
            call 5 (hit #4): next=5,  5%3 != 0            -> calls 4 -> 5
            call 6 (hit #5): next=6,  6%3 == 0  spot-check -> calls 5 -> 6
            call 7 (hit #6): next=7,  7%3 != 0            -> calls 6 -> 7
            call 8 (hit #7): next=8,  8%3 != 0            -> calls 7 -> 8
            call 9 (hit #8): next=9,  9%3 == 0  spot-check -> calls 8 -> 9

        So `stats["calls"] == 9` (every call, hit or miss, counted exactly
        once), with exactly 3 spot-checks among the 8 hits. Each spot-check
        recomputes via the unwrapped `fn`, so concrete (non-Tracer) `fn`
        executions == 1 miss + 3 spot-check recomputes == 4.
        """
        calls = {"concrete": 0}

        def f(x):
            if not isinstance(x, jax.core.Tracer):
                calls["concrete"] += 1
            return x * 2

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(spot_check_every=3))
        x = jnp.ones((3,), jnp.float32)

        wrapped(x)  # miss
        for _ in range(8):
            wrapped(x)  # hits, some spot-checked

        stats = wrapped.memo_get_stats()
        assert stats["calls"] == 9
        assert calls["concrete"] == 4

    def test_cr2_spot_check_entry_evicted_still_counts_call(self):
        """The early-return path in `_maybe_spot_check_unlocked` (the entry
        was evicted from the cache between the hit lookup and the recompute)
        must still count the call. Force eviction via `max_entries=1`: a
        second signature's miss evicts the first signature's only entry, so a
        third call that hits the (stale) key for the first entry finds it
        already gone."""

        def f(x):
            return x * 2

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(max_entries=1, spot_check_every=1))
        x = jnp.ones((3,), jnp.float32)
        y = jnp.ones((5,), jnp.float32)

        wrapped(x)  # miss, caches x's entry
        core = wrapped._memo_core
        digest = core._ensure_screened((x,), {})
        key_x = core.build_key(digest, (x,), {})

        wrapped(y)  # miss on a DIFFERENT shape -> evicts x's entry (max_entries=1)
        assert key_x not in core.cache

        calls_before = wrapped.memo_get_stats()["calls"]
        core._maybe_spot_check_unlocked(key_x, (x,), {})  # entry gone -> early return
        assert wrapped.memo_get_stats()["calls"] == calls_before + 1

    # -- CR-3: donation site wrapped_input_leaf_indices with static leaves --

    def test_cr3_donation_index_maps_through_static_leaves(self):
        """`d(flag, x)` holds `flag` (a bool) static (D-2), so only `x` is
        traced and `closed.jaxpr.invars` has length 1 — its sole invar is `x`,
        the flat leaf at index 1 (flag is index 0). The old identity mapping
        reported the donated invar's own position (0) as if it were the
        wrapped call's flat leaf index; the fix maps invar index 0 through
        `traced_positions` (which is `(1,)`) to the correct flat leaf index 1.
        """

        from xtrax.inference.memo import MemoDonationError

        def d(flag, x):
            return jax.jit(lambda y: y * 2, donate_argnums=0)(x)

        wrapped = memoize_jaxpr(d)
        x = jnp.ones((3,), jnp.float32)
        with pytest.raises(MemoDonationError) as exc_info:
            wrapped(True, x)
        assert exc_info.value.sites[0][3] == (1,)

    def test_cr3_donation_index_arrays_only_unaffected(self):
        """Regression: when every leaf is an array (nothing held static),
        `traced_positions` is the identity permutation, so `j == leaf index`
        exactly as before CR-3. Pins that the existing array-only donation
        index tests (TestDonation) keep their old assertions unchanged."""
        from xtrax.inference.memo import MemoDonationError

        def f(p, y):
            a, b = p
            return a + b + y

        wrapped = memoize_jaxpr(jax.jit(f, donate_argnums=1))
        with pytest.raises(MemoDonationError) as exc_info:
            wrapped((jnp.ones((4,)), jnp.ones((4,))), jnp.ones((4,)))
        assert exc_info.value.sites[0][3] == (2,)

    # -- CR-4: IndexError classification is too broad ------------------------

    def test_cr4_plain_indexerror_propagates_unchanged(self):
        """A user function indexing a plain Python TUPLE out of range (a real
        bug, unrelated to consulting an array's value) must surface as a bare
        `IndexError`, not be misreported as
        `MemoKeyUnsupportedLeafError` ("consults the value of an array
        argument"). Array-only args, so there is no STATIC retry to muddy the
        classification."""

        def ib(x, xs):
            return x + xs[5]

        wrapped = memoize_jaxpr(ib)
        x = jnp.ones((3,), jnp.float32)
        xs = (jnp.ones((3,), jnp.float32), jnp.ones((3,), jnp.float32))
        with pytest.raises(IndexError) as exc_info:
            wrapped(x, xs)
        assert not isinstance(exc_info.value, MemoKeyUnsupportedLeafError)
        assert "tuple index out of range" in str(exc_info.value)

    def test_cr4_boolean_indexing_still_classified(self):
        """AC-8 regression: `x[x > 0]` (JAX boolean/nonconcrete indexing) must
        still raise `MemoKeyUnsupportedLeafError` — this is
        `jax.errors.NonConcreteBooleanIndexError`, the narrowed subclass CR-4
        keeps classified, not a plain `IndexError`."""

        def f(x):
            return x[x > 0]

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoKeyUnsupportedLeafError) as exc_info:
            wrapped(x)
        assert "value of an array argument" in str(exc_info.value)

    # -- CR-5: the fallback discards the ABSTRACT exception -------------------

    def test_cr5_abstract_failure_kept_as_note_on_static_failure(self):
        """An `int` arg forces the ABSTRACT->STATIC fallback (§3.1). The body
        raises `ValueError("boom")` UNCONDITIONALLY, so both the ABSTRACT and
        the STATIC trace attempts fail with it. The STATIC failure is what
        propagates (retry policy unchanged); CR-5 additionally attaches a
        note recording that the ABSTRACT attempt failed first, so that
        information is not silently discarded."""

        def f(x, n):
            raise ValueError("boom")

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((3,), jnp.float32)
        with pytest.raises(ValueError) as exc_info:
            wrapped(x, 5)
        notes = list(getattr(exc_info.value, "__notes__", []))
        assert any("ABSTRACT-mode trace failed first" in n and "boom" in n for n in notes), notes

    # -- CR-6: the x64 toggle reuses a stale digest/key -----------------------

    def test_cr6_x64_toggle_forces_new_key(self):
        """`jnp.asarray(x) * 1.0` genuinely depends on the live x64 setting
        (a `np.float64` input truncates to float32 output with x64 disabled,
        stays float64 with it enabled). Calling the SAME signature once
        outside and once inside an `enable_x64()` context must MISS both
        times (not reuse the pre-toggle digest/key), and the inside result's
        dtype must reflect x64 being live."""

        def f(x):
            return jnp.asarray(x) * 1.0

        wrapped = memoize_jaxpr(f)
        x = np.ones((3,), np.float64)

        r_outside = wrapped(x)
        assert r_outside.dtype == np.float32

        x64_ctx = jax.enable_x64 if hasattr(jax, "enable_x64") else None
        if x64_ctx is None:
            pytest.skip("jax.enable_x64 context manager unavailable in this jax version")
        with x64_ctx():
            r_inside = wrapped(x)
            assert r_inside.dtype == np.float64

        stats = wrapped.memo_get_stats()
        assert stats["misses"] == 2, "x64 toggle must force a fresh trace/key, not reuse the entry"
        assert stats["hits"] == 0


# Reference helpers for AC-9: verbatim copies from 49f3def memo.py with _ref prefix
_ref_DonationSite = tuple[str, str, tuple[int, ...], tuple[int, ...]]


def _ref_iter_subjaxprs(value: Any, path: str = ""):
    """Yield (path, subjaxpr) for every subjaxpr-like object (anything
    exposing ``.eqns``) reachable from ``value``, recursing into tuple/list
    values (e.g. ``lax.cond``'s ``branches``) with NO depth cap.

    Shared traversal used by both _screen_jaxpr and _screen_donation to cover
    tuple/list-valued params generically (e.g. lax.cond's branches) while
    respecting structural nesting at any depth.
    """
    stack: list[tuple[str, Any]] = [(path, value)]
    while stack:
        p, v = stack.pop()
        if hasattr(v, "eqns"):
            yield p, v
        elif isinstance(v, (tuple, list)):
            for i, item in enumerate(v):
                stack.append((f"{p}[{i}]" if p else f"[{i}]", item))


def _ref_eqn_label(eqn) -> str:
    name = eqn.params.get("name")
    if name:
        return f"{eqn.primitive.name}[name={name}]"
    return eqn.primitive.name


def _ref_wrapped_leaf_indices(
    eqn,
    operand_indices: tuple[int, ...],
    closed_invars,
    invar_to_leaf: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """Identity lookup only (not provenance tracing): for each donated
    operand of a TOP-LEVEL equation, find `j` such that the operand var IS
    (identity) `closed_invars[j]` — an index into `closed.jaxpr.invars`.

    CR-3: `closed.jaxpr.invars` corresponds only to the TRACED flat leaves
    (`traced_positions` from `_mode_token`/`_trace_closed`), not to every
    flattened `(args, kwargs)` leaf, whenever some leaves are held static
    (bool/enum/str/bytes leaves, or scalars in STATIC mode). `invar_to_leaf`
    maps invar index `j` -> the true flat leaf index (`traced_positions[j]`).
    `None` means identity (every leaf was traced, e.g. an arrays-only call),
    which keeps `j == leaf index` and leaves existing array-only callers
    unaffected.
    """
    out: list[int] = []
    for i in operand_indices:
        operand = eqn.invars[i]
        for j, invar in enumerate(closed_invars):
            if operand is invar:
                out.append(invar_to_leaf[j] if invar_to_leaf is not None else j)
                break
    return tuple(out)


def _ref_eqn_donation_sites(
    eqn,
    path: str,
    top_level: bool,
    closed_invars,
    invar_to_leaf: tuple[int, ...] | None = None,
) -> list[_ref_DonationSite]:
    """Both donation carriers (D3, P5): `donated_invars` (jit/pjit/scan/...)
    and `device_put`'s `copy_semantics` DONATE_INPUT element. Duck-typed on
    `.name` — the private `ArrayCopySemantics` type is never imported."""
    sites: list[_ref_DonationSite] = []

    donated_invars = eqn.params.get("donated_invars")
    if donated_invars:
        idxs = tuple(i for i, d in enumerate(donated_invars) if d)
        if idxs:
            wrapped = (
                _ref_wrapped_leaf_indices(eqn, idxs, closed_invars, invar_to_leaf)
                if top_level
                else ()
            )
            sites.append((path, "donated_invars", idxs, wrapped))

    copy_semantics = eqn.params.get("copy_semantics")
    if copy_semantics:
        idxs = tuple(
            i for i, cs in enumerate(copy_semantics) if getattr(cs, "name", None) == "DONATE_INPUT"
        )
        if idxs:
            wrapped = (
                _ref_wrapped_leaf_indices(eqn, idxs, closed_invars, invar_to_leaf)
                if top_level
                else ()
            )
            sites.append((path, "copy_semantics", idxs, wrapped))

    return sites


def _ref_donation_message(sites: tuple[_ref_DonationSite, ...]) -> str:
    lines = [
        f"  - {path} (carrier={carrier}, eqn_operand_indices={op_idx}, "
        f"wrapped_input_leaf_indices={leaf_idx})"
        for path, carrier, op_idx, leaf_idx in sites
    ]
    return (
        "Function rejected by donation screen (spec §4.2 item 6): "
        "memoize_jaxpr never admits a function whose traced jaxpr carries a "
        "donation marker, at any depth. Sites:\n"
        + "\n".join(lines)
        + "\nRemedy: remove donate_argnums/donate_argnames/device_put(donate=True) "
        "from functions wrapped by memoize_jaxpr (spec §4.2 item 6)."
    )


def _reference_screen_jaxpr(closed) -> None:
    """Raise MemoImpurityError on detectably impure primitives. Traversal walks
    with an explicit stack via _iter_subjaxprs exclusively, covering tuple/list-
    valued params generically and with no depth cap."""
    banned = memo._STATEFUL_PRIMITIVES | memo._CALLBACK_PRIMITIVES | memo._RANDOM_PRIMITIVES

    offenders: list[tuple[str, str]] = []  # (primitive_name, path) pairs
    stack: list[tuple[Any, str]] = [(closed.jaxpr, "jaxpr")]
    while stack:
        jaxpr_obj, jaxpr_path = stack.pop()
        for eqn in jaxpr_obj.eqns:
            eqn_path = f"{jaxpr_path}.{_ref_eqn_label(eqn)}"
            name = eqn.primitive.name
            if name in banned:
                offenders.append((name, eqn_path))
            for param_name, param_val in eqn.params.items():
                for sub_path, sub_jaxpr in _ref_iter_subjaxprs(param_val, param_name):
                    stack.append((sub_jaxpr, f"{eqn_path}.{sub_path}"))
    if offenders:
        names = sorted(set(name for name, _ in offenders))
        paths = ", ".join(path for _, path in offenders)
        raise MemoImpurityError(
            f"Function rejected by purity screen: stateful/callback/random "
            f"primitives present: {names}. If you believe this "
            "function is pure, restructure to avoid these primitives; wrapping "
            f"is the purity attestation.\nPaths: {paths}"
        )


def _reference_screen_donation(closed, invar_to_leaf: tuple[int, ...] | None = None) -> None:
    """Collect ALL donation-hazard sites (both carriers, at any depth), then
    raise once (§3.3 "conservative" rule). Traversal walks with an explicit
    stack via `_iter_subjaxprs` exclusively, so it covers tuple/list-valued
    params generically and has no depth cap.

    CR-3: `invar_to_leaf` (typically the caller's `traced_positions`) maps a
    top-level invar index to the true flat `(args, kwargs)` leaf index, for
    callers that traced fewer leaves than the flattened arg count (D-2
    static leaves). `None` (the default) keeps the old identity mapping.
    """
    sites: list[_ref_DonationSite] = []
    closed_invars = closed.jaxpr.invars
    stack: list[tuple[Any, str, bool]] = [(closed.jaxpr, "jaxpr", True)]
    while stack:
        jaxpr_obj, jaxpr_path, top_level = stack.pop()
        for eqn in jaxpr_obj.eqns:
            eqn_path = f"{jaxpr_path}.{_ref_eqn_label(eqn)}"
            sites.extend(
                _ref_eqn_donation_sites(
                    eqn,
                    eqn_path,
                    top_level,
                    closed_invars,
                    invar_to_leaf if top_level else None,
                )
            )
            for param_name, param_val in eqn.params.items():
                for sub_path, sub_jaxpr in _ref_iter_subjaxprs(param_val, param_name):
                    stack.append((sub_jaxpr, f"{eqn_path}.{sub_path}", False))
    if sites:
        raise MemoDonationError(_ref_donation_message(tuple(sites)), sites=tuple(sites))


def _reference_screen(closed, traced):
    """Run purity screen first, then donation screen. Return the raised
    exception or None."""
    try:
        _reference_screen_jaxpr(closed)
        _reference_screen_donation(closed, traced)
    except (MemoImpurityError, MemoDonationError) as exc:
        return exc
    return None


# Reference helpers for AC-10: verbatim copy of _leaf_digest from memo.py:125-161
def _reference_leaf_digest(leaf: Any, sink: hashlib._Hash) -> None:
    """Fold one pytree leaf into the digest stream (spec §3.3).

    Arrays fold the container type before the dtype (D-4, AC-13b), so an
    `np.ndarray` and an equal `jax.Array` digest differently. `str`/`bytes`
    digest EXACTLY, with no Unicode normalization (D-3, AC-6/AC-6b).
    """
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        # House primitive core (zarr_integrity.update_array_digest recipe):
        # container type, then canonicalize + C-order bytes.
        arr = np.asarray(leaf)
        canon = np.ascontiguousarray(arr)
        sink.update(type(leaf).__qualname__.encode())
        sink.update(canon.dtype.name.encode())
        sink.update(repr(canon.shape).encode())
        sink.update(canon.tobytes(order="C"))
        weak = getattr(leaf, "weak_type", False)
        sink.update(f"|wt={bool(weak)}".encode())
        return
    if isinstance(leaf, bytes):
        # CR-1: length-prefixed so back-to-back leaf digests in one stream
        # (build_key has no separator between leaves) cannot alias across a
        # boundary, e.g. digest("a")+digest("str:b") == digest("astr:")+digest("b")
        # under the old unprefixed scheme.
        sink.update(b"bytes:%d:" % len(leaf) + leaf)
        return
    if isinstance(leaf, str):
        enc = leaf.encode("utf-8", "surrogatepass")
        sink.update(b"str:%d:" % len(enc) + enc)
        return
    if isinstance(leaf, (int, float, bool)):
        sink.update(f"{type(leaf).__name__}({leaf!r})".encode())
        return
    raise MemoKeyUnsupportedLeafError(
        f"Unsupported pytree leaf type {type(leaf).__name__!r} for memo key; "
        "admission restricted to arrays, scalars, bools and strings."
    )


def _reference_build_key(core, digest, args, kwargs):
    """Verbatim copy of build_key body from memo.py:724-732, inlining
    _structure_token and _pytree_leaves."""
    h = hashlib.sha256()
    h.update(digest.encode())
    h.update(repr((jax.tree_util.tree_structure((args, kwargs)),)).encode())
    for leaf in jax.tree_util.tree_leaves((args, kwargs)):
        _reference_leaf_digest(leaf, h)
    h.update(f"|x64={bool(jax.config.jax_enable_x64)}".encode())
    h.update(core.policy.salt.encode())
    h.update(core.stamp.encode())
    return h.hexdigest()


class TestSprint260923Pins:
    """AC-8 through AC-11 and AC-18: pins on memo.py:49f3def."""

    def test_ac8_both_hazards_raises_impurity_first(self):
        """AC-8: program with both hazards raises MemoImpurityError exactly."""

        def f(x):
            return jax.jit(lambda y: y * 2, donate_argnums=0)(x) + jax.random.uniform(
                jax.random.key(0), x.shape
            )

        wrapped = memoize_jaxpr(f)
        x = jnp.ones((4,), jnp.float32)
        with pytest.raises(MemoImpurityError) as exc_info:
            wrapped(x)

        # type is exactly MemoImpurityError, not MemoDonationError
        assert type(exc_info.value) is MemoImpurityError
        # message contains purity screen, not donation screen
        assert "purity screen" in str(exc_info.value)
        assert "donation screen" not in str(exc_info.value)
        # wrapper is latched
        assert wrapped._memo_core.screen_latched_error is not None
        # second call raises same type
        with pytest.raises(MemoImpurityError):
            wrapped(x)

    def test_ac9_f_imp_screen_equivalence(self):
        """AC-9: reference screen output matches live screen for F_imp."""

        def f(p, x):
            def body(y):
                return y + jax.random.uniform(jax.random.key(0), y.shape)

            return jax.jit(lambda z: z + jax.random.uniform(jax.random.key(1), z.shape))(
                lax.cond(p, body, lambda y: y, x)
            )

        args = (jnp.array(True), jnp.ones((4,), jnp.float32))
        leaves, treedef = jax.tree_util.tree_flatten((args, {}))
        closed = memo._trace_closed(f, leaves, treedef, tuple(range(len(leaves))))

        # Live call
        wrapped = memoize_jaxpr(f)
        live_exc = None
        try:
            wrapped(*args)
        except (MemoImpurityError, MemoDonationError) as e:
            live_exc = e

        # Reference screen
        ref_exc = _reference_screen(closed, tuple(range(len(leaves))))

        assert live_exc is not None and ref_exc is not None
        assert type(live_exc) is type(ref_exc)
        assert str(live_exc) == str(ref_exc)

    def test_ac9_f_don_screen_equivalence(self):
        """AC-9: reference screen output matches live screen for F_don."""

        def f(x):
            a = jax.jit(lambda y: y * 2, donate_argnums=0)(x)
            b = jax.device_put(a, donate=True)
            return jax.jit(lambda z: jax.jit(lambda w: w + 1, donate_argnums=0)(z))(b)

        args = (jnp.ones((4,), jnp.float32),)
        leaves, treedef = jax.tree_util.tree_flatten((args, {}))
        closed = memo._trace_closed(f, leaves, treedef, tuple(range(len(leaves))))

        # Live call
        wrapped = memoize_jaxpr(f)
        live_exc = None
        try:
            wrapped(*args)
        except (MemoImpurityError, MemoDonationError) as e:
            live_exc = e

        # Reference screen
        ref_exc = _reference_screen(closed, tuple(range(len(leaves))))

        assert live_exc is not None and ref_exc is not None
        assert type(live_exc) is type(ref_exc)
        assert str(live_exc) == str(ref_exc)
        if isinstance(live_exc, MemoDonationError):
            assert live_exc.sites == ref_exc.sites
            assert str(live_exc).startswith(
                "Function rejected by donation screen (spec §4.2 item 6): "
            )

    def test_ac9_f_both_screen_equivalence(self):
        """AC-9: reference screen output matches live screen for F_both."""

        def f(x):
            return jax.jit(lambda y: y * 2, donate_argnums=0)(x) + jax.random.uniform(
                jax.random.key(0), x.shape
            )

        args = (jnp.ones((4,), jnp.float32),)
        leaves, treedef = jax.tree_util.tree_flatten((args, {}))
        closed = memo._trace_closed(f, leaves, treedef, tuple(range(len(leaves))))

        # Live call
        wrapped = memoize_jaxpr(f)
        live_exc = None
        try:
            wrapped(*args)
        except (MemoImpurityError, MemoDonationError) as e:
            live_exc = e

        # Reference screen
        ref_exc = _reference_screen(closed, tuple(range(len(leaves))))

        assert live_exc is not None and ref_exc is not None
        assert type(live_exc) is type(ref_exc)
        assert str(live_exc) == str(ref_exc)

    def test_ac10_key_equivalence_fixture1(self):
        """AC-10: key equivalence for fixture 1."""

        def g(*args, **kwargs):
            return args[0] * 1.0

        wrapped = memoize_jaxpr(g, policy=MemoPolicy(salt="s1"))
        core = wrapped._memo_core

        args = (np.ones((3,), np.float32),)
        kwargs = {}
        digest = core._ensure_screened(args, kwargs)

        # Test that reference key equals live key
        ref_key = _reference_build_key(core, digest, args, kwargs)
        live_key = core.build_key(digest, args, kwargs)
        assert ref_key == live_key

        # Test that key is in cache after one call
        wrapped(*args, **kwargs)
        assert ref_key in core.cache

    def test_ac10_key_equivalence_fixture2(self):
        """AC-10: key equivalence for fixture 2."""

        def g(*args, **kwargs):
            return args[0] * 1.0

        wrapped = memoize_jaxpr(g, policy=MemoPolicy(salt="s1"))
        core = wrapped._memo_core

        args = (jnp.ones((2, 2), jnp.float32), 3, 2.5, -0.0, True, "é", b"\x00")
        kwargs = {}
        digest = core._ensure_screened(args, kwargs)

        ref_key = _reference_build_key(core, digest, args, kwargs)
        live_key = core.build_key(digest, args, kwargs)
        assert ref_key == live_key

        wrapped(*args, **kwargs)
        assert ref_key in core.cache

    def test_ac10_key_equivalence_fixture3(self):
        """AC-10: key equivalence for fixture 3."""

        def g(*args, **kwargs):
            return args[0] * 1.0

        wrapped = memoize_jaxpr(g, policy=MemoPolicy(salt="s1"))
        core = wrapped._memo_core

        args = (jnp.ones((2,), jnp.float32),)
        kwargs = {"a": [jnp.zeros((2,)), 1], "b": {"c": "s"}}
        digest = core._ensure_screened(args, kwargs)

        ref_key = _reference_build_key(core, digest, args, kwargs)
        live_key = core.build_key(digest, args, kwargs)
        assert ref_key == live_key

        wrapped(*args, **kwargs)
        assert ref_key in core.cache

    def test_ac11_fits_default_int_x64_off(self):
        """AC-11: _fits_default_int boundaries with x64 off."""
        # x64 off: default is int32
        assert memo._fits_default_int(2**31 - 1) is True
        assert memo._fits_default_int(-(2**31)) is True
        assert memo._fits_default_int(2**31) is False
        assert memo._fits_default_int(-(2**31) - 1) is False

    def test_ac11_fits_default_int_x64_on(self):
        """AC-11: _fits_default_int boundaries with x64 on."""
        x64_ctx = jax.enable_x64 if hasattr(jax, "enable_x64") else None
        if x64_ctx is None:
            pytest.skip("jax.enable_x64 context manager unavailable in this jax version")

        with x64_ctx():
            # x64 on: default is int64
            assert memo._fits_default_int(2**31) is True
            assert memo._fits_default_int(2**63 - 1) is True
            assert memo._fits_default_int(2**63) is False
            assert memo._fits_default_int(-(2**63) - 1) is False

    def test_golden_leaf_digests(self):
        """AC-18: golden leaf digests are unchanged."""
        fixtures = [
            (
                2.5,
                b"float(2.5)",
                "9fc15d7f6df8db99bc0dcde0447c9bf5ed6bc2aaccf3f4cfd46a28d4b9f72d98",
            ),
            (
                -0.0,
                b"float(-0.0)",
                "9494dcf6094b912eff00023aaee43d28149697b0be0765cd0e091cad8323e21e",
            ),
            (
                1e300,
                b"float(1e+300)",
                "361300e047c47d05ece511ef57c019d3c57f58cdafe81273922b1867ebeb431e",
            ),
            (
                5e-324,
                b"float(5e-324)",
                "cd705304fa1f0663015d3bb87b4d645c4d8ea0f4d162f46f385e274d6b8d9ce9",
            ),
            (
                3,
                b"int(3)",
                "3038d0e4056117cc63ca144b5436861036059825628dddffee5a4c3c0250d829",
            ),
            (
                True,
                b"bool(True)",
                "8fe0a14cc6b15c2a958819427d423b6ac1ea2b67fbb074167c1de6582629156c",
            ),
        ]

        for leaf, preimage, expected_hex in fixtures:
            # Verify preimage hash matches expected hex
            assert hashlib.sha256(preimage).hexdigest() == expected_hex

            # Verify live memo._leaf_digest gives same hex
            h = hashlib.sha256()
            memo._leaf_digest(leaf, h)
            assert h.hexdigest() == expected_hex

            # Verify reference leaf digest gives same hex
            h_ref = hashlib.sha256()
            _reference_leaf_digest(leaf, h_ref)
            assert h_ref.hexdigest() == expected_hex

    def test_ac16_keyword_only_default_pin(self):
        """AC-16: keyword-only parameter with default is screened on first call, not wrap time."""

        def f(*, a=1.0):
            return jax.random.uniform(jax.random.key(0), (4,)) * a

        # Wrapping should not raise
        wrapped = memoize_jaxpr(f)

        # screen_latched_error should be None (not screened at wrap time)
        assert wrapped._memo_core.screen_latched_error is None

        # First call with no arguments should raise MemoImpurityError
        with pytest.raises(MemoImpurityError):
            wrapped()

        # Error should latch: second call also raises
        assert wrapped._memo_core.screen_latched_error is not None
        with pytest.raises(MemoImpurityError):
            wrapped()

    def test_ac16_positional_default_pin(self):
        """AC-16: positional parameter with default is screened on first call, not wrap time."""

        def g(x=1.0):
            return jax.random.uniform(jax.random.key(0), (4,)) * x

        # Wrapping should not raise
        wrapped = memoize_jaxpr(g)

        # screen_latched_error should be None (not screened at wrap time)
        assert wrapped._memo_core.screen_latched_error is None

        # First call with no arguments should raise MemoImpurityError
        with pytest.raises(MemoImpurityError):
            wrapped()

        # Error should latch: second call also raises
        assert wrapped._memo_core.screen_latched_error is not None
        with pytest.raises(MemoImpurityError):
            wrapped()


def counting_wrapper(orig):
    """Counting wrapper that increments a counter when called."""
    count = {"value": 0}

    def wrapper(*args, **kwargs):
        count["value"] += 1
        return orig(*args, **kwargs)

    wrapper._count = count
    return wrapper


class TestSprint260923HotPath:
    """AC-13 and AC-14: tree flattening and int bounds caching optimizations."""

    def test_ac13_flatten_once_per_call(self, monkeypatch):
        """AC-13: Flatten once per call (not three times: flatten, leaves, structure)."""

        def f(x):
            return x * 2.0

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(copy_on_return=False, spot_check_every=0))
        x = np.ones((4,), np.float32)

        # Warm-up miss
        wrapped(x)

        # Install counting wrappers after warm-up
        orig_flatten = jax.tree_util.tree_flatten
        orig_leaves = jax.tree_util.tree_leaves
        orig_structure = jax.tree_util.tree_structure

        flatten_wrapper = counting_wrapper(orig_flatten)
        leaves_wrapper = counting_wrapper(orig_leaves)
        structure_wrapper = counting_wrapper(orig_structure)

        monkeypatch.setattr(jax.tree_util, "tree_flatten", flatten_wrapper)
        monkeypatch.setattr(jax.tree_util, "tree_leaves", leaves_wrapper)
        monkeypatch.setattr(jax.tree_util, "tree_structure", structure_wrapper)

        # One cache hit
        result = wrapped(x)

        # Verify result is correct
        assert result.shape == x.shape
        assert np.allclose(result, x * 2.0)

        # Verify call counts: flatten==1, leaves==0, structure==0
        assert flatten_wrapper._count["value"] == 1
        assert leaves_wrapper._count["value"] == 0
        assert structure_wrapper._count["value"] == 0

    def test_ac14_cached_int_bounds(self, monkeypatch):
        """AC-14: Cached int bounds — canonicalize_dtype not called on hit."""

        def f(x, n):
            return x + n

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(copy_on_return=False, spot_check_every=0))
        x = np.ones((4,), np.float32)
        n = 3

        # Warm-up miss
        wrapped(x, n)

        # Install counting wrapper for canonicalize_dtype after warm-up
        orig_canonicalize = jax.dtypes.canonicalize_dtype
        canonicalize_wrapper = counting_wrapper(orig_canonicalize)
        monkeypatch.setattr(jax.dtypes, "canonicalize_dtype", canonicalize_wrapper)

        # One cache hit on (x, 3)
        result = wrapped(x, n)

        # Verify result is correct
        assert result.shape == x.shape
        assert np.allclose(result, x + n)

        # Verify canonicalize_dtype was NOT called during the cache hit
        # (already cached during classification in _mode_token)
        assert canonicalize_wrapper._count["value"] == 0


class TestSprint260923NanKeys:
    """AC-17 and AC-18: NaN payloads are distinct keys; non-NaN floats unchanged."""

    def test_ac17_nan_payloads_distinct_static_mode(self):
        """AC-17 item 1: NaN payloads with different bits give different static tokens."""
        import struct

        # Control: assert the two NaN bit patterns are different
        nan_a = struct.unpack("<d", bytes.fromhex("000000000000f87f"))[0]
        nan_b = struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0]
        assert struct.pack("<d", nan_a) != struct.pack("<d", nan_b)

        # Both are NaN
        assert math.isnan(nan_a)
        assert math.isnan(nan_b)

        # _classify_leaf should return different descriptors in STATIC mode
        desc_a = memo._classify_leaf(nan_a, "STATIC")
        desc_b = memo._classify_leaf(nan_b, "STATIC")
        assert desc_a != desc_b

    def test_ac17_nan_payloads_cache_misses(self):
        """AC-17 item 2: Different NaN payloads cause cache misses, not hits."""
        import struct

        nan_a = struct.unpack("<d", bytes.fromhex("000000000000f87f"))[0]
        nan_b = struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0]
        assert struct.pack("<d", nan_a) != struct.pack("<d", nan_b)

        def f(x, s):
            return x * s

        wrapped = memoize_jaxpr(f, policy=MemoPolicy(copy_on_return=False, spot_check_every=0))
        x = jnp.ones((4,), jnp.float32)

        # First call with nan_a
        wrapped(x, nan_a)

        # Second call with nan_b (different NaN payload)
        wrapped(x, nan_b)

        # Third call with nan_a again (should hit)
        wrapped(x, nan_a)

        # Check stats
        stats = wrapped.memo_get_stats()
        assert stats["misses"] == 2, f"Expected 2 misses, got {stats['misses']}"
        assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"

    def test_ac17_nan_payloads_float_subclass(self):
        """AC-17 item 3: Float subclass NaN payloads are distinct in both modes."""
        import struct

        class F(float):
            pass

        nan_a = F(struct.unpack("<d", bytes.fromhex("000000000000f87f"))[0])
        nan_b = F(struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0])
        assert struct.pack("<d", float(nan_a)) != struct.pack("<d", float(nan_b))

        # In ABSTRACT mode, float subclass goes through _static_exact_token
        desc_a_abs = memo._classify_leaf(nan_a, "ABSTRACT")
        desc_b_abs = memo._classify_leaf(nan_b, "ABSTRACT")
        assert desc_a_abs != desc_b_abs

        # In STATIC mode, float subclass also goes through _static_exact_token
        desc_a_stat = memo._classify_leaf(nan_a, "STATIC")
        desc_b_stat = memo._classify_leaf(nan_b, "STATIC")
        assert desc_a_stat != desc_b_stat

    def test_ac18_non_nan_floats_unchanged(self):
        """AC-18: Non-NaN float classifications are unchanged."""
        # Classification for 1.5 in STATIC mode
        desc = memo._classify_leaf(1.5, "STATIC")
        assert desc == ("dyn", ("static", "float", "1.5"))

        # -0.0 and 0.0 should differ
        desc_neg_zero = memo._classify_leaf(-0.0, "STATIC")
        desc_pos_zero = memo._classify_leaf(0.0, "STATIC")
        assert desc_neg_zero != desc_pos_zero

    def test_ac18_golden_leaf_digests(self):
        """AC-18: Golden leaf digests from T2 still pass with NaN keying."""
        # Test data from AC-18 table
        test_cases = [
            (
                2.5,
                b"float(2.5)",
                "9fc15d7f6df8db99bc0dcde0447c9bf5ed6bc2aaccf3f4cfd46a28d4b9f72d98",
            ),
            (
                -0.0,
                b"float(-0.0)",
                "9494dcf6094b912eff00023aaee43d28149697b0be0765cd0e091cad8323e21e",
            ),
            (
                1e300,
                b"float(1e+300)",
                "361300e047c47d05ece511ef57c019d3c57f58cdafe81273922b1867ebeb431e",
            ),
            (
                5e-324,
                b"float(5e-324)",
                "cd705304fa1f0663015d3bb87b4d645c4d8ea0f4d162f46f385e274d6b8d9ce9",
            ),
            (3, b"int(3)", "3038d0e4056117cc63ca144b5436861036059825628dddffee5a4c3c0250d829"),
            (
                True,
                b"bool(True)",
                "8fe0a14cc6b15c2a958819427d423b6ac1ea2b67fbb074167c1de6582629156c",
            ),
        ]

        for leaf, preimage, expected_hexdigest in test_cases:
            # Verify preimage matches expected hash
            computed_hash = hashlib.sha256(preimage).hexdigest()
            assert computed_hash == expected_hexdigest, f"Preimage hash mismatch for {leaf}"

            # Verify live _leaf_digest gives the same hash
            h = hashlib.sha256()
            memo._leaf_digest(leaf, h)
            assert h.hexdigest() == expected_hexdigest, f"Live _leaf_digest mismatch for {leaf}"
