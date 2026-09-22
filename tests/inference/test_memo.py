"""Tests for xtrax.inference.memo — spec 260825 §4.2 (AC3-AC6, AC9, AC12-AC20)
and spec 260922 §3 donation, both directions (AC-1 to AC-17)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.inference.memo import (
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
        k1 = f1._memo_core.build_key((x,), {})
        k2 = f2._memo_core.build_key((x,), {})
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
        (mirroring _screen_jaxpr's own traversal) never descends into the
        tuple-valued `branches` param, so it misses the donation entirely.
        Demonstrates that recursing into tuple/list values (_iter_subjaxprs)
        is load-bearing for AC-4.
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


class TestDonationFailOpen:
    """T1 §3.3: the two documented, unfixed fail-open admission paths.

    #5214 and #5215 are NOT fixed in this sprint (§3.4 "Not in T1"); these
    tests PIN the fail-open behaviour with a strict xfail so a future fix (or
    accidental regression that starts catching them) is visible either way
    (AC-15, AC-16).
    """

    @pytest.mark.xfail(
        strict=True,
        raises=pytest.fail.Exception,
        reason="#5214: the screen runs once, keyed off the first call's abstract "
        "signature; a shape-dependent donation on a later call is not re-screened.",
    )
    def test_ac15_shape_dependent_donation_not_rescreened(self):
        from xtrax.inference.memo import MemoDonationError

        def f(x):
            if x.shape[0] > 4:
                return jax.jit(lambda y: y * 2, donate_argnums=0)(x)
            return x * 2

        wrapped = memoize_jaxpr(f)
        wrapped(jnp.ones((4,), jnp.float32))  # 1st call: shape (4,), no donation
        with pytest.raises(MemoDonationError):
            wrapped(jnp.ones((8,), jnp.float32))  # 2nd call: shape (8,), donates

    @pytest.mark.xfail(
        strict=True,
        raises=pytest.fail.Exception,
        reason="#5215: the screen traces probe(*args) only, never kwargs; a "
        "donation gated on a kwarg is never observed by the screen.",
    )
    def test_ac16_kwarg_dependent_donation_not_traced(self):
        from xtrax.inference.memo import MemoDonationError

        def f(x, *, fast=False):
            if fast:
                return jax.jit(lambda y: y * 2, donate_argnums=0)(x)
            return x * 2

        wrapped = memoize_jaxpr(f)
        with pytest.raises(MemoDonationError):
            wrapped(jnp.ones((4,), jnp.float32), fast=True)


class TestDonationDocs:
    def test_ac17_docs_mention_donation_and_fail_open_ids(self):
        import re
        from pathlib import Path

        docs_path = Path(__file__).resolve().parents[2] / "docs" / "api" / "inference.md"
        content = docs_path.read_text()
        for pattern in (r"memoize_jaxpr", r"donat", r"#5214", r"#5215"):
            assert re.search(pattern, content), f"missing {pattern!r} in docs/api/inference.md"


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
