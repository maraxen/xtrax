"""Tests for the fit pre-check (spec 261016 X6): xtrax.profiling.fit.fit_precheck."""

import jax
import jax.numpy as jnp

import xtrax.profiling.fit as fit_mod
from xtrax.profiling.fit import FitCheck, fit_precheck


def _tanh(x):
    return jnp.tanh(x) * 2.0


def test_small_cpu_function_fits_or_is_unknown_never_false(monkeypatch):
    # Real CPU reports no memory limit; pin that so the verdict is deterministic.
    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (None, "cpu"))
    check = fit_precheck(_tanh, (jnp.zeros((64, 128), jnp.float32),))
    assert isinstance(check, FitCheck)
    assert check.fits is None
    assert check.limit_bytes is None
    assert check.required_bytes is None
    assert "limit" in check.reason and "compile skipped" in check.reason


def test_no_device_limit_skips_compiling(monkeypatch):
    # A host loop at production extent unrolls once per element under jit; with
    # no limit to compare against the pre-check must not pay that compile.
    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (None, "cpu"))
    traced = []

    def records_trace(x):
        traced.append(1)
        return x

    fit_precheck(records_trace, (jnp.zeros((4,), jnp.float32),))
    assert traced == []


def test_small_function_passes_against_a_generous_real_limit():
    check = fit_precheck(_tanh, (jnp.zeros((64, 128), jnp.float32),))
    assert check.fits is not False


def test_lowering_never_allocates_the_real_inputs(monkeypatch):
    # 4 GiB of float32 would not be allocatable in a test; lowering must accept
    # the shape struct alone and report sizes without materialising it.
    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (2**40, "gpu"))
    huge = jax.ShapeDtypeStruct((2**20, 2**10), jnp.float32)
    check = fit_precheck(_tanh, (huge,))
    assert check.fits is not False
    if check.required_bytes is not None:
        assert check.required_bytes >= 2**20 * 2**10 * 4


def test_unjittable_function_returns_unknown_with_reason(monkeypatch):
    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (2**40, "gpu"))

    def not_jittable(x):
        return float(x)  # concretises a tracer: lowering raises

    check = fit_precheck(not_jittable, (jnp.zeros((4,), jnp.float32),))
    assert check.fits is None
    assert check.required_bytes is None
    assert "not jittable" in check.reason


def test_limit_smaller_than_required_gives_false(monkeypatch):
    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (1024, "gpu"))
    check = fit_precheck(_tanh, (jnp.zeros((64, 128), jnp.float32),))
    assert check.fits is False
    assert check.limit_bytes == 1024
    assert check.required_bytes > 1024


def test_limit_larger_than_required_gives_true(monkeypatch):
    monkeypatch.setattr(fit_mod, "_device_limit_bytes", lambda: (2**40, "gpu"))
    check = fit_precheck(_tanh, (jnp.zeros((64, 128), jnp.float32),))
    assert check.fits is True


def test_missing_memory_analysis_is_unknown_not_a_crash():
    class _NoAnalysis:
        def memory_analysis(self):
            return None

    class _PartialAnalysis:
        argument_size_in_bytes = 8

    class _Partial:
        def memory_analysis(self):
            return _PartialAnalysis()

    class _Raises:
        def memory_analysis(self):
            raise NotImplementedError("backend has no analysis")

    assert fit_mod._required_bytes(_NoAnalysis())[0] is None
    required, why = fit_mod._required_bytes(_Partial())
    assert required is None and "output_size_in_bytes" in why
    required, why = fit_mod._required_bytes(_Raises())
    assert required is None and "NotImplementedError" in why


def test_memory_analysis_sums_argument_output_and_temp():
    class _Analysis:
        argument_size_in_bytes = 10
        output_size_in_bytes = 20
        temp_size_in_bytes = 30

    class _Compiled:
        def memory_analysis(self):
            return _Analysis()

    assert fit_mod._required_bytes(_Compiled()) == (60, "")
