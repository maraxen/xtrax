"""Tests for Context-driven NaN/Inf numerical safety checking (#3042)."""

import jax.experimental.checkify as checkify
import jax.numpy as jnp
import pytest

from xtrax.intent import compile as intent_compile
from xtrax.intent.context import Context
from xtrax.safety.numerical import SafetyContext, check_numerical_safety


class TestContextSafety:
    """Verification controls for Context(safety=True) and check_numerical_safety."""

    def test_safe_execution_passes_cleanly(self):
        """Valid arithmetic passes without error when safety=True."""

        def safe_fn(x):
            return x * 2.0 + 1.0

        ctx = SafetyContext(safety=True)
        compiled = intent_compile(safe_fn, ctx=ctx)

        result = compiled(jnp.array([1.0, 2.0]))
        assert not jnp.isnan(result).any()

    def test_nan_division_raises_exception_when_safety_true(self):
        """Division by zero triggers an immediate host exception when safety=True."""

        def bad_fn(x, y):
            return x / y

        ctx = Context(safety=True, safety_checks=("nan", "div"))
        compiled = intent_compile(bad_fn, ctx=ctx)

        # Division by zero
        with pytest.raises((checkify.JaxRuntimeError, ValueError)):
            compiled(jnp.array(1.0), jnp.array(0.0))

    def test_unsafe_execution_returns_nan_when_safety_false(self):
        """Division by zero returns NaN or Inf without raising when safety=False."""

        def bad_fn(x, y):
            return x / y

        ctx = Context(safety=False)
        compiled = intent_compile(bad_fn, ctx=ctx)

        result = compiled(jnp.array(0.0), jnp.array(0.0))
        # Returns NaN without raising
        assert jnp.isnan(result)

    def test_check_numerical_safety_standalone(self):
        """check_numerical_safety instruments standalone callable with checkify."""

        @check_numerical_safety
        def compute(x):
            return jnp.log(x)

        # Valid input
        assert float(compute(jnp.array(1.0))) == 0.0

        # Unsafe input: log(-1) produces NaN
        with pytest.raises((checkify.JaxRuntimeError, ValueError)):
            compute(jnp.array(-1.0))
