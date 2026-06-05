import equinox as eqx
import jax.numpy as jnp
import pytest

from xtrax.safety.manager import SafetyManager, with_safety


def test_safety_manager_disabled_returns_identity():
    """SafetyManager(enabled=False) + wrap returns fn unchanged (strict identity)."""
    def dummy_fn(x):
        return x + 1.0

    mgr = SafetyManager(enabled=False, check_nans=False, check_infs=False)
    wrapped = mgr.wrap(dummy_fn)

    # Strict identity check — must be the exact same object
    assert wrapped is dummy_fn


def test_safety_manager_is_eqx_module():
    """SafetyManager is a valid equinox Module."""
    mgr = SafetyManager(enabled=True, check_nans=True, check_infs=True)
    assert isinstance(mgr, eqx.Module)


def test_safety_manager_enabled_detects_nan():
    """SafetyManager(enabled=True) wrapping a NaN-producing fn raises on call."""
    def produces_nan(x):
        # Division by zero creates inf (caught by float_checks)
        return jnp.array(1.0) / jnp.array(0.0)

    mgr = SafetyManager(enabled=True, check_nans=True, check_infs=True)
    wrapped = mgr.wrap(produces_nan)

    # Wrapped function should raise when called
    with pytest.raises(Exception):  # checkify raises JaxRuntimeError
        wrapped(0.0)


def test_with_safety_alias():
    """with_safety is an alias for manager.wrap."""
    def dummy_fn(x):
        return x + 1.0

    mgr = SafetyManager(enabled=False, check_nans=False, check_infs=False)
    wrapped = with_safety(dummy_fn, mgr)

    # Should also return identity when disabled
    assert wrapped is dummy_fn
