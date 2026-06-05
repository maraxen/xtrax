"""Safety and checkification management for JAX functions."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.experimental.checkify as checkify


class SafetyManager(eqx.Module):
    """Manages safety checks (NaN/Inf detection) for JAX computations.

    Attributes:
        enabled: If False, wrap() returns fn unchanged (zero overhead).
        check_nans: Include NaN checks in float_checks.
        check_infs: Include Inf checks in float_checks.
    """

    enabled: bool = eqx.field(static=True)
    check_nans: bool = eqx.field(static=True)
    check_infs: bool = eqx.field(static=True)

    def wrap(self, fn: Callable) -> Callable:
        """Wrap a function with optional checkify-based safety checks.

        If enabled=False, returns fn unchanged (strict identity, zero overhead).
        If enabled=True, returns a wrapper that applies jax.experimental.checkify
        and raises Python exceptions on host when NaN/Inf are detected.

        Args:
            fn: The function to wrap.

        Returns:
            If enabled=False: fn itself (identity).
            If enabled=True: A wrapped callable that checks for errors after execution.
        """
        if not self.enabled:
            # Strict identity: return fn unchanged, no wrapper overhead
            return fn

        # Build jit-compiled checkified fn ONCE (not per call)
        # float_checks covers NaN + Inf + overflow
        _checked_jit = jax.jit(checkify.checkify(fn, errors=checkify.float_checks))

        def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
            err, result = _checked_jit(*args, **kwargs)
            err.throw()  # outside jit — raises Python exception on host
            return result

        return wrapped_fn


def with_safety(fn: Callable, manager: SafetyManager) -> Callable:
    """Convenience alias for manager.wrap(fn).

    Args:
        fn: The function to wrap.
        manager: The SafetyManager to use.

    Returns:
        The wrapped function.
    """
    return manager.wrap(fn)
