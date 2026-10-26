"""Safety and checkification management for JAX functions."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.experimental.checkify as checkify

from xtrax._renamed import Deprecated, deprecated_getattr


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
        # float_checks is NaN checks + division-by-zero checks; it does not catch an Inf
        # that comes from overflow (jax/_src/checkify.py: float_checks = nan_checks | div_checks)
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


# Deprecated in the #6107 probe release: both names warn on access and are otherwise unchanged.
# `_SafetyManager` is the name xtrax's own code imports, so internal use never warns.
_REPLACEMENT = (
    "jax.jit(jax.experimental.checkify.checkify(fn, errors=checkify.float_checks)), "
    "then err.throw() on the host"
)
_NOTE = (
    "Note float_checks is NaN + division-by-zero, not Inf/overflow. "
    "xtrax debt #3042 tracks a Context-driven rebuild."
)
_DEPRECATED = {
    "SafetyManager": Deprecated(SafetyManager, _REPLACEMENT, _NOTE),
    "with_safety": Deprecated(with_safety, _REPLACEMENT, _NOTE),
}
_SafetyManager = SafetyManager
del SafetyManager, with_safety
__getattr__ = deprecated_getattr(__name__, _DEPRECATED)
