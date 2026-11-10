"""Context-driven numerical safety checking via checkify (#3042)."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.experimental.checkify as checkify

from xtrax.intent.context import Context


@dataclass(frozen=True)
class SafetyContext(Context):
    """Context specializing on runtime numerical safety checking.

    Attributes
    ----------
    safety : bool
        Whether to enforce numerical safety checks (default: True).
    safety_checks : tuple[str, ...]
        Which checks to enable (default: ("nan", "div")).
    """

    safety: bool = True
    safety_checks: tuple[str, ...] = ("nan", "div")


def check_numerical_safety(
    fn: Callable[..., Any] | None = None,
    *,
    checks: tuple[str, ...] = ("nan", "div"),
) -> Callable[..., Any]:
    """Decorator or wrapper to instrument a callable with JAX checkify.

    Parameters
    ----------
    fn : Callable or None
        Callable to wrap.
    checks : tuple[str, ...]
        Safety checks to enforce ("nan", "div", "inf").

    Returns
    -------
    Callable
        Wrapped callable that raises an exception on host if NaN or div-by-zero occurs.
    """

    def decorator(target_fn: Callable[..., Any]) -> Callable[..., Any]:
        errors = frozenset()
        if "nan" in checks:
            errors |= checkify.nan_checks
        if "div" in checks:
            errors |= checkify.div_checks
        if "inf" in checks:
            errors |= checkify.nan_checks | checkify.div_checks

        checked_target = checkify.checkify(target_fn, errors=errors)

        @functools.wraps(target_fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            err, out = checked_target(*args, **kwargs)
            err.throw()
            return out

        return wrapper

    if fn is None:
        return decorator
    return decorator(fn)
