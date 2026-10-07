"""Refuse to treat a run-twice self-comparison as parity."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["SelfParityError", "assert_distinct_callables"]


class SelfParityError(ValueError):
    """The candidate and the oracle are the same callable.

    Running one function twice measures determinism. Parity needs a second
    implementation.
    """


def _same_callable(candidate: Callable[..., Any], oracle: Callable[..., Any]) -> bool:
    if candidate is oracle:
        return True
    candidate_func = getattr(candidate, "__func__", None)
    oracle_func = getattr(oracle, "__func__", None)
    if candidate_func is None or oracle_func is None:
        return False
    return candidate_func is oracle_func and getattr(candidate, "__self__", None) is getattr(
        oracle, "__self__", None
    )


def assert_distinct_callables(candidate: Callable[..., Any], oracle: Callable[..., Any]) -> None:
    """Raise when ``candidate`` and ``oracle`` are one callable.

    Object identity counts, and so does a bound method: ``obj.method`` builds a
    new object on every access, so two lookups of the same method on the same
    instance still count as one function. Two ``functools.partial`` wrappers are
    distinct unless they are the same object.

    Args:
        candidate: Callable under test.
        oracle: Independent reference callable.

    Raises:
        SelfParityError: The two arguments are the same callable.
    """
    if _same_callable(candidate, oracle):
        raise SelfParityError(
            "candidate and oracle are the same callable; a run-twice "
            "self-comparison measures determinism, not parity"
        )
