"""One recursive walk of a JAX jaxpr.

``xtrax.export.safety`` and ``xtrax.profiling.loop_scaling`` each used to keep
a private copy of this walk. Nested primitives (``pjit`` / ``jit``, ``scan``,
``while`` cond and body, ``cond`` branches, ``custom_jvp`` / ``custom_vjp``,
``remat``) hide their equations one level down; a top-level-only walk reports
a false all-clear. This module is the single public walker.

jax is not imported here. Callers pass a ``Jaxpr`` (or anything with ``.eqns``
and, for a closed wrapper, ``.jaxpr``) and the walk is structural.
"""

from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["iter_jaxpr_eqns", "sub_jaxprs"]


def sub_jaxprs(value: Any) -> Iterator[Any]:  # noqa: ANN401 -- jaxpr containers are untyped
    """Yield every jaxpr stored in ``value``, one level deep.

    ``value`` may be an equation's ``params`` mapping, a jaxpr, a closed
    jaxpr (unwrapped via ``.jaxpr`` when that attribute is a different
    object), or a nested list/tuple of those. Containers are opened; the
    equations *inside* a yielded jaxpr are not. Callers that need the full
    tree use :func:`iter_jaxpr_eqns`.

    Args:
        value: A params mapping or one parameter value.

    Yields:
        Open jaxprs (objects with ``.eqns``), in encounter order.
    """
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from sub_jaxprs(item)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from sub_jaxprs(item)
        return
    inner = getattr(value, "jaxpr", None)
    # A Jaxpr's own ``.jaxpr`` is itself. Unwrapping only a *different* object
    # is what turns a closed wrapper into the walkable core, once.
    if inner is not None and inner is not value and hasattr(inner, "eqns"):
        yield inner
        return
    if hasattr(value, "eqns"):
        yield value


def iter_jaxpr_eqns(closed_jaxpr: Any) -> Iterator[Any]:  # noqa: ANN401
    """Yield every equation in ``closed_jaxpr``, including nested sub-jaxprs.

    Accepts a closed jaxpr or a bare jaxpr. Recurses through every nested
    jaxpr carried on an equation's params: ``pjit`` / ``jit`` call jaxprs,
    ``scan`` bodies, ``while`` cond and body, ``cond`` branches,
    ``custom_jvp`` / ``custom_vjp`` call jaxprs, and ``remat`` / checkpoint.

    Args:
        closed_jaxpr: The traced program. A closed wrapper is unwrapped once.

    Yields:
        Equations, parent before the equations of its sub-jaxprs.
    """
    jaxpr = closed_jaxpr
    inner = getattr(closed_jaxpr, "jaxpr", None)
    if inner is not None and inner is not closed_jaxpr and hasattr(inner, "eqns"):
        jaxpr = inner
    for eqn in jaxpr.eqns:
        yield eqn
        for sub in sub_jaxprs(eqn.params):
            yield from iter_jaxpr_eqns(sub)
