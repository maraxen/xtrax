"""Closure conversion that hoists every closed-over tracer.

:func:`jax.closure_convert` hoists only closed-over values that can carry a
tangent. A closed-over integer or boolean tracer (tangent dtype ``float0``)
stays inside the converted function. ``loop`` and ``map`` pass that function
to :func:`jax.custom_jvp` as a nondiff argument, so under ``jit`` the tracer
outlives its trace: a ``grad`` through a site inside a traced ``lax.cond``
fails to lower with "No constant handler for type: DynamicJaxprTracer".
:func:`hoist_closure` returns every closed-over tracer as an explicit argument
instead, and integer and boolean ones then get ``float0`` tangents. Concrete
closed-over values stay inside, as with :func:`jax.closure_convert`.
"""

from collections.abc import Callable
from typing import Any

import jax
from jax.core import Tracer
from jax.extend.core import ClosedJaxpr, jaxpr_as_fun


def hoist_closure(
    fun: Callable[..., Any], *example_args: Any
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Convert ``fun`` so that every closed-over tracer is a trailing argument.

    Parameters
    ----------
    fun : callable
        Function of ``example_args``' pytree structure.
    *example_args : pytree of jax.Array
        Example inputs that fix the traced shapes and pytree structure.

    Returns
    -------
    converted : callable
        ``converted(*args, *consts)`` evaluates ``fun(*args)`` with the
        closed-over values replaced by ``consts``.
    consts : tuple
        Every closed-over tracer, of any dtype, in the order ``converted``
        expects them.

    Raises
    ------
    TypeError
        ``converted`` is called with arguments whose pytree structure differs
        from ``example_args``.
    """
    closed, out_shape = jax.make_jaxpr(fun, return_shape=True)(*example_args)
    in_tree = jax.tree_util.tree_structure(example_args)
    out_tree = jax.tree_util.tree_structure(out_shape)
    jaxpr = closed.jaxpr
    all_consts = list(closed.consts)
    # Concrete values cannot outlive a trace, so they stay inside, as with closure_convert.
    traced = [isinstance(c, Tracer) for c in all_consts]
    consts = tuple(c for c, t in zip(all_consts, traced, strict=True) if t)

    def converted(*args_and_consts: Any) -> Any:
        split = len(args_and_consts) - len(consts)
        args, hoisted = args_and_consts[:split], iter(args_and_consts[split:])
        flat, tree = jax.tree_util.tree_flatten(tuple(args))
        if tree != in_tree:
            raise TypeError(
                "The inputs to the converted closure must have the same pytree "
                f"structure as the example arguments. Expected {in_tree}, got {tree}"
            )
        merged = [next(hoisted) if t else c for c, t in zip(all_consts, traced, strict=True)]
        out = jaxpr_as_fun(ClosedJaxpr(jaxpr, merged))(*flat)
        return jax.tree_util.tree_unflatten(out_tree, out)

    return converted, consts
