"""Experimental. Leading-axis map for intent resolution.

With no session, or with a session that has no decision, both the primal
and the derivative are :func:`jax.vmap`. A session decision selects
:class:`~xtrax.tiling.strategy.Vmap` or
:class:`~xtrax.tiling.strategy.ChunkedMap` independently on each path.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import jax
import jax.numpy as jnp

from xtrax.intent._trace import (
    _active_session,
    _retrace,
    _store_running_fn,
    _suspend_registration,
)
from xtrax.intent.errors import IntentError
from xtrax.intent.plan import SiteKey
from xtrax.intent.sites import Session, activate, call_label
from xtrax.tiling.strategy import ChunkedMap, Vmap
from xtrax.transforms.map import chunked_map

_Step = Callable[[tuple[Any, ...], Any], Any]
_Strategy = Vmap | ChunkedMap


def _rank(leaf: Any) -> int:
    """Rank of ``leaf``, or 0 when it has no ``ndim``."""
    ndim = getattr(leaf, "ndim", None)
    if isinstance(ndim, int):
        return ndim
    return len(getattr(leaf, "shape", ()))


def _require_leading(xs: Any) -> None:
    """Require every leaf to have rank at least 1 and the same leading size.

    Raises
    ------
    ValueError
        A leaf is scalar, or leading sizes differ. The text matches
        :func:`jax.vmap`: ``rank should be at least 1`` or
        ``inconsistent sizes``, and names each leading size.
    """
    leaves = jax.tree_util.tree_leaves(xs)
    ranks = [_rank(leaf) for leaf in leaves]
    if not leaves or any(rank < 1 for rank in ranks):
        bad = (
            None
            if not leaves
            else next(leaf for leaf, rank in zip(leaves, ranks, strict=True) if rank < 1)
        )
        shape = () if bad is None else tuple(getattr(bad, "shape", ()))
        rank = 0 if bad is None else _rank(bad)
        raise ValueError(
            "vmap was requested to map its argument along axis 0, which implies "
            f"that its rank should be at least 1, but is only {rank} "
            f"(its shape is {shape})"
        )
    sizes = [int(leaf.shape[0]) for leaf in leaves]
    if len(set(sizes)) > 1:
        rendered = ", ".join(str(size) for size in sizes)
        raise ValueError(f"vmap got inconsistent sizes for array axes to be mapped: {rendered}")


def _example_leaf(leaf: Any) -> Any:
    """One row of ``leaf``, or zeros when the leading axis is empty."""
    if leaf.shape[0] == 0:
        return jnp.zeros(leaf.shape[1:], leaf.dtype)
    return leaf[0]


def _example_xs(xs: Any) -> Any:
    """Slice ``xs`` at index 0 so closure conversion sees one element."""
    return jax.tree_util.tree_map(_example_leaf, xs)


def _convert(fun: Callable[..., Any], xs: Any) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Closure-convert ``fun`` specialized to one element of ``xs``."""
    converted, consts = jax.closure_convert(fun, _example_xs(xs))
    return converted, tuple(consts)


def _apply(fn: Callable[..., Any]) -> _Step:
    """Adapt a closure-converted map body to ``(consts, x)``."""

    def _step(consts: tuple[Any, ...], x: Any) -> Any:
        return fn(x, *consts)

    return _step


def _avals(xs: Any) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Abstract values of the leaves of ``xs``, including the leading axis."""
    return tuple(
        jax.ShapeDtypeStruct(tuple(leaf.shape), leaf.dtype)
        for leaf in jax.tree_util.tree_leaves(xs)
    )


def _accept(strategy: Any, path: str, key: SiteKey) -> _Strategy:
    """Return ``strategy`` when it is ``Vmap`` or ``ChunkedMap``.

    Raises
    ------
    IntentError
        ``strategy`` is neither ``Vmap`` nor ``ChunkedMap``. The error names
        the site, ``path``, and the strategy.
    """
    if isinstance(strategy, (Vmap, ChunkedMap)):
        return strategy
    raise IntentError(
        f"map {path} strategy must be Vmap or ChunkedMap, got {type(strategy).__name__}",
        site=str(key),
        path=path,
        constraint=repr(strategy),
    )


def _strategies(session: Session | None, key: SiteKey | None) -> tuple[_Strategy, _Strategy]:
    """Primal and diff strategies, or ``Vmap`` on both paths when undecided.

    A decision whose ``diff`` is ``None`` uses ``Vmap`` on the derivative.
    An explicit strategy that is neither ``Vmap`` nor ``ChunkedMap`` raises
    before the map runs.

    Raises
    ------
    IntentError
        The primal or the diff strategy is neither ``Vmap`` nor ``ChunkedMap``.
    """
    if session is None or key is None:
        return Vmap(), Vmap()
    decision = session.decision_for(key)
    if decision is None:
        return Vmap(), Vmap()
    primal = _accept(decision.primal, "primal", key)
    if decision.diff is None:
        return primal, Vmap()
    return primal, _accept(decision.diff, "diff", key)


def _execute(step: _Step, consts: tuple[Any, ...], xs: Any, strategy: _Strategy) -> Any:
    """Map ``step`` over axis 0. ``ChunkedMap`` chunks; ``Vmap`` vectorizes.

    A leading size of 0 uses :func:`jax.vmap`. :func:`chunked_map` already
    handles a leading size of 1.
    """

    def _apply_one(x: Any) -> Any:
        return step(consts, x)

    if isinstance(strategy, ChunkedMap):
        n = int(jax.tree_util.tree_leaves(xs)[0].shape[0])
        if n == 0:
            return jax.vmap(_apply_one)(xs)
        return chunked_map(_apply_one, xs, batch_size=strategy.batch_size)
    return jax.vmap(_apply_one)(xs)


def _runner(
    primal: _Strategy,
    diff: _Strategy,
    session: Session | None,
    key: SiteKey | None,
    fun: Callable[..., Any],
) -> Callable[..., Any]:
    """Build the ``custom_jvp`` whose nondiff argument is the hoisted step.

    When a session and key exist, the conversion that runs is a fresh
    closure conversion of ``fun`` under ``activate(session)`` and
    ``session.scope(key, path)``. It is applied to the ``custom_jvp``
    consts. The hoisted step is not called.
    """

    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def _run(step: _Step, consts: tuple[Any, ...], xs: Any) -> Any:
        if session is not None and key is not None:
            with activate(session), session.scope(key, "primal"):
                fn, _hoisted = _convert(_retrace(fun), xs)
            live = _apply(fn)
            _store_running_fn(session, key, fn)
            return _execute(live, consts, xs, primal)
        return _execute(step, consts, xs, primal)

    @_run.defjvp
    def _run_jvp(
        step: _Step,
        primals: tuple[Any, ...],
        tangents: tuple[Any, ...],
    ) -> Any:
        if session is None or key is None:
            return jax.jvp(
                lambda consts, xs: _execute(step, consts, xs, diff),
                primals,
                tangents,
            )
        _consts, xs = primals
        with activate(session), session.scope(key, "diff"):
            fn, _hoisted = _convert(_retrace(fun), xs)
            live = _apply(fn)
            return jax.jvp(
                lambda consts, xs: _execute(live, consts, xs, diff),
                primals,
                tangents,
            )

    return _run


def map(
    fun: Callable[..., Any],
    xs: Any,
    *,
    name: str | None = None,
) -> Any:
    """Map ``fun`` over the leading axis of every leaf of ``xs``.

    The result matches :func:`jax.vmap` of ``fun``. Closed-over values are
    hoisted with :func:`jax.closure_convert` and differentiated through a
    :func:`jax.custom_jvp`. With no session, or with a session that has no
    decision for this call, both paths are :class:`~xtrax.tiling.strategy.Vmap`.
    Otherwise the primal and the derivative each follow that path's decision:
    :class:`~xtrax.tiling.strategy.Vmap` or
    :class:`~xtrax.tiling.strategy.ChunkedMap`. The derivative re-enters the
    captured session in scope ``(key, "diff")``.

    When a session is active this records a site of kind ``"map"``,
    ``length=None``, and abstract values equal to the leaves of ``xs``.
    The label is ``name`` or the caller's ``file:line``.

    Parameters
    ----------
    fun : callable
        ``fun(x) -> y`` applied to each index along axis 0. ``x`` and ``y``
        may be pytrees. Closed-over values are hoisted.
    xs : pytree of jax.Array
        Values to map. Every leaf must have rank at least 1 and the same
        leading size.
    name : str or None
        Site label. ``None`` records the caller as ``file:line``.

    Returns
    -------
    pytree
        ``fun`` mapped over axis 0, with the same structure as
        ``jax.vmap(fun)(xs)``.

    Raises
    ------
    ValueError
        A leaf has rank 0, or leading sizes differ. Raised before ``fun`` is
        traced. The message matches :func:`jax.vmap` (``inconsistent sizes``
        or ``rank should be at least 1``) and names the sizes.
    IntentError
        The session's primal or diff strategy is neither ``Vmap`` nor
        ``ChunkedMap``. The error names the site, the path, and the strategy,
        and is raised before the map runs.
    """
    _require_leading(xs)
    label = call_label(name, stacklevel=2)
    session = _active_session()
    suspend = _suspend_registration() if session is not None else nullcontext()
    with suspend:
        fn, consts = _convert(fun, xs)
    key: SiteKey | None = None
    if session is not None:
        key = session.enter_site("map", label, _avals(xs), None, fn)
    primal, diff = _strategies(session, key)
    step = _apply(fn)
    run = _runner(primal, diff, session, key, fun)
    return run(step, consts, xs)
