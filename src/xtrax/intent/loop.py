"""Experimental. Static-length and conditional loops for intent resolution.

A static loop is one :func:`jax.lax.while_loop` on the primal path. JAX
differentiation traces that loop as :func:`jax.lax.scan`. A conditional
loop is a plain :func:`jax.lax.while_loop`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import jax
import jax.numpy as jnp

from xtrax.intent.errors import IntentError
from xtrax.intent.plan import SiteKey
from xtrax.intent.sites import Session, activate, call_label, current_session
from xtrax.tiling.strategy import Scan, StaticWhile

_REGISTER: ContextVar[bool] = ContextVar("xtrax.intent.loop.register", default=True)

_Step = Callable[[tuple[Any, ...], Any, Any], Any]
_Strategy = Scan | StaticWhile


@contextmanager
def _suspend_registration():
    """Hide the active session while a closure is converted for its record."""
    token = _REGISTER.set(False)
    try:
        yield
    finally:
        _REGISTER.reset(token)


def _active_session() -> Session | None:
    """Return the session that should record this call, if registration is on."""
    if not _REGISTER.get():
        return None
    return current_session()


def _example_x(xs: Any) -> Any:
    """First slice of ``xs`` on axis 0, or ``None`` when ``xs`` is omitted."""
    if xs is None:
        return None
    return jax.tree_util.tree_map(lambda leaf: leaf[0], xs)


def _index_xs(xs: Any, index: Any) -> Any:
    """Slice ``xs`` at a traced ``index``, or ``None`` when ``xs`` is omitted."""
    if xs is None:
        return None
    return jax.tree_util.tree_map(
        lambda leaf: jax.lax.dynamic_index_in_dim(leaf, index, axis=0, keepdims=False),
        xs,
    )


def _convert_scan(
    body: Callable[..., Any], init: Any, xs: Any
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Closure-convert a scan body ``(carry, x) -> (carry, y)``."""
    converted, consts = jax.closure_convert(body, init, _example_x(xs))
    return converted, tuple(consts)


def _convert_cond(
    body: Callable[..., Any], init: Any
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Closure-convert a conditional body ``carry -> carry``."""
    converted, consts = jax.closure_convert(body, init)
    return converted, tuple(consts)


def _retrace(body: Callable[..., Any]) -> Callable[..., Any]:
    """Return a new callable so a later closure conversion traces ``body`` again."""

    def _body(*args: Any) -> Any:
        return body(*args)

    return _body


def _apply_scan(fn: Callable[..., Any]) -> _Step:
    """Adapt a closure-converted scan body to ``(consts, carry, x)``."""

    def _step(consts: tuple[Any, ...], carry: Any, x: Any) -> Any:
        return fn(carry, x, *consts)

    return _step


def _apply_cond(fn: Callable[..., Any]) -> Callable[[tuple[Any, ...], Any], Any]:
    """Adapt a closure-converted conditional body to ``(consts, carry)``."""

    def _step(consts: tuple[Any, ...], carry: Any) -> Any:
        return fn(carry, *consts)

    return _step


def _avals(init: Any, xs: Any) -> tuple[jax.ShapeDtypeStruct, ...]:
    """Abstract values of ``init`` leaves, then ``xs`` leaves when ``xs`` is given."""
    tree = init if xs is None else (init, xs)
    return tuple(
        jax.ShapeDtypeStruct(tuple(leaf.shape), leaf.dtype)
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _static_length(length: int | None, xs: Any) -> int:
    """Trip count from ``length``, or from the leading axis of ``xs``."""
    if length is not None:
        return length
    leaf = jax.tree_util.tree_leaves(xs)[0]
    return int(leaf.shape[0])


def _allocate_ys(y_spec: Any, length: int) -> Any:
    """Zeros whose leading axis is ``length`` and whose leaves match ``y_spec``."""
    return jax.tree_util.tree_map(
        lambda spec: jnp.zeros((length, *spec.shape), spec.dtype),
        y_spec,
    )


def _write_ys(ys: Any, y: Any, index: Any) -> Any:
    """Write one step's ``y`` into axis 0 of ``ys`` at ``index``."""
    return jax.tree_util.tree_map(
        lambda buf, val: jax.lax.dynamic_update_index_in_dim(buf, val, index, 0),
        ys,
        y,
    )


def _scan(step: _Step, consts: tuple[Any, ...], init: Any, xs: Any, length: int) -> Any:
    """``lax.scan`` of ``step`` for ``length`` trips, with or without ``xs``."""

    def _body(carry: Any, x: Any) -> Any:
        return step(consts, carry, x)

    if xs is None:
        return jax.lax.scan(_body, init, None, length=length)
    return jax.lax.scan(_body, init, xs, length=length)


def _while_collect(step: _Step, consts: tuple[Any, ...], init: Any, xs: Any, length: int) -> Any:
    """Static ``while_loop`` that stacks per-step outputs on axis 0."""
    y_spec = jax.eval_shape(
        lambda carry, x: step(consts, carry, x)[1],
        init,
        _example_x(xs),
    )
    ys0 = _allocate_ys(y_spec, length)
    if length == 0:
        return init, ys0

    def _body(state: tuple[Any, Any, Any]) -> tuple[Any, Any, Any]:
        index, carry, ys = state
        carry2, y = step(consts, carry, _index_xs(xs, index))
        return index + 1, carry2, _write_ys(ys, y, index)

    _index, carry, ys = jax.lax.while_loop(
        lambda state: state[0] < length,
        _body,
        (jnp.int32(0), init, ys0),
    )
    return carry, ys


def _execute(
    step: _Step,
    consts: tuple[Any, ...],
    init: Any,
    xs: Any,
    length: int,
    strategy: _Strategy,
) -> Any:
    """Run the primal strategy. ``Scan`` scans; every other accepted strategy whiles."""
    if isinstance(strategy, Scan):
        return _scan(step, consts, init, xs, length)
    return _while_collect(step, consts, init, xs, length)


def _primal_strategy(session: Session | None, key: SiteKey | None, length: int) -> _Strategy:
    """Decision primal, or ``StaticWhile`` when the session has no decision.

    Raises
    ------
    IntentError
        The decision's primal is neither ``StaticWhile`` nor ``Scan``.
    """
    if session is None or key is None:
        return StaticWhile(length)
    decision = session.decision_for(key)
    if decision is None:
        return StaticWhile(length)
    primal = decision.primal
    if isinstance(primal, (StaticWhile, Scan)):
        return primal
    raise IntentError(
        f"loop primal strategy must be StaticWhile or Scan, got {type(primal).__name__}",
        site=str(key),
        path="primal",
        constraint=repr(primal),
    )


def _static_runner(
    strategy: _Strategy,
    session: Session | None,
    key: SiteKey | None,
    body: Callable[..., Any],
) -> Callable[..., Any]:
    """Build the ``custom_jvp`` whose nondiff arguments are the step and the length."""

    @functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
    def _run(
        step: _Step,
        length: int,
        consts: tuple[Any, ...],
        init: Any,
        xs: Any,
    ) -> Any:
        if session is not None and key is not None:
            with activate(session), session.scope(key, "primal"):
                _convert_scan(_retrace(body), init, xs)
        return _execute(step, consts, init, xs, length, strategy)

    @_run.defjvp
    def _run_jvp(
        step: _Step,
        length: int,
        primals: tuple[Any, ...],
        tangents: tuple[Any, ...],
    ) -> Any:
        if session is None or key is None:
            return jax.jvp(
                lambda consts, init, xs: _scan(step, consts, init, xs, length),
                primals,
                tangents,
            )
        _consts, init, xs = primals
        with activate(session), session.scope(key, "diff"):
            fn, _hoisted = _convert_scan(_retrace(body), init, xs)
            live = _apply_scan(fn)
            return jax.jvp(
                lambda consts, init, xs: _scan(live, consts, init, xs, length),
                primals,
                tangents,
            )

    return _run


def _loop_static(
    body: Callable[..., Any],
    init: Any,
    xs: Any,
    length: int | None,
    session: Session | None,
    label: str,
) -> Any:
    """Static-length loop, registering one site when ``session`` is active."""
    resolved = _static_length(length, xs)
    suspend = _suspend_registration() if session is not None else nullcontext()
    with suspend:
        fn, consts = _convert_scan(body, init, xs)
    key: SiteKey | None = None
    if session is not None:
        key = session.enter_site("loop", label, _avals(init, xs), resolved, fn)
    strategy = _primal_strategy(session, key, resolved)
    step = _apply_scan(fn)
    run = _static_runner(strategy, session, key, body)
    return run(step, resolved, consts, init, xs)


def _loop_cond(
    body: Callable[..., Any],
    init: Any,
    cond: Callable[..., Any],
    session: Session | None,
    label: str,
) -> Any:
    """Conditional loop: a plain ``while_loop``, with a site when a session is active."""
    suspend = _suspend_registration() if session is not None else nullcontext()
    with suspend:
        fn, consts = _convert_cond(body, init)
    if session is not None:
        key = session.enter_site("loop", label, _avals(init, None), None, fn)
        _primal_strategy(session, key, 0)
        with activate(session), session.scope(key, "primal"):
            fn, consts = _convert_cond(body, init)
    step = _apply_cond(fn)
    return jax.lax.while_loop(cond, lambda carry: step(consts, carry), init)


def loop(
    body: Callable[..., Any],
    init: Any,
    *,
    length: int | None = None,
    xs: Any = None,
    cond: Callable[..., Any] | None = None,
    name: str | None = None,
) -> Any:
    """Run ``body`` as a static loop or as a conditional ``while_loop``.

    The static form (``xs`` and/or ``length``) matches :func:`jax.lax.scan`:
    ``body(carry, x) -> (carry, y)`` and the result is ``(carry, ys)``. The
    primal writes ``ys`` with :func:`jax.lax.dynamic_update_index_in_dim` into
    buffers from :func:`jax.eval_shape` of one step, unless the session's
    primal decision is :class:`~xtrax.tiling.strategy.Scan`, which scans.
    Differentiation traces :func:`jax.lax.scan` after re-entering the captured
    session on the ``diff`` path. With no session, or with a session that has
    no decision, the primal is :class:`~xtrax.tiling.strategy.StaticWhile` and
    the derivative is :class:`~xtrax.tiling.strategy.Scan`.

    The conditional form (``cond``, with neither ``length`` nor ``xs``) matches
    :func:`jax.lax.while_loop`: ``body(carry) -> carry`` and the result is the
    final carry. It registers a ``loop`` site with ``length=None``.

    Parameters
    ----------
    body : callable
        Step function. Static form: ``body(carry, x) -> (carry, y)``.
        Conditional form: ``body(carry) -> carry``. Closed-over values are
        hoisted with :func:`jax.closure_convert`.
    init : pytree of jax.Array
        Initial carry.
    length : int or None
        Static trip count. ``None`` when only ``xs`` or ``cond`` is given.
    xs : pytree of jax.Array or None
        Per-step inputs stacked on axis 0. Omitted from the site when ``None``.
    cond : callable or None
        Continuation predicate ``cond(carry) -> bool`` for the conditional form.
    name : str or None
        Site label. ``None`` records the caller as ``file:line``.

    Returns
    -------
    pytree
        ``(carry, ys)`` for the static form, or the final carry for the
        conditional form.

    Raises
    ------
    ValueError
        ``cond`` is combined with ``length`` or ``xs``, or none of ``length``,
        ``xs``, and ``cond`` is given.
    IntentError
        The session's primal strategy is neither ``StaticWhile`` nor ``Scan``.
        The error names the site, the path ``primal``, and the strategy.
    """
    if cond is not None and (length is not None or xs is not None):
        raise ValueError("loop cond form does not take length or xs")
    if cond is None and length is None and xs is None:
        raise ValueError("loop requires length, xs, or cond")
    label = call_label(name, stacklevel=2)
    session = _active_session()
    if cond is not None:
        return _loop_cond(body, init, cond, session, label)
    return _loop_static(body, init, xs, length, session, label)
