"""Experimental. Static-length and conditional loops for intent resolution.

A static loop is one :func:`jax.lax.while_loop` on the primal path. JAX
differentiation traces that loop as :func:`jax.lax.scan`. A conditional
loop is a plain :func:`jax.lax.while_loop`.
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
    decision_for,
)
from xtrax.intent.errors import IntentError
from xtrax.intent.plan import SiteKey
from xtrax.intent.sites import Session, _leaf_struct, activate, call_label
from xtrax.tiling.strategy import Scan, StaticWhile, WhileCarry

_Step = Callable[[tuple[Any, ...], Any, Any], Any]
_Strategy = Scan | StaticWhile


def _example_slice(leaf: Any) -> Any:
    """One row of ``leaf``, or zeros when the leading axis is empty."""
    if leaf.shape[0] == 0:
        return jnp.zeros(leaf.shape[1:], leaf.dtype)
    return leaf[0]


def _example_x(xs: Any) -> Any:
    """Slice ``xs`` at index 0, or ``None`` when ``xs`` is omitted."""
    if xs is None:
        return None
    return jax.tree_util.tree_map(_example_slice, xs)


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
    return tuple(_leaf_struct(leaf) for leaf in jax.tree_util.tree_leaves(tree))


def _static_length(length: int | None, xs: Any) -> int:
    """Trip count from ``length``, or from the leading axis of ``xs``.

    When ``xs`` is given, every leaf's leading size must equal that count.
    The ``ValueError`` text matches :func:`jax.lax.scan`, and it is raised
    before the body is traced.
    """
    if xs is None:
        if length is None:
            raise ValueError("loop requires length, xs, or cond")
        return length
    sizes = [int(leaf.shape[0]) for leaf in jax.tree_util.tree_leaves(xs)]
    if length is not None:
        if any(size != length for size in sizes):
            raise ValueError(
                f"scan got `length` argument of {length} which disagrees with "
                f"leading axis sizes {sizes}."
            )
        return length
    unique = set(sizes)
    if len(unique) > 1:
        rendered = ", ".join(str(size) for size in sizes)
        raise ValueError(f"scan got values with different leading axis sizes: {rendered}.")
    if not sizes:
        raise ValueError("scan got no values to scan over and `length` not provided.")
    return sizes[0]


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


def _ys_buffer(step: _Step, consts: tuple[Any, ...], init: Any, xs: Any, length: int) -> Any:
    """Zeros shaped like ``length`` stacked per-step outputs of ``step``."""
    y_spec = jax.eval_shape(
        lambda carry, x: step(consts, carry, x)[1],
        init,
        _example_x(xs),
    )
    return _allocate_ys(y_spec, length)


def _scan_collect(step: _Step, consts: tuple[Any, ...], init: Any, xs: Any, length: int) -> Any:
    """``lax.scan`` that writes per-step outputs into a carried buffer.

    The values equal :func:`_scan`. The layout matches :func:`_while_collect`:
    under ``vmap`` the buffer is batched on axis 0, where stacked scan outputs
    are batched on axis 1. A ``StaticWhile`` primal's JVP rule needs the same
    layout as its primal, or ``custom_jvp`` batching under a chunked
    ``lax.map`` returns ``ys`` with time and batch axes interleaved.
    """
    ys0 = _ys_buffer(step, consts, init, xs, length)
    if length == 0:
        return init, ys0

    def _body(state: tuple[Any, Any, Any], x: Any) -> tuple[tuple[Any, Any, Any], None]:
        index, carry, ys = state
        carry2, y = step(consts, carry, x)
        return (index + 1, carry2, _write_ys(ys, y, index)), None

    (_index, carry, ys), _ = jax.lax.scan(_body, (jnp.int32(0), init, ys0), xs, length=length)
    return carry, ys


def _while_collect(step: _Step, consts: tuple[Any, ...], init: Any, xs: Any, length: int) -> Any:
    """Static ``while_loop`` that stacks per-step outputs on axis 0."""
    ys0 = _ys_buffer(step, consts, init, xs, length)
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
    decision = decision_for(session, key)
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


def _cond_primal(session: Session | None, key: SiteKey | None) -> None:
    """Accept an undecided cond loop, or a decision whose primal is ``WhileCarry``.

    Raises
    ------
    IntentError
        A decision exists and its primal is not ``WhileCarry``. ``site`` is
        the key, ``path`` is ``"primal"``, and ``constraint`` is ``repr`` of
        that primal.
    """
    if session is None or key is None:
        return
    decision = decision_for(session, key)
    if decision is None:
        return
    primal = decision.primal
    if isinstance(primal, WhileCarry):
        return
    raise IntentError(
        f"loop primal strategy must be WhileCarry, got {type(primal).__name__}",
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
                fn, _hoisted = _convert_scan(_retrace(body), init, xs)
            # This retrace is the primal that runs. ``consts`` is the custom_jvp
            # argument (closed-over tangents), matching the JVP rule; ``_hoisted``
            # is the retrace's own copy and is not the differentiated input.
            live = _apply_scan(fn)
            _store_running_fn(session, key, fn)
            return _execute(live, consts, init, xs, length, strategy)
        return _execute(step, consts, init, xs, length, strategy)

    @_run.defjvp
    def _run_jvp(
        step: _Step,
        length: int,
        primals: tuple[Any, ...],
        tangents: tuple[Any, ...],
    ) -> Any:
        _consts, init, xs = primals
        if session is None or key is None:
            live = step
        else:
            with activate(session), session.scope(key, "diff"):
                fn, _hoisted = _convert_scan(_retrace(body), init, xs)
            # ``_hoisted`` is this conversion's copy of the closed-over values.
            # The differentiated inputs are ``consts``; ``fn`` is applied to those.
            live = _apply_scan(fn)
        # The derivative is always a scan. A ``StaticWhile`` primal collects
        # ``ys`` in a carried buffer, so the scan does too (``_scan_collect``);
        # stacked scan outputs batch on a different axis and break custom_jvp
        # batching under a chunked ``lax.map``.
        collect = _scan if isinstance(strategy, Scan) else _scan_collect
        return jax.jvp(
            lambda cs, carry, inputs: collect(live, cs, carry, inputs, length),
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
    if session is None:
        fn, consts = _convert_cond(body, init)
    else:
        key = session.enter_site("loop", label, _avals(init, None), None, None)
        _cond_primal(session, key)
        with activate(session), session.scope(key, "primal"):
            fn, consts = _convert_cond(_retrace(body), init)
        _store_running_fn(session, key, fn)
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
    final carry. It registers a ``loop`` site with ``length=None``. With no
    decision the primal still runs :func:`jax.lax.while_loop`. A recorded
    decision must be :class:`~xtrax.tiling.strategy.WhileCarry`.

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
        ``xs``, and ``cond`` is given. An explicit ``length`` that disagrees
        with a leading size of ``xs`` raises the same ``ValueError`` as
        :func:`jax.lax.scan`.
    IntentError
        A static loop's primal strategy is neither ``StaticWhile`` nor
        ``Scan``, or a conditional loop's primal strategy is not
        ``WhileCarry``. The error names the site, the path ``primal``, and
        the strategy.
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
