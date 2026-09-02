"""Fold a BatchPlan into one traceable callable.

xtrax provides per-axis primitives and leaves the wiring to the caller: there is
no single call that dispatches a whole multi-strategy BatchPlan, because domain
libraries want to control that nesting themselves. ``jax.export.export`` needs
exactly the composer that leaves behind, so this module supplies it.

Scope here is one axis, plus the one certified two-axis shape: an outer ``Vmap``
axis wrapping an inner ``Scan`` axis.

This module is deliberately unaware of ``AxisBoundary.materialize``: it composes
whatever boundaries dict it is handed. Sink-stripping lives one layer up, in
``xtrax.export.pipeline``, so a caller reaching the composer directly still gets
the real, un-stripped sink.
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.executor import ExecutorError, execute_map_axis, execute_scan_axis
from xtrax.tiling.dispatch import axis_dispatch

__all__ = [
    "ComposerError",
    "MultiAxisCompositionError",
    "UnsupportedStrategyError",
    "build_traceable_callable",
    "compose_single_axis",
    "compose_vmap_of_scan",
]

_SUPPORTED = "Vmap/SafeMap/Scan/DedupGather"


class ComposerError(Exception):
    """Raised when a plan cannot be folded into a traceable callable."""


class UnsupportedStrategyError(ComposerError):
    """The axis's strategy is not one this composer routes.

    Bucket is host-tier: pad with bucketize() before the boundary. WhileCarry has
    an unbounded trip count: convert it to a Scan with a static length.
    """


class MultiAxisCompositionError(ComposerError):
    """Composing this plan would need a literal vmap around a lane-dependent axis.

    Wraps the underlying xtrax.stages.executor.ExecutorError; see its
    "Nesting: vmap-of-scan" docstring for the certified alternative.
    """


def compose_single_axis(
    step_fn: Callable[..., Any],
    decision: Any,
    boundary: AxisBoundary | None = None,
    *,
    scan_init: Any = None,
) -> Callable[[Any], Any]:
    """Build a traceable ``xs -> out`` callable for one axis decision.

    Args:
        step_fn: Per-element function. For a Scan axis this is instead a
            transition ``(carry, x) -> (carry, y)``.
        decision: An AxisDecision, or any structurally-compatible object whose
            ``strategy``'s class name matches a supported strategy.
        boundary: Optional AxisBoundary for this axis.
        scan_init: Initial carry, required when the strategy is Scan. Falls back
            to the strategy's own ``init`` when not given.

    Returns:
        A callable suitable for ``jax.jit`` / ``jax.export.export``.

    Raises:
        UnsupportedStrategyError: For host-tier or unbounded strategies.
        ComposerError: For a Scan axis with no initial carry available.
    """
    strategy = getattr(decision, "strategy", None)
    if strategy is None:
        msg = (
            f"axis decision {getattr(getattr(decision, 'spec', None), 'name', '?')!r} has "
            f"no strategy, so there is nothing to compose. Supported strategies are "
            f"{_SUPPORTED}."
        )
        raise UnsupportedStrategyError(msg)

    strategy_name = type(strategy).__name__

    if strategy_name in ("Vmap", "SafeMap"):

        def _run_map(xs: Any) -> Any:
            return execute_map_axis(step_fn, xs, strategy, boundary)

        return _run_map

    if strategy_name == "Scan":
        init = scan_init if scan_init is not None else getattr(strategy, "init", None)
        if init is None:
            msg = (
                "Scan axis needs an initial carry: pass scan_init= or set "
                "Scan(init=...). jax.lax.scan cannot infer it."
            )
            raise ComposerError(msg)

        def _run_scan(xs: Any) -> Any:
            # execute_scan_axis returns (final_carry, ys); the exported artifact
            # returns the (optionally fused) ys, matching the map-axis shape.
            _final_carry, ys = execute_scan_axis(step_fn, init, xs, boundary)
            return ys

        return _run_scan

    if strategy_name == "DedupGather":
        # All three dedup phases are in-trace gathers/scatters. The indices are
        # host NumPy computed at plan-build time, so they ride along as closure
        # constants at their static k_bucket shape.
        def _run_dedup(xs: Any) -> Any:
            return axis_dispatch(strategy, step_fn, xs)

        return _run_dedup

    msg = (
        f"strategy {strategy_name!r} cannot be composed into a traceable callable. "
        f"Supported strategies are {_SUPPORTED}. Bucket is host-tier (pad with "
        f"bucketize() before the boundary); WhileCarry has an unbounded trip count "
        f"(convert to Scan with a static length)."
    )
    raise UnsupportedStrategyError(msg)


def _apply_fuse(ys: Any, boundary: AxisBoundary | None) -> Any:
    """Apply ``boundary.fuse`` once to the stacked ``ys``, mirroring the executor's own."""
    if boundary is not None and boundary.fuse is not None:
        return boundary.fuse(ys)
    return ys


def _init_is_batched(init: Any, outer_n: int) -> bool:
    """True if every carry leaf carries the outer axis in its leading dimension.

    This is the batched-shape recipe's precondition: the outer axis is baked into
    the carry, so no ``jax.vmap`` is needed to iterate lanes. An empty carry
    cannot carry the axis, so it is False.
    """
    leaves = jax.tree_util.tree_leaves(init)
    if not leaves:
        return False
    return all(jnp.ndim(leaf) >= 1 and jnp.shape(leaf)[0] == outer_n for leaf in leaves)


def compose_vmap_of_scan(
    fn: Callable[[Any, Any], tuple[Any, Any]],
    outer_decision: Any,
    inner_decision: Any,
    outer_boundary: AxisBoundary | None = None,
    inner_boundary: AxisBoundary | None = None,
    *,
    scan_init: Any = None,
) -> Callable[[Any], Any]:
    """Compose an outer ``Vmap`` axis wrapping an inner ``Scan`` axis.

    Two routes, matching the two outcomes ``tests/stages/test_nested_ordering.py``
    certifies. Which one applies is decided by the initial carry's shape:

    **Batched-shape (the recommended recipe).** When every ``scan_init`` leaf
    carries the outer axis in its leading dimension, the outer axis is already
    baked into the carry, so no ``jax.vmap`` is needed at all. This builds a bare
    ``jax.lax.scan`` here and applies ``tap``/``sink`` inline, deliberately *not*
    calling ``execute_scan_axis``: ``TestBatchedShapeVmapOfScanPreservesOrder``
    scans a hand-written transition via bare ``jax.lax.scan`` and never exercises
    that helper in this shape, so routing through it would be an uncertified
    extrapolation wearing a certified badge.

    **Literal vmap.** When the carry is *not* batched, lanes can only be iterated
    by an actual ``jax.vmap``, so this defers to ``execute_map_axis``. If the
    inner axis has an ordered ``Tap``/``Sink`` whose value depends on the lane,
    JAX refuses and the executor raises ``ExecutorError``; that is re-raised as
    ``MultiAxisCompositionError`` with the message preserved rather than
    re-worded, so the certified guidance text has exactly one home. When the sunk
    value does not depend on the lane, this route composes fine -- the narrow
    success ``TestLiteralVmapOfScanOrdering::test_lane_independent_ordering_succeeds``
    certifies.

    The inner axis's per-step ``xs`` on the literal-vmap route comes from the
    plan's own declared cardinality, since each lane scans the whole inner axis.

    Boundaries belong on the inner axis. On the batched-shape route the outer
    axis has no per-lane call site at all -- it is a dimension, not a loop -- so
    an outer ``tap``/``sink``/``fuse`` has nowhere to fire and is refused rather
    than silently dropped.

    ``sink`` receives the exact value returned as ``y``, because both come from
    the single ``y`` that ``fn`` returned -- the same structural guarantee
    ``executor.py``'s ``_wrapped_transition`` gives, and what ``materialize``
    depends on: ``export_pipeline`` strips the sink and reads the returned ``ys``
    instead, so the two must be the same value. A transition that sinks one thing
    and returns another would break that silently, which is why this wrapper
    computes ``y`` once rather than letting the caller wire the sink itself.

    Args:
        fn: The scan transition ``(carry, x) -> (carry, y)``. On the batched-shape
            route it sees the whole batched carry at once, so its per-step logic
            must be ordinary broadcasting array ops.
        outer_decision: The outer axis's AxisDecision (``Vmap`` strategy).
        inner_decision: The inner axis's AxisDecision (``Scan`` strategy).
        outer_boundary: Must be absent or empty; see above.
        inner_boundary: Optional AxisBoundary applied per scan step.
        scan_init: Initial carry. Batched to the outer axis's cardinality selects
            the recommended route.

    Returns:
        A callable suitable for ``jax.jit`` / ``jax.export.export``.

    Raises:
        ComposerError: If no initial carry is available.
        MultiAxisCompositionError: If the outer axis carries a boundary, or if
            the literal-vmap route hits the lane-dependent ordering restriction.
    """
    outer_strategy = getattr(outer_decision, "strategy", None)
    inner_strategy = getattr(inner_decision, "strategy", None)

    if outer_strategy is None:
        outer_name = getattr(getattr(outer_decision, "spec", None), "name", "?")
        msg = (
            f"outer axis {outer_name!r} has no strategy, so there is nothing to map "
            f"lanes with. Supported strategies are {_SUPPORTED}."
        )
        raise UnsupportedStrategyError(msg)

    if outer_boundary is not None and (
        outer_boundary.fuse is not None
        or outer_boundary.tap is not None
        or outer_boundary.sink is not None
    ):
        outer_name = getattr(getattr(outer_decision, "spec", None), "name", "?")
        msg = (
            f"outer Vmap axis {outer_name!r} carries a boundary, but the certified "
            f"vmap-of-scan recipe bakes that axis into the carry's shape, leaving no "
            f"per-lane point for a fuse/tap/sink to fire. Attach the boundary to the "
            f"inner Scan axis instead."
        )
        raise MultiAxisCompositionError(msg)

    init = scan_init if scan_init is not None else getattr(inner_strategy, "init", None)
    if init is None:
        msg = (
            "multi-axis composition needs an initial carry: pass scan_init= or set "
            "Scan(init=...). For the recommended batched-shape recipe the carry must "
            "already carry the outer axis in its leading dimension."
        )
        raise ComposerError(msg)

    outer_n = getattr(getattr(outer_decision, "spec", None), "cardinality", None)
    inner_n = getattr(getattr(inner_decision, "spec", None), "cardinality", None)

    if outer_n is not None and _init_is_batched(init, outer_n):

        def _batched_transition(carry: Any, x: Any) -> tuple[Any, Any]:
            carry, y = fn(carry, x)
            if inner_boundary is not None:
                if inner_boundary.tap is not None:
                    y = inner_boundary.tap(y)
                if inner_boundary.sink is not None:
                    inner_boundary.sink(y)
            return carry, y

        def _run_batched(xs: Any) -> Any:
            _final_carry, ys = jax.lax.scan(_batched_transition, init, xs)
            return _apply_fuse(ys, inner_boundary)

        return _run_batched

    def _run_literal_vmap(xs: Any) -> Any:
        inner_xs = jnp.arange(inner_n) if inner_n is not None else None
        if inner_xs is None:
            msg = (
                "the inner Scan axis declares no cardinality, so the per-lane scan has "
                "no length to run for. Give the inner AxisSpec a cardinality, or batch "
                "scan_init to the outer axis so the recommended recipe applies."
            )
            raise ComposerError(msg)

        def _lane(lane_init: Any) -> Any:
            final_carry, _ys = execute_scan_axis(fn, lane_init, inner_xs, inner_boundary)
            return final_carry

        try:
            return execute_map_axis(_lane, xs, outer_strategy, None)
        except ExecutorError as exc:
            # Preserve the executor's own message verbatim: it is the certified
            # guidance text, and re-wording it here would give it two homes that
            # could drift apart.
            raise MultiAxisCompositionError(str(exc)) from exc

    return _run_literal_vmap


def build_traceable_callable(
    fn: Callable[..., Any],
    plan: Any,
    axis_boundaries: Mapping[str, AxisBoundary] | None = None,
    *,
    scan_init: Any = None,
) -> Callable[..., Any]:
    """Fold a BatchPlan into one traceable callable.

    Handles a one-axis plan, or the certified two-axis Vmap-over-Scan shape via
    ``compose_vmap_of_scan``. Deeper or differently-shaped nestings are refused.

    ``fn`` is always the transition used for a Scan axis. ``Scan.transition`` is
    read only by the eager ``xtrax.tiling.dispatch`` path and is never consulted
    here, so a caller who sets both gets ``fn`` exported -- which can differ from
    what an eager run of the same plan would do.

    Args:
        fn: Per-element function, or a Scan transition.
        plan: A BatchPlan, or any object exposing ``decisions``.
        axis_boundaries: Name-keyed boundaries. Composed as given; this function
            does not know about ``materialize``.
        scan_init: Initial carry for a Scan axis, threaded through.

    Returns:
        A callable suitable for ``jax.jit`` / ``jax.export.export``.

    Raises:
        ComposerError: If the plan has no axes.
        MultiAxisCompositionError: If the plan has more than two axes, a two-axis
            shape other than Vmap-over-Scan, or hits the lane-dependent ordering
            restriction.
        UnsupportedStrategyError: If the axis's strategy cannot be routed.
    """
    decisions = list(plan.decisions)
    if not decisions:
        msg = "plan has no axis decisions; nothing to compose."
        raise ComposerError(msg)
    boundaries = axis_boundaries or {}

    if len(decisions) == 2:
        outer, inner = decisions
        outer_name = type(getattr(outer, "strategy", None)).__name__
        inner_name = type(getattr(inner, "strategy", None)).__name__
        if outer_name != "Vmap" or inner_name != "Scan":
            msg = (
                f"two-axis plan is {outer_name}-over-{inner_name}; the only certified "
                f"multi-axis shape is an outer Vmap axis wrapping an inner Scan axis "
                f"(see tests/stages/test_nested_ordering.py). Compose other nestings "
                f"yourself from the per-axis primitives."
            )
            raise MultiAxisCompositionError(msg)
        return compose_vmap_of_scan(
            fn,
            outer,
            inner,
            boundaries.get(outer.spec.name),
            boundaries.get(inner.spec.name),
            scan_init=scan_init,
        )

    if len(decisions) > 2:
        names = ", ".join(repr(d.spec.name) for d in decisions)
        msg = (
            f"plan has {len(decisions)} axes ({names}); this composer handles one, or "
            f"two as the certified Vmap-over-Scan shape. Deeper nesting has no "
            f"certification harness behind it, so it is refused rather than guessed at."
        )
        raise MultiAxisCompositionError(msg)

    decision = decisions[0]
    return compose_single_axis(
        fn,
        decision,
        boundaries.get(decision.spec.name),
        scan_init=scan_init,
    )
