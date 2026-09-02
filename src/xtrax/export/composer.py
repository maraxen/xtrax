"""Fold a BatchPlan into one traceable callable.

xtrax provides per-axis primitives and leaves the wiring to the caller: there is
no single call that dispatches a whole multi-strategy BatchPlan, because domain
libraries want to control that nesting themselves. ``jax.export.export`` needs
exactly the composer that leaves behind, so this module supplies it.

Scope here is one axis. Multi-axis nesting follows the certified vmap-of-scan
recipe and lands separately.

This module is deliberately unaware of ``AxisBoundary.materialize``: it composes
whatever boundaries dict it is handed. Sink-stripping lives one layer up, in
``xtrax.export.pipeline``, so a caller reaching the composer directly still gets
the real, un-stripped sink.
"""

from collections.abc import Callable, Mapping
from typing import Any

from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.executor import execute_map_axis, execute_scan_axis
from xtrax.tiling.dispatch import axis_dispatch

__all__ = [
    "ComposerError",
    "MultiAxisCompositionError",
    "UnsupportedStrategyError",
    "build_traceable_callable",
    "compose_single_axis",
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


def build_traceable_callable(
    fn: Callable[..., Any],
    plan: Any,
    axis_boundaries: Mapping[str, AxisBoundary] | None = None,
    *,
    scan_init: Any = None,
) -> Callable[..., Any]:
    """Fold a single-axis BatchPlan into one traceable callable.

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
        ComposerError: If the plan has no axes, or more than one.
        UnsupportedStrategyError: If the axis's strategy cannot be routed.
    """
    decisions = list(plan.decisions)
    if not decisions:
        msg = "plan has no axis decisions; nothing to compose."
        raise ComposerError(msg)
    if len(decisions) > 1:
        names = ", ".join(repr(d.spec.name) for d in decisions)
        msg = (
            f"plan has {len(decisions)} axes ({names}); this composer handles one. "
            f"Multi-axis composition follows the certified vmap-of-scan recipe and "
            f"is not available here."
        )
        raise ComposerError(msg)

    decision = decisions[0]
    boundaries = axis_boundaries or {}
    return compose_single_axis(
        fn,
        decision,
        boundaries.get(decision.spec.name),
        scan_init=scan_init,
    )
