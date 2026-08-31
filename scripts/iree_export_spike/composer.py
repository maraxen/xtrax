"""Fold a BatchPlan into one traceable callable.

This is the piece xtrax does not ship. ``execute_map_axis``, ``execute_scan_axis``
and ``axis_dispatch`` are called from tests only -- never from another ``src/xtrax``
module -- because xtrax provides per-axis primitives and leaves the wiring to the
domain library. The shipped skill says so directly:

    "There is no single call that dispatches a whole multi-strategy BatchPlan --
    the caller iterates plan.decisions and routes each axis by strategy type."
    -- agent_assets/skills/using-xtrax/references/tiling.md:411

Export needs exactly that missing composer: something to hand to
``jax.export.export(jax.jit(fn))``.

Spike scope: **one axis**, plus an optional ``Fuse``. Multi-axis nesting is a
follow-up -- the certified vmap-of-scan recipe is in
``tests/stages/test_nested_ordering.py:102-125``, and the lane-dependent-ordering
counter-example that must NOT be copied is at ``:147-160``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from scripts.iree_export_spike.export_safety import assert_plan_export_safe
from xtrax.stages.executor import execute_map_axis, execute_scan_axis
from xtrax.tiling.dispatch import axis_dispatch


class ComposerError(Exception):
    """Raised when a plan cannot be folded into a traceable callable."""


def compose_single_axis(
    step_fn: Callable[..., Any],
    decision: Any,
    boundary: Any = None,
    *,
    scan_init: Any = None,
) -> Callable[[Any], Any]:
    """Build a traceable ``xs -> out`` callable for one axis decision.

    Args:
        step_fn: Per-element function. For a ``Scan`` axis this must instead be a
            transition ``(carry, x) -> (carry, y)``.
        decision: An ``AxisDecision``.
        boundary: Optional ``AxisBoundary``. Only ``fuse`` is honoured -- a
            ``tap``/``sink`` would pierce the export boundary and is rejected
            upstream by ``export_safety``.
        scan_init: Initial carry, required when the strategy is ``Scan``.

    Returns:
        A callable suitable for ``jax.jit`` / ``jax.export.export``.

    Raises:
        ComposerError: For host-tier or not-yet-supported strategies.
    """
    strategy = getattr(decision, "strategy", None)
    strategy_name = type(strategy).__name__

    if strategy_name in ("Vmap", "SafeMap"):

        def _run_map(xs: Any) -> Any:
            return execute_map_axis(step_fn, xs, strategy, boundary)

        return _run_map

    if strategy_name == "Scan":
        if scan_init is None:
            scan_init = getattr(strategy, "init", None)
        if scan_init is None:
            raise ComposerError(
                "Scan axis needs an initial carry: pass scan_init= or set "
                "Scan(init=...). jax.lax.scan cannot infer it."
            )
        init = scan_init

        def _run_scan(xs: Any) -> Any:
            # execute_scan_axis returns (final_carry, ys); the exported artifact
            # returns the (optionally fused) ys, matching the map-axis shape.
            _final_carry, ys = execute_scan_axis(step_fn, init, xs, boundary)
            return ys

        return _run_scan

    if strategy_name == "DedupGather":
        # All three dedup phases are in-trace gathers/scatters (dispatch.py:169-178).
        # The *indices* are host NumPy computed at plan-build time, so they ride along
        # as closure constants at their static k_bucket shape.
        def _run_dedup(xs: Any) -> Any:
            return axis_dispatch(strategy, step_fn, xs)

        return _run_dedup

    raise ComposerError(
        f"strategy {strategy_name!r} cannot be composed into a traceable callable. "
        "Bucket is host-tier (pad with bucketize() before the boundary); WhileCarry "
        "has an unbounded trip count (convert to Scan with a static length)."
    )


def compose_exportable(
    step_fn: Callable[..., Any],
    plan: Any,
    axis_boundaries: Mapping[str, Any] | None = None,
    *,
    scan_init: Any = None,
) -> Callable[[Any], Any]:
    """Gate a plan for export safety, then fold it into one traceable callable.

    Args:
        step_fn: Per-element function (or scan transition).
        plan: A ``BatchPlan``.
        axis_boundaries: Name-keyed boundaries.
        scan_init: Initial carry for a ``Scan`` axis.

    Raises:
        ExportUnsafeError: If any axis cannot cross the export boundary.
        ComposerError: If the plan is not single-axis (spike limitation).
    """
    decisions = list(getattr(plan, "decisions", ()))
    assert_plan_export_safe(decisions, axis_boundaries)

    if len(decisions) != 1:
        raise ComposerError(
            f"spike composes a single axis; got {len(decisions)}. Multi-axis nesting "
            "is a follow-up (see tests/stages/test_nested_ordering.py:102-125)."
        )

    decision = decisions[0]
    name = getattr(getattr(decision, "spec", None), "name", None)
    boundary = (axis_boundaries or {}).get(name)
    return compose_single_axis(step_fn, decision, boundary, scan_init=scan_init)


__all__ = ["ComposerError", "compose_exportable", "compose_single_axis"]
