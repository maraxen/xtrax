"""CLI verb for tiling strategy planning (E2 plan verb)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn
from xtrax.cli.shapes import parse_shapes
from xtrax.inference.api import infer_bundle
from xtrax.tiling.plan import BatchPlan, BatchPlanner
from xtrax.tiling.roles import AmbiguousAxisError


@dataclass
class PlanArgs:
    """Arguments for the plan verb.

    Attributes:
        fn: Import-path string to the function to plan. Format: 'module.path:symbol'.
            The function will be loaded lazily only when run_plan is invoked.
        shapes: Space-separated shape specification string. Format: 'name=(d0,d1,...)<dtype> ...'.
            Each name corresponds to a positional argument of the function, in order.
    """

    fn: str
    shapes: str = ""


def plan_from_fn(fn: Any, shapes: str) -> BatchPlan:
    """Infer a bundle for *fn* and plan its tiling strategy.

    Shared by `run_plan` (bare --fn/--shapes) and `run_graph_plan` (a callable sourced
    from a graph node instead of a bare import path) -- both converge on this exact
    function so a graph-sourced plan and a directly-inferred plan for the same callable +
    shapes are provably identical (T1-11, AC1).

    The shapes string order is assumed to match fn's positional-argument order: the first
    parsed shape corresponds to the first positional argument, etc.

    Args:
        fn: A resolved, traceable JAX function (from load_fn or a graph node's callable_ref).
        shapes: Space-separated shape specification string, see parse_shapes.

    Raises:
        CLIError: If BatchPlanner cannot resolve an axis role (AmbiguousAxisError), with an
            actionable message, not a raw traceback.
    """
    parsed = parse_shapes(shapes)
    abstract_inputs = list(parsed.values())
    _schema, axes = infer_bundle(fn, abstract_inputs)

    try:
        return BatchPlanner().plan(axes)
    except AmbiguousAxisError as e:
        raise CLIError(
            f"Unable to determine axis role for planning: {e}\n"
            f"Hint: decorate the function with @axis_config(AxisOverride(...)) "
            f"to resolve the axis role."
        ) from e


def run_plan(args: PlanArgs) -> None:
    """Execute the plan verb: infer bundle, plan tiling strategy, and print summary.

    Args:
        args: PlanArgs with fn and shapes strings.

    Raises:
        CLIError: On any user-facing error (bad import path, bad shapes, or
                  unresolved axis role from BatchPlanner). The error message
                  is clean and actionable, not a raw traceback.

    Example:
        >>> args = PlanArgs(fn="mylib:forward", shapes="x=(4,3)f32")
        >>> run_plan(args)
        # Prints tiling plan summary to stdout
    """
    fn = load_fn(args.fn)
    plan = plan_from_fn(fn, args.shapes)
    print_plan_summary(plan)


def print_plan_summary(plan: Any) -> None:
    """Print a human-readable summary of a tiling plan to stdout.

    Args:
        plan: A BatchPlan object (from xtrax.tiling.plan.BatchPlan).
    """
    print("Tiling Plan Summary")
    print("=" * 60)
    print(f"Number of axes: {len(plan.decisions)}")
    print()

    for i, decision in enumerate(plan.decisions, start=1):
        spec = decision.spec
        print(f"Axis {i}: {spec.name}")
        print(f"  Cardinality: {spec.cardinality}")
        print(f"  Default batch size: {spec.default_batch_size}")
        print(f"  Tile granularity: {spec.tile_granularity}")
        print(f"  Heterogeneous: {spec.heterogeneous}")
        print(f"  Dedup eligible: {spec.dedup_eligible}")
        print(f"  Role: {spec.role.value}")
        print(f"  Strategy: {type(decision.strategy).__name__}")
        print(f"  Reasoning: {decision.reasoning}")
        print()

    print("=" * 60)
