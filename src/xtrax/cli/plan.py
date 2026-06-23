"""CLI verb for tiling strategy planning (E2 plan verb)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn
from xtrax.cli.shapes import parse_shapes
from xtrax.inference.api import infer_bundle
from xtrax.tiling.plan import BatchPlanner
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


def run_plan(args: PlanArgs) -> None:
    """Execute the plan verb: infer bundle, plan tiling strategy, and print summary.

    This function orchestrates the full pipeline:
    1. Load the function from the import-path string.
    2. Parse the shapes string into ShapeDtypeStruct objects.
    3. Call infer_bundle to get the schema and axis specs.
    4. Call BatchPlanner.plan to get the tiling plan.
    5. Print a human-readable summary to stdout.

    The CLI shape order is assumed to match the function's positional-argument order:
    the first parsed shape corresponds to the first positional argument, etc.

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

    # Step 1: Load the function from import path.
    fn = load_fn(args.fn)

    # Step 2: Parse shape string to ShapeDtypeStruct dict.
    parsed = parse_shapes(args.shapes)

    # Step 3: Extract abstract inputs in positional order.
    # The CLI guarantees shape order matches fn positional-arg order.
    abstract_inputs = list(parsed.values())

    # Step 4: Infer bundle (schema + axes).
    schema, axes = infer_bundle(fn, abstract_inputs)

    # Step 5: Plan tiling strategy.
    # NOTE: BatchPlanner.plan() expects the AxisSpec LIST, not the schema.
    # Wrap it to catch AmbiguousAxisError and convert to CLIError.
    try:
        plan = BatchPlanner().plan(axes)
    except AmbiguousAxisError as e:
        raise CLIError(
            f"Unable to determine axis role for planning: {e}\n"
            f"Hint: decorate the function with @axis_config(AxisOverride(...)) "
            f"to resolve the axis role."
        ) from e

    # Step 6: Print a readable summary.
    _print_plan_summary(plan)


def _print_plan_summary(plan: Any) -> None:
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
