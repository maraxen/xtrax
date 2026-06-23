"""Explain verb for xtrax CLI (T5).

This module provides the run_explain() entry point: it loads a function,
parses shape specs, infers the bundle, plans tiling strategy, explains the
plan, and emits the result in the requested format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from xtrax.cli.emit import emit
from xtrax.cli.errors import CLIError
from xtrax.cli.loader import load_fn
from xtrax.cli.shapes import parse_shapes
from xtrax.eda.explain import explain_plan
from xtrax.inference.api import infer_bundle
from xtrax.tiling.plan import BatchPlanner
from xtrax.tiling.roles import AmbiguousAxisError


@dataclass
class ExplainArgs:
    """Arguments for the explain verb.

    Attributes:
        fn: Import-path string to the function to explain.
            Format: 'module.path:symbol'.
        shapes: Space-separated shape specification string.
            Format: 'name=(d0,d1,...)<dtype> ...'.
            Each name corresponds to a positional argument of the function, in order.
        fmt: Output format. One of "json" (default), "text", "html", "png".
            "json" is the machine contract: a single JSON object with a ``_meta``
            envelope and the PlanStatsDict fields.
    """

    fn: str
    shapes: str = ""
    fmt: Literal["json", "text", "html", "png"] = field(default="json")


def run_explain(args: ExplainArgs) -> None:
    """Execute the explain verb: infer bundle, plan, explain, and emit.

    Pipeline:
    1. Load the function from the import-path string via load_fn.
    2. Parse the shapes string into ShapeDtypeStruct objects via parse_shapes.
    3. Build abstract_inputs as an ordered list of ShapeDtypeStruct values.
    4. Call infer_bundle(fn, abstract_inputs) to get (schema, axes).
    5. Call BatchPlanner().plan(axes) to produce a BatchPlan.
       Catches AmbiguousAxisError and re-raises as CLIError with an
       @axis_config hint (same pattern as the plan verb).
    6. Call explain_plan(plan) to produce a PlanStatsDict.
    7. Call emit(stats, plan, args.fmt) to route output to the chosen format.

    Args:
        args: ExplainArgs with fn, shapes, and fmt.

    Raises:
        CLIError: On any user-facing error (bad import path, bad shapes,
                  unresolved axis role, or missing eda extra for html/png).
                  The message is clean and actionable, not a raw traceback.

    Example:
        >>> args = ExplainArgs(fn="mylib:forward", shapes="x=(4,3)f32")
        >>> run_explain(args)
        # Prints JSON explain output to stdout
    """
    # Step 1: Load the function from the import-path string.
    fn = load_fn(args.fn)

    # Step 2: Parse shape string to ShapeDtypeStruct dict.
    parsed = parse_shapes(args.shapes)

    # Step 3: Build ordered abstract inputs list (positional order matches fn args).
    abstract_inputs = list(parsed.values())

    # Step 4: Infer bundle (schema + axis specs).
    schema, axes = infer_bundle(fn, abstract_inputs)

    # Step 5: Plan tiling strategy.
    # Catch AmbiguousAxisError (role=UNKNOWN, undecorated fn) and convert to
    # a clean CLIError with an @axis_config hint.
    try:
        plan = BatchPlanner().plan(axes)
    except AmbiguousAxisError as exc:
        raise CLIError(
            f"Unable to determine axis role for planning: {exc}\n"
            f"Hint: decorate the function with @axis_config(AxisOverride(...)) "
            f"to resolve the axis role."
        ) from exc

    # Step 6: Explain plan (guaranteed non-empty reasoning).
    stats = explain_plan(plan)

    # Step 7: Emit in the requested format.
    emit(stats, plan, args.fmt)


__all__ = ["ExplainArgs", "run_explain"]
