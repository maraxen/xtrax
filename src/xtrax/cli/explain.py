"""Explain verb for xtrax CLI (T5).

This module provides the run_explain() entry point: it loads a function,
parses shape specs, infers the bundle, plans tiling strategy, explains the
plan, and emits the result in the requested format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

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
        out: Optional file path for html/png output. Contracts:
            - fmt="html": if out is None, HTML is printed to stdout; if out is given,
              file is written and a confirmation line is printed to stderr.
            - fmt="png": out is REQUIRED (binary output to a terminal is a footgun).
              If out is None, a CLIError is raised directing the user to supply --out.
              If out is given, the PNG file is written and a confirmation line is
              printed to stderr.
            Ignored for fmt="json" and fmt="text".
    """

    fn: str
    shapes: str = ""
    report: Literal["plan", "cse"] = field(default="plan")
    fmt: Literal["json", "text", "html", "png"] = field(default="json")
    out: str | None = field(default=None)


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
    7. Call emit(stats, plan, args.fmt, out=args.out) to route output to the
       chosen format. For html/png, args.out controls whether output goes to
       a file (out given) or stdout/error (out is None). See ExplainArgs.out
       for the per-format contract.

    Args:
        args: ExplainArgs with fn, shapes, fmt, and out.

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

    if args.report == "cse":
        _run_explain_cse(fn, abstract_inputs, args)
        return

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
    emit(stats, plan, args.fmt, out=args.out)


__all__ = ["ExplainArgs", "run_explain"]


def _run_explain_cse(fn, abstract_inputs, args) -> None:
    """CSE-report branch of the explain verb (spec 260825 §4.1).

    Emits a cse_report envelope (own schema_version beside the standard
    _meta key). json/text supported; html/png unsupported for CSE reports
    (no render semantics).
    """
    import json

    from xtrax.inference import analyze_cse

    report = analyze_cse(fn, abstract_inputs)

    def _dup_dict(c) -> dict[str, Any]:
        return {
            "primitive": c.primitive,
            "eqn_count": c.eqn_count,
            "params_fingerprint": c.params_fingerprint,
            "invar_shapes": [list(s) for s in c.invar_shapes],
            "est_wasted_bytes": c.est_wasted_bytes,
        }

    payload: dict[str, Any] = {
        "_meta": {"schema_version": 1},
        "cse_report": {
            "schema_version": 1,
            "total_eqns": report.total_eqns,
            "duplicate_eqns": report.duplicate_eqns,
            "trace_cache_hit": report.trace_cache_hit,
            "duplicates": [_dup_dict(c) for c in report.duplicates],
            "note": report.note,
        },
    }

    if args.fmt == "json":
        print(json.dumps(payload))
    elif args.fmt == "text":
        cr: dict[str, Any] = payload["cse_report"]
        print("CSE Report")
        print("=" * 40)
        print(f"Total eqns: {cr['total_eqns']}")
        print(f"Duplicate eqns: {cr['duplicate_eqns']}")
        if cr["trace_cache_hit"]:
            print(
                "NOTE: trace was served from cache; closure mutations after a "
                "previous analysis are NOT reflected. Use a fresh callable."
            )
        for c in cr["duplicates"]:
            print(f"  {c['primitive']} x{c['eqn_count']} (est. wasted {c['est_wasted_bytes']} B)")
        if not cr["duplicates"]:
            print("  No duplicate subexpressions detected.")
        print()
        print(cr["note"])
    else:
        raise CLIError(
            f'fmt={args.fmt!r} is not supported for --report cse (supported: "json", "text")'
        )
