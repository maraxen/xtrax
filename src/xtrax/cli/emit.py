"""Output router for the explain verb (T5).

This module provides emit(), which routes a PlanStatsDict + BatchPlan to the
requested output format: json (machine contract), text (human summary),
html, or png (via xtrax.eda render — requires xtrax[eda] extra).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from xtrax.cli.errors import CLIError

_VALID_FMTS = frozenset({"json", "text", "html", "png"})


def emit(stats: Mapping[str, Any], plan: object, fmt: str, out: str | None = None) -> None:
    """Route explain output to the requested format.

    Args:
        stats: The PlanStatsDict returned by explain_plan. Must be a dict whose
            top-level keys are the PlanStatsDict fields (axes, strategy_counts,
            total_axes, memory_warnings, dedup_stats, bucket_stats).
        plan: The BatchPlan object (used for html/png render).
        fmt: Output format. One of "json", "text", "html", "png".
            "json" is the machine contract: prints a single JSON object with
            a ``_meta`` envelope to stdout.
        out: Optional file path for html/png output. See _emit_render for the
            per-format contract. Ignored for json/text.

    Raises:
        CLIError: If fmt is unknown, or if html/png render fails because the
            eda extra is not installed, or if fmt="png" and out is None (binary
            output to a terminal is a footgun).
    """
    if fmt not in _VALID_FMTS:
        raise CLIError(f"Unknown output format {fmt!r}. Valid formats: {sorted(_VALID_FMTS)}")

    if fmt == "json":
        _emit_json(stats)
    elif fmt == "text":
        _emit_text(stats)
    elif fmt in {"html", "png"}:
        _emit_render(plan, fmt, out=out)


def _emit_json(stats: Mapping[str, Any]) -> None:
    """Emit stats as a single JSON object with _meta envelope to stdout.

    The _meta envelope is the machine contract: consumers MUST check
    schema_version before parsing the rest of the object.

    Args:
        stats: PlanStatsDict to serialise.
    """
    payload = {"_meta": {"schema_version": 1}, **stats}
    print(json.dumps(payload))


def _emit_text(stats: Mapping[str, Any]) -> None:
    """Emit a human-readable summary of the plan stats to stdout.

    Args:
        stats: PlanStatsDict to summarise.
    """
    total_axes = stats.get("total_axes", 0)
    strategy_counts = stats.get("strategy_counts", {})
    axes = stats.get("axes", [])
    memory_warnings = stats.get("memory_warnings", [])

    print("Plan Explanation Summary")
    print("=" * 60)
    print(f"Total axes: {total_axes}")

    if strategy_counts:
        strategies = ", ".join(f"{k}={v}" for k, v in sorted(strategy_counts.items()))
        print(f"Strategy counts: {strategies}")
    else:
        print("Strategy counts: (none)")

    print()

    for i, entry in enumerate(axes, start=1):
        name = entry.get("name", f"axis_{i}")
        strategy = entry.get("strategy", "?")
        cardinality = entry.get("cardinality", "?")
        batch_size = entry.get("batch_size", "?")
        reasoning = entry.get("reasoning", "")
        mem = entry.get("memory_estimate_bytes")

        print(f"  Axis {i}: {name}")
        print(f"    Strategy:    {strategy}")
        print(f"    Cardinality: {cardinality}")
        print(f"    Batch size:  {batch_size}")
        if mem is not None:
            print(f"    Memory est.: {mem} bytes")
        if reasoning:
            print(f"    Reasoning:   {reasoning}")
        print()

    if memory_warnings:
        print("Memory warnings:")
        for w in memory_warnings:
            print(f"  ! {w}")
        print()

    print("=" * 60)


def _emit_render(plan: object, fmt: str, out: str | None = None) -> None:
    """Emit html/png by delegating to xtrax.eda.render.

    The render function lazily imports matplotlib at call time. If the eda
    extra is not installed, the import will fail with ModuleNotFoundError;
    we catch that and re-raise as a clean CLIError.

    Per-format output contracts:
        html: If out is None, render returns the HTML string — printed to stdout.
              If out is given, render writes the file — prints "wrote <out>" to stderr.
        png:  PNG is binary. out is REQUIRED; if out is None, raises CLIError
              directing the user to supply --out <path>. If out is given, render
              writes the file — prints "wrote <out>" to stderr.

    Args:
        plan: The BatchPlan to render.
        fmt: "html" or "png".
        out: Optional file path. Required for png. For html, defaults to stdout.

    Raises:
        CLIError: If matplotlib or other eda dependencies are missing, or if
            fmt="png" and out is None.
    """
    if fmt == "png" and out is None:
        raise CLIError(
            "--fmt png requires --out <path> (binary output cannot be written to a terminal)"
        )

    try:
        from xtrax.eda import render

        result = render(plan, fmt=fmt, path=out)
    except (ModuleNotFoundError, ImportError) as exc:
        raise CLIError(f"--fmt {fmt!r} requires the eda extra: pip install xtrax[eda]") from exc

    if out is not None:
        # render() wrote the file; confirm to stderr
        print(f"wrote {out}", file=sys.stderr)
    else:
        # out is None — only reachable for html (png already guarded above)
        # result is the HTML string
        print(result)


__all__ = ["emit"]
