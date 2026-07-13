"""xtrax CLI graph-plan verb: GraphPlanArgs + run_graph_plan for REGISTRY wiring (T1-11,
AC1 graph-plan parity, #2181 entry-edge item).

Registered as the flat verb `graph-plan` (matching `graph-validate`'s naming convention from
T1-10) -- REGISTRY/entrypoint.py's tyro dispatch has no nested-subcommand support.

Proves the "consumed via the CLI" half of AC1: this verb is the only place a `HostPrepGraph`
node's `callable_ref` is turned into a `BatchPlan` through actual CLI code, converging on the
exact same `plan_from_fn` helper `run_plan`'s bare `--fn`/`--shapes` path uses -- a graph
node's callable_ref post-`load_graph` resolves to the same live Python function object
`load_fn` would resolve from a bare import-path string (both use the `module.path:symbol`
convention, see `xtrax.composition.serialize`'s own docstring), so the two paths are provably
convergent, not just coincidentally similar.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.cli.errors import CLIError
from xtrax.cli.plan import plan_from_fn, print_plan_summary
from xtrax.composition.errors import (
    GraphConstructionError,
    GraphSerializationError,
    SchemaVersionError,
)
from xtrax.composition.serialize import load_graph

# tyro is bound dynamically in entrypoint.py:main() to keep imports tyro-free
tyro: Any = None


@dataclass
class GraphPlanArgs:
    """Arguments for the graph-plan verb.

    Attributes:
        ir_path: Path to a D4 IR document (`<ir.json>`, T1-08's format).
        node_id: The id of the node whose callable_ref should be planned.
        shapes: Space-separated shape specification string, see xtrax.cli.shapes.parse_shapes.
    """

    ir_path: "tyro.conf.Positional[str]"
    node_id: str
    shapes: str = ""


def run_graph_plan(args: GraphPlanArgs) -> None:
    """Load `<ir.json>`, plan the named node's callable_ref, and print a summary.

    Fast/loud: a malformed document (missing/unknown schema_version, unresolvable
    callable_ref, duplicate ids, dangling edges) or an inaccessible path raises SystemExit
    with a clean message, not a raw traceback -- reusing the exact except-tuple T1-10's own
    audit hardened `graph-validate` with, applied here from the start.
    """
    path = Path(args.ir_path)
    try:
        graph = load_graph(path)
    except (
        SchemaVersionError,
        GraphSerializationError,
        GraphConstructionError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"graph-plan: failed to load {path}: {exc}") from None

    nodes_by_id = {node.id: node for node in graph.nodes}
    node = nodes_by_id.get(args.node_id)
    if node is None:
        raise CLIError(
            f"graph-plan: node id {args.node_id!r} not found in {path} "
            f"(available: {sorted(nodes_by_id)})"
        )

    plan = plan_from_fn(node.callable_ref, args.shapes)
    print_plan_summary(plan)


__all__ = ["GraphPlanArgs", "run_graph_plan"]
