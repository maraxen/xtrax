"""xtrax CLI graph-author verb: GraphAuthorArgs + run_graph_author for REGISTRY wiring
(T1-12, #3067): the default generate-then-validate authoring front-end.

Registered as the flat verb `graph-author` (matching `graph-validate`/`graph-plan`'s naming
convention from T1-10/T1-11). Free-generates a candidate IR document via
`xtrax.composition.author.TemplateGenerator` and validates it in-process against the same
`validate_graph` gate `graph-validate` uses -- "authors >=1 graph passing graph validate" is
enforced directly here as an in-process assertion (SystemExit(1) on failure), not left for a
caller to separately verify by running `graph-validate` afterward.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.composition.author import TemplateGenerator
from xtrax.composition.serialize import dump_graph
from xtrax.composition.validate import validate_graph

AUTHOR_ENVELOPE_SCHEMA_VERSION = 1

# tyro is bound dynamically in entrypoint.py:main() to keep imports tyro-free
tyro: Any = None


@dataclass
class GraphAuthorArgs:
    """Arguments for the graph-author verb.

    Attributes:
        out_path: Where to write the generated candidate IR document (`<ir.json>`, T1-08's
            format).
        seed: Seed for the deterministic free-generator.
        num_nodes: Number of nodes to author.
    """

    out_path: "tyro.conf.Positional[str]"
    seed: int = 0
    num_nodes: int = 3


def run_graph_author(args: GraphAuthorArgs) -> None:
    """Free-generate a candidate graph, validate it, and write it to `out_path`.

    Fast/loud: an invalid `num_nodes` or an unwritable `out_path` (missing parent directory,
    read-only directory) raises SystemExit with a clean message, not a raw traceback -- reusing
    the same discipline `graph-validate`/`graph-plan` established. Any node not verdict=PASS
    prints a structured JSON envelope (same shape as `graph-validate`'s) and exits 1.
    """
    try:
        graph = TemplateGenerator().generate(seed=args.seed, num_nodes=args.num_nodes)
    except ValueError as exc:
        raise SystemExit(f"graph-author: {exc}") from None

    result = validate_graph(graph, root=Path.cwd())

    try:
        dump_graph(result.graph, Path(args.out_path))
    except OSError as exc:
        raise SystemExit(f"graph-author: failed to write {args.out_path}: {exc}") from None

    envelope = {
        "schema_version": AUTHOR_ENVELOPE_SCHEMA_VERSION,
        "failure_count": len(result.failures),
        "failures": [
            {"node_id": f.node_id, "check": f.check, "message": f.message} for f in result.failures
        ],
    }
    print(json.dumps(envelope, indent=2))

    if not result.passed:
        raise SystemExit(1)


__all__ = ["AUTHOR_ENVELOPE_SCHEMA_VERSION", "GraphAuthorArgs", "run_graph_author"]
