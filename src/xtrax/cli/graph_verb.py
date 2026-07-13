"""xtrax CLI graph-validate verb: GraphValidateArgs + run_graph_validate for REGISTRY wiring
(T1-10, #2181 entry-edge item).

Registered as the flat verb `graph-validate` (`xtrax graph-validate <ir.json>`), not the DAG
doc's informal two-word `graph validate` prose -- REGISTRY/entrypoint.py's tyro dispatch is a
flat `dict[str, (ArgsClass, run_fn)]` with no nested-subcommand support today; adding one for a
single verb is out of this item's scope (see validate.py's module docstring / the plan).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.composition.errors import GraphSerializationError, SchemaVersionError
from xtrax.composition.serialize import dump_graph, load_graph
from xtrax.composition.validate import validate_graph

VALIDATE_ENVELOPE_SCHEMA_VERSION = 1

# tyro is bound dynamically in entrypoint.py:main() to keep imports tyro-free
tyro: Any = None


@dataclass
class GraphValidateArgs:
    """Arguments for the graph-validate verb.

    Attributes:
        ir_path: Path to a D4 IR document (`<ir.json>`, T1-08's format) to validate in place.
    """

    ir_path: "tyro.conf.Positional[str]"


def run_graph_validate(args: GraphValidateArgs) -> None:
    """Load `<ir.json>`, run the deterministic validation gate, write audit_verdict back.

    Fast/loud: a malformed document (missing/unknown schema_version, unresolvable
    callable_ref) raises SystemExit with a clean message, not a raw traceback. Any node not
    verdict=PASS prints a structured JSON envelope and exits 1.
    """
    path = Path(args.ir_path)
    try:
        graph = load_graph(path)
    except (
        SchemaVersionError,
        GraphSerializationError,
        ValueError,
        FileNotFoundError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"graph-validate: failed to load {path}: {exc}") from None

    result = validate_graph(graph, root=Path.cwd())
    dump_graph(result.graph, path)

    envelope = {
        "schema_version": VALIDATE_ENVELOPE_SCHEMA_VERSION,
        "failure_count": len(result.failures),
        "failures": [
            {"node_id": f.node_id, "check": f.check, "message": f.message} for f in result.failures
        ],
    }
    print(json.dumps(envelope, indent=2))

    if not result.passed:
        raise SystemExit(1)


__all__ = ["VALIDATE_ENVELOPE_SCHEMA_VERSION", "GraphValidateArgs", "run_graph_validate"]
