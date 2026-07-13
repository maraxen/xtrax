"""D4 graph serialization + schema-version load gate (T1-08, #3063, AC5/AC5-version-gate).

Serializes a HostPrepGraph to JSON and back. JSON, not TOML: T1-09's IR emitter and T1-10's
`graph validate` CLI verb both consume a JSON document (`<ir.json>`), so this is the format
the rest of the #2174/#2181 chain actually expects. The document shape is a strict superset
of the existing ad-hoc sample (`.praxia/composition/samples/valid_graph.json`, consumed by
`scripts/graph_auditor.py`, which reads only `id`/`metadata` per node and ignores unknown
keys) -- `callable_ref`, `frozen`, a top-level `schema_version`, and `edges` are additions,
not a break.

PM3 (design spec pre-mortem): a prior autopsy shipped a loader that tolerated a missing
`schema_version` with a default instead of failing, silently changing the meaning of old
graphs. `deserialize_graph` checks `schema_version` FIRST, before touching anything else, and
never default-fills it -- matching the same discipline `xtrax.cli.config.load_config` and
`xtrax.cli.manifest.read_manifest` already use for their own schema-version checks.

`callable_ref` is a live Python callable in HostPrepGraphNode's in-memory form (T1-06); this
module serializes it to the same `module.path:symbol` import-path format
`xtrax.cli.loader.load_fn` uses elsewhere in this codebase, but implements its own resolver
rather than importing `xtrax.cli` -- `cli/` is the auto-CLI surface built on core primitives,
and having the core composition substrate depend on it would invert that layering.
"""

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xtrax.composition.errors import GraphSerializationError, SchemaVersionError
from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode

GRAPH_SCHEMA_VERSION = 1
MIN_SUPPORTED_GRAPH_SCHEMA_VERSION = 1


def _callable_to_import_path(fn: Callable[..., Any]) -> str:
    module_name = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module_name or not qualname:
        msg = f"callable {fn!r} has no __module__/__qualname__ to serialize"
        raise GraphSerializationError(msg)
    if "." in qualname or "<" in qualname:
        msg = (
            f"callable {fn!r} (qualname {qualname!r}) is not a plain top-level module "
            "attribute -- lambdas, closures, locally-defined functions, and bound/nested "
            "methods cannot be serialized to a resolvable import path"
        )
        raise GraphSerializationError(msg)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"callable {fn!r}: module {module_name!r} could not be re-imported"
        raise GraphSerializationError(msg) from exc

    resolved = getattr(module, qualname, None)
    if resolved is not fn:
        msg = (
            f"callable {fn!r}: {module_name}.{qualname} does not resolve back to the same "
            "object -- not safely serializable"
        )
        raise GraphSerializationError(msg)

    return f"{module_name}:{qualname}"


def _resolve_import_path(path: str) -> Callable[..., Any]:
    if ":" not in path:
        msg = f"malformed import path {path!r}: expected 'module.path:symbol'"
        raise GraphSerializationError(msg)
    module_name, _, attr_name = path.rpartition(":")
    if not module_name or not attr_name:
        msg = f"malformed import path {path!r}: empty module or attribute name"
        raise GraphSerializationError(msg)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        msg = f"cannot import module {module_name!r} from path {path!r}"
        raise GraphSerializationError(msg) from exc

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        msg = f"module {module_name!r} has no attribute {attr_name!r} (from path {path!r})"
        raise GraphSerializationError(msg) from exc


def serialize_node(node: HostPrepGraphNode) -> dict[str, Any]:
    """Serialize one HostPrepGraphNode to a JSON-able dict."""
    return {
        "id": node.id,
        "callable_ref": _callable_to_import_path(node.callable_ref),
        "metadata": dict(node.metadata),
        "frozen": node.frozen,
    }


def deserialize_node(data: dict[str, Any]) -> HostPrepGraphNode:
    """Deserialize one node dict. Missing nl_description fails inside HostPrepGraphNode's own
    construction (T1-06's validate_node_metadata), not re-validated here.
    """
    for key in ("id", "callable_ref", "metadata"):
        if key not in data:
            msg = f"node missing required field {key!r}: {data!r}"
            raise ValueError(msg)

    return HostPrepGraphNode(
        id=data["id"],
        callable_ref=_resolve_import_path(data["callable_ref"]),
        metadata=data["metadata"],
        frozen=data.get("frozen", False),
    )


def serialize_graph(graph: HostPrepGraph) -> dict[str, Any]:
    """Serialize a HostPrepGraph to a JSON-able dict, tagged with GRAPH_SCHEMA_VERSION."""
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "nodes": [serialize_node(node) for node in graph.nodes],
        "edges": [{"src": edge.src, "dst": edge.dst} for edge in graph.edges],
    }


def deserialize_graph(data: dict[str, Any]) -> HostPrepGraph:
    """Deserialize a graph document. schema_version is checked FIRST and never default-filled
    (PM3): missing, non-int, or older than MIN_SUPPORTED_GRAPH_SCHEMA_VERSION all raise
    SchemaVersionError before anything else in `data` is touched.
    """
    if "schema_version" not in data:
        msg = "missing required field: schema_version"
        raise SchemaVersionError(msg)

    version = data["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool):
        msg = f"schema_version must be an int, got {version!r}"
        raise SchemaVersionError(msg)

    if version < MIN_SUPPORTED_GRAPH_SCHEMA_VERSION:
        msg = (
            f"schema_version {version} is older than the minimum supported version "
            f"{MIN_SUPPORTED_GRAPH_SCHEMA_VERSION}"
        )
        raise SchemaVersionError(msg)

    if "nodes" not in data:
        msg = "missing required field: nodes"
        raise ValueError(msg)

    nodes = tuple(deserialize_node(node) for node in data["nodes"])
    edges = tuple(GraphEdge(src=edge["src"], dst=edge["dst"]) for edge in data.get("edges", []))
    return HostPrepGraph(nodes=nodes, edges=edges)


def dump_graph(graph: HostPrepGraph, path: Path) -> None:
    """Serialize `graph` and write it to `path` as indented JSON."""
    path.write_text(json.dumps(serialize_graph(graph), indent=2), encoding="utf-8")


def load_graph(path: Path) -> HostPrepGraph:
    """Read and deserialize a graph document from `path`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return deserialize_graph(data)


__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "MIN_SUPPORTED_GRAPH_SCHEMA_VERSION",
    "deserialize_graph",
    "deserialize_node",
    "dump_graph",
    "load_graph",
    "serialize_graph",
    "serialize_node",
]
