"""Composition IR JSON Schema emitter, derived from E1 types (T1-09, #3064).

Inference-layer-owned (fork-13 of the design spec), not `xtrax.composition`: this module
describes the shape T1-08's `xtrax.composition.serialize` already reads/writes ("D4 == the NS
typed composition IR, one schema two consumers" -- chain-map read-only, the #2181 loop). The
emitted `schema_version` is `xtrax.composition.serialize.GRAPH_SCHEMA_VERSION`, imported rather
than reinvented -- a second, independent version counter here would recreate exactly the "two
competing graph formats" risk the design spec rejected a standalone shared schema package over.

Every `$defs` entry for a real dataclass/Protocol is built by introspecting the live type at
call time (`dataclasses.fields()` + `typing.get_type_hints()`), never a hand-typed field list --
this is what "derived from, not parallel to" (fork-13) means mechanically: an added/removed
field on `BundleSchema`/`AxisSpec`/`AxisOverride` changes the emitted schema automatically, with
no separate edit required here. `Fuse`/`Tap`/`Sink` (the closed axis-boundary vocabulary,
`xtrax.stages.boundaries`) are Protocols, not dataclasses, so their `$defs` entries are built by
walking their class-level type hints (`ordered: bool` for Tap/Sink; empty for Fuse, which is a
plain callable) instead.
"""

import dataclasses
import enum
import types
import typing
from typing import Any

from jax import ShapeDtypeStruct

from xtrax.composition.node_metadata import SlotDefinition, load_node_metadata_schema
from xtrax.composition.serialize import GRAPH_SCHEMA_VERSION
from xtrax.inference.config import AxisOverride
from xtrax.inference.schema import BundleSchema
from xtrax.stages.boundaries import Fuse, Sink, Tap
from xtrax.tiling.plan import AxisSpec

_IMPORT_PATH_DESCRIPTION = (
    "import-path 'module.path:symbol' string (xtrax.composition.serialize convention)"
)


def _import_path_schema() -> dict[str, Any]:
    return {"type": "string", "description": _IMPORT_PATH_DESCRIPTION}


def _make_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if "anyOf" in schema:
        return {"anyOf": [*schema["anyOf"], {"type": "null"}]}
    if "type" in schema:
        base_type = schema["type"]
        types = base_type if isinstance(base_type, list) else [base_type]
        return {**schema, "type": [*types, "null"]}
    return schema


def _type_to_json_schema(tp: Any) -> dict[str, Any]:
    if tp is Any or tp is typing.Any:
        return {}
    if tp is type(None):
        return {"type": "null"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp is ShapeDtypeStruct:
        return {
            "type": "object",
            "properties": {
                "shape": {"type": "array", "items": {"type": "integer"}},
                "dtype": {"type": "string"},
            },
        }
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return {"type": "string", "enum": [member.value for member in tp]}

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin is typing.Union or isinstance(tp, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        nullable = len(non_none) != len(args)
        if len(non_none) == 1:
            schema = _type_to_json_schema(non_none[0])
        else:
            schema = {"anyOf": [_type_to_json_schema(a) for a in non_none]}
        return _make_nullable(schema) if nullable else schema

    if origin in (tuple, list, frozenset, set):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _type_to_json_schema(item_type)}

    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {"type": "object", "additionalProperties": _type_to_json_schema(value_type)}

    if origin is not None and getattr(origin, "__name__", "") == "Callable":
        return _import_path_schema()

    return {}


def _dataclass_to_json_schema(cls: type) -> dict[str, Any]:
    hints = typing.get_type_hints(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        if f.name.startswith("_"):
            continue
        properties[f.name] = _type_to_json_schema(hints.get(f.name, Any))
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _protocol_to_json_schema(cls: type) -> dict[str, Any]:
    hints = {
        name: tp for name, tp in typing.get_type_hints(cls).items() if not name.startswith("_")
    }
    if not hints:
        return _import_path_schema()

    properties = {"callable_ref": _import_path_schema()}
    required = ["callable_ref"]
    for name, tp in hints.items():
        properties[name] = _type_to_json_schema(tp)
        required.append(name)
    return {"type": "object", "properties": properties, "required": required}


def _slot_to_json_schema(slot: SlotDefinition) -> dict[str, Any]:
    if slot.type == "string":
        schema: dict[str, Any] = {"type": "string"}
        if slot.min_length is not None:
            schema["minLength"] = slot.min_length
        return schema
    if slot.type == "enum":
        return {"type": "string", "enum": list(slot.values or ())}
    if slot.type == "array":
        return {"type": "array", "items": {"type": "object"}}
    if slot.type == "object":
        return {
            "type": "object",
            "properties": {name: {"type": "string"} for name in (slot.fields or {})},
        }
    return {}


def _node_metadata_json_schema() -> dict[str, Any]:
    schema = load_node_metadata_schema()
    properties: dict[str, Any] = {}
    required: list[str] = []
    for slot in schema.slots:
        properties[slot.id] = _slot_to_json_schema(slot)
        if slot.required:
            required.append(slot.id)
    return {"type": "object", "properties": properties, "required": required}


def _nullable_ref(def_name: str) -> dict[str, Any]:
    return {"anyOf": [{"$ref": f"#/$defs/{def_name}"}, {"type": "null"}]}


def _axis_boundary_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "fuse": _nullable_ref("Fuse"),
            "tap": _nullable_ref("Tap"),
            "sink": _nullable_ref("Sink"),
        },
    }


def emit_ir_schema() -> dict[str, Any]:
    """Emit the composition IR JSON Schema (v0.1), derived live from the real E1 types.

    The document shape at the top level (schema_version/nodes/edges) mirrors
    `xtrax.composition.serialize.serialize_graph`'s actual output -- this is the single
    typed composition IR, not a parallel format. `$defs` additionally documents the full
    axis/boundary/metadata vocabulary (`BundleSchema`, `AxisSpec`, `AxisOverride`,
    `AxisBoundary`, `Fuse`, `Tap`, `Sink`, `NodeMetadata`) for downstream consumers
    (T1-10's `graph validate`, T1-12's authoring front-end).
    """
    node_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "callable_ref": _import_path_schema(),
            "metadata": {"$ref": "#/$defs/NodeMetadata"},
            "frozen": {"type": "boolean"},
        },
        "required": ["id", "callable_ref", "metadata"],
    }
    edge_schema = {
        "type": "object",
        "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
        "required": ["src", "dst"],
    }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "xtrax://composition-ir",
        "schema_version": GRAPH_SCHEMA_VERSION,
        "title": "xtrax composition IR",
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer"},
            "nodes": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
            "edges": {"type": "array", "items": {"$ref": "#/$defs/Edge"}},
        },
        "required": ["schema_version", "nodes"],
        "$defs": {
            "Node": node_schema,
            "Edge": edge_schema,
            "NodeMetadata": _node_metadata_json_schema(),
            "BundleSchema": _dataclass_to_json_schema(BundleSchema),
            "AxisSpec": _dataclass_to_json_schema(AxisSpec),
            "AxisOverride": _dataclass_to_json_schema(AxisOverride),
            "AxisBoundary": _axis_boundary_json_schema(),
            "Fuse": _protocol_to_json_schema(Fuse),
            "Tap": _protocol_to_json_schema(Tap),
            "Sink": _protocol_to_json_schema(Sink),
        },
    }


__all__ = ["emit_ir_schema"]
