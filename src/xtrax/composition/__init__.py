"""Composition-layer data model: HostPrepGraph (D1, T1-06) + serialization (D4, T1-08)."""

from xtrax.composition.errors import (
    FrozenNodeError,
    GraphConstructionError,
    GraphSerializationError,
    SchemaVersionError,
)
from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode
from xtrax.composition.node_metadata import (
    NodeMetadataSchema,
    SlotDefinition,
    load_node_metadata_schema,
    validate_node_metadata,
)
from xtrax.composition.serialize import (
    GRAPH_SCHEMA_VERSION,
    MIN_SUPPORTED_GRAPH_SCHEMA_VERSION,
    deserialize_graph,
    deserialize_node,
    dump_graph,
    load_graph,
    serialize_graph,
    serialize_node,
)

__all__ = [
    "FrozenNodeError",
    "GraphConstructionError",
    "GraphEdge",
    "GraphSerializationError",
    "HostPrepGraph",
    "HostPrepGraphNode",
    "NodeMetadataSchema",
    "SchemaVersionError",
    "SlotDefinition",
    "load_node_metadata_schema",
    "validate_node_metadata",
    "GRAPH_SCHEMA_VERSION",
    "MIN_SUPPORTED_GRAPH_SCHEMA_VERSION",
    "deserialize_graph",
    "deserialize_node",
    "dump_graph",
    "load_graph",
    "serialize_graph",
    "serialize_node",
]
