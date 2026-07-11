"""Composition-layer data model: HostPrepGraph (D1, T1-06, #3059)."""

from xtrax.composition.errors import FrozenNodeError, GraphConstructionError
from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode
from xtrax.composition.node_metadata import (
    NodeMetadataSchema,
    SlotDefinition,
    load_node_metadata_schema,
    validate_node_metadata,
)

__all__ = [
    "FrozenNodeError",
    "GraphConstructionError",
    "GraphEdge",
    "HostPrepGraph",
    "HostPrepGraphNode",
    "NodeMetadataSchema",
    "SlotDefinition",
    "load_node_metadata_schema",
    "validate_node_metadata",
]
