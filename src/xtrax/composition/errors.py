"""Exception types for the xtrax.composition subpackage (T1-06 #3059, T1-08 #3063)."""


class FrozenNodeError(Exception):
    """Raised when mutating a HostPrepGraphNode whose `frozen` flag is True.

    Fires at mutation time, before any JAX trace begins -- frozen is a pure Python-level
    authoring-time guard, unrelated to tracing.
    """


class GraphConstructionError(Exception):
    """Raised for a structurally-invalid HostPrepGraph (duplicate node id, or an edge
    referencing an unknown node id) -- fires at graph-construction time, before any trace.
    """


class SchemaVersionError(Exception):
    """Raised when a serialized graph document's schema_version is missing, not an int, or
    older than MIN_SUPPORTED_GRAPH_SCHEMA_VERSION (T1-08, PM3) -- never silently default-filled.
    """


class GraphSerializationError(Exception):
    """Raised when a HostPrepGraphNode's callable_ref cannot be serialized to a resolvable
    'module.path:symbol' import-path string (a lambda, closure, bound method, or anything
    else that doesn't round-trip back to the same object via plain module attribute lookup).
    """


__all__ = [
    "FrozenNodeError",
    "GraphConstructionError",
    "GraphSerializationError",
    "SchemaVersionError",
]
