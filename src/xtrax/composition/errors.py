"""Exception types for the composition-layer HostPrepGraph data model (T1-06, #3059)."""


class FrozenNodeError(Exception):
    """Raised when mutating a HostPrepGraphNode whose `frozen` flag is True.

    Fires at mutation time, before any JAX trace begins -- frozen is a pure Python-level
    authoring-time guard, unrelated to tracing.
    """


class GraphConstructionError(Exception):
    """Raised for a structurally-invalid HostPrepGraph (duplicate node id, or an edge
    referencing an unknown node id) -- fires at graph-construction time, before any trace.
    """


__all__ = ["FrozenNodeError", "GraphConstructionError"]
