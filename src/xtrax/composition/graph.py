"""D1 HostPrepGraph data model (T1-06, #3059): typed pre-JIT nodes + edges.

The compile-time substrate the IR serializes (T1-08) and the #2181 evolve-block mutation
boundary maps onto (T1-09+). A `HostPrepGraphNode` wraps a real callable (authoring-time --
turning it into a serializable import-path string is T1-08's job, not this module's) plus
`node_metadata` slots enforced against `node_metadata_schema.toml` (see
`xtrax.composition.node_metadata`), and a per-node `frozen`/mutable flag -- the EVOLVE-BLOCK
analog: a frozen node rejects mutation before any trace begins (AC2).

`frozen` is a per-INSTANCE runtime flag, not a class-wide property, so `HostPrepGraphNode`
can't be `@dataclass(frozen=True)` at the class level (that would make every node immutable,
losing the mutable/EVOLVE-BLOCK side of the flag entirely). Instead it uses a custom
`__setattr__` gated on a `_constructed` sentinel: `_constructed` is flipped to True only at
the very end of `__post_init__`, via `object.__setattr__` (bypassing the override entirely),
so the guard never fires during the node's own construction -- this makes the guard
independent of dataclass field declaration order, unlike checking `self.frozen` directly
during `__init__` (which would only work by accident of `frozen` being the last field set).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from xtrax.composition.errors import FrozenNodeError, GraphConstructionError
from xtrax.composition.node_metadata import validate_node_metadata


@dataclass(slots=True)
class HostPrepGraphNode:
    """A single typed pre-JIT graph node: a callable ref, validated metadata, and a
    frozen/mutable flag.

    Attributes:
        id: Unique node identifier within a HostPrepGraph.
        callable_ref: The wrapped Python callable (live object, not a serialized reference).
        metadata: node_metadata slots, validated against node_metadata_schema.toml at
            construction time (required nl_description enforced immediately, not deferred).
        frozen: If True, any attribute mutation after construction raises FrozenNodeError.
    """

    id: str
    callable_ref: Callable[..., Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    frozen: bool = False
    _constructed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_node_metadata(dict(self.metadata))
        object.__setattr__(self, "_constructed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_constructed", False) and self.frozen:
            msg = f"cannot set {name!r} on frozen HostPrepGraphNode {self.id!r}"
            raise FrozenNodeError(msg)
        object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A directed edge between two HostPrepGraphNode ids."""

    src: str
    dst: str


@dataclass(frozen=True, slots=True)
class HostPrepGraph:
    """A typed pre-JIT graph: HostPrepGraphNodes plus directed edges between their ids.

    Validated at construction time: node ids must be unique, and every edge must reference
    known node ids -- both raise GraphConstructionError, before any trace begins.
    """

    nodes: tuple[HostPrepGraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()

    def __post_init__(self) -> None:
        ids = [node.id for node in self.nodes]
        dupes = {node_id for node_id in ids if ids.count(node_id) > 1}
        if dupes:
            msg = f"duplicate node id(s): {sorted(dupes)}"
            raise GraphConstructionError(msg)

        known = set(ids)
        for edge in self.edges:
            if edge.src not in known or edge.dst not in known:
                msg = f"edge {edge!r} references an unknown node id"
                raise GraphConstructionError(msg)


__all__ = ["GraphEdge", "HostPrepGraph", "HostPrepGraphNode"]
