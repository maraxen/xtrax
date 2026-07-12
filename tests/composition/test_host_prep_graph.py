"""Tests for the D1 HostPrepGraph data model (T1-06, #3059, AC2)."""

import pytest

from xtrax.composition.errors import FrozenNodeError, GraphConstructionError
from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode

VALID_METADATA = {"nl_description": "Prepare host-side tensors before JIT lowering."}


def _identity(x):
    return x


class TestFrozenFlag:
    def test_frozen_node_construction_succeeds(self) -> None:
        """Constructing a node with frozen=True must not trip the mutation guard on
        its own field-assignments during __init__."""
        node = HostPrepGraphNode(
            id="n1", callable_ref=_identity, metadata=VALID_METADATA, frozen=True
        )
        assert node.frozen is True
        assert node.id == "n1"

    def test_frozen_node_mutation_raises(self) -> None:
        node = HostPrepGraphNode(
            id="n1", callable_ref=_identity, metadata=VALID_METADATA, frozen=True
        )
        with pytest.raises(FrozenNodeError, match="frozen HostPrepGraphNode 'n1'"):
            node.metadata = {"nl_description": "changed"}

    def test_frozen_node_cannot_unfreeze_itself(self) -> None:
        node = HostPrepGraphNode(
            id="n1", callable_ref=_identity, metadata=VALID_METADATA, frozen=True
        )
        with pytest.raises(FrozenNodeError):
            node.frozen = False

    def test_mutable_node_mutation_succeeds(self) -> None:
        node = HostPrepGraphNode(
            id="n1", callable_ref=_identity, metadata=VALID_METADATA, frozen=False
        )
        node.metadata = {"nl_description": "updated description"}
        assert node.metadata == {"nl_description": "updated description"}


class TestNodeMetadataValidation:
    def test_missing_required_nl_description_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required slot 'nl_description'"):
            HostPrepGraphNode(id="n1", callable_ref=_identity, metadata={})

    def test_valid_metadata_constructs(self) -> None:
        node = HostPrepGraphNode(id="n1", callable_ref=_identity, metadata=VALID_METADATA)
        assert node.metadata == VALID_METADATA


class TestHostPrepGraphConstruction:
    def test_duplicate_node_id_raises(self) -> None:
        n1 = HostPrepGraphNode(id="dup", callable_ref=_identity, metadata=VALID_METADATA)
        n2 = HostPrepGraphNode(id="dup", callable_ref=_identity, metadata=VALID_METADATA)
        with pytest.raises(GraphConstructionError, match="duplicate node id"):
            HostPrepGraph(nodes=(n1, n2))

    def test_edge_referencing_unknown_node_raises(self) -> None:
        n1 = HostPrepGraphNode(id="n1", callable_ref=_identity, metadata=VALID_METADATA)
        with pytest.raises(GraphConstructionError, match="unknown node id"):
            HostPrepGraph(nodes=(n1,), edges=(GraphEdge(src="n1", dst="does-not-exist"),))

    def test_valid_two_node_graph_constructs(self) -> None:
        n1 = HostPrepGraphNode(id="host-prep", callable_ref=_identity, metadata=VALID_METADATA)
        n2 = HostPrepGraphNode(
            id="export-bundle",
            callable_ref=_identity,
            metadata={"nl_description": "Emit StableHLO bundle after purity review."},
        )
        edges = (GraphEdge(src="host-prep", dst="export-bundle"),)
        graph = HostPrepGraph(nodes=(n1, n2), edges=edges)
        assert len(graph.nodes) == 2
        assert graph.edges[0].src == "host-prep"
