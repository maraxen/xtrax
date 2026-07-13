"""Tests for D4 graph serialization + schema-version load gate (T1-08, #3063)."""

from pathlib import Path

import pytest

from xtrax.composition.errors import GraphSerializationError, SchemaVersionError
from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode
from xtrax.composition.serialize import (
    GRAPH_SCHEMA_VERSION,
    deserialize_graph,
    deserialize_node,
    dump_graph,
    load_graph,
    serialize_graph,
    serialize_node,
)

VALID_METADATA = {"nl_description": "Prepare host-side tensors before JIT lowering."}


def _reference_prep_fn(x):
    return x


def _reference_export_fn(x):
    return x


class TestCallableRefRoundTrip:
    def test_top_level_function_round_trips_to_same_object(self) -> None:
        node = HostPrepGraphNode(id="n1", callable_ref=_reference_prep_fn, metadata=VALID_METADATA)
        data = serialize_node(node)
        assert data["callable_ref"] == "tests.composition.test_serialize:_reference_prep_fn"

        restored = deserialize_node(data)
        assert restored.callable_ref is _reference_prep_fn

    def test_lambda_is_not_serializable(self) -> None:
        node = HostPrepGraphNode(id="n1", callable_ref=lambda x: x, metadata=VALID_METADATA)
        with pytest.raises(GraphSerializationError, match="not a plain top-level"):
            serialize_node(node)

    def test_nested_closure_is_not_serializable(self) -> None:
        def _local_fn(x):
            return x

        node = HostPrepGraphNode(id="n1", callable_ref=_local_fn, metadata=VALID_METADATA)
        with pytest.raises(GraphSerializationError, match="not a plain top-level"):
            serialize_node(node)


class TestGraphRoundTrip:
    def test_full_graph_round_trip_preserves_everything(self) -> None:
        n1 = HostPrepGraphNode(
            id="host-prep", callable_ref=_reference_prep_fn, metadata=VALID_METADATA, frozen=True
        )
        n2 = HostPrepGraphNode(
            id="export-bundle",
            callable_ref=_reference_export_fn,
            metadata={"nl_description": "Emit StableHLO bundle after purity review."},
            frozen=False,
        )
        edges = (GraphEdge(src="host-prep", dst="export-bundle"),)
        graph = HostPrepGraph(nodes=(n1, n2), edges=edges)

        restored = deserialize_graph(serialize_graph(graph))

        assert [node.id for node in restored.nodes] == ["host-prep", "export-bundle"]
        assert restored.nodes[0].callable_ref is _reference_prep_fn
        assert restored.nodes[0].frozen is True
        assert restored.nodes[0].metadata == VALID_METADATA
        assert restored.nodes[1].callable_ref is _reference_export_fn
        assert restored.nodes[1].frozen is False
        assert restored.edges == edges

    def test_serialize_graph_tags_current_schema_version(self) -> None:
        n1 = HostPrepGraphNode(id="n1", callable_ref=_reference_prep_fn, metadata=VALID_METADATA)
        data = serialize_graph(HostPrepGraph(nodes=(n1,)))
        assert data["schema_version"] == GRAPH_SCHEMA_VERSION

    def test_file_round_trip(self, tmp_path: Path) -> None:
        n1 = HostPrepGraphNode(id="n1", callable_ref=_reference_prep_fn, metadata=VALID_METADATA)
        graph = HostPrepGraph(nodes=(n1,))
        path = tmp_path / "graph.json"

        dump_graph(graph, path)
        restored = load_graph(path)

        assert restored.nodes[0].id == "n1"
        assert restored.nodes[0].callable_ref is _reference_prep_fn


class TestSchemaVersionGate:
    """AC5-version-gate (PM3): schema_version is checked FIRST and never default-filled."""

    def _valid_data(self) -> dict:
        n1 = HostPrepGraphNode(id="n1", callable_ref=_reference_prep_fn, metadata=VALID_METADATA)
        return serialize_graph(HostPrepGraph(nodes=(n1,)))

    def test_missing_schema_version_raises(self) -> None:
        data = self._valid_data()
        del data["schema_version"]
        with pytest.raises(SchemaVersionError, match="missing required field: schema_version"):
            deserialize_graph(data)

    def test_non_int_schema_version_raises(self) -> None:
        data = self._valid_data()
        data["schema_version"] = "1"
        with pytest.raises(SchemaVersionError, match="must be an int"):
            deserialize_graph(data)

    def test_bool_schema_version_raises(self) -> None:
        data = self._valid_data()
        data["schema_version"] = True
        with pytest.raises(SchemaVersionError, match="must be an int"):
            deserialize_graph(data)

    def test_schema_version_older_than_minimum_raises(self) -> None:
        data = self._valid_data()
        data["schema_version"] = 0
        with pytest.raises(SchemaVersionError, match="older than the minimum supported"):
            deserialize_graph(data)


class TestNodeFieldValidation:
    def test_missing_nl_description_fails_via_existing_validation(self) -> None:
        n1 = HostPrepGraphNode(id="n1", callable_ref=_reference_prep_fn, metadata=VALID_METADATA)
        data = serialize_graph(HostPrepGraph(nodes=(n1,)))
        del data["nodes"][0]["metadata"]["nl_description"]

        with pytest.raises(ValueError, match="missing required slot 'nl_description'"):
            deserialize_graph(data)

    @pytest.mark.parametrize("missing_key", ["id", "callable_ref", "metadata"])
    def test_node_missing_required_field_raises(self, missing_key: str) -> None:
        n1 = HostPrepGraphNode(id="n1", callable_ref=_reference_prep_fn, metadata=VALID_METADATA)
        data = serialize_node(n1)
        del data[missing_key]

        with pytest.raises(ValueError, match=f"missing required field {missing_key!r}"):
            deserialize_node(data)

    def test_graph_missing_nodes_raises(self) -> None:
        data = {"schema_version": GRAPH_SCHEMA_VERSION}
        with pytest.raises(ValueError, match="missing required field: nodes"):
            deserialize_graph(data)
