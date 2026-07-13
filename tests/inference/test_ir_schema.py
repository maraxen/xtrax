"""Drift tests for the composition IR JSON Schema emitter (T1-09, #3064).

Each test re-introspects the real E1 type directly (dataclasses.fields, load_node_metadata_
schema, AxisRole's own members) rather than importing a name list from the emitter -- this is
what makes these drift tests, not just format tests: if a field is added/removed on the real
type, the test's own expectation moves too, so it only stays green if the emitter's `$defs`
entry moved with it.
"""

from __future__ import annotations

import dataclasses

from xtrax.composition.node_metadata import load_node_metadata_schema
from xtrax.composition.serialize import GRAPH_SCHEMA_VERSION
from xtrax.inference.config import AxisOverride
from xtrax.inference.ir_schema import emit_ir_schema
from xtrax.inference.schema import BundleSchema
from xtrax.tiling.plan import AxisSpec
from xtrax.tiling.roles import AxisRole


def _real_field_names(cls: type) -> set[str]:
    return {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}


class TestSchemaVersion:
    def test_schema_carries_expected_version(self) -> None:
        schema = emit_ir_schema()
        assert schema["schema_version"] == GRAPH_SCHEMA_VERSION

    def test_top_level_document_shape_matches_serialize_graph(self) -> None:
        schema = emit_ir_schema()
        assert set(schema["properties"]) == {"schema_version", "nodes", "edges"}
        assert schema["required"] == ["schema_version", "nodes"]


class TestE1FieldDrift:
    def test_axis_spec_fields_all_present(self) -> None:
        schema = emit_ir_schema()
        properties = schema["$defs"]["AxisSpec"]["properties"]
        assert _real_field_names(AxisSpec) <= set(properties)

    def test_axis_override_fields_all_present(self) -> None:
        schema = emit_ir_schema()
        properties = schema["$defs"]["AxisOverride"]["properties"]
        assert _real_field_names(AxisOverride) <= set(properties)

    def test_bundle_schema_fields_all_present(self) -> None:
        schema = emit_ir_schema()
        properties = schema["$defs"]["BundleSchema"]["properties"]
        assert _real_field_names(BundleSchema) <= set(properties)

    def test_axis_spec_required_fields_match_no_default_fields(self) -> None:
        schema = emit_ir_schema()
        no_default = {
            f.name
            for f in dataclasses.fields(AxisSpec)
            if not f.name.startswith("_")
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        }
        assert set(schema["$defs"]["AxisSpec"]["required"]) == no_default

    def test_axis_role_enum_values_present(self) -> None:
        schema = emit_ir_schema()
        role_schema = schema["$defs"]["AxisSpec"]["properties"]["role"]
        assert set(role_schema["enum"]) == {member.value for member in AxisRole}


class TestNodeMetadataDrift:
    def test_node_metadata_slots_all_present(self) -> None:
        schema = emit_ir_schema()
        properties = schema["$defs"]["NodeMetadata"]["properties"]
        slot_ids = {slot.id for slot in load_node_metadata_schema().slots}
        assert slot_ids == set(properties)

    def test_node_metadata_required_slots_match(self) -> None:
        schema = emit_ir_schema()
        required_slots = {slot.id for slot in load_node_metadata_schema().slots if slot.required}
        assert set(schema["$defs"]["NodeMetadata"]["required"]) == required_slots


class TestBoundaryVocabulary:
    def test_axis_boundary_vocabulary_is_closed(self) -> None:
        schema = emit_ir_schema()
        properties = schema["$defs"]["AxisBoundary"]["properties"]
        assert set(properties) == {"fuse", "tap", "sink"}

    def test_tap_and_sink_expose_ordered(self) -> None:
        schema = emit_ir_schema()
        for name in ("Tap", "Sink"):
            assert "ordered" in schema["$defs"][name]["properties"]
            assert "callable_ref" in schema["$defs"][name]["properties"]

    def test_fuse_is_a_plain_callable_ref(self) -> None:
        schema = emit_ir_schema()
        assert schema["$defs"]["Fuse"] == {
            "type": "string",
            "description": (
                "import-path 'module.path:symbol' string (xtrax.composition.serialize convention)"
            ),
        }


class TestSingleSchemaSource:
    def test_emit_ir_schema_is_deterministic_across_calls(self) -> None:
        """Not a caching gate -- guards against accidental hidden mutable state making the
        'single source of truth' emitter non-reproducible between two callers (chain-map, #2181
        loop) in the same process.
        """
        assert emit_ir_schema() == emit_ir_schema()
