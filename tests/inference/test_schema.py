"""Tests for BundleSchema and extract_schema."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
from jax import ShapeDtypeStruct

from xtrax.inference.abstract import build_abstract_inputs
from xtrax.inference.schema import BundleSchema, extract_schema


class TestBundleSchema:
    """Test BundleSchema dataclass."""

    def test_basic_construction(self):
        """BundleSchema can be constructed with fields."""
        fields = {
            "output": ShapeDtypeStruct((2, 3), np.float32),
        }
        schema = BundleSchema(fields=fields)
        assert schema.fields == fields
        assert schema.carry_specs is None

    def test_carry_specs_passthrough(self):
        """carry_specs is stored and round-trips unchanged."""
        fields = {"output": ShapeDtypeStruct((2,), np.float32)}
        carry_specs = [object()]
        schema = BundleSchema(fields=fields, carry_specs=carry_specs)
        assert schema.carry_specs is carry_specs

    def test_carry_specs_default_none(self):
        """carry_specs defaults to None when not provided."""
        fields = {"output": ShapeDtypeStruct((2,), np.float32)}
        schema = BundleSchema(fields=fields)
        assert schema.carry_specs is None

    def test_fields_ordered(self):
        """fields dict preserves insertion order."""
        fields = {
            "a": ShapeDtypeStruct((2,), np.float32),
            "b": ShapeDtypeStruct((3,), np.int32),
            "c": ShapeDtypeStruct((4,), np.float64),
        }
        schema = BundleSchema(fields=fields)
        assert list(schema.fields.keys()) == ["a", "b", "c"]


class TestExtractSchema:
    """Test extract_schema function."""

    def test_extract_schema_eqx_module_keyed_fields(self):
        """Extract schema from a fn returning an eqx.Module with keyed fields."""

        class Output(eqx.Module):
            x: jax.Array
            y: jax.Array

        def fn(x):
            return Output(x=x * 2, y=x * 3)

        # Abstract inputs: one array (2, 3)
        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "x" in schema.fields
        assert "y" in schema.fields
        assert schema.fields["x"].shape == (2, 3)
        assert schema.fields["x"].dtype == np.float32
        assert schema.fields["y"].shape == (2, 3)
        assert schema.fields["y"].dtype == np.float32

    def test_extract_schema_eqx_module_two_fields(self):
        """Extract schema from a fn returning an eqx.Module with two fields."""

        class Output(eqx.Module):
            x: jax.Array
            y: jax.Array

        def fn(x):
            return Output(x=x * 2, y=x * 3)

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "x" in schema.fields
        assert "y" in schema.fields
        assert schema.fields["x"].shape == (2, 3)
        assert schema.fields["y"].shape == (2, 3)

    def test_extract_schema_bare_array(self):
        """Extract schema from a fn returning a bare array (positional out_0)."""

        def fn(x):
            return x * 2

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "out_0" in schema.fields
        assert schema.fields["out_0"].shape == (2, 3)
        assert schema.fields["out_0"].dtype == np.float32

    def test_extract_schema_tuple_of_arrays_positional(self):
        """Extract schema from a fn returning a tuple of arrays (positional names)."""

        def fn(x):
            return (x * 2, x * 3)

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "out_0" in schema.fields
        assert "out_1" in schema.fields
        assert schema.fields["out_0"].shape == (2, 3)
        assert schema.fields["out_1"].shape == (2, 3)

    def test_extract_schema_dict_output(self):
        """Extract schema from a fn returning a dict (DictKey)."""

        def fn(x):
            return {"logits": x * 2, "embeddings": x * 3}

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "logits" in schema.fields
        assert "embeddings" in schema.fields
        assert schema.fields["logits"].shape == (2, 3)
        assert schema.fields["embeddings"].shape == (2, 3)

    def test_extract_schema_no_annotations_fn(self):
        """Extract schema from a plain fn with NO jaxtyping or decorators."""

        def fn(x):
            return {"output": x * 2}

        abstract_inputs = [ShapeDtypeStruct((5, 10), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "output" in schema.fields
        assert schema.fields["output"].shape == (5, 10)
        assert schema.fields["output"].dtype == np.float32

    def test_extract_schema_no_compute(self):
        """Verify extract_schema uses eval_shape (no actual FLOPs)."""

        def fn(x):
            # This would fail if actually run (e.g., division by zero)
            return x / (x - x)  # Would be NaN/Inf at runtime

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        # Should NOT raise or error — eval_shape does zero compute
        schema = extract_schema(fn, abstract_inputs)

        assert isinstance(schema, BundleSchema)
        assert "out_0" in schema.fields
        # The result is abstract, not NaN
        assert isinstance(schema.fields["out_0"], ShapeDtypeStruct)

    def test_extract_schema_leaves_are_shapedtypestruct(self):
        """All leaves in schema.fields are ShapeDtypeStruct, not concrete arrays."""

        def fn(x):
            return {"a": x, "b": x + 1}

        abstract_inputs = [ShapeDtypeStruct((4, 5), np.int32)]
        schema = extract_schema(fn, abstract_inputs)

        for name, leaf in schema.fields.items():
            assert isinstance(leaf, ShapeDtypeStruct), f"Field {name} is not ShapeDtypeStruct"
            assert not isinstance(leaf, jax.Array), f"Field {name} is a concrete array!"

    def test_extract_schema_multiple_inputs(self):
        """Extract schema from a fn with multiple input args."""

        def fn(x, y):
            return {"z": x + y}

        abstract_inputs = [
            ShapeDtypeStruct((2, 3), np.float32),
            ShapeDtypeStruct((2, 3), np.float32),
        ]
        schema = extract_schema(fn, abstract_inputs)

        assert "z" in schema.fields
        assert schema.fields["z"].shape == (2, 3)

    def test_extract_schema_preserves_dtype(self):
        """Extract schema preserves dtype information."""

        def fn(x):
            return {"int_out": x.astype(np.int32), "float_out": x.astype(np.float32)}

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        assert schema.fields["int_out"].dtype == np.int32
        assert schema.fields["float_out"].dtype == np.float32

    def test_extract_schema_nested_structure(self):
        """Extract schema from a fn with nested pytree output."""

        def fn(x):
            return {
                "level1": {
                    "level2a": x * 2,
                    "level2b": x * 3,
                },
                "other": x * 4,
            }

        abstract_inputs = [ShapeDtypeStruct((2, 3), np.float32)]
        schema = extract_schema(fn, abstract_inputs)

        # All leaves should be flattened into the fields dict
        assert isinstance(schema, BundleSchema)
        # Nested paths recovered — exact naming depends on key_path recovery
        # At minimum, we have multiple fields
        assert len(schema.fields) >= 3


class TestExtractSchemaIntegration:
    """Integration tests with build_abstract_inputs."""

    def test_extract_schema_with_build_abstract_inputs(self):
        """Use build_abstract_inputs to create abstract inputs, then extract schema."""
        spec = {
            "x": ((2, 3), np.float32),
            "y": ((3,), np.float32),
        }
        abstract_inputs = build_abstract_inputs(spec)

        # Create a fn that uses both inputs
        def fn(x, y):
            # x is (2, 3), y is (3,); broadcast y
            return {"output": x + y}

        schema = extract_schema(fn, [abstract_inputs["x"], abstract_inputs["y"]])
        assert "output" in schema.fields
        assert schema.fields["output"].shape == (2, 3)
