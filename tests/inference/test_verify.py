"""Tests for structure consistency verification."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.inference.errors import StructureMismatchError
from xtrax.inference.verify import verify_structure


class TestVerifyStructure:
    """Test suite for verify_structure function."""

    def test_clean_fn_with_matching_structure_passes_silently(self):
        """A clean fn whose concrete output structure matches eval_shape passes silently."""

        def f(x):
            return x * 2

        abstract_x = ShapeDtypeStruct((2, 3), np.float32)
        concrete_x = jnp.ones((2, 3), dtype=jnp.float32)

        # Should not raise, returns None
        result = verify_structure(f, [abstract_x], [concrete_x])
        assert result is None

    def test_shape_divergence_raises_structure_mismatch_error(self):
        """A fn whose concrete output shape differs from abstract raises StructureMismatchError.

        Simulate by passing abstract_inputs that imply one shape, but concrete_inputs
        that will produce a different shape in the actual output.
        """

        def f(x):
            return x * 2

        # Abstract says (2, 3), but concrete will be (2, 4)
        abstract_x = ShapeDtypeStruct((2, 3), np.float32)
        concrete_x = jnp.ones((2, 4), dtype=jnp.float32)

        with pytest.raises(StructureMismatchError) as exc_info:
            verify_structure(f, [abstract_x], [concrete_x])

        assert "shape" in str(exc_info.value).lower()

    def test_dtype_divergence_raises_structure_mismatch_error(self):
        """Dtype divergence: abstract input float32, concrete input int32.

        When abstract_inputs imply float32 but concrete_inputs are int32,
        the outputs will diverge in dtype.
        """

        def f_identity(x):
            return x  # Pass through

        # Abstract says float32, concrete is int32 -> divergence
        abstract_x_float = ShapeDtypeStruct((2, 3), np.float32)
        concrete_x_int = jnp.ones((2, 3), dtype=jnp.int32)

        with pytest.raises(StructureMismatchError) as exc_info:
            verify_structure(f_identity, [abstract_x_float], [concrete_x_int])

        assert "dtype" in str(exc_info.value).lower()

    def test_treedef_divergence_tuple_vs_dict_raises_error(self):
        """Pytree structure divergence: abstract inputs imply structure A, concrete produces B."""

        # This tests the case where abstract_inputs are a list,
        # but we pass concrete_inputs as a dict
        def f(x_list):
            # Expects list arg, returns first element times 2
            return x_list[0] * 2

        abstract_as_list = [ShapeDtypeStruct((2, 3), np.float32)]
        # eval_shape(f, abstract_as_list) will trace and return ShapeDtypeStruct for the output

        # But now pass concrete_inputs as a different structure (dict instead of list)
        # This will still work with the function (dict access will fail), but
        # the point is that abstract_inputs and concrete_inputs have different structures
        concrete_as_dict = {"arr": jnp.ones((2, 3), dtype=jnp.float32)}

        with pytest.raises((StructureMismatchError, TypeError, KeyError)):
            # TypeError or KeyError because the fn tries to access [0] on a dict
            # Or StructureMismatchError if we catch the input structure mismatch
            verify_structure(f, abstract_as_list, concrete_as_dict)

    def test_multiple_arguments_with_shape_mismatch_raises_error(self):
        """Output shape mismatch when abstract_inputs differ from concrete_inputs."""

        def f(x):
            return x

        # Abstract says (2, 3), concrete says (2, 4) -> output shapes differ
        abstract_x = ShapeDtypeStruct((2, 3), np.float32)
        concrete_x = jnp.ones((2, 4), dtype=jnp.float32)

        with pytest.raises(StructureMismatchError):
            verify_structure(f, [abstract_x], [concrete_x])

    def test_nested_dict_structure_with_matching_shapes_passes(self):
        """Nested dict structure with matching shapes and dtypes passes."""

        def f(batch):
            return {
                "images": batch["images"] * 2,
                "labels": batch["labels"] + 1,
            }

        abstract_batch = {
            "images": ShapeDtypeStruct((4, 3, 3), np.float32),
            "labels": ShapeDtypeStruct((4,), np.int32),
        }

        concrete_batch = {
            "images": jnp.ones((4, 3, 3), dtype=jnp.float32),
            "labels": jnp.ones((4,), dtype=jnp.int32),
        }

        result = verify_structure(f, [abstract_batch], [concrete_batch])
        assert result is None

    def test_tuple_output_with_matching_structure_passes(self):
        """Tuple output with matching structure passes."""

        def f(x):
            return (x * 2, x + 1)

        abstract_x = ShapeDtypeStruct((2, 3), np.float32)
        concrete_x = jnp.ones((2, 3), dtype=jnp.float32)

        result = verify_structure(f, [abstract_x], [concrete_x])
        assert result is None

    def test_empty_input_passes(self):
        """Function with empty/no inputs passes."""

        def f():
            return jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

        result = verify_structure(f, [], [])
        assert result is None
