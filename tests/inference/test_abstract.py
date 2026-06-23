"""Tests for abstract input builder."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.inference.abstract import build_abstract_inputs


class TestBuildAbstractInputs:
    """Test suite for build_abstract_inputs function."""

    def test_roundtrip_dict_spec_with_shape_dtype_pairs(self):
        """Dict spec with (shape, dtype) pairs converts to ShapeDtypeStruct pytree."""
        spec = {
            "x": ((2, 3), np.float32),
            "y": ((4,), np.int32),
        }
        result = build_abstract_inputs(spec)

        assert isinstance(result, dict)
        assert "x" in result
        assert "y" in result

        assert isinstance(result["x"], ShapeDtypeStruct)
        assert result["x"].shape == (2, 3)
        assert result["x"].dtype == np.float32

        assert isinstance(result["y"], ShapeDtypeStruct)
        assert result["y"].shape == (4,)
        assert result["y"].dtype == np.int32

    def test_roundtrip_tuple_spec_with_shape_dtype_pairs(self):
        """Tuple spec with (shape, dtype) pairs converts to ShapeDtypeStruct pytree."""
        spec = (
            ((2, 3), np.float32),
            ((4,), np.int32),
        )
        result = build_abstract_inputs(spec)

        assert isinstance(result, tuple)
        assert len(result) == 2

        assert isinstance(result[0], ShapeDtypeStruct)
        assert result[0].shape == (2, 3)
        assert result[0].dtype == np.float32

        assert isinstance(result[1], ShapeDtypeStruct)
        assert result[1].shape == (4,)
        assert result[1].dtype == np.int32

    def test_identity_passthrough_shapedtypestruct(self):
        """ShapeDtypeStruct leaves pass through unchanged."""
        original = ShapeDtypeStruct((5, 10), np.float64)
        spec = {"abstract": original}
        result = build_abstract_inputs(spec)

        assert isinstance(result, dict)
        assert "abstract" in result
        assert result["abstract"] is original
        assert result["abstract"].shape == (5, 10)
        assert result["abstract"].dtype == np.float64

    def test_nested_dict_spec(self):
        """Nested dict spec preserves structure."""
        spec = {
            "batch": {
                "images": ((32, 224, 224, 3), np.uint8),
                "labels": ((32,), np.int32),
            },
            "metadata": {
                "size": ((1,), np.int64),
            },
        }
        result = build_abstract_inputs(spec)

        assert isinstance(result, dict)
        assert isinstance(result["batch"], dict)
        assert isinstance(result["metadata"], dict)

        assert isinstance(result["batch"]["images"], ShapeDtypeStruct)
        assert result["batch"]["images"].shape == (32, 224, 224, 3)
        assert result["batch"]["images"].dtype == np.uint8

        assert isinstance(result["batch"]["labels"], ShapeDtypeStruct)
        assert result["batch"]["labels"].shape == (32,)
        assert result["batch"]["labels"].dtype == np.int32

        assert isinstance(result["metadata"]["size"], ShapeDtypeStruct)
        assert result["metadata"]["size"].shape == (1,)
        assert result["metadata"]["size"].dtype == np.int64

    def test_mixed_shapedtypestruct_and_pairs(self):
        """Mixture of ShapeDtypeStruct and (shape, dtype) pairs."""
        spec = {
            "concrete": ShapeDtypeStruct((2, 3), np.float32),
            "pair": ((4,), np.int32),
        }
        result = build_abstract_inputs(spec)

        assert result["concrete"] is spec["concrete"]
        assert isinstance(result["pair"], ShapeDtypeStruct)
        assert result["pair"].shape == (4,)

    def test_jax_dtype_support(self):
        """jax dtype-like objects are accepted and converted."""
        spec = {
            "x": ((2, 3), jax.numpy.float32),
        }
        result = build_abstract_inputs(spec)

        assert isinstance(result["x"], ShapeDtypeStruct)
        assert result["x"].shape == (2, 3)

    def test_malformed_shape_not_tuple_raises_valueerror(self):
        """Non-tuple shape raises ValueError."""
        spec = {"bad": ("notashape", np.float32)}
        with pytest.raises((ValueError, TypeError)):
            build_abstract_inputs(spec)

    def test_malformed_negative_dimension_raises_valueerror(self):
        """Negative dimension in shape raises ValueError."""
        spec = {"bad": ((2, -3), np.float32)}
        with pytest.raises(ValueError):
            build_abstract_inputs(spec)

    def test_malformed_bare_string_leaf_raises_error(self):
        """Bare string leaf raises ValueError or TypeError."""
        spec = {"bad": "just_a_string"}
        with pytest.raises((ValueError, TypeError)):
            build_abstract_inputs(spec)

    def test_malformed_bare_int_leaf_raises_error(self):
        """Bare int leaf raises ValueError or TypeError."""
        spec = {"bad": 42}
        with pytest.raises((ValueError, TypeError)):
            build_abstract_inputs(spec)

    def test_malformed_incomplete_pair_raises_error(self):
        """Incomplete pair (only shape, no dtype) raises error."""
        spec = {"bad": ((2, 3),)}  # Missing dtype
        with pytest.raises((ValueError, TypeError)):
            build_abstract_inputs(spec)

    def test_empty_dict_preserves_structure(self):
        """Empty dict passes through with same structure."""
        spec = {}
        result = build_abstract_inputs(spec)
        assert result == {}

    def test_empty_tuple_preserves_structure(self):
        """Empty tuple passes through with same structure."""
        spec = ()
        result = build_abstract_inputs(spec)
        assert result == ()

    def test_list_not_treated_as_leaf(self):
        """List shapes (not tuples) in pairs should be normalized or rejected."""
        # The spec uses [2, 3] instead of (2, 3)
        # Should either normalize it to tuple or raise error
        spec = {"x": ([2, 3], np.float32)}
        result = build_abstract_inputs(spec)
        # If it passes, the shape should be normalized to tuple
        assert isinstance(result["x"], ShapeDtypeStruct)
        assert result["x"].shape == (2, 3)
