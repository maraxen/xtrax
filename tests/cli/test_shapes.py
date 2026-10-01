"""Tests for xtrax.cli.shapes module.

Tests the parse_shapes function with various input formats and error cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.cli.errors import ShapeParseError
from xtrax.cli.shapes import parse_shapes


class TestParseShapesSingleEntry:
    """Tests for parsing single shape entries."""

    def test_simple_2d_float32(self):
        """Parse a simple 2D float32 shape."""
        result = parse_shapes("x=(4,3)f32")
        assert len(result) == 1
        assert "x" in result
        spec = result["x"]
        assert spec.shape == (4, 3)
        assert spec.dtype == np.float32

    def test_1d_float32(self):
        """Parse a 1D float32 shape."""
        result = parse_shapes("x=(4,)f32")
        assert len(result) == 1
        assert "x" in result
        spec = result["x"]
        assert spec.shape == (4,)
        assert spec.dtype == np.float32

    def test_scalar_float32(self):
        """Parse a scalar (empty shape) float32."""
        result = parse_shapes("x=()f32")
        assert len(result) == 1
        assert "x" in result
        spec = result["x"]
        assert spec.shape == ()
        assert spec.dtype == np.float32


class TestParseShapesMultipleEntries:
    """Tests for parsing multiple shape entries."""

    def test_two_entries(self):
        """Parse multiple space-separated entries."""
        result = parse_shapes("x=(4,3)f32 mask=(4,)bool")
        assert len(result) == 2

        x_spec = result["x"]
        assert x_spec.shape == (4, 3)
        assert x_spec.dtype == np.float32

        mask_spec = result["mask"]
        assert mask_spec.shape == (4,)
        assert mask_spec.dtype == np.bool_

    def test_three_entries_mixed_dtypes(self):
        """Parse three entries with different dtypes."""
        result = parse_shapes("x=(2,3)f32 y=(5,)i32 z=()bool")
        assert len(result) == 3

        assert result["x"].shape == (2, 3)
        assert result["x"].dtype == np.float32

        assert result["y"].shape == (5,)
        assert result["y"].dtype == np.int32

        assert result["z"].shape == ()
        assert result["z"].dtype == np.bool_


class TestParseShapesDtypes:
    """Tests for various dtype specifications."""

    def test_f32_dtype(self):
        """Parse f32 (float32) dtype."""
        result = parse_shapes("x=(2,)f32")
        assert result["x"].dtype == np.float32

    def test_f64_dtype(self):
        """Parse f64 (float64) dtype."""
        result = parse_shapes("x=(2,)f64")
        assert result["x"].dtype == np.float64

    def test_i32_dtype(self):
        """Parse i32 (int32) dtype."""
        result = parse_shapes("x=(2,)i32")
        assert result["x"].dtype == np.int32

    def test_bool_dtype(self):
        """Parse bool (boolean) dtype."""
        result = parse_shapes("x=(2,)bool")
        assert result["x"].dtype == np.bool_


class TestParseShapesErrors:
    """Tests for error cases and malformed input."""

    def test_missing_equals(self):
        """Raise ShapeParseError when '=' is missing."""
        with pytest.raises(ShapeParseError):
            parse_shapes("x")

    def test_missing_open_paren(self):
        """Raise ShapeParseError when opening paren is missing."""
        with pytest.raises(ShapeParseError):
            parse_shapes("x=4,3)f32")

    def test_missing_close_paren(self):
        """Raise ShapeParseError when closing paren is missing."""
        with pytest.raises(ShapeParseError):
            parse_shapes("x=(4,3f32")

    def test_non_integer_dimension(self):
        """Raise ShapeParseError when a dimension is not an integer."""
        with pytest.raises(ShapeParseError):
            parse_shapes("x=(4,a)f32")

    def test_unknown_dtype(self):
        """Raise ShapeParseError when dtype is unknown."""
        with pytest.raises(ShapeParseError):
            parse_shapes("x=(4,3)f128")

    def test_empty_string(self):
        """Raise ShapeParseError for empty input string."""
        with pytest.raises(ShapeParseError):
            parse_shapes("")

    def test_whitespace_only(self):
        """Raise ShapeParseError for whitespace-only input."""
        with pytest.raises(ShapeParseError):
            parse_shapes("   ")

    def test_malformed_middle_entry(self):
        """Raise ShapeParseError when middle entry in multi-entry string is malformed."""
        with pytest.raises(ShapeParseError):
            parse_shapes("x=(4,3)f32 invalid y=(2,)f32")


class TestParseShapesShapeDtypeStruct:
    """Tests that returned values are proper ShapeDtypeStruct objects."""

    def test_returned_type_is_shapedtypestruct(self):
        """Verify returned values are ShapeDtypeStruct instances."""
        result = parse_shapes("x=(3,4)f32")
        assert isinstance(result["x"], ShapeDtypeStruct)

    def test_dict_return_type(self):
        """Verify return type is a dict."""
        result = parse_shapes("x=(3,4)f32")
        assert isinstance(result, dict)

    def test_dict_keys_are_names(self):
        """Verify dictionary keys match the provided names."""
        result = parse_shapes("a=(1,)f32 b=(2,)i32 c=(3,)bool")
        assert set(result.keys()) == {"a", "b", "c"}


class TestParseShapesDtypeForms:
    """#5174: the documented `<dtype>` read as literal syntax and was rejected."""

    @pytest.mark.parametrize(
        ("spec", "dtype"),
        [
            ("x=(4,3)float32", np.float32),
            ("x=(4,3)float64", np.float64),
            ("x=(4,3)int32", np.int32),
            ("x=(4,3)<f32>", np.float32),
            ("x=(4,3)<float32>", np.float32),
            ("x=(4,3)<bool>", np.bool_),
        ],
    )
    def test_long_aliases_and_bracketed_form(self, spec, dtype):
        assert parse_shapes(spec)["x"].dtype == dtype

    @pytest.mark.parametrize("spec", ["x=(4,3)<f32", "x=(4,3)f32>", "x=(4,3)<>", "x=(4,3)<f128>"])
    def test_malformed_brackets_and_unknown_bracketed_dtype_still_rejected(self, spec):
        with pytest.raises(ShapeParseError):
            parse_shapes(spec)

    def test_missing_dtype_names_the_problem(self):
        with pytest.raises(ShapeParseError, match="missing dtype"):
            parse_shapes("x=(4,3)")

    @pytest.mark.parametrize("spec", ["", "x", "=(4,)f32", "x=(4,)", "x=4,3)f32", "1x=(4,)f32"])
    def test_every_error_shows_a_concrete_example_not_a_placeholder(self, spec):
        with pytest.raises(ShapeParseError) as info:
            parse_shapes(spec)
        assert "<dtype>" not in str(info.value)
        assert "e.g. x=(4,3)f32" in str(info.value)


def test_cli_accepts_bracketed_long_dtype_end_to_end():
    """Subprocess-level: the documented form works through the real `xtrax explain`."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    code = "import sys; from xtrax.cli import main; sys.argv = ['xtrax', *sys.argv[1:]]; main()"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "explain",
            "--fn",
            "tests.cli.test_entrypoint:decorated_fn",
            "--shapes",
            "x=(4,)<float32>",
            "--fmt",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout)["total_axes"] == 1
