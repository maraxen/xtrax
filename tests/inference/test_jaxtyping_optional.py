"""Jaxtyping-optional gate: prove inference MVP works without jaxtyping (E1.7 / AC7)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jax import ShapeDtypeStruct

from xtrax.inference import BundleSchema, infer_bundle

# ---------------------------------------------------------------------------
# AC7.1: Static assertion — no unconditional jaxtyping imports in inference
# ---------------------------------------------------------------------------


class TestNoDirectJaxtypingImports:
    """Verify that xtrax.inference source files do NOT directly import jaxtyping."""

    def test_static_no_jaxtyping_import_in_inference_sources(self):
        """
        Verify xtrax.inference/ does not directly import jaxtyping.

        Note: jaxtyping may be a transitive dependency (e.g., via xtrax.tiling),
        but the inference module itself must not depend on it directly. This
        ensures the inference API is not on the critical path for jaxtyping
        availability.
        """
        # Locate repo root: from this test file, go up 3 levels to the worktree root
        test_file = Path(__file__).resolve()
        worktree_root = test_file.parent.parent.parent
        inference_src = worktree_root / "src" / "xtrax" / "inference"

        assert inference_src.exists(), f"Inference source dir not found: {inference_src}"

        py_files = sorted(inference_src.glob("*.py"))
        assert py_files, f"No .py files found in {inference_src}"

        forbidden_patterns = [
            "import jaxtyping",
            "from jaxtyping",
        ]

        offending_files = []
        for py_file in py_files:
            content = py_file.read_text()
            for pattern in forbidden_patterns:
                if pattern in content:
                    offending_files.append((py_file.name, pattern))

        if offending_files:
            msg = "Found direct jaxtyping imports in inference sources:\n"
            for fname, pattern in offending_files:
                msg += f"  {fname}: '{pattern}'\n"
            pytest.fail(msg)


# ---------------------------------------------------------------------------
# AC7.2: Runtime assertion — infer_bundle works with jaxtyping unavailable
# ---------------------------------------------------------------------------


class TestInferBundleWithoutJaxtyping:
    """Verify that infer_bundle succeeds when jaxtyping is blocked (import raises ImportError)."""

    @pytest.fixture(autouse=True)
    def block_jaxtyping_before_tests(self):
        """Block jaxtyping in sys.modules before test, restore after."""
        original_jaxtyping = sys.modules.pop("jaxtyping", None)
        sys.modules["jaxtyping"] = None  # type: ignore
        yield
        # Cleanup: restore or remove
        if original_jaxtyping is not None:
            sys.modules["jaxtyping"] = original_jaxtyping
        else:
            sys.modules.pop("jaxtyping", None)

    def test_fixture_blocks_jaxtyping_import(self):
        """Sanity check: verify that the fixture actually blocks jaxtyping imports."""
        with pytest.raises(ImportError):
            import jaxtyping  # noqa: F401

    def test_infer_bundle_works_without_jaxtyping(self):
        """Verify infer_bundle returns correct types when jaxtyping is blocked."""

        # Define a simple function
        def simple_fn(x: Any) -> dict[str, Any]:
            return {"result": x * 2}

        # Abstract inputs: one input with shape (4, 3) and dtype float32
        abstract_inputs = [ShapeDtypeStruct((4, 3), np.float32)]

        # Call infer_bundle — should NOT fail even though jaxtyping is blocked
        schema, axes = infer_bundle(simple_fn, abstract_inputs)

        # Assertions: verify return types and contents
        assert isinstance(schema, BundleSchema), f"Expected BundleSchema, got {type(schema)}"
        assert isinstance(axes, list), f"Expected list for axes, got {type(axes)}"

        # Verify schema has fields dict and it contains 'result'
        assert hasattr(schema, "fields"), "BundleSchema missing 'fields'"
        assert "result" in schema.fields, (
            f"Expected 'result' in schema.fields, got {schema.fields.keys()}"
        )

        # Verify the output field has the correct shape and dtype
        result_spec = schema.fields["result"]
        assert result_spec.shape == (4, 3), f"Expected shape (4, 3), got {result_spec.shape}"
        assert result_spec.dtype == np.float32, f"Expected dtype float32, got {result_spec.dtype}"

    def test_infer_bundle_multiple_inputs(self):
        """Verify infer_bundle with multiple inputs works without jaxtyping."""

        def multi_input_fn(x: Any, y: Any) -> dict[str, Any]:
            return {"sum": x + y, "product": x * y}

        # Abstract inputs: two inputs
        abstract_inputs = [
            ShapeDtypeStruct((4, 3), np.float32),
            ShapeDtypeStruct((4, 3), np.float32),
        ]

        # Call infer_bundle
        schema, axes = infer_bundle(multi_input_fn, abstract_inputs)

        # Assertions
        assert isinstance(schema, BundleSchema)
        assert isinstance(axes, list)
        assert "sum" in schema.fields
        assert "product" in schema.fields
        assert schema.fields["sum"].shape == (4, 3)
        assert schema.fields["product"].shape == (4, 3)
