"""Dtype gating over closure-held leaves, and the f64 rejection.

``abstract_inputs`` covers only what the caller passes at trace time. A model's
weights ride along in the exported callable's closure, so a gate that reads only
the arguments never sees them -- which is the commonest shape by far.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.export.safety import (
    DtypeNotSupportedError,
    check_export_safety,
    validate_export_safe,
)
from xtrax.export.targets import (
    METAL_SPIRV,
    NATIVE,
    VULKAN_SPIRV,
    WASM32,
    Target,
    VerificationLevel,
)
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

NARROW = Target(
    name="narrow",
    iree_backend="llvm-cpu",
    verification_level=VerificationLevel.CODEGEN_ONLY,
    supported_dtypes=frozenset({"f32", "i32", "bool"}),
    optional_dtypes=frozenset({"f16"}),
    optional_dtype_features={"f16": "shader-f16"},
)

F32_INPUT = (jax.ShapeDtypeStruct((4,), jnp.float32),)


def _decisions():
    return [
        AxisDecision(
            spec=AxisSpec(name="batch", cardinality=4, default_batch_size=0),
            batch_size=0,
            reasoning="test",
            strategy=Vmap(),
        )
    ]


class _WeightHolder(eqx.Module):
    """A model whose weight is never an argument -- exactly the missed case."""

    w: jax.Array

    def __call__(self, x):
        return x * self.w.astype(x.dtype)


class TestClosureLeafScan:
    def test_closure_held_bad_dtype_is_caught(self):
        """The leaf appears in no abstract input, so only a closure scan finds it."""
        model = _WeightHolder(w=jnp.ones((4,), dtype=jnp.bfloat16))
        blockers = check_export_safety(_decisions(), {}, F32_INPUT, model, NARROW)
        assert blockers, "a bf16 weight in the closure must be reported"
        assert any("bf16" in b.detail for b in blockers)

    def test_the_blocker_names_the_leaf_keypath(self):
        model = _WeightHolder(w=jnp.ones((4,), dtype=jnp.bfloat16))
        blockers = check_export_safety(_decisions(), {}, F32_INPUT, model, NARROW)
        assert any(b.axis.startswith("closure") for b in blockers), [b.axis for b in blockers]

    def test_validate_raises_for_a_closure_only_violation(self):
        model = _WeightHolder(w=jnp.ones((4,), dtype=jnp.bfloat16))
        with pytest.raises(DtypeNotSupportedError, match="bf16"):
            validate_export_safe(_decisions(), {}, F32_INPUT, model, NARROW)

    def test_a_supported_closure_dtype_passes(self):
        model = _WeightHolder(w=jnp.ones((4,), dtype=jnp.float32))
        assert check_export_safety(_decisions(), {}, F32_INPUT, model, NARROW) == []

    def test_a_plain_function_holds_no_leaves(self):
        assert check_export_safety(_decisions(), {}, F32_INPUT, lambda x: x, NARROW) == []

    def test_closure_leaves_are_gated_by_requested_features_too(self):
        """The optional-dtype rule applies wherever the leaf was found."""
        model = _WeightHolder(w=jnp.ones((4,), dtype=jnp.float16))
        assert check_export_safety(_decisions(), {}, F32_INPUT, model, NARROW), (
            "f16 needs a feature"
        )
        unlocked = check_export_safety(
            _decisions(), {}, F32_INPUT, model, NARROW, request_features=frozenset({"shader-f16"})
        )
        assert unlocked == []

    def test_both_an_argument_and_a_closure_leaf_are_reported(self):
        """Blockers are collected, so one pass shows every offending leaf."""
        model = _WeightHolder(w=jnp.ones((4,), dtype=jnp.bfloat16))
        inputs = (jax.ShapeDtypeStruct((4,), jnp.float64),)
        blockers = check_export_safety(_decisions(), {}, inputs, model, NARROW)
        where = {b.axis for b in blockers}
        assert any(w.startswith("abstract_inputs") for w in where), where
        assert any(w.startswith("closure") for w in where), where


class TestF64Rejection:
    @pytest.mark.parametrize("target", [NATIVE, WASM32, VULKAN_SPIRV, METAL_SPIRV])
    def test_every_registered_target_rejects_f64(self, target):
        inputs = (jax.ShapeDtypeStruct((4,), jnp.float64),)
        with pytest.raises(DtypeNotSupportedError, match="f64"):
            validate_export_safe(_decisions(), {}, inputs, lambda x: x, target)

    def test_the_message_explains_the_silent_demotion(self):
        """A bare "unsupported" sends the reader looking for a flag that does not exist."""
        inputs = (jax.ShapeDtypeStruct((4,), jnp.float64),)
        with pytest.raises(DtypeNotSupportedError) as excinfo:
            validate_export_safe(_decisions(), {}, inputs, lambda x: x, NATIVE)
        message = str(excinfo.value)
        assert "demotes f64 to f32" in message
        assert "signature" in message

    def test_other_dtypes_do_not_get_the_f64_note(self):
        inputs = (jax.ShapeDtypeStruct((4,), jnp.complex64),)
        with pytest.raises(DtypeNotSupportedError) as excinfo:
            validate_export_safe(_decisions(), {}, inputs, lambda x: x, NATIVE)
        assert "demotes f64" not in str(excinfo.value)
