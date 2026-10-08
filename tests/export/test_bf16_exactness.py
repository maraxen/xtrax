"""Does casting a bf16 model to f32 change what it computes?

Deliberately a different comparison from the parity check inside
``export_pipeline``. That one is f32-cast-against-f32-cast, so it is structurally
incapable of detecting this: bf16 -> f32 is an exact widening, and both sides
would move together. Here the reference is the *original bf16 model* evaluated in
plain JAX, with no export involved at all, so the two sides genuinely differ in
precision and the tolerance has to absorb bf16's ~3 decimal digits.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.export.pipeline import export_pipeline
from xtrax.export.targets import NATIVE
from xtrax.tiling.plan import AxisDecision, AxisSpec
from xtrax.tiling.strategy import Vmap

# bf16 carries ~8 bits of mantissa, so agreement to f32's 1e-5 is not on offer.
BF16_ATOL = 1e-2


class _Plan:
    def __init__(self, decisions):
        self.decisions = decisions


def _plan(cardinality: int) -> _Plan:
    return _Plan(
        [
            AxisDecision(
                spec=AxisSpec(name="batch", cardinality=cardinality, default_batch_size=0),
                batch_size=0,
                reasoning="test",
                strategy=Vmap(),
            )
        ]
    )


class TestBf16CastExactness:
    def test_cast_model_matches_the_original_bf16_forward_pass(self):
        pytest.importorskip("iree.compiler")
        pytest.importorskip("iree.runtime")

        weight_bf16 = jnp.asarray([0.5, 1.5, 2.5, 3.5], dtype=jnp.bfloat16)
        xs_bf16 = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.bfloat16)

        # Reference: the original bf16 model, plain JAX, never exported.
        bf16_reference = np.asarray(
            jax.vmap(lambda x, w: x * w + w)(xs_bf16, weight_bf16).astype(jnp.float32)
        )

        # Under test: the f32-cast model, exported and natively executed.
        weight_f32 = weight_bf16.astype(jnp.float32)
        xs_f32 = xs_bf16.astype(jnp.float32)

        def fn(pair):
            x, w = pair
            return x * w + w

        def reference_fn(inputs):
            return jax.vmap(fn)(inputs[0])

        stacked = (jnp.stack([xs_f32, weight_f32], axis=1),)
        abstract = (jax.ShapeDtypeStruct(stacked[0].shape, stacked[0].dtype),)

        results = export_pipeline(
            lambda row: row[0] * row[1] + row[1],
            _plan(4),
            abstract,
            stacked,
            targets=(NATIVE,),
            reference_fn=lambda inputs: jax.vmap(lambda r: r[0] * r[1] + r[1])(inputs[0]),
        )
        assert results["native"].verified is True

        from xtrax.export.compile import run_native_vmfb

        executed = np.asarray(run_native_vmfb(results["native"].path, np.asarray(stacked[0])))
        np.testing.assert_allclose(executed, bf16_reference, atol=BF16_ATOL)

    def test_f32_tolerance_would_not_have_been_appropriate(self):
        """Shows the tolerance above is doing work, not padding a passing test."""
        weight_bf16 = jnp.asarray([0.1, 0.3, 0.7, 1.1], dtype=jnp.bfloat16)
        as_f32 = weight_bf16.astype(jnp.float32)
        exact_f32 = jnp.asarray([0.1, 0.3, 0.7, 1.1], dtype=jnp.float32)

        gap = float(jnp.max(jnp.abs(as_f32 - exact_f32)))
        assert gap > 1e-5, "bf16 rounding must exceed the f32 parity tolerance"
        assert gap < BF16_ATOL, "and must sit inside the bf16 tolerance"
