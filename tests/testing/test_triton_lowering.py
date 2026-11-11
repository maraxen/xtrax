"""Tests for offline CUDA Triton lowering verifier (Debt #3197)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from xtrax.testing import (
    TritonLoweringError,
    TritonLoweringResult,
    verify_triton_lowering,
)


def _scale_body(x_ref, o_ref):
    o_ref[...] = x_ref[...] * 3.0


def scale_kernel(x):
    return pl.pallas_call(
        _scale_body,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        compiler_params=plt.CompilerParams(num_warps=4, num_stages=3),
    )(x)


def test_verify_triton_lowering_success():
    """Positive control: verify_triton_lowering succeeds on a valid Triton Pallas kernel."""
    x = jax.ShapeDtypeStruct((32,), jnp.float32)
    result = verify_triton_lowering(
        scale_kernel,
        x,
        expected_grid=(1, 1, 1),
        expected_num_warps=4,
        expected_num_stages=3,
    )
    assert isinstance(result, TritonLoweringResult)
    assert "@__gpu$xla.gpu.triton" in result.hlo_text
    assert result.grid == (1, 1, 1)
    assert result.num_warps == 4
    assert result.num_stages == 3
    assert result.kernel_name == "_scale_body"


def test_verify_triton_lowering_rejects_non_triton():
    """Negative control: non-Triton elementwise function raises TritonLoweringError."""

    def plain_fn(x):
        return x + 1.0

    x = jax.ShapeDtypeStruct((32,), jnp.float32)
    with pytest.raises(TritonLoweringError, match="@__gpu\\$xla\\.gpu\\.triton"):
        verify_triton_lowering(plain_fn, x)


def test_verify_triton_lowering_catches_invalid_grid():
    """Negative control: asserting an incorrect grid raises TritonLoweringError."""
    x = jax.ShapeDtypeStruct((32,), jnp.float32)
    with pytest.raises(TritonLoweringError, match="grid mismatch"):
        verify_triton_lowering(
            scale_kernel,
            x,
            expected_grid=(2, 1, 1),
        )


def test_verify_triton_lowering_catches_invalid_warps():
    """Negative control: asserting an incorrect num_warps raises TritonLoweringError."""
    x = jax.ShapeDtypeStruct((32,), jnp.float32)
    with pytest.raises(TritonLoweringError, match="num_warps mismatch"):
        verify_triton_lowering(
            scale_kernel,
            x,
            expected_num_warps=8,
        )


def test_verify_triton_lowering_catches_invalid_stages():
    """Negative control: asserting an incorrect num_stages raises TritonLoweringError."""
    x = jax.ShapeDtypeStruct((32,), jnp.float32)
    with pytest.raises(TritonLoweringError, match="num_stages mismatch"):
        verify_triton_lowering(
            scale_kernel,
            x,
            expected_num_stages=2,
        )
