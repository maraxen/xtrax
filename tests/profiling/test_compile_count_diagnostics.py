"""Tests for JIT cache miss and recompile diagnostics (Debt #3195)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.profiling import assert_no_recompile_after
from xtrax.profiling.compile_count import (
    diff_jit_signature,
    explain_cache_miss,
)


def test_diff_jit_signature_identical_args():
    """Identical arguments produce no signature diffs."""
    x = jnp.ones((8, 4), dtype=jnp.float32)
    diffs = diff_jit_signature((x, "static_mode"), (x, "static_mode"))
    assert diffs == []


def test_diff_jit_signature_shape_mismatch():
    """Shape divergence is detected with leaf path and indices."""
    x1 = jnp.ones((8, 4), dtype=jnp.float32)
    x2 = jnp.ones((16, 4), dtype=jnp.float32)
    diffs = diff_jit_signature((x1,), (x2,))
    assert len(diffs) == 1
    assert "shape changed from (8, 4) to (16, 4)" in diffs[0]


def test_diff_jit_signature_dtype_mismatch():
    """Dtype divergence is detected with leaf path and indices."""
    x1 = jnp.ones((8,), dtype=jnp.float32)
    x2 = jnp.ones((8,), dtype=jnp.int32)
    diffs = diff_jit_signature((x1,), (x2,))
    assert len(diffs) == 1
    assert "dtype changed from float32 to int32" in diffs[0]


def test_diff_jit_signature_pytree_structure_mismatch():
    """PyTree structure divergence is detected."""
    x = jnp.ones((4,))
    args1 = ({"a": x},)
    args2 = ({"a": x, "b": x},)
    diffs = diff_jit_signature(args1, args2)
    assert len(diffs) == 1
    assert "PyTree structure mismatch" in diffs[0]


def test_diff_jit_signature_static_arg_mismatch():
    """Static parameter divergence is detected."""
    x = jnp.ones((4,))
    diffs = diff_jit_signature((x, "train"), (x, "eval"))
    assert len(diffs) == 1
    assert "static value changed from 'train' to 'eval'" in diffs[0]


def test_diff_jit_signature_uncomparable_object_guard():
    """Finding C6: Objects whose __eq__ raises or returns an array are safely compared."""

    class CustomObj:
        def __eq__(self, other):
            raise RuntimeError("Equality comparison forbidden")

        def __repr__(self):
            return "CustomObj()"

    diffs = diff_jit_signature((CustomObj(),), (CustomObj(),))
    assert diffs == []


def test_explain_cache_miss_formatting():
    """explain_cache_miss formats actionable message with step index."""
    x1 = jnp.ones((8,))
    x2 = jnp.ones((16,))
    msg = explain_cache_miss((x1,), (x2,), step_index=2)
    assert "XLA cache miss detected at step 2" in msg
    assert "shape changed from (8,) to (16,)" in msg


def test_assert_no_recompile_emits_shape_diff_diagnostic():
    """assert_no_recompile_after failure includes signature divergence diff."""

    @jax.jit
    def step(x):
        return x * 2.0

    args = [
        (jnp.ones((8,), dtype=jnp.float32),),  # Warmup
        (jnp.ones((16,), dtype=jnp.float32),),  # Step 1: Shape change causes recompile
    ]
    with pytest.raises(AssertionError) as exc_info:
        assert_no_recompile_after(step, args, warmup=1)

    err = str(exc_info.value)
    assert "backend compile(s)" in err
    assert "shape changed from (8,) to (16,)" in err


def test_assert_no_recompile_emits_dtype_diff_diagnostic():
    """assert_no_recompile_after failure includes dtype divergence diff."""

    @jax.jit
    def step(x):
        return x * 2

    args = [
        (jnp.ones((8,), dtype=jnp.float32),),  # Warmup
        (jnp.ones((8,), dtype=jnp.int32),),  # Step 1: Dtype change causes recompile
    ]
    with pytest.raises(AssertionError) as exc_info:
        assert_no_recompile_after(step, args, warmup=1)

    err = str(exc_info.value)
    assert "backend compile(s)" in err
    assert "dtype changed from float32 to int32" in err


def test_assert_no_recompile_warmup_zero_step1_diagnostic():
    """Finding C6: warmup=0 with recompile on step 1 emits pre-warmed cache diagnostic."""

    # An unwarmed JIT function will compile at step 1
    def fresh_fn(x):
        return x + 1.0

    jitted = jax.jit(fresh_fn)

    args = [
        (jnp.ones((4,), dtype=jnp.float32),),
    ]
    with pytest.raises(AssertionError) as exc_info:
        assert_no_recompile_after(jitted, args, warmup=0)

    err = str(exc_info.value)
    assert "Function compiled at step 1 with warmup=0 (expected pre-warmed cache)" in err
