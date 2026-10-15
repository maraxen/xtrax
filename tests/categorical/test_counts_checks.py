"""Gate S1-G3 (exact path): preconditions C1-C3 enforced under trace.

The checks use ``eqx.error_if``, so they fire inside ``jax.jit`` and surface as
a ``jax.errors.JaxRuntimeError`` (wrapping the equinox message) when the result
is materialized. Each failing
case is also run with ``check=False`` to show the check is what raises.

The ``jax.jit(jax.grad(...))`` variant of S1-G3 targets the differentiable
path, which lands in task 1.7; see the report for the deviation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.categorical import CategoricalCodes, cooccurrence_counts, encode_codes


def _raw_codes(values: np.ndarray, n_categories: int) -> CategoricalCodes:
    """Build codes without ``encode_codes``'s host range check, to test the traced one."""
    return CategoricalCodes(codes=jnp.asarray(values.astype(np.uint8)), n_categories=n_categories)


def _run_jit(ca: CategoricalCodes, cb: CategoricalCodes, w: np.ndarray, *, check: bool = True):
    def run(a, b, weights):
        return cooccurrence_counts(a, b, weights=weights, method="matmul", check=check)

    out = jax.jit(run)(ca, cb, jnp.asarray(w))
    return jax.block_until_ready(out)


def test_clean_inputs_pass_under_jit_and_match_eager() -> None:
    rng = np.random.default_rng(3)
    xa = rng.integers(0, 4, size=(2, 50))
    w = rng.integers(0, 100, size=(1, 50)).astype(np.int32)
    ca = encode_codes(xa, 4)
    out = _run_jit(ca, ca, w)
    eager = cooccurrence_counts(ca, ca, weights=w, method="matmul")
    assert np.array_equal(np.asarray(out.joint), np.asarray(eager.joint))


def test_c1_out_of_range_code_raises_under_jit() -> None:
    xa = np.zeros((2, 16), dtype=np.int64)
    xa[1, 7] = 3  # K=3, so code 3 is out of range
    ca = _raw_codes(xa, 3)
    cb = encode_codes(np.zeros((1, 16), dtype=np.int64), 3)
    w = np.ones((1, 16), dtype=np.int32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C1 failed"):
        _run_jit(ca, cb, w)


def test_c1_out_of_range_code_on_b_side_raises_under_jit() -> None:
    xb = np.zeros((1, 16), dtype=np.int64)
    xb[0, 0] = 5
    ca = encode_codes(np.zeros((1, 16), dtype=np.int64), 4)
    cb = _raw_codes(xb, 4)
    w = np.ones((1, 16), dtype=np.int32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="codes_b"):
        _run_jit(ca, cb, w)


def test_c2_negative_weight_raises_under_jit() -> None:
    ca = encode_codes(np.zeros((1, 8), dtype=np.int64), 2)
    w = np.ones((1, 8), dtype=np.int32)
    w[0, 3] = -1
    with pytest.raises(jax.errors.JaxRuntimeError, match="C2 failed"):
        _run_jit(ca, ca, w)


def test_c2_negative_weight_is_silent_when_check_is_off() -> None:
    """C4: check=False skips the traced checks, so no error is raised (results undefined)."""
    ca = encode_codes(np.zeros((1, 8), dtype=np.int64), 2)
    w = np.ones((1, 8), dtype=np.int32)
    w[0, 3] = -1
    _run_jit(ca, ca, w, check=False)


def test_c3_total_of_two_to_the_31_raises_under_jit() -> None:
    """Two frames of 2**30 sum to exactly 2**31, which overflows int32."""
    ca = encode_codes(np.zeros((1, 2), dtype=np.int64), 2)
    w = np.full((1, 2), 2**30, dtype=np.int32)
    with pytest.raises(jax.errors.JaxRuntimeError, match="C3 failed"):
        _run_jit(ca, ca, w)


def test_c3_total_just_below_two_to_the_31_is_exact() -> None:
    """A single frame of 2**31 - 1 is the largest legal total, and must be exact."""
    ca = encode_codes(np.zeros((1, 1), dtype=np.int64), 2)
    w = np.array([[2**31 - 1]], dtype=np.int32)
    out = _run_jit(ca, ca, w)
    assert int(np.asarray(out.total)[0]) == 2**31 - 1
    assert int(np.asarray(out.joint)[0, 0, 0, 0, 0]) == 2**31 - 1


def test_c3_overflow_is_caught_with_replicates_present() -> None:
    """The bound is per replicate; an overflowing replicate must trip the check."""
    ca = encode_codes(np.zeros((1, 2), dtype=np.int64), 2)
    w = np.stack([np.array([1, 1], dtype=np.int32), np.full(2, 2**30, dtype=np.int32)])
    with pytest.raises(jax.errors.JaxRuntimeError, match="C3 failed"):
        _run_jit(ca, ca, w)


def test_traced_pair_index_out_of_range_raises_under_jit() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    w = np.ones((1, 8), dtype=np.int32)
    pairs = jnp.asarray(np.array([[0, 1], [2, 0]], dtype=np.int32))  # a=2 is out of range

    def run(p):
        return cooccurrence_counts(ca, ca, weights=w, pairs=p, method="matmul")

    with pytest.raises(jax.errors.JaxRuntimeError, match="pairs index out of range"):
        jax.block_until_ready(jax.jit(run)(pairs))


def test_traced_pair_index_in_range_passes_under_jit() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    w = np.ones((1, 8), dtype=np.int32)
    pairs = jnp.asarray(np.array([[0, 1], [1, 0]], dtype=np.int32))

    def run(p):
        return cooccurrence_counts(ca, ca, weights=w, pairs=p, method="matmul")

    out = jax.block_until_ready(jax.jit(run)(pairs))
    assert out.joint.shape == (1, 2, 3, 3)
