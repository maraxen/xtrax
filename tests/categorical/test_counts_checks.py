"""Gate S1-G3 (exact path): preconditions C1-C3 and the pair range.

The checks are on-device reductions into ``CooccurrenceCounts.status``; there is
no host callback. An eager call (no traced input) raises ``ValueError`` before the
kernel runs. Under ``jax.jit``/``jax.vmap`` nothing raises: the failing bits come
back in ``status``, and ``raise_if_invalid()`` raises once the result is read
outside the trace. Each failing case is also run with ``check=False`` to show the
check is what sets the bit.

The ``jax.jit(jax.grad(...))`` variant of S1-G3 targets the differentiable
and float-strategy paths and lives in ``test_counts_grad.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.categorical import (
    STATUS_CODES_A,
    STATUS_CODES_B,
    STATUS_NEGATIVE_WEIGHT,
    STATUS_PAIRS_RANGE,
    STATUS_TOTAL_OVERFLOW,
    CategoricalCodes,
    cooccurrence_counts,
    encode_codes,
)


def _raw_codes(values: np.ndarray, n_categories: int) -> CategoricalCodes:
    """Build codes without ``encode_codes``'s host range check, to test the device one."""
    return CategoricalCodes(codes=jnp.asarray(values.astype(np.uint8)), n_categories=n_categories)


def _run_jit(ca: CategoricalCodes, cb: CategoricalCodes, w: np.ndarray, *, check: bool = True):
    def run(a, b, weights):
        return cooccurrence_counts(a, b, weights=weights, method="matmul", check=check)

    out = jax.jit(run)(ca, cb, jnp.asarray(w))
    return jax.block_until_ready(out)


def _status(out) -> np.ndarray:
    return np.asarray(out.status)


def test_clean_inputs_pass_under_jit_and_match_eager() -> None:
    rng = np.random.default_rng(3)
    xa = rng.integers(0, 4, size=(2, 50))
    w = rng.integers(0, 100, size=(1, 50)).astype(np.int32)
    ca = encode_codes(xa, 4)
    out = _run_jit(ca, ca, w)
    eager = cooccurrence_counts(ca, ca, weights=w, method="matmul")
    assert np.array_equal(np.asarray(out.joint), np.asarray(eager.joint))
    assert np.array_equal(_status(out), np.zeros(1, np.int32))
    assert np.array_equal(_status(eager), np.zeros(1, np.int32))
    out.raise_if_invalid()


def test_checks_compile_to_no_host_callback() -> None:
    """The point of the status design: no callback, so no device-to-host stall."""
    ca = encode_codes(np.zeros((2, 16), dtype=np.int64), 3)
    w = jnp.ones((1, 16), dtype=jnp.int32)
    hlo = (
        jax.jit(lambda a, w_: cooccurrence_counts(a, a, weights=w_, method="matmul"))
        .lower(ca, w)
        .as_text()
    )
    assert "callback" not in hlo.lower()


def test_c1_out_of_range_code_sets_status_under_jit() -> None:
    xa = np.zeros((2, 16), dtype=np.int64)
    xa[1, 7] = 3  # K=3, so code 3 is out of range
    ca = _raw_codes(xa, 3)
    cb = encode_codes(np.zeros((1, 16), dtype=np.int64), 3)
    w = np.ones((1, 16), dtype=np.int32)
    out = _run_jit(ca, cb, w)
    assert _status(out)[0] == STATUS_CODES_A
    with pytest.raises(ValueError, match="C1 failed: codes_a"):
        out.raise_if_invalid()
    assert _status(_run_jit(ca, cb, w, check=False))[0] == 0


def test_c1_out_of_range_code_raises_eagerly() -> None:
    xa = np.zeros((2, 16), dtype=np.int64)
    xa[1, 7] = 3
    ca = _raw_codes(xa, 3)
    cb = encode_codes(np.zeros((1, 16), dtype=np.int64), 3)
    with pytest.raises(ValueError, match="C1 failed: codes_a"):
        cooccurrence_counts(ca, cb, weights=np.ones((1, 16), np.int32), method="matmul")


def test_c1_out_of_range_code_on_b_side_sets_status_under_jit() -> None:
    xb = np.zeros((1, 16), dtype=np.int64)
    xb[0, 0] = 5
    ca = encode_codes(np.zeros((1, 16), dtype=np.int64), 4)
    cb = _raw_codes(xb, 4)
    w = np.ones((1, 16), dtype=np.int32)
    out = _run_jit(ca, cb, w)
    assert _status(out)[0] == STATUS_CODES_B
    with pytest.raises(ValueError, match="codes_b"):
        out.raise_if_invalid()


def test_c2_negative_weight_sets_status_under_jit() -> None:
    ca = encode_codes(np.zeros((1, 8), dtype=np.int64), 2)
    w = np.ones((1, 8), dtype=np.int32)
    w[0, 3] = -1
    out = _run_jit(ca, ca, w)
    assert _status(out)[0] & STATUS_NEGATIVE_WEIGHT
    with pytest.raises(ValueError, match="C2 failed"):
        out.raise_if_invalid()


def test_c2_negative_weight_raises_eagerly() -> None:
    ca = encode_codes(np.zeros((1, 8), dtype=np.int64), 2)
    w = np.ones((1, 8), dtype=np.int32)
    w[0, 3] = -1
    with pytest.raises(ValueError, match="C2 failed"):
        cooccurrence_counts(ca, ca, weights=w, method="matmul")


def test_c2_negative_weight_is_silent_when_check_is_off() -> None:
    """C4: check=False skips the checks: status stays 0 and nothing raises (results undefined)."""
    ca = encode_codes(np.zeros((1, 8), dtype=np.int64), 2)
    w = np.ones((1, 8), dtype=np.int32)
    w[0, 3] = -1
    assert _status(_run_jit(ca, ca, w, check=False))[0] == 0
    eager = cooccurrence_counts(ca, ca, weights=w, method="matmul", check=False)
    assert _status(eager)[0] == 0


def test_c3_total_of_two_to_the_31_sets_status_under_jit() -> None:
    """Two frames of 2**30 sum to exactly 2**31, which overflows int32."""
    ca = encode_codes(np.zeros((1, 2), dtype=np.int64), 2)
    w = np.full((1, 2), 2**30, dtype=np.int32)
    out = _run_jit(ca, ca, w)
    assert _status(out)[0] == STATUS_TOTAL_OVERFLOW
    with pytest.raises(ValueError, match="C3 failed"):
        out.raise_if_invalid()


def test_c3_total_just_below_two_to_the_31_is_exact() -> None:
    """A single frame of 2**31 - 1 is the largest legal total, and must be exact."""
    ca = encode_codes(np.zeros((1, 1), dtype=np.int64), 2)
    w = np.array([[2**31 - 1]], dtype=np.int32)
    out = _run_jit(ca, ca, w)
    assert _status(out)[0] == 0
    assert int(np.asarray(out.total)[0]) == 2**31 - 1
    assert int(np.asarray(out.joint)[0, 0, 0, 0, 0]) == 2**31 - 1


def test_c3_overflow_is_per_replicate() -> None:
    """The bound is per replicate: only the overflowing replicate carries the bit."""
    ca = encode_codes(np.zeros((1, 2), dtype=np.int64), 2)
    w = np.stack([np.array([1, 1], dtype=np.int32), np.full(2, 2**30, dtype=np.int32)])
    out = _run_jit(ca, ca, w)
    assert np.array_equal(_status(out), np.array([0, STATUS_TOTAL_OVERFLOW], np.int32))
    with pytest.raises(ValueError, match="C3 failed"):
        cooccurrence_counts(ca, ca, weights=w, method="matmul")


def test_status_under_vmap_is_per_batch_element() -> None:
    """Under vmap each batch element gets its own status word; nothing raises."""
    ca = encode_codes(np.zeros((1, 4), dtype=np.int64), 2)
    w = np.ones((3, 1, 4), dtype=np.int32)
    w[1, 0, 2] = -1

    def run(w_):
        return cooccurrence_counts(ca, ca, weights=w_, method="matmul").status

    status = np.asarray(jax.jit(jax.vmap(run))(jnp.asarray(w)))
    assert status.shape == (3, 1)
    assert np.array_equal(status[:, 0], np.array([0, STATUS_NEGATIVE_WEIGHT, 0], np.int32))


def test_traced_pair_index_out_of_range_sets_status_under_jit() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    w = np.ones((1, 8), dtype=np.int32)
    pairs = jnp.asarray(np.array([[0, 1], [2, 0]], dtype=np.int32))  # a=2 is out of range

    def run(p):
        return cooccurrence_counts(ca, ca, weights=w, pairs=p, method="matmul")

    out = jax.block_until_ready(jax.jit(run)(pairs))
    assert _status(out)[0] == STATUS_PAIRS_RANGE
    with pytest.raises(ValueError, match="pairs index out of range"):
        out.raise_if_invalid()


def test_traced_pair_index_in_range_passes_under_jit() -> None:
    ca = encode_codes(np.zeros((2, 8), dtype=np.int64), 3)
    w = np.ones((1, 8), dtype=np.int32)
    pairs = jnp.asarray(np.array([[0, 1], [1, 0]], dtype=np.int32))

    def run(p):
        return cooccurrence_counts(ca, ca, weights=w, pairs=p, method="matmul")

    out = jax.block_until_ready(jax.jit(run)(pairs))
    assert out.joint.shape == (1, 2, 3, 3)
    assert _status(out)[0] == 0
