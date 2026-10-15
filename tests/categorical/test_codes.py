"""Gate S1-G1: code and weight dtype rules, and encode_codes validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.categorical import CategoricalCodes, code_dtype, encode_codes, weight_dtype


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, jnp.uint8),
        (16, jnp.uint8),
        (17, jnp.uint8),
        (256, jnp.uint8),
        (257, jnp.uint16),
        (65536, jnp.uint16),
        (65537, jnp.int32),
    ],
)
def test_code_dtype_boundaries(k: int, expected: object) -> None:
    assert code_dtype(k) == jnp.dtype(expected)


def test_code_dtype_rejects_uint4_in_stage1() -> None:
    with pytest.raises(NotImplementedError, match="Stage 1b"):
        code_dtype(16, allow_uint4=True)


@pytest.mark.parametrize("k", [0, -1])
def test_code_dtype_rejects_nonpositive(k: int) -> None:
    with pytest.raises(ValueError, match="n_categories"):
        code_dtype(k)


def test_code_dtype_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        code_dtype(4.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("max_weight", "expected"),
    [
        (0, jnp.int8),
        (127, jnp.int8),
        (128, jnp.int16),
        (32767, jnp.int16),
        (32768, jnp.int32),
    ],
)
def test_weight_dtype_boundaries(max_weight: int, expected: object) -> None:
    assert weight_dtype(max_weight) == jnp.dtype(expected)


def test_weight_dtype_rejects_negative() -> None:
    with pytest.raises(ValueError, match="max_weight"):
        weight_dtype(-1)


def test_encode_codes_casts_to_compact_dtype_without_relabel() -> None:
    raw = np.array([[0, 2, 1, 2], [3, 3, 0, 1]], dtype=np.int64)
    enc = encode_codes(raw, n_categories=4)
    assert isinstance(enc, CategoricalCodes)
    assert enc.codes.dtype == jnp.uint8
    assert enc.n_categories == 4
    np.testing.assert_array_equal(np.asarray(enc.codes), raw)


def test_encode_codes_uses_uint16_above_256() -> None:
    raw = np.array([[0, 299]], dtype=np.int32)
    enc = encode_codes(raw, n_categories=300)
    assert enc.codes.dtype == jnp.uint16
    np.testing.assert_array_equal(np.asarray(enc.codes), raw)


@pytest.mark.parametrize("bad", [[[0, 4]], [[-1, 0]]])
def test_encode_codes_rejects_out_of_range_concrete(bad: list) -> None:
    with pytest.raises(ValueError, match=r"\[0, 4\)"):
        encode_codes(np.array(bad), n_categories=4)


def test_encode_codes_rejects_non_integer_dtype() -> None:
    with pytest.raises(TypeError, match="integer"):
        encode_codes(np.array([[0.0, 1.0]]), n_categories=2)


@pytest.mark.parametrize("shape", [(4,), (2, 2, 2)])
def test_encode_codes_rejects_non_2d(shape: tuple) -> None:
    with pytest.raises(ValueError, match="2-D"):
        encode_codes(np.zeros(shape, dtype=np.int32), n_categories=2)


def test_encode_codes_traced_input_keeps_dtype_rule() -> None:
    def f(x: jax.Array) -> jax.Array:
        return encode_codes(x, n_categories=3).codes

    out = jax.jit(f)(jnp.zeros((2, 5), dtype=jnp.int32))
    assert out.dtype == jnp.uint8
    assert out.shape == (2, 5)
