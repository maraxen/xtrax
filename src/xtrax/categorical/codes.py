"""Compact categorical code dtypes and validated encoding.

Codes are stored in the smallest unsigned dtype that holds ``k`` categories.
Arithmetic never happens in that storage dtype: callers cast to int32 first
(spec §4.1), so a uint8 code can never wrap inside an index computation.

Spec: .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (§3, §3.1)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Int32, UInt

_INT32_MAX = 2**31 - 1
_INT8_MAX = 127
_INT16_MAX = 32767


def _as_count(value: object, name: str) -> int:
    """Validate an integer count argument (rejects bool and non-integers)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def code_dtype(n_categories: int, *, allow_uint4: bool = False) -> jnp.dtype:
    """Return the storage dtype for codes drawn from ``n_categories`` values.

    Args:
        n_categories: Number of categories ``k``; codes lie in ``[0, k)``.
        allow_uint4: Reserved for Stage 1b (uint4 storage, gated by the GPU
            probe). Stage 1 rejects it.

    Returns:
        ``uint8`` if ``k <= 256``, ``uint16`` if ``k <= 65536``, else ``int32``.

    Raises:
        ValueError: If ``k < 1`` or ``k`` exceeds the int32 range.
        NotImplementedError: If ``allow_uint4`` is True (Stage 1b only).
    """
    k = _as_count(n_categories, "n_categories")
    if k < 1:
        raise ValueError(f"n_categories must be >= 1, got {k}")
    if k > _INT32_MAX:
        raise ValueError(f"n_categories={k} exceeds the int32 code range")
    if allow_uint4:
        raise NotImplementedError(
            "uint4 codes are Stage 1b only (gated by the GPU probe O5); call with allow_uint4=False"
        )
    if k <= 256:
        return jnp.dtype(jnp.uint8)
    if k <= 65536:
        return jnp.dtype(jnp.uint16)
    return jnp.dtype(jnp.int32)


def weight_dtype(max_weight: int) -> jnp.dtype:
    """Return the smallest signed integer dtype that holds ``max_weight``.

    Args:
        max_weight: Largest weight value that will be stored.

    Returns:
        ``int8`` if ``<= 127``, ``int16`` if ``<= 32767``, else ``int32``.

    Raises:
        ValueError: If ``max_weight`` is negative or exceeds the int32 range.
    """
    w = _as_count(max_weight, "max_weight")
    if w < 0:
        raise ValueError(f"max_weight must be >= 0, got {w}")
    if w <= _INT8_MAX:
        return jnp.dtype(jnp.int8)
    if w <= _INT16_MAX:
        return jnp.dtype(jnp.int16)
    if w <= _INT32_MAX:
        return jnp.dtype(jnp.int32)
    raise ValueError(f"max_weight={w} exceeds the int32 range")


class CategoricalCodes(eqx.Module):
    """Hard categorical codes for ``P`` variables over ``T`` frames.

    Attributes:
        codes: Codes of shape ``(P, T)``. Dtype equals
            ``code_dtype(n_categories)``; built by ``encode_codes``.
        n_categories: Number of categories ``k`` shared by every variable.
    """

    codes: UInt[Array, "P T"] | Int32[Array, "P T"]  # noqa: F722 (jaxtyping shape string)
    n_categories: int = eqx.field(static=True)


def encode_codes(codes: ArrayLike, n_categories: int) -> CategoricalCodes:
    """Validate and cast integer codes to their compact storage dtype.

    Never relabels or compacts: code values are kept exactly as given.

    Args:
        codes: Integer codes of shape ``(P, T)``.
        n_categories: Number of categories ``k``; every code must lie in
            ``[0, k)``.

    Returns:
        A ``CategoricalCodes`` whose ``codes`` have dtype
        ``code_dtype(n_categories)``.

    Raises:
        ValueError: If the input is not 2-D, or (for concrete input) any code
            lies outside ``[0, k)``.
        TypeError: If the input is not integer-valued (concrete input only).
    """
    k = _as_count(n_categories, "n_categories")
    dt = code_dtype(k)
    if isinstance(codes, jax.core.Tracer):
        if codes.ndim != 2:
            raise ValueError(f"codes must be 2-D (P, T), got shape {codes.shape}")
        arr = codes
    else:
        host = np.asarray(codes)
        if host.ndim != 2:
            raise ValueError(f"codes must be 2-D (P, T), got shape {host.shape}")
        if not np.issubdtype(host.dtype, np.integer):
            raise TypeError(f"codes must be integer-valued, got dtype {host.dtype}")
        if host.size and (host.min() < 0 or host.max() >= k):
            raise ValueError(
                f"codes must lie in [0, {k}); found range [{host.min()}, {host.max()}]"
            )
        arr = jnp.asarray(host)
    return CategoricalCodes(codes=arr.astype(dt), n_categories=k)
