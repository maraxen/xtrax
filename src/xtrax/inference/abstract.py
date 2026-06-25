"""Builder for abstract (ShapeDtypeStruct) input specifications."""

from __future__ import annotations

from typing import Any

import jax
from jax import ShapeDtypeStruct


def build_abstract_inputs(spec: Any) -> Any:
    """Convert a pytree spec to abstract inputs (ShapeDtypeStruct leaves).

    This function transforms a pytree specification where leaves are either:
    - jax.ShapeDtypeStruct: returned unchanged (identity passthrough)
    - (shape, dtype) tuple pairs: converted to ShapeDtypeStruct

    The pytree structure (dicts, tuples, etc.) is preserved.

    Args:
        spec: A pytree whose leaves are either ShapeDtypeStruct instances or
            (shape, dtype) pairs where shape is a tuple/sequence of ints and
            dtype is a numpy/jax dtype-like object.

    Returns:
        A pytree with the same structure as spec, with every leaf as a
        ShapeDtypeStruct.

    Raises:
        ValueError: If a leaf is neither ShapeDtypeStruct nor a valid
            (shape, dtype) pair, or if shape contains non-integer or
            negative dimensions.
        TypeError: If a (shape, dtype) pair is malformed (e.g., wrong
            number of elements).

    Examples:
        >>> import numpy as np
        >>> spec = {"x": ((2, 3), np.float32), "y": ((4,), np.int32)}
        >>> result = build_abstract_inputs(spec)
        >>> result["x"].shape
        (2, 3)
        >>> result["x"].dtype
        dtype('float32')

        >>> # ShapeDtypeStruct passes through unchanged
        >>> from jax import ShapeDtypeStruct
        >>> spec = {"abstract": ShapeDtypeStruct((5,), np.float64)}
        >>> result = build_abstract_inputs(spec)
        >>> result["abstract"] is spec["abstract"]
        True
    """

    def _is_shape_dtype_pair(x: Any) -> bool:
        """Check if x is a (shape, dtype) pair."""
        if not isinstance(x, tuple) or len(x) != 2:
            return False
        shape, dtype = x
        # shape must be a tuple or sequence of ints; dtype must be dtype-like
        try:
            if not isinstance(shape, (tuple, list)):
                return False
            # Check all elements in shape are integers
            for dim in shape:
                if not isinstance(dim, int):
                    return False
            # dtype must be convertible to a numpy dtype
            _ = jax.numpy.dtype(dtype)
            return True
        except (TypeError, ValueError):
            return False

    def _convert_leaf(x: Any) -> ShapeDtypeStruct:
        """Convert a single leaf to ShapeDtypeStruct."""
        # If already ShapeDtypeStruct, return unchanged
        if isinstance(x, ShapeDtypeStruct):
            return x

        # If (shape, dtype) pair, convert it
        if _is_shape_dtype_pair(x):
            shape, dtype = x
            # Normalize shape to tuple
            shape = tuple(shape)
            # Validate shape: no negative dimensions
            for i, dim in enumerate(shape):
                if dim < 0:
                    raise ValueError(
                        f"shape dimension {i} is negative: {dim}. "
                        "All shape dimensions must be non-negative."
                    )
            # Convert dtype to numpy dtype
            dtype_obj = jax.numpy.dtype(dtype)
            return ShapeDtypeStruct(shape, dtype_obj)

        # Malformed leaf
        raise ValueError(
            f"leaf is neither ShapeDtypeStruct nor (shape, dtype) pair: {type(x).__name__}. "
            "Expected ShapeDtypeStruct or a tuple of (shape, dtype)."
        )

    # Use tree_map with is_leaf to treat ShapeDtypeStruct and (shape, dtype)
    # tuples as leaves (not containers to recurse into)
    def _is_leaf(x: Any) -> bool:
        return isinstance(x, ShapeDtypeStruct) or _is_shape_dtype_pair(x)

    return jax.tree_util.tree_map(_convert_leaf, spec, is_leaf=_is_leaf)
