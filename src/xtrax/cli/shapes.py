"""Parser for shape specification strings.

This module provides a parser for shape specification strings used in xtrax
command-line operations. Shape specifications define the static structure
(shape and dtype) of array inputs.

Grammar
-------

A shape specification string consists of one or more space-separated entries,
each specifying a single input array. Each entry follows the format::

    name=(d0,d1,...)<dtype>

where:

- **name**: An identifier for the input (e.g., ``x``, ``mask``, ``weights``).
  This name is used only for display and input identification purposes.
  It does NOT bind to xtrax's AxisSpec/@axis_config axis names — axis naming
  in E1 is handled positionally by synthesize_axes, which names axes as
  ``axis_<i>`` based on leading dimension order.

- **shape**: A tuple of dimensions enclosed in parentheses. Each dimension
  must be a non-negative integer. Examples: ``(4,3)`` for a 2D array,
  ``(4,)`` for a 1D array, ``()`` for a scalar.

- **dtype**: A dtype specifier from the supported vocabulary. Supported dtypes:

  - ``f32`` → ``numpy.float32``
  - ``f64`` → ``numpy.float64``
  - ``i32`` → ``numpy.int32``
  - ``bool`` → ``numpy.bool_``

Examples
--------

Parse a single float32 array of shape (4, 3)::

    >>> parse_shapes('x=(4,3)f32')
    {'x': ShapeDtypeStruct(shape=(4, 3), dtype=float32)}

Parse multiple inputs with different dtypes::

    >>> parse_shapes('x=(4,3)f32 mask=(4,)bool')
    {'x': ShapeDtypeStruct(shape=(4, 3), dtype=float32),
     'mask': ShapeDtypeStruct(shape=(4,), dtype=bool_)}

Parse a scalar input::

    >>> parse_shapes('x=()f32')
    {'x': ShapeDtypeStruct(shape=(), dtype=float32)}

Errors
------

Malformed entries raise ``ShapeParseError``, which quotes the expected
grammar to guide correction. Common errors:

- Missing ``=`` separator
- Missing or mismatched parentheses
- Non-integer dimensions
- Unknown dtype specifiers
"""

from __future__ import annotations

import re
from typing import Any

import jax
import numpy as np

from xtrax.cli.errors import ShapeParseError

# Map dtype strings to numpy/jax dtype objects
_DTYPE_MAP: dict[str, Any] = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "bool": np.bool_,
}


def parse_shapes(s: str) -> dict[str, jax.ShapeDtypeStruct]:
    """Parse a shape specification string into a dictionary of ShapeDtypeStruct.

    Parameters
    ----------
    s : str
        A space-separated list of shape entries, each in the format
        ``name=(d0,d1,...)<dtype>``.

    Returns
    -------
    dict[str, jax.ShapeDtypeStruct]
        A dictionary mapping names to ShapeDtypeStruct objects, where each
        ShapeDtypeStruct contains the shape tuple and dtype for the named input.

    Raises
    ------
    ShapeParseError
        If any entry is malformed (missing ``=``, invalid parentheses,
        non-integer dimensions, or unknown dtype).
    """
    s = s.strip()

    if not s:
        raise ShapeParseError("Empty shape specification. Expected: name=(d0,d1,...)<dtype>")

    entries = s.split()
    result: dict[str, jax.ShapeDtypeStruct] = {}

    for entry in entries:
        if not entry:
            continue

        _parse_single_entry(entry, result)

    if not result:
        raise ShapeParseError(
            "No valid entries in shape specification. Expected: name=(d0,d1,...)<dtype>"
        )

    return result


def _parse_single_entry(entry: str, result: dict[str, jax.ShapeDtypeStruct]) -> None:
    """Parse a single shape entry and add it to the result dictionary.

    Parameters
    ----------
    entry : str
        A single shape entry in the format ``name=(d0,d1,...)<dtype>``.
    result : dict
        The result dictionary to which the parsed entry is added.

    Raises
    ------
    ShapeParseError
        If the entry is malformed.
    """
    # Check for '=' separator
    if "=" not in entry:
        raise ShapeParseError(
            f"Malformed entry '{entry}': missing '=' separator. Expected: name=(d0,d1,...)<dtype>"
        )

    name, rest = entry.split("=", 1)
    name = name.strip()

    if not name:
        raise ShapeParseError(
            f"Malformed entry '{entry}': name is empty. Expected: name=(d0,d1,...)<dtype>"
        )

    # Validate name is a valid identifier
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
        raise ShapeParseError(
            f"Malformed entry '{entry}': name '{name}' is not a valid identifier. "
            f"Expected: name=(d0,d1,...)<dtype>"
        )

    # Parse shape and dtype from rest
    # Pattern: (d0,d1,...)dtype
    match = re.match(r"^\(([^)]*)\)([a-zA-Z0-9]+)$", rest)

    if not match:
        raise ShapeParseError(
            f"Malformed entry '{entry}': expected shape in parentheses followed by dtype. "
            f"Expected: name=(d0,d1,...)<dtype>"
        )

    shape_str, dtype_str = match.groups()

    # Parse shape
    shape = _parse_shape(shape_str, entry)

    # Parse dtype
    if dtype_str not in _DTYPE_MAP:
        raise ShapeParseError(
            f"Malformed entry '{entry}': unknown dtype '{dtype_str}'. "
            f"Supported dtypes: {', '.join(sorted(_DTYPE_MAP.keys()))}"
        )

    dtype = _DTYPE_MAP[dtype_str]

    # Create ShapeDtypeStruct and add to result
    result[name] = jax.ShapeDtypeStruct(shape=shape, dtype=dtype)


def _parse_shape(shape_str: str, full_entry: str) -> tuple[int, ...]:
    """Parse a shape string into a tuple of integers.

    Parameters
    ----------
    shape_str : str
        The contents of the shape parentheses, e.g., "4,3" or "4," or "".
    full_entry : str
        The full entry string (for error messages).

    Returns
    -------
    tuple[int, ...]
        A tuple of dimension integers.

    Raises
    ------
    ShapeParseError
        If shape_str contains non-integer values or is malformed.
    """
    shape_str = shape_str.strip()

    if not shape_str:
        # Empty shape: scalar
        return ()

    # Split by comma and filter empty strings (handles trailing commas like "4,")
    dim_strs = [s.strip() for s in shape_str.split(",") if s.strip()]
    shape = []

    for dim_str in dim_strs:
        try:
            dim = int(dim_str)
            if dim < 0:
                raise ValueError("negative dimension")
            shape.append(dim)
        except ValueError:
            raise ShapeParseError(
                f"Malformed entry '{full_entry}': '{dim_str}' is not a valid non-negative integer. "
                f"Expected: name=(d0,d1,...)<dtype>"
            )

    return tuple(shape)
