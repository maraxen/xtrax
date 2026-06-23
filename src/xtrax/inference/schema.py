"""BundleSchema: runtime schema extraction from jax.eval_shape output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
from jax import ShapeDtypeStruct


@dataclass
class BundleSchema:
    """Schema extracted from a computation's output pytree via jax.eval_shape.

    Attributes:
        fields: Ordered mapping of leaf field-name (str) to ShapeDtypeStruct.
            Field names are recovered from the eval_shape output pytree's key_path:
            - GetAttrKey.attribute: dataclass/eqx.Module field names
            - DictKey.key: dict keys
            - Positional fallback: `out_{i}` for SequenceKey or bare leaves
        carry_specs: Optional list of CarrySpec objects, deferred for future use.
            Stored as-is, no processing in MVP. Defaults to None.
    """

    fields: dict[str, ShapeDtypeStruct] = field(default_factory=dict)
    carry_specs: list[Any] | None = None


def extract_schema(
    fn: Any,
    abstract_inputs: list[ShapeDtypeStruct] | tuple[ShapeDtypeStruct, ...],
) -> BundleSchema:
    """Extract BundleSchema from a fn and abstract inputs via jax.eval_shape.

    This function uses jax.eval_shape to trace fn with abstract (shape/dtype only)
    inputs, producing a pytree of ShapeDtypeStruct objects mirroring the output
    structure. It then walks the output pytree to recover field names from key_path
    and returns a BundleSchema.

    Field-name recovery:
    - jax.tree_util.GetAttrKey: use `.name` attribute (dataclass/eqx.Module fields)
    - jax.tree_util.DictKey: use `str(.key)` (dict keys)
    - jax.tree_util.SequenceKey or no key_path: positional fallback `out_{i}`

    Args:
        fn: A pure, traceable JAX function. Should accept positional arguments
            matching abstract_inputs and return a pytree of arrays.
        abstract_inputs: Sequence (list or tuple) of jax.ShapeDtypeStruct objects,
            one per positional argument to fn.

    Returns:
        BundleSchema with fields dict (name -> ShapeDtypeStruct leaves) and
        carry_specs=None (for future CarryInferencePass hooks).

    Raises:
        Any error raised by jax.eval_shape (e.g., tracing errors).

    Examples:
        >>> from jax import ShapeDtypeStruct
        >>> import numpy as np
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass
        ... class Output:
        ...     logits: jax.Array
        ...     embeddings: jax.Array
        >>>
        >>> def fn(x):
        ...     return Output(logits=x * 2, embeddings=x * 3)
        >>>
        >>> abstract_x = ShapeDtypeStruct((2, 3), np.float32)
        >>> schema = extract_schema(fn, [abstract_x])
        >>> schema.fields["logits"].shape
        (2, 3)
        >>> schema.fields["logits"].dtype
        dtype('float32')
    """
    # Use jax.eval_shape to trace fn without FLOPs.
    # Pass unpacked abstract_inputs as positional args.
    output = jax.eval_shape(fn, *abstract_inputs)

    # Flatten the output pytree, keeping path information.
    # Returns list of (key_path, leaf) tuples where leaf is ShapeDtypeStruct.
    flat_with_paths = jax.tree_util.tree_flatten_with_path(output)[0]

    fields: dict[str, ShapeDtypeStruct] = {}
    for i, (key_path, leaf) in enumerate(flat_with_paths):
        # Recover the field name from the last key entry in the path.
        name = _recover_field_name(key_path, i)
        fields[name] = leaf

    return BundleSchema(fields=fields, carry_specs=None)


def _recover_field_name(key_path: tuple[Any, ...], positional_index: int) -> str:
    """Recover a field name from a key_path and positional fallback.

    Args:
        key_path: Tuple of jax.tree_util.KeyEntry objects representing the path
            to a leaf in the pytree.
        positional_index: Enumeration index for positional fallback.

    Returns:
        Field name (str): either from key_path or positional fallback.
    """
    # Empty path or no meaningful keys -> positional
    if not key_path:
        return f"out_{positional_index}"

    # Get the last key entry (innermost node)
    last_key = key_path[-1]

    # GetAttrKey (dataclass/eqx.Module field)
    if _is_get_attr_key(last_key):
        name = getattr(last_key, "name", None)
        if name is not None:
            return name

    # DictKey (dict key)
    if _is_dict_key(last_key):
        key = getattr(last_key, "key", None)
        if key is not None:
            return str(key)

    # SequenceKey or other: use positional fallback
    return f"out_{positional_index}"


def _is_get_attr_key(key_entry: Any) -> bool:
    """Check if key_entry is a GetAttrKey."""
    return type(key_entry).__name__ == "GetAttrKey"


def _is_dict_key(key_entry: Any) -> bool:
    """Check if key_entry is a DictKey."""
    return type(key_entry).__name__ == "DictKey"
