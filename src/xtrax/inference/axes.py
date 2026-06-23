"""Axis synthesis and role resolution for signature inference (E1.3a MVP)."""

from __future__ import annotations

from typing import Any

import jax
from jax import ShapeDtypeStruct

from xtrax.inference.errors import AxisRole
from xtrax.tiling.plan import AxisSpec


def resolve_axis_role(override: Any) -> AxisRole:
    """Resolve an axis role from an optional override value.

    Args:
        override: The override value for this axis (from the @axis_config hook,
            E1.4). None means no override was provided.

    Returns:
        AxisRole.KNOWN if override is not None, else AxisRole.UNKNOWN.
    """
    return AxisRole.KNOWN if override is not None else AxisRole.UNKNOWN


def synthesize_axes(
    abstract_inputs: Any,
    overrides: dict[str, object] | None = None,
) -> list[AxisSpec]:
    """Synthesize AxisSpec objects from abstract (ShapeDtypeStruct) inputs.

    For each ShapeDtypeStruct leaf with ndim >= 1, one AxisSpec is synthesized
    for its leading axis.

    INVARIANT (D1): this function ALWAYS sets role EXPLICITLY on every AxisSpec
    it creates. With overrides=None, EVERY returned AxisSpec has
    role == AxisRole.UNKNOWN.

    Args:
        abstract_inputs: A pytree whose leaves are ShapeDtypeStruct instances.
            Typically produced by build_abstract_inputs().
        overrides: Optional mapping of axis name to override value. An override
            value of anything other than None causes role=KNOWN. None (or absent
            key) causes role=UNKNOWN. E1.4 will wire real override values here;
            for now the presence/absence of a key is the signal.

    Returns:
        List of AxisSpec objects, one per qualifying leaf (ndim >= 1), in
        tree-leaf order.
    """

    def _is_shape_dtype_struct(x: Any) -> bool:
        return isinstance(x, ShapeDtypeStruct)

    leaves: list[ShapeDtypeStruct] = jax.tree_util.tree_leaves(
        abstract_inputs,
        is_leaf=_is_shape_dtype_struct,
    )

    result: list[AxisSpec] = []
    for i, leaf in enumerate(leaves):
        if not isinstance(leaf, ShapeDtypeStruct):
            continue
        if leaf.ndim < 1:
            continue
        name = f"axis_{i}"
        cardinality = int(leaf.shape[0])
        # Placeholder: UNKNOWN axes fail loud before tiling (E1.3b adds the guard).
        # The placeholder is never used by the planner in the UNKNOWN path.
        default_batch_size = cardinality
        override_val = overrides.get(name) if overrides is not None else None
        role = resolve_axis_role(override_val)
        spec = AxisSpec(
            name=name,
            cardinality=cardinality,
            default_batch_size=default_batch_size,
            role=role,
        )
        result.append(spec)

    return result
