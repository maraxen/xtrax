"""Axis synthesis and role resolution for signature inference (E1.3a MVP, E1.4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
from jax import ShapeDtypeStruct

from xtrax.inference.config import AxisOverride
from xtrax.inference.errors import AxisRole

if TYPE_CHECKING:
    from xtrax.tiling.plan import AxisSpec


def resolve_axis_role(override: AxisOverride | None) -> AxisRole:
    """Resolve an axis role from an optional AxisOverride.

    Args:
        override: The :class:`AxisOverride` for this axis (from the
            @axis_config hook, E1.4). None means no override was provided.

    Returns:
        AxisRole.KNOWN if override is not None, else AxisRole.UNKNOWN.
    """
    return AxisRole.KNOWN if override is not None else AxisRole.UNKNOWN


def synthesize_axes(
    abstract_inputs: Any,
    overrides: Sequence[AxisOverride] | None = None,
) -> list[AxisSpec]:
    """Synthesize AxisSpec objects from abstract (ShapeDtypeStruct) inputs.

    For each ShapeDtypeStruct leaf with ndim >= 1, one AxisSpec is synthesized
    for its leading axis.

    INVARIANT (D1): this function ALWAYS sets role EXPLICITLY on every AxisSpec
    it creates. With overrides=None, EVERY returned AxisSpec has
    role == AxisRole.UNKNOWN.

    Overrides are applied POSITIONALLY: synthesized axis i receives
    ``overrides[i]`` when that index exists.  Axes with no positional override
    keep a positional name (``axis_<i>``) and role=UNKNOWN.

    Args:
        abstract_inputs: A pytree whose leaves are ShapeDtypeStruct instances.
            Typically produced by build_abstract_inputs().
        overrides: Optional sequence of :class:`AxisOverride` instances,
            applied positionally.  ``overrides[i]`` configures the i-th axis:
            name, default_batch_size, cardinality (if not None), and other
            tiling fields are taken from the override, and the axis role is
            set to KNOWN.  Axes beyond the length of *overrides* (or when
            *overrides* is None) receive positional names and role=UNKNOWN.

    Returns:
        List of AxisSpec objects, one per qualifying leaf (ndim >= 1), in
        tree-leaf order.
    """

    from xtrax.tiling.plan import AxisSpec  # deferred to break inference<->tiling cycle

    def _is_shape_dtype_struct(x: Any) -> bool:
        return isinstance(x, ShapeDtypeStruct)

    leaves: list[ShapeDtypeStruct] = jax.tree_util.tree_leaves(
        abstract_inputs,
        is_leaf=_is_shape_dtype_struct,
    )

    result: list[AxisSpec] = []
    axis_idx = 0  # index into the synthesized-axis list (skips ndim=0 leaves)
    for i, leaf in enumerate(leaves):
        if not isinstance(leaf, ShapeDtypeStruct):
            continue
        if leaf.ndim < 1:
            continue

        leading_dim = int(leaf.shape[0])

        # Positional override lookup
        ov: AxisOverride | None = (
            overrides[axis_idx]
            if overrides is not None and axis_idx < len(overrides)
            else None
        )

        if ov is not None:
            # Override present: use override fields
            name = ov.name
            cardinality = ov.cardinality if ov.cardinality is not None else leading_dim
            default_batch_size = ov.default_batch_size
            role = resolve_axis_role(ov)
            spec = AxisSpec(
                name=name,
                cardinality=cardinality,
                default_batch_size=default_batch_size,
                tile_granularity=ov.tile_granularity,
                heterogeneous=ov.heterogeneous,
                dedup_eligible=ov.dedup_eligible,
                bucket_boundaries=ov.bucket_boundaries,
                role=role,
            )
        else:
            # No override: positional name, UNKNOWN role
            name = f"axis_{i}"
            cardinality = leading_dim
            # Placeholder: UNKNOWN axes fail loud before tiling (E1.3b adds the guard).
            # The placeholder is never used by the planner in the UNKNOWN path.
            default_batch_size = cardinality
            role = resolve_axis_role(None)
            spec = AxisSpec(
                name=name,
                cardinality=cardinality,
                default_batch_size=default_batch_size,
                role=role,
            )

        result.append(spec)
        axis_idx += 1

    return result
