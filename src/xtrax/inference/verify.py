"""Structure consistency verification for signature inference.

This module provides guardrails against "bedrock betrayal" — when jax.eval_shape's
abstract output structure silently diverges from what the function actually produces
on a concrete batch (e.g., due to data-dependent control flow or side-effecting
tree operations).
"""

from __future__ import annotations

from typing import Any

import jax
from jax import ShapeDtypeStruct

from xtrax.inference.errors import StructureMismatchError


def verify_structure(fn: Any, abstract_inputs: list[Any], concrete_inputs: list[Any]) -> None:
    """Guard against divergence between abstract (eval_shape) and concrete execution.

    Compares the pytree structure and leaf shapes/dtypes of the abstract output
    (from jax.eval_shape) with the concrete output (from actual execution).

    Args:
        fn: The function to verify.
        abstract_inputs: List of abstract inputs (ShapeDtypeStruct or pytree of them).
        concrete_inputs: List of concrete inputs (actual arrays/data).

    Raises:
        StructureMismatchError: If the abstract and concrete outputs differ in:
            - Pytree structure (treedef)
            - Any leaf's shape
            - Any leaf's dtype

    Returns:
        None if verification passes (structures match).
    """
    # Get abstract output via eval_shape (no actual computation)
    abstract_out = jax.eval_shape(fn, *abstract_inputs)

    # Get concrete output via actual execution
    concrete_out = fn(*concrete_inputs)

    # Compare pytree structures
    abstract_treedef = jax.tree_util.tree_structure(abstract_out)
    concrete_treedef = jax.tree_util.tree_structure(concrete_out)

    if abstract_treedef != concrete_treedef:
        raise StructureMismatchError(
            f"Pytree structure mismatch: abstract has structure {abstract_treedef}, "
            f"but concrete has structure {concrete_treedef}"
        )

    # Compare leaves: shape and dtype
    abstract_leaves = jax.tree_util.tree_leaves(abstract_out)
    concrete_leaves = jax.tree_util.tree_leaves(concrete_out)

    # tree_structure already verified they have same count, but be explicit
    if len(abstract_leaves) != len(concrete_leaves):
        raise StructureMismatchError(
            f"Leaf count mismatch: abstract has {len(abstract_leaves)} leaves, "
            f"concrete has {len(concrete_leaves)} leaves"
        )

    for idx, (abstract_leaf, concrete_leaf) in enumerate(
        zip(abstract_leaves, concrete_leaves, strict=True)
    ):
        # Extract shape and dtype from abstract leaf
        if isinstance(abstract_leaf, ShapeDtypeStruct):
            abstract_shape = abstract_leaf.shape
            abstract_dtype = abstract_leaf.dtype
        else:
            # Fallback: treat as array-like
            abstract_shape = getattr(abstract_leaf, "shape", None)
            abstract_dtype = getattr(abstract_leaf, "dtype", None)

        # Extract shape and dtype from concrete leaf
        concrete_shape = getattr(concrete_leaf, "shape", None)
        concrete_dtype = getattr(concrete_leaf, "dtype", None)

        # Check shape
        if abstract_shape != concrete_shape:
            raise StructureMismatchError(
                f"Shape mismatch at leaf {idx}: abstract shape {abstract_shape}, "
                f"concrete shape {concrete_shape}"
            )

        # Check dtype
        if abstract_dtype != concrete_dtype:
            raise StructureMismatchError(
                f"Dtype mismatch at leaf {idx}: abstract dtype {abstract_dtype}, "
                f"concrete dtype {concrete_dtype}"
            )
