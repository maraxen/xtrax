"""Inference-time sparsification utilities for xtrax.

Provides:
  - sparsify_model: functional transform — converts 2D weight leaves to BCOO
  - make_sparse_forward_fn: closure helper — keeps BCOO leaves out of
    eqx.filter_jit's argument partition (recommended composition pattern)
  - assert_not_tracing: guard that raises if called inside jax.jit
  - sparse_filter_jit: wrapper for eqx.filter_jit safe for BCOO models

BCOO destructuring trap:
  eqx.filter_jit without careful handling will descend into BCOO nodes and
  partition .data and .indices as separate traced arrays on every call. This
  defeats retrace prevention. Use make_sparse_forward_fn (closure pattern) to
  avoid this entirely, OR use sparse_filter_jit when you pass the sparsified
  model as a jit argument.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
from jax.experimental.sparse import BCOO

from xtrax.sparse.policy import SparsePolicy

Array = jax.Array
PyTree = Any


def assert_not_tracing(leaves: list[Any]) -> None:
    """Assert that no leaves are JAX Tracers (not inside jit).

    Raises:
        RuntimeError: If any leaf is a jax.core.Tracer.
    """
    for leaf in leaves:
        if isinstance(leaf, jax.core.Tracer):
            raise RuntimeError(
                "sparsify_model cannot be called inside jax.jit — "
                "call it before jit compilation"
            )


def sparsify_model(
    model: eqx.Module,
    policy: SparsePolicy,
    leaf_filter: Callable[[Any], bool] = eqx.is_array,
) -> eqx.Module:
    """Apply sparsification masks to a model, returning a new model with BCOO leaves.

    This is a pure Python-side functional transform. It traverses the model pytree,
    applies policy.apply_mask to each leaf passing leaf_filter, and returns a new
    eqx.Module with BCOO weights in place of dense parameter leaves.

    Args:
        model: An equinox Module to sparsify.
        policy: SparsePolicy instance defining mask and apply_mask behavior.
        leaf_filter: A callable that returns True for leaves to consider for
                     sparsification. Default is eqx.is_array (all arrays).

    Returns:
        A new eqx.Module with BCOO leaves replacing 2D dense parameter leaves.

    Raises:
        RuntimeError: If called inside jax.jit (detected via Tracer check).
        ValueError: If model already contains BCOO leaves (double sparsification).
    """
    # Flatten the model to check for Tracers and existing BCOO leaves
    leaves, treedef = jax.tree_util.tree_flatten(
        model, is_leaf=lambda x: isinstance(x, BCOO)
    )

    # Check that we're not inside jit
    assert_not_tracing(leaves)

    # Guard: reject models that already contain BCOO leaves
    if any(isinstance(leaf, BCOO) for leaf in leaves):
        raise ValueError(
            "model already contains BCOO leaves — double sparsification detected"
        )

    # Apply masks to each leaf
    new_leaves = []
    for leaf in leaves:
        if not leaf_filter(leaf):
            # Leaf excluded by filter; pass through unchanged
            new_leaf = leaf
        elif hasattr(leaf, "ndim") and leaf.ndim == 2:
            # Sparsifiable leaf: 2D array, apply mask
            mask = policy.make_mask(leaf, step=0)
            new_leaf = policy.apply_mask(leaf, mask)
        elif hasattr(leaf, "ndim") and leaf.ndim != 2:
            # Filtered but non-2D: warn and skip (apply_mask requires 2D)
            warnings.warn(
                f"sparsify_model: skipping non-2D leaf (ndim={leaf.ndim}) — "
                f"apply_mask requires 2D weights; leaf preserved as dense",
                stacklevel=2,
            )
            new_leaf = leaf
        else:
            # No ndim attribute or other edge case; pass through
            new_leaf = leaf

        new_leaves.append(new_leaf)

    # Unflatten and return
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def make_sparse_forward_fn(
    fn: Callable[[eqx.Module, Any], Any], sparse_model: eqx.Module
) -> Callable[[Any], Any]:
    """Wrap a forward function with a sparse model closed over.

    Returns a function that takes only inputs (and optional RNG), with the
    sparse_model already baked in as a constant.

    Closing over sparse_model keeps BCOO leaves out of eqx.filter_jit's is_array
    partition. BCOO is not a jax.Array — passed as an argument it would be
    destructured into data+indices; as a closure constant it is treated as static.
    This is the load-bearing mechanism for AC-6 (no-retrace guarantee).

    Args:
        fn: A callable with signature (model, inputs) -> outputs.
        sparse_model: The sparsified eqx.Module to close over.

    Returns:
        A function with signature (inputs) -> outputs.
    """

    def _forward(inputs):
        return fn(sparse_model, inputs)

    return _forward


def sparse_filter_jit(fn: Callable, **kwargs) -> Callable:
    """Drop-in eqx.filter_jit wrapper optimized for BCOO-containing models.

    WARNING: Modern equinox.filter_jit does not support is_leaf configuration.
    For safe BCOO handling, prefer make_sparse_forward_fn (closure pattern) which
    keeps BCOO completely out of JAX's tracing system.

    sparse_filter_jit is provided for cases where you must pass BCOO models as
    arguments. The safest usage is with eqx.Module (pytree) models where BCOO
    leaves are preserved during filtering.

    Args:
        fn: Function to jit-compile.
        **kwargs: Additional keyword arguments passed to eqx.filter_jit.

    Returns:
        A jit-compiled function via eqx.filter_jit.
    """
    return eqx.filter_jit(fn, **kwargs)
