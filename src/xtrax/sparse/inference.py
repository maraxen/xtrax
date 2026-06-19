"""Inference-time sparsification utilities for xtrax.

Provides:
  - sparsify_model: functional transform — converts 2D weight leaves to BCOO format
  - make_sparse_forward_fn: closure helper — keeps BCOO leaves out of JAX's tracing
    system by closing over them (recommended composition pattern)
  - assert_not_tracing: guard that raises if called inside jax.jit
  - sparse_filter_jit: wrapper for eqx.filter_jit that explicitly documents BCOO safety

BCOO destructuring trap & solutions:
  BCOO is not a jax.Array — it is a pytree node containing .data and .indices arrays.
  Without careful handling, naive jit could treat .data and .indices as separate
  traced arrays, triggering retrace on each call even if model is unchanged.

  eqx.filter_jit naturally avoids this: it traces JAX arrays and holds non-arrays
  (including BCOO) as static. This is sufficient, but two patterns are available:

  1. make_sparse_forward_fn (closure pattern) — RECOMMENDED
     Closes over sparse_model to keep it completely outside JAX tracing.
     Simplest and most explicit.

  2. sparse_filter_jit (argument pattern)
     Pass sparsified model as an argument. eqx.filter_jit's default behavior
     treats BCOO as static, preventing destructuring and retrace.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.core
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
                "sparsify_model cannot be called inside jax.jit — call it before jit compilation"
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
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=lambda x: isinstance(x, BCOO))

    # Check that we're not inside jit
    assert_not_tracing(leaves)

    # Guard: reject models that already contain BCOO leaves
    if any(isinstance(leaf, BCOO) for leaf in leaves):
        raise ValueError("model already contains BCOO leaves — double sparsification detected")

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
    """Drop-in eqx.filter_jit wrapper safe for passing BCOO-containing models.

    By design, eqx.filter_jit treats all JAX/NumPy arrays as traced and all
    non-arrays (including BCOO) as static. Since BCOO is not a jax.Array,
    it is automatically held static and never destructured into .data/.indices
    during jit compilation. This prevents unintended retrace even when the
    sparsified model is passed as an argument.

    This function documents the correct pattern and ensures BCOO safety without
    requiring equinox-specific configuration (modern equinox does not expose
    is_leaf parameter on filter_jit).

    For maximum safety and control, prefer make_sparse_forward_fn (closure pattern)
    which keeps BCOO outside JAX's tracing system entirely.

    Args:
        fn: Function to jit-compile.
        **kwargs: Additional keyword arguments passed to eqx.filter_jit
                  (e.g., donate='all', donate='all-except-first').

    Returns:
        A jit-compiled function via eqx.filter_jit with BCOO leaves held static.

    Example:
        >>> import equinox as eqx
        >>> import jax.numpy as jnp
        >>> @sparse_filter_jit
        ... def forward(model, x):
        ...     return model(x)
        >>> # Call with sparse model: BCOO is static, no retrace on repeated calls
    """
    return eqx.filter_jit(fn, **kwargs)
