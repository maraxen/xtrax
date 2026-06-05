"""Gradient accumulation using jax.lax.scan."""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


def accumulate_grads(
    loss_fn: Callable[[Any, Any], jax.Array],
    params: Any,
    microbatches: Any,  # pre-stacked PyTree; leading axis = num_microbatches
    filter_spec: Callable | None = None,
) -> tuple[Any, jax.Array]:
    """Accumulate gradients over microbatches using jax.lax.scan.

    Args:
        loss_fn: Callable that takes (params, microbatch) and returns scalar loss.
        params: Model parameters (PyTree).
        microbatches: Pre-stacked PyTree with leading axis num_microbatches.
                     All leaves must have same leading axis size.
        filter_spec: Filter function (defaults to eqx.is_array).

    Returns:
        (mean_grads, mean_loss): Tuple of mean gradients and mean loss.
                                mean_grads is a PyTree matching params structure.
                                mean_loss is a scalar Array.

    Raises:
        ValueError: If microbatch shapes are inconsistent.
    """
    if filter_spec is None:
        filter_spec = eqx.is_array

    # Validate: all leaves must have the same leading axis size
    def get_leading_size(x):
        if isinstance(x, jax.Array):
            return x.shape[0] if x.ndim > 0 else None
        return None

    leading_sizes = jax.tree.map(get_leading_size, microbatches)
    leading_sizes_flat = jax.tree.leaves(
        jax.tree.map(lambda x: x, leading_sizes, is_leaf=lambda x: isinstance(x, int))
    )
    leading_sizes_flat = [s for s in leading_sizes_flat if s is not None]

    if leading_sizes_flat:
        first_size = leading_sizes_flat[0]
        if not all(s == first_size for s in leading_sizes_flat):
            raise ValueError(
                f"All leaves of microbatches must have the same leading axis size, "
                f"but got: {set(leading_sizes_flat)}"
            )
    else:
        raise ValueError("microbatches must contain at least one array leaf")

    # Validate: all leaves must have the same second-axis (microbatch size)
    def get_second_size(x):
        if isinstance(x, jax.Array):
            return x.shape[1] if x.ndim > 1 else None
        return None

    second_sizes = jax.tree.map(get_second_size, microbatches)
    second_sizes_flat = jax.tree.leaves(
        jax.tree.map(lambda x: x, second_sizes, is_leaf=lambda x: isinstance(x, int))
    )
    second_sizes_flat = [s for s in second_sizes_flat if s is not None]

    if second_sizes_flat:
        first_second_size = second_sizes_flat[0]
        if not all(s == first_second_size for s in second_sizes_flat):
            raise ValueError(
                f"All leaves of microbatches must have the same second axis size, "
                f"but got: {set(second_sizes_flat)}"
            )

    # Define scan body: process one microbatch at a time
    def scan_fn(carry, microbatch):
        """Compute loss and grad for one microbatch.

        Args:
            carry: None (or any carry state; we don't use it)
            microbatch: One slice along the leading axis of the input PyTree

        Returns:
            (carry, (grads, loss)): carry unchanged, grads and loss as output
        """
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, microbatch)
        return carry, (grads, loss)

    # Run scan over leading axis
    _, (all_grads, all_losses) = jax.lax.scan(scan_fn, None, microbatches)

    # Compute mean gradients and mean loss
    mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), all_grads)
    mean_loss = jnp.mean(all_losses)

    return mean_grads, mean_loss
