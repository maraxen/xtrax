"""Gradient accumulation using jax.lax.scan."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from xtrax._tree import is_array, is_inexact_array, value_and_grad


def accumulate_grads(
    loss_fn: Callable[..., Any],
    params: Any,
    microbatches: Any,  # pre-stacked PyTree; leading axis = num_microbatches
    filter_spec: Callable | None = None,
    *,
    has_aux: bool = False,
) -> tuple[Any, ...]:
    """Accumulate gradients over microbatches using jax.lax.scan.

    Args:
        loss_fn: Callable that takes ``(params, microbatch)``. Returns a scalar
            loss, or ``(loss, aux)`` when ``has_aux`` is True. ``aux`` is a
            PyTree of arrays with the same structure on every microbatch.
        params: Model parameters (PyTree).
        microbatches: Pre-stacked PyTree with leading axis num_microbatches.
                     All leaves must have same leading axis size.
        filter_spec: Predicate ``leaf -> bool`` (default ``xtrax.is_array``) choosing
            which leaves of ``params`` are differentiated. Leaves it rejects, and
            any leaf that is not a floating or complex array, are held constant and
            come back as ``None`` in ``mean_grads``.
        has_aux: When True, ``loss_fn`` returns ``(loss, aux)`` and the result
            is ``(mean_grads, mean_loss, mean_aux)``. ``has_aux`` is a static
            mode flag (a Python bool), not a traced value. Default False
            returns ``(mean_grads, mean_loss)``.

    Returns:
        ``has_aux=False``: ``(mean_grads, mean_loss)``.
        ``has_aux=True``: ``(mean_grads, mean_loss, mean_aux)``.

        ``mean_grads`` matches ``params``. ``mean_loss`` is a scalar.
        ``mean_aux`` has the same PyTree structure as one microbatch's aux.
        Every aux leaf is reduced with ``jnp.mean(..., axis=0)``, the same
        reduction as ``mean_loss``. Microbatches are already required to share
        a leading size and a second-axis size, so this is an unweighted mean
        of the per-microbatch aux values (there is no size reweighting).

    Raises:
        ValueError: If microbatch shapes are inconsistent.

    When to use ``optax.MultiSteps`` instead: ``MultiSteps`` accumulates across
    optimizer calls, so each microbatch is its own jitted step and the inner
    optimizer runs once per ``k`` calls. Prefer it when the microbatches arrive
    one at a time (a streaming loader, a step you already jit) or when you want
    the optimizer state to see the accumulation. Prefer ``accumulate_grads`` when
    all microbatches are stacked in memory and you want one fused ``lax.scan``.
    Both average the per-microbatch gradients and agree with each other to float
    tolerance (parity probe, bathos run 72c46a5b). Neither reproduces the
    full-batch gradient of a batch-coupled loss (batch norm, a contrastive or
    in-batch-negative loss, anything that is not a mean of per-example terms):
    each microbatch's loss only sees its own examples. For a per-example mean
    over equal-sized microbatches the accumulated gradient equals the full-batch
    one; for a batch-centered loss on a spike (4 microbatches of 5), the maximum
    absolute difference from the full-batch gradient was 0.12 against 1e-7. For
    the full-batch gradient of such a loss, do not accumulate: use a larger
    device batch or shard the batch.
    """
    if filter_spec is None:
        filter_spec = is_array

    def diff_pred(x):
        return bool(filter_spec(x)) and is_inexact_array(x)

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

    # has_aux is a Python bool, so exactly one of these scan bodies is traced.
    def scan_fn(carry, microbatch):
        """Compute loss and grad for one microbatch.

        Args:
            carry: None (or any carry state; we don't use it)
            microbatch: One slice along the leading axis of the input PyTree

        Returns:
            (carry, outputs): carry unchanged. outputs is (grads, loss), or
            (grads, loss, aux) when has_aux is True.
        """
        if has_aux:
            (loss, aux), grads = value_and_grad(loss_fn, has_aux=True, pred=diff_pred)(
                params, microbatch
            )
            return carry, (grads, loss, aux)
        loss, grads = value_and_grad(loss_fn, pred=diff_pred)(params, microbatch)
        return carry, (grads, loss)

    # Run scan over leading axis. Mean over axis 0 is the microbatch reduction
    # for grads, the scalar loss, and (when requested) every aux leaf.
    if has_aux:
        _, (all_grads, all_losses, all_aux) = jax.lax.scan(scan_fn, None, microbatches)
        mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), all_grads)
        mean_loss = jnp.mean(all_losses)
        mean_aux = jax.tree.map(lambda a: jnp.mean(a, axis=0), all_aux)
        return mean_grads, mean_loss, mean_aux

    _, (all_grads, all_losses) = jax.lax.scan(scan_fn, None, microbatches)
    mean_grads = jax.tree.map(lambda g: jnp.mean(g, axis=0), all_grads)
    mean_loss = jnp.mean(all_losses)
    return mean_grads, mean_loss
