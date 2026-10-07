from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

T: type


def _none_leaf(node: Any) -> bool:
    """None is an unmapped prefix, not an empty pytree node."""
    return node is None


def _broadcast_axes(xs: Any, in_axes: Any) -> list[Any]:
    """One axis per leaf of ``xs``. None means that leaf is not mapped."""
    broadcasted = jax.tree.broadcast(in_axes, xs, is_leaf=_none_leaf)
    return jax.tree.flatten(broadcasted, is_leaf=_none_leaf)[0]


def _is_size1_axis(xs: Any, in_axes: Any) -> bool:
    """True when every mapped leaf has length 1 on its mapped axis.

    ``in_axes`` is an int, None, or a tree prefix of ``xs``. A None node leaves
    that subtree unmapped, the same rule ``jax.vmap`` uses for in-axes.
    """
    leaves = jax.tree.leaves(xs)
    if not leaves:
        return False
    try:
        axes = _broadcast_axes(xs, in_axes)
        if len(axes) != len(leaves):
            return False
        mapped = [(leaf, axis) for leaf, axis in zip(leaves, axes, strict=True) if axis is not None]
        return bool(mapped) and all(leaf.shape[axis] == 1 for leaf, axis in mapped)
    except (IndexError, TypeError, ValueError):
        return False


def _apply_size1(fn: Callable[..., Any], xs: Any, in_axes: Any = 0) -> Any:
    """Apply fn along a length-1 axis without vmap.

    Output axis 0 matches ``jax.vmap``'s default ``out_axes``. A length-1 vmap
    miscompiles on some GPUs (#2520, aminx #2391). ``in_axes`` follows the same
    tree-prefix rule as ``_is_size1_axis``.
    """
    broadcasted = jax.tree.broadcast(in_axes, xs, is_leaf=_none_leaf)

    def squeeze_leaf(leaf: Any, axis: Any) -> Any:
        if axis is None:
            return leaf
        return jnp.squeeze(leaf, axis=axis)

    squeezed = jax.tree.map(squeeze_leaf, xs, broadcasted, is_leaf=_none_leaf)
    return jax.tree.map(lambda leaf: jnp.expand_dims(leaf, axis=0), fn(squeezed))


def chunked_map[T](fn: Callable[[T], T], xs: T, batch_size: int | None = None) -> T:
    """Apply a function to a pytree using vmap or lax.map depending on size.

    Uses jax.vmap when batch_size is None or n <= batch_size, except when the
    leading axis has length 1. Otherwise uses jax.lax.map with the specified
    batch_size for memory efficiency. A cardinality that is not a multiple of
    batch_size is fine: full chunks are scanned and the remainder is one smaller
    chunk, so peak memory stays bounded by batch_size and nothing is padded (#5565).

    A mapped axis of length 1 is never a vmap (#2520). ``batch_size=1`` is a
    sequential ``lax.map``, a remainder of 1 is a direct call on the last element,
    and a leading axis of length 1 is a direct call. vmap over a length-1 axis
    miscompiles on some GPUs.

    Args:
        fn: Function to apply to each element.
        xs: Input pytree where the first dimension is the batch dimension.
        batch_size: Batch size for lax.map. If None, uses vmap (unless n == 1).

    Returns:
        Result of applying fn to xs with same structure as xs.
    """
    # Get the batch size from the first leaf. Shapes are static under jit, so
    # these branches are Python, not traced, control flow.
    n = jax.tree.leaves(xs)[0].shape[0]

    # vmap (and lax.map's per-chunk vmap) over a length-1 axis miscompiles (#2520).
    if n == 1:
        return _apply_size1(fn, xs)

    # Use vmap if batch_size is None or n <= batch_size
    if batch_size is None or n <= batch_size:
        return jax.vmap(fn)(xs)

    # lax.map(batch_size=1) vmaps every element. Sequential map does not.
    if batch_size == 1:
        return jax.lax.map(fn, xs)

    # Peel a remainder of 1 so the final chunk is not a vmap-of-1. The prefix
    # length is a multiple of batch_size, and batch_size > 1 here.
    if batch_size > 1 and n % batch_size == 1:
        prefix = jax.tree.map(lambda leaf: leaf[:-1], xs)
        last = jax.tree.map(lambda leaf: leaf[-1:], xs)
        prefix_ys = jax.lax.map(fn, prefix, batch_size=batch_size)
        last_y = _apply_size1(fn, last)
        return jax.tree.map(
            lambda bulk, tail: jnp.concatenate([bulk, tail], axis=0),
            prefix_ys,
            last_y,
        )

    # lax.map handles any other ragged final chunk itself (no padding; see docstring).
    return jax.lax.map(fn, xs, batch_size=batch_size)
