from collections.abc import Callable

import jax

T: type


def safe_map[T](fn: Callable[[T], T], xs: T, batch_size: int | None = None) -> T:
    """Apply a function to a pytree using vmap or lax.map depending on size.

    Uses jax.vmap when batch_size is None or n <= batch_size. Otherwise uses
    jax.lax.map with the specified batch_size for memory efficiency. A cardinality
    that is not a multiple of batch_size is fine: jax.lax.map runs the n // batch_size
    full chunks as a scan and the n % batch_size remainder as one smaller vmapped
    chunk, so peak memory stays bounded by batch_size and nothing is padded (#5565).

    Args:
        fn: Function to apply to each element.
        xs: Input pytree where the first dimension is the batch dimension.
        batch_size: Batch size for lax.map. If None, uses vmap.

    Returns:
        Result of applying fn to xs with same structure as xs.
    """
    # Get the batch size from the first leaf
    n = jax.tree.leaves(xs)[0].shape[0]

    # Use vmap if batch_size is None or n <= batch_size
    if batch_size is None or n <= batch_size:
        return jax.vmap(fn)(xs)

    # lax.map handles a ragged final chunk itself (no padding; see docstring).
    return jax.lax.map(fn, xs, batch_size=batch_size)
