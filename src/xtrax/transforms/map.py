from collections.abc import Callable

import jax

T: type


def safe_map[T](
    fn: Callable[[T], T], xs: T, batch_size: int | None = None
) -> T:
    """Apply a function to a pytree using vmap or lax.map depending on size.

    Uses jax.vmap when batch_size is None or n <= batch_size. Otherwise uses
    jax.lax.map with the specified batch_size for memory efficiency.

    Args:
        fn: Function to apply to each element.
        xs: Input pytree where the first dimension is the batch dimension.
        batch_size: Batch size for lax.map. If None, uses vmap.

    Returns:
        Result of applying fn to xs with same structure as xs.

    Raises:
        ValueError: If n > batch_size and n is not divisible by batch_size.
    """
    # Get the batch size from the first leaf
    n = jax.tree.leaves(xs)[0].shape[0]

    # Use vmap if batch_size is None or n <= batch_size
    if batch_size is None or n <= batch_size:
        return jax.vmap(fn)(xs)

    # Validate divisibility for lax.map path
    if n % batch_size != 0:
        raise ValueError(
            f"safe_map: n={n} is not divisible by batch_size={batch_size}. "
            "Ensure cardinality is a multiple of batch_size."
        )

    # Use lax.map for batched processing
    return jax.lax.map(fn, xs, batch_size=batch_size)
