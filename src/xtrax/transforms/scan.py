from collections.abc import Callable

import jax


def safe_scan[Carry, X, Y](
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
) -> tuple[Carry, Y]:
    """Safe JAX scan with validation against zero-length inputs.

    Wraps jax.lax.scan with a pre-trace validation that raises ValueError if
    the input sequence has length 0 (either inferred from xs or explicit via
    the length parameter).

    Args:
        fn: Transition function (carry, x) -> (new_carry, y).
        init: Initial carry state. Can be None (which is a legal lax.scan carry).
        xs: Input sequence (pytree with a leading batch axis).
        length: Optional explicit length. If None, inferred from the first leaf
                of xs. If 0 or inferred length is 0, raises ValueError.
        reverse: If True, iterate in reverse.
        unroll: Loop unrolling factor for JAX compiler.

    Returns:
        Tuple of (final_carry, all_ys) where all_ys has the same structure
        as the outputs of fn applied to each input.

    Raises:
        ValueError: If length is 0 or if the inferred leading axis of xs is 0.
    """
    # Determine effective length
    effective_length = length
    if effective_length is None:
        leaves = jax.tree.leaves(xs)
        effective_length = leaves[0].shape[0] if leaves else 0

    # Validate: length must not be 0
    if effective_length == 0:
        raise ValueError("safe_scan: xs has leading axis of length 0.")

    # Delegate to jax.lax.scan with all parameters
    return jax.lax.scan(fn, init, xs, length=length, reverse=reverse, unroll=unroll)
