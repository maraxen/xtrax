from collections.abc import Callable

import jax

from xtrax._renamed import Deprecated, deprecated_getattr


def safe_scan[Carry, X, Y](
    fn: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
) -> tuple[Carry, Y]:
    """Safe JAX scan with validation against zero-length inputs.

    .. deprecated:: 0.4.1b9
        Deprecated in the #6107 probe release (``"safe"`` was a misnomer: this
        wrapper neither chunks nor dispatches). Call :func:`jax.lax.scan`
        directly, or use :func:`xtrax.tiling.axis_dispatch` /
        :func:`xtrax.tiling.make_axis_dispatch` with :class:`~xtrax.tiling.Scan`
        when strategy-dispatched. Note that :func:`jax.lax.scan` treats a
        length-0 input as a no-op returning ``(init, empty_ys)`` rather than
        raising :class:`ValueError`.

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


# Deprecated in the #6107 probe release: warns on access and is otherwise unchanged.
_DEPRECATED = {
    "safe_scan": Deprecated(
        safe_scan,
        "jax.lax.scan(fn, init, xs)",
        "For strategy-dispatched scan iteration, use xtrax.tiling.axis_dispatch(Scan(...), ...) "
        "or make_axis_dispatch(Scan(...)). Note: jax.lax.scan treats length-0 xs as a no-op "
        "rather than raising ValueError.",
    ),
}
del safe_scan
__getattr__ = deprecated_getattr(__name__, _DEPRECATED)
