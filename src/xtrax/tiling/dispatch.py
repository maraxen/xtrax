"""make_axis_dispatch: unified dispatch over AxisStrategy variants."""

from collections.abc import Callable
from typing import Any

import jax

from xtrax.tiling.strategy import AxisStrategy, DedupGather, SafeMap, Scan, Vmap
from xtrax.transforms.map import safe_map
from xtrax.transforms.scan import safe_scan


def make_axis_dispatch(
    strategy: AxisStrategy,
    fn: Callable,
    xs: Any,
    init: Any = None,
) -> Any:
    """Dispatch execution to the appropriate JAX transform based on strategy.

    Exhaustively handles all four AxisStrategy variants: Vmap, SafeMap, Scan, and
    DedupGather. Each variant applies a different execution pattern for axis iteration.

    Args:
        strategy: The AxisStrategy to dispatch with.
            One of: Vmap | SafeMap | Scan | DedupGather.
        fn: Callable to apply. Unused for Scan (uses strategy.transition instead).
        xs: Input pytree to process.
        init: Initial carry state. Required for Scan, ignored for others.

    Returns:
        Result of applying the strategy, with shape and structure matching the
        desired execution pattern.

    Raises:
        ValueError: For Scan if init is None; for SafeMap if xs leading axis
                    is not divisible by batch_size.
        TypeError: If strategy is not a recognized AxisStrategy variant.
    """
    if isinstance(strategy, Vmap):
        # Vmap: vectorize over the leading axis
        return jax.vmap(fn)(xs)

    elif isinstance(strategy, SafeMap):
        # SafeMap: chunked vmap for memory efficiency
        return safe_map(fn, xs, batch_size=strategy.batch_size)

    elif isinstance(strategy, Scan):
        # Scan: carry-bearing sequential iteration
        # strategy.transition is the actual computation; fn is ignored
        if init is None:
            raise ValueError(
                "make_axis_dispatch: Scan strategy requires a non-None init carry."
            )
        return safe_scan(strategy.transition, init, xs)

    elif isinstance(strategy, DedupGather):
        # DedupGather: three-phase: dedup -> map -> gather
        deduped_xs, gather_indices = strategy.dedup_fn(xs)  # Explicit unpacking
        deduped_ys = safe_map(fn, deduped_xs, batch_size=None)  # vmap over deduped
        return strategy.gather_fn(deduped_ys, gather_indices)

    else:
        raise TypeError(f"Unknown strategy type: {type(strategy)}")
