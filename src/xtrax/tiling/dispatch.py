"""make_axis_dispatch: unified dispatch over AxisStrategy variants."""

from collections.abc import Callable
from typing import Any

import jax

from xtrax.tiling.strategy import (
    AxisStrategy,
    Bucket,
    DedupGather,
    SafeMap,
    Scan,
    Vmap,
)
from xtrax.transforms.map import safe_map
from xtrax.transforms.scan import safe_scan


def make_axis_dispatch(
    strategy: AxisStrategy,
    fn: Callable,
    xs: Any,
    init: Any = None,
) -> Any:
    """Dispatch execution to the appropriate JAX transform based on strategy.

    Handles the four device-tier AxisStrategy variants: Vmap, SafeMap, Scan, and
    DedupGather. Each applies a different on-device execution pattern for axis
    iteration.

    ``Bucket`` is intentionally NOT executed here: bucketing is a host-side,
    pre-JIT concern (JAX has no dynamic shapes inside ``jit``). Pad to a bucket
    shape with ``select_bucket``/``bucketize`` on the host, then dispatch the
    per-bucket compute with a device-tier strategy. Passing a ``Bucket`` raises
    ``TypeError`` with that guidance.

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
        TypeError: For Bucket (host-side strategy); or if strategy is not a
                   recognized AxisStrategy variant.
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
        # Use strategy.init if available, else fall back to init parameter
        carry_init = strategy.init if strategy.init is not None else init
        if carry_init is None:
            raise ValueError(
                "make_axis_dispatch: Scan strategy requires a non-None init carry. "
                "Provide via Scan.init field or init parameter."
            )
        return safe_scan(strategy.transition, carry_init, xs)

    elif isinstance(strategy, DedupGather):
        # DedupGather: three-phase: dedup -> map -> gather
        deduped_xs, gather_indices = strategy.dedup_fn(xs)  # Explicit unpacking
        deduped_ys = safe_map(fn, deduped_xs, batch_size=None)  # vmap over deduped
        return strategy.gather_fn(deduped_ys, gather_indices)

    elif isinstance(strategy, Bucket):
        # Bucket is host-side: bucketing pads variable-length inputs *before* the
        # JIT boundary, so it cannot be executed inside this device-tier dispatch.
        raise TypeError(
            "Bucket is a host-side strategy and is not executed by "
            "make_axis_dispatch. Pad to a bucket shape on the host with "
            "select_bucket()/bucketize() before your jitted step, then dispatch "
            "the per-bucket compute with a device-tier strategy (e.g. Vmap/SafeMap)."
        )

    else:
        raise TypeError(f"Unknown strategy type: {type(strategy)}")
