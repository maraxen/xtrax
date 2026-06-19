"""Library-side axis dispatch factory contract.

Converts AxisStrategy instances into typed iterators, with rejection logic
for invalid axis/strategy pairs. Part of the composable library surface.

Backward compatibility: axis_dispatch() preserves the old eager 4-arg API,
handling DedupGather and Bucket that make_axis_dispatch() rejects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from xtrax.tiling.strategy import AxisStrategy, SafeMap, Scan, Vmap
from xtrax.transforms.map import safe_map
from xtrax.transforms.scan import safe_scan


class DispatchRejected(Exception):
    """Raised when an AxisStrategy is rejected for a given axis.

    Example: Scan on a heterogeneous axis (state) cannot be used because
    jax.lax.scan requires static carry shape, which incompatible with
    variable-geometry state elements.
    """


def make_axis_dispatch(
    strategy: AxisStrategy,
    *,
    axis: str = "",
    heterogeneous_axes: set[str] | None = None,
) -> object:
    """Dispatch an AxisStrategy to a typed iterator.

    Converts a strategy (Vmap, SafeMap, Scan) into a corresponding iterator
    (VmapIterator, SafeMapIterator, JaxScanIterator). Enforces topological
    constraints: e.g., Scan on a heterogeneous axis is rejected.

    Parameters
    ----------
    strategy : AxisStrategy
        One of Vmap, SafeMap, Scan, or DedupGather.
    axis : str, optional
        Name of the axis being dispatched. Default "". Used to detect
        heterogeneous axes via heterogeneous_axes parameter.
    heterogeneous_axes : set[str] | None, optional
        Set of axis names that are heterogeneous (shapes vary per element).
        If provided, Scan is rejected for any axis in this set.
        Default None (no heterogeneous constraints).

    Returns
    -------
    object
        A MapIterator (VmapIterator or SafeMapIterator) or ScanIterator
        (JaxScanIterator). Concrete types are imported at call time.

    Raises
    ------
    DispatchRejected
        If strategy is Scan and axis is in heterogeneous_axes,
        or if strategy is DedupGather (which is handled elsewhere,
        not by make_axis_dispatch).

    """
    # Lazy import to avoid circular dependency with iterator.py.
    from xtrax.tiling.iterator import (
        JaxScanIterator,
        SafeMapIterator,
        VmapIterator,
    )
    from xtrax.tiling.strategy import DedupGather

    # Reject DedupGather (handled by _dispatch_axis, not here).
    if isinstance(strategy, DedupGather):
        raise DispatchRejected(
            "DedupGather strategy is handled elsewhere, not by make_axis_dispatch. "
            "Use DedupGather via BatchPlanner + _dispatch_axis, not make_axis_dispatch.",
        )

    # Reject Scan on heterogeneous axes.
    if isinstance(strategy, Scan):
        het_axes = heterogeneous_axes or set()
        if axis in het_axes:
            raise DispatchRejected(
                f"Cannot use Scan strategy on {axis} axis: {axis} axis contains "
                "heterogeneous (variable-shape) state elements. Scan requires "
                "static carry shape across all iterations.",
            )

    # Dispatch by strategy type.
    if isinstance(strategy, Vmap):
        return VmapIterator()
    if isinstance(strategy, SafeMap):
        return SafeMapIterator(tile=strategy.batch_size)
    if isinstance(strategy, Scan):
        return JaxScanIterator()
    # Exhaustiveness check: should never reach here if AxisStrategy is sealed.
    raise TypeError(f"Unknown strategy type: {type(strategy)}")


def axis_dispatch(
    strategy: AxisStrategy, fn: Callable[..., Any] | None, xs: Any, init: Any = None
) -> Any:
    """Eager shim for backward compatibility with old 4-arg eager API.

    Preserves the old eager dispatch behavior for DedupGather and Bucket,
    which make_axis_dispatch() rejects (they're handled elsewhere).
    New code should use make_axis_dispatch(strategy) and call the iterator directly.

    Args:
        strategy: The AxisStrategy to dispatch with.
        fn: Callable to apply.
        xs: Input pytree to process.
        init: Initial carry state (for Scan strategies).

    Returns:
        Result of applying the strategy.
    """
    from xtrax.tiling.strategy import Bucket, DedupGather

    if isinstance(strategy, Vmap):
        # Vmap: vectorize over the leading axis
        return jax.vmap(fn)(xs)

    elif isinstance(strategy, SafeMap):
        # SafeMap: chunked vmap for memory efficiency
        if fn is None:
            raise ValueError("axis_dispatch: SafeMap requires fn")
        return safe_map(fn, xs, batch_size=strategy.batch_size)

    elif isinstance(strategy, Scan):
        # Scan: carry-bearing sequential iteration
        # strategy.transition is the actual computation; fn is ignored
        if strategy.transition is None:
            raise ValueError(
                "axis_dispatch: Scan strategy requires a non-None transition function. "
                "Provide via Scan(transition=...) argument."
            )
        # Use strategy.init if available, else fall back to init parameter
        carry_init = strategy.init if strategy.init is not None else init
        if carry_init is None:
            raise ValueError(
                "axis_dispatch: Scan strategy requires a non-None init carry. "
                "Provide via Scan.init field or init parameter."
            )
        return safe_scan(strategy.transition, carry_init, xs)

    elif isinstance(strategy, DedupGather):
        # DedupGather: three-phase dedup -> map -> gather
        if fn is None:
            raise ValueError("axis_dispatch: DedupGather requires fn")
        # Phase 1: dedup — select K unique elements from N using unique_indices
        deduped_xs = strategy.dedup_fn(xs, strategy.unique_indices)
        # Phase 2: map — apply fn to K unique elements
        deduped_ys = safe_map(fn, deduped_xs, batch_size=None)  # vmap over deduped
        # Phase 3: gather — scatter K results back to N positions using index_map
        return strategy.gather_fn(deduped_ys, strategy.index_map)

    elif isinstance(strategy, Bucket):
        # Bucket is host-side: bucketing pads variable-length inputs *before* the
        # JIT boundary, so it cannot be executed inside this device-tier dispatch.
        raise TypeError(
            "Bucket is a host-side strategy and is not executed by "
            "axis_dispatch. Pad to a bucket shape on the host with "
            "select_bucket()/bucketize() before your jitted step, then dispatch "
            "the per-bucket compute with a device-tier strategy (e.g. Vmap/SafeMap)."
        )

    else:
        raise TypeError(f"Unknown strategy type: {type(strategy)}")
