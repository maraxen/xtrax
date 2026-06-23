"""Axis configuration override types and decorator for signature inference (E1.4)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisOverride:
    """Override specification for a single axis in signature inference.

    All fields except ``name`` and ``default_batch_size`` are optional and
    fall back to structural defaults when not supplied.

    Attributes:
        name: Human-readable axis name (e.g., "batch", "sequence"). Required.
        default_batch_size: Default batch size for this axis (Assumption A3:
            NOT inferable from shape alone). Required.
        cardinality: Override for the number of elements along this axis.
            When None (default), the cardinality is inferred from the leading
            dimension of the corresponding abstract input.
        tile_granularity: Alignment granularity for tiling. Default 1 (no
            constraint).
        heterogeneous: Whether elements have variable shapes. Default False.
        dedup_eligible: Whether this axis is eligible for deduplication.
            Default False.
        bucket_boundaries: Sorted, strictly-ascending bucket sizes for
            length-based padding. None disables bucketing (default).
    """

    name: str
    default_batch_size: int
    cardinality: int | None = None
    tile_granularity: int = 1
    heterogeneous: bool = False
    dedup_eligible: bool = False
    bucket_boundaries: tuple[int, ...] | None = None


_AXIS_CONFIG_ATTR = "__xtrax_axis_config__"


def axis_config(*overrides: AxisOverride) -> Callable:
    """Decorator factory that attaches axis overrides to a function.

    The overrides are stored as an ordered tuple on the decorated function
    under the attribute ``__xtrax_axis_config__``.  The decorator does NOT
    alter the function's call behavior in any way.

    Args:
        *overrides: Zero or more :class:`AxisOverride` instances, one per
            leading axis (positional order matches axis index).

    Returns:
        A decorator that attaches the overrides tuple and returns the function
        unchanged.

    Example::

        @axis_config(AxisOverride(name="batch", default_batch_size=32))
        def my_model(x):
            ...

        cfg = get_axis_config(my_model)
        specs = synthesize_axes(abstract_inputs, overrides=cfg)
    """
    override_tuple: tuple[AxisOverride, ...] = tuple(overrides)

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _AXIS_CONFIG_ATTR, override_tuple)
        return fn

    return decorator


def get_axis_config(fn: Callable) -> tuple[AxisOverride, ...] | None:
    """Return the axis overrides attached to *fn*, or None if absent.

    Args:
        fn: Any callable, decorated or not.

    Returns:
        The tuple of :class:`AxisOverride` objects attached by
        :func:`axis_config`, or ``None`` if the function was not decorated.
    """
    return getattr(fn, _AXIS_CONFIG_ATTR, None)
