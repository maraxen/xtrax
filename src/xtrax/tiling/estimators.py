"""Building blocks for MemoryBudget estimators.

``device_memory_budget`` and ``lowered_memory_estimate`` hook joint-budget
planning into JAX/XLA's own memory accounting:

- ``device_memory_budget`` derives a budget from the runtime's
  ``Device.memory_stats()`` (the allocator's actual ``bytes_limit``).
- ``lowered_memory_estimate`` compiles a function ahead-of-time from abstract
  inputs and reads XLA's buffer-assignment numbers via
  ``Compiled.memory_analysis()`` — the compiler's own static memory plan, not
  a heuristic.

``estimate_memory_theoretical`` is the domain-free product-of-extents
fallback when a compile is unnecessary.

The two JAX helpers fail loud when the backend cannot answer (no silent
defaults), matching budget mode's strictness contract. Callers that still need
a number when the device cannot answer (``BatchPlanner``'s per-axis estimator,
``xtrax.config.resolve_memory_budget``) log and use
``DEFAULT_DEVICE_MEMORY_BYTES`` instead of substituting it silently.
A typical ``MemoryBudget.estimate`` calls ``lowered_memory_estimate`` on a
representative tile of the computation for the candidate decisions and scales
by the plan's live tile counts.

Spec: .praxia/docs/specs/260706_joint-budget-batch-planner.md
"""

from collections.abc import Callable, Mapping
from typing import Any

import jax

# Documented fallback when the runtime does not report bytes_limit.
# ``device_memory_budget`` itself still raises; BatchPlanner and
# ``resolve_memory_budget`` log once and use this (headroom scales it there).
DEFAULT_DEVICE_MEMORY_BYTES: int = 4 * 1024**3


def device_memory_budget(fraction: float = 0.9, device: Any | None = None) -> int:
    """Derive a MemoryBudget byte count from the device allocator's limit.

    Reads ``device.memory_stats()["bytes_limit"]`` — the XLA allocator's real
    limit for the device — and applies a safety fraction.

    Args:
        fraction: Fraction of the allocator limit to budget (0 < fraction <= 1).
            Default 0.9 leaves headroom for allocator fragmentation and
            non-plan buffers.
        device: Device to query. Defaults to ``jax.devices()[0]``.

    Returns:
        Budget in bytes: ``int(bytes_limit * fraction)``.

    Raises:
        ValueError: If fraction is not in (0, 1].
        RuntimeError: If the device does not report memory stats (e.g. some
            CPU backends). No silent fallback — pass an explicit budget
            instead when the runtime cannot answer.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    dev = device if device is not None else jax.devices()[0]
    stats = dev.memory_stats()
    if not stats or "bytes_limit" not in stats:
        raise RuntimeError(
            f"device {dev} does not report memory_stats()['bytes_limit']; "
            f"pass an explicit MemoryBudget byte count instead."
        )
    return int(stats["bytes_limit"] * fraction)


def lowered_memory_estimate(fn: Callable[..., Any], *abstract_args: Any) -> int:
    """Estimate peak memory of ``fn`` from XLA's own buffer assignment.

    Lowers and compiles ``fn`` ahead-of-time for the given abstract inputs
    (``jax.ShapeDtypeStruct`` or concrete arrays — only shapes/dtypes are
    used) and reads ``Compiled.memory_analysis()``: the static memory plan
    XLA computed at compile time. Returns argument + output + temp bytes.

    This is a compile, so it costs seconds per distinct shape signature.
    Inside a greedy MemoryBudget.estimate (called once per demotion step),
    estimate a representative tile and scale analytically, or memoize on the
    decision signature.

    Args:
        fn: JAX-traceable callable to analyze.
        *abstract_args: Abstract inputs, e.g.
            ``jax.ShapeDtypeStruct((1024, 128), jnp.float32)``.

    Returns:
        Estimated peak bytes (argument_size + output_size + temp_size from
        XLA buffer assignment).

    Raises:
        RuntimeError: If the backend provides no memory analysis for this
            computation. No silent fallback.
    """
    compiled = jax.jit(fn).lower(*abstract_args).compile()
    analysis = compiled.memory_analysis()
    sizes = [
        getattr(analysis, attr, None)
        for attr in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
    ]
    if analysis is None or any(size is None for size in sizes):
        raise RuntimeError(
            "backend returned no usable memory_analysis() for this computation; "
            "provide a hand-written MemoryBudget.estimate instead."
        )
    return int(sum(size for size in sizes if size is not None))


def estimate_memory_theoretical(
    extents: Mapping[str, int],
    bytes_per_element: int | float,
    *,
    dtype_bytes: int = 1,
    activation_multiplier: int | float = 1.0,
) -> int:
    """Estimate peak bytes as the product of live extents times per-element bytes.

    Domain-free form of the theoretical product used by joint-budget callers:
    each extent is how many elements of that axis are live at once (full
    cardinality when the axis is mapped, tile size when only one tile is
    live). The product of those extents is the number of elements live
    together. Multiply by ``bytes_per_element``.

    ``dtype_bytes`` defaults to 1 so a ``bytes_per_element`` that already
    includes the dtype width is unchanged. Pass ``dtype_bytes`` when
    ``bytes_per_element`` counts values rather than bytes (for example
    ``dtype_bytes=4`` for float32). ``activation_multiplier`` scales the
    product for activation overhead; it defaults to 1.

    An empty ``extents`` mapping has product 1 (the base element only).
    The result is truncated toward zero to an int, which is the unit
    ``MemoryBudget.estimate`` returns.

    Args:
        extents: Axis name to live-element count. Names document the product;
            only the values are multiplied.
        bytes_per_element: Bytes (or values, if ``dtype_bytes`` is set) for
            one live element. Must be >= 0.
        dtype_bytes: Positive byte width applied on top of
            ``bytes_per_element``. Default 1.
        activation_multiplier: Non-negative scale factor. Default 1.

    Returns:
        Estimated peak bytes.

    Raises:
        TypeError: If an extent or ``dtype_bytes`` is not an int, or if
            ``bytes_per_element`` / ``activation_multiplier`` is not a real
            number. Bool is rejected.
        ValueError: If an extent or ``bytes_per_element`` is negative, if
            ``dtype_bytes`` is not positive, or if ``activation_multiplier``
            is negative.
    """
    if isinstance(bytes_per_element, bool) or not isinstance(bytes_per_element, (int, float)):
        raise TypeError(
            f"bytes_per_element must be a real number, got {type(bytes_per_element).__name__}"
        )
    if bytes_per_element < 0:
        raise ValueError(f"bytes_per_element must be >= 0, got {bytes_per_element}")
    if isinstance(dtype_bytes, bool) or not isinstance(dtype_bytes, int):
        raise TypeError(f"dtype_bytes must be an int, got {type(dtype_bytes).__name__}")
    if dtype_bytes <= 0:
        raise ValueError(f"dtype_bytes must be positive, got {dtype_bytes}")
    if isinstance(activation_multiplier, bool) or not isinstance(
        activation_multiplier, (int, float)
    ):
        raise TypeError(
            "activation_multiplier must be a real number, "
            f"got {type(activation_multiplier).__name__}"
        )
    if activation_multiplier < 0:
        raise ValueError(f"activation_multiplier must be >= 0, got {activation_multiplier}")

    product = 1
    for name, extent in extents.items():
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"extent for {name!r} must be an int, got {type(extent).__name__}")
        if extent < 0:
            raise ValueError(f"extent for {name!r} must be >= 0, got {extent}")
        product *= extent
    return int(product * bytes_per_element * dtype_bytes * activation_multiplier)
