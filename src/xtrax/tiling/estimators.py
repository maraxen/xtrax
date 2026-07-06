"""Native-JAX-backed building blocks for MemoryBudget estimators.

These helpers hook xtrax's joint-budget planning into JAX/XLA's own memory
accounting instead of hand-rolled byte math:

- ``device_memory_budget`` derives a budget from the runtime's
  ``Device.memory_stats()`` (the allocator's actual ``bytes_limit``).
- ``lowered_memory_estimate`` compiles a function ahead-of-time from abstract
  inputs and reads XLA's buffer-assignment numbers via
  ``Compiled.memory_analysis()`` — the compiler's own static memory plan, not
  a heuristic.

Both fail loud when the backend cannot answer (no silent defaults), matching
budget mode's strictness contract. A typical ``MemoryBudget.estimate`` calls
``lowered_memory_estimate`` on a representative tile of the computation for
the candidate decisions and scales by the plan's live tile counts.

Spec: .praxia/docs/specs/260706_joint-budget-batch-planner.md
"""

from collections.abc import Callable
from typing import Any

import jax


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
