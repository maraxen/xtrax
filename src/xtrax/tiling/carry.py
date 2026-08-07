"""CarrySpec — declare a carry-bearing scan or while-loop on a named axis.

Used by SamplingSpecification (and custom experiment specs) to indicate which
axes should use jax.lax.scan with a carry, rather than safe_map (stateless).
BatchPlanner.plan() reads CarrySpec list in Phase 0 and pre-demotes matching
axes to Scan(init, transition) decisions before Phases 1 and 2 -- or, when
collect_outputs=False, to WhileCarry(init, body, cond) instead.

CONSTRAINT: Heterogeneous axes (shapes vary per element) cannot be scanned —
jax.lax.scan requires static carry shape. CarrySpec validation is delegated to
BatchPlanner, which accepts heterogeneous_axes as an injected parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xtrax.tiling.strategy import ScanTransition, WhileBodyFn, WhileCondFn


@dataclass(frozen=True)
class CarrySpec:
    """Declare a carry-bearing scan or while-loop on a named axis.

    Attributes:
        axis_name: Name of the axis (must match AxisSpec.name in the planner,
            e.g. "n_noises", "n_samples", "n_temperatures").
        init: Initial carry value. May contain JAX arrays (traced leaves).
            Shape must be static at JAX trace time.
        transition: (carry, x) -> (carry, y) function when collect_outputs=True
            (must be a ScanTransition); carry -> carry when collect_outputs=False
            (must be a WhileBodyFn).
        ordered_sinks: If True, any Sink/Tap on this axis uses ordered=True
            in io_callback (step-ordered guarantees). Default: True.
        collect_outputs: If True (default), pre-demotes to Scan (per-step output
            collection). If False, pre-demotes to WhileCarry instead -- a
            carry-only lax.while_loop with no per-step output collection.
        cond: carry -> bool (traced scalar) continuation predicate. Only used
            when collect_outputs=False; ignored (may be None) otherwise.

    Note:
        Validation that axis_name is not heterogeneous is performed by
        BatchPlanner.plan(), which checks against the heterogeneous_axes
        parameter passed at initialization.

    """

    axis_name: str
    init: Any
    transition: ScanTransition | WhileBodyFn
    ordered_sinks: bool = True
    collect_outputs: bool = True
    cond: WhileCondFn | None = None


__all__ = ["CarrySpec"]
