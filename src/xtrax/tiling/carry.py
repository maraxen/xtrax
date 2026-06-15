"""CarrySpec — declare a carry-bearing scan on a named axis.

Used by SamplingSpecification (and custom experiment specs) to indicate which
axes should use jax.lax.scan with a carry, rather than safe_map (stateless).
BatchPlanner.plan() reads CarrySpec list in Phase 0 and pre-demotes matching
axes to Scan(init, transition) decisions before Phases 1 and 2.

CONSTRAINT: Heterogeneous axes (shapes vary per element) cannot be scanned —
jax.lax.scan requires static carry shape. CarrySpec validation is delegated to
BatchPlanner, which accepts heterogeneous_axes as an injected parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xtrax.tiling.strategy import ScanTransition


@dataclass(frozen=True)
class CarrySpec:
  """Declare carry-bearing scan on a named axis.

  Attributes:
      axis_name: Name of the axis (must match AxisSpec.name in the planner,
          e.g. "n_noises", "n_samples", "n_temperatures").
      init: Initial carry value. May contain JAX arrays (traced leaves).
          Shape must be static at JAX trace time.
      transition: (carry, x) -> (carry, y) function. Must be a ScanTransition.
      ordered_sinks: If True, any Sink/Tap on this axis uses ordered=True
          in io_callback (step-ordered guarantees). Default: True.

  Note:
      Validation that axis_name is not heterogeneous is performed by
      BatchPlanner.plan(), which checks against the heterogeneous_axes
      parameter passed at initialization.

  """

  axis_name: str
  init: Any
  transition: ScanTransition
  ordered_sinks: bool = True


__all__ = ["CarrySpec"]
