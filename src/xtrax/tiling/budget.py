"""MemoryBudget — declare a joint memory budget for BatchPlanner.

Used by BatchPlanner callers who need whole-plan memory planning: start every
eligible axis at Vmap, then greedily demote axes to SafeMap (in the order specs
were given) until the joint estimate fits the budget. Mirrors the
CarrySpec/DedupSpec pattern: caller declares, planner enforces.

Unlike the per-axis ``memory_estimator`` (which swallows estimator errors and
reads the device limit implicitly), budget mode is strict: estimator exceptions
propagate, and a plan that cannot fit raises BudgetInfeasibleError.

Spec: .praxia/docs/specs/260706_joint-budget-batch-planner.md
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xtrax.tiling.plan import AxisDecision


class BudgetInfeasibleError(Exception):
    """Raised when no demotion sequence brings the plan under its MemoryBudget.

    Every demotion candidate has been demoted to SafeMap and the joint
    estimate still exceeds ``MemoryBudget.bytes``. The message names the
    budget, the final estimate, and the per-axis strategy state so the caller
    can see exactly what was tried.
    """


@dataclass(frozen=True)
class MemoryBudget:
    """Total-plan memory budget with a joint estimator.

    Attributes:
        bytes: Total memory budget for the whole plan, in bytes. Must be a
            positive int.
        estimate: Function mapping the full decision sequence (in spec order,
            including carry/dedup/bucket decisions) to an estimated peak
            memory in bytes. Exceptions raised by this function propagate
            unchanged — there is no silent fallback in budget mode.
    """

    bytes: int
    estimate: Callable[[Sequence["AxisDecision"]], int]

    def __post_init__(self) -> None:
        """Validate budget fields, failing loud at construction time."""
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise TypeError(f"MemoryBudget.bytes must be an int, got {type(self.bytes).__name__}")
        if self.bytes <= 0:
            raise ValueError(f"MemoryBudget.bytes must be positive, got {self.bytes}")
        if not callable(self.estimate):
            raise TypeError("MemoryBudget.estimate must be callable")
