"""Tiling module for composable axis strategy selection and execution.

CORE exports (stable, always available):
    AxisSpec, BatchPlanner, BatchPlan, AxisDecision, declare_varying_inputs
    Vmap, ChunkedMap, Scan, ScanTransition, Bucket, select_bucket, bucketize,
    BUCKET_LADDER, valid_span, select_rung, trim_axis, pad_axis
    WhileCarry, StaticWhile, WhileBodyFn, WhileCondFn, fixed_step_count_cond
    make_axis_dispatch, axis_dispatch, DispatchRejected
    CarrySpec, CarryShape
    MemoryBudget, BudgetInfeasibleError, device_memory_budget, lowered_memory_estimate,
    estimate_memory_theoretical, plan_axis
    VmapIterator, ChunkedMapIterator, JaxScanIterator, WhileLoopIterator,
    WhileLoopWithYsIterator, BucketIterator,
    MapIterator, ScanIterator

OPTIONAL (dedup/gather machinery — import from submodules):
    xtrax.tiling.strategy: DedupGather, DedupFn, GatherFn
    xtrax.tiling.dedup:    DedupSpec, get_k_bucket
"""

from typing import TYPE_CHECKING, Any

from xtrax.tiling._plan_wrapper import _BatchPlanWrapper
from xtrax.tiling.bucket import (
    BUCKET_LADDER,
    bucketize,
    pad_axis,
    select_bucket,
    select_rung,
    trim_axis,
    valid_span,
)
from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.carry_shape import CarryShape
from xtrax.tiling.dispatch import DispatchRejected, axis_dispatch, make_axis_dispatch
from xtrax.tiling.estimators import (
    device_memory_budget,
    estimate_memory_theoretical,
    lowered_memory_estimate,
)
from xtrax.tiling.plan import (
    AxisDecision,
    AxisSpec,
    BatchPlan,
    BatchPlanner,
    declare_varying_inputs,
    plan_axis,
)
from xtrax.tiling.strategy import (
    Bucket,
    ChunkedMap,
    Scan,
    ScanTransition,
    StaticWhile,
    Vmap,
    WhileBodyFn,
    WhileCarry,
    WhileCondFn,
    fixed_step_count_cond,
)

__all__ = [
    "AxisSpec",
    "AxisDecision",
    "BatchPlan",
    "BatchPlanner",
    "declare_varying_inputs",
    "Vmap",
    "ChunkedMap",
    "Scan",
    "Bucket",
    "BUCKET_LADDER",
    "select_bucket",
    "bucketize",
    "valid_span",
    "select_rung",
    "trim_axis",
    "pad_axis",
    "ScanTransition",
    "WhileCarry",
    "StaticWhile",
    "WhileBodyFn",
    "WhileCondFn",
    "fixed_step_count_cond",
    "make_axis_dispatch",
    "axis_dispatch",
    "DispatchRejected",
    "CarrySpec",
    "CarryShape",
    "MemoryBudget",
    "BudgetInfeasibleError",
    "device_memory_budget",
    "lowered_memory_estimate",
    "estimate_memory_theoretical",
    "plan_axis",
    "VmapIterator",
    "ChunkedMapIterator",
    "JaxScanIterator",
    "WhileLoopIterator",
    "WhileLoopWithYsIterator",
    "BucketIterator",
    "MapIterator",
    "ScanIterator",
    "_BatchPlanWrapper",
]

if TYPE_CHECKING:
    from xtrax.tiling.iterator import (
        BucketIterator,
        ChunkedMapIterator,
        JaxScanIterator,
        MapIterator,
        ScanIterator,
        VmapIterator,
        WhileLoopIterator,
        WhileLoopWithYsIterator,
    )

_LAZY_ITERATORS = frozenset(
    {
        "BucketIterator",
        "ChunkedMapIterator",
        "JaxScanIterator",
        "MapIterator",
        "ScanIterator",
        "VmapIterator",
        "WhileLoopIterator",
        "WhileLoopWithYsIterator",
    }
)


def __getattr__(name: str) -> Any:
    """Load iterator classes on first access.

    Eager import of :mod:`xtrax.tiling.iterator` pulls in equinox. Callers
    that only need the rest of this package, including ``xtrax.intent``,
    therefore do not load equinox until they touch an iterator name.

    Parameters
    ----------
    name : str
        Attribute being accessed.

    Returns
    -------
    object
        The iterator class from :mod:`xtrax.tiling.iterator`.

    Raises
    ------
    AttributeError
        ``name`` is not a lazy iterator export.
    """
    if name in _LAZY_ITERATORS:
        import importlib

        return getattr(importlib.import_module("xtrax.tiling.iterator"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return sorted list of public attributes."""
    return sorted(__all__)
