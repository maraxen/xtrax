"""Tiling module for composable axis strategy selection and execution.

CORE exports (stable, always available):
    AxisSpec, BatchPlanner, BatchPlan, AxisDecision
    Vmap, SafeMap, Scan, ScanTransition, Bucket, select_bucket, bucketize
    WhileCarry, WhileBodyFn, WhileCondFn, fixed_step_count_cond
    make_axis_dispatch, axis_dispatch, DispatchRejected
    CarrySpec, CarryShape
    MemoryBudget, BudgetInfeasibleError, device_memory_budget, lowered_memory_estimate
    VmapIterator, SafeMapIterator, JaxScanIterator, WhileLoopIterator, BucketIterator,
    MapIterator, ScanIterator

OPTIONAL (dedup/gather machinery — import from submodules):
    xtrax.tiling.strategy: DedupGather, DedupFn, GatherFn
    xtrax.tiling.dedup:    DedupSpec, get_k_bucket
"""

from xtrax.tiling._plan_wrapper import _BatchPlanWrapper
from xtrax.tiling.bucket import bucketize, select_bucket
from xtrax.tiling.budget import BudgetInfeasibleError, MemoryBudget
from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.carry_shape import CarryShape
from xtrax.tiling.dispatch import DispatchRejected, axis_dispatch, make_axis_dispatch
from xtrax.tiling.estimators import device_memory_budget, lowered_memory_estimate
from xtrax.tiling.iterator import (
    BucketIterator,
    JaxScanIterator,
    MapIterator,
    SafeMapIterator,
    ScanIterator,
    VmapIterator,
    WhileLoopIterator,
)
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan, BatchPlanner
from xtrax.tiling.strategy import (
    Bucket,
    SafeMap,
    Scan,
    ScanTransition,
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
    "Vmap",
    "SafeMap",
    "Scan",
    "Bucket",
    "select_bucket",
    "bucketize",
    "ScanTransition",
    "WhileCarry",
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
    "VmapIterator",
    "SafeMapIterator",
    "JaxScanIterator",
    "WhileLoopIterator",
    "BucketIterator",
    "MapIterator",
    "ScanIterator",
    "_BatchPlanWrapper",
]
