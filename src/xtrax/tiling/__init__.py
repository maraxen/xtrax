"""Tiling module for composable axis strategy selection and execution.

CORE exports (stable, always available):
    AxisSpec, BatchPlanner, BatchPlan, AxisDecision
    Vmap, ChunkedMap, Scan, ScanTransition, Bucket, select_bucket, bucketize,
    BUCKET_LADDER, valid_span, select_rung, trim_axis, pad_axis
    WhileCarry, WhileBodyFn, WhileCondFn, fixed_step_count_cond
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
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan, BatchPlanner, plan_axis
from xtrax.tiling.strategy import (
    Bucket,
    ChunkedMap,
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
