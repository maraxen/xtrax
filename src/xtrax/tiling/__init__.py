"""Tiling module for composable axis strategy selection and execution.

CORE exports (stable, always available):
    AxisSpec, BatchPlanner, BatchPlan, AxisDecision
    Vmap, SafeMap, Scan, ScanTransition, Bucket, select_bucket, bucketize
    make_axis_dispatch, axis_dispatch, DispatchRejected
    CarrySpec, CarryShape
    VmapIterator, SafeMapIterator, JaxScanIterator, BucketIterator, MapIterator, ScanIterator

OPTIONAL (dedup/gather machinery — import from submodules):
    xtrax.tiling.strategy: DedupGather, DedupFn, GatherFn
    xtrax.tiling.dedup:    DedupSpec, get_k_bucket
"""

from xtrax.tiling.bucket import bucketize, select_bucket
from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.carry_shape import CarryShape
from xtrax.tiling.dispatch import DispatchRejected, axis_dispatch, make_axis_dispatch
from xtrax.tiling.iterator import (
    BucketIterator,
    JaxScanIterator,
    MapIterator,
    SafeMapIterator,
    ScanIterator,
    VmapIterator,
)
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan, BatchPlanner
from xtrax.tiling.strategy import (
    Bucket,
    SafeMap,
    Scan,
    ScanTransition,
    Vmap,
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
    "make_axis_dispatch",
    "axis_dispatch",
    "DispatchRejected",
    "CarrySpec",
    "CarryShape",
    "VmapIterator",
    "SafeMapIterator",
    "JaxScanIterator",
    "BucketIterator",
    "MapIterator",
    "ScanIterator",
]
