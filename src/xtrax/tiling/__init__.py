"""Tiling module for composable axis strategy selection and execution."""

from xtrax.tiling.bucket import bucketize, select_bucket
from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan, BatchPlanner
from xtrax.tiling.strategy import (
    Bucket,
    DedupFn,
    DedupGather,
    GatherFn,
    SafeMap,
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
    "DedupGather",
    "Bucket",
    "select_bucket",
    "bucketize",
    "ScanTransition",
    "DedupFn",
    "GatherFn",
    "make_axis_dispatch",
]
