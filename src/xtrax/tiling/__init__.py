"""Tiling module for composable axis strategy selection and execution."""

from xtrax.tiling.bucket import bucketize, select_bucket
from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.carry_shape import CarryShape
from xtrax.tiling.dedup import DedupSpec, get_k_bucket
from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan, BatchPlanner
from xtrax.tiling.strategy import (
    Bucket,
    DedupFn,
    DedupGather,
    GatherFn,
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
    "DedupGather",
    "Bucket",
    "select_bucket",
    "bucketize",
    "ScanTransition",
    "DedupFn",
    "GatherFn",
    "make_axis_dispatch",
    "CarrySpec",
    "CarryShape",
    "DedupSpec",
    "get_k_bucket",
]
