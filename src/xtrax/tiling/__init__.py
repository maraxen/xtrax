"""Tiling module for composable axis strategy selection and execution."""

from xtrax.tiling.dispatch import make_axis_dispatch
from xtrax.tiling.plan import AxisDecision, AxisSpec, BatchPlan, BatchPlanner
from xtrax.tiling.strategy import (
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
    "ScanTransition",
    "DedupFn",
    "GatherFn",
    "make_axis_dispatch",
]
