"""Signature inference package for xtrax.

This package provides tools for inferring and validating axis signatures,
roles, and structural properties of batched computations.
"""

from __future__ import annotations

from xtrax.inference.api import infer_bundle
from xtrax.inference.axes import synthesize_axes
from xtrax.inference.config import AxisOverride, axis_config
from xtrax.inference.cse import CseDuplicateClass, CseReport, analyze_cse
from xtrax.inference.errors import (
    AmbiguousAxisError,
    AxisRole,
    CseTraceError,
    MemoImpurityError,
    MemoKeyUnsupportedLeafError,
    MemoMultiDeviceError,
    MemoStalenessError,
    StructureMismatchError,
)
from xtrax.inference.ir_schema import emit_ir_schema
from xtrax.inference.memo import MemoPolicy, memoize_jaxpr
from xtrax.inference.schema import BundleSchema

__all__ = [
    "AmbiguousAxisError",
    "AxisOverride",
    "AxisRole",
    "BundleSchema",
    "CseDuplicateClass",
    "CseReport",
    "CseTraceError",
    "MemoImpurityError",
    "MemoKeyUnsupportedLeafError",
    "MemoMultiDeviceError",
    "MemoPolicy",
    "MemoStalenessError",
    "StructureMismatchError",
    "analyze_cse",
    "axis_config",
    "emit_ir_schema",
    "infer_bundle",
    "memoize_jaxpr",
    "synthesize_axes",
]
