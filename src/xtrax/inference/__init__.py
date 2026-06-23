"""Signature inference package for xtrax.

This package provides tools for inferring and validating axis signatures,
roles, and structural properties of batched computations.
"""

from __future__ import annotations

from xtrax.inference.api import infer_bundle
from xtrax.inference.axes import synthesize_axes
from xtrax.inference.config import AxisOverride, axis_config
from xtrax.inference.errors import AmbiguousAxisError, AxisRole, StructureMismatchError
from xtrax.inference.schema import BundleSchema

__all__ = [
    "AmbiguousAxisError",
    "AxisOverride",
    "AxisRole",
    "BundleSchema",
    "StructureMismatchError",
    "axis_config",
    "infer_bundle",
    "synthesize_axes",
]
