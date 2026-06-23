"""Signature inference package for xtrax.

This package provides tools for inferring and validating axis signatures,
roles, and structural properties of batched computations.
"""

from __future__ import annotations

from xtrax.inference.errors import AmbiguousAxisError, AxisRole, StructureMismatchError

__all__ = [
    "AmbiguousAxisError",
    "StructureMismatchError",
    "AxisRole",
]
