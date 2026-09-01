"""Compile an xtrax pipeline to a standalone artifact via StableHLO and IREE.

This package folds a ``BatchPlan`` into one traceable callable, exports it as
StableHLO, and compiles that for one or more targets. Importing it requires
nothing beyond the base install; the IREE toolchain is imported lazily, so a
missing ``export`` extra surfaces as a clear error at compile time rather than
an ImportError at module load.

This module is a re-export shim and deliberately holds no logic of its own.
"""

from xtrax.export.spirv import SpirvValidationResult
from xtrax.export.targets import (
    ALL_TARGETS,
    NATIVE,
    WASM32,
    Target,
    VerificationLevel,
    target_by_name,
)

__all__ = [
    "ALL_TARGETS",
    "NATIVE",
    "WASM32",
    "SpirvValidationResult",
    "Target",
    "VerificationLevel",
    "target_by_name",
]
