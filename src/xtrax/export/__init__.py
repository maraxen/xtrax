"""Compile an xtrax pipeline to a standalone artifact via StableHLO and IREE.

This package folds a ``BatchPlan`` into one traceable callable, exports it as
StableHLO, and compiles that for one or more targets. Importing it requires
nothing beyond the base install; the IREE toolchain is imported lazily, so a
missing ``export`` extra surfaces as a clear error at compile time rather than
an ImportError at module load.

This module is a re-export shim and deliberately holds no logic of its own.
"""

from xtrax.export.compile import (
    CompileError,
    CompileResult,
    compile_for_target,
    run_native_vmfb,
)
from xtrax.export.composer import (
    ComposerError,
    MultiAxisCompositionError,
    UnsupportedStrategyError,
    build_traceable_callable,
    compose_single_axis,
)
from xtrax.export.parity import ParityResult, compare, verify_native_parity
from xtrax.export.pipeline import ExportResult, export_pipeline
from xtrax.export.safety import (
    DtypeNotSupportedError,
    ExportBlocker,
    ExportSafetyError,
    check_export_safety,
    find_bcoo_leaves,
    validate_export_safe,
)
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
    "CompileError",
    "CompileResult",
    "ComposerError",
    "DtypeNotSupportedError",
    "ExportBlocker",
    "ExportResult",
    "ExportSafetyError",
    "MultiAxisCompositionError",
    "ParityResult",
    "SpirvValidationResult",
    "Target",
    "UnsupportedStrategyError",
    "VerificationLevel",
    "build_traceable_callable",
    "check_export_safety",
    "compare",
    "compile_for_target",
    "compose_single_axis",
    "export_pipeline",
    "find_bcoo_leaves",
    "run_native_vmfb",
    "target_by_name",
    "validate_export_safe",
    "verify_native_parity",
]
