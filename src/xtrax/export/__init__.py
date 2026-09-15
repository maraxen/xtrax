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
    compose_vmap_of_scan,
)
from xtrax.export.divergence import (
    SLACK,
    ULP_FLOOR,
    DivergenceClass,
    LeafDivergence,
    MissingBudgetError,
    ProbeDependencyError,
    ProbeReport,
    ProbeResolution,
    ProbeStructureError,
    RingResult,
    Severity,
    budget_key,
    budget_leaf,
    classify_probes,
    compare_pytree,
    probe_resolution,
    validate_probe_deps,
)
from xtrax.export.hf_weights import (
    HFWeightsError,
    LoadedWeights,
    WeightReport,
    load_hf_weights,
)
from xtrax.export.parity import ParityResult, compare, verify_native_parity
from xtrax.export.pipeline import ExportResult, export_pipeline
from xtrax.export.rings import (
    BUCKET_LADDER,
    InputClassResult,
    magnitude_extremes,
    nominal,
    r0_replay_gate,
    r1_target_isa,
    r2a_fusion,
    r2b_lowering,
    r3_probe,
    recover_probe_names,
    run_ladder,
    sub_k_neighbours,
    symmetric_geometry,
)
from xtrax.export.safety import (
    DtypeNotSupportedError,
    ExportBlocker,
    ExportSafetyError,
    check_export_safety,
    find_bcoo_leaves,
    validate_export_safe,
)
from xtrax.export.spirv import SpirvValidationResult, is_spirv, spirv_binaries_in
from xtrax.export.targets import (
    ALL_TARGETS,
    METAL_SPIRV,
    NATIVE,
    NATIVE_PORTABLE,
    VULKAN_SPIRV,
    WASM32,
    Target,
    VerificationLevel,
    target_by_name,
)

__all__ = [
    "ALL_TARGETS",
    "BUCKET_LADDER",
    "METAL_SPIRV",
    "NATIVE",
    "NATIVE_PORTABLE",
    "SLACK",
    "ULP_FLOOR",
    "VULKAN_SPIRV",
    "WASM32",
    "CompileError",
    "CompileResult",
    "ComposerError",
    "DivergenceClass",
    "DtypeNotSupportedError",
    "ExportBlocker",
    "ExportResult",
    "ExportSafetyError",
    "HFWeightsError",
    "InputClassResult",
    "LeafDivergence",
    "LoadedWeights",
    "MissingBudgetError",
    "MultiAxisCompositionError",
    "ParityResult",
    "ProbeDependencyError",
    "ProbeReport",
    "ProbeResolution",
    "ProbeStructureError",
    "RingResult",
    "Severity",
    "SpirvValidationResult",
    "Target",
    "UnsupportedStrategyError",
    "VerificationLevel",
    "WeightReport",
    "budget_key",
    "budget_leaf",
    "build_traceable_callable",
    "check_export_safety",
    "classify_probes",
    "compare",
    "compare_pytree",
    "compile_for_target",
    "compose_single_axis",
    "compose_vmap_of_scan",
    "export_pipeline",
    "find_bcoo_leaves",
    "is_spirv",
    "load_hf_weights",
    "magnitude_extremes",
    "nominal",
    "probe_resolution",
    "r0_replay_gate",
    "r1_target_isa",
    "r2a_fusion",
    "r2b_lowering",
    "r3_probe",
    "recover_probe_names",
    "run_ladder",
    "run_native_vmfb",
    "spirv_binaries_in",
    "sub_k_neighbours",
    "symmetric_geometry",
    "target_by_name",
    "validate_export_safe",
    "validate_probe_deps",
    "verify_native_parity",
]
