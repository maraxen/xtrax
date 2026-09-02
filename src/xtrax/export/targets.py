"""Compilation targets and the depth to which each one is verified.

A ``Target`` is a named IREE backend plus the dtype vocabulary that backend
accepts and the flags it needs. ``VerificationLevel`` records how far this
package is willing to vouch for the resulting artifact, which is deliberately
not the same for every target:

- ``native`` is compiled AND executed, so its numerics can be checked against an
  independently-computed oracle.
- ``wasm32`` is compiled only. Executing it needs an emsdk-built IREE runtime,
  which has no published package, so claiming more would be dishonest.
- ``vulkan-spirv`` and ``metal-spirv`` are compiled only. Executing either needs
  a device this package does not require, and the WebGPU shader-validity gate
  that would once have raised ``vulkan-spirv`` to ``VALIDATED`` was falsified
  before it was built (see ``xtrax.export.spirv``).

No target is registered at ``VALIDATED``; ``export_pipeline`` refuses one, since
it has nothing to populate the result with.

Nothing here imports IREE. Target selection is plain data, available with only
the base install; the toolchain is touched lazily by ``xtrax.export.compile``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "ALL_TARGETS",
    "METAL_SPIRV",
    "NATIVE",
    "VULKAN_SPIRV",
    "WASM32",
    "Target",
    "VerificationLevel",
    "target_by_name",
]


class VerificationLevel(StrEnum):
    """How far a compiled artifact's correctness has actually been established.

    Attributes:
        EXECUTED: Numerics were verified against an independent JAX oracle via
            IREE's native runtime. This bounds *lowering* fidelity (XLA vs.
            IREE); it does not by itself prove the plan was composed correctly.
        CODEGEN_ONLY: The artifact compiled. It was never executed or otherwise
            validated, so ``ExportResult.verified`` is unconditionally False.
        VALIDATED: SPIR-V accepted by a shader validator; never executed.
            ``ExportResult.verified`` mirrors the validation result.
    """

    EXECUTED = "executed"
    CODEGEN_ONLY = "codegen_only"
    VALIDATED = "validated"


@dataclass(frozen=True)
class Target:
    """One compilation target: an IREE backend plus its dtype and flag envelope.

    Attributes:
        name: Key used in ``export_pipeline``'s result dict, e.g. ``"native"``.
        iree_backend: Value passed to ``--iree-hal-target-backends``.
        verification_level: How far this target's artifact is verified.
        supported_dtypes: Dtype names accepted unconditionally.
        optional_dtypes: Dtype names accepted only when the corresponding
            feature in ``optional_dtype_features`` has been requested and is
            available.
        optional_dtype_features: Maps an optional dtype to the device feature
            that unlocks it, e.g. ``{"f16": "shader-f16"}``.
        extra_compiler_flags: Backend-specific flags appended to every
            ``iree-compile`` invocation for this target.
        emits_spirv: Whether compiling for this backend produces SPIR-V shader
            binaries worth extracting. Drives the executable dump in
            ``xtrax.export.compile``; a backend that emits something else keeps
            ``CompileResult.spirv_bytes`` at None.
    """

    name: str
    iree_backend: str
    verification_level: VerificationLevel
    supported_dtypes: frozenset[str]
    optional_dtypes: frozenset[str] = frozenset()
    optional_dtype_features: Mapping[str, str] = field(default_factory=dict)
    extra_compiler_flags: tuple[str, ...] = ()
    emits_spirv: bool = False


# The dtype envelopes below are measured (260902, iree-base-compiler and
# iree-base-runtime 3.11.0), and they split by verification level rather than by
# backend: every backend *compiles* the same set, but a target that promises
# EXECUTED has to survive the runtime as well.
#
# f64 is excluded everywhere, and that is a correction rather than a
# restriction. IREE's ConvertTypesPass demotes f64 to f32 on every backend and
# rewrites the entry point's public signature to match -- `@main(tensor<8xf64>)`
# becomes `@main(tensor<8xf32>)` -- emitting a warning, not an error. Neither
# consequence is acceptable to pass silently:
#
#   - EXECUTED: invoking the artifact with the f64 array the caller asked about
#     fails inside the runtime with `input0 element type mismatch; expected f32
#     but have f64`, a buffer-level diagnostic far from the cause.
#   - CODEGEN_ONLY: nothing is ever invoked, so the mismatch never surfaces at
#     all. The caller receives an artifact silently taking and returning f32.
#
# bf16 is the one genuine level-dependent difference. It compiles on every
# backend and its signature is left alone, but IREE's Python runtime cannot map
# its buffers back to numpy (`Unsupported VM Buffer -> numpy dtype mapping`), so
# an EXECUTED target cannot run it and therefore cannot verify it. A
# CODEGEN_ONLY target never runs anything, so carrying bf16 costs it nothing.
# To get a bf16 model verified, cast it to f32 first -- then the precision loss
# is a decision rather than a surprise.
_EXECUTABLE_DTYPES = frozenset({"f32", "f16", "i32", "i64", "i8", "u32", "bool"})
_CODEGEN_DTYPES = _EXECUTABLE_DTYPES | {"bf16"}

# Native builds target the machine doing the compiling. That is optimal here and
# correct: this artifact is a parity oracle, not something distributed.
NATIVE = Target(
    name="native",
    iree_backend="llvm-cpu",
    verification_level=VerificationLevel.EXECUTED,
    supported_dtypes=_EXECUTABLE_DTYPES,
    extra_compiler_flags=("--iree-llvmcpu-target-cpu=host",),
)

# +simd128 is the meaningful perf lever for wasm CPU codegen; atomics and
# bulk-memory are required by the threaded runtime variants. target-cpu must be
# set explicitly: left unset, IREE warns while creating the CPU target and falls
# back to a generic CPU whose generated code it documents as poorly performing.
WASM32 = Target(
    name="wasm32",
    iree_backend="llvm-cpu",
    verification_level=VerificationLevel.CODEGEN_ONLY,
    supported_dtypes=_CODEGEN_DTYPES,
    extra_compiler_flags=(
        "--iree-llvmcpu-target-triple=wasm32-unknown-emscripten",
        "--iree-llvmcpu-target-cpu=generic",
        "--iree-llvmcpu-target-cpu-features=+simd128,+atomics,+bulk-memory",
    ),
)

# Both SPIR-V targets are CODEGEN_ONLY. The spec originally registered
# vulkan-spirv at VALIDATED behind a WebGPU shader-validity gate; that gate was
# falsified before implementation (IREE's Vulkan HAL uses push constants, which
# WebGPU has no capability for) and neither target is executed here, so
# CODEGEN_ONLY is the whole of what compiling establishes.
#
# Their dtype envelope is _CODEGEN_DTYPES, the same as wasm32's. The spec gave
# them a narrower, WebGPU-derived table ({f32, i32, bool} plus f16 behind a
# "shader-f16" feature), but no backend-dependent dtype behaviour turned up in
# IREE 3.11 at all -- the emitted SPIR-V declares only Shader and Matrix, for
# every dtype, never Float16 or Float64. Populating optional_dtypes here would
# invent a distinction the toolchain does not make; the machinery stays
# available for a target that genuinely needs it.
VULKAN_SPIRV = Target(
    name="vulkan-spirv",
    iree_backend="vulkan-spirv",
    verification_level=VerificationLevel.CODEGEN_ONLY,
    supported_dtypes=_CODEGEN_DTYPES,
    emits_spirv=True,
)

# metal-spirv is named for its input dialect, not its output: it dumps Metal
# Shading Language source, whose head is the ASCII "#inc" of an #include line
# rather than the SPIR-V magic. Hence emits_spirv=False -- there is no SPIR-V to
# extract, and the magic filter would reject the MSL anyway.
METAL_SPIRV = Target(
    name="metal-spirv",
    iree_backend="metal-spirv",
    verification_level=VerificationLevel.CODEGEN_ONLY,
    supported_dtypes=_CODEGEN_DTYPES,
    emits_spirv=False,
)

ALL_TARGETS: tuple[Target, ...] = (NATIVE, WASM32, VULKAN_SPIRV, METAL_SPIRV)


def target_by_name(name: str) -> Target:
    """Look up a registered target by its ``name``.

    Args:
        name: A target name, e.g. ``"native"``.

    Returns:
        The matching ``Target``.

    Raises:
        KeyError: If no registered target has that name.
    """
    for target in ALL_TARGETS:
        if target.name == name:
            return target
    known = ", ".join(sorted(t.name for t in ALL_TARGETS))
    msg = f"unknown target {name!r}; registered targets are: {known}"
    raise KeyError(msg)
