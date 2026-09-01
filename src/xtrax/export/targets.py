"""Compilation targets and the depth to which each one is verified.

A ``Target`` is a named IREE backend plus the dtype vocabulary that backend
accepts and the flags it needs. ``VerificationLevel`` records how far this
package is willing to vouch for the resulting artifact, which is deliberately
not the same for every target:

- ``native`` is compiled AND executed, so its numerics can be checked against an
  independently-computed oracle.
- ``wasm32`` is compiled only. Executing it needs an emsdk-built IREE runtime,
  which has no published package, so claiming more would be dishonest.

Nothing here imports IREE. Target selection is plain data, available with only
the base install; the toolchain is touched lazily by ``xtrax.export.compile``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "ALL_TARGETS",
    "NATIVE",
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
    """

    name: str
    iree_backend: str
    verification_level: VerificationLevel
    supported_dtypes: frozenset[str]
    optional_dtypes: frozenset[str] = frozenset()
    optional_dtype_features: Mapping[str, str] = field(default_factory=dict)
    extra_compiler_flags: tuple[str, ...] = ()


# Native builds target the machine doing the compiling. That is optimal here and
# correct: this artifact is a parity oracle, not something distributed.
NATIVE = Target(
    name="native",
    iree_backend="llvm-cpu",
    verification_level=VerificationLevel.EXECUTED,
    supported_dtypes=frozenset({"f32", "f64", "i32", "i64", "bf16", "f16", "bool"}),
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
    supported_dtypes=frozenset({"f32", "f64", "i32", "i64", "bf16", "f16", "bool"}),
    extra_compiler_flags=(
        "--iree-llvmcpu-target-triple=wasm32-unknown-emscripten",
        "--iree-llvmcpu-target-cpu=generic",
        "--iree-llvmcpu-target-cpu-features=+simd128,+atomics,+bulk-memory",
    ),
)

ALL_TARGETS: tuple[Target, ...] = (NATIVE, WASM32)


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
