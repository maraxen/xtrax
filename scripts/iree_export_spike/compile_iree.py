"""StableHLO -> IREE vmfb, for a native target and a wasm32 target.

Two targets, verified to different depths on purpose:

- ``native`` (llvm-cpu, host triple): compiled AND executed, so numerics can be
  checked against JAX.
- ``wasm32`` (llvm-cpu, wasm32-unknown-emscripten): compiled only. Executing it
  needs an emsdk-built IREE runtime -- IREE's browser runtime lives in
  ``experimental/web`` with no published npm package -- so the spike verifies
  codegen, not execution. Claiming otherwise would be dishonest.

Version skew note: StableHLO's compatibility guarantees apply to *portable
artifacts* serialized at a target version. ``Exported.mlir_module()`` emits
current-version text, so if IREE rejects it we retry through a portable artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

NATIVE_TARGET = "native"
WASM32_TARGET = "wasm32"

# +simd128 is the meaningful perf lever for wasm CPU codegen; atomics/bulk-memory
# are required for the threaded runtime variants. target-cpu must be set explicitly:
# left unset, IREE warns "while creating CPU target" and falls back to a generic CPU
# whose generated code it documents as having "poor performance".
WASM32_FLAGS = (
    "--iree-llvmcpu-target-triple=wasm32-unknown-emscripten",
    "--iree-llvmcpu-target-cpu=generic",
    "--iree-llvmcpu-target-cpu-features=+simd128,+atomics,+bulk-memory",
)

# Native builds target the machine doing the compiling -- optimal here, and this is a
# parity oracle, not a distributable artifact.
NATIVE_FLAGS = ("--iree-llvmcpu-target-cpu=host",)


class IREECompileError(Exception):
    """Raised when iree-compile is unavailable or rejects the input."""


@dataclass(frozen=True)
class CompileResult:
    """Outcome of one iree-compile invocation."""

    target: str
    path: Path
    size_bytes: int
    downgraded_stablehlo: bool


def _require_compiler() -> Any:
    try:
        from iree.compiler import tools as iree_tools
    except ImportError as exc:
        raise IREECompileError(
            "iree-base-compiler is not installed: uv sync --group export-spike"
        ) from exc
    return iree_tools


def _downgrade_to_portable(mlir_text: str) -> bytes:
    """Re-serialize as a version-pinned StableHLO portable artifact.

    Only called as a fallback: if IREE's bundled StableHLO is older than the one
    jax emits, the current-version text can carry ops it cannot parse.
    """
    try:
        from iree.compiler.dialects import stablehlo as iree_stablehlo
    except ImportError as exc:
        raise IREECompileError(
            "IREE rejected the StableHLO and its Python stablehlo bindings are "
            "unavailable, so it cannot be downgraded. Pin a matching jax/IREE pair."
        ) from exc

    version = iree_stablehlo.get_minimum_version()
    return iree_stablehlo.serialize_portable_artifact(mlir_text, version)


def compile_stablehlo(
    mlir_text: str,
    out_path: Path,
    *,
    target: str = NATIVE_TARGET,
) -> CompileResult:
    """Compile StableHLO text to a vmfb at ``out_path``.

    Args:
        mlir_text: StableHLO MLIR, e.g. from ``Exported.mlir_module()``.
        out_path: Destination for the vmfb.
        target: ``"native"`` or ``"wasm32"``.

    Raises:
        IREECompileError: Compiler missing, unknown target, or compilation failed
            even after a portable-artifact downgrade.
    """
    tools = _require_compiler()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if target == NATIVE_TARGET:
        extra_args: tuple[str, ...] = NATIVE_FLAGS
    elif target == WASM32_TARGET:
        extra_args = WASM32_FLAGS
    else:
        raise IREECompileError(
            f"unknown target {target!r}; expected {NATIVE_TARGET!r} or {WASM32_TARGET!r}"
        )

    args = ["--iree-hal-target-backends=llvm-cpu", *extra_args]

    downgraded = False
    try:
        binary = tools.compile_str(mlir_text, input_type="stablehlo", extra_args=args)
    except Exception as first_exc:
        # Retry once through a version-pinned portable artifact before giving up.
        try:
            portable = _downgrade_to_portable(mlir_text)
            binary = tools.compile_str(portable, input_type="stablehlo", extra_args=args)
            downgraded = True
        except IREECompileError:
            raise
        except Exception as retry_exc:
            raise IREECompileError(
                f"iree-compile failed for target {target!r}.\n"
                f"  direct:   {first_exc}\n"
                f"  portable: {retry_exc}"
            ) from first_exc

    out_path.write_bytes(binary)
    return CompileResult(
        target=target,
        path=out_path,
        size_bytes=len(binary),
        downgraded_stablehlo=downgraded,
    )


def run_native_vmfb(vmfb_path: Path, *args: Any, function: str = "main") -> Any:
    """Execute a native vmfb through the IREE runtime and return its output.

    Native only. A wasm32 vmfb cannot be executed here.

    The VM module's name is derived from the traced function (jax names it
    ``jit_<fn>``), so it is resolved from the module rather than hardcoded.
    """
    try:
        import iree.runtime as ireert
    except ImportError as exc:
        raise IREECompileError(
            "iree-base-runtime is not installed: uv sync --group export-spike"
        ) from exc

    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.mmap(ctx.instance, str(vmfb_path))
    ctx.add_vm_module(vm_module)

    loaded = ctx.modules[vm_module.name]
    try:
        entry = loaded[function]
    except (KeyError, AttributeError) as exc:
        raise IREECompileError(
            f"vmfb {vmfb_path.name} (module {vm_module.name!r}) has no entry point {function!r}."
        ) from exc
    return entry(*args)


__all__ = [
    "NATIVE_FLAGS",
    "NATIVE_TARGET",
    "WASM32_FLAGS",
    "WASM32_TARGET",
    "CompileResult",
    "IREECompileError",
    "compile_stablehlo",
    "run_native_vmfb",
]
