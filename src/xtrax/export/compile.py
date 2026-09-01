"""StableHLO -> IREE vmfb, for any registered target.

The spike this was promoted from hardcoded two string targets; here the backend
and flags come off the ``Target`` object, so adding a target is data rather than
a new branch.

Version skew note: StableHLO's compatibility guarantees apply to *portable
artifacts* serialized at a target version. ``Exported.mlir_module()`` emits
current-version text, so if IREE rejects it we retry once through a portable
artifact pinned to IREE's own minimum version.

IREE is imported lazily inside each function. Importing this module requires
only the base install; a missing toolchain surfaces as a CompileError naming the
extra to install, at the point of use.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.export.targets import Target

__all__ = [
    "CompileError",
    "CompileResult",
    "compile_for_target",
    "run_native_vmfb",
]

_MISSING_EXTRA = "install the export toolchain with: pip install xtrax[export]"


class CompileError(Exception):
    """Wraps an IREE compiler failure with target name, backend, and stderr."""


@dataclass(frozen=True)
class CompileResult:
    """Outcome of one iree-compile invocation.

    Attributes:
        target: The target that was compiled for.
        path: The written vmfb. A real file, because the only proven executor
            mmaps it.
        size_bytes: Size of the vmfb.
        spirv_bytes: Extracted SPIR-V keyed by executable name, for targets that
            emit it; None otherwise.
        downgraded_stablehlo: Whether the portable-artifact retry was needed.
        stderr: Compiler diagnostics, empty on a clean first compile.
    """

    target: Target
    path: Path
    size_bytes: int
    spirv_bytes: dict[str, bytes] | None
    downgraded_stablehlo: bool
    stderr: str


def _require_compiler() -> Any:
    """Import iree.compiler.tools, or raise CompileError naming the extra."""
    try:
        from iree.compiler import tools as iree_tools
    except ImportError as exc:
        msg = f"iree-base-compiler is not installed: {_MISSING_EXTRA}"
        raise CompileError(msg) from exc
    return iree_tools


def _downgrade_to_portable(mlir_text: str) -> bytes:
    """Re-serialize as a version-pinned StableHLO portable artifact.

    Only called as a fallback: if IREE's bundled StableHLO is older than the one
    jax emits, the current-version text can carry ops it cannot parse.

    Args:
        mlir_text: Current-version StableHLO MLIR.

    Returns:
        The portable artifact bytes.

    Raises:
        CompileError: If IREE's stablehlo bindings are unavailable.
    """
    try:
        from iree.compiler.dialects import stablehlo as iree_stablehlo
    except ImportError as exc:
        msg = (
            "IREE rejected the StableHLO and its Python stablehlo bindings are "
            "unavailable, so it cannot be downgraded. Pin a matching jax/IREE pair."
        )
        raise CompileError(msg) from exc

    version = iree_stablehlo.get_minimum_version()
    return iree_stablehlo.serialize_portable_artifact(mlir_text, version)


def compile_for_target(
    mlir_text: str,
    target: Target,
    *,
    out_path: Path | None = None,
) -> CompileResult:
    """Compile StableHLO text to a vmfb for ``target``.

    Args:
        mlir_text: StableHLO MLIR, e.g. from ``Exported.mlir_module()``.
        target: The target to compile for; supplies the IREE backend and flags.
        out_path: Destination for the vmfb. A fresh temp file is used if omitted.

    Returns:
        A CompileResult describing the written artifact.

    Raises:
        CompileError: Compiler missing, or compilation failed even after a
            portable-artifact downgrade.
    """
    tools = _require_compiler()

    if out_path is None:
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed immediately below
            suffix=f".{target.name}.vmfb", delete=False
        )
        handle.close()
        out_path = Path(handle.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        f"--iree-hal-target-backends={target.iree_backend}",
        *target.extra_compiler_flags,
    ]

    downgraded = False
    stderr = ""
    try:
        binary = tools.compile_str(mlir_text, input_type="stablehlo", extra_args=args)
    except Exception as first_exc:
        stderr = str(first_exc)
        # Retry once through a version-pinned portable artifact before giving up.
        try:
            portable = _downgrade_to_portable(mlir_text)
            binary = tools.compile_str(portable, input_type="stablehlo", extra_args=args)
            downgraded = True
        except CompileError:
            raise
        except Exception as retry_exc:
            msg = (
                f"iree-compile failed for target {target.name!r} "
                f"(backend {target.iree_backend!r}).\n"
                f"  direct:   {first_exc}\n"
                f"  portable: {retry_exc}"
            )
            raise CompileError(msg) from first_exc

    out_path.write_bytes(binary)
    return CompileResult(
        target=target,
        path=out_path,
        size_bytes=len(binary),
        spirv_bytes=None,
        downgraded_stablehlo=downgraded,
        stderr=stderr,
    )


def run_native_vmfb(vmfb_path: Path, *args: Any, function: str = "main") -> Any:
    """Execute a native vmfb through the IREE runtime and return its output.

    Native only. A wasm32 or SPIR-V vmfb cannot be executed here.

    The VM module's name is derived from the traced function -- jax names it
    ``jit_<fn>`` -- so it is resolved from the module rather than hardcoded.

    Args:
        vmfb_path: Path to a native vmfb.
        *args: Concrete arguments to pass to the entry point.
        function: Entry point name within the module.

    Returns:
        Whatever the entry point returns.

    Raises:
        CompileError: If the runtime is missing, or the module has no such entry
            point.
    """
    try:
        import iree.runtime as ireert
    except ImportError as exc:
        msg = f"iree-base-runtime is not installed: {_MISSING_EXTRA}"
        raise CompileError(msg) from exc

    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.mmap(ctx.instance, str(vmfb_path))
    ctx.add_vm_module(vm_module)

    loaded = ctx.modules[vm_module.name]
    try:
        entry = loaded[function]
    except (KeyError, AttributeError) as exc:
        msg = (
            f"vmfb {vmfb_path.name} (module {vm_module.name!r}) has no entry "
            f"point {function!r}."
        )
        raise CompileError(msg) from exc
    return entry(*args)
