"""Offline CPU-only CUDA Triton lowering verifier using abstract meshes (Debt #3197)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class TritonLoweringError(AssertionError):
    """Raised when a Pallas kernel fails to lower to valid CUDA Triton HLO."""


@dataclass(frozen=True, slots=True)
class TritonLoweringResult:
    """Diagnostic outcome of an offline CUDA Triton lowering verification."""

    hlo_text: str
    grid: tuple[int, int, int]
    num_warps: int | None
    num_stages: int | None
    kernel_name: str | None
    raw_backend_config: str


_GRID_X_RE = re.compile(r"grid_x\s*=\s*(\d+)\s*:\s*i32")
_GRID_Y_RE = re.compile(r"grid_y\s*=\s*(\d+)\s*:\s*i32")
_GRID_Z_RE = re.compile(r"grid_z\s*=\s*(\d+)\s*:\s*i32")
_NUM_WARPS_RE = re.compile(r"num_warps\s*=\s*(\d+)\s*:\s*i32")
_NUM_STAGES_RE = re.compile(r"num_stages\s*=\s*(\d+)\s*:\s*i32")
_KERNEL_NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')
_BACKEND_CONFIG_RE = re.compile(r"mhlo\.backend_config\s*=\s*({[^}]+})")


def verify_triton_lowering(
    kernel_fn: Callable[..., Any],
    *args: Any,
    compiler_params: Any | None = None,
    expected_grid: tuple[int, int, int] | None = None,
    expected_num_warps: int | None = None,
    expected_num_stages: int | None = None,
    device_kind: str = "NVIDIA A100",
    **kwargs: Any,
) -> TritonLoweringResult:
    """Verify that `kernel_fn(*args, **kwargs)` lowers to CUDA Triton on CPU offline.

    Uses an abstract CUDA mesh to drive JAX/XLA lowering without GPU hardware.
    Confirms that StableHLO emits `@__gpu$xla.gpu.triton` and validates grid
    and compiler configurations when specified.

    Args:
        kernel_fn: Pallas kernel callable or function wrapping `pl.pallas_call`.
        *args: Example inputs (arrays or `jax.ShapeDtypeStruct` instances).
        compiler_params: Optional Triton compiler parameters (`plt.CompilerParams`).
        expected_grid: Expected (grid_x, grid_y, grid_z) tuple, if asserted.
        expected_num_warps: Expected num_warps setting, if asserted.
        expected_num_stages: Expected num_stages setting, if asserted.
        device_kind: Abstract device model name (defaults to "NVIDIA A100").
        **kwargs: Additional keyword arguments passed to `kernel_fn`.

    Returns:
        A `TritonLoweringResult` containing the extracted HLO text, grid,
        warps, stages, and kernel metadata.

    Raises:
        TritonLoweringError: If lowering fails, emits Mosaic GPU or generic HLO
            instead of Triton, or mismatches expected grid/warp/stage constraints.
    """
    import jax
    from jax._src import mesh as mesh_lib

    device = mesh_lib.AbstractDevice(device_kind=device_kind, num_cores=None, platform="cuda")
    mesh = mesh_lib.AbstractMesh((1,), ("x",), abstract_device=device)

    try:
        with mesh_lib.use_abstract_mesh(mesh):
            traced = jax.jit(kernel_fn).trace(*args, **kwargs)
            lowered = traced.lower(lowering_platforms=("cuda",))
            hlo_text = lowered.as_text()
    except Exception as exc:
        raise TritonLoweringError(
            f"Failed to lower kernel to Triton on abstract CUDA device: {exc}"
        ) from exc

    if "@__gpu$xla.gpu.triton" not in hlo_text:
        fallback = "mosaic_gpu" if "mosaic_gpu" in hlo_text else "plain StableHLO"
        raise TritonLoweringError(
            f"Kernel did not lower to Triton custom call (@__gpu$xla.gpu.triton) -- "
            f"emitted {fallback}. HLO:\n{hlo_text}"
        )

    gx = _GRID_X_RE.search(hlo_text)
    gy = _GRID_Y_RE.search(hlo_text)
    gz = _GRID_Z_RE.search(hlo_text)
    grid = (
        int(gx.group(1)) if gx else 1,
        int(gy.group(1)) if gy else 1,
        int(gz.group(1)) if gz else 1,
    )

    warps_m = _NUM_WARPS_RE.search(hlo_text)
    num_warps = int(warps_m.group(1)) if warps_m else None

    stages_m = _NUM_STAGES_RE.search(hlo_text)
    num_stages = int(stages_m.group(1)) if stages_m else None

    name_m = _KERNEL_NAME_RE.search(hlo_text)
    kernel_name = name_m.group(1) if name_m else None

    config_m = _BACKEND_CONFIG_RE.search(hlo_text)
    raw_backend_config = config_m.group(1) if config_m else ""

    if expected_grid is not None and grid != expected_grid:
        raise TritonLoweringError(f"Triton grid mismatch: expected {expected_grid}, got {grid}")

    if expected_num_warps is not None and num_warps != expected_num_warps:
        raise TritonLoweringError(
            f"Triton num_warps mismatch: expected {expected_num_warps}, got {num_warps}"
        )

    if expected_num_stages is not None and num_stages != expected_num_stages:
        raise TritonLoweringError(
            f"Triton num_stages mismatch: expected {expected_num_stages}, got {num_stages}"
        )

    return TritonLoweringResult(
        hlo_text=hlo_text,
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        kernel_name=kernel_name,
        raw_backend_config=raw_backend_config,
    )
