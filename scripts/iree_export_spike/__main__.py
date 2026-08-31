"""Driver: HF weights -> xtrax plan -> composer -> StableHLO -> IREE -> parity.

NOT a CI gate and NOT a CLI verb. Run it directly:

    uv run --group export-spike python -m scripts.iree_export_spike --random
    uv run --group export-spike python -m scripts.iree_export_spike --model <hf-id>

Every stage prints a line so a failure is attributable to a stage, not to the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from scripts.iree_export_spike.compile_iree import (
    NATIVE_TARGET,
    WASM32_TARGET,
    IREECompileError,
    compile_stablehlo,
    run_native_vmfb,
)
from scripts.iree_export_spike.composer import compose_exportable
from scripts.iree_export_spike.export_safety import find_bcoo_leaves
from scripts.iree_export_spike.hf_weights import mlp_from_hf, random_mlp
from scripts.iree_export_spike.parity import compare
from xtrax.stages.boundaries import AxisBoundary
from xtrax.tiling.plan import AxisSpec, BatchPlanner

logger = logging.getLogger("iree_export_spike")

IN_DIM = 8
HIDDEN = 16
OUT_DIM = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iree_export_spike",
        description="Feasibility spike: xtrax pipeline -> IREE -> wasm32.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="HuggingFace repo id to pull weights from.")
    src.add_argument(
        "--random",
        action="store_true",
        help="Use deterministic random weights instead of HuggingFace.",
    )
    p.add_argument(
        "--filename",
        default="model.safetensors",
        help="File within the HF repo (default: model.safetensors).",
    )
    p.add_argument("--cardinality", type=int, default=32, help="Axis cardinality.")
    p.add_argument("--batch-size", type=int, default=8, help="Axis batch size.")
    p.add_argument(
        "--fuse",
        action="store_true",
        help="Attach a mean-reducing Fuse to the axis (stays inside the boundary).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".xtrax/iree_spike"),
        help="Where to write .mlir and .vmfb artifacts.",
    )
    p.add_argument("--atol", type=float, default=1e-5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # --- 1. weights -------------------------------------------------------
    if args.random:
        model = random_mlp(IN_DIM, HIDDEN, OUT_DIM)
        logger.info("[1/7] weights: deterministic random (no HF fetch)")
    else:
        model, report = mlp_from_hf(
            args.model,
            filename=args.filename,
            in_dim=IN_DIM,
            hidden=HIDDEN,
            out_dim=OUT_DIM,
        )
        logger.info(
            "[1/7] weights: %s (%d tensors seen, %d used)%s",
            report.source,
            report.tensors_seen,
            report.tensors_used,
            f" casts={report.dtypes_cast}" if report.dtypes_cast else "",
        )

    bcoo = find_bcoo_leaves(model)
    if bcoo:
        logger.info("      BCOO leaves baked as constants: %s", bcoo)

    # --- 2. plan ----------------------------------------------------------
    spec = AxisSpec(
        name="batch",
        cardinality=args.cardinality,
        default_batch_size=args.batch_size,
    )
    plan = BatchPlanner().plan([spec])
    decision = plan.decisions[0]
    logger.info(
        "[2/7] plan: axis 'batch' -> %s (%s)",
        type(decision.strategy).__name__,
        decision.reasoning,
    )

    boundaries = {}
    if args.fuse:
        boundaries = {"batch": AxisBoundary(fuse=lambda ys: jnp.mean(ys, axis=0))}
        logger.info("      fuse attached (mean over axis 0)")

    # --- 3. compose (gate runs inside) ------------------------------------
    # The model is closed over, so jax.export bakes its weights into the
    # StableHLO as constants -- the self-contained artifact we want.
    forward = compose_exportable(model, plan, boundaries)
    logger.info("[3/7] export-safety gate: PASS; composed single-axis callable")

    # --- 4. export --------------------------------------------------------
    aval = jax.ShapeDtypeStruct((args.cardinality, IN_DIM), jnp.float32)
    exported = jax.export.export(jax.jit(forward))(aval)
    mlir_text = exported.mlir_module()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = args.out_dir / "model.mlir"
    mlir_path.write_text(mlir_text)
    logger.info(
        "[4/7] StableHLO: %s (%d bytes), out_avals=%s",
        mlir_path,
        len(mlir_text),
        [str(a) for a in exported.out_avals],
    )

    xs = jnp.linspace(-1.0, 1.0, args.cardinality * IN_DIM, dtype=jnp.float32).reshape(
        args.cardinality, IN_DIM
    )
    reference = jax.jit(forward)(xs)

    # --- 5. native compile + parity ---------------------------------------
    try:
        native = compile_stablehlo(
            mlir_text, args.out_dir / "model_native.vmfb", target=NATIVE_TARGET
        )
    except IREECompileError as exc:
        logger.error("[5/7] native compile FAILED: %s", exc)
        return 1
    logger.info(
        "[5/7] native vmfb: %s (%d bytes)%s",
        native.path,
        native.size_bytes,
        " [stablehlo downgraded]" if native.downgraded_stablehlo else "",
    )

    try:
        actual = run_native_vmfb(native.path, xs)
    except IREECompileError as exc:
        logger.error("[6/7] native execution FAILED: %s", exc)
        return 1
    result = compare(reference, actual, atol=args.atol, rtol=args.atol)
    logger.info("[6/7] parity vs JAX: %s", result.summary())
    if not result.passed:
        return 1

    # --- 7. wasm32 codegen (compile only, by design) ----------------------
    try:
        wasm = compile_stablehlo(
            mlir_text, args.out_dir / "model_wasm32.vmfb", target=WASM32_TARGET
        )
    except IREECompileError as exc:
        logger.error("[7/7] wasm32 compile FAILED: %s", exc)
        return 1
    if wasm.size_bytes <= 0:
        logger.error("[7/7] wasm32 vmfb is empty")
        return 1
    logger.info(
        "[7/7] wasm32 vmfb: %s (%d bytes)%s -- compiled, NOT executed "
        "(needs an emsdk-built IREE runtime)",
        wasm.path,
        wasm.size_bytes,
        " [stablehlo downgraded]" if wasm.downgraded_stablehlo else "",
    )

    logger.info("SPIKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
