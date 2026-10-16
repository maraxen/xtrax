#!/usr/bin/env python3
"""Calibrate the dispatch-ratio threshold of ``xtrax profile`` on real entrypoints.

Spec 261016_jax-smoke-profiling-scope, item X5 (calibration half). The
provisional rule is ``executions_per_unit / cardinality < 0.5`` ("batched");
``>= 0.5`` is a host loop. This script measures a fixed, pre-registered list
of entrypoints with :func:`xtrax.profiling.unit_probe.probe_unit` and checks
the labels in the sidecar ``prof_calibrate_dispatch_ratio.bth.toml``.

The ENTRYPOINTS table below is the pre-registration: every label was written
down, and committed with the sidecar, before this script was run for numbers.
Labels and the threshold are not changed after seeing numbers.

Entrypoint families:

- ``controls.host_loop`` (host_loop) and ``controls.vmapped_twin`` (batched)
  at B in {8, 64, 256}.
- The tiling strategies of ``scripts/prof_stage1_tiling_micro.py`` (Vmap,
  ChunkedMap with batch sizes 8 and 32, DedupGather), each its own unit at
  rows in {64, 256} (batched).
- A ``lax.scan`` over the batch inside one jit (batched), at B in {64, 256}.
- A Python loop over chunks of size C calling a jitted chunk kernel
  (labelled batched: a chunk loop should sit far under 0.5 because each
  chunk is one execution, about 1/C per element).
- Optional: one call of the aminx sampler (``make_sample_sequences``) on a
  reduced random-weight Aminx, run in aminx's own venv in a subprocess with
  ``PYTHONPATH`` pointing at this worktree's ``src``. Its failure is recorded
  as ``skipped`` and does not count toward ``n_failed``.

CPU timings are relative only. The dispatch ratio is the scale-free quantity
being calibrated.

Usage (through bathos, from the worktree root):
    bth run --project-slug xtrax -- uv run --no-sync python3 \\
        scripts/experiments/prof_calibrate_dispatch_ratio.py \\
        --out-dir outputs/profiling/calibration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
THRESHOLD = 0.5
HOST_LOOP = "host_loop"
BATCHED = "batched"
N_STEADY = 3
N_WALL = 3
AMINX_PYTHON = Path.home() / "projects" / "aminx" / ".venv" / "bin" / "python"
AMINX_LENGTH = 32
RESULT_FILENAME = "dispatch_ratio_calibration.json"

log = logging.getLogger("prof_calibrate_dispatch_ratio")


@dataclass(frozen=True)
class Entry:
    """One pre-registered entrypoint.

    Attributes:
        name: Stable identifier used in the output table.
        expected: Pre-registered label, ``HOST_LOOP`` or ``BATCHED``.
        family: Grouping used in the output table.
        cardinality: Scale-axis extent (elements per unit).
        make: Zero-argument factory returning ``(fn, args)``. Built lazily so
            a failure in one entrypoint does not stop the others.
    """

    name: str
    expected: str
    family: str
    cardinality: int
    make: Callable[[], tuple[Callable[..., Any], tuple[Any, ...]]]


def _build_xtrax_entries() -> list[Entry]:
    """Return the pre-registered xtrax-side entrypoints. Imports jax lazily."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    from xtrax.profiling import controls
    from xtrax.tiling.dispatch import axis_dispatch, make_axis_dispatch
    from xtrax.tiling.strategy import ChunkedMap, DedupGather, Vmap

    _, _, weights = controls._kernels()

    def workload(row, w):
        h = row
        for _ in range(controls.N_LAYERS):
            h = jnp.tanh(h @ w)
        return h

    def inputs(batch: int, width: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.normal(size=(batch, width)), jnp.float32)

    chunk_step = jax.jit(jax.vmap(workload, in_axes=(0, None)))
    scan_unit = jax.jit(
        lambda x: lax.scan(lambda c, r: (c, workload(r, weights)), None, x)[1]
    )

    def chunked_host_loop(chunk: int):
        def fn(x):
            return jnp.concatenate(
                [chunk_step(x[i : i + chunk], weights) for i in range(0, x.shape[0], chunk)]
            )

        return fn

    def tile_core(xs):
        return (jnp.sin(xs) * 2.0).sum(axis=-1)

    def tiling_unit(kind: str, rows: int, batch_size: int = 0):
        if kind == "dedup":
            k = 8
            dedup = DedupGather(
                unique_indices=jnp.arange(k, dtype=jnp.int32),
                index_map=jnp.arange(rows, dtype=jnp.int32) % k,
                k=k,
                k_bucket=k,
                dedup_fn=lambda x, i: x[i],
                gather_fn=lambda y, i: y[i],
            )
            return jax.jit(lambda xs: axis_dispatch(dedup, tile_core, xs))
        strategy = Vmap() if kind == "vmap" else ChunkedMap(batch_size=batch_size)
        it = make_axis_dispatch(strategy)
        return jax.jit(lambda xs: it(tile_core, xs))

    entries: list[Entry] = []

    for batch in (8, 64, 256):
        x = inputs(batch, controls.FEATURES)
        entries.append(
            Entry(
                f"controls.host_loop B={batch}",
                HOST_LOOP,
                "controls",
                batch,
                lambda x=x: (controls.host_loop, (x,)),
            )
        )
        entries.append(
            Entry(
                f"controls.vmapped_twin B={batch}",
                BATCHED,
                "controls",
                batch,
                lambda x=x: (controls.vmapped_twin, (x,)),
            )
        )

    tiling_specs = [
        ("tiling.vmap", "vmap", 0),
        ("tiling.chunked_map bs=8", "chunked", 8),
        ("tiling.chunked_map bs=32", "chunked", 32),
        ("tiling.dedup_gather", "dedup", 0),
    ]
    for rows in (64, 256):
        xs = inputs(rows, 32, seed=2)
        for label, kind, bs in tiling_specs:
            entries.append(
                Entry(
                    f"{label} rows={rows}",
                    BATCHED,
                    "tiling",
                    rows,
                    lambda kind=kind, rows=rows, bs=bs, xs=xs: (
                        tiling_unit(kind, rows, bs),
                        (xs,),
                    ),
                )
            )

    for batch in (64, 256):
        x = inputs(batch, controls.FEATURES)
        entries.append(
            Entry(
                f"scan_over_batch B={batch}",
                BATCHED,
                "scan",
                batch,
                lambda x=x: (scan_unit, (x,)),
            )
        )

    x256 = inputs(256, controls.FEATURES)
    for chunk in (16, 64):
        entries.append(
            Entry(
                f"chunked_host_loop B=256 C={chunk}",
                BATCHED,
                "chunked_host_loop",
                256,
                lambda chunk=chunk, x=x256: (chunked_host_loop(chunk), (x,)),
            )
        )
    return entries


def _measure(entry: Entry) -> dict[str, Any]:
    from xtrax.profiling.unit_probe import probe_unit

    row: dict[str, Any] = {
        "name": entry.name,
        "family": entry.family,
        "expected": entry.expected,
        "cardinality": entry.cardinality,
        "source": "xtrax",
    }
    try:
        fn, args = entry.make()
        result = probe_unit(fn, args, n_steady=N_STEADY, n_wall=N_WALL)
        row.update(
            status="ok",
            executions_per_unit=float(result.executions_per_unit),
            exec_per_element=float(result.executions_per_element(entry.cardinality)),
            steady_compile_count=int(result.steady_compile_count),
            warmup_compile_count=int(result.warmup_compile_count),
            wall_seconds_per_unit=float(result.wall_seconds_per_unit),
            device_seconds_per_unit=float(result.device_seconds_per_unit),
            host_seconds_per_unit=float(result.host_seconds_per_unit),
        )
    except Exception as exc:  # recorded, counted as n_failed
        log.exception("entrypoint %s failed", entry.name)
        row.update(status=f"error: {type(exc).__name__}: {exc}")
    log.info(
        "%-36s expected=%-9s card=%-4s exec/el=%s steady_compiles=%s",
        entry.name,
        entry.expected,
        entry.cardinality,
        row.get("exec_per_element"),
        row.get("steady_compile_count"),
    )
    return row


def _aminx_worker(out_path: Path, worktree: Path) -> int:
    """Run inside aminx's venv: measure the aminx sampler and write JSON."""
    import jax
    import jax.numpy as jnp

    import xtrax

    xtrax_file = Path(xtrax.__file__).resolve()
    if worktree.resolve() not in xtrax_file.parents:
        raise SystemExit(f"xtrax resolved to {xtrax_file}, not under {worktree}")

    from aminx.model import Aminx
    from aminx.sampling import make_sample_sequences

    from xtrax.profiling.unit_probe import probe_unit

    key = jax.random.PRNGKey(0)
    model = Aminx(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=16,
        key=key,
    )
    length = AMINX_LENGTH
    coords = jax.random.normal(key, (length, 37, 3)) * 5.0
    mask = jnp.ones((length,))
    residue_index = jnp.arange(length)
    chain_index = jnp.zeros((length,))
    sample_fn = jax.jit(make_sample_sequences(model, sampling_strategy="temperature"))
    result = probe_unit(
        sample_fn,
        (jax.random.PRNGKey(2), coords, mask, residue_index, chain_index),
        n_steady=N_STEADY,
        n_wall=N_WALL,
    )
    payload = {
        "name": f"aminx.make_sample_sequences L={length}",
        "family": "aminx",
        "expected": BATCHED,
        "cardinality": length,
        "source": "aminx",
        "status": "ok",
        "executions_per_unit": float(result.executions_per_unit),
        "exec_per_element": float(result.executions_per_element(length)),
        "steady_compile_count": int(result.steady_compile_count),
        "warmup_compile_count": int(result.warmup_compile_count),
        "wall_seconds_per_unit": float(result.wall_seconds_per_unit),
        "device_seconds_per_unit": float(result.device_seconds_per_unit),
        "host_seconds_per_unit": float(result.host_seconds_per_unit),
        "xtrax_file": str(xtrax_file),
        "jax_version": jax.__version__,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return 0


def _run_aminx_arm(scratch: Path) -> dict[str, Any]:
    """Run the optional aminx arm in aminx's venv. Never raises."""
    base: dict[str, Any] = {
        "name": f"aminx.make_sample_sequences L={AMINX_LENGTH}",
        "family": "aminx",
        "expected": BATCHED,
        "cardinality": AMINX_LENGTH,
        "source": "aminx",
    }
    if not AMINX_PYTHON.exists():
        return {**base, "status": f"skipped: no aminx venv at {AMINX_PYTHON}"}
    out = scratch / "aminx_arm.json"
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    }
    cmd = [
        str(AMINX_PYTHON),
        str(Path(__file__).resolve()),
        "--aminx-worker",
        "--worker-out",
        str(out),
        "--worktree",
        str(ROOT),
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {**base, "status": "skipped: timeout after 900 s"}
    if proc.returncode != 0 or not out.exists():
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return {**base, "status": f"skipped: worker exit {proc.returncode}: {tail[0][:300]}"}
    return json.loads(out.read_text())


def _summarise(rows: list[dict[str, Any]], xtrax_file: str, jax_version: str) -> dict[str, Any]:
    xt = [r for r in rows if r["source"] == "xtrax"]
    ok = [r for r in rows if r.get("status") == "ok"]
    host = [r for r in ok if r["expected"] == HOST_LOOP]
    bat = [r for r in ok if r["expected"] == BATCHED]
    mislabelled = [r for r in host if r["exec_per_element"] < THRESHOLD] + [
        r for r in bat if r["exec_per_element"] >= THRESHOLD
    ]
    host_eps = [r["exec_per_element"] for r in host]
    bat_eps = [r["exec_per_element"] for r in bat]
    bat_xt = [r["exec_per_element"] for r in bat if r["source"] == "xtrax"]
    min_host = min(host_eps) if host_eps else None
    max_bat = max(bat_eps) if bat_eps else None
    max_bat_xt = max(bat_xt) if bat_xt else None
    margin = min_host / max_bat if (min_host is not None and max_bat) else None
    aminx = next((r for r in rows if r["source"] == "aminx"), None)
    aminx_status = aminx["status"] if aminx else "not run"
    aminx_eps = (
        aminx["exec_per_element"] if aminx and aminx.get("status") == "ok" else None
    )
    return {
        "n_units_xtrax": len(xt),
        "n_failed": sum(1 for r in xt if r.get("status") != "ok"),
        "n_host_loop": len(host),
        "n_batched_xtrax": sum(1 for r in bat if r["source"] == "xtrax"),
        "n_batched": len(bat),
        "n_mislabelled": len(mislabelled),
        "n_steady_nonzero": sum(1 for r in ok if r["steady_compile_count"] > 0),
        "min_host_loop_exec_per_element": min_host,
        "max_batched_xtrax_exec_per_element": max_bat_xt,
        "max_batched_exec_per_element": max_bat,
        "margin": margin,
        "aminx_status": aminx_status,
        "aminx_exec_per_element": aminx_eps,
        "threshold": THRESHOLD,
        "xtrax_file": xtrax_file,
        "jax_version": jax_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--skip-aminx", action="store_true", help="Skip the optional aminx arm.")
    parser.add_argument("--aminx-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worktree", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.aminx_worker:
        return _aminx_worker(args.worker_out, args.worktree)
    if args.out_dir is None:
        parser.error("--out-dir is required")

    import jax

    import xtrax

    xtrax_file = str(Path(xtrax.__file__).resolve())
    if str(SRC.resolve()) not in xtrax_file:
        raise SystemExit(f"xtrax resolved to {xtrax_file}, not under {SRC}")
    log.info("xtrax from %s; jax %s", xtrax_file, jax.__version__)

    rows = [_measure(e) for e in _build_xtrax_entries()]
    with tempfile.TemporaryDirectory() as scratch:
        if args.skip_aminx:
            rows.append(
                {
                    "name": f"aminx.make_sample_sequences L={AMINX_LENGTH}",
                    "family": "aminx",
                    "expected": BATCHED,
                    "cardinality": AMINX_LENGTH,
                    "source": "aminx",
                    "status": "skipped: --skip-aminx",
                }
            )
        else:
            rows.append(_run_aminx_arm(Path(scratch)))

    summary = _summarise(rows, xtrax_file, jax.__version__)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / RESULT_FILENAME
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    log.info("wrote %s", out_path)

    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(summary))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
