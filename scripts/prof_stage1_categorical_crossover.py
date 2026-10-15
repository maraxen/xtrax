#!/usr/bin/env python3
"""Stage-1 CPU crossover probe: matmul vs scatter for exact co-occurrence counts.

Task 1.6 of .praxia/docs/specs/261015_exact-categorical-cooccurrence.md (sections 4.4 and 9).
The hypothesis, grid and outcome criteria are pre-registered in
scripts/prof_stage1_categorical_crossover.bth.toml; this script must not change them.

Each grid cell (dense layout, P variables per side, K categories, T frames, independent
uniform codes for A and B) is measured in its own subprocess with its own timeout. Both
methods of xtrax.categorical.cooccurrence_counts are jitted, warmed up, and timed over
several trials (median reported). Counts must be bit-identical int32 across the two
methods in every cell (parity). Completed cells are persisted atomically under --work-dir
with a sha256 of their config and of the implementation files; a rerun reuses a cell only
when both still match, and the manifest records which cells were reused.

Controls (synthetic timings, always run first) check that select_threshold recovers a
clean crossover and rejects an alternating pattern. The result JSON carries exactly the
pre-registered fields and is written to --out and to $BTH_RESULTS_PATH when set.

Usage (smoke; caller caps threads):
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run --no-sync python \
        scripts/prof_stage1_categorical_crossover.py --smoke --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
LOGGER = logging.getLogger("prof_stage1_categorical_crossover")

PROBE_ID = "stage1_categorical_crossover"
FULL_P: tuple[int, ...] = (80, 400)
FULL_K: tuple[int, ...] = (2, 4, 8, 12, 16, 32)
FULL_T: tuple[int, ...] = (10_000, 100_000)
SMOKE_P: tuple[int, ...] = (8,)
SMOKE_K: tuple[int, ...] = (2, 4)
SMOKE_T: tuple[int, ...] = (1_000,)
METHODS: tuple[str, ...] = ("matmul", "scatter")
SEED = 20261015
DEFAULT_TIMEOUT_S = 300.0
DEFAULT_TRIALS = 5
DEFAULT_WARMUP = 2
MIN_TRIALS = 5
IMPL_FILES: tuple[str, ...] = (
    "src/xtrax/categorical/counts.py",
    "src/xtrax/categorical/codes.py",
    "src/xtrax/categorical/plan.py",
)


# --- hashing and atomic persistence -------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(canonical.encode())


def implementation_sha256() -> str:
    parts = []
    for rel in IMPL_FILES:
        path = ROOT / rel
        digest = sha256_file(path) if path.is_file() else "missing"
        parts.append(f"{rel}:{digest}")
    return sha256_bytes("\n".join(parts).encode())


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a per-writer temp file and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# --- selection routine (pure) -------------------------------------------------------


def select_threshold(
    times_matmul: list[float] | tuple[float, ...],
    times_scatter: list[float] | tuple[float, ...],
    kk: list[int] | tuple[int, ...],
) -> tuple[int, float, float]:
    """Pick the single threshold c on Ka*Kb that best matches the faster method.

    For a candidate c, a cell chooses matmul when ``Ka*Kb <= c`` and scatter otherwise.
    The choice is correct when its time equals the faster of the two times. Candidates
    are the distinct ``kk`` values; the best candidate maximises the fraction correct,
    then minimises the worst slowdown (chosen/faster), then the smaller c.

    Args:
        times_matmul: Matmul time per cell (seconds).
        times_scatter: Scatter time per cell (seconds), same order.
        kk: Ka*Kb per cell, same order.

    Returns:
        ``(c_best, frac_correct, max_slowdown)`` where ``max_slowdown`` is the worst
        chosen/faster ratio over all cells at ``c_best``.

    Raises:
        ValueError: On empty or mismatched inputs, or non-positive timings.
    """
    tm = [float(x) for x in times_matmul]
    ts = [float(x) for x in times_scatter]
    kks = [int(x) for x in kk]
    if not kks or not (len(tm) == len(ts) == len(kks)):
        raise ValueError("select_threshold needs equal-length, non-empty inputs")
    if any(x <= 0.0 for x in tm + ts):
        raise ValueError("select_threshold timings must be positive")

    faster = [min(m, s) for m, s in zip(tm, ts, strict=True)]
    best_key: tuple[float, float, int] | None = None
    best_row: tuple[int, float, float] | None = None
    for c in sorted(set(kks)):
        chosen = [m if k <= c else s for m, s, k in zip(tm, ts, kks, strict=True)]
        correct = sum(1 for ch, fb in zip(chosen, faster, strict=True) if ch <= fb)
        frac = correct / len(kks)
        worst = max(ch / fb for ch, fb in zip(chosen, faster, strict=True))
        key = (-frac, worst, c)
        if best_key is None or key < best_key:
            best_key = key
            best_row = (c, frac, worst)
    assert best_row is not None
    return best_row


# --- controls (synthetic timings, always run) --------------------------------------


def build_grid(smoke: bool) -> list[tuple[int, int, int]]:
    ps, ks, ts = (SMOKE_P, SMOKE_K, SMOKE_T) if smoke else (FULL_P, FULL_K, FULL_T)
    return [(p, k, t) for p in ps for k in ks for t in ts]


def run_controls() -> dict[str, Any]:
    """Positive: one clean crossover must be recovered. Negative: alternation must fail."""
    cells = build_grid(smoke=False)
    kks = [k * k for _, k, _ in cells]
    ks = [k for _, k, _ in cells]

    pos_m = [1.0 if kk <= 64 else 2.0 for kk in kks]
    pos_s = [2.0 if kk <= 64 else 1.0 for kk in kks]
    pos_c, pos_frac, pos_slow = select_threshold(pos_m, pos_s, kks)
    positive_ok = pos_c == 64 and pos_frac == 1.0 and pos_slow == 1.0

    k_rank = {k: i for i, k in enumerate(FULL_K)}
    neg_m: list[float] = []
    neg_s: list[float] = []
    for k in ks:
        if k_rank[k] % 2 == 0:
            neg_m.append(1.0)
            neg_s.append(2.0)
        else:
            neg_m.append(2.0)
            neg_s.append(1.0)
    neg_c, neg_frac, neg_slow = select_threshold(neg_m, neg_s, kks)
    negative_ok = neg_frac < 0.9

    return {
        "positive": {"c_best": pos_c, "frac_correct": pos_frac, "max_slowdown": pos_slow},
        "positive_ok": positive_ok,
        "negative": {"c_best": neg_c, "frac_correct": neg_frac, "max_slowdown": neg_slow},
        "negative_ok": negative_ok,
        "controls_ok": positive_ok and negative_ok,
    }


# --- per-cell child: measure one cell ------------------------------------------------


def measure_cell(config: dict[str, Any], out_path: Path) -> int:
    """Time both methods on one cell, check parity, and persist the cell atomically."""
    import jax
    import numpy as np

    from xtrax.categorical import cooccurrence_counts, encode_codes

    p, k, t = int(config["P"]), int(config["K"]), int(config["T"])
    trials, warmup = int(config["trials"]), int(config["warmup"])
    rng = np.random.default_rng(int(config["seed"]))
    codes_a = encode_codes(rng.integers(0, k, size=(p, t), dtype=np.int32), k)
    codes_b = encode_codes(rng.integers(0, k, size=(p, t), dtype=np.int32), k)

    def compile_method(method: str):
        def run(xa, xb):
            return cooccurrence_counts(xa, xb, method=method).joint

        return jax.jit(run).lower(codes_a, codes_b).compile()

    times: dict[str, list[float]] = {}
    outputs: dict[str, np.ndarray] = {}
    for method in METHODS:
        fn = compile_method(method)
        for _ in range(warmup):
            fn(codes_a, codes_b).block_until_ready()
        samples: list[float] = []
        out = None
        for _ in range(trials):
            t0 = time.perf_counter()
            out = fn(codes_a, codes_b)
            out.block_until_ready()
            samples.append(time.perf_counter() - t0)
        times[method] = samples
        outputs[method] = np.asarray(out)
        LOGGER.info("%s %s: median %.6f s", config["cell_id"], method, statistics.median(samples))

    mm, ms = outputs["matmul"], outputs["scatter"]
    expected_shape = (1, p, p, k, k)
    parity_ok = bool(
        mm.dtype == np.int32
        and ms.dtype == np.int32
        and mm.shape == expected_shape
        and ms.shape == expected_shape
        and np.array_equal(mm, ms)
    )
    mismatch = int(np.count_nonzero(mm != ms)) if mm.shape == ms.shape else -1

    payload = {
        "cell_id": config["cell_id"],
        "config": config,
        "config_sha256": config_sha256(config),
        "backend": jax.default_backend(),
        "times_matmul_s": times["matmul"],
        "times_scatter_s": times["scatter"],
        "median_matmul_s": statistics.median(times["matmul"]),
        "median_scatter_s": statistics.median(times["scatter"]),
        "parity_ok": parity_ok,
        "parity_mismatch_count": mismatch,
        "output_shape": list(mm.shape),
        "output_dtype": str(mm.dtype),
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(out_path, payload)
    LOGGER.info("%s parity_ok=%s", config["cell_id"], parity_ok)
    return 0


# --- parent: orchestrate cells with resume and per-cell timeouts ---------------------


def build_cell_config(
    p: int,
    k: int,
    t: int,
    *,
    trials: int,
    warmup: int,
    jax_version: str,
    impl_sha: str,
    script_sha: str,
) -> dict[str, Any]:
    return {
        "cell_id": f"dense_P{p}_K{k}_T{t}",
        "layout": "dense",
        "P": p,
        "K": k,
        "T": t,
        "methods": list(METHODS),
        "trials": trials,
        "warmup": warmup,
        "seed": SEED,
        "jax_version": jax_version,
        "impl_sha256": impl_sha,
        "script_sha256": script_sha,
    }


def try_reuse(cell_path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    """Return a persisted cell only if its config hash and medians still verify."""
    if not cell_path.is_file():
        return None
    try:
        data = json.loads(cell_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("unreadable cell file %s (%s); re-measuring", cell_path, exc)
        return None
    if data.get("config") != config or data.get("config_sha256") != config_sha256(config):
        LOGGER.info("stale cell file %s (config or implementation changed)", cell_path.name)
        return None
    try:
        ok = (
            statistics.median(data["times_matmul_s"]) == data["median_matmul_s"]
            and statistics.median(data["times_scatter_s"]) == data["median_scatter_s"]
        )
    except (KeyError, TypeError, statistics.StatisticsError):
        ok = False
    if not ok:
        LOGGER.warning("inconsistent cell file %s; re-measuring", cell_path.name)
        return None
    return data


def run_cell_subprocess(
    config: dict[str, Any], cell_path: Path, log_path: Path, timeout_s: float
) -> tuple[str, str]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--run-cell-config",
        json.dumps(config, sort_keys=True),
        "--run-cell-out",
        str(cell_path),
    ]
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "timeout", f"exceeded {timeout_s:g} s"
    if proc.returncode != 0:
        return "failed", f"exit {proc.returncode}"
    if not cell_path.is_file():
        return "failed", "no cell file written"
    return "measured", ""


def _manifest_entry(
    config: dict[str, Any], status: str, cell_path: Path, detail: str = ""
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "cell_id": config["cell_id"],
        "config_sha256": config_sha256(config),
        "status": status,
        "detail": detail,
    }
    if cell_path.is_file() and status in ("measured", "reused"):
        entry["cell_file"] = str(cell_path)
        entry["cell_file_sha256"] = sha256_file(cell_path)
    return entry


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    grid = build_grid(args.smoke)
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    impl_sha = implementation_sha256()
    jax_version = package_version("jax")
    script_sha = sha256_file(SCRIPT_PATH)

    entries: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    for p, k, t in grid:
        config = build_cell_config(
            p,
            k,
            t,
            trials=args.trials,
            warmup=args.warmup,
            jax_version=jax_version,
            impl_sha=impl_sha,
            script_sha=script_sha,
        )
        cell_path = work_dir / f"{config['cell_id']}.json"
        log_path = work_dir / f"{config['cell_id']}.log"
        data = try_reuse(cell_path, config)
        if data is not None:
            LOGGER.info("reusing verified cell %s", config["cell_id"])
            status = "reused"
        else:
            if cell_path.exists():
                cell_path.unlink()
            LOGGER.info("measuring cell %s (timeout %g s)", config["cell_id"], args.timeout)
            status, detail = run_cell_subprocess(config, cell_path, log_path, args.timeout)
            if status != "measured":
                LOGGER.error("cell %s %s: %s", config["cell_id"], status, detail)
                entries.append(_manifest_entry(config, status, cell_path, detail))
                continue
            data = json.loads(cell_path.read_text())
        entries.append(_manifest_entry(config, status, cell_path))
        measured.append(
            {
                "cell_id": config["cell_id"],
                "kk": k * k,
                "median_matmul_s": float(data["median_matmul_s"]),
                "median_scatter_s": float(data["median_scatter_s"]),
                "parity_ok": bool(data["parity_ok"]),
                "backend": str(data.get("backend", "unknown")),
                "status": status,
            }
        )
    return {
        "grid": [list(cell) for cell in grid],
        "entries": entries,
        "measured": measured,
        "impl_sha256": impl_sha,
        "script_sha256": script_sha,
        "jax_version": jax_version,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=None, help="Result JSON path.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs" / "profiling" / "stage1" / "crossover_cells",
        help="Per-cell JSON, logs and manifest are persisted here.",
    )
    parser.add_argument(
        "--probe-out",
        type=Path,
        default=None,
        help="If set, also write an xtrax ProbeRecord (stage 1) to this path.",
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Per-cell seconds."
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS, help="Timed trials per method."
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--smoke", action="store_true", help="Tiny grid; result is not citable.")
    parser.add_argument("--run-cell-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-cell-out", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run_cell_config is not None:
        if args.run_cell_out is None:
            parser.error("--run-cell-config requires --run-cell-out")
        return measure_cell(json.loads(args.run_cell_config), args.run_cell_out)

    if args.trials < MIN_TRIALS:
        parser.error(f"--trials must be >= {MIN_TRIALS} (median of trials)")

    controls = run_controls()
    LOGGER.info("controls: %s", json.dumps(controls, sort_keys=True))
    if not controls["controls_ok"]:
        LOGGER.error("controls failed; result will be instrument_invalid")

    run = run_grid(args)
    measured = run["measured"]
    grid_size = len(run["grid"])
    n_cells = len(measured)

    parity_ok = bool(n_cells > 0 and n_cells == grid_size and all(m["parity_ok"] for m in measured))
    if n_cells:
        c_best, frac, slow = select_threshold(
            [m["median_matmul_s"] for m in measured],
            [m["median_scatter_s"] for m in measured],
            [m["kk"] for m in measured],
        )
    else:
        LOGGER.error("no cells measured")
        c_best, frac, slow = 0, 0.0, 0.0

    result: dict[str, Any] = {
        "n_cells": int(n_cells),
        "smoke": bool(args.smoke),
        "controls_ok": bool(controls["controls_ok"]),
        "parity_ok": parity_ok,
        "c_best": int(c_best),
        "frac_correct": float(frac),
        "max_slowdown": float(slow),
    }

    targets: list[Path] = []
    if args.out is not None:
        targets.append(args.out)
    bth_path = os.environ.get("BTH_RESULTS_PATH")
    if bth_path:
        targets.append(Path(bth_path))
    if not targets:
        LOGGER.warning("no --out and no BTH_RESULTS_PATH; result not persisted")
    for target in targets:
        atomic_write_json(target, result)
        LOGGER.info("wrote result to %s", target)

    reused = sum(1 for e in run["entries"] if e["status"] == "reused")
    manifest = {
        "probe_id": PROBE_ID,
        "result": result,
        "controls": controls,
        "grid": run["grid"],
        "grid_size": grid_size,
        "reused_cells": reused,
        "entries": run["entries"],
        "impl_sha256": run["impl_sha256"],
        "script_sha256": run["script_sha256"],
        "jax_version": run["jax_version"],
        "smoke": bool(args.smoke),
        "written_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(args.work_dir / "crossover_manifest.json", manifest)

    if args.probe_out is not None and measured:
        from xtrax.profiling.emitters import emit_probe_record

        emit_probe_record(
            path=args.probe_out,
            probe_id=PROBE_ID,
            stage=1,
            n_atoms=max(p for p, _, _ in run["grid"]),
            platform=measured[0]["backend"],
            metrics={
                "n_cells": int(n_cells),
                "grid_cells": int(grid_size),
                "reused_cells": int(reused),
                "c_best": int(c_best),
                "frac_correct": float(frac),
                "max_slowdown": float(slow),
                "controls_ok": int(controls["controls_ok"]),
                "parity_ok": int(parity_ok),
                "smoke": int(args.smoke),
            },
            config={
                "layout": "dense",
                "comparison": "matmul_vs_scatter",
                "grid": json.dumps(run["grid"]),
                "statistic": f"median_of_{args.trials}_trials",
                "axis_note": "n_atoms == max P in grid",
            },
        )
        LOGGER.info("wrote probe record %s", args.probe_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
