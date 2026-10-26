#!/usr/bin/env python3
"""Run the pairwise-MI forms campaign: one subprocess per cell, then ONE aggregate JSON.

Companion of ``pairwise_mi_forms.py`` (backlog #6099). The cell list comes from that
script's ``plan_cells`` (imported, not copied): forms {A, B, C_triton, D} x entropy
{float64, float32} x shape {full, tree, boundary} x K {11, 129}, default tiles, with
``n_used`` = K for K=11 and 23 for K=129 (``--n-used-map``). Each cell runs as its own
Python process under its own ``--cell-timeout-s``; a cell whose stamp verifies (config
hash, cell-script hash, artifact sha256) is skipped and the reuse logged to
``<out-dir>/reuse_log.jsonl``. A timed-out or crashed cell leaves no artifact; the
timeout is recorded in ``<cell_id>.timeout.json`` and counted, never raised. An fsynced
campaign heartbeat (``campaign.heartbeat.jsonl``) is written between cells.

The aggregate is flat and contains only scalars, written to ``--out`` and, when set,
``$BTH_RESULTS_PATH``. A per-cell table goes to ``<out-dir>/campaign_cells.json``.

Definitions used by the aggregate (the script decides nothing beyond these booleans):

- a cell is ``ok`` when a verified artifact exists with an empty ``error``; ``error`` when
  the artifact has an error or no artifact exists without a timeout marker; ``timeout``
  when only a timeout marker exists;
- ``correct`` (per-form fields): ok and counts_exact and rowsums_ok and float64
  mi_max_abs_diff <= 1e-9 and not mi_all_zero;
- ``a_b_correct``: at least one ok A/B cell, and every ok A/B cell has counts_exact,
  rowsums_ok, not mi_all_zero, mi_nonfinite == 0 and, for float64 cells, diff <= 1e-9;
- ``a_b_full_ok`` and ``boundary_b_ok`` are False when no such cell is planned (not
  available), so a subset run reports them False rather than vacuously True;
- ``fastest_k<k>`` is "" when no correct full-shape float64 cell exists.

Usage:
    uv run --no-sync python scripts/benchmarks/pairwise_mi_forms_campaign.py \\
        --out-dir OUT --out OUT/campaign_result.json

Smoke job (A and B, full shape, K=11, float64 only):
    ... --forms A,B --shapes full --Ks 11 --entropy-dtypes float64
"""

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CELL_SCRIPT = HERE / "pairwise_mi_forms.py"
REPO_ROOT = HERE.parents[1]
log = logging.getLogger("pairwise_mi_forms_campaign")


def _load_cell_module() -> Any:
    spec = importlib.util.spec_from_file_location("pairwise_mi_forms", CELL_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pairwise_mi_forms", mod)
    spec.loader.exec_module(mod)
    return mod


pmf = _load_cell_module()

FORM_KEYS = {"a": "A", "b": "B", "c_triton": "C_triton", "d": "D"}
KS = (11, 129)
DEFAULT_FORMS = "A,B,C_triton,D"
DEFAULT_SHAPES = "full,tree,boundary"
DEFAULT_N_USED_MAP = "11:11,129:23"
F64_DIFF_MAX = 1e-9
F32_DIFF_MAX = 1e-6

FIELDS = (
    ["n_cells_planned", "n_cells_ok", "n_cells_error", "n_cells_timeout", "max_recompiles"]
    + ["a_b_correct", "a_b_full_ok", "boundary_b_ok"]
    + [f"step_{f}_k{k}" for f in FORM_KEYS for k in KS]
    + [f"correct_{f}_k{k}" for f in FORM_KEYS for k in KS]
    + [f"{name}_k{k}" for k in KS for name in
       ("fastest", "best_kernel_step", "ratio_b_over_kernel", "kernel_ok")]
    + [f"f32_entropy_ok_{f}" for f in FORM_KEYS]
    + ["jax_version", "device_kind", "script_sha256", "denxity_commit"]
)  # fmt: skip


@dataclass
class CellRec:
    """One planned cell and what became of it."""

    cell_id: str
    form: str
    entropy_dtype: str
    shape: str
    K: int
    status: str = "error"  # ok | error | timeout
    result: dict[str, Any] | None = None
    note: str = ""
    argv: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- planning


def _parse_n_used_map(text: str) -> dict[int, int]:
    out = {}
    for item in text.split(","):
        if item:
            k, n = item.split(":")
            out[int(k)] = int(n)
    return out


def plan_argvs(args: argparse.Namespace) -> list[list[str]]:
    """Cell argvs from ``pmf.plan_cells``, one call per K so each K gets its own n_used."""
    n_used_map = _parse_n_used_map(args.n_used_map)
    argvs: list[list[str]] = []
    for k in [int(x) for x in args.Ks.split(",") if x]:
        flags = [
            "--plan", "--out-dir", str(args.out_dir), "--forms", args.forms,
            "--shapes", args.shapes, "--entropy-dtypes", args.entropy_dtypes,
            "--Ks", str(k), "--seed", str(args.seed), "--reps", str(args.reps),
            "--warmup", str(args.warmup), "--n-sample", str(args.n_sample),
            "--pair-chunks", str(args.pair_chunk), "--pair-tiles", str(args.pair_tile),
            "--frame-tiles", str(args.frame_tile), "--a-batch-sizes", str(args.a_batch_size),
            "--n-used", str(n_used_map.get(k, k)),
        ]  # fmt: skip
        for opt, val in (("--n-res", args.n_res), ("--n-frames", args.n_frames),
                         ("--n-pairs", args.n_pairs)):  # fmt: skip
            if val is not None:
                flags += [opt, str(val)]
        if args.corrupt:
            flags += ["--corrupt", args.corrupt]
        if args.interpret:
            flags.append("--interpret")
        argvs += pmf.plan_cells(pmf.build_parser().parse_args(flags))
    return argvs


def _cell_of(argv: list[str]) -> Any:
    return pmf.cell_from_args(pmf.build_parser().parse_args(argv))


# --------------------------------------------------------------------------- running


def run_cell_subprocess(argv: list[str], out_dir: Path, cid: str, timeout_s: float) -> str:
    """Run one cell as ``python pairwise_mi_forms.py <argv>``; return ok | timeout | crash."""
    # --force: the campaign already decided this cell must run (e.g. --retry-errors).
    cmd = [sys.executable, str(CELL_SCRIPT), *argv, "--force"]
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    with (out_dir / f"{cid}.log").open("w", encoding="utf-8") as fh:
        try:
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                timeout=timeout_s, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return "timeout"
    return "ok" if proc.returncode == 0 else f"crash:{proc.returncode}"


def _atomic_write(path: Path, text: str) -> None:
    pmf._atomic_write_text(path, text)


def run_campaign(args: argparse.Namespace) -> list[CellRec]:
    """Run (or reuse) every planned cell, then classify each from what is on disk."""
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hb = pmf.Heartbeat(out_dir / "campaign.heartbeat.jsonl")
    recs: list[CellRec] = []
    scr = pmf.script_sha256()
    try:
        argvs = plan_argvs(args)
        hb.beat("campaign_start", n_cells=len(argvs))
        for i, argv in enumerate(argvs):
            cell = _cell_of(argv)
            cid = pmf.cell_id(cell)
            rec = CellRec(cid, cell.form, cell.entropy_dtype, cell.shape, cell.K, argv=argv)
            recs.append(rec)
            reusable = None
            if not args.force:
                reusable = pmf.find_reusable(
                    out_dir, cid, cell.config_sha256(), scr, args.retry_errors
                )
            if reusable is not None and not args.aggregate_only:
                pmf.record_reuse(out_dir, cid, out_dir / f"{cid}.stamp.json")
                hb.beat("cell_reused", i=i, cell_id=cid)
                continue
            if args.aggregate_only:
                continue
            hb.beat("cell_start", i=i, cell_id=cid)
            t0 = time.perf_counter()
            outcome = run_cell_subprocess(argv, out_dir, cid, args.cell_timeout_s)
            marker = out_dir / f"{cid}.timeout.json"
            if outcome == "timeout":
                _atomic_write(
                    marker, json.dumps({"cell_id": cid, "timeout_s": args.cell_timeout_s})
                )
            elif marker.exists():
                marker.unlink()
            hb.beat("cell_done", i=i, cell_id=cid, outcome=outcome,
                    seconds=time.perf_counter() - t0)  # fmt: skip
        for rec in recs:
            _classify(rec, out_dir, args.retry_errors)
        hb.beat("campaign_done", n_cells=len(recs))
    finally:
        hb.close()
    return recs


def _classify(rec: CellRec, out_dir: Path, retry_errors: bool) -> None:
    """Set ``status``/``result``: verified artifact > timeout marker > missing (error)."""
    cell = _cell_of(rec.argv)
    result = pmf.find_reusable(
        out_dir, rec.cell_id, cell.config_sha256(), pmf.script_sha256(), False
    )
    if result is not None:
        rec.result = result
        rec.status = "ok" if not result.get("error") else "error"
        rec.note = result.get("error", "")
    elif (out_dir / f"{rec.cell_id}.timeout.json").exists():
        rec.status, rec.note = "timeout", "no artifact; timeout marker"
    else:
        rec.status, rec.note = "error", "no verified artifact (crash, not run, or stale stamp)"


# --------------------------------------------------------------------------- aggregate


def _ab_correct(r: dict[str, Any]) -> bool:
    ok = r["counts_exact"] and r["rowsums_ok"] and not r["mi_all_zero"] and r["mi_nonfinite"] == 0
    if r["entropy_dtype"] == "float64":
        ok = ok and r["mi_max_abs_diff"] <= F64_DIFF_MAX
    return bool(ok)


def _correct64(r: dict[str, Any]) -> bool:
    return bool(
        r["counts_exact"]
        and r["rowsums_ok"]
        and not r["mi_all_zero"]
        and 0 <= r["mi_max_abs_diff"] <= F64_DIFF_MAX
    )


def aggregate(recs: list[CellRec]) -> dict[str, Any]:
    """Flatten per-cell records into the campaign result (see module docstring)."""
    ok = [r for r in recs if r.status == "ok"]
    out: dict[str, Any] = {
        "n_cells_planned": len(recs),
        "n_cells_ok": len(ok),
        "n_cells_error": sum(r.status == "error" for r in recs),
        "n_cells_timeout": sum(r.status == "timeout" for r in recs),
        "max_recompiles": max((r.result["recompiles"] for r in ok), default=-1),
    }
    ab_ok = [r for r in ok if r.form in ("A", "B")]
    out["a_b_correct"] = bool(ab_ok) and all(_ab_correct(r.result) for r in ab_ok)
    ab_full = [r for r in recs if r.form in ("A", "B") and r.shape == "full"
               and r.entropy_dtype == "float64"]  # fmt: skip
    out["a_b_full_ok"] = bool(ab_full) and all(r.status == "ok" for r in ab_full)
    b_boundary = [r for r in recs if r.form == "B" and r.shape == "boundary"]
    out["boundary_b_ok"] = bool(b_boundary) and all(
        r.status == "ok" and _ab_correct(r.result) for r in b_boundary
    )

    def full64(form: str, k: int) -> CellRec | None:
        hits = [r for r in recs if r.form == form and r.K == k and r.shape == "full"
                and r.entropy_dtype == "float64"]  # fmt: skip
        return hits[0] if hits else None

    steps: dict[tuple[str, int], float] = {}
    for key, form in FORM_KEYS.items():
        for k in KS:
            r = full64(form, k)
            good = r is not None and r.status == "ok" and _correct64(r.result)
            out[f"correct_{key}_k{k}"] = bool(good)
            step = float(r.result["step_median_s"]) if good else -1.0
            steps[(form, k)] = step if step > 0 else -1.0
            out[f"step_{key}_k{k}"] = steps[(form, k)]
    for k in KS:
        positive = {f: s for (f, kk), s in steps.items() if kk == k and s > 0}
        out[f"fastest_k{k}"] = min(positive, key=positive.__getitem__) if positive else ""
        kern = [steps[(f, k)] for f in ("C_triton", "D") if steps[(f, k)] > 0]
        best = min(kern) if kern else -1.0
        out[f"best_kernel_step_k{k}"] = best
        step_b = steps[("B", k)]
        out[f"ratio_b_over_kernel_k{k}"] = step_b / best if best > 0 and step_b > 0 else -1.0
        out[f"kernel_ok_k{k}"] = best > 0
    for key, form in FORM_KEYS.items():
        f32 = [r for r in recs if r.form == form and r.shape == "full"
               and r.entropy_dtype == "float32"]  # fmt: skip
        out[f"f32_entropy_ok_{key}"] = bool(f32) and all(
            r.status == "ok" and 0 <= r.result["mi_max_abs_diff"] <= F32_DIFF_MAX for r in f32
        )
    first = ok[0].result if ok else None
    cell_sha = pmf.script_sha256()
    camp_sha = pmf.sha256_file(Path(__file__).resolve())
    out["jax_version"] = first["jax_version"] if first else pmf.jax.__version__
    out["device_kind"] = first["device_kind"] if first else "unknown"
    out["script_sha256"] = hashlib.sha256((camp_sha + ":" + cell_sha).encode()).hexdigest()
    out["denxity_commit"] = pmf.DENXITY_COMMIT
    out["campaign_script_sha256"] = camp_sha
    out["cell_script_sha256"] = cell_sha
    ordered = {name: out[name] for name in FIELDS}
    ordered.update({k: v for k, v in out.items() if k not in ordered})
    return ordered


def write_outputs(recs: list[CellRec], result: dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(result, indent=2, allow_nan=False)
    out_path = args.out or (args.out_dir / "campaign_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_path, text)
    table = [
        {"cell_id": r.cell_id, "form": r.form, "entropy_dtype": r.entropy_dtype,
         "shape": r.shape, "K": r.K, "status": r.status, "note": r.note[:300],
         "step_median_s": r.result["step_median_s"] if r.result else -1.0}
        for r in recs
    ]  # fmt: skip
    _atomic_write(args.out_dir / "campaign_cells.json", json.dumps(table, indent=2))
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(result, allow_nan=False))


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--out", type=Path, help="aggregate JSON (default <out-dir>/campaign_result.json)"
    )
    p.add_argument("--cell-timeout-s", type=float, default=1800.0)
    p.add_argument("--forms", default=DEFAULT_FORMS)
    p.add_argument("--Ks", default="11,129")
    p.add_argument("--shapes", default=DEFAULT_SHAPES)
    p.add_argument("--entropy-dtypes", default="float64,float32")
    p.add_argument("--n-used-map", default=DEFAULT_N_USED_MAP, help="K:n_used,... (default K)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--n-sample", type=int, default=64)
    p.add_argument("--pair-chunk", type=int, default=2048)
    p.add_argument("--pair-tile", type=int, default=4)
    p.add_argument("--frame-tile", type=int, default=1024)
    p.add_argument("--a-batch-size", type=int, default=4096)
    p.add_argument("--n-res", type=int, help="with --shapes custom (tests)")
    p.add_argument("--n-frames", type=int, help="with --shapes custom (tests)")
    p.add_argument("--n-pairs", type=int, help="with --shapes custom (tests)")
    p.add_argument("--corrupt", choices=pmf.CORRUPTIONS, default="", help="negative control")
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--force", action="store_true", help="re-run even if a stamp verifies")
    p.add_argument("--retry-errors", action="store_true", help="do not reuse errored cells")
    p.add_argument("--aggregate-only", action="store_true", help="classify existing artifacts only")
    p.add_argument("--dry-run", action="store_true", help="print the cell commands and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        for cell_argv in plan_argvs(args):
            print(" ".join([sys.executable, str(CELL_SCRIPT), *cell_argv]))
        return 0
    recs = run_campaign(args)
    result = aggregate(recs)
    write_outputs(recs, result, args)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
