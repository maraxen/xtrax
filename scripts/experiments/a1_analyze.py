# ruff: noqa: UP031, E501
"""A1 analysis: torn reads of ``os.replace`` on Engaging scratch, pool and home (spec 261030 section 4).

Reads the raw JSONL logs written by ``a1_replace_atomicity.py`` and computes the pre-registered
cells (3 filesystems x 4 payload sizes). Outcome rules live in ``a1_analyze.bth.toml``.

SUBMISSION (do not run from here; the job produces raw logs only, no finding). From the project
root, with the worktree pushed to Engaging:

    myxcel submit-job engaging xtrax --preset cpu --override time=01:15:00 \\
        --override nodes=3 --override ntasks=3 --override cpus_per_task=2 \\
        --command "bash scripts/experiments/a1_replace_atomicity.sh"

``nodes`` and ``ntasks`` are in myxcel's override table (``myxcel/slurm.py``
``_OVERRIDE_FIELD_TYPES``); ``--ntasks-per-node=1`` is set on the ``srun`` inside the wrapper. The
job writes logs to ``outputs/probes/a1/raw_<jobid>/`` (about 47 min of phases plus 1 min lead).
Then pull that directory and run the analysis locally, under bathos:

    bth run --project-slug xtrax --output-paths outputs/probes/a1/a1.result.json -- \\
        uv run --no-sync python scripts/experiments/a1_analyze.py \\
        --logs outputs/probes/a1/raw_<jobid> --out outputs/probes/a1/a1.result.json

Design rule: a zero torn count is evidence only if the in-place negative control fired in the same
cell. Missing control, too few reads, too short an overlap or fewer than two reader hosts distinct
from the writer make the cell, and so the run, ``inconclusive``.
"""

import argparse
import glob
import json
import logging
import os
import statistics
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("a1_analyze")

EXPECTED_FS = ("scratch", "pool", "home")
EXPECTED_SIZES = (4096, 262144, 4194304, 16777216)
MIN_READER_HOSTS = 2
MIN_OVERLAP_S = 150.0
MIN_REAL_READS = 2000
COUNTERS = ("ok", "torn", "enoent", "estale_open", "estale_read", "other")


def load(logs_dir: Path) -> list[dict]:
    rows = []
    for name in sorted(glob.glob(os.path.join(str(logs_dir), "a1_*_task*.jsonl"))):
        with open(name, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_file"] = os.path.basename(name)
                    rows.append(r)
    return rows


def analyze(rows: list[dict]) -> dict:
    meta = {r["task"]: r for r in rows if r["ev"] == "meta"}
    writers = [t for t, m in meta.items() if m["role"] == "writer"]
    writer = writers[0] if len(writers) == 1 else None
    writer_host = meta[writer]["host"] if writer is not None else None
    edges: dict = defaultdict(dict)  # (task, fs, size, phase) -> {"start": t, "end": t}
    counts: dict = defaultdict(lambda: dict.fromkeys(COUNTERS, 0))
    for r in rows:
        if r["ev"] == "phase":
            edges[(r["task"], r["fs"], r["size"], r["phase"])][r["edge"]] = r["t"]
        elif r["ev"] == "agg":
            c = counts[(r["task"], r["fs"], r["size"], r["phase"])]
            for k in COUNTERS:
                c[k] += r.get(k, 0)

    reader_tasks = [t for t, m in meta.items() if m["role"] == "reader"]
    cells = []
    for fs in EXPECTED_FS:
        for size in EXPECTED_SIZES:
            w = edges.get((writer, fs, size, "real"), {})
            qual_hosts, reads_q, overlaps = set(), 0, {}
            ctrl_torn = real_torn_all = 0
            tot = dict.fromkeys(COUNTERS, 0)
            for t in meta:
                real_torn_all += counts[(t, fs, size, "real")]["torn"]
            for t in reader_tasks:
                host = meta[t]["host"]
                if host == writer_host:
                    continue
                ctrl_torn += counts[(t, fs, size, "control")]["torn"]
                c = counts[(t, fs, size, "real")]
                for k in COUNTERS:
                    tot[k] += c[k]
                e = edges.get((t, fs, size, "real"), {})
                if {"start", "end"} <= set(e) and {"start", "end"} <= set(w):
                    ov = min(e["end"], w["end"]) - max(e["start"], w["start"])
                else:
                    ov = 0.0
                overlaps[host] = ov
                if ov >= MIN_OVERLAP_S:
                    qual_hosts.add(host)
                    reads_q += c["ok"] + c["torn"]
            cell_ok = (
                writer is not None
                and len(qual_hosts) >= MIN_READER_HOSTS
                and reads_q >= MIN_REAL_READS
                and ctrl_torn > 0
            )
            cells.append(
                {
                    "fs": fs,
                    "size": size,
                    "cell_ok": cell_ok,
                    "n_reader_hosts_qualifying": len(qual_hosts),
                    "overlap_s_by_reader_host": overlaps,
                    "real_reads_qualifying": reads_q,
                    "control_torn": ctrl_torn,
                    "real_torn_all_tasks": real_torn_all,
                    "real_distinct_reader": tot,
                }
            )
    reader_hosts = sorted({meta[t]["host"] for t in reader_tasks if meta[t]["host"] != writer_host})
    result = {
        "n_cells_expected": len(EXPECTED_FS) * len(EXPECTED_SIZES),
        "n_cells_ok": sum(c["cell_ok"] for c in cells),
        "real_torn_total": sum(c["real_torn_all_tasks"] for c in cells),
        "control_torn_total": sum(c["control_torn"] for c in cells),
        "estale_total": sum(
            c["real_distinct_reader"]["estale_open"] + c["real_distinct_reader"]["estale_read"]
            for c in cells
        ),
        "enoent_total": sum(c["real_distinct_reader"]["enoent"] for c in cells),
        "other_error_total": sum(c["real_distinct_reader"]["other"] for c in cells),
        "writer_host": writer_host or "",
        "reader_hosts": reader_hosts,
        "n_reader_hosts": len(reader_hosts),
        "cells": cells,
        "visibility": visibility(rows, meta),
    }
    result["vis_lag_median_s"] = result["visibility"]["median_s"]
    return result


def visibility(rows: list[dict], meta: dict) -> dict:
    """Report only: time from the writer's create to a reader's first successful open."""
    created = {(r["fs"], r["i"]): r["t"] for r in rows if r["ev"] == "vis_create"}
    lags: dict = defaultdict(list)
    timeouts = defaultdict(int)
    for r in rows:
        if r["ev"] == "vis_seen" and (r["fs"], r["i"]) in created:
            lags[r["fs"]].append(r["t"] - created[(r["fs"], r["i"])])
        elif r["ev"] == "vis_timeout":
            timeouts[r["fs"]] += 1
    out = {}
    for fs, v in lags.items():
        v = sorted(v)
        out[fs] = {
            "n": len(v),
            "median_s": statistics.median(v),
            "p95_s": v[min(len(v) - 1, int(0.95 * len(v)))],
            "max_s": v[-1],
            "min_s": v[0],
            "timeouts": timeouts[fs],
        }
    allv = [x for v in lags.values() for x in v]
    return {
        "by_fs": out,
        "median_s": statistics.median(allv) if allv else -1.0,
        "note": "report only; cross-node clock skew enters directly (negative min_s means skew)",
    }


def outcome_of(r: dict) -> str:
    """Python mirror of the sidecar [outcomes]; used by the selftest only."""
    if r["analyzer_selftest_ok"] and r["real_torn_total"] > 0:
        return "fail"
    if (
        r["analyzer_selftest_ok"]
        and r["real_torn_total"] == 0
        and r["n_cells_ok"] == r["n_cells_expected"]
    ):
        return "pass"
    return "inconclusive"


def synth(
    real_torn=0, control_torn=5, reads=3000, overlap=170.0, reader_hosts=("r1", "r2")
) -> list[dict]:
    """Synthetic logs with a known answer for the selftest."""
    rows = [{"ev": "meta", "task": 0, "host": "w", "role": "writer", "t": 0.0}]
    for i, h in enumerate(reader_hosts, start=1):
        rows.append({"ev": "meta", "task": i, "host": h, "role": "reader", "t": 0.0})
    for fs in EXPECTED_FS:
        for size in EXPECTED_SIZES:
            for task in range(1 + len(reader_hosts)):
                s, e = 1000.0, 1000.0 + overlap
                rows.append(
                    {
                        "ev": "phase",
                        "task": task,
                        "fs": fs,
                        "size": size,
                        "phase": "real",
                        "edge": "start",
                        "t": s,
                    }
                )
                rows.append(
                    {
                        "ev": "phase",
                        "task": task,
                        "fs": fs,
                        "size": size,
                        "phase": "real",
                        "edge": "end",
                        "t": e,
                    }
                )
                if task:
                    base = {"ev": "agg", "task": task, "fs": fs, "size": size, "t": e}
                    zeros = dict.fromkeys(COUNTERS, 0)
                    rows.append(
                        {
                            **base,
                            **zeros,
                            "phase": "real",
                            "ok": reads,
                            "torn": real_torn if task == 1 else 0,
                        }
                    )
                    rows.append(
                        {**base, **zeros, "phase": "control", "ok": 10, "torn": control_torn}
                    )
    return rows


def selftest() -> bool:
    """Positive and negative controls for the analyzer's own logic (known synthetic answers)."""
    cases = [
        (synth(), "pass"),
        (synth(real_torn=1), "fail"),  # positive control: a real torn read must be seen
        (synth(control_torn=0), "inconclusive"),  # instrument cannot see tearing
        (synth(reads=100), "inconclusive"),  # too few reads
        (synth(overlap=100.0), "inconclusive"),  # too short an overlap
        (synth(reader_hosts=("r1",)), "inconclusive"),  # one reader host
        (synth(reader_hosts=("w", "r1")), "inconclusive"),  # a reader on the writer host
    ]
    ok = True
    for rows, want in cases:
        res = analyze(rows)
        res["analyzer_selftest_ok"] = True
        got = outcome_of(res)
        if got != want:
            log.error("selftest case expected %s got %s", want, got)
            ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    st = selftest()
    rows = load(args.logs)
    if not rows:
        raise SystemExit("no a1_*_task*.jsonl rows under %s" % args.logs)
    result = analyze(rows)
    result["analyzer_selftest_ok"] = st
    result["n_log_rows"] = len(rows)
    result["logs_dir"] = str(args.logs)
    log.info(
        "selftest=%s cells_ok=%s/%s real_torn=%s control_torn=%s estale=%s",
        st,
        result["n_cells_ok"],
        result["n_cells_expected"],
        result["real_torn_total"],
        result["control_torn_total"],
        result["estale_total"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
