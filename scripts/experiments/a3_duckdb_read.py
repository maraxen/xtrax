# ruff: noqa: UP031
"""A3 probe (spec 261030 §3.1 Read side): DuckDB read time over a unit index at 1e6 rows.

Builds a deterministic unit-index fixture under ``--root`` (segments/, sealed/, rebuilt/) for each
requested layout, then times the spec's DuckDB view:

    read_json([...], format='newline_delimited', ignore_errors=true, columns={...})
    + DISTINCT ON (store, locator, record_digest)

as (a) a full count of the deduplicated view and (b) one point lookup on (unit_key, input_digest),
each in a fresh connection, ``--repeats`` times. It also times a stdlib reader that applies the same
rule (the canonical ``iter_rows`` stand-in) and runs the correctness controls:

* dedup count == expected unique rows (fixture holds exact duplicates in rebuilt/, a sealed/segment
  overlap, ``{"closed": true}`` trailers and torn last lines);
* negative control: without DISTINCT ON the count is higher by exactly the planted duplicates;
* negative control: without ignore_errors DuckDB refuses the torn tails;
* the stdlib reader and DuckDB agree on the deduplicated count.

Writes one raw JSON (``--out``); judging is done by ``a3_analyze.py`` under its sidecar.
Needs only the stdlib and ``duckdb`` (run with ``uv run --no-sync --with duckdb==2.0.0``).
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import socket
import time
from pathlib import Path

import duckdb

log = logging.getLogger("a3")

COLUMNS = {
    "schema_version": "INTEGER",
    "store": "VARCHAR",
    "locator": "VARCHAR",
    "unit_key": "VARCHAR",
    "input_digest": "VARCHAR",
    "content_digest": "VARCHAR",
    "record_digest": "VARCHAR",
    "adapter": "VARCHAR",
    "writer_id": "VARCHAR",
    "run_id": "VARCHAR",
    "ts": "DOUBLE",
    "reused_from": "VARCHAR",
}
SEALED_FILE_ROWS = 100_000
DUP_REBUILT = 1000  # exact copies of sealed/segment rows written to rebuilt/
DUP_OVERLAP = 1000  # rows present in both a sealed file and a segment (crash between seal and move)
TORN_EVERY = 10  # every 10th segment ends with a torn last line
STRICT_MULTI_TRIES = 5
VISIBILITY_WAIT_S = 120.0
OPEN_EVERY = 10  # every 10th segment (offset 5) has no trailer (live writer)

# name -> (n_segments, rows held in segments); the rest of --rows is sealed
LAYOUTS = {
    "steady": (100, 10_000),
    "busy": (10_000, 100_000),
    "uncompacted": (100_000, 1_000_000),
}


def _h(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def make_row(i: int, writer_id: str) -> dict:
    store = "/fake/store%d" % (i % 4)
    locator = "units/%07d/record.json" % i
    content = _h("content", i)
    return {
        "schema_version": 1,
        "store": store,
        "locator": locator,
        "unit_key": "u%07d" % i,
        "input_digest": _h("input", i),
        "content_digest": content,
        "record_digest": _h("record", store, locator, content),
        "adapter": "zarr_commit",
        "writer_id": writer_id,
        "run_id": "run-%d" % (i % 97),
        "ts": 1.8e9 + i,
        "reused_from": None,
    }


def build_fixture(root: Path, n_rows: int, n_segments: int, seg_rows: int) -> dict:
    if root.exists():
        shutil.rmtree(root)
    for d in ("segments", "sealed", "rebuilt"):
        (root / d).mkdir(parents=True)
    sealed_rows = n_rows - seg_rows
    t0 = time.perf_counter()
    # sealed: rows [0, sealed_rows)
    for f0 in range(0, sealed_rows, SEALED_FILE_ROWS):
        with open(root / "sealed" / ("sealed-%07d.jsonl" % f0), "w") as fh:
            for i in range(f0, min(f0 + SEALED_FILE_ROWS, sealed_rows)):
                fh.write(json.dumps(make_row(i, "sealed")) + "\n")
    # segments: rows [sealed_rows, n_rows), round-robin into n_segments files
    per = seg_rows // n_segments
    torn = 0
    overlap_written = 0
    for s in range(n_segments):
        wid = "node%04d-%d-%d-0-%08x" % (s % 1000, 1000 + s, 27000000 + s, s)
        lo = sealed_rows + s * per
        hi = n_rows if s == n_segments - 1 else lo + per
        with open(root / "segments" / (wid + ".jsonl"), "w") as fh:
            for i in range(lo, hi):
                fh.write(json.dumps(make_row(i, wid)) + "\n")
            # overlap: re-append a few already-sealed rows (identical record) to early segments
            if sealed_rows and overlap_written < DUP_OVERLAP:
                k = min(DUP_OVERLAP - overlap_written, max(1, DUP_OVERLAP // n_segments))
                for j in range(k):
                    fh.write(json.dumps(make_row(overlap_written + j, "sealed")) + "\n")
                overlap_written += k
            if s % OPEN_EVERY != 5:
                fh.write('{"closed": true}\n')
            if s % TORN_EVERY == 0:
                fh.write('{"schema_version": 1, "store": "/fake/st')  # torn, no newline
                torn += 1
    with open(root / "rebuilt" / "index-0.jsonl", "w") as fh:
        for j in range(DUP_REBUILT):
            fh.write(json.dumps(make_row((j * 997) % n_rows, "rebuilt")) + "\n")
    n_files = sum(1 for _ in root.rglob("*.jsonl"))
    return {
        "n_rows_unique": n_rows,
        "n_dup_planted": DUP_REBUILT + overlap_written,
        "n_torn": torn,
        "n_segments": n_segments,
        "n_sealed_files": len(list((root / "sealed").glob("*.jsonl"))),
        "n_files": n_files,
        "bytes": sum(p.stat().st_size for p in root.rglob("*.jsonl")),
        "build_s": time.perf_counter() - t0,
    }


def _globs(root: Path) -> list:
    # DuckDB errors on a glob that matches nothing, so only non-empty directories are listed.
    return [
        str(root / d / "*.jsonl")
        for d in ("segments", "sealed", "rebuilt")
        if any((root / d).glob("*.jsonl"))
    ]


def _source(root: Path, ignore_errors: bool = True) -> str:
    cols = "{" + ", ".join("'%s': '%s'" % kv for kv in COLUMNS.items()) + "}"
    files = "[" + ", ".join("'%s'" % g for g in _globs(root)) + "]"
    return "read_json(%s, format='newline_delimited', ignore_errors=%s, columns=%s)" % (
        files,
        "true" if ignore_errors else "false",
        cols,
    )


def _view(root: Path) -> str:
    return (
        "SELECT DISTINCT ON (store, locator, record_digest) * FROM %s WHERE store IS NOT NULL"
        % _source(root)
    )


def _con(threads: int) -> "duckdb.DuckDBPyConnection":
    con = duckdb.connect()
    con.execute("SET threads = %d" % threads)
    con.execute("SET memory_limit = '4GB'")
    return con


def time_queries(root: Path, threads: int, repeats: int, probe_i: int) -> dict:
    probe = make_row(probe_i, "")
    out = {"view_count_s": [], "lookup_s": [], "view_count": None, "lookup_rows": None}
    for _ in range(repeats):
        con = _con(threads)
        t0 = time.perf_counter()
        n = con.execute("SELECT count(*) FROM (%s)" % _view(root)).fetchone()[0]
        out["view_count_s"].append(time.perf_counter() - t0)
        out["view_count"] = n
        con.close()
        con = _con(threads)
        t0 = time.perf_counter()
        rows = con.execute(
            "SELECT locator FROM (%s) WHERE unit_key = ? AND input_digest = ?" % _view(root),
            [probe["unit_key"], probe["input_digest"]],
        ).fetchall()
        out["lookup_s"].append(time.perf_counter() - t0)
        out["lookup_rows"] = len(rows)
        con.close()
    return out


def controls(root: Path, threads: int) -> dict:
    con = _con(threads)
    raw = con.execute("SELECT count(*) FROM %s WHERE store IS NOT NULL" % _source(root)).fetchone()[
        0
    ]
    con.close()
    strict_refused, strict_error = _strict_refuses(root, threads=1)
    # DuckDB 2.0.0, smoke 2026-11-03: with threads > 1 a strict read sometimes drops the torn
    # line silently instead of raising (3/10 refusals at 4 threads, 10/10 at 1). The gating
    # control runs single-threaded; the multi-thread refusal rate is recorded, not gated.
    multi = [_strict_refuses(root, threads)[0] for _ in range(STRICT_MULTI_TRIES)]
    return {
        "count_no_distinct": raw,
        "strict_refused": strict_refused,
        "strict_error": strict_error,
        "strict_multi_threads": threads,
        "strict_multi_refusals": sum(multi),
        "strict_multi_tries": STRICT_MULTI_TRIES,
    }


def _strict_refuses(root: Path, threads: int) -> tuple:
    con = _con(threads)
    try:
        # count(store), not count(*): a bare count(*) can skip parsing (smoke run, 2026-11-03).
        con.execute("SELECT count(store) FROM %s" % _source(root, ignore_errors=False)).fetchone()
        return False, ""
    except duckdb.Error as e:
        return True, type(e).__name__ + ": " + str(e)[:200]
    finally:
        con.close()


def stdlib_reader(root: Path) -> dict:
    """Same rule as the canonical reader: skip torn/non-row lines, dedup on the triple."""
    t0 = time.perf_counter()
    seen = set()
    skipped = 0
    for d in ("segments", "sealed", "rebuilt"):
        for p in sorted((root / d).glob("*.jsonl")):
            with open(p, "rb") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        skipped += 1
                        continue
                    if not isinstance(row, dict) or row.get("store") is None:
                        continue
                    seen.add((row["store"], row["locator"], row["record_digest"]))
    return {"count": len(seen), "skipped_lines": skipped, "s": time.perf_counter() - t0}


def build_phase(a: argparse.Namespace) -> None:
    for name in a.layouts.split(","):
        scale = a.rows / 1_000_000  # layouts are defined at 1e6 rows; scale them for smoke runs
        n_seg = max(1, int(LAYOUTS[name][0] * scale))
        seg_rows = max(n_seg, int(LAYOUTS[name][1] * scale))
        root = a.root / name
        log.info("building %s: %d segments, %d segment rows", name, n_seg, seg_rows)
        fx = build_fixture(root, a.rows, n_seg, seg_rows)
        fx.update(
            build_host=socket.gethostname(),
            rows=a.rows,
            build_slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        (root / "fixture.json").write_text(json.dumps(fx, indent=2) + "\n")
        log.info("built %s: %s", name, fx)


def read_phase(a: argparse.Namespace) -> None:
    host = socket.gethostname()
    result = {
        "env": a.env,
        "host": host,
        "python": platform.python_version(),
        "duckdb": duckdb.__version__,
        "threads": a.threads,
        "repeats": a.repeats,
        "rows": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "root": str(a.root),
        "layouts": {},
    }
    for name in a.layouts.split(","):
        root = a.root / name
        fx = json.loads((root / "fixture.json").read_text())
        result["rows"] = fx["rows"]
        fx["cross_node"] = fx["build_host"] != host
        # NFS: a file made on another node can take up to ~30 s to appear (spec §3.1 ESTALE note).
        # Wait (bounded) until every built file is listed, so timing never runs on a partial set.
        t0 = time.perf_counter()
        seen = sum(1 for _ in root.rglob("*.jsonl"))
        while seen < fx["n_files"] and time.perf_counter() - t0 < VISIBILITY_WAIT_S:
            time.sleep(2.0)
            seen = sum(1 for _ in root.rglob("*.jsonl"))
        fx["visibility_wait_s"] = time.perf_counter() - t0
        fx["files_visible"] = seen
        log.info(
            "%s: %d/%d files visible after %.1fs",
            name,
            seen,
            fx["n_files"],
            fx["visibility_wait_s"],
        )
        q = time_queries(root, a.threads, a.repeats, probe_i=fx["rows"] - 7)
        log.info("timed %s: %s", name, q)
        c = controls(root, a.threads)
        py = stdlib_reader(root)
        result["layouts"][name] = {"fixture": fx, "queries": q, "controls": c, "stdlib": py}
        if not a.keep:
            shutil.rmtree(root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    log.info("wrote %s", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--phase",
        choices=["build", "read", "all"],
        default="all",
        help="build fixtures, read (time) existing ones, or both",
    )
    ap.add_argument("--root", type=Path, required=True, help="scratch dir for the fixture")
    ap.add_argument("--env", default="local", help="label, e.g. local or engaging_scratch")
    ap.add_argument("--layouts", default="steady,busy", help="comma list of %s" % list(LAYOUTS))
    ap.add_argument("--rows", type=int, default=1_000_000, help="build phase only")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="keep fixtures after reading")
    ap.add_argument("--out", type=Path, help="raw result JSON (read/all phases)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.phase in ("build", "all"):
        build_phase(a)
    if a.phase in ("read", "all"):
        if a.out is None:
            ap.error("--out is required for the read phase")
        read_phase(a)


if __name__ == "__main__":
    main()
