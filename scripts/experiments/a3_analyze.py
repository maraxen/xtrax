# ruff: noqa: UP031
"""Judge an A3 raw result (from a3_duckdb_read.py) against the a3_analyze.bth.toml outcomes.

Reads one raw JSON, checks every control per layout, and reports the gate time: the largest single
query time (deduplicated-view count or point lookup, any repeat, first run included) over the
gating layouts ``steady`` and ``busy``. ``uncompacted`` (1e5 segments) is reported, not gated.
Writes the flat result JSON that bathos evaluates (``--out``; also ``$BTH_RESULTS_PATH`` if set).
"""

import argparse
import hashlib
import json
import logging
import os
import statistics
from pathlib import Path

log = logging.getLogger("a3_analyze")
GATING = ("steady", "busy")


def layout_controls(v: dict) -> dict:
    f, q, c, s = v["fixture"], v["queries"], v["controls"], v["stdlib"]
    return {
        "dedup_exact": q["view_count"] == f["n_rows_unique"],
        "lookup_one_row": q["lookup_rows"] == 1,
        "dups_detected": c["count_no_distinct"] - f["n_rows_unique"] == f["n_dup_planted"]
        and f["n_dup_planted"] > 0,
        "strict_refuses_torn": c["strict_refused"] and f["n_torn"] > 0,
        "readers_agree": s["count"] == q["view_count"],
        "stdlib_skips_torn": s["skipped_lines"] == f["n_torn"],
    }


def analyze(raw: dict) -> dict:
    lay = raw["layouts"]
    checks = {name: layout_controls(v) for name, v in lay.items()}
    for name, ch in checks.items():
        log.info("%s controls: %s", name, ch)
    controls_ok = bool(checks) and all(all(ch.values()) for ch in checks.values())
    gating_present = all(g in lay for g in GATING)

    def times(name: str) -> list:
        q = lay[name]["queries"]
        return q["view_count_s"] + q["lookup_s"]

    gate = [t for g in GATING if g in lay for t in times(g)]
    first = [
        lay[g]["queries"][k][0] for g in GATING if g in lay for k in ("view_count_s", "lookup_s")
    ]
    unc = times("uncompacted") if "uncompacted" in lay else []
    return {
        "env": raw["env"],
        "host": raw["host"],
        "duckdb": raw["duckdb"],
        "threads": raw["threads"],
        "rows": raw["rows"],
        "slurm_job_id": raw.get("slurm_job_id") or "",
        "controls_ok": controls_ok,
        "gating_layouts_present": gating_present,
        "gate_max_s": max(gate) if gate else -1.0,
        "gate_median_s": statistics.median(gate) if gate else -1.0,
        "gate_first_run_max_s": max(first) if first else -1.0,
        "steady_max_s": max(times("steady")) if "steady" in lay else -1.0,
        "busy_max_s": max(times("busy")) if "busy" in lay else -1.0,
        "uncompacted_max_s": max(unc) if unc else -1.0,
        "stdlib_reader_max_s": max(v["stdlib"]["s"] for v in lay.values()),
        "busy_n_files": lay["busy"]["fixture"]["n_files"] if "busy" in lay else -1,
        "cross_node": all(v["fixture"].get("cross_node", False) for v in lay.values()),
        "strict_multi_refusals": sum(
            v["controls"].get("strict_multi_refusals", 0) for v in lay.values()
        ),
        "strict_multi_tries": sum(v["controls"].get("strict_multi_tries", 0) for v in lay.values()),
        "failed_controls": sorted(
            "%s.%s" % (n, k) for n, ch in checks.items() for k, ok in ch.items() if not ok
        ),
    }


def self_test() -> bool:
    """Positive and negative control for the judge itself on a synthetic raw result."""

    def layout(dup_ok: bool) -> dict:
        return {
            "fixture": {"n_rows_unique": 10, "n_dup_planted": 2, "n_torn": 1, "n_files": 3},
            "queries": {
                "view_count": 10,
                "lookup_rows": 1,
                "view_count_s": [1.0],
                "lookup_s": [2.0],
            },
            "controls": {"count_no_distinct": 12 if dup_ok else 10, "strict_refused": True},
            "stdlib": {"count": 10, "skipped_lines": 1, "s": 0.5},
        }

    base = {"env": "x", "host": "h", "duckdb": "2", "threads": 1, "rows": 10}
    good = analyze({**base, "layouts": {"steady": layout(True), "busy": layout(True)}})
    bad = analyze({**base, "layouts": {"steady": layout(True), "busy": layout(False)}})
    return good["controls_ok"] and good["gate_max_s"] == 2.0 and not bad["controls_ok"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = a.raw.read_bytes()
    res = analyze(json.loads(data))
    res["self_test_ok"] = self_test()
    res["raw_path"] = str(a.raw)
    res["raw_sha256"] = hashlib.sha256(data).hexdigest()
    res["failed_controls"] = ",".join(res["failed_controls"])
    text = json.dumps(res, indent=2) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    if os.environ.get("BTH_RESULTS_PATH"):
        Path(os.environ["BTH_RESULTS_PATH"]).write_text(text)
    log.info("%s", text)


if __name__ == "__main__":
    main()
