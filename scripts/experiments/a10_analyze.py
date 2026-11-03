# ruff: noqa: UP031, E501
"""A10 analysis: false "writer is dead" verdicts under preemption and requeue (spec 261030 section 4).

Reads the raw JSONL written by ``a10_controller.py`` (samples of three rules on identical scheduler
inputs, plus each segment's heartbeat wall times) and scores each rule against ground truth.

Ground truth: rule R makes a FALSE POSITIVE at sample time t for segment S if R says dead at t and S
later holds a heartbeat row with wall time > t + 5 s. A TRUE POSITIVE is a dead verdict on a segment
with no such later heartbeat; time-to-dead is measured from the segment's last heartbeat. Samples
taken under the deliberately broken query ("broken" input) count as well: a rule that calls a live
writer dead because the scheduler could not be queried is exactly the failure under test.

Outcome rules are in ``a10_analyze.bth.toml``. Submission and run commands for the whole probe are in
the docstring of ``a10_controller.py`` (reproduced here):

    myxcel submit-job engaging xtrax --preset cpu --override time=00:55:00 --override cpus_per_task=1 \\
        --command "/usr/bin/python3.11 scripts/experiments/a10_controller.py \\
        --segdir /orcd/scratch/orcd/011/maarxaru/a10_probe --out-dir outputs/probes/a10/raw"

    bth run --project-slug xtrax --output-paths outputs/probes/a10/a10.result.json -- \\
        uv run --no-sync python scripts/experiments/a10_analyze.py \\
        --logs outputs/probes/a10/raw --out outputs/probes/a10/a10.result.json

The controller submits the writer jobs itself with ``sbatch``; whether Engaging allows ``sbatch`` from
a compute node is UNVERIFIED. If it does not, submit the two writer jobs with myxcel (W with
``--override requeue=true``) and pass their ids as ``--w-job``/``--a-job``.
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path

log = logging.getLogger("a10_analyze")

MARGIN_S = 5.0
RULES = ("SHIPPING", "NAIVE_ERR", "NAIVE_NOGRACE")
TERMINAL = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
)
MIN_BROKEN_SAMPLES = 3


def load(logs: Path) -> list[dict]:
    rows = []
    for name in sorted(glob.glob(os.path.join(str(logs), "a10_*_controller.jsonl"))):
        with open(name, encoding="utf-8") as f:
            rows.extend(json.loads(ln) for ln in f if ln.strip())
    return rows


def w_state(calls: list[dict], jid: str) -> tuple[str, str]:
    """``(squeue_state, sacct_state)`` of job ``jid`` from one sample's raw calls."""
    sq = sa = ""
    for c in calls:
        a = c.get("argv", [])
        if jid not in a:
            continue
        if a[0] == "squeue":
            if c.get("rc") == 0 and c["out"].strip():
                sq = c["out"].strip().split("\n")[0].strip()
            elif "invalid job id" in (c.get("err", "") + c.get("out", "")).lower():
                sq = "GONE"
            else:
                sq = "ERROR"
        elif a[0] == "sacct" and c.get("rc") == 0:
            sa = c["out"].strip().split("\n")[0].strip()
    return sq, sa


def phases_of_w(samples: list[dict], w_job: str) -> dict:
    ph = dict.fromkeys(
        ("running", "requeued_pending", "running_restart1", "completed", "aged_out"), False
    )
    for s in samples:
        sq, sa = w_state(s["calls"], w_job)
        restart1 = any(g["job_id"] == w_job and g["restart"] >= 1 for g in s["segments"])
        if sq == "RUNNING" and not ph["running"]:
            ph["running"] = True
        elif (
            ph["running"]
            and not ph["running_restart1"]
            and sq not in ("RUNNING", "GONE", "ERROR", "")
            and not sq.startswith(TERMINAL)
            and not sa.startswith(TERMINAL)
        ):
            ph["requeued_pending"] = True
        if ph["requeued_pending"] and sq == "RUNNING" and restart1:
            ph["running_restart1"] = True
        if ph["running_restart1"] and (sq.startswith(TERMINAL) or sa.startswith(TERMINAL)):
            ph["completed"] = True
        if ph["completed"] and sq == "GONE":
            ph["aged_out"] = True
    return ph


def analyze(rows: list[dict]) -> dict:
    sub = next((r for r in rows if r["ev"] == "submitted"), None)
    samples = sorted((r for r in rows if r["ev"] == "sample"), key=lambda r: r["t"])
    finals = {r["name"]: r for r in rows if r["ev"] == "final"}
    w_job = sub["w_job"] if sub else ""
    walls = {n: sorted(f["walls"]) for n, f in finals.items()}
    last_wall = {n: (w[-1] if w else 0.0) for n, w in walls.items()}

    fp = {k: {"live": 0, "broken": 0} for k in RULES}
    first_dead: dict = {k: {} for k in RULES}
    err_dead_in_broken = ship_dead_in_broken = 0
    n_broken_samples = 0
    for s in samples:
        had_broken = False
        for g in s["segments"]:
            name = g["name"]
            for kind, vd in g["verdicts"].items():
                had_broken = had_broken or kind == "broken"
                for rule in RULES:
                    if not vd[rule]["dead"]:
                        continue
                    if last_wall.get(name, 0.0) > s["t"] + MARGIN_S:
                        fp[rule][kind] += 1
                    elif kind == "live":
                        first_dead[rule].setdefault(name, s["t"])
                    if kind == "broken" and rule == "NAIVE_ERR":
                        err_dead_in_broken += 1
                    if kind == "broken" and rule == "SHIPPING":
                        ship_dead_in_broken += 1
        n_broken_samples += had_broken

    segs = {}
    for name in sorted(walls):
        base = name[: -len(".jsonl")]
        job = base.rsplit("-", 4)[-3] if base.count("-") >= 4 else ""
        restart = base.rsplit("-", 4)[-2] if base.count("-") >= 4 else ""
        segs[name] = {
            "job": job,
            "restart": restart,
            "closed": finals[name]["closed"],
            "last_wall": last_wall[name],
            "time_to_dead_s": {
                r: (first_dead[r][name] - last_wall[name]) if name in first_dead[r] else None
                for r in RULES
            },
        }
    shipping_missed = [n for n in segs if n not in first_dead["SHIPPING"]]
    missed_closed = [n for n in shipping_missed if segs[n]["closed"]]
    ph = phases_of_w(samples, w_job)
    array_sampled = any(
        g["job_id"] != w_job and g["job_id"] is not None for s in samples for g in s["segments"]
    )
    fp_tot = {k: fp[k]["live"] + fp[k]["broken"] for k in RULES}
    naive_fp_total = fp_tot["NAIVE_ERR"] + fp_tot["NAIVE_NOGRACE"]
    phases_ok = all(ph.values())
    coverage_ok = (
        phases_ok
        and n_broken_samples >= MIN_BROKEN_SAMPLES
        and array_sampled
        and naive_fp_total >= 1
    )
    return {
        "shipping_fp": fp_tot["SHIPPING"],
        "naive_err_fp": fp_tot["NAIVE_ERR"],
        "naive_nograce_fp": fp_tot["NAIVE_NOGRACE"],
        "naive_fp_total": naive_fp_total,
        "shipping_missed_dead": len(shipping_missed),
        "shipping_missed_closed": len(missed_closed),
        "phases_ok": phases_ok,
        "broken_window_samples": n_broken_samples,
        "array_sampled": array_sampled,
        "coverage_ok": coverage_ok,
        "n_samples": len(samples),
        "n_segments": len(segs),
        "naive_err_dead_in_broken": err_dead_in_broken,
        "shipping_dead_in_broken": ship_dead_in_broken,
        "fp_by_rule_and_input": fp,
        "w_phases": ph,
        "segments": segs,
        "shipping_missed_names": shipping_missed,
    }


def outcome_of(r: dict) -> str:
    """Python mirror of the sidecar [outcomes]; used by the selftest only."""
    if not r["analyzer_selftest_ok"]:
        return "inconclusive"
    if r["shipping_fp"] > 0:
        return "fail"
    if r["coverage_ok"] and r["shipping_missed_dead"] > 0:
        return "marginal"
    if r["coverage_ok"] and r["shipping_missed_dead"] == 0:
        return "pass"
    return "inconclusive"


# ---- selftest: synthetic traces with known answers ---------------------------------------------


def _calls(job, sq, sa):
    out = [
        {"argv": ["squeue", "-h", "-j", job, "-o", "%T"]},
        {"argv": ["sacct", "-j", job, "-n", "-X", "-P", "-o", "State"]},
    ]
    if sq == "GONE":
        out[0].update(rc=1, out="", err="slurm_load_jobs error: Invalid job id specified")
    else:
        out[0].update(rc=0, out=sq + "\n", err="")
    out[1].update(rc=0, out=sa + "\n", err="")
    return out


def synth(
    *,
    naive_err_fp=True,
    shipping_fp=False,
    shipping_blind=False,
    broken=True,
    array=True,
    skip_requeue=False,
):
    """W=1 (killed restart 0, closed restart 1) and A=2; samples every 10 s."""
    segdef = {
        "h-1-1-0-aaaaaaaa.jsonl": ("1", 0, 0.0, 100.0, False),
        "h-2-1-1-bbbbbbbb.jsonl": ("1", 1, 160.0, 300.0, True),
        "h-3-2-0-cccccccc.jsonl": ("2", 0, 0.0, 60.0, True),
    }
    if not array:
        segdef.pop("h-3-2-0-cccccccc.jsonl")
    rows = [{"ev": "submitted", "w_job": "1", "a_job": "2", "t": 0.0}]
    for name, (_j, _r, w0, w1, closed) in segdef.items():
        walls = [w0 + i * 2.0 for i in range(int((w1 - w0) / 2) + 1)]
        rows.append({"ev": "final", "name": name, "walls": walls, "closed": closed, "t": 999.0})
    for t in range(0, 430, 10):
        t = float(t)
        if t < 100:
            sq, sa = "RUNNING", ""
        elif t < 150:
            sq, sa = ("RUNNING", "") if skip_requeue else ("PENDING", "")
        elif t < 300:
            sq, sa = "RUNNING", ""
        elif t < 400:
            sq, sa = "COMPLETED", "COMPLETED"
        else:
            sq, sa = "GONE", "COMPLETED"
        segs = []
        for name, (job, restart, w0, w1, _c) in segdef.items():
            if w0 > t:
                continue
            last = min(t, w1)
            in_broken = broken and 200 <= t <= 280
            kinds = ["live"] + (["broken"] if in_broken else [])
            vd = {}
            for kind in kinds:
                dead = t > w1 + 10  # correct rule: dead only after the last heartbeat
                v = {r: {"dead": dead, "reason": ""} for r in RULES}
                if shipping_blind:
                    v["SHIPPING"]["dead"] = False
                if shipping_fp and name.startswith("h-2") and t == 220.0 and kind == "live":
                    v["SHIPPING"]["dead"] = True
                if naive_err_fp and kind == "broken" and name.startswith("h-2"):
                    v["NAIVE_ERR"]["dead"] = True
                vd[kind] = v
            segs.append(
                {"name": name, "job_id": job, "restart": restart, "mtime": last, "verdicts": vd}
            )
        rows.append(
            {
                "ev": "sample",
                "t": t,
                "server_now": t,
                "broken_active": bool(broken and 200 <= t <= 280),
                "calls": _calls("1", sq, sa)
                + _calls("2", "RUNNING" if t < 100 else "GONE", "COMPLETED"),
                "broken_calls": [],
                "segments": segs,
            }
        )
    return rows


def selftest() -> bool:
    cases = [
        (synth(), "pass"),
        (synth(shipping_fp=True), "fail"),  # positive control: a shipping false positive is seen
        (
            synth(naive_err_fp=False),
            "inconclusive",
        ),  # no naive rule fired: cannot show the probe can see FPs
        (synth(broken=False), "inconclusive"),  # no broken-query window
        (synth(array=False), "inconclusive"),  # array job never sampled
        (synth(skip_requeue=True), "inconclusive"),  # no requeued/pending phase
        (synth(shipping_blind=True), "marginal"),  # safe but never marks anything dead
    ]
    ok = True
    for rows, want in cases:
        res = analyze(rows)
        res["analyzer_selftest_ok"] = True
        got = outcome_of(res)
        if got != want:
            log.error(
                "selftest expected %s got %s (%s)",
                want,
                got,
                {
                    k: res[k]
                    for k in ("shipping_fp", "coverage_ok", "phases_ok", "shipping_missed_dead")
                },
            )
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
        raise SystemExit("no a10_*_controller.jsonl under %s" % args.logs)
    result = analyze(rows)
    result["analyzer_selftest_ok"] = st
    result["logs_dir"] = str(args.logs)
    log.info(
        "selftest=%s shipping_fp=%s naive_fp=%s missed=%s coverage_ok=%s phases=%s",
        st,
        result["shipping_fp"],
        result["naive_fp_total"],
        result["shipping_missed_dead"],
        result["coverage_ok"],
        result["w_phases"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
