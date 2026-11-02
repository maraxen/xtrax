#!/usr/bin/env python3
# ruff: noqa: UP031, E501
"""A10 controller: requeue-injection probe for the "writer is dead" rule (spec 261030 section 4, A10).

Runs as its own CPU job on Engaging under Python >= 3.9 (``/usr/bin/python3.11``; the default
``python3`` is 3.6.8, spec A8). Stdlib only. It imports the SHIPPING liveness code by file path from
``src/xtrax/run/writer_liveness.py`` (no copy) and logs raw events only. The outcome is computed by
``a10_analyze.py`` under ``bth run`` against the pulled logs.

What it does
------------
1. ``sbatch`` writer job W (``--requeue``) and a 1-task array writer job A, both running
   ``a10_writer.py`` into a shared segment directory (or adopt already-submitted ids with
   ``--w-job/--a-job`` if ``sbatch`` is not allowed from compute nodes: unverified assumption).
2. Every ``--sample-s`` seconds (2 s for 90 s after the injection) it queries squeue and sacct for
   each job ONCE (memoised per sample, so all rules see identical raw inputs) and, for every segment
   present, evaluates three rules on those inputs:
   * SHIPPING: ``segment_writer_dead(grace_s=300, now_server=server_now)``.
   * NAIVE_ERR: dead iff a trailer exists, or the squeue query "gone" (nonzero rc, error or empty
     output, no sacct) in this sample AND the previous one (a one-sample debounce, its "grace").
   * NAIVE_NOGRACE: the shipping state logic with ``grace_s=0`` and the local clock.
3. Injection: ``--requeue-after-s`` after W is first RUNNING, ``scontrol requeue W`` then
   ``scontrol hold W`` for ``--hold-s`` (a deterministic pending phase), then ``scontrol release``.
   Once the restart-1 segment has been alive ``--broken-after-s``, for ``--broken-len-s`` every rule
   is ALSO evaluated against a deliberately broken scheduler query (``SLURM_CONF=/nonexistent``):
   the shipping rule must say not-dead (UNKNOWN), NAIVE_ERR says dead.
4. Samples until W and A are aged out of squeue (MinJobAge 120 s) plus 2 x grace, then copies the
   segments to ``--out-dir`` and logs each segment's heartbeat wall times for ground truth.
Total time is bounded by ``--max-s`` (default 3000 s).

SUBMISSION (do not run here). From the project root on Engaging:

    myxcel submit-job engaging xtrax --preset cpu --override time=00:55:00 --override cpus_per_task=1 \\
        --command "/usr/bin/python3.11 scripts/experiments/a10_controller.py \\
        --segdir /orcd/scratch/orcd/011/maarxaru/a10_probe --out-dir outputs/probes/a10/raw"

Then pull ``outputs/probes/a10/raw`` and run, locally, under bathos:

    bth run --project-slug xtrax --output-paths outputs/probes/a10/a10.result.json -- \\
        uv run --no-sync python scripts/experiments/a10_analyze.py \\
        --logs outputs/probes/a10/raw --out outputs/probes/a10/a10.result.json
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("a10_controller")
PY = "/usr/bin/python3.11"


def load_liveness(repo):
    path = os.path.join(repo, "src", "xtrax", "run", "writer_liveness.py")
    spec = importlib.util.spec_from_file_location("writer_liveness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["writer_liveness"] = mod
    spec.loader.exec_module(mod)
    return mod, path


class Memo:
    """Runner that caches each argv for one sample, records the raw result, replays exceptions."""

    def __init__(self, env=None):
        self.env = env
        self.calls = {}

    def __call__(self, argv, timeout):
        key = tuple(argv)
        if key not in self.calls:
            try:
                p = subprocess.run(
                    argv, capture_output=True, text=True, timeout=timeout, env=self.env, check=False
                )
                self.calls[key] = {
                    "argv": list(argv),
                    "rc": p.returncode,
                    "out": p.stdout,
                    "err": p.stderr,
                }
            except (OSError, subprocess.SubprocessError) as e:
                self.calls[key] = {"argv": list(argv), "exc": type(e).__name__}
        c = self.calls[key]
        if "exc" in c:
            raise OSError(c["exc"])
        return c["rc"], c["out"], c["err"]

    def dump(self):
        return list(self.calls.values())


def run(argv, timeout=30):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, "", type(e).__name__


def submit(args, name, run_s, array, requeue):
    script = os.path.join(args.out_dir, "a10_%s_%s.sbatch" % (args.tag, name))
    lines = [
        "#!/bin/bash",
        "#SBATCH -J a10_%s_%s" % (args.tag, name),
        "#SBATCH -p %s" % args.partition,
        "#SBATCH -t %s" % args.writer_time,
        "#SBATCH -c 1",
        "#SBATCH --mem=1G",
        "#SBATCH -D %s" % os.path.abspath(args.repo),
        "#SBATCH -o %s"
        % os.path.join(os.path.abspath(args.out_dir), "a10_%s_%s_%%j.log" % (args.tag, name)),
    ]
    if requeue:
        lines.append("#SBATCH --requeue")
    if array:
        lines.append("#SBATCH --array=0-0")
    lines.append(
        "exec %s scripts/experiments/a10_writer.py --segdir %s --duration-s %d"
        % (PY, os.path.abspath(args.segdir), run_s)
    )
    with open(script, "w") as f:
        f.write("\n".join(lines) + "\n")
    rc, out, err = run(["sbatch", "--parsable", script])
    if rc != 0:
        raise SystemExit("sbatch failed for %s: rc=%d %s" % (name, rc, err))
    return out.split(";")[0]


def job_gone(calls, jid):
    """True once squeue answered 'invalid job id' for ``jid``."""
    for c in calls:
        if (
            c.get("argv", [""])[0] == "squeue"
            and jid in c["argv"]
            and "invalid job id" in (c.get("err", "") + c.get("out", "")).lower()
        ):
            return True
    return False


def squeue_state(calls, jid):
    for c in calls:
        if c.get("argv", [""])[0] == "squeue" and jid in c["argv"] and c.get("rc") == 0:
            return c["out"].strip().split("\n")[0].strip()
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segdir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--partition", default="mit_normal")
    ap.add_argument("--writer-time", default="00:12:00")
    ap.add_argument("--w-run-s", type=int, default=240)
    ap.add_argument("--a-run-s", type=int, default=120)
    ap.add_argument("--sample-s", type=float, default=10.0)
    ap.add_argument("--fast-sample-s", type=float, default=2.0)
    ap.add_argument("--grace-s", type=float, default=300.0)
    ap.add_argument("--requeue-after-s", type=float, default=120.0)
    ap.add_argument("--hold-s", type=float, default=30.0)
    ap.add_argument("--broken-after-s", type=float, default=40.0)
    ap.add_argument("--broken-len-s", type=float, default=100.0)
    ap.add_argument("--start-timeout-s", type=float, default=1500.0)
    ap.add_argument("--max-s", type=float, default=3000.0)
    ap.add_argument("--w-job", default="")
    ap.add_argument("--a-job", default="")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    wl, wl_path = load_liveness(args.repo)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.segdir, exist_ok=True)
    out = open(os.path.join(args.out_dir, "a10_%s_controller.jsonl" % args.tag), "a", buffering=1)

    def row(ev, **kw):
        kw["ev"] = ev
        kw.setdefault("t", time.time())
        out.write(json.dumps(kw, sort_keys=True) + "\n")

    with open(wl_path, "rb") as f:
        wl_sha = hashlib.sha256(f.read()).hexdigest()
    rev = run(["git", "-C", args.repo, "rev-parse", "HEAD"])[1]
    row(
        "meta",
        host=socket.gethostname(),
        py=sys.version.split()[0],
        args=vars(args),
        writer_liveness_path=wl_path,
        writer_liveness_sha256=wl_sha,
        git_head=rev,
    )

    t_start = time.time()
    if args.w_job and args.a_job:
        w_job, a_job = args.w_job, args.a_job
    else:
        w_job = submit(args, "W", args.w_run_s, array=False, requeue=True)
        a_job = submit(args, "A", args.a_run_s, array=True, requeue=False)
    row("submitted", w_job=w_job, a_job=a_job)
    log.info("W=%s A=%s", w_job, a_job)

    w_running_since = None
    requeued_at = None
    released = False
    fast_until = 0.0
    restart1_seen_at = None
    gone_at = None
    prev_gone = {}
    status = "completed"
    try:
        while True:
            now = time.time()
            if now - t_start > args.max_s:
                status = "deadline"
                break
            if w_running_since is None and now - t_start > args.start_timeout_s:
                status = "start_timeout"
                break
            server = wl.server_now(args.segdir)
            memo = Memo()
            broken_active = (
                restart1_seen_at is not None
                and restart1_seen_at + args.broken_after_s
                <= now
                < restart1_seen_at + args.broken_after_s + args.broken_len_s
            )
            bmemo = Memo(env=dict(os.environ, SLURM_CONF="/nonexistent")) if broken_active else None
            # the two main jobs, same argv shapes the shipping code uses
            for jid in (w_job, a_job):
                for argv in (
                    ["squeue", "-h", "-j", jid, "-o", "%T"],
                    ["sacct", "-j", jid, "-n", "-X", "-P", "-o", "State"],
                ):
                    try:
                        memo(argv, 20)
                    except OSError:
                        pass
            segs = []
            for path in sorted(glob.glob(os.path.join(args.segdir, "*.jsonl"))):
                name = os.path.basename(path)
                try:
                    wid = wl.parse_writer_id(name[: -len(".jsonl")])
                except ValueError:
                    continue
                st_ = os.stat(path)
                rec = {
                    "name": name,
                    "job_id": wid.job_id,
                    "restart": wid.restart,
                    "mtime": st_.st_mtime,
                    "size": st_.st_size,
                    "verdicts": {},
                }
                if wid.job_id == w_job and wid.restart >= 1 and restart1_seen_at is None:
                    restart1_seen_at = now
                kinds = [("live", memo)] + ([("broken", bmemo)] if bmemo else [])
                for kind, runner in kinds:
                    ship = wl.segment_writer_dead(
                        path, grace_s=args.grace_s, now_server=server, runner=runner
                    )
                    nog = wl.segment_writer_dead(path, grace_s=0, now_server=now, runner=runner)
                    trailer = wl._has_trailer(Path(path))
                    gone = False
                    if wid.job_id is not None:
                        try:
                            rc, o, e = runner(["squeue", "-h", "-j", wid.job_id, "-o", "%T"], 20)
                            gone = rc != 0 or not o.strip()
                        except OSError:
                            gone = True
                    key = (name, kind)
                    err_dead = trailer or (
                        wid.job_id is not None and gone and prev_gone.get(key, False)
                    )
                    prev_gone[key] = gone
                    rec["verdicts"][kind] = {
                        "SHIPPING": {"dead": ship.dead, "reason": ship.reason},
                        "NAIVE_ERR": {
                            "dead": bool(err_dead),
                            "reason": "trailer" if trailer else "gone=%s" % gone,
                        },
                        "NAIVE_NOGRACE": {"dead": nog.dead, "reason": nog.reason},
                    }
                segs.append(rec)
            row(
                "sample",
                t=now,
                server_now=server,
                broken_active=broken_active,
                calls=memo.dump(),
                broken_calls=bmemo.dump() if bmemo else [],
                segments=segs,
            )

            w_state = squeue_state(memo.dump(), w_job)
            if w_state == "RUNNING" and w_running_since is None:
                w_running_since = now
            if (
                w_running_since is not None
                and requeued_at is None
                and now - w_running_since >= args.requeue_after_s
            ):
                requeued_at = now
                fast_until = now + 90.0
                r1 = run(["scontrol", "requeue", w_job])
                r2 = run(["scontrol", "hold", w_job])
                row("inject", action="requeue+hold", requeue=r1, hold=r2)
            if requeued_at is not None and not released and now - requeued_at >= args.hold_s:
                released = True
                row("inject", action="release", release=run(["scontrol", "release", w_job]))
            if job_gone(memo.dump(), w_job) and job_gone(memo.dump(), a_job):
                gone_at = gone_at or now
                if now - gone_at >= 2 * args.grace_s:
                    break
            time.sleep(args.fast_sample_s if now < fast_until else args.sample_s)
    finally:
        if status != "completed":
            for j in (w_job, a_job):
                row("cleanup", job=j, scancel=run(["scancel", j]))
        dest = os.path.join(args.out_dir, "segments_%s" % args.tag)
        shutil.copytree(args.segdir, dest, dirs_exist_ok=True)
        for path in sorted(glob.glob(os.path.join(dest, "*.jsonl"))):
            walls, closed = [], False
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("closed") is True:
                        closed = True
                    elif "wall" in r:
                        walls.append(r["wall"])
            row("final", name=os.path.basename(path), walls=walls, closed=closed)
        row("end", status=status, w_job=w_job, a_job=a_job)
        out.close()
    log.info("controller finished: %s", status)
    return 0 if status == "completed" else 3


if __name__ == "__main__":
    sys.exit(main())
