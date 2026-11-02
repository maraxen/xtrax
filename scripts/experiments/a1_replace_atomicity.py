#!/usr/bin/env python3
# ruff: noqa: UP031
"""A1 harness: is ``os.replace`` of a sibling temp file atomic across NFS clients?

Spec .praxia/docs/specs/261030_ledger-pack-fileguard.md, section 4, row A1 (task
261030_xtrax-ledger-pack). This script only RECORDS raw events (JSONL, one file per task). It
produces no finding. The outcome is computed by ``a1_analyze.py`` under ``bth run`` against the
pulled logs, per ``a1_analyze.bth.toml``.

Stdlib only; written for Python 3.9+ (Engaging's default ``python3`` is 3.6.8, see spec A8, so the
wrapper ``a1_replace_atomicity.sh`` selects ``/usr/bin/python3.11``).

Roles (SLURM_PROCID): 0 = writer, >= 1 = readers. One task per node, so readers run on hosts
distinct from the writer. All tasks derive the same phase schedule from ``--start-epoch``.

Per filesystem (``--fs name=dir``) and payload size, in order:

1. CONTROL phase (negative control): the writer rewrites the target IN PLACE (``O_TRUNC``, 64 KiB
   chunks, 1 ms sleeps, no rename). Readers must see torn reads; if they do not, the analysis is
   inconclusive because the instrument cannot see tearing.
2. REAL phase (production write path): temp file in the same directory, write, ``fsync``,
   ``os.replace``, ``fsync`` of the directory. The directory fsync mirrors
   ``fsync_directory`` in ``src/xtrax/run/zarr_integrity.py:258-264`` (open ``O_RDONLY``,
   ``os.fsync``, close).

After the sizes of one filesystem, a visibility block (report only): the writer creates fresh
uniquely named files, readers poll for each name from its slot time and log when it first opens.

Payload: first line ``{"gen":N,"len":L,"sha":sha256(body)}`` then L body bytes derived from N. A
read is TORN when the bytes fail header parse, length, or sha256. Readers open fresh on every read
(close-to-open). Counters: ok, torn, enoent (open), estale_open, estale_read, other. Counters are
aggregated every ``--agg-s`` seconds; the first ``DETAIL_CAP`` non-ok events per (cell, phase) are
also logged individually.

Local smoke: ``uv run --no-sync python scripts/experiments/a1_replace_atomicity.py --smoke``
(1 writer + 1 reader process on this machine, seconds long; run directly, not under bth).
"""

import argparse
import errno
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time

log = logging.getLogger("a1_replace_atomicity")

CHUNK = 64 * 1024
SETUP_S = 4.0
DETAIL_CAP = 200
DEFAULT_SIZES = "4096,262144,4194304,16777216"
VIS_N = 15
VIS_PERIOD_S = 1.0
VIS_TAIL_S = 60.0


# ---- payload ---------------------------------------------------------------------------------


def make_payload(gen, size):
    """Header line + ``size`` body bytes, all determined by ``gen``."""
    body = hashlib.shake_256(("a1-%d" % gen).encode()).digest(size)
    hdr = json.dumps({"gen": gen, "len": size, "sha": hashlib.sha256(body).hexdigest()})
    return hdr.encode() + b"\n" + body


def verify(data):
    """``(ok, reason, gen)`` for the bytes one reader got."""
    nl = data.find(b"\n")
    if nl < 0:
        return False, "no_header" if len(data) else "empty", None
    try:
        hdr = json.loads(data[:nl].decode("utf-8"))
        gen, length, sha = hdr["gen"], hdr["len"], hdr["sha"]
    except (ValueError, KeyError, UnicodeDecodeError):
        return False, "bad_header", None
    body = data[nl + 1 :]
    if len(body) != length:
        return False, "length %d != %d" % (len(body), length), gen
    if hashlib.sha256(body).hexdigest() != sha:
        return False, "sha_mismatch", gen
    return True, "", gen


# ---- logging / time --------------------------------------------------------------------------


class Jsonl:
    def __init__(self, path, task):
        self.f = open(path, "a", buffering=1)
        self.task = task

    def row(self, ev, **kw):
        kw["ev"] = ev
        kw["task"] = self.task
        kw.setdefault("t", time.time())
        self.f.write(json.dumps(kw, sort_keys=True) + "\n")

    def close(self):
        self.f.close()


def wait_until(t):
    while True:
        d = t - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 0.25))


def fsync_directory(path):
    """Mirror of src/xtrax/run/zarr_integrity.py:258-264."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_all(fd, data, sleep_s=0.0):
    mv = memoryview(data)
    off = 0
    while off < len(mv):
        n = os.write(fd, mv[off : off + CHUNK])
        off += n
        if sleep_s:
            time.sleep(sleep_s)


def atomic_write(target, data):
    """The production path: temp sibling, write, fsync, os.replace, fsync directory."""
    tmp = "%s.tmp.%d" % (target, os.getpid())
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, target)
    fsync_directory(os.path.dirname(target))


def inplace_write(target, data):
    """The negative control: truncate and rewrite in place, slowly, no rename."""
    fd = os.open(target, os.O_WRONLY | os.O_TRUNC)
    try:
        time.sleep(0.001)
        write_all(fd, data, sleep_s=0.001)
    finally:
        os.close(fd)


# ---- schedule --------------------------------------------------------------------------------


def build_schedule(start, fs_names, sizes, control_s, real_s):
    """List of ``("cell", fs, size, t0)`` and ``("vis", fs, 0, t0)`` entries, same on every task."""
    out = []
    t = start
    cell_len = SETUP_S + control_s + SETUP_S + real_s
    for fs in fs_names:
        for size in sizes:
            out.append(("cell", fs, size, t))
            t += cell_len
        out.append(("vis", fs, 0, t))
        t += SETUP_S + VIS_N * VIS_PERIOD_S + VIS_TAIL_S
    return out, t


# ---- writer ----------------------------------------------------------------------------------


def writer_cell(lg, d, fs, size, t0, control_s, real_s, agg_s):
    for phase, t_phase, dur in (
        ("control", t0 + SETUP_S, control_s),
        ("real", t0 + 2 * SETUP_S + control_s, real_s),
    ):
        target = os.path.join(d, "target.%d.%s" % (size, phase))
        wait_until(t_phase - SETUP_S + 0.5)
        atomic_write(target, make_payload(0, size))  # initial generation, present for readers
        wait_until(t_phase)
        t_end = t_phase + dur
        lg.row("phase", fs=fs, size=size, phase=phase, edge="start")
        gen, n, last = 0, 0, time.time()
        while time.time() < t_end:
            gen += 1
            data = make_payload(gen, size)
            if phase == "control":
                inplace_write(target, data)
            else:
                atomic_write(target, data)
                time.sleep(0.01)
            n += 1
            if time.time() - last >= agg_s:
                lg.row("wagg", fs=fs, size=size, phase=phase, writes=n, gen=gen)
                n, last = 0, time.time()
        lg.row("wagg", fs=fs, size=size, phase=phase, writes=n, gen=gen)
        lg.row("phase", fs=fs, size=size, phase=phase, edge="end", last_gen=gen)


def writer_vis(lg, d, fs, t0, tag):
    for i in range(VIS_N):
        wait_until(t0 + SETUP_S + i * VIS_PERIOD_S)
        p = os.path.join(d, "vis.%s.%d" % (tag, i))
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, b"x" * 64)
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(d)
        lg.row("vis_create", fs=fs, i=i)


# ---- reader ----------------------------------------------------------------------------------


def read_once(path):
    """``(kind, data_or_reason)``; kind in ok/enoent/estale_open/estale_read/other."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return "enoent", ""
    except OSError as e:
        return ("estale_open" if e.errno == errno.ESTALE else "other"), "open: %r" % (e,)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return "data", b"".join(chunks)
    except OSError as e:
        return ("estale_read" if e.errno == errno.ESTALE else "other"), "read: %r" % (e,)
    finally:
        os.close(fd)


def reader_phase(lg, d, fs, size, phase, t_phase, dur, agg_s):
    target = os.path.join(d, "target.%d.%s" % (size, phase))
    wait_until(t_phase)
    t_end = t_phase + dur
    lg.row("phase", fs=fs, size=size, phase=phase, edge="start")
    c = {"ok": 0, "torn": 0, "enoent": 0, "estale_open": 0, "estale_read": 0, "other": 0}
    detail, last = 0, time.time()
    while True:
        now = time.time()
        if now >= t_end:
            break
        kind, payload = read_once(target)
        if kind == "data":
            ok, reason, gen = verify(payload)
            kind = "ok" if ok else "torn"
            payload = (reason, gen, len(payload))
        c[kind] += 1
        if kind != "ok" and detail < DETAIL_CAP:
            detail += 1
            lg.row("event", fs=fs, size=size, phase=phase, kind=kind, info=repr(payload))
        if now - last >= agg_s:
            lg.row("agg", fs=fs, size=size, phase=phase, **c)
            c = dict.fromkeys(c, 0)
            last = now
    lg.row("agg", fs=fs, size=size, phase=phase, **c)
    lg.row("phase", fs=fs, size=size, phase=phase, edge="end", detail_logged=detail)


def reader_cell(lg, d, fs, size, t0, control_s, real_s, agg_s):
    reader_phase(lg, d, fs, size, "control", t0 + SETUP_S, control_s, agg_s)
    reader_phase(lg, d, fs, size, "real", t0 + 2 * SETUP_S + control_s, real_s, agg_s)


def reader_vis(lg, d, fs, t0, tag):
    """Poll each name from its slot time on; log when it first opens."""
    start = t0 + SETUP_S
    deadline = start + VIS_N * VIS_PERIOD_S + VIS_TAIL_S - 5.0
    wait_until(start)
    seen = set()
    polls = [0] * VIS_N
    while len(seen) < VIS_N and time.time() < deadline:
        now = time.time()
        for i in range(VIS_N):
            if i in seen or now < start + i * VIS_PERIOD_S:
                continue
            polls[i] += 1
            try:
                fd = os.open(os.path.join(d, "vis.%s.%d" % (tag, i)), os.O_RDONLY)
            except OSError:
                continue
            os.close(fd)
            seen.add(i)
            lg.row("vis_seen", fs=fs, i=i, polls=polls[i])
        time.sleep(0.005)
    for i in range(VIS_N):
        if i not in seen:
            lg.row("vis_timeout", fs=fs, i=i, polls=polls[i])


# ---- driver ----------------------------------------------------------------------------------


def run_task(args):
    procid = int(args.procid)
    role = "writer" if procid == 0 else "reader"
    host = socket.gethostname()
    os.makedirs(args.out_dir, exist_ok=True)
    lg = Jsonl(
        os.path.join(args.out_dir, "a1_%s_task%d_%s.jsonl" % (args.tag, procid, host)), procid
    )
    fs_dirs = dict(kv.split("=", 1) for kv in args.fs)
    sizes = [int(s) for s in args.sizes.split(",")]
    lg.row(
        "meta",
        host=host,
        role=role,
        pid=os.getpid(),
        jobid=os.environ.get("SLURM_JOB_ID", ""),
        nodelist=os.environ.get("SLURM_JOB_NODELIST", ""),
        py=sys.version.split()[0],
        fs_dirs=fs_dirs,
        sizes=sizes,
        control_s=args.control_s,
        real_s=args.real_s,
        start_epoch=args.start_epoch,
        tag=args.tag,
    )
    probe = {fs: os.path.join(p, "a1_" + args.tag) for fs, p in fs_dirs.items()}
    if role == "writer":
        for p in probe.values():
            os.makedirs(p, exist_ok=True)
    sched, t_all_end = build_schedule(
        args.start_epoch, list(fs_dirs), sizes, args.control_s, args.real_s
    )
    for kind, fs, size, t0 in sched:
        d = probe[fs]
        if kind == "cell":
            f = writer_cell if role == "writer" else reader_cell
            f(lg, d, fs, size, t0, args.control_s, args.real_s, args.agg_s)
        else:
            f = writer_vis if role == "writer" else reader_vis
            f(lg, d, fs, t0, args.tag)
    lg.row("done")
    lg.close()


def smoke(args):
    """1 writer + 1 reader process on this machine, tiny phases, local tmp dir."""
    work = tempfile.mkdtemp(prefix="a1_smoke_")
    out = os.path.join(work, "logs")
    start = time.time() + 1.5
    common = [
        "--fs",
        "local=" + os.path.join(work, "probe"),
        "--sizes",
        "4096,262144",
        "--control-s",
        str(args.smoke_control_s),
        "--real-s",
        str(args.smoke_real_s),
        "--agg-s",
        "1",
        "--vis-tail-s",
        "8",
        "--start-epoch",
        repr(start),
        "--out-dir",
        out,
        "--tag",
        "smoke",
    ]
    os.makedirs(os.path.join(work, "probe"))
    procs = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__), "--procid", str(i)] + common)
        for i in (0, 1)
    ]
    for p in procs:
        p.wait()
    totals, vis = {}, {}
    for name in sorted(os.listdir(out)):
        with open(os.path.join(out, name)) as f:
            for line in f:
                r = json.loads(line)
                if r["ev"] in ("vis_seen", "vis_timeout"):
                    vis[r["ev"]] = vis.get(r["ev"], 0) + 1
                if r["ev"] == "agg":
                    k = (r["size"], r["phase"])
                    t = totals.setdefault(k, {"ok": 0, "torn": 0, "enoent": 0})
                    for key in t:
                        t[key] += r[key]
    print("smoke dir:", work, "vis:", vis)
    for k, v in sorted(totals.items()):
        print("size=%d phase=%s %s" % (k[0], k[1], v))
    control_fired = all(v["torn"] > 0 for k, v in totals.items() if k[1] == "control")
    real_torn = sum(v["torn"] for k, v in totals.items() if k[1] == "real")
    print("control_fired=%s real_torn=%d" % (control_fired, real_torn))
    return 0 if control_fired and real_torn == 0 else 1


def main():
    global VIS_TAIL_S
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--procid", default=os.environ.get("SLURM_PROCID", "0"))
    ap.add_argument("--fs", action="append", default=[], help="name=dir, repeatable")
    ap.add_argument("--sizes", default=DEFAULT_SIZES)
    ap.add_argument("--control-s", type=float, default=30.0)
    ap.add_argument("--real-s", type=float, default=180.0)
    ap.add_argument("--agg-s", type=float, default=5.0)
    ap.add_argument("--start-epoch", type=float, default=0.0)
    ap.add_argument("--out-dir", default="outputs/probes/a1/raw")
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local"))
    ap.add_argument("--vis-tail-s", type=float, default=VIS_TAIL_S)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-control-s", type=float, default=3.0)
    ap.add_argument("--smoke-real-s", type=float, default=3.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    VIS_TAIL_S = args.vis_tail_s
    if args.smoke:
        sys.exit(smoke(args))
    if not args.fs or not args.start_epoch:
        ap.error("--fs and --start-epoch are required")
    run_task(args)


if __name__ == "__main__":
    main()
