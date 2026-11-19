#!/usr/bin/env python3
# ruff: noqa: UP031
"""A10 writer: a stand-in unit-index writer for the requeue-injection probe.

Stdlib only, Python 3.9+. Mirrors the writer side of spec 261030 section 3.1 (Segments):
``<segdir>/<writer_id>.jsonl`` with ``writer_id = <host>-<pid>-<SLURM_JOB_ID>-<SLURM_RESTART_COUNT>-
<uuid8>`` (``nojob`` when outside SLURM), one heartbeat row every ``--hb-s`` seconds, ``fsync`` per
row, ``fsync`` of the directory once at creation, and a ``{"closed": true}`` trailer on a CLEAN
exit after ``--duration-s``.

SIGTERM exits at once WITHOUT the trailer: that simulates a preemption or requeue kill, which is
exactly the case "writer is dead, no trailer" that compaction must classify correctly.
"""

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
import uuid

log = logging.getLogger("a10_writer")


def writer_id():
    return "%s-%d-%s-%s-%s" % (
        socket.gethostname(),
        os.getpid(),
        os.environ.get("SLURM_JOB_ID", "nojob"),
        os.environ.get("SLURM_RESTART_COUNT", "0"),
        uuid.uuid4().hex[:8],
    )


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append(fd, row):
    os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode())
    os.fsync(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segdir", required=True)
    ap.add_argument("--duration-s", type=float, default=240.0)
    ap.add_argument("--hb-s", type=float, default=2.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    os.makedirs(args.segdir, exist_ok=True)
    wid = writer_id()
    path = os.path.join(args.segdir, wid + ".jsonl")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o644)
    fsync_dir(args.segdir)
    log.info("writer %s start, duration %.0fs", wid, args.duration_s)

    def on_term(signum, frame):  # preemption/requeue kill: no trailer, no flush
        log.info("writer %s got signal %d, exiting without trailer", wid, signum)
        os._exit(143)

    signal.signal(signal.SIGTERM, on_term)
    t_end = time.monotonic() + args.duration_s
    seq = 0
    while time.monotonic() < t_end:
        append(fd, {"seq": seq, "wall": time.time(), "mono": time.monotonic(), "writer_id": wid})
        seq += 1
        time.sleep(args.hb_s)
    append(fd, {"closed": True})
    os.close(fd)
    log.info("writer %s closed cleanly after %d rows", wid, seq)


if __name__ == "__main__":
    sys.exit(main())
