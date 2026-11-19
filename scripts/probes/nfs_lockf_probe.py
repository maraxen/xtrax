"""Spike (throwaway, spec 261030 §3.3): do POSIX record locks work, and exclude across nodes?

Run as a 2-task SLURM array. Each task, for each directory given, opens ``<dir>/.lockf_probe``,
tries ``fcntl.lockf(LOCK_EX | LOCK_NB)``, and if it gets the lock holds it for ``--hold`` s. Task 1
starts ``--stagger`` s after task 0, so if locks exclude across nodes task 1 sees EAGAIN/EACCES
while task 0 holds. Prints one JSON line per directory. Python 3.9+, stdlib only.
"""

import argparse
import errno
import fcntl
import json
import os
import socket
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--hold", type=float, default=40.0)
    ap.add_argument("--stagger", type=float, default=15.0)
    args = ap.parse_args()
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if task:
        time.sleep(args.stagger)
    held = []
    for d in args.dirs:
        d = os.path.expanduser(d)
        row = {"task": task, "host": socket.gethostname(), "dir": d, "t": time.time()}
        try:
            st = os.statvfs(d)
            row["fs_id"] = st.f_fsid
            path = os.path.join(d, ".lockf_probe")
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                row["result"] = "acquired"
                held.append(fd)
            except OSError as exc:
                row["result"] = "busy" if exc.errno in (errno.EAGAIN, errno.EACCES) else "error"
                row["errno"] = errno.errorcode.get(exc.errno, str(exc.errno))
                os.close(fd)
        except OSError as exc:
            row["result"] = "error"
            row["errno"] = errno.errorcode.get(exc.errno or 0, str(exc))
        with open("/proc/mounts") as fh:
            best = ""
            for line in fh:
                parts = line.split()
                if d.startswith(parts[1]) and len(parts[1]) > len(best):
                    best, row["fstype"], row["mount_opts"] = parts[1], parts[2], parts[3]
            row["mount"] = best
        print(json.dumps(row), flush=True)
    time.sleep(args.hold)
    for fd in held:
        os.close(fd)


if __name__ == "__main__":
    main()
