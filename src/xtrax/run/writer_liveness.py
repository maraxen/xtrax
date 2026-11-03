# ruff: noqa: UP031
"""Is a unit-index segment's writer dead? (spec 261030 §3.1, Compact; assumption A10.)

Stdlib only and Python 3.9 compatible, so it can be embedded on compute nodes the way
``fileguard/_node.py`` is. Internal for now: nothing is exported from ``xtrax.run``.

Writer id encoding (segment file stem)::

    <host>-<pid>-<SLURM_JOB_ID>-<SLURM_RESTART_COUNT>-<uuid8>

``host`` may itself contain ``-``, so the id is parsed from the right (four ``-`` splits).
A writer outside SLURM has no job id; it writes the literal ``nojob`` (:data:`NOJOB`) in the job
slot. Such a writer is never judged by the scheduler: its segment is dead only through the
``{"closed": true}`` trailer.

Three-state job lookup (:func:`job_state`):

* ``PRESENT``: ``squeue`` lists the job (any state), or ``sacct`` reports a non-terminal state.
* ``ABSENT``: ``squeue`` says "Invalid job id" AND ``sacct`` positively reports a terminal state.
* ``UNKNOWN``: everything else (command missing, timeout, controller unreachable, unparseable or
  empty output). ``UNKNOWN`` never makes a writer dead.

On Engaging, ``squeue -j <aged-out id>`` exits 1 with "slurm_load_jobs error: Invalid job id
specified" (MinJobAge is 120 s), which a return code alone cannot tell from a broken query.
That is why absence needs a positive ``sacct`` answer.

Dead rule (:func:`segment_writer_dead`): trailer present, OR (job id known AND ``ABSENT`` AND the
segment's mtime is older than ``grace_s`` by NFS *server* time, :func:`server_now`). The default
grace of 300 s is above the NFS attribute-cache ceiling (``acregmax`` 60 s) plus MinJobAge
(120 s).
"""

from __future__ import annotations

import enum
import json
import os
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

NOJOB = "nojob"
DEFAULT_GRACE_S = 300.0
DEFAULT_TIMEOUT_S = 20.0

Runner = Callable[[list[str], float], tuple[int, str, str]]

_TERMINAL = frozenset(
    {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE"}
)
# Known states that mean the job may still run (or be rescheduled).
_NONTERMINAL = frozenset(
    {
        "PENDING",
        "RUNNING",
        "COMPLETING",
        "SUSPENDED",
        "CONFIGURING",
        "PREEMPTED",
        "RESIZING",
        "RESV_DEL_HOLD",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "SIGNALING",
        "STAGE_OUT",
        "STOPPED",
        "SPECIAL_EXIT",
        "STAGE_IN",
        "POWER_UP_NODE",
    }
)
_INVALID_ID = "invalid job id"


class State(enum.Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class JobState(NamedTuple):
    state: State
    raw: str  # scheduler state string, or "" when none was obtained
    detail: str  # which query decided, for logs


class WriterId(NamedTuple):
    host: str
    pid: int
    job_id: str | None  # None when the writer is outside SLURM ("nojob")
    restart: int
    uuid8: str


class Verdict(NamedTuple):
    dead: bool
    reason: str


def parse_writer_id(s: str) -> WriterId:
    """Parse ``<host>-<pid>-<job>-<restart>-<uuid8>`` from the right; ValueError if malformed."""
    parts = s.rsplit("-", 4)
    if len(parts) != 5 or not parts[0]:
        raise ValueError("not a writer id: %r" % (s,))
    host, pid, job, restart, u = parts
    if not (pid.isdigit() and restart.isdigit() and u):
        raise ValueError("not a writer id: %r" % (s,))
    if job == NOJOB:
        job_id: str | None = None
    elif job.isdigit():
        job_id = job
    else:
        raise ValueError("not a writer id (job slot): %r" % (s,))
    return WriterId(host, int(pid), job_id, int(restart), u)


def _subprocess_runner(argv: list[str], timeout: float) -> tuple[int, str, str]:
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    return p.returncode, p.stdout, p.stderr


def _classify(raw: str) -> State | None:
    """PRESENT/ABSENT for a recognised sacct/squeue state word, else None."""
    word = raw.strip().split(" ")[0].rstrip("+")
    if word in _TERMINAL or word.startswith("CANCELLED"):
        return State.ABSENT
    if word in _NONTERMINAL:
        return State.PRESENT
    return None


def job_state(job_id: str, runner: Runner | None = None) -> JobState:
    """Three-state scheduler lookup; see the module docstring for the rules."""
    run = runner or _subprocess_runner
    unknown = lambda why: JobState(State.UNKNOWN, "", why)  # noqa: E731
    try:
        rc, out, err = run(["squeue", "-h", "-j", job_id, "-o", "%T"], DEFAULT_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as e:
        return unknown("squeue unavailable: %s" % type(e).__name__)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if rc == 0 and lines:
        return JobState(State.PRESENT, lines[0], "squeue")
    if rc != 0 and _INVALID_ID not in (err + out).lower():
        return unknown("squeue rc=%d: %s" % (rc, err.strip()[:200]))
    # squeue does not list it: only a positive sacct terminal state counts as absent.
    try:
        rc, out, err = run(
            ["sacct", "-j", job_id, "-n", "-X", "-P", "-o", "State"], DEFAULT_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError) as e:
        return unknown("sacct unavailable: %s" % type(e).__name__)
    if rc != 0:
        return unknown("sacct rc=%d: %s" % (rc, err.strip()[:200]))
    states = [ln.strip() for ln in out.splitlines() if ln.strip()]
    if not states:
        return unknown("sacct returned no rows")
    verdicts = [_classify(s) for s in states]
    if any(v is None for v in verdicts):
        return JobState(State.UNKNOWN, states[0], "sacct unrecognised state")
    if any(v is State.PRESENT for v in verdicts):
        return JobState(State.PRESENT, states[0], "sacct non-terminal")
    return JobState(State.ABSENT, states[0], "sacct terminal")


def server_now(directory: str | Path) -> float:
    """NFS server time: create a probe file in ``directory`` and read its mtime.

    A freshly created file's timestamps are set by the file server, so this is the clock the
    segment mtimes were written with, not the caller's. The probe is removed afterwards.
    """
    p = Path(directory) / (".server_now.%s.probe" % uuid.uuid4().hex[:12])
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.close(fd)
        return os.stat(str(p)).st_mtime
    finally:
        try:
            os.unlink(str(p))
        except OSError:
            pass


def _has_trailer(path: Path) -> bool:
    """True iff the last non-empty line is exactly a ``{"closed": true}`` row."""
    size = path.stat().st_size
    with open(str(path), "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read()
    lines = [ln for ln in tail.split(b"\n") if ln.strip()]
    if not lines:
        return False
    try:
        row = json.loads(lines[-1].decode("utf-8"))
    except ValueError:
        return False
    return isinstance(row, dict) and row.get("closed") is True


def segment_writer_dead(
    segment_path: str | Path,
    *,
    grace_s: int | float = DEFAULT_GRACE_S,
    now_server: float | None = None,
    runner: Runner | None = None,
) -> Verdict:
    """Decide whether a segment's writer is dead. UNKNOWN is never dead."""
    path = Path(segment_path)
    if _has_trailer(path):
        return Verdict(True, "closed trailer")
    try:
        wid = parse_writer_id(path.stem)
    except ValueError:
        return Verdict(False, "unparseable writer id, no trailer")
    if wid.job_id is None:
        return Verdict(False, "no job id, no trailer")
    js = job_state(wid.job_id, runner=runner)
    if js.state is State.PRESENT:
        return Verdict(False, "job present (%s)" % js.raw)
    if js.state is State.UNKNOWN:
        return Verdict(False, "job state unknown (%s)" % js.detail)
    now = now_server if now_server is not None else server_now(path.parent)
    age = now - path.stat().st_mtime
    if age > grace_s:
        return Verdict(
            True, "job absent (%s), mtime age %.1fs > grace %.0fs" % (js.raw, age, grace_s)
        )
    return Verdict(
        False, "job absent (%s) but mtime age %.1fs <= grace %.0fs" % (js.raw, age, grace_s)
    )
