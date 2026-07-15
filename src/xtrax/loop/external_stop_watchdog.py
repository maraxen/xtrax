"""External-stop watchdog (T2-09, #2181, AC-13, F9, PM-7).

PM-7's failure mode (`.praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md`):
F9's watchdog was an in-process Python driver -- "termination outside agent context" was
nominally true (the loop's own iteration code never decided when to stop) but a hang or OOM in
a candidate wedged the controller too, because it shared a process -- and a GIL -- with the
very thing it was meant to police. This module is the fix: `start_watchdog` launches a
genuinely SEPARATE, DETACHED OS process (`python -m xtrax.loop.external_stop_watchdog`, via
`subprocess.Popen(..., start_new_session=True)`) that polls a target PID's wall-clock budget and
SIGKILLs it on expiry, entirely independent of whatever state the target process -- or the
process that launched the watchdog -- is in.

Design-history note (260715 audit): the first implementation here used
`multiprocessing.Process(daemon=True)` for the "separate process" requirement. That was wrong in
a way only visible under a specific failure path: Python's `multiprocessing.util._exit_function`
(registered via `atexit`) tears down every daemonic child whenever the LAUNCHING process exits
through any shutdown path, including an uncaught exception -- verified empirically during
review. That reintroduced a variant of PM-7's exact failure, just via lifecycle coupling instead
of shared-process/GIL coupling: the loop process could revoke the watchdog's kill authority
simply by exiting, even accidentally, which directly contradicts AC-13's "kill authority
unrevokable by the loop." `subprocess.Popen(start_new_session=True)` has no such coupling --
the subprocess module registers no `atexit` hook to reap children, and `start_new_session=True`
detaches the watchdog into its own session (new process group, no controlling terminal), so it
keeps running regardless of what happens to -- or inside -- the process that launched it. As a
side effect this also isolates the watchdog from the target's process group: a candidate that
tried to `killpg` its own process group to take the watchdog down with it cannot reach a
watchdog living in a different session.

Kill-authority scope: this module owns exactly the "hard-kill on wall-clock budget exhaustion"
half of AC-13. It does NOT track or enforce a *compute* budget (CPU/GPU-hours) -- AC-13 names
both wall-clock and compute budget, and only the former is implemented here; a compute-budget
watchdog is a distinct, not-yet-built primitive. It does NOT decide what "convergence reached"
means either -- that is loop-controller-specific and no #2181 loop controller exists yet (same
"no consumer to validate a stronger contract against" stance
`xtrax.loop.admission`/`closure_lock`/`evaluator_completeness` already took); a caller wanting a
convergence-triggered stop calls `WatchdogHandle.stop()` itself once its own convergence check
fires. It also does NOT implement git-as-memory crash-atomicity (AC-13's "lineage preserved via
git" success clause) -- that is T2-10's ratchet-crash-atomicity gate, a separate DAG item; a
hard-killed target's git state staying recoverable is a property of T2-10's
tmp-commit+fsync+atomic-ref-update discipline, not of this module's kill mechanism.

Known limitation (PID reuse): `_watchdog_run`'s final kill on budget expiry checks only
`time.monotonic()` against the budget, with no process-start-time/identity re-verification of
`target_pid` immediately beforehand. In the (narrow) window where `target_pid` exits and the OS
recycles that PID for an unrelated process within one `poll_interval_seconds`, the watchdog
could SIGKILL the wrong process. Mitigated by keeping `poll_interval_seconds` small relative to
typical process churn; not eliminated. A stronger fix (e.g. comparing `/proc/<pid>/stat` start
time) is Linux-specific and deferred as a documented gap rather than added speculatively.

`WatchdogCriteria` is a frozen dataclass, constructed once by whatever process launches the
loop (human or orchestrator) and handed to `start_watchdog`. Nothing here exposes a way to
mutate an already-running watchdog's budget, or to reach from the loop process into the
watchdog's separate process to edit it -- which is what AC-13's "termination criteria never
visible/editable to the agent" requires: the agent operates inside the loop process, and the
loop process has no handle back into the watchdog process's own memory.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


class WatchdogError(Exception):
    """Base for AC-13 watchdog failures."""


class WatchdogAlreadyStoppedError(WatchdogError):
    """`WatchdogHandle.stop()` was called after the watchdog process already exited on its own."""


@dataclass(frozen=True, slots=True)
class WatchdogCriteria:
    """Immutable kill criteria for one watchdog: a wall-clock budget plus a poll cadence.

    Frozen so a value handed to `start_watchdog` cannot be mutated afterward by whatever holds
    a reference to it -- the loop process included.
    """

    wall_clock_budget_seconds: float
    poll_interval_seconds: float = 0.5


def _process_alive(pid: int) -> bool:
    """True iff `pid` still exists, via a zero-signal probe (does not require kill permission)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _hard_kill(pid: int) -> None:
    """SIGKILL -- AC-13 requires a hard-kill a hung/wedged target cannot ignore or catch."""
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _watchdog_run(
    target_pid: int,
    budget_seconds: float,
    poll_interval_seconds: float,
    started_at: float,
) -> None:
    """The SEPARATE, DETACHED watchdog process's own loop body (see `start_watchdog`).

    Polls only two things: elapsed wall clock (`time.monotonic`) and `target_pid` liveness via
    a zero-signal probe -- never anything the monitored loop/candidate could write to signal
    "I'm fine", which would let a wedged or compromised target suppress its own kill.
    """
    while True:
        if time.monotonic() - started_at >= budget_seconds:
            _hard_kill(target_pid)
            return
        if not _process_alive(target_pid):
            return
        time.sleep(poll_interval_seconds)


def _watchdog_main() -> None:
    """Process entry point for `python -m xtrax.loop.external_stop_watchdog` (see `start_watchdog`).

    This process *is* the watchdog -- there is nothing further to spawn from here. The wall
    clock starts now, at this process's own launch, rather than being computed by the launcher
    and passed across the process boundary; that keeps this entry point fully self-contained
    and avoids depending on the parent's and this process's `time.monotonic()` epochs agreeing
    (true on the POSIX platforms this module targets, but not worth relying on when avoiding it
    costs nothing).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--poll-interval-seconds", type=float, required=True)
    args = parser.parse_args()
    _watchdog_run(
        args.target_pid, args.budget_seconds, args.poll_interval_seconds, time.monotonic()
    )


@dataclass
class WatchdogHandle:
    """A handle to a running, detached watchdog process.

    Deliberately narrow: `is_alive`/`join` observe; `stop` is best-effort cleanup of the
    watchdog process itself once the caller's own convergence check fires -- it is not a way to
    revoke the watchdog's kill authority over `target_pid` mid-flight. Once started, the
    watchdog runs to completion on its own schedule in its own detached session regardless of
    what the launching process does next -- including exiting entirely -- short of something
    explicitly killing the watchdog process itself.
    """

    process: subprocess.Popen
    target_pid: int

    def is_alive(self) -> bool:
        """Whether the watchdog process is still running (has not yet killed or exited)."""
        return self.process.poll() is None

    def join(self, timeout: float | None = None) -> None:
        """Block until the watchdog process exits (budget expired or target exited first)."""
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def stop(self) -> None:
        """Terminate the watchdog process itself (not `target_pid`) once no longer needed.

        Raises:
            WatchdogAlreadyStoppedError: the watchdog process already exited on its own
                (budget expired and it hard-killed the target, or it observed the target exit
                first) -- there is nothing left to stop.
        """
        if self.process.poll() is not None:
            msg = f"watchdog for target_pid={self.target_pid} already stopped"
            raise WatchdogAlreadyStoppedError(msg)
        self.process.terminate()
        self.process.wait()


def start_watchdog(target_pid: int, criteria: WatchdogCriteria) -> WatchdogHandle:
    """Launch a SEPARATE, DETACHED OS process that hard-kills `target_pid` once the budget elapses.

    Uses `subprocess.Popen(..., start_new_session=True)` running this module's own
    `python -m xtrax.loop.external_stop_watchdog` entry point (`_watchdog_main`) -- see the
    module docstring's design-history note for why this replaced an earlier
    `multiprocessing.Process(daemon=True)` implementation that had a lifecycle-coupling flaw.
    Output streams are discarded (`DEVNULL`): the watchdog has nothing to report and no console
    to report it to once detached.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xtrax.loop.external_stop_watchdog",
            "--target-pid",
            str(target_pid),
            "--budget-seconds",
            str(criteria.wall_clock_budget_seconds),
            "--poll-interval-seconds",
            str(criteria.poll_interval_seconds),
        ],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return WatchdogHandle(process=process, target_pid=target_pid)


__all__ = [
    "WatchdogAlreadyStoppedError",
    "WatchdogCriteria",
    "WatchdogError",
    "WatchdogHandle",
    "start_watchdog",
]


if __name__ == "__main__":
    _watchdog_main()
