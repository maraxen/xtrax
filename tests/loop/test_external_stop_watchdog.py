"""Tests for the external-stop watchdog (T2-09, #2181, AC-13, F9, PM-7)."""

import dataclasses
import json
import os
import subprocess
import sys
import time

import pytest

from xtrax.loop.external_stop_watchdog import (
    WatchdogAlreadyStoppedError,
    WatchdogCriteria,
    start_watchdog,
)

_SLEEP_LONG = [sys.executable, "-c", "import time; time.sleep(30)"]
_SLEEP_SHORT = [sys.executable, "-c", "import time; time.sleep(0.2)"]
_IGNORE_SIGTERM_AND_SLEEP = [
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
]

_JOIN_TIMEOUT_S = 10.0


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestWatchdogCriteria:
    def test_is_frozen(self) -> None:
        criteria = WatchdogCriteria(wall_clock_budget_seconds=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            criteria.wall_clock_budget_seconds = 2.0  # type: ignore[misc]


class TestStartWatchdog:
    def test_hard_kills_target_after_budget_expires(self) -> None:
        target = subprocess.Popen(_SLEEP_LONG)
        try:
            handle = start_watchdog(
                target.pid,
                WatchdogCriteria(wall_clock_budget_seconds=0.3, poll_interval_seconds=0.05),
            )
            handle.join(timeout=_JOIN_TIMEOUT_S)
            assert not handle.is_alive()

            returncode = target.wait(timeout=_JOIN_TIMEOUT_S)
            # POSIX: a process killed by signal N reports returncode -N.
            assert returncode == -9
        finally:
            _terminate(target)

    def test_survives_a_target_that_ignores_sigterm(self) -> None:
        """PM-7's actual claim: a wedged/uncooperative target cannot wedge the watchdog."""
        target = subprocess.Popen(_IGNORE_SIGTERM_AND_SLEEP)
        try:
            time.sleep(0.1)  # let the target install its SIGTERM handler
            handle = start_watchdog(
                target.pid,
                WatchdogCriteria(wall_clock_budget_seconds=0.3, poll_interval_seconds=0.05),
            )
            handle.join(timeout=_JOIN_TIMEOUT_S)
            assert not handle.is_alive()

            returncode = target.wait(timeout=_JOIN_TIMEOUT_S)
            assert returncode == -9  # SIGKILL cannot be ignored, unlike SIGTERM
        finally:
            _terminate(target)

    def test_does_not_kill_a_target_that_exits_before_the_budget(self) -> None:
        target = subprocess.Popen(_SLEEP_SHORT)
        try:
            handle = start_watchdog(
                target.pid,
                WatchdogCriteria(wall_clock_budget_seconds=5.0, poll_interval_seconds=0.05),
            )
            returncode = target.wait(timeout=_JOIN_TIMEOUT_S)
            assert returncode == 0

            handle.join(timeout=_JOIN_TIMEOUT_S)
            assert not handle.is_alive()
        finally:
            _terminate(target)

    def test_stop_terminates_the_watchdog_not_the_target(self) -> None:
        target = subprocess.Popen(_SLEEP_LONG)
        try:
            handle = start_watchdog(
                target.pid,
                WatchdogCriteria(wall_clock_budget_seconds=30.0, poll_interval_seconds=0.05),
            )
            assert handle.is_alive()
            handle.stop()
            assert not handle.is_alive()
            assert target.poll() is None  # target is untouched by stop()
        finally:
            _terminate(target)

    def test_stop_raises_once_watchdog_already_exited(self) -> None:
        target = subprocess.Popen(_SLEEP_SHORT)
        try:
            handle = start_watchdog(
                target.pid,
                WatchdogCriteria(wall_clock_budget_seconds=0.3, poll_interval_seconds=0.05),
            )
            handle.join(timeout=_JOIN_TIMEOUT_S)
            assert not handle.is_alive()

            with pytest.raises(WatchdogAlreadyStoppedError, match=str(target.pid)):
                handle.stop()
        finally:
            _terminate(target)

    def test_watchdog_survives_the_launching_process_exiting(self) -> None:
        """Regression test (260715 audit): the watchdog must outlive its launcher.

        A `multiprocessing.Process(daemon=True)` watchdog would be torn down by the launcher's
        own `atexit` handling the moment the launcher process exits -- even via an unhandled
        exception -- which silently revokes AC-13's "kill authority unrevokable by the loop".
        This spawns a short-lived "launcher" subprocess that starts a watchdog against an
        orphaned target and then exits immediately, without ever calling stop()/join(). The
        watchdog must still hard-kill the target afterward, entirely on its own.
        """
        launcher_script = (
            "import json, subprocess, sys\n"
            "from xtrax.loop.external_stop_watchdog import WatchdogCriteria, start_watchdog\n"
            "target = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
            "start_watchdog(\n"
            "    target.pid,\n"
            "    WatchdogCriteria(wall_clock_budget_seconds=0.4, poll_interval_seconds=0.05),\n"
            ")\n"
            "print(json.dumps({'target_pid': target.pid}))\n"
            # No stop()/join() -- the launcher exits here, immediately.
        )
        launcher = subprocess.run(
            [sys.executable, "-c", launcher_script],
            capture_output=True,
            text=True,
            timeout=_JOIN_TIMEOUT_S,
        )
        assert launcher.returncode == 0, launcher.stderr
        target_pid = json.loads(launcher.stdout)["target_pid"]

        try:
            deadline = time.monotonic() + _JOIN_TIMEOUT_S
            while time.monotonic() < deadline and _pid_alive(target_pid):
                time.sleep(0.05)
            assert not _pid_alive(target_pid), (
                "orphaned target outlived its watchdog's budget -- the watchdog did not "
                "survive its launcher exiting"
            )
        finally:
            if _pid_alive(target_pid):
                os.kill(target_pid, 9)
