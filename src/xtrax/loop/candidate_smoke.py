"""Candidate-smoke (T2-14, #2181, AC-4, F4).

AC-4's claim: a structurally-valid candidate (T2-13) must pass an L1 dry-run + L2 CPU smoke test,
under the pinned uv lockfile, both exiting 0 in under 60s -- reject pre-budget, sanitized failure
summary, no GPU/cluster submission. Unlike T2-11/T2-12/T2-13 (which all import-and-call a
candidate in-process), this gate needs a genuinely SEPARATE process: a real PID for T2-09's
watchdog to police, and an isolated environment `uv run --locked` can pin to the exact locked
dependency set. No existing dry-run/smoke-test pattern or `uv run --locked` precedent exists
anywhere in this repo -- this module is new infrastructure, not glue around something already
built (contrast T2-12/T2-13).

Reconciling the two candidate shapes: rather than inventing a second, incompatible contract for
"subprocess-runnable candidates," this module generates a small bootstrap script (a Python source
string written to a tempfile at call time -- never a packaged repo asset, avoiding the
"non-.py data file needs an explicit force-include entry or it silently won't ship" pitfall this
project has hit before) that itself calls `xtrax.loop.schema_gate.resolve_candidate_callable` --
the exact same resolution T2-11/T2-12/T2-13 use -- just inside the child process. Candidate authors
see one contract, not two.

L1 ("dry-run"): the bootstrap resolves the candidate and exits 0 without calling it -- proving the
candidate imports cleanly under the pinned lockfile, in a fresh interpreter. This is meaningfully
different from T2-11's own clean-import check, which runs in the orchestrator's already-warm
process and can't catch an issue specific to a cold start under the exact locked dependency set.
L2 ("CPU smoke"): the bootstrap additionally loads concrete inputs (serialized to `.npz` by the
parent) and actually calls the candidate once, CPU-only (`JAX_PLATFORMS=cpu` +
`CUDA_VISIBLE_DEVICES=""` on the subprocess environment).

Budget: ONE combined wall-clock deadline spans both phases (not two separate per-phase budgets) --
computed once at entry, each phase gets whatever time remains. Enforced via T2-09's
`xtrax.loop.external_stop_watchdog` (`start_watchdog`/`WatchdogCriteria`) directly -- a real
composition dependency, matching the DAG's own `T2-14 depends_on: [T2-09]` edge -- rather than
`subprocess.run(timeout=...)`: the watchdog's SIGKILL is what T2-09 was built to guarantee even a
wedged/SIGTERM-ignoring child can't survive, and this repo has no other timeout-enforcement idiom
to prefer instead.

Process-boundary tradeoff, documented rather than silently accepted: a candidate-resolution
failure (bad import, missing callable) now surfaces as a nonzero subprocess exit + stderr text,
not as a typed `CandidateResolutionError` the parent can catch -- the real exception object can't
cross a process boundary. `CandidateSmokeFailedError`'s message carries a truncated stderr tail
(mirroring `xtrax.devtools.empirical_oracle.run_pytest_repro`'s existing tail-2000-chars
convention, the only prior "sanitize subprocess output" precedent in this repo) so the failure
reason is still visible, just not typed.

Extension seam (mirrors every other T2-0X module's stance): this module raises and stops. It does
not decide what a future loop controller does after a `CandidateSmokeTimeoutError`/
`CandidateSmokeFailedError` (retry with a different candidate? escalate? log to a campaign ledger?)
-- no #2181 loop controller exists yet to validate that design against.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from xtrax.loop.external_stop_watchdog import WatchdogCriteria, start_watchdog

ROOT = Path(__file__).resolve().parents[3]

_OUTPUT_TRUNCATE_CHARS = 2000

_BOOTSTRAP_SOURCE = """\
import sys
from pathlib import Path

import numpy as np

from xtrax.loop.schema_gate import resolve_candidate_callable

resolve_only = sys.argv[1] == "--resolve-only"
candidate_path = Path(sys.argv[2])
callable_name = sys.argv[3]

fn = resolve_candidate_callable(candidate_path, callable_name)

if resolve_only:
    sys.exit(0)

inputs_path = sys.argv[4]
with np.load(inputs_path) as npz:
    inputs = [npz[key] for key in sorted(npz.files, key=lambda k: int(k.removeprefix("arr_")))]

fn(*inputs)
sys.exit(0)
"""


class CandidateSmokeError(Exception):
    """Base for AC-4's candidate-smoke rejection errors. Reject; no GPU/cluster submission."""


class CandidateSmokeTimeoutError(CandidateSmokeError):
    """The combined L1+L2 wall-clock budget was exhausted (pre-budget, or watchdog SIGKILL)."""


class CandidateSmokeFailedError(CandidateSmokeError):
    """A phase exited nonzero for a reason other than the watchdog -- a genuine crash."""


def _cpu_only_env() -> dict[str, str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _write_bootstrap(tmp_dir: Path) -> Path:
    path = tmp_dir / "_candidate_smoke_bootstrap.py"
    path.write_text(_BOOTSTRAP_SOURCE, encoding="utf-8")
    return path


def _write_concrete_inputs(concrete_inputs: list[Any], tmp_dir: Path) -> Path:
    path = tmp_dir / "_candidate_smoke_inputs.npz"
    # Positional args (not **kwargs unpacking) -- numpy's own default "arr_N" naming, in call
    # order. Avoids a ty false positive on np.savez's **kwds: ArrayLike overload when passed a
    # dict-comprehension unpack (ty can't rule out an `allow_pickle` collision at the type level).
    np.savez(path, *[np.asarray(x) for x in concrete_inputs])
    return path


def _truncate(text: str) -> str:
    return text[-_OUTPUT_TRUNCATE_CHARS:] if len(text) > _OUTPUT_TRUNCATE_CHARS else text


def _run_phase_under_deadline(
    cmd: list[str],
    *,
    deadline: float,
    env: dict[str, str],
    root: Path,
    phase_name: str,
    poll_interval_seconds: float,
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        msg = f"{phase_name}: combined wall-clock budget exhausted before this phase could start"
        raise CandidateSmokeTimeoutError(msg)

    # `uv run` spawns the actual python interpreter as a CHILD process rather than exec-replacing
    # itself (confirmed empirically: the two have distinct PIDs) -- SIGKILLing only `proc.pid`
    # leaves that grandchild running, still holding the stdout/stderr pipes open, which hangs
    # `communicate()` until the grandchild exits on its own (reproduced directly: a 10s-sleeping,
    # SIGTERM-ignoring grandchild made `communicate()` block for ~9s after killing just the `uv`
    # PID). Fix: launch in a new session (`start_new_session=True`, making `proc.pid` the process
    # GROUP id too) and target the watchdog at `-proc.pid` -- POSIX `kill`/`os.kill` treat a
    # negative pid as "signal the whole process group," reliably taking out `uv` and its
    # grandchild together. Verified empirically: this drops `communicate()`'s post-kill wait from
    # ~9s to ~2ms for the same hanging-grandchild scenario. No change needed to T2-09's watchdog
    # itself -- its own `_process_alive`/`_hard_kill` just forward whatever pid they're given.
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    handle = start_watchdog(
        -proc.pid,
        WatchdogCriteria(
            wall_clock_budget_seconds=remaining, poll_interval_seconds=poll_interval_seconds
        ),
    )
    stdout, stderr = proc.communicate()
    if handle.is_alive():
        handle.stop()

    if proc.returncode == 0:
        return
    if proc.returncode == -9:
        msg = f"{phase_name}: exceeded the combined {remaining:.1f}s remaining budget, SIGKILLed"
        raise CandidateSmokeTimeoutError(msg)

    combined_output = _truncate(f"{stdout}{stderr}".strip())
    msg = f"{phase_name}: exited {proc.returncode}: {combined_output}"
    raise CandidateSmokeFailedError(msg)


def assert_candidate_smoke(
    candidate_path: Path,
    callable_name: str,
    *,
    concrete_inputs: list[Any],
    wall_clock_budget_seconds: float = 60.0,
    poll_interval_seconds: float = 0.5,
    root: Path | None = None,
) -> None:
    """The composed AC-4 gate: L1 dry-run + L2 CPU smoke, under one combined wall-clock budget.

    Args:
        candidate_path: source file to resolve (same contract as T2-11/T2-12/T2-13).
        callable_name: attribute on the resolved module to call for L2.
        concrete_inputs: real (CPU) arguments for L2's single execution.
        wall_clock_budget_seconds: combined budget spanning both L1 and L2. `<= 0` rejects
            pre-budget with no subprocess spawned at all.
        poll_interval_seconds: passed through to `WatchdogCriteria` for both phases.
        root: working directory for `uv run --locked` (defaults to the repo root).

    Raises:
        CandidateSmokeTimeoutError: the combined budget was exhausted before or during either
            phase (watchdog SIGKILL, or already-exhausted budget before a phase could start).
        CandidateSmokeFailedError: a phase exited nonzero for a reason other than the watchdog --
            message carries a truncated stdout+stderr tail.
    """
    deadline = time.monotonic() + wall_clock_budget_seconds
    if wall_clock_budget_seconds <= 0:
        msg = "candidate-smoke: wall_clock_budget_seconds <= 0, rejecting pre-budget"
        raise CandidateSmokeTimeoutError(msg)

    resolved_root = root or ROOT
    env = _cpu_only_env()

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        bootstrap_path = _write_bootstrap(tmp_dir)
        inputs_path = _write_concrete_inputs(concrete_inputs, tmp_dir)

        _run_phase_under_deadline(
            [
                "uv",
                "run",
                "--locked",
                "python",
                str(bootstrap_path),
                "--resolve-only",
                str(candidate_path),
                callable_name,
            ],
            deadline=deadline,
            env=env,
            root=resolved_root,
            phase_name="L1 dry-run",
            poll_interval_seconds=poll_interval_seconds,
        )

        _run_phase_under_deadline(
            [
                "uv",
                "run",
                "--locked",
                "python",
                str(bootstrap_path),
                "--run",
                str(candidate_path),
                callable_name,
                str(inputs_path),
            ],
            deadline=deadline,
            env=env,
            root=resolved_root,
            phase_name="L2 CPU smoke",
            poll_interval_seconds=poll_interval_seconds,
        )


__all__ = [
    "CandidateSmokeError",
    "CandidateSmokeFailedError",
    "CandidateSmokeTimeoutError",
    "assert_candidate_smoke",
]
