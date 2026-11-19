# ruff: noqa: UP031
"""Tests for xtrax.run.writer_liveness: three-state job lookup and the segment dead-writer rule."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from xtrax.run.writer_liveness import (
    NOJOB,
    State,
    job_state,
    parse_writer_id,
    segment_writer_dead,
    server_now,
)

Res = tuple[int, str, str]
INVALID = "slurm_load_jobs error: Invalid job id specified"


def fake(squeue: Res, sacct: Res = (0, "", "")) -> Callable[[list[str], float], Res]:
    """Runner answering squeue/sacct with fixed results; any other command is missing."""

    def run(argv: list[str], timeout: float) -> Res:
        if argv[0] == "squeue":
            return squeue
        if argv[0] == "sacct":
            return sacct
        raise FileNotFoundError(argv[0])

    return run


def raising(exc: BaseException) -> Callable[[list[str], float], Res]:
    def run(argv: list[str], timeout: float) -> Res:
        raise exc

    return run


# ---- parse_writer_id -------------------------------------------------------


def test_parse_simple() -> None:
    w = parse_writer_id("node1600-4242-27692525-1-deadbeef")
    assert (w.host, w.pid, w.job_id, w.restart, w.uuid8) == (
        "node1600",
        4242,
        "27692525",
        1,
        "deadbeef",
    )


def test_parse_host_with_dashes() -> None:
    w = parse_writer_id("my-gpu-node-07-99-123-0-0a1b2c3d")
    assert w.host == "my-gpu-node-07"
    assert w.job_id == "123"


def test_parse_nojob() -> None:
    w = parse_writer_id("laptop-12-%s-0-aaaaaaaa" % NOJOB)
    assert w.job_id is None
    assert w.host == "laptop"


@pytest.mark.parametrize(
    "bad", ["", "a-b", "h-xx-1-0-aaaaaaaa", "h-1-j1-0-aaaaaaaa", "h-1-1-x-aaaaaaaa"]
)
def test_parse_malformed_raises(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_writer_id(bad)


# ---- job_state -------------------------------------------------------------


@pytest.mark.parametrize(
    "st", ["PENDING", "RUNNING", "COMPLETING", "SUSPENDED", "REQUEUED", "CONFIGURING"]
)
def test_squeue_lists_job_is_present(st: str) -> None:
    js = job_state("1", runner=fake((0, st + "\n", "")))
    assert js.state is State.PRESENT and js.raw == st


def test_invalid_id_and_sacct_terminal_is_absent() -> None:
    js = job_state("1", runner=fake((1, "", INVALID), (0, "COMPLETED\n", "")))
    assert js.state is State.ABSENT and js.raw == "COMPLETED"


@pytest.mark.parametrize(
    "st",
    [
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "CANCELLED by 1000",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "BOOT_FAIL",
        "DEADLINE",
    ],
)
def test_terminal_states_are_absent(st: str) -> None:
    assert job_state("1", runner=fake((1, "", INVALID), (0, st + "\n", ""))).state is State.ABSENT


@pytest.mark.parametrize(
    "st", ["PREEMPTED", "REQUEUED", "REQUEUE_HOLD", "RESIZING", "PENDING", "RUNNING"]
)
def test_nonterminal_sacct_is_present(st: str) -> None:
    assert job_state("1", runner=fake((1, "", INVALID), (0, st + "\n", ""))).state is State.PRESENT


def test_invalid_id_and_sacct_failing_is_unknown() -> None:
    js = job_state("1", runner=fake((1, "", INVALID), (1, "", "sacct: error: slurmdbd down")))
    assert js.state is State.UNKNOWN


def test_invalid_id_and_sacct_empty_is_unknown() -> None:
    assert job_state("1", runner=fake((1, "", INVALID), (0, "", ""))).state is State.UNKNOWN


def test_invalid_id_and_sacct_gibberish_is_unknown() -> None:
    assert (
        job_state("1", runner=fake((1, "", INVALID), (0, "WHATEVER\n", ""))).state is State.UNKNOWN
    )


def test_controller_error_is_unknown() -> None:
    err = "squeue: error: Unable to contact slurm controller (connect failure)"
    js = job_state("1", runner=fake((1, "", err), (0, "COMPLETED\n", "")))
    assert js.state is State.UNKNOWN  # sacct is NOT consulted: rc != 0 without the invalid-id text


def test_missing_command_is_unknown() -> None:
    assert job_state("1", runner=raising(FileNotFoundError("squeue"))).state is State.UNKNOWN


def test_timeout_is_unknown() -> None:
    exc = subprocess.TimeoutExpired("squeue", 1)
    assert job_state("1", runner=raising(exc)).state is State.UNKNOWN


def test_sacct_timeout_is_unknown() -> None:
    def run(argv: list[str], timeout: float) -> Res:
        if argv[0] == "squeue":
            return (1, "", INVALID)
        raise subprocess.TimeoutExpired("sacct", 1)

    assert job_state("1", runner=run).state is State.UNKNOWN


def test_squeue_rc0_empty_falls_to_sacct() -> None:
    js = job_state("1", runner=fake((0, "", ""), (0, "FAILED\n", "")))
    assert js.state is State.ABSENT


# ---- segment_writer_dead ---------------------------------------------------

WID = "node-1-777-0-abcdef12"


def seg(tmp_path: Path, rows: list[str], name: str = WID, age: float = 1000.0) -> Path:
    p = tmp_path / (name + ".jsonl")
    p.write_text("".join(r + "\n" for r in rows))
    st = p.stat()
    os.utime(p, (st.st_atime, st.st_mtime - age))
    return p


ABSENT = fake((1, "", INVALID), (0, "COMPLETED\n", ""))
PRESENT = fake((0, "RUNNING\n", ""))
UNKNOWN = fake((1, "", "squeue: error: Unable to contact slurm controller"))


def now_of(p: Path) -> float:
    return p.stat().st_mtime + 1000.0


def test_trailer_is_dead_regardless_of_job_state(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}', json.dumps({"closed": True})], age=0.0)
    for r in (PRESENT, UNKNOWN, ABSENT):
        v = segment_writer_dead(p, grace_s=300, now_server=p.stat().st_mtime, runner=r)
        assert v.dead and "trailer" in v.reason


def test_trailer_without_job_id(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"closed": true}'], name="h-1-%s-0-aaaaaaaa" % NOJOB)
    assert segment_writer_dead(p, now_server=now_of(p), runner=ABSENT).dead


def test_no_job_id_no_trailer_never_dead(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], name="h-1-%s-0-aaaaaaaa" % NOJOB, age=10_000.0)
    assert not segment_writer_dead(p, now_server=now_of(p) + 1e6, runner=ABSENT).dead


def test_absent_and_old_is_dead(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'])
    assert segment_writer_dead(p, grace_s=300, now_server=now_of(p), runner=ABSENT).dead


def test_absent_but_young_not_dead(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], age=0.0)
    v = segment_writer_dead(p, grace_s=300, now_server=p.stat().st_mtime + 10, runner=ABSENT)
    assert not v.dead


def test_grace_boundary_is_strict(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], age=0.0)
    m = p.stat().st_mtime
    assert not segment_writer_dead(p, grace_s=300, now_server=m + 300.0, runner=ABSENT).dead
    assert segment_writer_dead(p, grace_s=300, now_server=m + 300.5, runner=ABSENT).dead


def test_present_not_dead(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'])
    assert not segment_writer_dead(p, now_server=now_of(p), runner=PRESENT).dead


def test_unknown_never_dead(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], age=1e6)
    assert not segment_writer_dead(p, now_server=now_of(p) + 1e6, runner=UNKNOWN).dead


@pytest.mark.parametrize(
    "exc",
    [subprocess.TimeoutExpired("squeue", 20), OSError("broken scheduler config")],
    ids=["timeout", "oserror"],
)
def test_raising_query_never_dead(tmp_path: Path, exc: BaseException) -> None:
    # A10 on Engaging: with SLURM_CONF broken, squeue hung to its timeout (run 9040eb50) or
    # raised OSError (run 306cbabb). Either must leave an old, trailerless segment alive.
    p = seg(tmp_path, ['{"a": 1}'], age=1e6)
    assert not segment_writer_dead(p, now_server=now_of(p) + 1e6, runner=raising(exc)).dead


def test_torn_trailer_is_not_a_trailer(tmp_path: Path) -> None:
    p = tmp_path / (WID + ".jsonl")
    p.write_text('{"a": 1}\n{"clos')
    assert not segment_writer_dead(p, now_server=0.0, runner=PRESENT).dead


def test_closed_false_is_not_a_trailer(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"closed": false}'])
    assert not segment_writer_dead(p, now_server=now_of(p), runner=PRESENT).dead


def test_unparseable_name_only_trailer(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], name="garbage")
    assert not segment_writer_dead(p, now_server=now_of(p) + 1e6, runner=ABSENT).dead


def test_host_with_dashes_segment(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], name="gpu-node-07-5-31-2-0a1b2c3d")
    assert segment_writer_dead(p, now_server=now_of(p), runner=ABSENT).dead


def test_server_now_cleans_up_and_is_close(tmp_path: Path) -> None:
    t = server_now(tmp_path)
    assert abs(t - time.time()) < 5
    assert list(tmp_path.iterdir()) == []


def test_default_now_uses_server_now(tmp_path: Path) -> None:
    p = seg(tmp_path, ['{"a": 1}'], age=1000.0)
    assert segment_writer_dead(p, grace_s=300, runner=ABSENT).dead
    assert [q.name for q in tmp_path.iterdir()] == [p.name]
