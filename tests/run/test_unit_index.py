"""Tests for xtrax.run.unit_index: segment writer, trailer, read side, sealing and compaction.

Spec 261030 §3.1 and §5: each refusal has a negative control (torn tail, R-S6 conflict, recompute,
live writer, UNKNOWN, nojob without trailer, crash at each step, interleaved compactors).
"""

import dataclasses
import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax.run import unit_index as ui
from xtrax.run import writer_liveness as wl
from xtrax.run.unit_index import IndexRow

WID = "node-1-777-0-abcdef12"
WID2 = "node-2-778-0-abcdef13"
INVALID = "slurm_load_jobs error: Invalid job id specified"
Res = tuple[int, str, str]


def fake(squeue: Res, sacct: Res = (0, "", "")):
    calls: list[list[str]] = []

    def run(argv: list[str], timeout: float) -> Res:
        calls.append(argv)
        return squeue if argv[0] == "squeue" else sacct

    run.calls = calls  # type: ignore[attr-defined]
    return run


def absent():
    return fake((1, "", INVALID), (0, "COMPLETED\n", ""))


def present():
    return fake((0, "RUNNING\n", ""))


def unknown():
    return fake((1, "", "squeue: error: Unable to contact slurm controller"))


def row(
    i: int,
    *,
    writer: str = WID,
    content: str | None = None,
    input_digest: str | None = None,
    unit_key: str | None = None,
    store: str = "/s",
    locator: str | None = None,
    record: str | None = None,
    ts: float | None = None,
) -> IndexRow:
    return IndexRow(
        store=store,
        locator=locator or f"units/{i}",
        unit_key=unit_key or f"u{i}",
        input_digest=input_digest or f"in{i}",
        content_digest=content or f"c{i}",
        record_digest=record or f"r{i}-{content or ''}-{store}-{locator or ''}",
        adapter="zarr_commit",
        writer_id=writer,
        run_id="run-1",
        ts=float(i) if ts is None else ts,
    )


def write_segment(root: Path, rows: list[IndexRow], writer: str = WID, close: bool = True) -> Path:
    w = ui.SegmentWriter(root, writer_id=writer).open()
    for r in rows:
        w.append(dataclasses.replace(r, writer_id=writer))
    if close:
        w.close()
    else:
        os.close(w._fd)  # simulate a killed writer: no trailer
        w._fd = None
    return w.path


def keys(root: Path) -> set[tuple[str, str, str]]:
    return {r.read_key for r in ui.iter_rows(root)}


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def old(path: Path) -> float:
    """A now_server that makes ``path`` older than the default grace."""
    return path.stat().st_mtime + 1000.0


# ---- root ----------------------------------------------------------------------------------


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(ui.ROOT_ENV, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    proj = tmp_path / "proj"
    proj.mkdir()
    return proj


def test_root_unset_is_off(clean_env: Path) -> None:
    assert ui.root_source(start=clean_env) == (None, "off")


def test_root_argument_wins_and_none_disables(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ui.ROOT_ENV, "/from/env")
    assert ui.root_source("/explicit", start=clean_env) == (Path("/explicit"), "arg")
    assert ui.root_source("none", start=clean_env) == (None, "arg (disabled)")


def test_root_env_beats_pyproject_and_empty_disables(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (clean_env / "pyproject.toml").write_text('[tool.xtrax]\nunit_index_root = "/from/pyproject"\n')
    monkeypatch.setenv(ui.ROOT_ENV, "/from/env")
    assert ui.root_source(start=clean_env) == (Path("/from/env"), "env")
    monkeypatch.setenv(ui.ROOT_ENV, "  ")
    assert ui.root_source(start=clean_env) == (None, "env (disabled)")
    monkeypatch.setenv(ui.ROOT_ENV, "None")
    assert ui.root_source(start=clean_env) == (None, "env (disabled)")


def test_root_pyproject_then_user_config(clean_env: Path, tmp_path: Path) -> None:
    cfg = tmp_path / "xdg" / "xtrax"
    cfg.mkdir(parents=True)
    (cfg / "config.toml").write_text('unit_index_root = "/from/user"\n')
    assert ui.root_source(start=clean_env) == (Path("/from/user"), "user_config")
    (clean_env / "pyproject.toml").write_text('[tool.xtrax]\nunit_index_root = "/from/pyproject"\n')
    assert ui.root_source(start=clean_env) == (Path("/from/pyproject"), "pyproject")


def test_root_relative_config_path_refused(clean_env: Path) -> None:
    (clean_env / "pyproject.toml").write_text('[tool.xtrax]\nunit_index_root = "rel/idx"\n')
    with pytest.raises(ValueError, match="relative"):
        ui.root_source(start=clean_env)


def test_root_malformed_config_raises(clean_env: Path) -> None:
    (clean_env / "pyproject.toml").write_text("[tool.xtrax\nunit_index_root = \n")
    with pytest.raises(ValueError):
        ui.root_source(start=clean_env)


def test_open_writer_off_writes_nothing(clean_env: Path) -> None:
    w = ui.open_writer(start=clean_env)
    assert not w.enabled
    with w:
        w.append(row(1))
    assert list(clean_env.rglob("*")) == []


def test_open_writer_on(clean_env: Path, tmp_path: Path) -> None:
    w = ui.open_writer(tmp_path / "idx", writer_id=WID)
    assert w.enabled
    with w:
        w.append(row(1))
    assert keys(tmp_path / "idx") == {row(1).read_key}


# ---- writer id -----------------------------------------------------------------------------


def test_writer_id_round_trips_through_liveness_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "27790386")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "2")
    wid = ui.new_writer_id(host="node2511.engaging", pid=42)
    w = wl.parse_writer_id(Path(f"{wid}.jsonl").stem)
    assert (w.host, w.pid, w.job_id, w.restart, len(w.uuid8)) == ("node2511", 42, "27790386", 2, 8)


def test_writer_id_outside_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    w = wl.parse_writer_id(ui.new_writer_id(host="my/la ptop", pid=7))
    assert (w.host, w.job_id, w.restart) == ("my_la_ptop", None, 0)


def test_writer_ids_are_unique() -> None:
    assert len({ui.new_writer_id() for _ in range(200)}) == 200


# ---- writer --------------------------------------------------------------------------------


def test_segment_rows_then_trailer(tmp_path: Path) -> None:
    p = write_segment(tmp_path, [row(1), row(2)])
    lines = p.read_bytes().split(b"\n")
    assert [json.loads(x)["unit_key"] for x in lines[:2]] == ["u1", "u2"]
    assert lines[2] == b'{"closed": true}' and lines[3] == b""
    assert wl._has_trailer(p)
    assert keys(tmp_path) == {row(1).read_key, row(2).read_key}


def test_trailer_written_when_body_raises(tmp_path: Path) -> None:
    w = ui.SegmentWriter(tmp_path, writer_id=WID)
    with pytest.raises(RuntimeError, match="boom"):
        with w:
            w.append(row(1))
            raise RuntimeError("boom")
    assert wl._has_trailer(w.path)


def test_same_writer_id_twice_refused(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1)])
    with pytest.raises(FileExistsError):
        ui.SegmentWriter(tmp_path, writer_id=WID).open()


def test_writer_misuse_refused(tmp_path: Path) -> None:
    w = ui.SegmentWriter(tmp_path, writer_id=WID)
    with pytest.raises(RuntimeError):
        w.append(row(1))  # not open
    w.open()
    with pytest.raises(RuntimeError):
        w.open()
    with pytest.raises(ValueError, match="writer_id"):
        w.append(row(1, writer=WID2))
    w.close()
    w.close()  # idempotent
    with pytest.raises(RuntimeError):
        w.append(row(1))
    with pytest.raises(ValueError):
        ui.SegmentWriter(tmp_path, writer_id="not-a-writer-id")


def test_guard_refusal_leaves_no_file(tmp_path: Path) -> None:
    class Refuse:
        def check(self, n_new_files: int) -> int:
            raise RuntimeError("over the file cap")

    with pytest.raises(RuntimeError, match="file cap"):
        ui.SegmentWriter(tmp_path / "idx", writer_id=WID, guard=Refuse()).open()
    assert not (tmp_path / "idx").exists()


def test_guard_checked_once_before_create(tmp_path: Path) -> None:
    seen: list[tuple[int, bool]] = []
    path = tmp_path / "segments" / f"{WID}.jsonl"

    class Count:
        def check(self, n_new_files: int) -> int:
            seen.append((n_new_files, path.exists()))
            return 0

    with ui.SegmentWriter(tmp_path, writer_id=WID, guard=Count()) as w:
        w.append(row(1))
    assert seen == [(1, False)]


def test_failed_write_does_not_swallow_trailer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = ui._write_all
    state = {"fail": False}

    def flaky(fd: int, data: bytes) -> None:
        if state["fail"]:
            state["fail"] = False
            os.write(fd, data[:10])
            raise OSError(errno.ENOSPC, "No space left on device")
        real(fd, data)

    monkeypatch.setattr(ui, "_write_all", flaky)
    w = ui.SegmentWriter(tmp_path, writer_id=WID).open()
    w.append(row(1))
    state["fail"] = True
    with pytest.raises(OSError):
        w.append(row(2))
    w.append(row(3))
    w.close()
    assert wl._has_trailer(w.path)
    rep = ui.ReadReport()
    assert {r.unit_key for r in ui.iter_rows(tmp_path, report=rep)} == {"u1", "u3"}
    assert [b.kind for b in rep.torn] == ["torn"]


# ---- read ----------------------------------------------------------------------------------


def test_read_dedups_across_dirs(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1), row(2)])
    for d in ("sealed", "rebuilt"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "x.jsonl").write_text(row(1).to_json_line() + row(3).to_json_line())
    rep = ui.ReadReport()
    got = list(ui.iter_rows(tmp_path, report=rep))
    assert {r.unit_key for r in got} == {"u1", "u2", "u3"}
    assert (rep.files, rep.rows_read, rep.duplicates) == (3, 6, 3)


def test_read_classifies_bad_lines(tmp_path: Path) -> None:
    seg = tmp_path / "segments"
    seg.mkdir()
    bad = dict(row(9).to_dict(), schema_version=99)
    missing = {k: v for k, v in row(8).to_dict().items() if k != "store"}
    (seg / f"{WID}.jsonl").write_text(
        row(1).to_json_line()
        + json.dumps(bad)
        + "\n"
        + json.dumps(missing)
        + "\n"
        + "[1, 2]\n"
        + '{"closed": true}\n'
        + '{"schema_version": 1, "sto'
    )
    rep = ui.ReadReport()
    assert [r.unit_key for r in ui.iter_rows(tmp_path, report=rep)] == ["u1"]
    assert [b.line for b in rep.torn] == [6]
    assert [b.line for b in rep.unreadable] == [2, 3, 4]
    assert "schema_version 99" in rep.unreadable[0].reason


def test_read_missing_root_is_empty(tmp_path: Path) -> None:
    assert list(ui.iter_rows(tmp_path / "nothing")) == []


def test_estale_retried_with_fresh_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_segment(tmp_path, [row(1)])
    opens: list[str] = []
    real_open = open

    def flaky_open(path: object, mode: str = "r", *a: object, **k: object):
        opens.append(str(path))
        if str(path) == str(p) and len(opens) <= 2:
            raise OSError(errno.ESTALE, "Stale file handle")
        return real_open(path, mode, *a, **k)

    monkeypatch.setattr(ui, "open", flaky_open, raising=False)
    monkeypatch.setattr(ui.time, "sleep", lambda s: None)
    assert keys(tmp_path) == {row(1).read_key}
    assert opens.count(str(p)) == 3


def test_estale_gives_up_after_bounded_tries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_segment(tmp_path, [row(1)])

    def always(path: object, mode: str = "r", *a: object, **k: object):
        raise OSError(errno.ESTALE, "Stale file handle")

    monkeypatch.setattr(ui, "open", always, raising=False)
    monkeypatch.setattr(ui.time, "sleep", lambda s: None)
    with pytest.raises(OSError) as e:
        list(ui.iter_rows(tmp_path))
    assert e.value.errno == errno.ESTALE


def test_vanished_file_restarts_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_segment(tmp_path, [row(1)])
    write_segment(tmp_path, [row(2)], writer=WID2)
    real = ui._read_bytes
    state = {"n": 0}

    def once(path: Path) -> bytes:
        state["n"] += 1
        if state["n"] == 2:
            raise FileNotFoundError(path)
        return real(path)

    monkeypatch.setattr(ui, "_read_bytes", once)
    assert keys(tmp_path) == {row(1).read_key, row(2).read_key}


def test_always_changing_index_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_segment(tmp_path, [row(1)])

    def gone(path: Path) -> bytes:
        raise FileNotFoundError(path)

    monkeypatch.setattr(ui, "_read_bytes", gone)
    with pytest.raises(ui.UnitIndexError, match="kept changing"):
        list(ui.iter_rows(tmp_path))


# ---- compaction: which segments --------------------------------------------------------------


def test_closed_segments_sealed_and_archived(tmp_path: Path) -> None:
    a = write_segment(tmp_path, [row(1), row(2)])
    b = write_segment(tmp_path, [row(3)], writer=WID2)
    before = keys(tmp_path)
    rep = ui.compact(tmp_path, runner=unknown())
    assert set(rep.taken) == {a.name, b.name}
    assert rep.rows_sealed == 3 and rep.sealed_path is not None
    assert keys(tmp_path) == before
    assert sorted(p.name for p in (tmp_path / "archive").iterdir()) == sorted(
        [f"segment-{a.name}", f"segment-{b.name}"]
    )
    assert list((tmp_path / "segments").iterdir()) == []
    assert rep.scheduler_calls == 0  # trailers decide; no scheduler query


def test_live_writer_segment_never_moves(tmp_path: Path) -> None:
    live = write_segment(tmp_path, [row(1)], close=False)
    rep = ui.compact(tmp_path, runner=present(), now_server=old(live))
    assert rep.taken == () and live.exists()
    assert not (tmp_path / "sealed").exists()


def test_negative_control_closed_twin_does_move(tmp_path: Path) -> None:
    p = write_segment(tmp_path, [row(1)])  # same content, but closed
    rep = ui.compact(tmp_path, runner=present(), now_server=old(p))
    assert rep.taken == (p.name,) and not p.exists()


def test_unknown_scheduler_never_dead(tmp_path: Path) -> None:
    p = write_segment(tmp_path, [row(1)], close=False)
    rep = ui.compact(tmp_path, runner=unknown(), now_server=old(p) + 1e6)
    assert rep.taken == () and p.exists()


def test_nojob_without_trailer_never_compacted(tmp_path: Path) -> None:
    p = write_segment(
        tmp_path,
        [row(1, writer="lap-1-nojob-0-aaaaaaaa")],
        writer="lap-1-nojob-0-aaaaaaaa",
        close=False,
    )
    rep = ui.compact(tmp_path, runner=absent(), now_server=old(p) + 1e6)
    assert rep.taken == () and p.exists()


def test_dead_writer_taken_after_grace(tmp_path: Path) -> None:
    p = write_segment(tmp_path, [row(1)], close=False)
    young = ui.compact(tmp_path, runner=absent(), now_server=p.stat().st_mtime + 10)
    assert young.taken == () and p.exists()
    rep = ui.compact(tmp_path, runner=absent(), now_server=old(p))
    assert rep.taken == (p.name,) and keys(tmp_path) == {row(1).read_key}


def test_scheduler_queries_cached_per_pass(tmp_path: Path) -> None:
    for k in range(5):
        wid = f"node-{k}-777-0-abcdef{k:02d}"  # same job 777
        write_segment(tmp_path, [row(k, writer=wid)], writer=wid, close=False)
    run = absent()
    rep = ui.compact(tmp_path, runner=run, now_server=1e12)
    assert len(rep.taken) == 5
    assert len(run.calls) == 2  # one squeue + one sacct for the shared job


def test_broken_scheduler_queried_once_per_pass(tmp_path: Path) -> None:
    for k in range(5):
        wid = f"node-{k}-{700 + k}-0-abcdef{k:02d}"  # five different jobs
        write_segment(tmp_path, [row(k, writer=wid)], writer=wid, close=False)
    calls: list[str] = []

    def hung(argv: list[str], timeout: float) -> Res:
        calls.append(argv[0])
        raise subprocess.TimeoutExpired(argv, timeout)

    rep = ui.compact(tmp_path, runner=hung, now_server=1e12)
    assert rep.taken == () and calls == ["squeue"]


# ---- compaction: what is sealed ------------------------------------------------------------


def test_idle_pass_changes_nothing(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1)])
    ui.compact(tmp_path, runner=unknown())
    live = write_segment(tmp_path, [row(2)], writer=WID2, close=False)
    before = snapshot(tmp_path)
    rep = ui.compact(tmp_path, runner=present(), now_server=old(live))
    assert rep.taken == () and rep.sealed_path is None
    assert snapshot(tmp_path) == before


def test_dry_run_writes_nothing_and_matches_real_pass(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1), row(2, content="x")])
    write_segment(tmp_path, [row(2, content="y")], writer=WID2)
    before = snapshot(tmp_path)
    dry = ui.compact(tmp_path, dry_run=True, runner=unknown(), now_server=1e12)
    assert snapshot(tmp_path) == before
    assert dry.sealed_path is None and dry.archived == ()
    real = ui.compact(tmp_path, runner=unknown(), now_server=1e12)
    for f in ("segments", "rows_in", "rows_sealed", "collapsed", "copies", "conflicts"):
        assert getattr(dry, f) == getattr(real, f), f


def test_rs6_conflict_keeps_both_lists_pair_and_does_not_wedge(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1, content="A"), row(5)])
    write_segment(tmp_path, [row(1, content="B")], writer=WID2)
    rep = ui.compact(tmp_path, runner=unknown())
    assert rep.conflicts == (ui.Conflict("u1", "in1", ("A", "B")),)
    assert {r.content_digest for r in ui.iter_rows(tmp_path) if r.unit_key == "u1"} == {"A", "B"}
    assert len(rep.taken) == 2 and rep.rows_sealed == 3
    # next pass: the conflict is listed again and a new key still seals
    write_segment(tmp_path, [row(6)], writer="node-3-779-0-abcdef14")
    again = ui.compact(tmp_path, runner=unknown())
    assert again.conflicts == rep.conflicts
    assert "u6" in {r.unit_key for r in ui.iter_rows(tmp_path)}


def test_recompute_with_new_input_digest_keeps_both(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1, input_digest="old", content="A")])
    write_segment(tmp_path, [row(1, input_digest="new", content="B")], writer=WID2)
    rep = ui.compact(tmp_path, runner=unknown())
    assert rep.conflicts == () and rep.rows_sealed == 2


def test_same_place_collapses_other_places_kept(tmp_path: Path) -> None:
    write_segment(
        tmp_path,
        [
            row(1, content="A", record="r-early", ts=1.0),
            row(1, content="A", record="r-late", ts=2.0),  # same place, re-committed
            row(1, content="A", store="/copy"),  # same content, another store
        ],
    )
    rep = ui.compact(tmp_path, runner=unknown())
    assert (rep.rows_in, rep.rows_sealed, rep.collapsed, rep.copies) == (3, 2, 1, 1)
    assert rep.conflicts == ()
    got = {(r.store, r.record_digest) for r in ui.iter_rows(tmp_path)}
    assert ("/s", "r-late") in got and ("/s", "r-early") not in got
    assert any(s == "/copy" for s, _ in got)


def test_torn_tail_in_dead_segment_dropped_and_reported(tmp_path: Path) -> None:
    p = write_segment(tmp_path, [row(1)], close=False)
    with open(p, "ab") as fh:
        fh.write(b'{"schema_version": 1, "store": "/s", "loc')
    rep = ui.compact(tmp_path, runner=absent(), now_server=old(p))
    assert rep.taken == (p.name,) and [b.kind for b in rep.torn] == ["torn"]
    assert keys(tmp_path) == {row(1).read_key}


def test_segment_with_unreadable_row_left_in_place(tmp_path: Path) -> None:
    good = write_segment(tmp_path, [row(1)])
    bad = write_segment(tmp_path, [row(2)], writer=WID2, close=False)
    with open(bad, "ab") as fh:
        fh.write(
            (json.dumps(dict(row(3, writer=WID2).to_dict(), schema_version=2)) + "\n").encode()
        )
        fh.write(b'{"closed": true}\n')
    rep = ui.compact(tmp_path, runner=unknown())
    assert rep.taken == (good.name,) and rep.left_unreadable == (bad.name,)
    assert bad.exists() and keys(tmp_path) == {row(1).read_key, row(2).read_key}


def test_unreadable_sealed_file_refuses_whole_pass(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1)])
    ui.compact(tmp_path, runner=unknown())
    sealed = next((tmp_path / "sealed").glob("*.jsonl"))
    with open(sealed, "ab") as fh:
        fh.write((json.dumps(dict(row(9).to_dict(), schema_version=2)) + "\n").encode())
    write_segment(tmp_path, [row(2)], writer=WID2)
    before = snapshot(tmp_path)
    with pytest.raises(ui.UnitIndexError, match="out of view"):
        ui.compact(tmp_path, runner=unknown())
    assert snapshot(tmp_path) == before


def test_second_pass_folds_previous_sealed(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1)])
    first = ui.compact(tmp_path, runner=unknown())
    write_segment(tmp_path, [row(2)], writer=WID2)
    second = ui.compact(tmp_path, runner=unknown())
    assert second.sealed_inputs == (Path(first.sealed_path).name,)
    assert [p.name for p in (tmp_path / "sealed").glob("*.jsonl")] == [
        Path(second.sealed_path).name
    ]
    assert keys(tmp_path) == {row(1).read_key, row(2).read_key}
    assert f"sealed-{Path(first.sealed_path).name}" in os.listdir(tmp_path / "archive")


# ---- crashes and concurrency ---------------------------------------------------------------

STEPS = ["sealed_tmp_written", "sealed_replaced", "first_input_archived"]


class Crash(Exception):
    pass


def _setup_three(root: Path) -> set[tuple[str, str, str]]:
    write_segment(root, [row(1)])
    ui.compact(root, runner=unknown())  # an existing sealed file joins the next pass
    write_segment(root, [row(2), row(3)], writer=WID2)
    write_segment(root, [row(4)], writer="node-3-779-0-abcdef14")
    return keys(root)


@pytest.mark.parametrize("step", STEPS)
def test_crash_at_each_step_loses_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    before = _setup_three(tmp_path)

    def crash(s: str) -> None:
        if s == step:
            raise Crash(s)

    monkeypatch.setattr(ui, "_fault", crash)
    with pytest.raises(Crash):
        ui.compact(tmp_path, runner=unknown())
    assert keys(tmp_path) == before  # duplicates allowed, losses not
    monkeypatch.undo()
    ui.compact(tmp_path, runner=unknown())  # recovers
    assert keys(tmp_path) == before
    assert list((tmp_path / "segments").glob("*.jsonl")) == []


@pytest.mark.parametrize("step", STEPS)
def test_interleaved_compactors_lose_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    before = _setup_three(tmp_path)
    real_fault = ui._fault
    state = {"done": False}

    def run_other(s: str) -> None:
        if s == step and not state["done"]:
            state["done"] = True
            mid.append(keys(tmp_path))  # a reader inside the first compactor's window
            ui.compact(tmp_path, runner=unknown())  # a second compactor runs to completion here
            mid.append(keys(tmp_path))  # and again, with the first one still mid-pass
        real_fault(s)

    mid: list[set[tuple[str, str, str]]] = []

    monkeypatch.setattr(ui, "_fault", run_other)
    ui.compact(tmp_path, runner=unknown())
    assert state["done"]
    assert mid == [before, before]
    assert keys(tmp_path) == before
    monkeypatch.undo()
    ui.compact(tmp_path, runner=unknown())  # nothing left to take: idle
    assert keys(tmp_path) == before


def test_reader_rescans_when_a_sealed_file_appears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_segment(tmp_path, [row(1)])
    real = ui._read_bytes
    state = {"done": False}

    def compact_during_read(path: Path) -> bytes:
        data = real(path)
        if not state["done"]:
            state["done"] = True
            write_segment(tmp_path, [row(2)], writer=WID2)
            ui.compact(tmp_path, runner=unknown())  # real archive + new sealed mid-scan
        return data

    monkeypatch.setattr(ui, "_read_bytes", compact_during_read)
    assert keys(tmp_path) == {row(1).read_key, row(2).read_key}


def test_writer_id_must_be_a_plain_file_name(tmp_path: Path) -> None:
    for bad in ("a/b-1-nojob-0-aaaaaaaa", ".hidden-1-nojob-0-aaaaaaaa"):
        with pytest.raises(ValueError):
            ui.SegmentWriter(tmp_path, writer_id=bad)


def test_reused_writer_id_never_overwrites_archive(tmp_path: Path) -> None:
    write_segment(tmp_path, [row(1)])
    ui.compact(tmp_path, runner=unknown())
    write_segment(tmp_path, [row(2)])  # same explicit writer id, after the first was archived
    ui.compact(tmp_path, runner=unknown())
    archived = sorted(p.name for p in (tmp_path / "archive").glob(f"segment-{WID}*"))
    assert len(archived) == 2
    assert keys(tmp_path) == {row(1).read_key, row(2).read_key}


def test_stale_partials_swept_fresh_kept(tmp_path: Path) -> None:
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    stale, fresh = sealed / ".old.partial", sealed / ".new.partial"
    for p in (stale, fresh):
        p.write_text("x")
    t = stale.stat().st_mtime - 2 * ui.PARTIAL_MAX_AGE_S
    os.utime(stale, (t, t))
    write_segment(tmp_path, [row(1)])
    rep = ui.compact(tmp_path, runner=unknown())
    assert rep.partials_removed == 1 and not stale.exists() and fresh.exists()


def test_append_fails_then_close_puts_trailer_on_own_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = ui._write_all
    state = {"fail": False}

    def flaky(fd: int, data: bytes) -> None:
        if state["fail"]:
            state["fail"] = False
            os.write(fd, data[:10])
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        real(fd, data)

    monkeypatch.setattr(ui, "_write_all", flaky)
    w = ui.SegmentWriter(tmp_path, writer_id=WID).open()
    w.append(row(1))
    state["fail"] = True
    with pytest.raises(OSError):
        w.append(row(2))
    w.close()
    assert w.path.read_bytes().endswith(b'\n{"closed": true}\n')
    assert wl._has_trailer(w.path)


def test_real_process_kill_after_replace(tmp_path: Path) -> None:
    before = _setup_three(tmp_path)
    env = dict(os.environ, **{ui.FAULT_ENV: "unit_index:sealed_replaced"})
    code = (
        "import sys\n"
        "from xtrax.run import unit_index as ui\n"
        "ui.compact(sys.argv[1], runner=lambda a, t: (1, '', 'down'))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, check=False)
    assert proc.returncode == 137
    assert len(list((tmp_path / "sealed").glob("*.jsonl"))) == 2  # old + new, nothing archived
    assert keys(tmp_path) == before
