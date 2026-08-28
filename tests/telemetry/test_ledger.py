"""Ledger behaviour: fail-closed opens, guaranteed rows, opt-out, concurrency."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from xtrax.telemetry.ledger import (
    MAX_SEGMENT_BYTES,
    LedgerUnavailableError,
    RunLedger,
    find_run,
    iter_rows,
    resolve_root,
)
from xtrax.telemetry.record import (
    KIND_EVAL,
    KIND_TRAIN,
    STATUS_COMPLETE,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_OPTED_OUT,
    IRRef,
)


@pytest.fixture(autouse=True)
def _no_optout(monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)


# --- the fail-closed contract ----------------------------------------------


def test_open_fails_closed_when_the_root_is_unwritable(tmp_path):
    """The whole point: a run that cannot be recorded does not start."""
    root = tmp_path / "ledger"
    root.mkdir()
    root.chmod(0o500)
    try:
        with pytest.raises(LedgerUnavailableError, match="not writable"):
            RunLedger.open("run-1", root=root)
    finally:
        root.chmod(0o700)


def test_the_unwritable_error_names_the_ways_out(tmp_path):
    """A fatal error must tell the caller what to do about it."""
    root = tmp_path / "ledger"
    root.mkdir()
    root.chmod(0o500)
    try:
        with pytest.raises(LedgerUnavailableError) as excinfo:
            RunLedger.open("run-1", root=root)
    finally:
        root.chmod(0o700)
    message = str(excinfo.value)
    assert "XTRAX_LEDGER_ROOT" in message
    assert "XTRAX_TELEMETRY_OPTOUT" in message


def test_writability_is_proven_at_open_not_at_close(tmp_path):
    """Discovering this after six hours of training is too late."""
    root = tmp_path / "nested" / "deep" / "ledger"
    with RunLedger.open("run-1", root=root):
        assert (root / "segments").is_dir()


# --- every run leaves exactly one row ---------------------------------------


def test_a_successful_run_writes_one_complete_row(tmp_path):
    root = tmp_path / "ledger"
    with RunLedger.open("run-abc", kind=KIND_TRAIN, root=root):
        pass
    rows = list(iter_rows(root))
    assert len(rows) == 1
    assert rows[0].run_id == "run-abc"
    assert rows[0].telemetry_status == STATUS_COMPLETE
    assert rows[0].is_citable


def test_a_raising_run_still_writes_a_row_and_reraises(tmp_path):
    """A crashed run is when the record matters most."""
    root = tmp_path / "ledger"
    with pytest.raises(ValueError, match="boom"):
        with RunLedger.open("run-crash", root=root):
            raise ValueError("boom")
    rows = list(iter_rows(root))
    assert len(rows) == 1
    assert rows[0].telemetry_status == STATUS_FAILED
    assert "ValueError" in rows[0].status_reason
    assert "boom" in rows[0].status_reason
    assert not rows[0].is_citable


def test_the_exception_is_not_swallowed(tmp_path):
    """Trading a visible training failure for an invisible one would be worse."""
    root = tmp_path / "ledger"
    with pytest.raises(KeyError):
        with RunLedger.open("run-x", root=root):
            raise KeyError("nope")


def test_double_close_is_refused(tmp_path):
    root = tmp_path / "ledger"
    ledger = RunLedger.open("run-1", root=root)
    ledger.close()
    with pytest.raises(Exception, match="already closed"):
        ledger.close()


# --- opt-out ----------------------------------------------------------------


def test_optout_still_writes_a_row(tmp_path, monkeypatch):
    """Opting out of capture does not opt out of the record."""
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    root = tmp_path / "ledger"
    with RunLedger.open("run-opt", root=root) as ledger:
        assert ledger.opted_out
    rows = list(iter_rows(root))
    assert len(rows) == 1
    assert rows[0].telemetry_status == STATUS_OPTED_OUT
    assert not rows[0].is_citable
    assert "XTRAX_TELEMETRY_OPTOUT" in rows[0].status_reason


def test_optout_is_not_overwritten_by_a_later_degrade(tmp_path, monkeypatch):
    """The reason must name the caller's own choice, not a downstream symptom."""
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    root = tmp_path / "ledger"
    with RunLedger.open("run-opt", root=root) as ledger:
        ledger.set_status(STATUS_DEGRADED, "ir capture incomplete")
    assert list(iter_rows(root))[0].telemetry_status == STATUS_OPTED_OUT


def test_optout_is_still_overridden_by_an_actual_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    root = tmp_path / "ledger"
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-opt", root=root):
            raise RuntimeError("crashed anyway")
    assert list(iter_rows(root))[0].telemetry_status == STATUS_FAILED


# --- lineage ----------------------------------------------------------------


def test_derived_from_is_recorded(tmp_path):
    root = tmp_path / "ledger"
    with RunLedger.open("run-child", root=root, derived_from="run-parent"):
        pass
    assert list(iter_rows(root))[0].derived_from == "run-parent"


def test_rows_form_a_derivable_dag(tmp_path):
    """No graph store needed: the edges are a field."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-a", root=root):
        pass
    with RunLedger.open("run-b", root=root, derived_from="run-a"):
        pass
    with RunLedger.open("run-c", root=root, derived_from="run-b"):
        pass
    edges = {r.run_id: r.derived_from for r in iter_rows(root)}
    assert edges == {"run-a": None, "run-b": "run-a", "run-c": "run-b"}


# --- reading ----------------------------------------------------------------


def test_find_run_returns_the_latest_row_for_an_id(tmp_path):
    """A later failed row must out-vote an earlier complete one."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-1", root=root):
        pass
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-1", root=root):
            raise RuntimeError("second attempt failed")
    assert find_run("run-1", root).telemetry_status == STATUS_FAILED


def test_find_run_returns_none_for_an_unknown_id(tmp_path):
    assert find_run("nope", tmp_path / "ledger") is None


def test_a_corrupt_row_does_not_make_the_history_unreadable(tmp_path):
    """One bad line must not cost the whole ledger."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-good", root=root):
        pass
    segment = next((root / "segments").glob("*.jsonl"))
    with segment.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    with pytest.warns(UserWarning, match="skipping unreadable ledger row"):
        rows = list(iter_rows(root))
    assert [r.run_id for r in rows] == ["run-good"]


def test_strict_mode_surfaces_the_error_instead(tmp_path):
    root = tmp_path / "ledger"
    with RunLedger.open("run-good", root=root):
        pass
    segment = next((root / "segments").glob("*.jsonl"))
    with segment.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    with pytest.raises(Exception, match="not valid JSON"):
        list(iter_rows(root, strict=True))


def test_blank_lines_are_ignored(tmp_path):
    root = tmp_path / "ledger"
    with RunLedger.open("run-1", root=root):
        pass
    segment = next((root / "segments").glob("*.jsonl"))
    with segment.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
    assert len(list(iter_rows(root))) == 1


def test_iter_rows_on_a_missing_ledger_is_empty_not_an_error(tmp_path):
    assert list(iter_rows(tmp_path / "never-created")) == []


# --- storage layout ---------------------------------------------------------


def test_rows_are_appended_not_overwritten(tmp_path):
    """The append-only property, stated as a test."""
    root = tmp_path / "ledger"
    for i in range(5):
        with RunLedger.open(f"run-{i}", root=root):
            pass
    assert len(list(iter_rows(root))) == 5


def test_each_row_is_exactly_one_json_line(tmp_path):
    root = tmp_path / "ledger"
    for i in range(3):
        with RunLedger.open(f"run-{i}", root=root):
            pass
    segment = next((root / "segments").glob("*.jsonl"))
    lines = segment.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        json.loads(line)


def test_ir_refs_are_carried_into_the_row(tmp_path):
    root = tmp_path / "ledger"
    ref = IRRef(kind="jaxpr", sha256="d" * 64, bytes=7)
    with RunLedger.open("run-ir", root=root) as ledger:
        ledger.record_ir(ref)
    assert list(iter_rows(root))[0].ir[0].sha256 == "d" * 64


def test_record_ir_accepts_a_single_ref_or_a_sequence(tmp_path):
    root = tmp_path / "ledger"
    refs = (
        IRRef(kind="jaxpr", sha256="a" * 64, bytes=1),
        IRRef(kind="stablehlo", sha256="b" * 64, bytes=2),
    )
    with RunLedger.open("run-ir", root=root) as ledger:
        ledger.record_ir(refs)
    assert len(list(iter_rows(root))[0].ir) == 2


def test_segment_rollover_is_bounded(tmp_path):
    assert MAX_SEGMENT_BYTES > 0


# --- root resolution --------------------------------------------------------


def test_root_resolution_prefers_the_explicit_argument(monkeypatch, tmp_path):
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(tmp_path / "from-env"))
    assert resolve_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_root_resolution_falls_back_to_the_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("XTRAX_LEDGER_ROOT", str(tmp_path / "from-env"))
    assert resolve_root() == tmp_path / "from-env"


def test_root_resolution_default_is_under_dot_xtrax(monkeypatch):
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)
    assert resolve_root() == Path(".xtrax/ledger")


# --- concurrency ------------------------------------------------------------


_CONCURRENT_WRITER = """
import sys
from xtrax.telemetry.ledger import RunLedger

root, run_id = sys.argv[1], sys.argv[2]
# Long payloads make a torn write detectable: interleaving two short lines can
# coincidentally still parse, whereas interleaving 4 KB of padding cannot.
with RunLedger.open(run_id, root=root) as ledger:
    ledger.status_padding = "x" * 4096
"""


def test_concurrent_appends_do_not_interleave(tmp_path):
    """The reason this uses flock rather than findings.py's unlocked append.

    Real separate processes are the only honest test of an advisory file lock:
    threads in one interpreter would share the file descriptor and prove
    nothing. subprocess rather than os.fork() because JAX is multithreaded and
    forking a multithreaded process risks a deadlock -- the test would then be
    a CI flake rather than a guarantee.
    """
    root = tmp_path / "ledger"
    RunLedger.open("run-warmup", root=root).close()

    procs = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", _CONCURRENT_WRITER, str(root), f"run-child-{i}"],
        )
        for i in range(8)
    ]
    for proc in procs:
        assert proc.wait(timeout=120) == 0

    for segment in (root / "segments").glob("*.jsonl"):
        for line in segment.read_text().splitlines():
            if line.strip():
                json.loads(line)  # a torn line would fail here

    ids = {r.run_id for r in iter_rows(root)}
    assert ids == {"run-warmup", *{f"run-child-{i}" for i in range(8)}}


def test_eval_runs_are_recorded_too(tmp_path):
    """The inference path is not exempt."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-eval", kind=KIND_EVAL, root=root):
        pass
    assert list(iter_rows(root))[0].kind == KIND_EVAL
