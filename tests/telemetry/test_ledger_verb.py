"""The `xtrax ledger` verb: list, show, verify, compact."""

import pytest

from xtrax.cli.ledger_verb import LedgerArgs, run_ledger
from xtrax.telemetry.ledger import RunLedger, blobs_dir, iter_rows
from xtrax.telemetry.record import STATUS_FAILED, IRRef
from xtrax.telemetry.store import BlobStore


@pytest.fixture
def ledger_root(tmp_path, monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)
    root = tmp_path / "ledger"
    for i in range(3):
        with RunLedger.open(f"run-{i:04d}", root=root):
            pass
    return root


def _args(root, **kw):
    return LedgerArgs(root=str(root), **kw)


# --- list -------------------------------------------------------------------


def test_list_prints_every_row(ledger_root, capsys):
    run_ledger(_args(ledger_root, action="list"))
    out = capsys.readouterr().out
    for i in range(3):
        assert f"run-{i:04d}" in out


def test_list_on_an_empty_ledger_says_so(tmp_path, capsys):
    run_ledger(_args(tmp_path / "nothing", action="list"))
    assert "no ledger rows" in capsys.readouterr().out


def test_list_marks_non_citable_rows_and_explains_the_marker(ledger_root, capsys):
    """A bare glyph the reader has to decode is not an explanation."""
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-bad", root=ledger_root):
            raise RuntimeError("broke")
    run_ledger(_args(ledger_root, action="list"))
    out = capsys.readouterr().out
    assert "! = not citable" in out


def test_list_respects_the_limit(ledger_root, capsys):
    run_ledger(_args(ledger_root, action="list", limit=1))
    out = capsys.readouterr().out
    assert "run-0002" in out
    assert "run-0000" not in out


# --- show -------------------------------------------------------------------


def test_show_reports_provenance_and_context(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    root = tmp_path / "ledger"
    with RunLedger.open("run-detail", root=root, context={"config_hash": "abc123"}):
        pass
    run_ledger(_args(root, action="show", run_id="run-detail"))
    out = capsys.readouterr().out
    assert "run-detail" in out
    assert "provenance_source" in out
    assert "config_hash" in out
    assert "abc123" in out


def test_show_requires_a_run_id(ledger_root):
    with pytest.raises(SystemExit) as excinfo:
        run_ledger(_args(ledger_root, action="show"))
    assert excinfo.value.code == 2


def test_show_exits_nonzero_for_an_unknown_run(ledger_root):
    with pytest.raises(SystemExit) as excinfo:
        run_ledger(_args(ledger_root, action="show", run_id="run-nope"))
    assert excinfo.value.code == 1


def test_show_flags_a_missing_blob(tmp_path, capsys, monkeypatch):
    """An IR digest whose blob has vanished must be visible, not implied."""
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    root = tmp_path / "ledger"
    store = BlobStore(blobs_dir(root))
    sha, size = store.put("some ir")
    with RunLedger.open("run-ir", root=root) as ledger:
        ledger.record_ir(IRRef(kind="jaxpr", sha256=sha, bytes=size))
    store.delete(sha)
    run_ledger(_args(root, action="show", run_id="run-ir"))
    assert "MISSING" in capsys.readouterr().out


# --- verify -----------------------------------------------------------------


def test_verify_passes_on_an_intact_ledger(ledger_root, capsys):
    run_ledger(_args(ledger_root, action="verify"))
    assert "PASS" in capsys.readouterr().out


def test_verify_exits_nonzero_when_a_blob_is_gone(tmp_path, monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    root = tmp_path / "ledger"
    store = BlobStore(blobs_dir(root))
    sha, size = store.put("doomed ir")
    with RunLedger.open("run-ir", root=root) as ledger:
        ledger.record_ir(IRRef(kind="jaxpr", sha256=sha, bytes=size))
    store.delete(sha)
    with pytest.raises(SystemExit) as excinfo:
        run_ledger(_args(root, action="verify"))
    assert excinfo.value.code == 1


# --- compact ----------------------------------------------------------------


def test_compact_reports_what_it_did(ledger_root, capsys):
    run_ledger(_args(ledger_root, action="compact", force=True))
    out = capsys.readouterr().out
    assert "compact: read 3 rows, kept 3" in out


def test_dry_run_changes_nothing(ledger_root, capsys):
    run_ledger(_args(ledger_root, action="compact", force=True, dry_run=True))
    assert "would compact" in capsys.readouterr().out
    assert not (ledger_root / "sealed").exists()


def test_compact_below_threshold_explains_the_no_op(ledger_root, capsys):
    run_ledger(_args(ledger_root, action="compact"))
    assert "--force" in capsys.readouterr().out


def test_compact_preserves_the_rows(ledger_root):
    before = {r.run_id for r in iter_rows(ledger_root)}
    run_ledger(_args(ledger_root, action="compact", force=True))
    assert {r.run_id for r in iter_rows(ledger_root)} == before


def test_compact_keeps_the_latest_row_per_run(ledger_root):
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-0000", root=ledger_root):
            raise RuntimeError("second attempt")
    run_ledger(_args(ledger_root, action="compact", force=True))
    rows = {r.run_id: r for r in iter_rows(ledger_root)}
    assert rows["run-0000"].telemetry_status == STATUS_FAILED


# --- dispatch ---------------------------------------------------------------


def test_an_unknown_action_exits_two(ledger_root):
    with pytest.raises(SystemExit) as excinfo:
        run_ledger(_args(ledger_root, action="frobnicate"))
    assert excinfo.value.code == 2


def test_the_verb_is_registered():
    from xtrax.cli.registry import REGISTRY

    assert "ledger" in REGISTRY
    args_cls, run_fn = REGISTRY["ledger"]
    assert args_cls is LedgerArgs
    assert run_fn is run_ledger
