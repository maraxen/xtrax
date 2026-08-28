"""Compaction: dedup, archival (never deletion), conservative blob GC."""

import pytest

from xtrax.telemetry.compact import (
    ARCHIVE_DIRNAME,
    COMPACTION_THRESHOLD,
    compact_ledger,
    should_compact,
    verify_ledger,
)
from xtrax.telemetry.ledger import RunLedger, blobs_dir, iter_rows
from xtrax.telemetry.record import STATUS_FAILED
from xtrax.telemetry.store import BlobStore


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)


def _write_runs(root, count, kind="train"):
    for i in range(count):
        with RunLedger.open(f"run-{i:04d}", kind=kind, root=root):
            pass


# --- threshold --------------------------------------------------------------


def test_compaction_is_a_noop_below_the_threshold(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 3)
    result = compact_ledger(root)
    assert result.rows_read == 0
    assert not (root / "sealed").exists()


def test_force_compacts_regardless_of_threshold(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 3)
    result = compact_ledger(root, force=True)
    assert result.rows_read == 3
    assert result.rows_kept == 3


def test_should_compact_tracks_segment_count(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 2)
    assert not should_compact(root)
    assert should_compact(root, threshold=1)


def test_default_threshold_matches_bathos(tmp_path):
    assert COMPACTION_THRESHOLD == 50


# --- nothing is destroyed ---------------------------------------------------


def test_segments_are_archived_not_deleted(tmp_path):
    """bathos's real invariant: compaction destroys nothing."""
    root = tmp_path / "ledger"
    _write_runs(root, 3)
    before = sorted(p.name for p in (root / "segments").glob("*.jsonl"))
    compact_ledger(root, force=True)
    archived = sorted(p.name for p in (root / ARCHIVE_DIRNAME).glob("segment-*.jsonl"))
    assert archived == [f"segment-{name}" for name in before]


def test_compaction_does_not_lose_rows(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 5)
    before = {r.run_id for r in iter_rows(root)}
    compact_ledger(root, force=True)
    assert {r.run_id for r in iter_rows(root)} == before


def test_rows_are_not_double_counted_after_compaction(tmp_path):
    """The reason segments are moved rather than left in place."""
    root = tmp_path / "ledger"
    _write_runs(root, 4)
    compact_ledger(root, force=True)
    assert len(list(iter_rows(root))) == 4


def test_repeated_compaction_is_stable(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 4)
    compact_ledger(root, force=True)
    compact_ledger(root, force=True)
    compact_ledger(root, force=True)
    assert len(list(iter_rows(root))) == 4


def test_compaction_after_new_runs_keeps_everything(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 3)
    compact_ledger(root, force=True)
    with RunLedger.open("run-later", root=root):
        pass
    compact_ledger(root, force=True)
    ids = {r.run_id for r in iter_rows(root)}
    assert "run-later" in ids
    assert len(ids) == 4


# --- dedup ------------------------------------------------------------------


def test_a_later_row_supersedes_an_earlier_one_for_the_same_run(tmp_path):
    """A reopened run must be described by its latest outcome."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-1", root=root):
        pass
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-1", root=root):
            raise RuntimeError("second attempt failed")

    result = compact_ledger(root, force=True)
    assert result.rows_read == 2
    assert result.rows_kept == 1
    assert result.rows_superseded == 1
    rows = list(iter_rows(root))
    assert len(rows) == 1
    assert rows[0].telemetry_status == STATUS_FAILED


def test_the_same_run_id_under_a_different_kind_is_not_deduped(tmp_path):
    """train and eval for one run_id are different facts, not duplicates."""
    root = tmp_path / "ledger"
    with RunLedger.open("run-1", kind="train", root=root):
        pass
    with RunLedger.open("run-1", kind="eval", root=root):
        pass
    compact_ledger(root, force=True)
    assert len(list(iter_rows(root))) == 2


def test_dedup_preserves_chronological_order(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 5)
    compact_ledger(root, force=True)
    assert [r.run_id for r in iter_rows(root)] == [f"run-{i:04d}" for i in range(5)]


# --- blob GC ----------------------------------------------------------------


def test_orphan_blobs_are_collected(tmp_path):
    """The genuine orphan case: a crash between storing IR and committing a row."""
    root = tmp_path / "ledger"
    _write_runs(root, 2)
    store = BlobStore(blobs_dir(root))
    store.put("orphaned ir text that no row references")
    assert store.count() == 1

    result = compact_ledger(root, force=True)
    assert result.blobs_deleted == 1
    assert store.count() == 0
    assert result.bytes_reclaimed > 0


def test_referenced_blobs_survive_gc(tmp_path):
    root = tmp_path / "ledger"
    store = BlobStore(blobs_dir(root))
    sha, size = store.put("referenced ir")
    from xtrax.telemetry.record import IRRef

    with RunLedger.open("run-1", root=root) as ledger:
        ledger.record_ir(IRRef(kind="jaxpr", sha256=sha, bytes=size))

    result = compact_ledger(root, force=True)
    assert result.blobs_deleted == 0
    assert store.has(sha)


def test_blobs_referenced_only_by_archived_rows_survive(tmp_path):
    """The archive must stay readable, so GC is conservative by design."""
    from xtrax.telemetry.record import IRRef

    root = tmp_path / "ledger"
    store = BlobStore(blobs_dir(root))
    sha, size = store.put("ir from a superseded row")
    with RunLedger.open("run-1", root=root) as ledger:
        ledger.record_ir(IRRef(kind="jaxpr", sha256=sha, bytes=size))
    # Supersede it with a row carrying no IR at all.
    with RunLedger.open("run-1", root=root):
        pass

    compact_ledger(root, force=True)
    assert store.has(sha), "a blob referenced by an archived row must not be collected"


def test_gc_can_be_disabled(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 2)
    store = BlobStore(blobs_dir(root))
    store.put("orphan")
    result = compact_ledger(root, force=True, gc_blobs=False)
    assert result.blobs_deleted == 0
    assert store.count() == 1


# --- dry run ----------------------------------------------------------------


def test_dry_run_changes_nothing(tmp_path):
    root = tmp_path / "ledger"
    _write_runs(root, 3)
    store = BlobStore(blobs_dir(root))
    store.put("orphan")

    result = compact_ledger(root, force=True, dry_run=True)
    assert result.dry_run
    assert result.rows_read == 3
    assert result.blobs_deleted == 1  # what it *would* delete
    assert store.count() == 1  # but did not
    assert not (root / "sealed").exists()
    assert len(list((root / "segments").glob("*.jsonl"))) == 1


# --- backends ---------------------------------------------------------------


def test_an_unimplemented_backend_raises_rather_than_falling_back(tmp_path):
    """Asking for a tier that does not exist must be immediately visible."""
    with pytest.raises(ValueError, match="only 'jsonl' is implemented"):
        compact_ledger(tmp_path / "ledger", force=True, backend="duckdb")


# --- verification -----------------------------------------------------------


def test_verify_is_clean_for_an_intact_ledger(tmp_path):
    from xtrax.telemetry.record import IRRef

    root = tmp_path / "ledger"
    store = BlobStore(blobs_dir(root))
    sha, size = store.put("intact ir")
    with RunLedger.open("run-1", root=root) as ledger:
        ledger.record_ir(IRRef(kind="jaxpr", sha256=sha, bytes=size))
    assert verify_ledger(root) == []


def test_verify_reports_a_missing_blob(tmp_path):
    from xtrax.telemetry.record import IRRef

    root = tmp_path / "ledger"
    store = BlobStore(blobs_dir(root))
    sha, size = store.put("soon to vanish")
    with RunLedger.open("run-1", root=root) as ledger:
        ledger.record_ir(IRRef(kind="jaxpr", sha256=sha, bytes=size))
    store.delete(sha)
    problems = verify_ledger(root)
    assert len(problems) == 1
    assert "missing" in problems[0]


def test_verify_ignores_hash_only_refs(tmp_path):
    """A hash_only ref never claimed a blob exists, so it is not a problem."""
    from xtrax.telemetry.record import IRRef

    root = tmp_path / "ledger"
    with RunLedger.open("run-1", root=root) as ledger:
        ledger.record_ir(
            IRRef(kind="jaxpr", sha256="a" * 64, bytes=1, mode="hash_only", reason="too big")
        )
    assert verify_ledger(root) == []
