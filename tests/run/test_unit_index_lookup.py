"""Tests for the unit-index read API: lookup, record adapters, rebuild and the DuckDB view.

Spec 261030 §3.1. Every lookup outcome has a negative control: stale input digest, a row whose
artifact is not visible yet (Missing, never Corrupt), a schema-version-2 row, and a row whose
record_digest no longer matches the artifact (classified from the artifact, which is the truth).
"""

import dataclasses
import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from xtrax.run import unit_index as ui
from xtrax.run import zarr_commit as zc

WID = "node-1-777-0-abcdef12"
W2 = "node-2-778-0-abcdef22"


def make_store(tmp_path: Path, name: str = "store") -> Path:
    store = tmp_path / name
    zc.create_store(
        store,
        identity_payload={"type": "test"},
        creator_run_id="creator",
        prefixes=(("chunks",),),
        writer_id="writer",
    )
    return store


def commit(store: Path, k: str, digest: str, values: object = None) -> zc.CommitRecord:
    staged = zc.staging_root(store) / "writer" / k
    zc.write_staged_group(staged, {"arr": np.arange(5) if values is None else values})
    out = zc.commit_key(store, ("chunks", k), staged, input_digest=digest, run_id="run1", env={})
    return out.record


def index_store(root: Path, store: Path, writer: str = WID) -> list[ui.IndexRow]:
    """Append one index row per committed key, the way a consumer would after each commit."""
    rows = [
        ui.make_row(store, locator, core, adapter=ui.ZARR_COMMIT, writer_id=writer)
        for locator, core in ui.ZarrCommitAdapter().find(store)
    ]
    with ui.SegmentWriter(root, writer_id=writer) as w:
        for r in rows:
            w.append(r)
    return rows


# ---- lookup outcomes -------------------------------------------------------------------------


def test_lookup_reuse(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    rec = commit(store, "k0", "d0")
    index_store(tmp_path / "idx", store)
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, ui.IndexReuse)
    assert got.locator == "chunks/k0" and got.store == os.path.realpath(store)
    assert got.core.record_digest == rec.record_digest


def test_lookup_stale_input_digest(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    index_store(tmp_path / "idx", store)
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "other")
    assert isinstance(got, ui.IndexStale) and got.core.input_digest == "d0"


def test_lookup_missing_without_row(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")  # committed but never indexed: a rebuild would add it
    assert isinstance(ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0"), zc.Missing)


def test_lookup_artifact_not_visible_is_missing_not_corrupt(tmp_path: Path) -> None:
    """A1: a new artifact can take seconds to appear on another node; its row may arrive first."""
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    index_store(tmp_path / "idx", store)
    os.rename(store / "chunks" / "k0", tmp_path / "elsewhere")
    assert isinstance(ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0"), zc.Missing)


def test_lookup_corrupt_content_and_verify_off(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    index_store(tmp_path / "idx", store)
    chunk = next(
        p
        for p in (store / "chunks" / "k0" / "arr").rglob("*")
        if p.is_file() and p.name != "zarr.json"
    )
    chunk.write_bytes(chunk.read_bytes() + b"x")
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, zc.Corrupt) and "digest" in got.reason
    assert isinstance(
        ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0", verify=False), ui.IndexReuse
    )


def test_lookup_schema_mismatch(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    (row,) = index_store(tmp_path / "other", store)
    seg = tmp_path / "idx" / "segments"
    seg.mkdir(parents=True)
    (seg / f"{WID}.jsonl").write_text(json.dumps(dict(row.to_dict(), schema_version=2)) + "\n")
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, ui.SchemaMismatch) and "schema_version 2" in got.reason


def test_lookup_classifies_from_the_artifact_not_the_row(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    rec = commit(store, "k0", "d0")
    (row,) = index_store(tmp_path / "other", store)
    stale_row = dataclasses.replace(row, record_digest="not-the-artifact", input_digest="old")
    with ui.SegmentWriter(tmp_path / "idx", writer_id=WID) as w:
        w.append(stale_row)
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, ui.IndexReuse) and got.core.record_digest == rec.record_digest


def test_snapshot_scans_the_index_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path)
    for i in range(5):
        commit(store, f"k{i}", f"d{i}")
    index_store(tmp_path / "idx", store)
    real, calls = ui._scan, []

    def counting(root: Path):  # noqa: ANN202
        calls.append(root)
        return real(root)

    monkeypatch.setattr(ui, "_scan", counting)
    snap = ui.load_index(tmp_path / "idx")
    for _ in range(4):
        for i in range(5):
            assert isinstance(ui.lookup(snap, store, f"chunks/k{i}", f"d{i}"), ui.IndexReuse)
    assert len(calls) == 1


def test_symlinked_store_spellings_meet(tmp_path: Path) -> None:
    """Rows carry the store's resolved path, so a symlinked spelling finds the same rows."""
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    link = tmp_path / "link"
    link.symlink_to(store)
    index_store(tmp_path / "idx", link)  # written through the symlink
    assert isinstance(ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0"), ui.IndexReuse)
    assert isinstance(ui.lookup(tmp_path / "idx", link, "chunks/k0", "d0"), ui.IndexReuse)
    (row,) = ui.iter_rows(tmp_path / "idx")
    assert row.store == os.path.realpath(store)


def test_segment_writer_canonicalizes_store(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    link = tmp_path / "link"
    link.symlink_to(store)
    (row,) = index_store(tmp_path / "other", store)
    with ui.SegmentWriter(tmp_path / "idx", writer_id=WID) as w:
        w.append(dataclasses.replace(row, store=str(link)))
    (got,) = ui.iter_rows(tmp_path / "idx")
    assert got.store == os.path.realpath(store)


# ---- several rows for one unit -----------------------------------------------------------------


@dataclasses.dataclass
class FakeAdapter:
    name: str
    outcomes: dict[str, object]

    def find(self, store: Path) -> Iterator[tuple[str, ui.CoreRecord]]:
        return iter(())

    def classify(self, store: Path, locator: str, input_digest: str, *, verify: bool) -> object:
        return self.outcomes[locator]


def _rows(root: Path, store: Path, locators: list[str], adapter: str) -> None:
    with ui.SegmentWriter(root, writer_id=WID) as w:
        for loc in locators:
            core = ui.CoreRecord("u", "d", "c", f"r-{loc}", "run")
            w.append(ui.make_row(store, loc, core, adapter=adapter, writer_id=WID))


def test_any_reusable_copy_wins(tmp_path: Path) -> None:
    """The reusable copy is the OLDER row, so newest-first order classifies Corrupt first."""
    core = ui.CoreRecord("u", "d", "c", "r", "run")
    reuse = ui.IndexReuse(str(tmp_path), "b", core)
    fake = FakeAdapter("fake", {"a": zc.Corrupt("bad"), "b": reuse})
    _rows(tmp_path / "idx", tmp_path, ["b", "a"], "fake")
    rows = sorted(ui.iter_rows(tmp_path / "idx"), key=lambda r: -r.ts)
    assert rows[0].locator == "a"  # the Corrupt copy really is tried first
    assert ui.lookup(tmp_path / "idx", tmp_path, "u", "d", adapters={"fake": fake}) == reuse


def test_row_pointing_at_another_units_record_is_not_reuse(tmp_path: Path) -> None:
    """A row for chunks/k0 whose locator is chunks/k1 must not reuse k1's artifact."""
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    commit(store, "k1", "d0")  # same input digest, another unit
    rows = {r.locator: r for r in index_store(tmp_path / "other", store)}
    bad = dataclasses.replace(rows["chunks/k1"], unit_key="chunks/k0", record_digest="bad")
    with ui.SegmentWriter(tmp_path / "idx", writer_id=WID) as w:
        w.append(bad)
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, zc.Corrupt) and "chunks/k1" in got.reason
    with ui.SegmentWriter(tmp_path / "idx", writer_id=W2) as w:
        w.append(
            dataclasses.replace(rows["chunks/k0"], writer_id=W2)
        )  # the real row: the bad copy no longer hides it
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, ui.IndexReuse) and got.locator == "chunks/k0"


def test_reuse_for_another_input_is_corrupt(tmp_path: Path) -> None:
    core = ui.CoreRecord("u", "other", "c", "r", "run")
    fake = FakeAdapter("fake", {"a": ui.IndexReuse(str(tmp_path), "a", core)})
    _rows(tmp_path / "idx", tmp_path, ["a"], "fake")
    got = ui.lookup(tmp_path / "idx", tmp_path, "u", "d", adapters={"fake": fake})
    assert isinstance(got, zc.Corrupt) and "reuse" in got.reason


def test_illegal_zarr_locator_is_corrupt_and_scan_continues(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    (row,) = index_store(tmp_path / "idx", store)
    with ui.SegmentWriter(tmp_path / "idx", writer_id=W2) as w:
        w.append(
            dataclasses.replace(
                row, locator="chunks//k0", record_digest="x", ts=row.ts + 9, writer_id=W2
            )
        )
    ad = ui.ZarrCommitAdapter()
    assert isinstance(ad.classify(store, "chunks/..", "d0", verify=True), zc.Corrupt)
    assert isinstance(ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0"), ui.IndexReuse)


def test_relative_and_tilde_store_spellings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="relative"):
        ui.canonical_store("store")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ui.canonical_store("~/s") == os.path.realpath(tmp_path / "s")


def test_schema_mismatch_found_through_a_symlinked_spelling(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    link = tmp_path / "link"
    link.symlink_to(store)
    (row,) = index_store(tmp_path / "other", store)
    seg = tmp_path / "idx" / "segments"
    seg.mkdir(parents=True)
    raw = dict(row.to_dict(), schema_version=2, store=str(link) + "/")
    (seg / f"{WID}.jsonl").write_text(json.dumps(raw) + "\n")
    assert isinstance(ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0"), ui.SchemaMismatch)


def test_make_row_refuses_what_readers_drop(tmp_path: Path) -> None:
    core = ui.CoreRecord("u", "", "c", "r", "run")
    with pytest.raises(ui.RowError, match="input_digest"):
        ui.make_row(tmp_path, "a", core, adapter="fake", writer_id=WID)


def test_rebuild_skips_a_record_no_reader_would_see(tmp_path: Path) -> None:
    @dataclasses.dataclass
    class Finder:
        name: str = "fake"

        def find(self, store: Path) -> Iterator[tuple[str, ui.CoreRecord]]:
            yield "good", ui.CoreRecord("u", "d", "c", "r", "run")
            yield "bad", ui.CoreRecord("v", "", "c", "r2", "run")

        def classify(self, *a: object, **k: object) -> object:
            raise AssertionError

    rep = ui.rebuild(tmp_path / "idx", [(tmp_path, Finder())])
    assert rep.rows == 1 and rep.skipped == (f"{os.path.realpath(tmp_path)}:bad",)
    assert [r.locator for r in ui.iter_rows(tmp_path / "idx")] == ["good"]


def test_corrupt_outranks_stale_and_missing(tmp_path: Path) -> None:
    core = ui.CoreRecord("u", "old", "c", "r", "run")
    fake = FakeAdapter(
        "fake",
        {"a": zc.Missing(), "b": ui.IndexStale(str(tmp_path), "b", core), "c": zc.Corrupt("x")},
    )
    _rows(tmp_path / "idx", tmp_path, ["a", "b", "c"], "fake")
    got = ui.lookup(tmp_path / "idx", tmp_path, "u", "d", adapters={"fake": fake})
    assert got == zc.Corrupt("x")
    fake.outcomes["c"] = zc.Missing()
    got = ui.lookup(tmp_path / "idx", tmp_path, "u", "d", adapters={"fake": fake})
    assert isinstance(got, ui.IndexStale)


def test_unknown_adapter_refused(tmp_path: Path) -> None:
    _rows(tmp_path / "idx", tmp_path, ["a"], "nobody")
    with pytest.raises(ui.UnitIndexError, match="nobody"):
        ui.lookup(tmp_path / "idx", tmp_path, "u", "d")


# ---- rebuild -----------------------------------------------------------------------------------


def test_rebuild_restores_a_missing_row(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    rec = commit(store, "k0", "d0")
    commit(store, "k1", "d1")
    assert isinstance(ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0"), zc.Missing)
    rep = ui.rebuild(tmp_path / "idx", [(store, ui.ZarrCommitAdapter())])
    assert rep.rows == 2 and Path(rep.path).parent.name == "rebuilt"
    got = ui.lookup(tmp_path / "idx", store, "chunks/k0", "d0")
    assert isinstance(got, ui.IndexReuse) and got.core.record_digest == rec.record_digest


def test_rebuild_twice_adds_a_file_and_no_rows(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    index_store(tmp_path / "idx", store)
    before = {r.read_key for r in ui.iter_rows(tmp_path / "idx")}
    ui.rebuild(tmp_path / "idx", [(store, ui.ZarrCommitAdapter())])
    ui.rebuild(tmp_path / "idx", [(store, ui.ZarrCommitAdapter())])
    assert {r.read_key for r in ui.iter_rows(tmp_path / "idx")} == before
    assert len(list((tmp_path / "idx" / "rebuilt").glob("*.jsonl"))) == 2


def test_rebuild_of_nothing_writes_nothing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    rep = ui.rebuild(tmp_path / "idx", [(store, ui.ZarrCommitAdapter())])
    assert rep.rows == 0 and rep.path is None and not (tmp_path / "idx").exists()


# ---- DuckDB view -------------------------------------------------------------------------------


def _messy_index(root: Path, store: Path) -> None:
    """Segments with a torn tail and a trailer, the same rows sealed (deferred archive), a
    trailer-only segment, a manifest, a schema-2 row and a rebuilt copy."""
    rows = index_store(root, store)
    ui.compact(root, runner=lambda a, t: (1, "", "down"))  # sealed + manifest, inputs stay
    lone = ui.SegmentWriter(root, writer_id="node-2-778-0-abcdef13").open()
    lone.close()  # trailer only
    torn = ui.SegmentWriter(root, writer_id="node-3-779-0-abcdef14").open()
    extra = dataclasses.replace(
        rows[0], locator="chunks/extra", record_digest="r-extra", writer_id=torn.writer_id
    )
    torn.append(extra)
    os.write(torn._fd, b'{"schema_version": 1, "store": "/x", "loc')  # torn tail
    os.write(torn._fd, b"\n" + json.dumps(dict(rows[0].to_dict(), schema_version=2)).encode())
    os.write(torn._fd, b"\n")
    ui.rebuild(root, [(store, ui.ZarrCommitAdapter())])
    assert list((root / "sealed").glob(f"*{ui.MANIFEST_SUFFIX}"))


def test_duckdb_view_agrees_with_iter_rows(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    store = make_store(tmp_path)
    for i in range(3):
        commit(store, f"k{i}", f"d{i}")
    _messy_index(tmp_path / "idx", store)
    report = ui.ReadReport()
    canonical = list(ui.iter_rows(tmp_path / "idx", report=report))
    assert report.torn and report.unreadable and report.duplicates  # the fixture bites
    assert _view_rows(tmp_path / "idx") == sorted(_full(r) for r in canonical)


def _full(row: ui.IndexRow) -> tuple:
    d = row.to_dict()
    d["reused_from"] = tuple(d["reused_from"]) if d["reused_from"] is not None else None
    return tuple(d[c] for c in ui._VIEW_COLUMNS)


def _view_rows(root: Path) -> list[tuple]:
    con = ui.duckdb_view(root)
    out = con.execute(f"SELECT {', '.join(ui._VIEW_COLUMNS)} FROM unit_index").fetchall()
    return sorted(tuple(tuple(v) if isinstance(v, list) else v for v in r) for r in out)


def test_duckdb_view_applies_the_readers_types(tmp_path: Path) -> None:
    """Values DuckDB would coerce (bool ts, string ts, a numeric store, a short reused_from,
    a boolean schema_version) are dropped by iter_rows, and so by the view."""
    pytest.importorskip("duckdb")
    good = ui.make_row(
        "/s", "a", ui.CoreRecord("u", "d", "c", "r", "run"), adapter="x", writer_id=WID, ts=1.0
    )
    bads = [
        dict(good.to_dict(), ts=True, record_digest="b1"),
        dict(good.to_dict(), ts="1.5", record_digest="b2"),
        dict(good.to_dict(), store=5, record_digest="b3"),
        dict(good.to_dict(), reused_from=["only", "two"], record_digest="b4"),
        dict(good.to_dict(), schema_version=True, record_digest="b5"),
        dict(good.to_dict(), run_id=7, record_digest="b6"),
    ]
    seg = tmp_path / "idx" / "segments"
    seg.mkdir(parents=True)
    lines = [json.dumps(good.to_dict())] + [json.dumps(b) for b in bads]
    (seg / f"{WID}.jsonl").write_text("\n".join(lines) + "\n")
    canonical = list(ui.iter_rows(tmp_path / "idx"))
    assert [r.record_digest for r in canonical] == ["r"]
    assert _view_rows(tmp_path / "idx") == [_full(good)]


def test_duckdb_view_keeps_the_copy_iter_rows_keeps(tmp_path: Path) -> None:
    """Duplicates of a read key differ in ts/writer_id; both readers keep the first in scan order
    (segments before sealed before rebuilt)."""
    pytest.importorskip("duckdb")
    core = ui.CoreRecord("u", "d", "c", "r", "run")
    other = "node-9-1-0-abcdef19"
    with ui.SegmentWriter(tmp_path / "idx", writer_id=other) as w:
        w.append(ui.make_row("/s", "a", core, adapter="x", writer_id=other, ts=50.0))
    rebuilt = tmp_path / "idx" / "rebuilt"
    rebuilt.mkdir()
    early = ui.make_row("/s", "a", core, adapter="x", writer_id=WID, ts=1.0)
    (rebuilt / "index-0-x.jsonl").write_text(early.to_json_line())
    (kept,) = ui.iter_rows(tmp_path / "idx")
    assert kept.writer_id == other  # a min-ts rule would have picked the rebuilt copy
    assert _view_rows(tmp_path / "idx") == [_full(kept)]


def test_duckdb_view_without_dedup_would_overcount(tmp_path: Path) -> None:
    """Negative control: the fixture really holds duplicates the view must remove."""
    duckdb = pytest.importorskip("duckdb")
    store = make_store(tmp_path)
    commit(store, "k0", "d0")
    _messy_index(tmp_path / "idx", store)
    files = [str(p) for d in ui.READ_DIRS for p in ui._listing(tmp_path / "idx", d)]
    (raw,) = (
        duckdb.connect()
        .execute(
            "SELECT count(*) FROM read_json(?, format='newline_delimited', ignore_errors=true, "
            "columns={'store': 'VARCHAR', 'schema_version': 'BIGINT'}) "
            "WHERE store IS NOT NULL AND schema_version = 1",
            [files],
        )
        .fetchone()
    )
    assert raw > len(list(ui.iter_rows(tmp_path / "idx")))


def test_duckdb_view_of_empty_index(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    con = ui.duckdb_view(tmp_path / "nothing")
    assert con.execute("SELECT count(*) FROM unit_index").fetchone() == (0,)


def test_duckdb_view_name_must_be_an_identifier(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="identifier"):
        ui.duckdb_view(tmp_path, name="x; DROP TABLE y")
