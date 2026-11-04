"""Tests for the aminx done-marker v3 and alynxr fold-stats fragment record adapters.

Formats read from origin/main on 2026-11-03: aminx ``host/campaign.py`` (``_write_done_marker``,
``_validate_done_marker``; 7291397c) and alynxr ``fold_stats/layout.py`` (``write_part``,
``reusable``; f08806c). Each adapter has a negative control per classification.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from xtrax import pack
from xtrax.run import unit_index as ui
from xtrax.run import unit_index_adapters as ua
from xtrax.run import zarr_commit as zc
from xtrax.run.zarr_integrity import zarr_content_digest

# ---- aminx done marker v3 ------------------------------------------------------------------------


def aminx_unit(
    store: Path,
    name: str,
    *,
    row: str,
    inp: str | None,
    schema: str = ua.AMINX_V3,
    values: object = None,
) -> Path:
    out = store / name
    zc.write_staged_group(out, {"arr": np.arange(4) if values is None else values})
    marker = {
        "schema_version": schema,
        "manifest_row_hash": row,
        "attempt_id": f"att-{name}",
        "output_h5_path": str(out.resolve()),
        "content_digest_sha256": zarr_content_digest(out),
        "lock_backend": "file",
        "completed_at_unix_s": 1.0,
    }
    if inp is not None:
        marker["input_hash"] = inp
    out.with_name(out.name + ".done.json").write_text(json.dumps(marker, sort_keys=True))
    return out


def test_aminx_find_maps_v3_markers_only(tmp_path: Path) -> None:
    aminx_unit(tmp_path, "a.zarr", row="rowA", inp="inA")
    aminx_unit(tmp_path / "sub", "b.zarr", row="rowB", inp="inB")
    aminx_unit(tmp_path, "old.zarr", row="rowC", inp=None, schema=ua.AMINX_V2)
    aminx_unit(tmp_path, "nohash.zarr", row="rowD", inp=None)  # v3 without input_hash
    (tmp_path / "x.zarr.done.json.tmp.att").write_text("{}")  # a temp marker
    got = dict(ua.AminxDoneMarkerAdapter().find(tmp_path))
    assert set(got) == {"a.zarr", "sub/b.zarr"}
    core = got["a.zarr"]
    assert (core.unit_key, core.input_digest, core.run_id) == ("rowA", "inA", "att-a.zarr")
    assert core.content_digest == zarr_content_digest(tmp_path / "a.zarr")
    assert dict(ua.AminxDoneMarkerAdapter().find(tmp_path))["a.zarr"] == core  # stable digest


def test_aminx_lookup_outcomes(tmp_path: Path) -> None:
    store, idx = tmp_path / "camp", tmp_path / "idx"
    out = aminx_unit(store, "a.zarr", row="rowA", inp="inA")
    ui.rebuild(idx, [(store, ua.AminxDoneMarkerAdapter())])
    snap = ui.load_index(idx)
    got = ui.lookup(snap, store, "rowA", "inA")
    assert isinstance(got, ui.IndexReuse) and got.locator == "a.zarr"
    assert isinstance(ui.lookup(snap, store, "rowA", "other"), ui.IndexStale)
    assert isinstance(ui.lookup(snap, store, "rowB", "inA"), zc.Missing)
    chunk = next(p for p in (out / "arr").rglob("*") if p.is_file() and p.name != "zarr.json")
    chunk.write_bytes(chunk.read_bytes() + b"x")
    bad = ui.lookup(snap, store, "rowA", "inA")
    assert isinstance(bad, zc.Corrupt) and "content" in bad.reason
    assert isinstance(ui.lookup(snap, store, "rowA", "inA", verify=False), ui.IndexReuse)


def test_aminx_marker_without_output_is_corrupt(tmp_path: Path) -> None:
    out = aminx_unit(tmp_path, "a.zarr", row="rowA", inp="inA")
    ad = ua.AminxDoneMarkerAdapter()
    import shutil

    shutil.rmtree(out)
    got = ad.classify(tmp_path, "a.zarr", "inA", verify=True)
    assert isinstance(got, zc.Corrupt) and "output" in got.reason


def test_aminx_marker_gone_is_missing_and_v2_is_schema_mismatch(tmp_path: Path) -> None:
    ad = ua.AminxDoneMarkerAdapter()
    assert isinstance(ad.classify(tmp_path, "nothing.zarr", "inA", verify=True), zc.Missing)
    aminx_unit(tmp_path, "old.zarr", row="rowC", inp=None, schema=ua.AMINX_V2)
    got = ad.classify(tmp_path, "old.zarr", "inA", verify=True)
    assert isinstance(got, ui.SchemaMismatch) and ua.AMINX_V2 in got.reason


def test_aminx_locator_outside_the_store_is_corrupt(tmp_path: Path) -> None:
    """A row is not trusted: a verifying artifact outside the store must not be reused."""
    store = tmp_path / "camp"
    store.mkdir()
    aminx_unit(tmp_path, "evil.zarr", row="rowA", inp="inA")  # beside the store, not in it
    (store / "link.zarr").symlink_to(tmp_path / "evil.zarr")
    (store / "link.zarr.done.json").write_text((tmp_path / "evil.zarr.done.json").read_text())
    ad = ua.AminxDoneMarkerAdapter()
    for loc in ("../evil.zarr", str(tmp_path / "evil.zarr"), "./x.zarr", "a//b"):
        got = ad.classify(store, loc, "inA", verify=True)
        assert isinstance(got, zc.Corrupt) and "inside the store" in got.reason, loc
    # A symlink inside the store is the owner's (an output dir on scratch, say): it is followed.
    assert isinstance(ad.classify(store, "link.zarr", "inA", verify=True), ui.IndexReuse)
    aminx_unit(store, "ok.zarr", row="rowB", inp="inB")
    assert isinstance(ad.classify(store, "ok.zarr", "inB", verify=True), ui.IndexReuse)


def test_aminx_rows_need_no_explicit_adapter_table(tmp_path: Path) -> None:
    """Rows name their adapter; lookup finds the built-in ones without an ``adapters`` table."""
    store = tmp_path / "camp"
    aminx_unit(store, "a.zarr", row="rowA", inp="inA")
    ui.rebuild(tmp_path / "idx", [(store, ua.AminxDoneMarkerAdapter())])
    assert isinstance(ui.lookup(tmp_path / "idx", store, "rowA", "inA"), ui.IndexReuse)


def test_adapter_schema_mismatch_ranks_below_corrupt_above_stale(tmp_path: Path) -> None:
    store = tmp_path / "camp"
    aminx_unit(store, "a.zarr", row="rowA", inp="inA")
    ui.rebuild(tmp_path / "idx", [(store, ua.AminxDoneMarkerAdapter())])
    # The marker is rewritten as v2 after indexing: the artifact, not the row, decides.
    marker = store / "a.zarr.done.json"
    d = json.loads(marker.read_text())
    marker.write_text(json.dumps(dict(d, schema_version=ua.AMINX_V2)))
    assert isinstance(ui.lookup(tmp_path / "idx", store, "rowA", "inA"), ui.SchemaMismatch)


# ---- alynxr fold-stats fragment ------------------------------------------------------------------

PART = "metrics/arm=a1/source=derived/fn=ipsae@1/part-0123456789ab-000000000000.parquet"


def alynxr_part(
    store: Path,
    rel: str = PART,
    *,
    inp: str = "inS",
    data: bytes = b"PAR1data",
    run_id: str | None = "run-9",
) -> Path:
    part = store / rel
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(data)
    frag = {
        "schema": ua.ALYNXR_FRAGMENT,
        "part": rel,
        "table": rel.split("/")[0],
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "file_bytes": len(data),
        "logical_sha256": "L" * 64,
        "n_rows": 3,
        "inputs_sha256": inp,
        "fn": "ipsae@1",
        "design_ids": [],
        "run_id": run_id,
        "written_at": "2026-11-03T00:00:00Z",
    }
    (store / "parts").mkdir(parents=True, exist_ok=True)
    name = rel.removesuffix(".parquet").replace("/", "__") + ".json"
    (store / "parts" / name).write_text(json.dumps(frag))
    return part


def test_alynxr_find_and_lookup(tmp_path: Path) -> None:
    store, idx = tmp_path / "fs", tmp_path / "idx"
    part = alynxr_part(store)
    (core,) = [c for _, c in ua.AlynxrFragmentAdapter().find(store)]
    assert (core.unit_key, core.input_digest, core.content_digest) == (PART, "inS", "L" * 64)
    ui.rebuild(idx, [(store, ua.AlynxrFragmentAdapter())])
    snap = ui.load_index(idx)
    got = ui.lookup(snap, store, PART, "inS")
    assert isinstance(got, ui.IndexReuse) and got.locator == PART
    assert isinstance(ui.lookup(snap, store, PART, "other"), ui.IndexStale)
    part.write_bytes(b"PAR1tampered")
    assert isinstance(ui.lookup(snap, store, PART, "inS"), zc.Corrupt)
    assert isinstance(ui.lookup(snap, store, PART, "inS", verify=False), ui.IndexReuse)
    part.unlink()
    assert isinstance(
        ui.lookup(snap, store, PART, "inS"), zc.Missing
    )  # not visible yet, as zarr_commit (A1)


def test_alynxr_null_run_id_and_missing_fragment(tmp_path: Path) -> None:
    alynxr_part(tmp_path, run_id=None)
    ((_, core),) = ua.AlynxrFragmentAdapter().find(tmp_path)
    assert core.run_id == ""
    ad = ua.AlynxrFragmentAdapter()
    other = PART.replace("0123456789ab", "ba9876543210")
    assert isinstance(ad.classify(tmp_path, other, "inS", verify=True), zc.Missing)


def test_alynxr_current_schema_missing_fields_is_corrupt(tmp_path: Path) -> None:
    alynxr_part(tmp_path)
    frag_path = next((tmp_path / "parts").glob("*.json"))
    d = json.loads(frag_path.read_text())
    del d["logical_sha256"]
    frag_path.write_text(json.dumps(d))
    got = ua.AlynxrFragmentAdapter().classify(tmp_path, PART, "inS", verify=True)
    assert isinstance(got, zc.Corrupt) and "logical_sha256" in got.reason


def test_alynxr_part_not_visible_is_missing_either_way(tmp_path: Path) -> None:
    alynxr_part(tmp_path).unlink()
    got = ua.AlynxrFragmentAdapter().classify(tmp_path, PART, "inS", verify=False)
    assert got == zc.Missing()
    assert ua.AlynxrFragmentAdapter().classify(tmp_path, PART, "inS", verify=True) == zc.Missing()


def test_alynxr_refuses_paths_outside_the_store(tmp_path: Path) -> None:
    ad = ua.AlynxrFragmentAdapter()
    for bad in ("../x.parquet", "/etc/x.parquet", "predictions/../../x.parquet"):
        got = ad.classify(tmp_path, bad, "inS", verify=True)
        assert isinstance(got, zc.Corrupt) and "part path" in got.reason


def test_alynxr_fragment_naming_another_part_is_corrupt(tmp_path: Path) -> None:
    alynxr_part(tmp_path)
    frag_path = next((tmp_path / "parts").glob("*.json"))
    d = json.loads(frag_path.read_text())
    frag_path.write_text(json.dumps(dict(d, part="predictions/arm=z/part-x.parquet")))
    got = ua.AlynxrFragmentAdapter().classify(tmp_path, PART, "inS", verify=True)
    assert isinstance(got, zc.Corrupt)
    with pytest.raises(ui.UnitIndexError):
        list(ua.AlynxrFragmentAdapter().find(tmp_path))


# ---- unit.json -----------------------------------------------------------------------------------


def json_unit(store: Path, name: str, *, inp: str = "inU", run_id: str = "run-1") -> Path:
    unit = store / name
    (unit / "fold").mkdir(parents=True)
    (unit / "a.txt").write_bytes(b"alpha")
    (unit / "fold" / "b.txt").write_bytes(b"beta")
    pack.write_unit_record(unit, input_digest=inp, run_id=run_id)
    return unit


def test_unit_json_find_maps_the_record_and_stops_at_the_unit(tmp_path: Path) -> None:
    unit = json_unit(tmp_path, "arm/u1")
    json_unit(tmp_path, "arm/u2")
    (unit / "fold" / "nested").mkdir()
    (unit / "fold" / "nested" / "unit.json").write_text("not a unit, a member")
    got = dict(ua.UnitJsonAdapter().find(tmp_path))
    assert set(got) == {"arm/u1", "arm/u2"}
    record = json.loads((unit / "unit.json").read_text())
    core = got["arm/u1"]
    assert core == ui.CoreRecord(
        "arm/u1", "inU", record["content_digest"], record["record_digest"], "run-1"
    )


def test_unit_json_row_stays_valid_after_packing(tmp_path: Path) -> None:
    """§3.1: a pack adds ``storage`` outside the digest, so the index needs no rewrite."""
    store, idx = tmp_path / "s", tmp_path / "idx"
    unit = json_unit(store, "u1")
    ui.rebuild(idx, [(store, ua.UnitJsonAdapter())])
    snap = ui.load_index(idx)
    loose = ui.lookup(snap, store, "u1", "inU")
    assert isinstance(loose, ui.IndexReuse)
    pack.pack_unit(unit)
    packed = ui.lookup(snap, store, "u1", "inU")
    assert isinstance(packed, ui.IndexReuse) and packed.core == loose.core
    assert dict(ua.UnitJsonAdapter().find(store))["u1"] == loose.core


@pytest.mark.parametrize("packed", [False, True])
def test_unit_json_lookup_outcomes(tmp_path: Path, packed: bool) -> None:
    store, idx = tmp_path / "s", tmp_path / "idx"
    unit = json_unit(store, "u1")
    if packed:
        pack.pack_unit(unit)
    ui.rebuild(idx, [(store, ua.UnitJsonAdapter())])
    snap = ui.load_index(idx)
    assert isinstance(ui.lookup(snap, store, "u1", "inU"), ui.IndexReuse)
    assert isinstance(ui.lookup(snap, store, "u1", "other"), ui.IndexStale)
    assert isinstance(ui.lookup(snap, store, "nope", "inU"), zc.Missing)


def test_unit_json_tampered_loose_member_is_corrupt(tmp_path: Path) -> None:
    unit = json_unit(tmp_path, "u1")
    ad = ua.UnitJsonAdapter()
    (unit / "a.txt").write_bytes(b"tampered")
    got = ad.classify(tmp_path, "u1", "inU", verify=True)
    assert isinstance(got, zc.Corrupt) and "content" in got.reason
    assert isinstance(ad.classify(tmp_path, "u1", "inU", verify=False), ui.IndexReuse)


def test_unit_json_extra_loose_member_is_corrupt(tmp_path: Path) -> None:
    unit = json_unit(tmp_path, "u1")
    (unit / "stray.txt").write_bytes(b"not committed")
    got = ua.UnitJsonAdapter().classify(tmp_path, "u1", "inU", verify=True)
    assert isinstance(got, zc.Corrupt)


def test_unit_json_tampered_pack_is_corrupt(tmp_path: Path) -> None:
    unit = json_unit(tmp_path, "u1")
    pack.pack_unit(unit)
    entry = json.loads((unit / pack.PACK_INDEX).read_text())["members"]["a.txt"]
    raw = bytearray((unit / pack.PACK_FILE).read_bytes())
    raw[entry["offset"]] ^= 0xFF
    (unit / pack.PACK_FILE).write_bytes(bytes(raw))
    ad = ua.UnitJsonAdapter()
    assert isinstance(ad.classify(tmp_path, "u1", "inU", verify=True), zc.Corrupt)
    assert isinstance(ad.classify(tmp_path, "u1", "inU", verify=False), ui.IndexReuse)


@pytest.mark.parametrize("name", [pack.PACK_FILE, pack.PACK_INDEX])
def test_unit_json_pack_not_visible_is_missing(tmp_path: Path, name: str) -> None:
    """Spec A1: a file another node wrote may not be visible yet; that is Missing, not Corrupt."""
    unit = json_unit(tmp_path, "u1")
    pack.pack_unit(unit)
    (unit / name).unlink()
    assert ua.UnitJsonAdapter().classify(tmp_path, "u1", "inU", verify=True) == zc.Missing()


def test_unit_json_record_faults_classify(tmp_path: Path) -> None:
    unit = json_unit(tmp_path, "u1")
    ad = ua.UnitJsonAdapter()
    path = unit / "unit.json"
    good = path.read_text()
    record = json.loads(good)

    path.write_text(json.dumps(dict(record, input_digest="forged")))
    forged = ad.classify(tmp_path, "u1", "forged", verify=False)
    assert isinstance(forged, zc.Corrupt) and "record_digest" in forged.reason

    path.write_text("{torn")
    assert isinstance(ad.classify(tmp_path, "u1", "inU", verify=True), zc.Corrupt)
    path.write_text("[1]")
    assert isinstance(ad.classify(tmp_path, "u1", "inU", verify=True), zc.Corrupt)
    path.write_text(json.dumps(dict(record, schema="xtrax.unit/99")))
    assert isinstance(ad.classify(tmp_path, "u1", "inU", verify=True), ui.SchemaMismatch)
    path.unlink()
    assert ad.classify(tmp_path, "u1", "inU", verify=True) == zc.Missing()


def test_unit_json_find_raises_on_a_damaged_record_and_skips_a_foreign_one(tmp_path: Path) -> None:
    unit = json_unit(tmp_path, "u1")
    ad = ua.UnitJsonAdapter()
    path = unit / "unit.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps(dict(record, schema="someone-elses/1")))
    assert list(ad.find(tmp_path)) == []
    for bad in ("{torn", "[1]", json.dumps(dict(record, input_digest="forged"))):
        path.write_text(bad)
        with pytest.raises(ui.UnitIndexError):
            list(ad.find(tmp_path))


def test_unit_json_refuses_locators_outside_the_store(tmp_path: Path) -> None:
    store = tmp_path / "s"
    store.mkdir()
    json_unit(tmp_path, "evil")
    ad = ua.UnitJsonAdapter()
    for loc in ("../evil", str(tmp_path / "evil"), "./x", "a//b", ""):
        got = ad.classify(store, loc, "inU", verify=True)
        assert isinstance(got, zc.Corrupt) and "inside the store" in got.reason, loc


def test_unit_json_rows_need_no_explicit_adapter_table(tmp_path: Path) -> None:
    json_unit(tmp_path / "s", "u1")
    ui.rebuild(tmp_path / "idx", [(tmp_path / "s", ua.UnitJsonAdapter())])
    assert isinstance(ui.lookup(tmp_path / "idx", tmp_path / "s", "u1", "inU"), ui.IndexReuse)
