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

from xtrax.run import unit_index as ui
from xtrax.run import unit_index_adapters as ua
from xtrax.run import zarr_commit as zc
from xtrax.run.zarr_integrity import zarr_content_digest

# ---- aminx done marker v3 ----------------------------------------------------------------------


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
    for loc in ("../evil.zarr", str(tmp_path / "evil.zarr"), "./x.zarr", "a//b", "link.zarr"):
        got = ad.classify(store, loc, "inA", verify=True)
        assert isinstance(got, zc.Corrupt) and "inside the store" in got.reason, loc
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


# ---- alynxr fold-stats fragment ----------------------------------------------------------------

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
    assert isinstance(ui.lookup(snap, store, PART, "inS"), zc.Corrupt)


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


def test_alynxr_missing_part_is_corrupt_without_verify(tmp_path: Path) -> None:
    alynxr_part(tmp_path).unlink()
    got = ua.AlynxrFragmentAdapter().classify(tmp_path, PART, "inS", verify=False)
    assert isinstance(got, zc.Corrupt) and "missing" in got.reason


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
