"""xtrax.pack: lifecycle, crash recovery, refusals, and the format shared with alynxr.

The fixture ``fixtures/alynxr_cd8e5a9`` was written by alynxr's own ``unit_pack.py`` at cd8e5a9 and
is checked in as bytes, so the cross-implementation read cannot drift with this code.
"""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

from xtrax import pack
from xtrax.pack import _node
from xtrax.run import zarr_commit
from xtrax.run.zarr_integrity import canonical_json_bytes, normalize_json_value

ALYNXR_FIXTURE = Path(__file__).parent / "fixtures" / "alynxr_cd8e5a9"
THREE_FILES = ["members.idx.json", "members.tar.zst", "unit.json"]


def make_unit(unit: Path, *, commit: bool = True) -> dict[str, bytes]:
    """A unit with a nested member, a dotfile, and a nested file named like a pack artefact."""
    files = {
        "query.fasta": b">d1\nACDE\n",
        ".hidden": b"a dotfile is a member\n",
        "fold/scores.json": b'{"ptm": 0.85}\n',
        "fold/empty.txt": b"",
        "fold/deep/model.cif": b"data_x\n" * 400,
        "fold/members.tar.zst": b"not this unit's pack\n",
    }
    for rel, data in files.items():
        path = unit / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if commit:
        pack.write_unit_record(unit, input_digest="in-1", run_id="run-1")
    return files


def rewrite_index(unit: Path, mutate: Any) -> None:
    """Tamper with the index and re-sign it in the record, so it is self-consistent."""
    idx_path = unit / pack.PACK_INDEX
    doc = json.loads(idx_path.read_text())
    mutate(doc["members"])
    _node.write_json_atomic(idx_path, doc)
    rec_path = unit / pack.UNIT_RECORD
    record = json.loads(rec_path.read_text())
    record["storage"]["index_sha256"] = hashlib.sha256(idx_path.read_bytes()).hexdigest()
    _node.write_json_atomic(rec_path, record)


def crash_at(unit: Path, step: str) -> None:
    with pytest.raises(RuntimeError, match=f"injected crash after {step}"):
        pack.pack_unit(unit, _crash_after=step)


# ---- lifecycle -----------------------------------------------------------------------------------


def test_pack_leaves_three_files_and_every_member_reads_back(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    assert pack.list_unit_members(unit) == sorted(files)
    record = pack.pack_unit(unit)
    assert sorted(p.name for p in unit.iterdir()) == THREE_FILES
    assert record["storage"]["n_members"] == len(files)
    assert pack.is_packed(unit)
    assert pack.list_unit_members(unit) == sorted(files)
    for rel, data in files.items():
        assert pack.read_unit_member(unit, rel) == data


def test_content_digest_is_the_same_loose_and_packed(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    before = json.loads((unit / pack.UNIT_RECORD).read_text())
    loose = pack.unit_content_digest(unit)
    after = pack.pack_unit(unit)
    assert loose == before["content_digest"]
    assert pack.unit_content_digest(unit) == loose
    # Packing never changes the record's identity (§3.1).
    assert after["record_digest"] == before["record_digest"]
    assert pack.compute_record_digest(after) == before["record_digest"]


def test_content_digest_changes_with_any_member_and_any_name(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    base = pack.unit_content_digest(unit)
    (unit / "query.fasta").write_bytes(b">d1\nACDF\n")
    assert pack.unit_content_digest(unit) != base
    (unit / "query.fasta").write_bytes(b">d1\nACDE\n")
    assert pack.unit_content_digest(unit) == base
    (unit / "query.fasta").rename(unit / "renamed.fasta")
    assert pack.unit_content_digest(unit) != base


def test_digest_is_canonical_json_of_sorted_pairs() -> None:
    shas = {"b": "2", "a": "1", "é": "3"}
    pairs = [["a", "1"], ["b", "2"], ["é", "3"]]
    text = json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)
    want = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert pack.digest_of_members(shas) == want


def test_whole_pack_is_a_plain_tar_after_decompression(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    pack.pack_unit(unit)
    tar_bytes = _node.zstd_decompress((unit / pack.PACK_FILE).read_bytes())
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        assert sorted(tf.getnames()) == sorted(files)
        for rel, data in files.items():
            member = tf.extractfile(rel)
            assert member is not None and member.read() == data
            assert tf.getmember(rel).mtime == 0


def test_random_access_touches_only_the_requested_frame(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    pack.pack_unit(unit)
    entry = json.loads((unit / pack.PACK_INDEX).read_text())["members"]["query.fasta"]
    raw = bytearray((unit / pack.PACK_FILE).read_bytes())
    raw[entry["offset"]] ^= 0xFF
    (unit / pack.PACK_FILE).write_bytes(bytes(raw))
    assert pack.read_unit_member(unit, "fold/scores.json") == files["fold/scores.json"]
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.read_unit_member(unit, "query.fasta")
    assert e.value.rid == "R-U2"
    # A digest reads every member, so it sees the damage a single read did not.
    with pytest.raises(pack.UnitPackRefused):
        pack.unit_content_digest(unit)


def test_pack_is_idempotent(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    first = pack.pack_unit(unit)
    pack_bytes = (unit / pack.PACK_FILE).read_bytes()
    assert pack.pack_unit(unit) == first
    assert (unit / pack.PACK_FILE).read_bytes() == pack_bytes


def test_min_members_is_a_count_of_member_files(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    record = pack.pack_unit(unit, min_members=len(files) + 1)
    assert "storage" not in record and not (unit / pack.PACK_FILE).exists()
    assert "storage" in pack.pack_unit(unit, min_members=len(files))


def test_a_unit_with_no_members_stays_loose(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    unit.mkdir()
    pack.write_unit_record(unit, input_digest="i", run_id="r")
    assert "storage" not in pack.pack_unit(unit)


def test_find_packed_unit(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    assert pack.find_packed_unit(unit / "fold" / "scores.json") is None
    pack.pack_unit(unit)
    assert pack.find_packed_unit(unit / "fold" / "scores.json") == (
        str(unit),
        "fold/scores.json",
    )
    assert pack.find_packed_unit(tmp_path / "elsewhere" / "x") is None


# ---- member set ----------------------------------------------------------------------------------


def test_member_set_includes_dotfiles_and_nested_lookalikes(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    members = pack.collect_loose_members(unit)
    assert ".hidden" in members and "fold/members.tar.zst" in members
    assert pack.UNIT_RECORD not in members


def test_member_set_excludes_record_artefacts_and_transients(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    (unit / "fold").mkdir(parents=True)
    (unit / "keep").write_bytes(b"k")
    for name in (
        "unit.json",
        "unit.json.tmp",
        "members.tar.zst",
        "members.tar.zst.partial",
        "members.idx.json",
        ".a.1.tmp",
        ".nfs0001",
        "fold/.b.2.tmp",
        "fold/.nfs0002",
    ):
        (unit / name).write_bytes(b"x")
    assert pack.collect_loose_members(unit) == ["keep"]
    (unit / "real.tmp").write_bytes(b"not a temp: no leading dot")
    assert "real.tmp" in pack.collect_loose_members(unit)


@pytest.mark.parametrize("kind", ["file", "dir"])
def test_symlink_in_a_unit_is_refused_not_dropped(tmp_path: Path, kind: str) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    target = tmp_path / "target"
    if kind == "file":
        target.write_bytes(b"t")
    else:
        target.mkdir()
    (unit / "link").symlink_to(target)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.write_unit_record(unit, input_digest="i", run_id="r")
    assert e.value.rid == "R-U3"
    assert not (unit / pack.UNIT_RECORD).exists()


def test_special_file_is_refused(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    unit.mkdir()
    os.mkfifo(unit / "pipe")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.collect_loose_members(unit)
    assert e.value.rid == "R-U3"


def test_path_ustar_cannot_hold_is_refused_and_leaves_the_unit_loose(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    long_dir = unit / ("d" * 120) / ("e" * 120) / ("f" * 120)
    long_dir.mkdir(parents=True)
    (long_dir / "m.txt").write_bytes(b"m")
    pack.write_unit_record(unit, input_digest="i", run_id="r")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_unit(unit)
    assert e.value.rid == "R-U3"
    assert not pack.is_packed(unit) and (long_dir / "m.txt").is_file()


# ---- crash recovery ------------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["tar_written", "tar_verified", "tar_renamed", "index_written"])
def test_crash_before_the_record_names_the_pack_keeps_every_loose_file(
    tmp_path: Path, step: str
) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    digest = pack.unit_content_digest(unit)
    crash_at(unit, step)
    assert not pack.is_packed(unit)
    for rel, data in files.items():
        assert (unit / rel).read_bytes() == data
    assert pack.unit_content_digest(unit) == digest
    pack.pack_unit(unit)
    assert sorted(p.name for p in unit.iterdir()) == THREE_FILES
    assert pack.unit_content_digest(unit) == digest


def test_crash_after_the_record_rewrite_is_finished_by_the_next_call(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    assert pack.is_packed(unit) and (unit / "query.fasta").is_file()
    pack.pack_unit(unit)  # the early-return path sweeps (§3.3, C6)
    assert sorted(p.name for p in unit.iterdir()) == THREE_FILES
    for rel, data in files.items():
        assert pack.read_unit_member(unit, rel) == data


def test_crash_partway_through_deletion_is_finished_by_the_sweep(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    crash_at(unit, "loose_member_deleted")
    assert sum((unit / rel).is_file() for rel in files) == len(files) - 1
    pack.pack_sweep(unit)
    assert sorted(p.name for p in unit.iterdir()) == THREE_FILES


_KILLER = """
import os, sys
from xtrax.pack import _node
step = sys.argv[2]
def hook(name, _crash_after):
    if name == step:
        os._exit(9)
_node._hook = hook
_node.pack_unit(sys.argv[1])
"""


@pytest.mark.parametrize("step", ["tar_written", "index_written", "stamp_rewritten"])
def test_a_killed_process_recovers(tmp_path: Path, step: str) -> None:
    """The process really dies: no finally, no flush, no cleanup after the step."""
    unit = tmp_path / "u"
    files = make_unit(unit)
    proc = subprocess.run(
        [sys.executable, "-c", _KILLER, str(unit), step], capture_output=True, check=False
    )
    assert proc.returncode == 9, proc.stderr
    pack.pack_unit(unit)
    assert sorted(p.name for p in unit.iterdir()) == THREE_FILES
    for rel, data in files.items():
        assert pack.read_unit_member(unit, rel) == data


# ---- refusals ------------------------------------------------------------------------------------


def test_unit_without_a_record_is_refused(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_unit(unit)
    assert e.value.rid == "R-U1"
    assert not (unit / pack.PACK_FILE).exists()


@pytest.mark.parametrize("content", [b"{not json", b"[1, 2]"])
def test_unreadable_record_is_refused(tmp_path: Path, content: bytes) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    (unit / pack.UNIT_RECORD).write_bytes(content)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_unit(unit)
    assert e.value.rid == "R-U1"


def test_members_that_differ_from_the_record_are_refused_before_the_pack_exists(
    tmp_path: Path,
) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    (unit / "query.fasta").write_bytes(b">tampered\n")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_unit(unit)
    assert e.value.rid == "R-U1" and "content_digest" in e.value.detail
    assert not (unit / pack.PACK_FILE).exists() and not pack.is_packed(unit)
    assert (unit / "fold/scores.json").read_bytes() == files["fold/scores.json"]


def test_a_tampered_record_core_is_refused(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    record = json.loads((unit / pack.UNIT_RECORD).read_text())
    record["input_digest"] = "other"
    _node.write_json_atomic(unit / pack.UNIT_RECORD, record)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_unit(unit)
    assert e.value.rid == "R-U1" and "record_digest" in e.value.detail


@pytest.mark.parametrize(
    ("relpath", "rid"),
    [
        ("fold/nope.cif", "R-U-unknown-member"),
        ("../outside.txt", "R-U-unknown-member"),
        ("/etc/passwd", "R-U-unknown-member"),
        ("fold/../query.fasta", "R-U-unknown-member"),
        ("unit.json", "R-U-unknown-member"),
        ("members.tar.zst", "R-U-unknown-member"),
    ],
)
def test_reads_refuse_what_is_not_a_member(tmp_path: Path, relpath: str, rid: str) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    (tmp_path / "outside.txt").write_bytes(b"outside")
    pack.pack_unit(unit)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.read_unit_member(unit, relpath)
    assert e.value.rid == rid


def test_loose_reads_refuse_traversal_too(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    (tmp_path / "outside.txt").write_bytes(b"outside")
    with pytest.raises(pack.UnitPackRefused):
        pack.read_unit_member(unit, "../outside.txt")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_bytes(b"x")
    (unit / "lnk").symlink_to(tmp_path / "out")
    with pytest.raises(pack.UnitPackRefused):
        pack.read_unit_member(unit, "lnk/x")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.read_unit_member(unit, "fold/nope.cif")
    assert e.value.rid == "R-U-unknown-member"  # the same id as for a packed unit


def test_index_that_does_not_match_the_record_is_refused(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    pack.pack_unit(unit)
    (unit / pack.PACK_INDEX).write_bytes((unit / pack.PACK_INDEX).read_bytes() + b" ")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.list_unit_members(unit)
    assert e.value.rid == "R-U2"


@pytest.mark.parametrize("missing", [pack.PACK_FILE, pack.PACK_INDEX])
def test_sweep_refuses_a_missing_pack_or_index(tmp_path: Path, missing: str) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    (unit / missing).unlink()
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_sweep(unit)
    assert e.value.rid == "R-U2"
    assert (unit / "query.fasta").is_file()


def test_sweep_refuses_a_pack_that_fails_its_sha(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    (unit / pack.PACK_FILE).write_bytes((unit / pack.PACK_FILE).read_bytes() + b"\0")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_sweep(unit)
    assert e.value.rid == "R-U2" and (unit / "query.fasta").is_file()


# ---- negative controls: an index is not trusted --------------------------------------------------


@pytest.mark.parametrize("key", ["../outside.txt", "/abs.txt", "fold/../../outside.txt", "./x"])
def test_sweep_never_deletes_through_a_self_consistent_index_key(tmp_path: Path, key: str) -> None:
    """alynxr's sweep at cd8e5a9 deletes ../outside.txt given this index; this one must not."""
    unit = tmp_path / "u"
    make_unit(unit)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"keep me")
    crash_at(unit, "stamp_rewritten")
    rewrite_index(unit, lambda m: m.__setitem__(key, m["query.fasta"]))
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_sweep(unit)
    assert e.value.rid == "R-U2"
    assert outside.read_bytes() == b"keep me"
    assert (unit / "query.fasta").is_file()  # refused before anything was deleted


@pytest.mark.parametrize("key", ["unit.json", "members.tar.zst", "members.idx.json"])
def test_sweep_never_deletes_the_record_or_pack(tmp_path: Path, key: str) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    rewrite_index(unit, lambda m: m.__setitem__(key, m["query.fasta"]))
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_sweep(unit)
    assert e.value.rid == "R-U2" and (unit / key).is_file()


@pytest.mark.parametrize("field", ["pack", "index"])
def test_record_cannot_point_outside_the_unit(tmp_path: Path, field: str) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    pack.pack_unit(unit)
    record = json.loads((unit / pack.UNIT_RECORD).read_text())
    record["storage"][field] = "../elsewhere"
    _node.write_json_atomic(unit / pack.UNIT_RECORD, record)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.read_unit_member(unit, "query.fasta")
    assert e.value.rid == "R-U2"


def test_sweep_keeps_a_loose_file_that_differs_from_the_pack(tmp_path: Path) -> None:
    """It may be the only copy of those bytes. Nothing at all is deleted."""
    unit = tmp_path / "u"
    files = make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    (unit / "query.fasta").write_bytes(b">edited after the pack\n")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_sweep(unit)
    assert e.value.rid == "R-U2"
    assert (unit / "query.fasta").read_bytes() == b">edited after the pack\n"
    assert (unit / "fold/scores.json").read_bytes() == files["fold/scores.json"]


def test_sweep_does_not_delete_through_a_symlinked_parent(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    moved = tmp_path / "moved_fold"
    shutil.move(str(unit / "fold"), moved)
    (unit / "fold").symlink_to(moved)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_sweep(unit)
    assert e.value.rid == "R-U2"
    assert (moved / "scores.json").read_bytes() == files["fold/scores.json"]


def test_negative_index_values_are_refused(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    pack.pack_unit(unit)
    rewrite_index(unit, lambda m: m["query.fasta"].__setitem__("offset", -1))
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.read_unit_member(unit, "query.fasta")
    assert e.value.rid == "R-U2"


def test_unknown_pack_format_is_refused(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    pack.pack_unit(unit)
    record = json.loads((unit / pack.UNIT_RECORD).read_text())
    record["storage"]["format"] = "tar.gz+frame-index/9"
    _node.write_json_atomic(unit / pack.UNIT_RECORD, record)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.list_unit_members(unit)
    assert e.value.rid == "R-U2"


# ---- the record ----------------------------------------------------------------------------------


def test_record_digest_equals_zarr_commits_over_the_same_fields() -> None:
    """§3.1: "the same fields as CommitRecord", so one digest rule covers both."""
    core = {
        "input_digest": "in",
        "input_payload": {"é": 1, "b": [1, 2.5, None, True], "a": {"z": "ñ"}},
        "meta": {"k": "v"},
        "content_digest": "c",
        "run_id": "r",
        "env": {},
        "committed_at": "2026-11-04T00:00:00+00:00",
    }
    assert pack.compute_record_digest(core) == zarr_commit._compute_record_digest(core)
    # Extra fields, such as ``storage`` and ``schema``, are outside the digest.
    assert pack.compute_record_digest({**core, "storage": {"pack": "p"}, "schema": "s"}) == (
        zarr_commit._compute_record_digest(core)
    )


def test_record_normalization_matches_normalize_json_value() -> None:
    value = {"é": "é", "k": (1, b"by", {3: 1, 10: 2.5}), "n": None, "t": True}
    assert _node._normalize(value) == normalize_json_value(value)
    nfd_key = {"é": 1}  # a non-NFC key is kept as written, by both
    assert pack.compute_record_digest({"meta": nfd_key}) == zarr_commit._compute_record_digest(
        {k: ({"meta": nfd_key}.get(k)) for k in _node._CORE_FIELDS}
    )
    assert canonical_json_bytes(_node._normalize(value)) == canonical_json_bytes(
        normalize_json_value(value)
    )


def test_record_is_written_once(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    before = (unit / pack.UNIT_RECORD).read_bytes()
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.write_unit_record(unit, input_digest="other", run_id="r")
    assert e.value.rid == "R-U1"
    assert (unit / pack.UNIT_RECORD).read_bytes() == before


@pytest.mark.parametrize("bad", [{"x": object()}, {"x": float("nan")}])
def test_record_values_must_be_json(tmp_path: Path, bad: dict[str, Any]) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    with pytest.raises((pack.UnitPackRefused, ValueError)):
        pack.write_unit_record(unit, input_digest="i", run_id="r", meta=bad)
    assert not (unit / pack.UNIT_RECORD).exists()


@pytest.mark.parametrize(
    ("change", "needle"),
    [
        ({"input_digest": ""}, "input_digest"),
        ({"content_digest": 7}, "content_digest"),
        ({"meta": []}, "meta"),
        ({"record_digest": "x"}, "record_digest"),
    ],
)
def test_record_problem_names_the_defect(
    tmp_path: Path, change: dict[str, Any], needle: str
) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    record = json.loads((unit / pack.UNIT_RECORD).read_text())
    assert pack.record_problem(record) is None
    problem = pack.record_problem({**record, **change})
    assert problem is not None and needle in problem


def test_empty_run_id_is_allowed(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    assert pack.record_problem(pack.write_unit_record(unit, input_digest="i", run_id="")) is None


# ---- codecs --------------------------------------------------------------------------------------


def use_codec(monkeypatch: pytest.MonkeyPatch, kind: str, impl: Any) -> None:
    monkeypatch.setattr(_node, "_codec", lambda: (kind, impl))


def test_zstd_cli_path_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    binary = shutil.which("zstd")
    if binary is None:
        pytest.skip("no zstd CLI")
    use_codec(monkeypatch, "cli", binary)
    data = b"x" * 5000
    assert _node.zstd_decompress(_node.zstd_compress(data)) == data
    with pytest.raises(pack.UnitPackRefused):
        _node.zstd_decompress(b"not zstd", "m")


def test_zstandard_path_reads_concatenated_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    zstandard = pytest.importorskip("zstandard")
    use_codec(monkeypatch, "zstandard", zstandard)
    assert _node.zstd_decompress(_node.zstd_compress(b"a") + _node.zstd_compress(b"b")) == b"ab"


def test_codec_is_resolved_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(_node, "_CODEC_CACHE", [])
    monkeypatch.setattr(_node, "_find_codec", lambda: calls.append(1) or ("none", None))
    for _ in range(3):
        with pytest.raises(pack.CodecUnavailable):
            _node.zstd_compress(b"x")
    assert calls == [1]


def test_no_codec_is_environmental_and_keeps_the_loose_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not a refusal and not corruption: the unit is fine, the host has no zstd."""
    unit = tmp_path / "u"
    files = make_unit(unit)
    use_codec(monkeypatch, "none", None)
    with pytest.raises(pack.CodecUnavailable) as e:
        pack.pack_unit(unit)
    assert isinstance(e.value, OSError) and not isinstance(e.value, pack.UnitPackRefused)
    assert sorted(p.name for p in unit.iterdir()) == sorted(
        {*{r.split("/")[0] for r in files}, "unit.json"}
    )
    for rel, data in files.items():
        assert (unit / rel).read_bytes() == data


def test_vanished_cli_binary_is_environmental(monkeypatch: pytest.MonkeyPatch) -> None:
    use_codec(monkeypatch, "cli", "/nonexistent/zstd")
    with pytest.raises(pack.CodecUnavailable):
        _node.zstd_decompress(b"anything", "m")


def test_garbage_frame_is_a_refusal_not_a_crash() -> None:
    with pytest.raises(pack.UnitPackRefused) as e:
        _node.zstd_decompress(b"this is not a zstd frame", "m")
    assert e.value.rid == "R-U2"


# ---- the format shared with alynxr ---------------------------------------------------------------


def alynxr_copy(tmp_path: Path) -> Path:
    unit = tmp_path / "alynxr_unit"
    shutil.copytree(ALYNXR_FIXTURE, unit)
    return unit


def test_reads_a_pack_alynxr_wrote(tmp_path: Path) -> None:
    unit = alynxr_copy(tmp_path)
    name = "alynxr_unit.json"
    assert pack.is_packed(unit, record_name=name)
    assert pack.list_unit_members(unit, record_name=name) == [
        "fold/alynxr_run.json",
        "fold/d1_scores.json",
        "query.fasta",
    ]
    assert pack.read_unit_member(unit, "query.fasta", record_name=name) == b">d1\nACDE\n"
    assert pack.read_unit_member(unit, "fold/d1_scores.json", record_name=name) == (
        b'{"ptm": 0.85}\n'
    )
    assert pack.pack_sweep(unit, record_name=name)["storage"]["format"] == pack.PACK_FORMAT


def test_negative_control_alynxr_pack_needs_its_record_name(tmp_path: Path) -> None:
    unit = alynxr_copy(tmp_path)
    assert not pack.is_packed(unit)


def test_writes_what_alynxrs_reader_needs(tmp_path: Path) -> None:
    """The keys alynxr's ``_read_member_from_pack`` and ``_read_verified_index`` read, by name."""
    unit = tmp_path / "u"
    make_unit(unit)
    storage = pack.pack_unit(unit)["storage"]
    theirs = json.loads((ALYNXR_FIXTURE / "alynxr_unit.json").read_text())["storage"]
    # Every storage key matches alynxr's; only the writer's ``schema`` value differs.
    assert set(storage) == set(theirs)
    assert {k: v for k, v in storage.items() if k in ("pack", "index", "format", "codec")} == {
        k: v for k, v in theirs.items() if k in ("pack", "index", "format", "codec")
    }
    mine = json.loads((unit / pack.PACK_INDEX).read_text())["members"]["query.fasta"]
    their_entry = json.loads((ALYNXR_FIXTURE / "members.idx.json").read_text())["members"][
        "query.fasta"
    ]
    assert set(mine) == set(their_entry)


def test_consumer_stamp_and_check_replace_the_digest_check(tmp_path: Path) -> None:
    """How a consumer with its own stamp packs: its name, its completion rule."""
    unit = tmp_path / "u"
    files = make_unit(unit, commit=False)
    _node.write_json_atomic(unit / "own_stamp.json", {"complete": True})
    seen: list[str] = []

    def check(_unit: Any, record: dict[str, Any], digest: str) -> None:
        if record.get("complete") is not True:
            raise pack.UnitPackRefused("R-U1", "not complete")
        seen.append(digest)

    pack.pack_unit(unit, record_name="own_stamp.json", check=check)
    assert seen and pack.is_packed(unit, record_name="own_stamp.json")
    for rel, data in files.items():
        assert pack.read_unit_member(unit, rel, record_name="own_stamp.json") == data
    assert seen[0] == pack.digest_of_members(
        {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}
    )


def test_consumer_check_can_refuse_before_the_pack_exists(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    _node.write_json_atomic(unit / "own_stamp.json", {"complete": False})

    def check(_unit: Any, _record: dict[str, Any], _digest: str) -> None:
        raise pack.UnitPackRefused("R-U1", "not complete")

    with pytest.raises(pack.UnitPackRefused):
        pack.pack_unit(unit, record_name="own_stamp.json", check=check)
    assert not (unit / pack.PACK_FILE).exists()


# ---- CLI, as an embedded script runs it ----------------------------------------------------------


def test_cli_round_trip(tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    assert _node.main(["digest", str(unit)]) == 0
    digest = capfdbinary.readouterr().out.decode().strip()
    assert digest == pack.unit_content_digest(unit)
    assert _node.main(["pack", str(unit)]) == 0
    capfdbinary.readouterr()
    assert _node.main(["list", str(unit)]) == 0
    assert capfdbinary.readouterr().out.decode().split() == sorted(files)
    assert _node.main(["sweep", str(unit)]) == 0
    capfdbinary.readouterr()


def test_cli_read_writes_raw_bytes_in_a_subprocess(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    pack.pack_unit(unit)
    src = tmp_path / "embedded.py"
    src.write_text(pack.node_source(), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(src), "read", str(unit), "fold/deep/model.cif"],
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == files["fold/deep/model.cif"]


def test_cli_exit_codes(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    assert _node.main(["pack", str(unit)]) == 6
    assert _node.main(["bogus"]) == 2
    assert _node.main(["read", str(unit)]) == 2


# ---- review findings (PR #262) -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["meta/stamp.json", "../stamp.json", "", "members.tar.zst"])
def test_record_name_must_be_a_bare_file_name(tmp_path: Path, name: str) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.pack_unit(unit, record_name=name, check=lambda *_a: None)
    assert e.value.rid == "R-U3"
    assert not (unit / pack.PACK_FILE).exists()


def test_unreadable_record_is_an_error_not_a_loose_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EIO on the record of a packed unit must not read as 'loose, with no members'."""
    unit = tmp_path / "u"
    make_unit(unit)
    pack.pack_unit(unit)
    real = _node._read_bytes

    def eio(path: Any) -> bytes:
        if str(path).endswith(pack.UNIT_RECORD):
            raise OSError(5, "Input/output error")
        return real(path)

    monkeypatch.setattr(_node, "_read_bytes", eio)
    for call in (pack.unit_content_digest, pack.list_unit_members):
        with pytest.raises(OSError):
            call(unit)


def test_estale_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    import errno as errno_mod

    path = tmp_path / "f"
    path.write_bytes(b"ok")
    real_open = builtins.open
    fails = [2]

    def flaky(file: Any, *args: Any, **kwargs: Any) -> Any:
        if str(file) == str(path) and fails[0]:
            fails[0] -= 1
            raise OSError(errno_mod.ESTALE, "Stale file handle")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky)
    assert _node._read_bytes(path) == b"ok"


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"input_digest": ""}, "input_digest"),
        ({"run_id": 7}, "run_id"),
        ({"meta": []}, "meta"),
        ({"committed_at": 5}, "committed_at"),
    ],
)
def test_write_unit_record_refuses_an_invalid_record(
    tmp_path: Path, kwargs: dict[str, Any], needle: str
) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    args: dict[str, Any] = {"input_digest": "i", "run_id": "r", **kwargs}
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.write_unit_record(unit, **args)
    assert e.value.rid == "R-U3" and needle in e.value.detail
    assert not (unit / pack.UNIT_RECORD).exists()


@pytest.mark.parametrize("name", ["members.tar.zst", "members.idx.json", "members.tar.zst.partial"])
def test_write_unit_record_refuses_root_files_named_like_artefacts(
    tmp_path: Path, name: str
) -> None:
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    (unit / name).write_bytes(b"user data")
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.write_unit_record(unit, input_digest="i", run_id="r")
    assert e.value.rid == "R-U3"
    assert (unit / name).read_bytes() == b"user data"


def test_write_unit_record_never_replaces_a_record_that_appears_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check-then-write race: another writer's record lands after the existence check."""
    unit = tmp_path / "u"
    make_unit(unit, commit=False)
    real = _node._write_tmp

    def racing(path: str, data: bytes) -> str:
        tmp = real(path, data)
        if path.endswith(pack.UNIT_RECORD):
            (unit / pack.UNIT_RECORD).write_bytes(b'{"theirs": true}')
        return tmp

    monkeypatch.setattr(_node, "_write_tmp", racing)
    with pytest.raises(pack.UnitPackRefused) as e:
        pack.write_unit_record(unit, input_digest="i", run_id="r")
    assert e.value.rid == "R-U1"
    assert (unit / pack.UNIT_RECORD).read_bytes() == b'{"theirs": true}'
    assert not [p for p in unit.iterdir() if p.name.endswith(".tmp")]


@pytest.mark.parametrize("fault", ["tamper", "no_codec"])
def test_a_refusal_after_step_one_leaves_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    if fault == "tamper":
        (unit / "query.fasta").write_bytes(b">tampered\n")
        expected: Any = pack.UnitPackRefused
    else:
        use_codec(monkeypatch, "none", None)
        expected = pack.CodecUnavailable
    with pytest.raises(expected):
        pack.pack_unit(unit)
    names = {p.name for p in unit.iterdir()}
    assert not names & {
        "members.tar.zst.partial",
        pack.PACK_FILE,
        pack.PACK_INDEX,
        "members.tar.zst.lock",
    }


def test_loose_read_refuses_a_symlinked_leaf_and_transient_files(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    (tmp_path / "secret").write_bytes(b"outside the unit")
    (unit / "leaf").symlink_to(tmp_path / "secret")
    (unit / ".x.1.tmp").write_bytes(b"temp")
    (unit / ".nfs0001").write_bytes(b"nfs")
    for rel in ("leaf", ".x.1.tmp", ".nfs0001"):
        with pytest.raises(pack.UnitPackRefused) as e:
            pack.read_unit_member(unit, rel)
        assert e.value.rid == "R-U-unknown-member", rel


def test_a_reader_racing_the_sweep_falls_over_to_the_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader sees the unit loose, then the packer sweeps the file it was about to read."""
    unit = tmp_path / "u"
    files = make_unit(unit)
    digest = pack.unit_content_digest(unit)
    crash_at(unit, "stamp_rewritten")
    record = json.loads((unit / pack.UNIT_RECORD).read_text())
    loose_view = {k: v for k, v in record.items() if k != "storage"}
    reads = [0]
    real = _node._read_record

    def stale_then_fresh(unit_dir: Any, name: str) -> Any:
        reads[0] += 1
        if reads[0] == 1:
            pack.pack_sweep(unit)  # the packer finishes between the reader's two looks
            return loose_view
        return real(unit_dir, name)

    monkeypatch.setattr(_node, "_read_record", stale_then_fresh)
    assert pack.read_unit_member(unit, "query.fasta") == files["query.fasta"]
    reads[0] = 0
    crash_at_unit = tmp_path / "u2"
    make_unit(crash_at_unit)
    monkeypatch.setattr(_node, "_read_record", real)
    crash_at(crash_at_unit, "stamp_rewritten")
    record2 = json.loads((crash_at_unit / pack.UNIT_RECORD).read_text())
    loose2 = {k: v for k, v in record2.items() if k != "storage"}

    def stale_digest(unit_dir: Any, name: str) -> Any:
        reads[0] += 1
        if reads[0] == 1:
            pack.pack_sweep(crash_at_unit)
            return loose2
        return real(unit_dir, name)

    monkeypatch.setattr(_node, "_read_record", stale_digest)
    assert pack.unit_content_digest(crash_at_unit) == record2["content_digest"]
    assert digest == record["content_digest"]


_HOLD_LOCK = """
import sys, time
from xtrax.pack import _node
with _node._pack_lock(sys.argv[1]):
    print("held", flush=True)
    time.sleep(30)
"""


def test_a_second_packer_is_refused_while_the_first_holds_the_lock(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(unit)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(pack.UnitPackRefused) as e:
            pack.pack_unit(unit)
        assert e.value.rid == "R-U4"
        assert not (unit / "members.tar.zst.partial").exists()
    finally:
        holder.kill()
        holder.wait()
    # The holder died without cleaning up: its lock file is left, and the lock is free.
    assert (unit / "members.tar.zst.lock").exists()
    pack.pack_unit(unit)
    assert sorted(p.name for p in unit.iterdir()) == THREE_FILES
    for rel, data in files.items():
        assert pack.read_unit_member(unit, rel) == data


def test_lock_file_is_not_a_member_and_is_removed(tmp_path: Path) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    (unit / "members.tar.zst.lock").write_bytes(b"")  # a dead packer's leftover
    assert "members.tar.zst.lock" not in pack.collect_loose_members(unit)
    pack.pack_unit(unit)
    assert not (unit / "members.tar.zst.lock").exists()


def test_sweep_tolerates_a_directory_that_gains_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = tmp_path / "u"
    make_unit(unit)
    crash_at(unit, "stamp_rewritten")
    real_rmdir = os.rmdir

    def busy(path: Any) -> None:
        raise OSError(39, "Directory not empty", str(path))

    monkeypatch.setattr(_node.os, "rmdir", busy)
    pack.pack_sweep(unit)  # finishes; leftover empty dirs are harmless
    monkeypatch.setattr(_node.os, "rmdir", real_rmdir)
    assert pack.is_packed(unit) and not (unit / "query.fasta").exists()


def test_pack_unit_does_not_reverify_what_it_just_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = tmp_path / "u"
    files = make_unit(unit)
    calls = [0]
    real = _node.zstd_decompress

    def counting(data: bytes, name: str = "<member>") -> bytes:
        calls[0] += 1
        return real(data, name)

    monkeypatch.setattr(_node, "zstd_decompress", counting)
    pack.pack_unit(unit)
    assert calls[0] == len(files)  # step 2 only
