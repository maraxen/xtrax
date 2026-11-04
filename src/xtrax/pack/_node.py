"""Per-unit ``members.tar.zst`` + ``members.idx.json`` packer and random-access reader.

Spec 261030 §3.2, §3.3. SELF-CONTAINED BY DESIGN: this file imports only the standard library
and runs under Python 3.9, so :func:`xtrax.pack.node_source` can embed it in a rendered batch
script on a node with no xtrax installed. Embedders must call a registered interpreter >= 3.9,
never bare ``python3`` (3.6 on Engaging compute nodes, spec A8).

Lifted from alynxr ``unit_pack.py`` (cd8e5a9) on the owner's decision, so there is one format.
The bytes on disk are alynxr's: each member's USTAR header and padded data is its own zstd frame,
followed by a zero EOF frame, and the index gives each frame's ``offset`` and ``length``. A pack
written by either implementation reads through the other. What differs from alynxr:

* The completion check is the record's ``content_digest``, not a fold-specific run record;
  ``check=`` replaces it for a consumer with its own stamp, and ``record_name=`` names that stamp.
* Index keys are not trusted. A key that leaves the unit, or names the record or a pack artefact,
  is refused before anything is read or deleted.
* A symlink or special file in a unit is refused, not silently left out of the member set.
  Dotfiles are members, except the temp files of an atomic write and NFS ``.nfs*`` placeholders.
* The sweep deletes a loose file only if it is byte-identical to its packed copy, and it checks
  every file before deleting any.
* ``min_members`` is the per-unit-class threshold; below it a unit stays loose.

Commit order. (1) write ``members.tar.zst.partial``; (2) re-read every member from it, verify its
sha256 and run the completion check; (3) rename the pack into place and write the index; (4) rewrite
the record with its ``storage`` field; (5) delete the loose files. A crash before (4) leaves every
loose file and no ``storage``; after (4), :func:`pack_sweep` finishes the job.

A compressor is needed to pack, and a decompressor to read: PEP 784 ``compression.zstd``,
``zstandard``, or the ``zstd`` CLI (present on Engaging compute nodes). With none, packing refuses
and the loose files stay. Members are read whole into memory, so a unit's largest member must fit.

Refusals. ``R-U1``: the unit is not complete (record missing or unreadable, completion check
failed). ``R-U2``: a pack or member failed verification. ``R-U3``: the unit cannot be represented
(symlink, special file, a path or size USTAR cannot hold). ``R-U-unknown-member``: not a member.

Extract by hand with ``zstd -dc members.tar.zst | tar -x``; GNU tar before 1.31 has no ``--zstd``.
"""

import datetime
import hashlib
import importlib
import io
import json
import os
import socket
import stat
import subprocess
import sys
import tarfile
import unicodedata
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

PathLike = Union[str, os.PathLike]

UNIT_RECORD = "unit.json"
RECORD_SCHEMA = "xtrax.unit/1"
PACKER = "xtrax.pack/1"
STORAGE_SCHEMA = "xtrax.unit_pack/1"
INDEX_SCHEMA = "xtrax.unit_pack_index/1"
PACK_FORMAT = "tar.zst+frame-index/1"
PACK_FILE = "members.tar.zst"
PACK_INDEX = "members.idx.json"

_PARTIAL = PACK_FILE + ".partial"
# The record's immutable core: the fields ``zarr_commit._compute_record_digest`` covers (§3.1).
_CORE_FIELDS = (
    "input_digest",
    "input_payload",
    "meta",
    "content_digest",
    "run_id",
    "env",
    "committed_at",
)
_CORE_STR_FIELDS = ("input_digest", "content_digest", "run_id", "committed_at")
_CORE_DICT_FIELDS = ("input_payload", "meta", "env")


class UnitPackRefused(ValueError):
    """The unit cannot be packed or a packed unit failed verification; ``rid`` names the rule."""

    def __init__(self, rid: str, message: str) -> None:
        super().__init__("[{}] {}".format(rid, message))
        self.rid = rid
        self.detail = message


# ---- hashing and atomic writes -------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(path: PathLike) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pretty(obj: object) -> bytes:
    text = json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _write_bytes_atomic(path: PathLike, data: bytes) -> None:
    abspath = os.path.abspath(path)
    directory = os.path.dirname(abspath)
    tmp_name = ".{}.{}.{}.tmp".format(os.path.basename(abspath), socket.gethostname(), os.getpid())
    tmp = os.path.join(directory, tmp_name)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, abspath)
    _fsync_dir(directory)


def write_json_atomic(path: PathLike, obj: object) -> None:
    _write_bytes_atomic(path, _pretty(obj))


def _hook(step: str, crash_after: Optional[str]) -> None:
    if crash_after == step:
        raise RuntimeError("injected crash after {}".format(step))


# ---- zstd ----------------------------------------------------------------------------------------


def _optional_module(name: str) -> Any:  # noqa: ANN401
    """Import ``name`` if this interpreter has it, by name, so a static checker needs neither."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def zstd_compress(data: bytes, level: int = 3) -> bytes:
    """Compress ``data`` into a single RFC 8878 Zstandard frame."""
    stdlib = _optional_module("compression.zstd")  # Python 3.14+ (PEP 784)
    if stdlib is not None:
        return stdlib.compress(data, level=level)
    zstandard = _optional_module("zstandard")
    if zstandard is not None:
        return zstandard.ZstdCompressor(level=level).compress(data)
    try:
        r = subprocess.run(
            ["zstd", "-{}".format(int(level)), "-q", "-c"],
            input=data,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise UnitPackRefused("R-U2", "no zstd compressor available ({})".format(exc)) from None
    return r.stdout


def zstd_decompress(data: bytes, name: str = "<member>") -> bytes:
    """Decompress a Zstandard frame, or concatenated frames."""
    try:
        stdlib = _optional_module("compression.zstd")
        if stdlib is not None:
            return stdlib.decompress(data)
        zstandard = _optional_module("zstandard")
        if zstandard is not None:
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
                return reader.read()
        args = ["zstd", "-d", "-q", "-c"]
        return subprocess.run(args, input=data, capture_output=True, check=True).stdout
    except Exception as exc:  # noqa: BLE001 - every decode failure is the same refusal
        message = "{}: zstd member unreadable ({})".format(name, exc)
        raise UnitPackRefused("R-U2", message) from None


# ---- member set and content identity -------------------------------------------------------------


def is_plain_relpath(path: str) -> bool:
    """``path`` is a plain relative POSIX path: no ``..``, ``.``, empty part, NUL or leading ``/``.

    The check is lexical. Indexes and locators are not trusted, and those spellings reach outside
    the directory they are joined to.
    """
    if not path or "\0" in path or path.startswith("/"):
        return False
    return not any(p in ("", ".", "..") for p in path.split("/"))


def _excluded_names(record_name: str) -> "frozenset[str]":
    return frozenset({record_name, record_name + ".tmp", PACK_FILE, _PARTIAL, PACK_INDEX})


def _is_transient(fname: str) -> bool:
    """An atomic write's temp file, or NFS's placeholder for a deleted, still-open file."""
    return (fname.startswith(".") and fname.endswith(".tmp")) or fname.startswith(".nfs")


def collect_loose_members(unit_dir: PathLike, record_name: str = UNIT_RECORD) -> List[str]:
    """Sorted POSIX relative paths of the member set (§3.2).

    Refuses (``R-U3``) what a pack cannot hold, so a file never silently falls out of a unit's
    identity: a symlink, a special file, or a path that is not valid UTF-8.
    """
    root = os.path.abspath(unit_dir)
    excluded = _excluded_names(record_name)
    members: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                raise UnitPackRefused("R-U3", "{}: symlink in unit".format(name))
        for fname in filenames:
            if _is_transient(fname) or (dirpath == root and fname in excluded):
                continue
            full = os.path.join(dirpath, fname)
            mode = os.lstat(full).st_mode
            if not stat.S_ISREG(mode):
                raise UnitPackRefused("R-U3", "{}: not a regular file".format(full))
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError:
                raise UnitPackRefused("R-U3", "{!r}: path is not UTF-8".format(rel)) from None
            members.append(rel)
    return sorted(members)


def digest_of_members(shas: Dict[str, str]) -> str:
    """``content_digest``: sha256 of canonical JSON ``[[relpath, sha256], ...]``, by relpath."""
    pairs = [[rel, shas[rel]] for rel in sorted(shas)]
    text = json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def _loose_shas(unit_dir: PathLike, record_name: str) -> Dict[str, str]:
    return {
        rel: _sha256_file(os.path.join(unit_dir, *rel.split("/")))
        for rel in collect_loose_members(unit_dir, record_name)
    }


def unit_content_digest(unit_dir: PathLike, *, record_name: str = UNIT_RECORD) -> str:
    """``content_digest`` of a unit, loose or packed; the same value either way (§3.2).

    For a packed unit this reads and verifies every member through the index, so a damaged pack
    refuses (``R-U2``) rather than hashing what the index merely claims.
    """
    record = _read_record(unit_dir, record_name)
    if record is None or not is_packed(unit_dir, record):
        return digest_of_members(_loose_shas(unit_dir, record_name))
    storage = record["storage"]
    pack_path, members = _verified_members(unit_dir, storage, record_name)
    for relpath, entry in sorted(members.items()):
        _read_member_from_pack(pack_path, relpath, entry)
    return digest_of_members({rel: str(entry["sha256"]) for rel, entry in members.items()})


# ---- the unit record -----------------------------------------------------------------------------


def _normalize(value: Any) -> Any:  # noqa: ANN401
    """JSON-safe, canonically ordered, NFC strings: as ``zarr_integrity.normalize_json_value``."""
    if isinstance(value, dict):
        # Keys are stringified but not NFC-normalized, exactly as normalize_json_value does.
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise UnitPackRefused("R-U3", "record value of type {}".format(type(value).__name__))


def compute_record_digest(record: Dict[str, Any]) -> str:
    """Digest of the immutable core, so packing (``storage``) never changes it (§3.1)."""
    core = _normalize({k: record.get(k) for k in _CORE_FIELDS})
    text = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def record_problem(record: Dict[str, Any]) -> Optional[str]:
    """Why ``record`` is not a well-formed unit record whose digest verifies; None if it is."""
    for field in _CORE_STR_FIELDS + ("record_digest",):
        value = record.get(field)
        if not isinstance(value, str) or (not value and field != "run_id"):
            return "record lacks {}".format(field)
    for field in _CORE_DICT_FIELDS:
        if not isinstance(record.get(field), dict):
            return "record field {} is not an object".format(field)
    if compute_record_digest(record) != record["record_digest"]:
        return "record_digest does not match the record's core"
    return None


def write_unit_record(
    unit_dir: PathLike,
    *,
    input_digest: str,
    run_id: str,
    input_payload: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    env: Optional[Dict[str, Any]] = None,
    committed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit a loose unit: hash its members and write ``unit.json`` last, by temp + replace.

    Refuses (``R-U1``) if the unit already has a record. A record is immutable, and overwriting
    one that names a pack would orphan the only copy of the members.
    """
    path = os.path.join(unit_dir, UNIT_RECORD)
    if os.path.lexists(path):
        raise UnitPackRefused("R-U1", "{}: unit already has a record".format(path))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record: Dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "input_digest": input_digest,
        "input_payload": input_payload or {},
        "meta": meta or {},
        "content_digest": digest_of_members(_loose_shas(unit_dir, UNIT_RECORD)),
        "run_id": run_id,
        "env": env or {},
        "committed_at": committed_at or now,
    }
    record["record_digest"] = compute_record_digest(record)
    try:
        payload = _pretty(record)
    except (TypeError, ValueError) as exc:
        raise UnitPackRefused("R-U3", "record is not JSON ({})".format(exc)) from None
    _write_bytes_atomic(path, payload)
    return record


def _read_record(unit_dir: PathLike, record_name: str) -> Optional[Dict[str, Any]]:
    """The record, or None when it is absent, unreadable or not a JSON object."""
    try:
        with open(os.path.join(unit_dir, record_name), encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _load_record(unit_dir: PathLike, record_name: str) -> Dict[str, Any]:
    path = os.path.join(unit_dir, record_name)
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except FileNotFoundError:
        raise UnitPackRefused("R-U1", "{}: unit has no record".format(path)) from None
    except (OSError, ValueError) as exc:
        raise UnitPackRefused("R-U1", "{}: unreadable record ({})".format(path, exc)) from None
    if not isinstance(record, dict):
        raise UnitPackRefused("R-U1", "{}: record is not a JSON object".format(path))
    return record


def is_packed(
    unit_dir: PathLike, record: Optional[Dict[str, Any]] = None, *, record_name: str = UNIT_RECORD
) -> bool:
    """True iff the unit's record names a committed ``storage.pack``."""
    if record is None:
        record = _read_record(unit_dir, record_name)
    storage = record.get("storage") if isinstance(record, dict) else None
    return isinstance(storage, dict) and bool(storage.get("pack"))


# ---- reading a pack ------------------------------------------------------------------------------


def _storage_name(storage: Dict[str, Any], key: str, default: str) -> str:
    name = storage.get(key, default)
    if not isinstance(name, str) or not is_plain_relpath(name) or "/" in name:
        raise UnitPackRefused("R-U2", "storage.{} {!r} is not a file in the unit".format(key, name))
    return name


def _verified_members(
    unit_dir: PathLike, storage: Dict[str, Any], record_name: str
) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """The pack path and index members, with the index checked against the record's hash.

    Every key must be a member path inside the unit. An index that names ``../x``, or the record
    itself, would otherwise be read from and swept.
    """
    if storage.get("format") != PACK_FORMAT:
        raise UnitPackRefused("R-U2", "unknown pack format {!r}".format(storage.get("format")))
    idx_path = os.path.join(unit_dir, _storage_name(storage, "index", PACK_INDEX))
    pack_path = os.path.join(unit_dir, _storage_name(storage, "pack", PACK_FILE))
    try:
        with open(idx_path, "rb") as fh:
            idx_bytes = fh.read()
    except OSError as exc:
        raise UnitPackRefused("R-U2", "pack index {} missing ({})".format(idx_path, exc)) from None
    if sha256_bytes(idx_bytes) != storage.get("index_sha256"):
        raise UnitPackRefused("R-U2", "{}: does not match index_sha256".format(idx_path))
    try:
        doc = json.loads(idx_bytes.decode("utf-8"))
    except ValueError as exc:
        raise UnitPackRefused("R-U2", "{}: invalid JSON ({})".format(idx_path, exc)) from None
    members = doc.get("members") if isinstance(doc, dict) else None
    if not isinstance(members, dict):
        raise UnitPackRefused("R-U2", "{}: missing 'members' mapping".format(idx_path))
    excluded = _excluded_names(record_name)
    for rel, entry in members.items():
        if not is_plain_relpath(rel) or rel in excluded or not isinstance(entry, dict):
            raise UnitPackRefused("R-U2", "{}: bad member {!r}".format(idx_path, rel))
    return pack_path, members


def _read_member_from_pack(pack_path: PathLike, relpath: str, entry: Dict[str, Any]) -> bytes:
    try:
        offset = int(entry["offset"])
        length = int(entry["length"])
        hlen = int(entry.get("header_length", 512))
        size = int(entry["size"])
        want_sha = str(entry["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        message = "{}: malformed index entry ({})".format(relpath, exc)
        raise UnitPackRefused("R-U2", message) from None
    if offset < 0 or length < 0 or hlen < 0 or size < 0:
        raise UnitPackRefused("R-U2", "{}: negative value in index entry".format(relpath))
    try:
        with open(pack_path, "rb") as fh:
            fh.seek(offset)
            frame = fh.read(length)
    except OSError as exc:
        message = "pack file {} missing or unreadable ({})".format(pack_path, exc)
        raise UnitPackRefused("R-U2", message) from None
    if len(frame) != length:
        raise UnitPackRefused("R-U2", "{}: pack truncated at offset {}".format(relpath, offset))
    block = zstd_decompress(frame, relpath)
    data = block[hlen : hlen + size]
    if len(data) != size or sha256_bytes(data) != want_sha:
        raise UnitPackRefused("R-U2", "{}: sha256 mismatch inside pack".format(relpath))
    return data


def list_unit_members(unit_dir: PathLike, *, record_name: str = UNIT_RECORD) -> List[str]:
    """Sorted POSIX relative paths of a unit's members, loose or packed."""
    record = _read_record(unit_dir, record_name)
    if record is None or not is_packed(unit_dir, record):
        return collect_loose_members(unit_dir, record_name)
    return sorted(_verified_members(unit_dir, record["storage"], record_name)[1])


def _no_symlinked_parent(unit_dir: PathLike, relpath: str) -> bool:
    current = str(unit_dir)
    for part in relpath.split("/")[:-1]:
        current = os.path.join(current, part)
        if os.path.islink(current):
            return False
    return True


def read_unit_member(unit_dir: PathLike, relpath: str, *, record_name: str = UNIT_RECORD) -> bytes:
    """Read a member, loose or packed.

    A packed read seeks to the member's own frame, decompresses only it, and verifies its sha256
    against the index, which is itself verified against the record.
    """
    unknown = "{} is not a member of unit {}".format(relpath, unit_dir)
    if not is_plain_relpath(relpath) or relpath in _excluded_names(record_name):
        raise UnitPackRefused("R-U-unknown-member", unknown)
    record = _read_record(unit_dir, record_name)
    if record is None or not is_packed(unit_dir, record):
        if not _no_symlinked_parent(unit_dir, relpath):
            raise UnitPackRefused("R-U-unknown-member", unknown)
        try:
            with open(os.path.join(unit_dir, *relpath.split("/")), "rb") as fh:
                return fh.read()
        except OSError as exc:
            message = "{}: loose member unreadable ({})".format(relpath, exc)
            raise UnitPackRefused("R-U2", message) from None
    pack_path, members = _verified_members(unit_dir, record["storage"], record_name)
    entry = members.get(relpath)
    if entry is None:
        raise UnitPackRefused("R-U-unknown-member", unknown)
    return _read_member_from_pack(pack_path, relpath, entry)


def find_packed_unit(
    path: PathLike, *, record_name: str = UNIT_RECORD, max_depth: int = 4
) -> Optional[Tuple[str, str]]:
    """If ``path`` lies inside a packed unit directory, return ``(unit_dir, relpath)``."""
    abspath = os.path.abspath(path)
    cur = os.path.dirname(abspath)
    for _ in range(max_depth):
        if not cur or cur == os.path.dirname(cur):
            break
        if is_packed(cur, record_name=record_name):
            return cur, os.path.relpath(abspath, cur).replace(os.sep, "/")
        cur = os.path.dirname(cur)
    return None


# ---- packing -------------------------------------------------------------------------------------


def pack_sweep(
    unit_dir: PathLike, *, record_name: str = UNIT_RECORD, _crash_after: Optional[str] = None
) -> Dict[str, Any]:
    """Finish step 5: verify the pack the record names, then delete the loose copies.

    Nothing is deleted until the whole pack verifies and every loose copy is byte-identical to its
    packed member. A loose file that differs is refused (``R-U2``), because it may be the only copy
    of those bytes. A record with no pack is returned untouched.
    """
    record = _load_record(unit_dir, record_name)
    if not is_packed(unit_dir, record):
        return record
    storage = record["storage"]
    pack_path, members = _verified_members(unit_dir, storage, record_name)
    if not os.path.isfile(pack_path):
        raise UnitPackRefused("R-U2", "pack file {} is missing".format(pack_path))
    if _sha256_file(pack_path) != storage.get("pack_sha256"):
        raise UnitPackRefused("R-U2", "pack file {} does not match pack_sha256".format(pack_path))
    for relpath, entry in sorted(members.items()):
        _read_member_from_pack(pack_path, relpath, entry)

    to_delete: List[str] = []
    for relpath, entry in sorted(members.items()):
        loose = os.path.join(unit_dir, *relpath.split("/"))
        if not _no_symlinked_parent(unit_dir, relpath):
            raise UnitPackRefused("R-U2", "{}: reached through a symlink".format(relpath))
        if not os.path.lexists(loose):
            continue
        if os.path.islink(loose) or not os.path.isfile(loose):
            raise UnitPackRefused("R-U2", "{}: loose copy is not a regular file".format(relpath))
        if _sha256_file(loose) != entry["sha256"]:
            raise UnitPackRefused("R-U2", "{}: loose copy differs from the pack".format(relpath))
        to_delete.append(loose)
    for loose in to_delete:
        os.remove(loose)
        _hook("loose_member_deleted", _crash_after)

    root = os.path.abspath(unit_dir)
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath != root and not os.listdir(dirpath):
            os.rmdir(dirpath)
    _fsync_dir(root)
    _hook("loose_deleted", _crash_after)
    return record


def _tar_member_block(relpath: str, payload: bytes) -> Tuple[bytes, int]:
    """Deterministic USTAR header + 512-byte padded payload block: ``(block, header_len)``."""
    ti = tarfile.TarInfo(name=relpath)
    ti.size = len(payload)
    ti.mtime = 0
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""
    ti.mode = 0o644
    ti.type = tarfile.REGTYPE
    try:
        hdr = ti.tobuf(tarfile.USTAR_FORMAT)
    except ValueError as exc:
        raise UnitPackRefused("R-U3", "{}: not representable ({})".format(relpath, exc)) from None
    pad = (512 - (len(payload) % 512)) % 512
    return hdr + payload + (b"\x00" * pad), len(hdr)


def _check_content_digest(unit_dir: PathLike, record: Dict[str, Any], digest: str) -> None:
    """The default completion check: the unit holds exactly the bytes its record committed."""
    problem = record_problem(record)
    if problem is not None:
        raise UnitPackRefused("R-U1", "{}: {}".format(unit_dir, problem))
    if digest != record["content_digest"]:
        raise UnitPackRefused("R-U1", "{}: members do not match content_digest".format(unit_dir))


def pack_unit(
    unit_dir: PathLike,
    *,
    level: int = 3,
    min_members: int = 1,
    record_name: str = UNIT_RECORD,
    check: Optional[Callable[[PathLike, Dict[str, Any], str], None]] = None,
    _crash_after: Optional[str] = None,
) -> Dict[str, Any]:
    """Pack a completed unit into ``members.tar.zst`` + ``members.idx.json``; returns the record.

    Idempotent: a record that already names a pack goes to :func:`pack_sweep`. A unit with fewer
    than ``min_members`` members stays loose. The threshold is per unit class and counts files
    in the member set, not queries.

    Args:
        unit_dir: The unit directory.
        level: The zstd compression level.
        min_members: Below this member count, the unit is left loose.
        record_name: The record file, for a consumer with its own stamp.
        check: ``check(unit_dir, record, digest)`` raises :class:`UnitPackRefused` if the unit is
            not complete; ``digest`` is the verified :func:`digest_of_members` of what was packed.
            The default compares it to the record's ``content_digest``.
    """
    record = _load_record(unit_dir, record_name)
    if is_packed(unit_dir, record):
        return pack_sweep(unit_dir, record_name=record_name, _crash_after=_crash_after)
    relpaths = collect_loose_members(unit_dir, record_name)
    if len(relpaths) < max(min_members, 1):
        return record

    partial_path = os.path.join(unit_dir, _PARTIAL)
    pack_path = os.path.join(unit_dir, PACK_FILE)
    idx_path = os.path.join(unit_dir, PACK_INDEX)

    # Step 1: one zstd frame per USTAR member block, then the end-of-archive marker in its own.
    members: Dict[str, Dict[str, Any]] = {}
    offset = 0
    with open(partial_path, "wb") as out_fh:
        for relpath in relpaths:
            with open(os.path.join(unit_dir, *relpath.split("/")), "rb") as in_fh:
                raw = in_fh.read()
            block, hlen = _tar_member_block(relpath, raw)
            frame = zstd_compress(block, level=level)
            out_fh.write(frame)
            members[relpath] = {
                "offset": offset,
                "length": len(frame),
                "header_length": hlen,
                "size": len(raw),
                "sha256": sha256_bytes(raw),
            }
            offset += len(frame)
        out_fh.write(zstd_compress(b"\x00" * 1024, level=level))
        out_fh.flush()
        os.fsync(out_fh.fileno())
    _hook("tar_written", _crash_after)

    # Step 2: re-read every member from the partial on disk, and check it against the record.
    for relpath, entry in sorted(members.items()):
        _read_member_from_pack(partial_path, relpath, entry)
    digest = digest_of_members({rel: entry["sha256"] for rel, entry in members.items()})
    (check or _check_content_digest)(unit_dir, record, digest)
    _hook("tar_verified", _crash_after)

    # Step 3: rename the pack into place and write the index.
    os.replace(partial_path, pack_path)
    _fsync_dir(unit_dir)
    _hook("tar_renamed", _crash_after)
    index = {
        "schema": INDEX_SCHEMA,
        "format": PACK_FORMAT,
        "codec": "zstd",
        "packer": PACKER,
        "members": members,
    }
    _write_bytes_atomic(idx_path, _pretty(index))
    _hook("index_written", _crash_after)

    # Step 4: name the pack in the record, outside its digest.
    new_record = dict(record)
    new_record["storage"] = {
        "schema": STORAGE_SCHEMA,
        "pack": PACK_FILE,
        "index": PACK_INDEX,
        "format": PACK_FORMAT,
        "codec": "zstd",
        "pack_sha256": _sha256_file(pack_path),
        "index_sha256": _sha256_file(idx_path),
        "n_members": len(members),
    }
    write_json_atomic(os.path.join(unit_dir, record_name), new_record)
    _hook("stamp_rewritten", _crash_after)

    # Step 5: delete the loose files.
    return pack_sweep(unit_dir, record_name=record_name, _crash_after=_crash_after)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] not in ("pack", "sweep", "read", "list", "digest"):
        sys.stderr.write(
            "usage: _node.py <pack|sweep|list|digest> <unit_dir> | read <unit_dir> <relpath>\n"
        )
        return 2
    cmd, unit_dir = args[0], args[1]
    try:
        if cmd == "pack":
            record = pack_unit(unit_dir)
            sys.stdout.write(json.dumps(record.get("storage"), sort_keys=True) + "\n")
        elif cmd == "sweep":
            record = pack_sweep(unit_dir)
            sys.stdout.write(json.dumps(record.get("storage"), sort_keys=True) + "\n")
        elif cmd == "list":
            for member in list_unit_members(unit_dir):
                sys.stdout.write(member + "\n")
        elif cmd == "digest":
            sys.stdout.write(unit_content_digest(unit_dir) + "\n")
        else:
            if len(args) != 3:
                sys.stderr.write("usage: _node.py read <unit_dir> <relpath>\n")
                return 2
            sys.stdout.buffer.write(read_unit_member(unit_dir, args[2]))
    except UnitPackRefused as exc:
        sys.stderr.write("pack refused {}\n".format(exc))
        return 6
    except OSError as exc:
        sys.stderr.write("pack input error: {}\n".format(exc))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
