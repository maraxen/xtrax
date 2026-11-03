"""Unit index: lock-free per-writer segments over per-unit commit records (spec 261030 §3.1).

The index is derived (decision D1). A unit is complete when its artifact carries a verifying
commit record; an index row only says where to look. Internal for now: nothing is exported from
``xtrax.run``.

Layout under the index root::

    segments/<writer_id>.jsonl   one append-only file per writer; fsync after every row
    sealed/<ns>-<uuid8>.jsonl    compaction output, written by temp + os.replace
    sealed/<ns>-<uuid8>.inputs.json   that file's inputs and their sizes, until archived
    rebuilt/index-<ts>.jsonl     rebuild output (not written here)
    archive/                     compacted inputs, moved here and never deleted

* **Writer.** :class:`SegmentWriter` opens its segment with ``O_EXCL``, fsyncs the directory
  once, fsyncs after each row, and appends a ``{"closed": true}`` trailer on close (also when
  the ``with`` body raises). An optional file guard is checked before the segment exists.
* **Read.** :func:`iter_rows` covers ``segments/``, ``sealed/`` and ``rebuilt/`` and dedups on
  ``(store, locator, record_digest)``. An unparseable line is *torn*: skipped and reported. A
  parseable row this code cannot accept (unknown ``schema_version``, missing field) is
  *unreadable*: skipped and reported. Every read and listing retries on ``ESTALE``; a file that
  vanishes mid-scan (a compaction, or a stale NFS listing) restarts the scan after a backoff
  longer than the NFS directory-cache window.
* **Compact.** :func:`compact` takes closed segments, and segments whose writer is dead by
  :func:`xtrax.run.writer_liveness.segment_writer_dead` (UNKNOWN is never dead). It writes one
  new sealed file holding every row it read from the taken segments and the existing sealed
  files, plus a manifest of those inputs. A later pass moves exactly those inputs to
  ``archive/`` once the sealed file is ``archive_after_s`` old (decision D6), so a reader on a
  host with a stale ``sealed/`` listing never misses a row. Rows collapse only when they agree on
  ``(unit_key, input_digest, content_digest, store, locator)``: copies at other locations stay,
  and one ``(unit_key, input_digest)`` with two content digests (R-S6) keeps both rows and is
  listed, never resolved. A new ``input_digest`` for a ``unit_key`` is a recompute, a new key.

There is no compaction lock. Two compactors at once are safe because every sealed and temp name
is unique and each compactor archives only files it read into its own output: at worst rows are
duplicated, and the read side removes duplicates. A crash at any step leaves every row readable.
"""

import dataclasses
import errno
import json
import os
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xtrax.config import resolve_layered
from xtrax.run import writer_liveness as _wl
from xtrax.run.writer_liveness import (
    DEFAULT_GRACE_S,
    NOJOB,
    Runner,
    parse_writer_id,
    segment_writer_dead,
    server_now,
)
from xtrax.run.zarr_commit import FAULT_ENV

SCHEMA_VERSION = 1
ROOT_ENV = "XTRAX_UNIT_INDEX_ROOT"
ROOT_KEY = "unit_index_root"
SEGMENTS_DIRNAME = "segments"
SEALED_DIRNAME = "sealed"
REBUILT_DIRNAME = "rebuilt"
ARCHIVE_DIRNAME = "archive"
TRAILER = b'{"closed": true}\n'
READ_DIRS = (SEGMENTS_DIRNAME, SEALED_DIRNAME, REBUILT_DIRNAME)
ESTALE_TRIES = 6
# Sleeps between read rescans. They sum past the NFS directory attribute cache ceiling (kernel
# default acdirmax = 60 s; Engaging mounts set no actimeo/acdir* option, read 2026-11-03), so a
# reader whose cached listing still names an archived segment waits until the listing refreshes.
RESCAN_SLEEPS_S = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
MAX_RESCANS = len(RESCAN_SLEEPS_S) + 1
# Compaction archives a sealed file's inputs only once that file is this old (NFS server time):
# 3x acdirmax, so every host's sealed/ listing shows it before its inputs leave segments/.
DEFAULT_ARCHIVE_AFTER_S = 180.0
PARTIAL_MAX_AGE_S = 3600.0
MANIFEST_SUFFIX = ".inputs.json"
_ARCHIVE_PREFIX = {SEGMENTS_DIRNAME: "segment-", SEALED_DIRNAME: "sealed-"}
# archive/segment-<id>.jsonl plus a ".<uuid8>" collision suffix must fit NAME_MAX (255 bytes).
WRITER_ID_MAX_BYTES = 255 - len("segment-") - len(".jsonl") - 9

_REQUIRED_STR = ("store", "locator", "unit_key", "input_digest", "content_digest", "record_digest")
_STR = ("adapter", "writer_id", "run_id")
ROW_FIELDS = (
    "schema_version",
    *_REQUIRED_STR,
    *_STR,
    "ts",
    "reused_from",
)


class UnitIndexError(Exception):
    """The index cannot be read or compacted safely; nothing was changed."""


class RowError(ValueError):
    """A parseable JSON line is not a valid index row."""


class SchemaMismatchError(RowError):
    """A row's ``schema_version`` is not the one this code writes."""


@runtime_checkable
class FileGuard(Protocol):
    """What :class:`SegmentWriter` needs from :class:`xtrax.fileguard.FileCapGuard`."""

    def check(self, n_new_files: int) -> int: ...


# ---- rows ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class IndexRow:
    """One index row: where a unit's committed artifact is, and which commit it carries."""

    store: str
    locator: str
    unit_key: str
    input_digest: str
    content_digest: str
    record_digest: str
    adapter: str
    writer_id: str
    run_id: str
    ts: float
    reused_from: tuple[str, str, str] | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def read_key(self) -> tuple[str, str, str]:
        """The read-side dedup key: ``(store, locator, record_digest)``."""
        return (self.store, self.locator, self.record_digest)

    @property
    def unit_id(self) -> tuple[str, str]:
        """The compaction key: ``(unit_key, input_digest)``."""
        return (self.unit_key, self.input_digest)

    def to_dict(self) -> dict[str, Any]:
        d = {f: getattr(self, f) for f in ROW_FIELDS}
        d["reused_from"] = list(self.reused_from) if self.reused_from is not None else None
        return d

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IndexRow":
        version = d.get("schema_version")
        if version != SCHEMA_VERSION:
            raise SchemaMismatchError(f"schema_version {version!r}, expected {SCHEMA_VERSION}")
        for f in _REQUIRED_STR:
            v = d.get(f)
            if not isinstance(v, str) or not v:
                raise RowError(f"field {f!r} must be a non-empty string, got {v!r}")
        for f in _STR:
            if not isinstance(d.get(f), str):
                raise RowError(f"field {f!r} must be a string, got {d.get(f)!r}")
        ts = d.get("ts")
        if isinstance(ts, bool) or not isinstance(ts, int | float):
            raise RowError(f"field 'ts' must be a number, got {ts!r}")
        reused = d.get("reused_from")
        if reused is not None:
            if not (
                isinstance(reused, list)
                and len(reused) == 3
                and all(isinstance(x, str) and x for x in reused)
            ):
                raise RowError(f"field 'reused_from' must be null or 3 strings, got {reused!r}")
            reused = (reused[0], reused[1], reused[2])
        return cls(
            store=d["store"],
            locator=d["locator"],
            unit_key=d["unit_key"],
            input_digest=d["input_digest"],
            content_digest=d["content_digest"],
            record_digest=d["record_digest"],
            adapter=d["adapter"],
            writer_id=d["writer_id"],
            run_id=d["run_id"],
            ts=float(ts),
            reused_from=reused,
        )


# ---- root ----------------------------------------------------------------------------------


def _disabled(value: str) -> bool:
    return value.strip().lower() in {"", "none"}


def root_source(
    root: Path | str | None = None, *, start: Path | str | None = None
) -> tuple[Path | None, str]:
    """``(root or None, which layer decided)``. No default path: unset means the index is off.

    Order: ``root`` argument > ``$XTRAX_UNIT_INDEX_ROOT`` > ``unit_index_root`` in
    ``[tool.xtrax]`` of the nearest ``pyproject.toml`` > ``unit_index_root`` in
    ``${XDG_CONFIG_HOME:-~/.config}/xtrax/config.toml`` > off. ``none`` or empty in the argument
    or the variable switches the index off. A config-file value must be an absolute path (``~``
    and ``$VARS`` expand); a malformed config file raises ``ValueError`` naming it.
    """
    if root is not None:
        if _disabled(str(root)):
            return None, "arg (disabled)"
        return Path(str(root).strip()).expanduser().absolute(), "arg"
    if ROOT_ENV in os.environ:
        value = os.environ[ROOT_ENV]
        if _disabled(value):
            return None, "env (disabled)"
        return Path(value.strip()).expanduser().absolute(), "env"
    value, source = resolve_layered("xtrax", ROOT_KEY, start=start)
    if value is None:
        return None, "off"
    if not isinstance(value, str):
        raise ValueError(f"{ROOT_KEY} from {source} must be a string path, got {value!r}")
    if _disabled(value):
        return None, f"{source} (disabled)"
    path = Path(os.path.expandvars(value.strip())).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{ROOT_KEY} from {source} is relative ({value!r}); use an absolute path")
    return path, source


def resolve_root(root: Path | str | None = None, *, start: Path | str | None = None) -> Path | None:
    """The index root, or None when the index is off. See :func:`root_source`."""
    return root_source(root, start=start)[0]


# ---- low-level IO --------------------------------------------------------------------------


def _fault(step: str) -> None:
    """Crash here when ``XTRAX_FAULT_INJECT=unit_index:<step>`` (crash-injection tests)."""
    if os.environ.get(FAULT_ENV) == f"unit_index:{step}":
        os._exit(137)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_dir(path: Path) -> None:
    """mkdir -p, fsyncing the parent of every directory this call creates."""
    missing = []
    p = path
    while not p.exists():
        missing.append(p)
        p = p.parent
    for d in reversed(missing):
        try:
            d.mkdir()
        except FileExistsError:
            pass  # another process won the race; its parent fsync may not have happened yet
        _fsync_dir(d.parent)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _read_bytes(path: Path) -> bytes:
    """Whole-file read; on ESTALE reopen and reread (never reuse the descriptor)."""
    for attempt in range(ESTALE_TRIES):
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError as e:
            if e.errno != errno.ESTALE or attempt == ESTALE_TRIES - 1:
                raise
            time.sleep(min(0.05 * 2**attempt, 1.0))
    raise AssertionError("unreachable")  # pragma: no cover


# ---- writer --------------------------------------------------------------------------------


def new_writer_id(*, host: str | None = None, pid: int | None = None) -> str:
    """``<host>-<pid>-<SLURM_JOB_ID|nojob>-<SLURM_RESTART_COUNT|0>-<uuid8>``."""
    h = (host if host is not None else socket.gethostname()).split(".")[0]
    h = re.sub(r"[^A-Za-z0-9-]", "_", h) or "host"
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    restart = os.environ.get("SLURM_RESTART_COUNT", "").strip()
    p = os.getpid() if pid is None else pid
    j = job if job.isdigit() else NOJOB
    r = restart if restart.isdigit() else "0"
    return f"{h}-{p}-{j}-{r}-{uuid.uuid4().hex[:8]}"


class SegmentWriter:
    """This writer's append-only segment. Use as a context manager.

    ``with SegmentWriter(root) as w: w.append(row)``. Commit the unit's record first, then
    append its row: a crash in between leaves a record with no row, which rebuild restores.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        writer_id: str | None = None,
        guard: FileGuard | None = None,
    ) -> None:
        self.root = Path(root)
        self.writer_id = writer_id if writer_id is not None else new_writer_id()
        parse_writer_id(self.writer_id)  # ValueError on a malformed id
        # The id is a file name in segments/: a "/" would nest the file outside the read glob.
        if (
            Path(self.writer_id).name != self.writer_id
            or self.writer_id.startswith(".")
            or "\0" in self.writer_id
        ):
            raise ValueError(f"writer_id {self.writer_id!r} is not a plain file name")
        if len(self.writer_id.encode("utf-8")) > WRITER_ID_MAX_BYTES:
            raise ValueError(
                f"writer_id is {len(self.writer_id.encode('utf-8'))} bytes; the archived segment "
                f"name needs it to be at most {WRITER_ID_MAX_BYTES}"
            )
        self.path = self.root / SEGMENTS_DIRNAME / f"{self.writer_id}.jsonl"
        self.rows_written = 0
        self._guard = guard
        self._fd: int | None = None
        self._closed = False
        self._clean = True  # False while a write is in flight or after one failed part-way

    @property
    def enabled(self) -> bool:
        return True

    def open(self) -> "SegmentWriter":
        if self._fd is not None or self._closed:
            raise RuntimeError(f"segment {self.path} was already opened")
        if self._guard is not None:
            self._guard.check(1)  # refuses before the segment exists
        seg_dir = self.path.parent
        _ensure_dir(seg_dir)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_CLOEXEC
        self._fd = os.open(self.path, flags, 0o644)
        _fsync_dir(seg_dir)
        return self

    def _write(self, data: bytes) -> None:
        if self._fd is None:
            raise RuntimeError(f"segment {self.path} is not open")
        if not self._clean:
            data = b"\n" + data  # end the fragment a failed write left, so this line is whole
        self._clean = False
        _write_all(self._fd, data)
        os.fsync(self._fd)
        self._clean = True

    def append(self, row: IndexRow) -> None:
        """Append one row and fsync it."""
        if self._closed:
            raise RuntimeError(f"segment {self.path} is closed")
        if row.writer_id != self.writer_id:
            raise ValueError(f"row writer_id {row.writer_id!r} is not {self.writer_id!r}")
        self._write(row.to_json_line().encode("utf-8"))
        self.rows_written += 1

    def close(self) -> None:
        """Append the closed trailer and close. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._fd is None:
            return
        try:
            self._write(TRAILER)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "SegmentWriter":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()


class DisabledWriter:
    """Stand-in when the index is off: accepts rows and writes nothing."""

    writer_id = ""
    rows_written = 0
    path = None

    @property
    def enabled(self) -> bool:
        return False

    def open(self) -> "DisabledWriter":
        return self

    def append(self, row: IndexRow) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "DisabledWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def open_writer(
    root: Path | str | None = None,
    *,
    start: Path | str | None = None,
    guard: FileGuard | None = None,
    writer_id: str | None = None,
) -> SegmentWriter | DisabledWriter:
    """A writer for the resolved root (:func:`root_source`), or a no-op one when it is off."""
    resolved = resolve_root(root, start=start)
    if resolved is None:
        return DisabledWriter()
    return SegmentWriter(resolved, writer_id=writer_id, guard=guard)


# ---- read ----------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BadLine:
    """A line skipped on read: ``kind`` is ``torn`` or ``unreadable``."""

    path: str
    line: int
    kind: str
    reason: str


@dataclasses.dataclass(slots=True)
class _Parsed:
    rows: list[IndexRow]
    torn: list[BadLine]
    unreadable: list[BadLine]
    trailer: bool


def _parse(path: Path, data: bytes) -> _Parsed:
    out = _Parsed([], [], [], False)
    for i, raw in enumerate(data.split(b"\n"), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as e:
            out.torn.append(BadLine(str(path), i, "torn", f"not JSON: {type(e).__name__}"))
            continue
        if isinstance(obj, dict) and obj.get("closed") is True:
            out.trailer = True
            continue
        if not isinstance(obj, dict):
            out.unreadable.append(BadLine(str(path), i, "unreadable", "not a JSON object"))
            continue
        try:
            out.rows.append(IndexRow.from_dict(obj))
        except RowError as e:
            out.unreadable.append(BadLine(str(path), i, "unreadable", str(e)))
    return out


def _names(d: Path, suffix: str) -> list[Path]:
    """Sorted non-hidden entries of ``d`` ending in ``suffix``; ``[]`` if ``d`` is missing.

    ``os.listdir`` rather than ``Path.glob``: glob swallows ``scandir`` errors and would turn a
    failed listing into an empty directory. ESTALE is retried; any other error raises.
    """
    for attempt in range(ESTALE_TRIES):
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            return []
        except OSError as e:
            if e.errno != errno.ESTALE or attempt == ESTALE_TRIES - 1:
                raise
            time.sleep(min(0.05 * 2**attempt, 1.0))
            continue
        return sorted(d / n for n in names if n.endswith(suffix) and not n.startswith("."))
    raise AssertionError("unreachable")  # pragma: no cover


def _listing(root: Path, dirname: str) -> list[Path]:
    return _names(root / dirname, ".jsonl")


def _scan(root: Path) -> dict[Path, _Parsed]:
    """Parse every readable file. Restart when a listed file vanishes or a sealed file appears.

    ``segments/`` is listed before ``sealed/`` so a compaction that moves segments after this
    listing is caught: the moved file vanishes, or a new sealed file appears. Rescans back off
    (:data:`RESCAN_SLEEPS_S`) past the NFS listing-cache window: a segment is archived only
    after the sealed file holding its rows is visible everywhere, so a vanished segment means a
    stale listing, which a wait cures.
    """
    for attempt in range(MAX_RESCANS):
        if attempt:
            time.sleep(RESCAN_SLEEPS_S[attempt - 1])
        listed = [p for d in READ_DIRS for p in _listing(root, d)]
        sealed_before = {p for p in listed if p.parent.name == SEALED_DIRNAME}
        parsed: dict[Path, _Parsed] = {}
        vanished = False
        for p in listed:
            try:
                parsed[p] = _parse(p, _read_bytes(p))
            except FileNotFoundError:
                vanished = True
                break
        if not vanished and set(_listing(root, SEALED_DIRNAME)) <= sealed_before:
            return parsed
    raise UnitIndexError(f"{root}: the index kept changing during {MAX_RESCANS} read attempts")


@dataclasses.dataclass(slots=True)
class ReadReport:
    """Filled by :func:`iter_rows` when passed in."""

    files: int = 0
    rows_read: int = 0
    duplicates: int = 0
    torn: list[BadLine] = dataclasses.field(default_factory=list)
    unreadable: list[BadLine] = dataclasses.field(default_factory=list)


def iter_rows(root: Path | str, *, report: ReadReport | None = None) -> Iterator[IndexRow]:
    """Every row in ``segments/``, ``sealed/`` and ``rebuilt/``, deduplicated on the read key."""
    parsed = _scan(Path(root))
    rep = report if report is not None else ReadReport()
    rep.files = len(parsed)
    seen: set[tuple[str, str, str]] = set()
    for p in parsed.values():
        rep.torn.extend(p.torn)
        rep.unreadable.extend(p.unreadable)
        for row in p.rows:
            rep.rows_read += 1
            if row.read_key in seen:
                rep.duplicates += 1
                continue
            seen.add(row.read_key)
            yield row


# ---- compaction ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SegmentDecision:
    name: str
    taken: bool
    reason: str


@dataclasses.dataclass(frozen=True, slots=True)
class Conflict:
    """R-S6: one ``(unit_key, input_digest)`` with more than one content digest."""

    unit_key: str
    input_digest: str
    content_digests: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class CompactReport:
    """What one compaction pass did (or, with ``dry_run``, would do)."""

    dry_run: bool
    segments: tuple[SegmentDecision, ...] = ()
    sealed_inputs: tuple[str, ...] = ()
    rows_in: int = 0
    rows_sealed: int = 0
    collapsed: int = 0
    copies: int = 0
    conflicts: tuple[Conflict, ...] = ()
    torn: tuple[BadLine, ...] = ()
    left_unreadable: tuple[str, ...] = ()
    sealed_path: str | None = None
    archived: tuple[str, ...] = ()
    pending_archive: tuple[str, ...] = ()
    scheduler_calls: int = 0
    partials_removed: int = 0

    @property
    def taken(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.segments if d.taken)


class _PassRunner:
    """One pass's scheduler runner: caches each query, and after the first failure answers
    every later query with a failure (UNKNOWN, never dead) instead of waiting on a hung squeue."""

    def __init__(self, inner: Runner | None) -> None:
        self._inner = inner if inner is not None else _wl._subprocess_runner
        self._cache: dict[tuple[str, ...], tuple[int, str, str]] = {}
        self._broken: str | None = None
        self.calls = 0

    def __call__(self, argv: list[str], timeout: float) -> tuple[int, str, str]:
        key = tuple(argv)
        if key in self._cache:
            return self._cache[key]
        if self._broken is not None:
            raise OSError(f"scheduler query skipped after an earlier {self._broken}")
        self.calls += 1
        try:
            result = self._inner(argv, timeout)
        except (OSError, subprocess.SubprocessError) as e:
            self._broken = type(e).__name__
            raise
        self._cache[key] = result
        return result


def _decide(
    segments: list[Path],
    *,
    grace_s: int | float,
    now_server: float | None,
    runner: _PassRunner,
) -> list[SegmentDecision]:
    out = []
    now = now_server
    for p in segments:
        try:
            if now is None and not _wl._has_trailer(p):
                now = server_now(p.parent)  # once per pass, and only if a liveness test needs it
            v = segment_writer_dead(p, grace_s=grace_s, now_server=now, runner=runner)
        except FileNotFoundError:
            out.append(SegmentDecision(p.name, False, "vanished (moved by another compactor)"))
            continue
        out.append(SegmentDecision(p.name, v.dead, v.reason))
    return out


def _collapse(rows: list[IndexRow]) -> tuple[list[IndexRow], int, int, list[Conflict]]:
    """Group by unit_id; one row per (content_digest, store, locator), the latest by ts."""
    by_place: dict[tuple[str, ...], IndexRow] = {}
    collapsed = 0
    for r in rows:
        k = (*r.unit_id, r.content_digest, r.store, r.locator)
        cur = by_place.get(k)
        if cur is None:
            by_place[k] = r
            continue
        collapsed += 1
        if (r.ts, r.record_digest) > (cur.ts, cur.record_digest):
            by_place[k] = r
    kept = sorted(
        by_place.values(), key=lambda r: (*r.unit_id, r.content_digest, r.store, r.locator)
    )
    contents: dict[tuple[str, str], set[str]] = {}
    places: dict[tuple[str, str, str], int] = {}
    for r in kept:
        contents.setdefault(r.unit_id, set()).add(r.content_digest)
        k3 = (*r.unit_id, r.content_digest)
        places[k3] = places.get(k3, 0) + 1
    conflicts = [
        Conflict(uk, ind, tuple(sorted(cds)))
        for (uk, ind), cds in sorted(contents.items())
        if len(cds) > 1
    ]
    copies = sum(n - 1 for n in places.values())
    return kept, collapsed, copies, conflicts


# What identifies the file that was read: (size, inode, mtime_ns). All three come from the NFS
# server, so another client sees the same values; st_dev is client-local and is left out.
_Ident = tuple[int, int, int]


def _ident(st: os.stat_result) -> _Ident:
    return (st.st_size, st.st_ino, st.st_mtime_ns)


@dataclasses.dataclass(frozen=True, slots=True)
class _Manifest:
    """``sealed/<stem>.inputs.json``: the inputs a sealed file holds, as identified when read."""

    path: Path
    sealed: Path
    inputs: tuple[tuple[Path, _Ident], ...]


def _load_manifest(root: Path, path: Path) -> _Manifest | None:
    try:
        raw = _read_bytes(path)
    except FileNotFoundError:
        return None  # removed by a concurrent compactor
    try:
        d = json.loads(raw)
        sealed = root / SEALED_DIRNAME / d["sealed"]
        inputs = []
        for item in d["inputs"]:
            rel = Path(item["path"])
            if len(rel.parts) != 2 or rel.parts[0] not in _ARCHIVE_PREFIX:
                raise ValueError(f"input path {item['path']!r} is outside segments/ and sealed/")
            ident = (item["size"], item["ino"], item["mtime_ns"])
            if rel.parts[1].startswith(".") or not all(type(x) is int for x in ident):
                raise ValueError(f"bad input entry {item!r}")
            inputs.append((root / rel, ident))
        if Path(d["sealed"]).name != d["sealed"]:
            raise ValueError(f"sealed name {d['sealed']!r} is not a plain file name")
    except (ValueError, KeyError, TypeError) as e:
        raise UnitIndexError(f"refusing to compact {root}: manifest {path.name} is malformed: {e}")
    return _Manifest(path, sealed, tuple(inputs))


def _manifests(root: Path, *, remove_orphans: bool = False) -> dict[Path, _Manifest]:
    """Manifests whose sealed file exists, keyed by that sealed file.

    A manifest whose sealed file is gone (archived by a concurrent compactor's later pass) is an
    orphan: it is ignored, and removed when ``remove_orphans``. Ignoring or removing a manifest
    can only make its inputs be sealed again (duplicates), never archive them.
    """
    out = {}
    for p in _names(root / SEALED_DIRNAME, MANIFEST_SUFFIX):
        m = _load_manifest(root, p)
        if m is None:
            continue
        if m.sealed.exists():
            out[m.sealed] = m
        elif remove_orphans:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
    return out


def _write_atomic(path: Path, data: bytes) -> None:
    """temp (``O_EXCL``) + fsync + ``os.replace`` + fsync of the directory."""
    tmp = path.with_name(f".{path.name}.partial")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644)
    try:
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    if path.suffix == ".jsonl":
        _fault("sealed_tmp_written")
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _archive_due(
    root: Path,
    manifests: dict[Path, _Manifest],
    *,
    archive_after_s: int | float,
    archive_now: float | None,
    still_dead: Callable[[Path], bool],
) -> list[str]:
    """Archive the inputs of every sealed file at least ``archive_after_s`` old (server time).

    An input that is itself a sealed file with a manifest has its own inputs archived too: they
    are inside it, and so inside the old-enough file. An input stays where it is when its
    ``(size, inode, mtime_ns)`` differs from what the manifest recorded (it grew, or the path now
    names another file), or when it is a segment without a trailer whose writer
    ``still_dead`` no longer judges dead; once its manifest is gone it is taken again. The
    manifest is removed last, after the moves are fsynced.
    """
    if not manifests:
        return []
    now = 0.0
    if archive_after_s > 0:
        now = archive_now if archive_now is not None else server_now(root / SEALED_DIRNAME)
    due = []
    for sealed, m in manifests.items():
        if archive_after_s > 0:
            try:
                age = now - sealed.stat().st_mtime
            except FileNotFoundError:
                continue
            if age < archive_after_s:
                continue
        due.append(m)
    if not due:
        return []
    archive_dir = root / ARCHIVE_DIRNAME
    _ensure_dir(archive_dir)
    archived: list[str] = []
    done: list[_Manifest] = []
    stack = list(due)
    seen: set[Path] = set()
    while stack:
        m = stack.pop()
        if m.path in seen:
            continue
        seen.add(m.path)
        done.append(m)
        for p, ident in m.inputs:
            try:
                if _ident(p.stat()) != ident:
                    continue  # not the file that was read: its rows may not be in the sealed file
                inner = manifests.get(p)
                if inner is not None:
                    stack.append(inner)  # the file that was read, so its own inputs are inside
                if p.parent.name == SEGMENTS_DIRNAME and not _wl._has_trailer(p):
                    if not still_dead(p):
                        continue  # an open descriptor would follow the rename out of view
            except FileNotFoundError:
                continue  # archived already, by this pass or a concurrent compactor
            name = _ARCHIVE_PREFIX[p.parent.name] + p.name
            dest = archive_dir / name
            if dest.exists():  # never overwrite an archived file (a reused explicit writer_id)
                dest = archive_dir / f"{name}.{uuid.uuid4().hex[:8]}"
            try:
                os.replace(p, dest)
            except FileNotFoundError:
                continue
            archived.append(p.name)
            if len(archived) == 1:
                _fault("first_input_archived")
    for d in (archive_dir, root / SEALED_DIRNAME, root / SEGMENTS_DIRNAME):
        if d.is_dir():
            _fsync_dir(d)
    for m in done:
        try:
            m.path.unlink()
        except FileNotFoundError:
            pass
    _fsync_dir(root / SEALED_DIRNAME)
    return archived


def compact(
    root: Path | str,
    *,
    grace_s: int | float = DEFAULT_GRACE_S,
    dry_run: bool = False,
    runner: Runner | None = None,
    now_server: float | None = None,
    archive_after_s: int | float = DEFAULT_ARCHIVE_AFTER_S,
    archive_now: float | None = None,
) -> CompactReport:
    """Fold closed and dead segments, plus the existing sealed files, into one new sealed file.

    A pass has two halves (decision D6):

    1. **Archive.** For every sealed file at least ``archive_after_s`` old (NFS server time;
       ``archive_now`` overrides the probe), move the inputs its manifest lists to ``archive/``,
       then remove the manifest. Inputs stay readable until then, so a reader on a host whose
       ``sealed/`` listing is cached never loses sight of a row.
    2. **Seal.** Inputs still listed by a manifest are skipped. The rest (closed segments, dead
       writers' segments, other sealed files) are read and written to one new sealed file (temp,
       fsync, ``os.replace``, fsync ``sealed/``), then its manifest is written the same way.
       With ``archive_after_s <= 0`` the archive half runs again at once.

    A pass with nothing due and no segment to take changes nothing. A segment holding an
    unreadable row stays in place; a sealed file holding one refuses the pass
    (:class:`UnitIndexError`, nothing written). Torn lines in taken segments are dropped and
    reported. ``dry_run`` reports the same decisions, archives nothing and writes nothing but the
    server-time probe.
    """
    root = Path(root)
    pass_runner = _PassRunner(runner)
    liveness_now: list[float] = [] if now_server is None else [now_server]

    def still_dead(p: Path) -> bool:
        """Re-ask liveness for a trailer-less segment just before archiving it."""
        if not liveness_now:
            liveness_now.append(server_now(p.parent))
        try:
            v = segment_writer_dead(
                p, grace_s=grace_s, now_server=liveness_now[0], runner=pass_runner
            )
        except FileNotFoundError:
            return False
        return v.dead

    archived: list[str] = []
    if not dry_run:
        archived += _archive_due(
            root,
            _manifests(root, remove_orphans=True),
            archive_after_s=archive_after_s,
            archive_now=archive_now,
            still_dead=still_dead,
        )
    pending = {p for m in _manifests(root).values() for p, _ in m.inputs}
    segments = [p for p in _listing(root, SEGMENTS_DIRNAME) if p not in pending]
    decisions = _decide(segments, grace_s=grace_s, now_server=now_server, runner=pass_runner)

    taken: dict[Path, _Parsed] = {}
    idents: dict[Path, _Ident] = {}

    def read_with_ident(p: Path) -> tuple[bytes, _Ident | None]:
        """Read, then stat; ``None`` ident when the file changed while it was being read."""
        raw = _read_bytes(p)
        ident = _ident(p.stat())
        return raw, (ident if ident[0] == len(raw) else None)

    left_unreadable: list[str] = []
    final: list[SegmentDecision] = [
        SegmentDecision(p.name, False, "already sealed; archive pending")
        for p in sorted(pending)
        if p.parent.name == SEGMENTS_DIRNAME
    ]
    for p, d in zip(segments, decisions, strict=True):
        if not d.taken:
            final.append(d)
            continue
        try:
            raw, ident = read_with_ident(p)
        except FileNotFoundError:
            final.append(SegmentDecision(d.name, False, "vanished (moved by another compactor)"))
            continue
        if ident is None:
            final.append(SegmentDecision(d.name, False, "changed while being read; next pass"))
            continue
        parsed = _parse(p, raw)
        if parsed.unreadable:
            left_unreadable.append(d.name)
            final.append(
                SegmentDecision(d.name, False, f"left in place: {parsed.unreadable[0].reason}")
            )
            continue
        taken[p] = parsed
        idents[p] = ident
        final.append(d)

    def report(**kw: Any) -> CompactReport:
        return CompactReport(
            dry_run=dry_run,
            segments=tuple(final),
            left_unreadable=tuple(left_unreadable),
            scheduler_calls=pass_runner.calls,
            archived=tuple(archived),
            pending_archive=tuple(sorted(p.name for p in pending)),
            **kw,
        )

    if not taken:
        return report()

    sealed_in: dict[Path, _Parsed] = {}
    for p in _listing(root, SEALED_DIRNAME):
        if p in pending:
            continue  # already inside a newer sealed file that is waiting to archive it
        try:
            raw, ident = read_with_ident(p)
        except FileNotFoundError:
            continue  # archived by a concurrent compactor; its rows are in that one's output
        if ident is None:
            continue  # sealed files never change; skipping one only leaves duplicates
        parsed = _parse(p, raw)
        if parsed.unreadable or parsed.torn:
            bad = (parsed.unreadable or parsed.torn)[0]
            raise UnitIndexError(
                f"refusing to compact {root}: sealed file {p.name} line {bad.line} is "
                f"{bad.kind} ({bad.reason}); sealing now would archive it out of view"
            )
        sealed_in[p] = parsed
        idents[p] = ident

    rows = [r for p in (*sealed_in.values(), *taken.values()) for r in p.rows]
    kept, collapsed, copies, conflicts = _collapse(rows)
    torn = tuple(b for p in taken.values() for b in p.torn)
    summary = dict(
        sealed_inputs=tuple(p.name for p in sealed_in),
        rows_in=len(rows),
        rows_sealed=len(kept),
        collapsed=collapsed,
        copies=copies,
        conflicts=tuple(conflicts),
        torn=torn,
    )
    if dry_run:
        return report(**summary)

    sealed_dir = root / SEALED_DIRNAME
    _ensure_dir(sealed_dir)
    partials_removed = _sweep_partials(sealed_dir)
    target = sealed_dir / f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.jsonl"
    _write_atomic(target, "".join(r.to_json_line() for r in kept).encode("utf-8"))
    _fault("sealed_replaced")
    manifest = {
        "sealed": target.name,
        "inputs": [
            {
                "path": f"{p.parent.name}/{p.name}",
                "size": idents[p][0],
                "ino": idents[p][1],
                "mtime_ns": idents[p][2],
            }
            for p in (*sealed_in, *taken)
        ],
    }
    manifest_path = sealed_dir / f"{target.stem}{MANIFEST_SUFFIX}"
    _write_atomic(manifest_path, json.dumps(manifest, sort_keys=True).encode("utf-8"))
    _fault("manifest_written")
    if archive_after_s <= 0:
        m = _load_manifest(root, manifest_path)
        if m is not None:
            archived += _archive_due(
                root, {target: m}, archive_after_s=0, archive_now=None, still_dead=still_dead
            )
    return report(
        **summary,
        sealed_path=str(target),
        partials_removed=partials_removed,
    )


def _sweep_partials(sealed_dir: Path) -> int:
    """Remove sealed temp files a crashed compactor left, once older than ``PARTIAL_MAX_AGE_S``.

    Ages use NFS server time. A live compactor's temp file exists for seconds, so the age bound
    never touches one in flight; the temp holds no row that is not still in its inputs.
    """
    partials = list(sealed_dir.glob(".*.partial"))
    if not partials:
        return 0
    now = server_now(sealed_dir)
    removed = 0
    for p in partials:
        try:
            if now - p.stat().st_mtime > PARTIAL_MAX_AGE_S:
                p.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed
