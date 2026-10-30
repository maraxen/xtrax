"""File-count (inode) guard core: stdlib-only, Python >= 3.9, node-embeddable.

Lifted from alynxr ``storage.py`` (spec 261030_ledger-pack-fileguard.md section 3.4) with its
semantics intact: the real quota is read at construction and re-read every ``reread_every``
counted files; between reads the local counter only interpolates. A check that would push the
projected count above ``cap_fraction`` of the limit is refused before the write, including mid-run.

SELF-CONTAINED BY DESIGN. This module imports only the standard library and nothing from xtrax,
uses no syntax newer than Python 3.9, and is exported as source text by
:func:`xtrax.fileguard.node_source` so a cluster script can embed it where xtrax is not installed.
``tests/fileguard/test_node_standalone.py`` pins both properties (an ``ast`` import check and an
isolated ``-I`` run under a 3.9 interpreter).

Quota sources:

* Lustre: ``lfs quota -u <user>`` (and ``-g <group>`` when the target directory is setgid, since
  files created there count against the group). If ``lfs`` is missing, fails or does not parse,
  the read is REFUSED; ``df -i`` is filesystem-wide on Lustre and is never a substitute.
* Elsewhere (NFS, ext4, ...): the ``quota -w -u`` (and ``-g``) row for the path's ``df`` device,
  plus the filesystem-wide ``df -i -P`` reading. On a shared NFS server ``df -i`` is the whole
  server and never the per-user cap, so a
  path with no ``quota`` row emits ``quota_limit_unknown`` for the user scope instead of being
  trusted silently. An unknown filesystem type (``stat -f`` fails) is refused.

Events go through an injectable ``emit(name, fields)`` callable. The default logs one JSON line
per event on the ``xtrax.fileguard`` logger.
"""

import getpass
import grp
import json
import logging
import os
import shlex
import stat as _stat
import subprocess
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, Union

Runner = Callable[[List[str]], Tuple[Optional[int], str]]
Emitter = Callable[[str, Dict[str, object]], None]
PathLike = Union[str, Path]

#: ``stat -f -c %T`` name of a Lustre mount.
LUSTRE = "lustre"
REFUSAL_EVENT = "xtrax.fileguard.refusal"
READING_EVENT = "xtrax.fileguard.quota_reading"
LIMIT_UNKNOWN_EVENT = "xtrax.fileguard.quota_limit_unknown"

_logger = logging.getLogger("xtrax.fileguard")


def log_emitter(name: str, fields: Dict[str, object]) -> None:
    """Default emitter: one JSON line per event on the ``xtrax.fileguard`` logger."""
    _logger.info("%s %s", name, json.dumps(fields, sort_keys=True, default=str))


class QuotaRefused(ValueError):
    """A write would push file usage above the guard's cap fraction of the real limit."""

    rid = "R-X1"


class QuotaUnreadable(QuotaRefused):
    """The real quota cannot be read, so nothing is guessed (R-X1)."""


@dataclass(frozen=True)
class QuotaReading:
    """One real reading of a file quota."""

    files_used: int
    files_limit: Optional[int]
    source: str  # "lfs" | "df" | "synthetic"
    command: str
    raw: str
    path: str
    taken_at: str  # ISO 8601, UTC
    fs_type: Optional[str] = None
    scope: str = "user"  # "user" | "group" | "filesystem"
    principal: Optional[str] = None  # user or group name the reading is for

    @property
    def fraction(self) -> Optional[float]:
        if not self.files_limit:
            return None
        return self.files_used / self.files_limit


@dataclass(frozen=True)
class LocationInfo:
    """Ownership facts of the directory files will be created in."""

    path: str  # the nearest existing ancestor that was inspected
    gid: int
    group: Optional[str]
    setgid: bool


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(token: str, where: str) -> int:
    """Parse one numeric cell; a trailing ``*`` marks over-quota in lfs output."""
    raw = token[:-1] if token.endswith("*") else token
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"unparseable {where} value {token!r}") from None


def _limit_from(hard: int, soft: int) -> Optional[int]:
    """Hard limit wins; the soft quota is used if the hard limit is unset; both 0 is unlimited."""
    if hard > 0:
        return hard
    if soft > 0:
        return soft
    return None


def parse_lfs_quota(text: str, path: PathLike) -> QuotaReading:
    """Parse ``lfs quota -u|-g <who> <path>`` output.

    Accepts the single-row layout and the variant where the filesystem name sits on its own line
    with the numbers on the next. Raises ValueError on anything else.
    """
    lines = text.splitlines()
    header = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("Filesystem")), None)
    if header is not None:
        body = lines[header + 1 :]
    else:
        body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("Disk quotas")]
    tokens = " ".join(body).split()

    fs_idx = next((i for i, t in enumerate(tokens) if t.startswith("/")), None)
    if fs_idx is None:
        raise ValueError("lfs quota output has no filesystem row")
    if fs_idx != 0:
        raise ValueError(f"unexpected tokens before filesystem row: {tokens[:fs_idx]!r}")
    rest = tokens[fs_idx + 1 :]
    if len(rest) != 8:
        raise ValueError(f"expected 8 quota cells after filesystem, got {len(rest)}: {rest!r}")

    _num(rest[0], "kbytes")
    _num(rest[1], "kbytes quota")
    _num(rest[2], "kbytes limit")
    files_used = _num(rest[4], "files")
    files_soft = _num(rest[5], "files quota")
    files_hard = _num(rest[6], "files limit")
    return QuotaReading(
        files_used=files_used,
        files_limit=_limit_from(files_hard, files_soft),
        source="lfs",
        command="",  # filled in by read_quota, which knows the argv
        raw=text,
        path=str(path),
        taken_at=_now_utc(),
    )


def parse_df_inodes(text: str, path: PathLike) -> QuotaReading:
    """Parse ``df -i -P <path>`` (POSIX): Filesystem Inodes IUsed IFree IUse% Mounted on."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 2 or not lines[0].lstrip().startswith("Filesystem"):
        raise ValueError(f"unexpected df -i -P output: {text!r}")
    parts = lines[1].split(None, 5)
    if len(parts) < 6:
        raise ValueError(f"df -i -P data row has {len(parts)} fields, expected 6: {lines[1]!r}")
    inodes, iused = parts[1], parts[2]
    files_used = _num(iused, "IUsed")
    files_limit = None if inodes == "-" else _num(inodes, "Inodes")
    return QuotaReading(
        files_used=files_used,
        files_limit=files_limit or None,
        source="df",
        command="",
        raw=text,
        path=str(path),
        taken_at=_now_utc(),
        scope="filesystem",
    )


def default_runner(argv: List[str]) -> Tuple[Optional[int], str]:
    """Run ``argv`` with a 30 s timeout; returns ``(returncode, stdout)``."""
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)
    return proc.returncode, proc.stdout


def _run(run: Runner, argv: List[str]) -> Tuple[Optional[int], str]:
    try:
        return run(argv)
    except FileNotFoundError:
        return None, ""


def existing_ancestor(path: PathLike) -> Path:
    """``path`` itself if it exists, else its nearest existing ancestor (files go there first)."""
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return p


def describe_location(path: PathLike) -> LocationInfo:
    """Group and setgid bit of the directory new files under ``path`` will be created in."""
    p = existing_ancestor(path)
    st = p.stat()
    try:
        group: Optional[str] = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = None
    return LocationInfo(
        path=str(p), gid=st.st_gid, group=group, setgid=bool(st.st_mode & _stat.S_ISGID)
    )


def _refuse(
    emit: Emitter, path: PathLike, fs_type: Optional[str], command: str, detail: str
) -> NoReturn:
    """Emit the R-X1 refusal event and raise. Never returns."""
    emit(
        REFUSAL_EVENT,
        {
            "refusal_id": QuotaRefused.rid,
            "path": str(path),
            "fs_type": fs_type,
            "command": command,
            "detail": detail,
        },
    )
    raise QuotaUnreadable(f"R-X1: {detail}")


def fs_type(path: PathLike, runner: Optional[Runner] = None, emit: Emitter = log_emitter) -> str:
    """Filesystem type of ``path`` (or its nearest existing ancestor) from ``stat -f -c %T``.

    Refuses (R-X1) when the type cannot be read: which quota applies is then unknown.
    """
    run = runner or default_runner
    argv = ["stat", "-f", "-c", "%T", str(existing_ancestor(path))]
    rc, out = _run(run, argv)
    name = out.strip()
    if rc != 0 or not name or "\n" in name:
        _refuse(
            emit,
            path,
            None,
            shlex.join(argv),
            f"cannot determine the filesystem type of {path} (stat rc={rc}); refusing to "
            "guess which file quota applies",
        )
    return name


def _read_lfs(
    run: Runner, emit: Emitter, path: PathLike, kind: str, flag: str, who: str
) -> QuotaReading:
    argv = ["lfs", "quota", flag, who, str(existing_ancestor(path))]
    command = shlex.join(argv)
    rc, out = _run(run, argv)
    scope = "user" if flag == "-u" else "group"
    if rc != 0:
        _refuse(
            emit,
            path,
            kind,
            command,
            f"cannot read the per-{scope} quota on Lustre for {path} (lfs rc={rc}); refusing "
            "rather than using a filesystem-wide df -i, which never reaches the per-user cap",
        )
    try:
        r = parse_lfs_quota(out, path)
    except ValueError as e:
        _refuse(
            emit,
            path,
            kind,
            command,
            f"cannot read the per-{scope} quota on Lustre for {path} (lfs output did not "
            f"parse: {e}); refusing rather than using a filesystem-wide df -i",
        )
    return replace(r, command=command, fs_type=kind, scope=scope, principal=who)


def read_quota(
    path: PathLike,
    runner: Optional[Runner] = None,
    user: Optional[str] = None,
    emit: Emitter = log_emitter,
) -> QuotaReading:
    """Read the tightest-scoped file quota for ``path``: the user's when one is reported.

    On Lustre: ``lfs quota -u``; a missing, failing or unparseable ``lfs`` is REFUSED (R-X1).
    Elsewhere: the ``quota -w -u`` row for the path's device when there is one, else the
    filesystem-wide ``df -i -P`` reading (see :func:`read_quotas`). An unknown filesystem type
    is refused.
    """
    return read_quotas(path, runner=runner, user=user, emit=emit, group=None)[0]


def parse_quota_rows(text: str) -> Dict[str, Tuple[int, Optional[int]]]:
    """``{device: (files_used, files_limit)}`` from ``quota -w -u|-g <who>`` output.

    Columns are ``Filesystem blocks quota limit grace files quota limit grace``; a grace cell
    is present only when the used value before it is over quota (marked ``*``). ``-w`` keeps
    long device names on one line. A device listed twice keeps its first row. Output such as
    ``Disk quotas for group g (gid 1): none`` has no rows.
    """
    rows: Dict[str, Tuple[int, Optional[int]]] = {}
    seen_header = False
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "Filesystem":
            seen_header = True
            continue
        if not seen_header or tokens[0] == "Disk":
            continue
        rest = tokens[1:]
        i = 4 if rest and rest[0].endswith("*") else 3
        if len(rest) < i + 3:
            raise ValueError(f"quota row has {len(rest)} cells after the device: {line!r}")
        files_used = _num(rest[i], "files")
        limit = _limit_from(_num(rest[i + 2], "files limit"), _num(rest[i + 1], "files quota"))
        rows.setdefault(tokens[0], (files_used, limit))
    return rows


def _df_device(df_text: str) -> str:
    """The device column of a validated ``df -i -P`` data row."""
    lines = [ln for ln in df_text.splitlines() if ln.strip()]
    return lines[1].split(None, 1)[0]


def _read_quota_cmd(
    run: Runner,
    emit: Emitter,
    path: PathLike,
    kind: str,
    device: str,
    flag: str,
    who: str,
) -> Optional[QuotaReading]:
    """The ``quota -w`` reading for ``device``, or None (with an event) when none is reported."""
    argv = ["quota", "-w", flag, who]
    command = shlex.join(argv)
    scope = "user" if flag == "-u" else "group"
    rc, out = _run(run, argv)
    detail = None
    rows: Dict[str, Tuple[int, Optional[int]]] = {}
    if rc is None:
        detail = "the quota command is not installed"
    else:
        try:
            rows = parse_quota_rows(out)
        except ValueError as e:
            detail = f"quota output did not parse: {e}"
        if detail is None and device not in rows:
            detail = f"quota reports no {scope} row for device {device} (rc={rc})"
    if detail is not None:
        # Not a refusal: the filesystem reading still applies. The guard is blind to a
        # per-principal cap here, and the operator must see that rather than infer it.
        emit(
            LIMIT_UNKNOWN_EVENT,
            {
                "refusal_id": "R-X1-limit-unknown",
                "path": str(path),
                "command": command,
                "scope": scope,
                "principal": who,
                "device": device,
                "detail": detail,
            },
        )
        return None
    files_used, files_limit = rows[device]
    return QuotaReading(
        files_used=files_used,
        files_limit=files_limit,
        source="quota",
        command=command,
        raw=out,
        path=str(path),
        taken_at=_now_utc(),
        fs_type=kind,
        scope=scope,
        principal=who,
    )


def read_quotas(
    path: PathLike,
    runner: Optional[Runner] = None,
    user: Optional[str] = None,
    emit: Emitter = log_emitter,
    group: Optional[str] = None,
) -> List[QuotaReading]:
    """Every quota that bounds new files under ``path``, tightest scope first.

    Lustre: the user's ``lfs quota -u`` reading, plus ``lfs quota -g <group>`` when ``group``
    is given. Elsewhere (NFS, ext4, ...): the ``quota -w -u`` (and ``-g``) row whose device
    matches the path's ``df`` device, then the filesystem-wide ``df -i -P`` reading. On a
    shared NFS server the ``df`` figure is the whole server, so it is never the per-user cap;
    when ``quota`` reports no row for the device, a ``quota_limit_unknown`` event says the
    guard cannot see a per-user cap there.
    """
    run = runner or default_runner
    who = user or getpass.getuser()
    kind = fs_type(path, run, emit)
    if kind == LUSTRE:
        readings = [_read_lfs(run, emit, path, kind, "-u", who)]
        if group is not None:
            readings.append(_read_lfs(run, emit, path, kind, "-g", group))
        return readings
    argv = ["df", "-i", "-P", str(existing_ancestor(path))]
    rc, out = _run(run, argv)
    if rc != 0:
        raise RuntimeError(f"cannot read file quota for {path}: df rc={rc}")
    fs_reading = replace(parse_df_inodes(out, path), command=shlex.join(argv), fs_type=kind)
    device = _df_device(out)
    readings: List[QuotaReading] = []
    principals = [("-u", who)] + ([("-g", group)] if group is not None else [])
    for flag, principal in principals:
        r = _read_quota_cmd(run, emit, path, kind, device, flag, principal)
        if r is not None:
            readings.append(r)
    readings.append(fs_reading)
    return readings


def check_concurrency_slack(
    files_limit: Optional[int], *, n_jobs: int, reread_every: int, cap_fraction: float
) -> None:
    """Plan-time refusal when N concurrent guards could jointly overshoot the real limit.

    Each guard interpolates up to ``reread_every`` files between real reads without seeing the
    others, so N guards can overshoot by up to ``n_jobs * reread_every``. That must fit in the
    slack above the cap: ``n_jobs * reread_every < (1 - cap_fraction) * files_limit``.
    """
    if n_jobs < 1 or reread_every < 1:
        raise ValueError(f"n_jobs and reread_every must be >= 1, got {n_jobs}, {reread_every}")
    if not files_limit:
        return
    overshoot = n_jobs * reread_every
    slack = (1.0 - cap_fraction) * files_limit
    if overshoot >= slack:
        raise QuotaRefused(
            f"R-X1: {n_jobs} concurrent guards x reread_every={reread_every} can overshoot by "
            f"{overshoot} files, which does not fit the {slack:.0f}-file slack above "
            f"cap_fraction={cap_fraction:.0%} of files_limit {files_limit}; lower reread_every, "
            "lower n_jobs, or lower cap_fraction"
        )


def _persist(record_path: Path, readings: List[QuotaReading], location: LocationInfo) -> None:
    """Overwrite ``record_path`` atomically (temp file in the same directory, then os.replace)."""
    record_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("command", "fs_type", "source", "scope", "principal", "files_used", "files_limit")
    doc = {
        "readings": [{k: getattr(r, k) for k in (*fields, "taken_at", "path")} for r in readings],
        "location": asdict(location),
    }
    tmp = record_path.with_name(f".{record_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, record_path)
    finally:
        if tmp.exists():
            tmp.unlink()


class FileCapGuard:
    """Refuses writes that would push real file usage above ``cap_fraction`` of the limit.

    The real readings are taken at construction and re-taken once ``reread_every`` files have
    been counted since the last read. When the target directory is setgid, the owning
    group's quota is read and enforced alongside the user's (files created there count against
    the group). A reading with no limit never refuses; it emits ``quota_limit_unknown``.

    ``record_path``, when given, receives the latest readings and location as JSON after every
    real read. There is no default location.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        cap_fraction: float = 0.80,
        reread_every: int = 1000,
        runner: Optional[Runner] = None,
        user: Optional[str] = None,
        emit: Emitter = log_emitter,
        record_path: Optional[PathLike] = None,
        check_setgid_group: bool = True,
    ) -> None:
        if not 0.0 < cap_fraction <= 1.0:
            raise ValueError(f"cap_fraction must be in (0, 1], got {cap_fraction}")
        if reread_every < 1:
            raise ValueError(f"reread_every must be >= 1, got {reread_every}")
        self.path = path
        self.cap_fraction = cap_fraction
        self.reread_every = reread_every
        self.record_path = None if record_path is None else Path(record_path)
        self._runner = runner
        self._user = user
        self._emit = emit
        self.location = describe_location(path)
        self._group = self.location.group if check_setgid_group and self.location.setgid else None
        self.last_readings: List[QuotaReading] = self._take()
        self.files_written_since_read = 0
        self.n_reads = 1

    @property
    def last_reading(self) -> QuotaReading:
        """The user (or filesystem) reading; group readings are in :attr:`last_readings`."""
        return self.last_readings[0]

    def _take(self) -> List[QuotaReading]:
        readings = read_quotas(
            self.path, runner=self._runner, user=self._user, emit=self._emit, group=self._group
        )
        for r in readings:
            self._emit(
                READING_EVENT,
                {
                    "path": r.path,
                    "command": r.command,
                    "fs_type": r.fs_type,
                    "source": r.source,
                    "scope": r.scope,
                    "principal": r.principal,
                    "files_used": r.files_used,
                    "files_limit": r.files_limit,
                    "taken_at": r.taken_at,
                    "setgid": self.location.setgid,
                    "group": self.location.group,
                },
            )
            if r.files_limit is None:
                # An unset quota is legitimate: no refusal, but the operator should see it.
                self._emit(
                    LIMIT_UNKNOWN_EVENT,
                    {
                        "refusal_id": "R-X1-limit-unknown",
                        "path": r.path,
                        "command": r.command,
                        "scope": r.scope,
                        "files_used": r.files_used,
                    },
                )
        if self.record_path is not None:
            _persist(self.record_path, readings, self.location)
        return readings

    def check(self, n_new_files: int) -> int:
        """Admit ``n_new_files`` or raise QuotaRefused. Returns the projected user file count."""
        if n_new_files < 0:
            raise ValueError(f"n_new_files must be >= 0, got {n_new_files}")
        projections = []
        for r in self.last_readings:
            projected = r.files_used + self.files_written_since_read + n_new_files
            projections.append(projected)
            if r.files_limit and projected > self.cap_fraction * r.files_limit:
                self._emit(
                    REFUSAL_EVENT,
                    {
                        "refusal_id": QuotaRefused.rid,
                        "scope": r.scope,
                        "principal": r.principal,
                        "files_used": r.files_used,
                        "files_limit": r.files_limit,
                        "files_written_since_read": self.files_written_since_read,
                        "n_new_files": n_new_files,
                        "projected": projected,
                        "cap_fraction": self.cap_fraction,
                        "source": r.source,
                        "command": r.command,
                    },
                )
                raise QuotaRefused(
                    f"R-X1: refusing {n_new_files} new file(s) at {self.path}: projected "
                    f"{projected} exceeds {self.cap_fraction:.0%} of the {r.scope} files_limit "
                    f"{r.files_limit} (used {r.files_used} at read + "
                    f"{self.files_written_since_read} written since; source={r.source}, "
                    f"command={r.command!r})"
                )
        self.files_written_since_read += n_new_files
        if self.files_written_since_read >= self.reread_every:
            self.last_readings = self._take()
            self.files_written_since_read = 0
            self.n_reads += 1
        return projections[0]

    def record(self) -> Dict[str, object]:
        """The guard's state: last readings, location, cap and counters."""
        return {
            "last_reading": asdict(self.last_reading),
            "last_readings": [asdict(r) for r in self.last_readings],
            "location": asdict(self.location),
            "cap_fraction": self.cap_fraction,
            "files_written_since_read": self.files_written_since_read,
            "n_reads": self.n_reads,
        }
