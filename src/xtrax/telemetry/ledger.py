"""Append-only, lock-protected run ledger.

Two failure postures live here, and the asymmetry is deliberate:

*Opening* fails closed. If a row cannot be written, the run does not start.
Provenance is the one thing that cannot be captured retroactively, so a
silently-dropped ledger is exactly the outcome this module exists to prevent --
and a run that produced no record is indistinguishable, later, from a run that
never happened.

*Capturing* fails open. A failed git shellout, an un-exportable function, or an
absent cisternal degrades the row and records why. Telemetry must never be the
thing that takes down the run it is observing.

Bridging the two: no degradation is silent. ``RunLedgerRecord.__post_init__``
makes a non-``complete`` status unrepresentable without a ``status_reason``, so
"we did not capture this" is a fact in the data rather than an absence a reader
has to infer.

Locking uses ``fcntl.flock``, following cisternal's ``append_manifest`` rather
than this repo's own ``xtrax.findings.append_finding`` -- the latter appends with
a bare ``open("a")`` and no lock, which lets concurrent writers interleave
mid-line during a parallel sweep. That is a real corruption path for a
newline-delimited format, and it is the local precedent worth *not* following.
"""

import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from xtrax.telemetry.record import (
    KIND_TRAIN,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_OPTED_OUT,
    IRRef,
    LedgerRecordError,
    RunLedgerRecord,
    RunProvenance,
    SchemaVersionMismatchError,
    telemetry_opted_out,
)

try:
    import fcntl

    _HAVE_FLOCK = True
except ImportError:  # pragma: no cover - exercised only on non-POSIX platforms
    _HAVE_FLOCK = False

DEFAULT_LEDGER_ROOT = Path(".xtrax/ledger")
SEGMENTS_DIRNAME = "segments"
BLOBS_DIRNAME = "blobs"
SEALED_DIRNAME = "sealed"

# Roll to a new segment past this size. Segments are only ever appended to and
# never rewritten, so rollover exists to bound the cost of a compaction read,
# not to reclaim space.
MAX_SEGMENT_BYTES = 8 * 1024 * 1024


class LedgerUnavailableError(RuntimeError):
    """Raised when the ledger cannot be opened for writing.

    Deliberately fatal. A caller that genuinely cannot record telemetry -- a
    read-only checkout, a full disk -- must either fix that or opt out
    explicitly via XTRAX_TELEMETRY_OPTOUT, which still writes a row and marks
    the run non-citable. Proceeding silently is not one of the options.
    """


def _segments_dir(root: Path) -> Path:
    return root / SEGMENTS_DIRNAME


def blobs_dir(root: "Path | str" = DEFAULT_LEDGER_ROOT) -> Path:
    return Path(root) / BLOBS_DIRNAME


def _active_segment(root: Path) -> Path:
    """Highest-numbered segment under the size cap, or the next one.

    Only numerically-named segments are candidates for *writing*, and ordering
    is by that number rather than lexicographic. A stray file -- a manual copy,
    ``00001.bak.jsonl``, an editor's backup -- would otherwise sort last and
    make ``int(stem)`` raise, and under fail-closed semantics an uncaught
    ValueError here aborts the run rather than merely misfiling a row.

    Such a file is still *read* by ``iter_rows`` (its rows are data, and
    silently ignoring them would lose history); it simply never becomes the
    append target.
    """
    seg_dir = _segments_dir(root)
    numbered: list[tuple[int, Path]] = []
    for path in seg_dir.glob("*.jsonl"):
        if not path.is_file():
            continue
        try:
            numbered.append((int(path.stem), path))
        except ValueError:
            continue
    if not numbered:
        return seg_dir / "00001.jsonl"
    index, newest = max(numbered)
    if newest.stat().st_size < MAX_SEGMENT_BYTES:
        return newest
    return seg_dir / f"{index + 1:05d}.jsonl"


def _append_line(path: Path, line: str) -> None:
    """Append one line under an exclusive lock.

    The lock is held across the write *and* the flush, so a concurrent writer
    cannot interleave a partial line. Opening in append mode means every write
    goes to the current end of file even if another process extended it while we
    waited for the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        if _HAVE_FLOCK:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            if _HAVE_FLOCK:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _assert_writable(root: Path) -> None:
    """Prove the ledger is writable now, rather than discovering it at close.

    A run that trains for six hours and only then finds it cannot record itself
    has already lost the thing we were trying to keep.

    The probe filename carries this process's pid, and the unlink tolerates an
    already-absent file. A shared, fixed probe name is a genuine race: under a
    parallel sweep, several processes open the same ledger at once, and one
    would delete another's probe out from under it -- turning a perfectly
    writable directory into a spurious LedgerUnavailableError that aborts a run.
    Found by tests/telemetry/test_ledger.py::test_concurrent_appends_do_not_interleave.
    """
    try:
        _segments_dir(root).mkdir(parents=True, exist_ok=True)
        probe = _segments_dir(root) / f".write_probe.{os.getpid()}"
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise LedgerUnavailableError(
            f"the xtrax run ledger at {root} is not writable ({exc}). Telemetry is "
            "enforced: fix the path, point XTRAX_LEDGER_ROOT elsewhere, or set "
            "XTRAX_TELEMETRY_OPTOUT=1 to proceed with a run recorded as non-citable."
        ) from exc


def resolve_root(root: "Path | str | None" = None) -> Path:
    """Ledger root: explicit argument, then XTRAX_LEDGER_ROOT, then the default."""
    if root is not None:
        return Path(root)
    env = (os.environ.get("XTRAX_LEDGER_ROOT") or "").strip()
    return Path(env) if env else DEFAULT_LEDGER_ROOT


class RunLedger:
    """A single run's ledger entry, written once at close.

    Used as a context manager::

        with RunLedger.open(run_id, kind="train") as ledger:
            ledger.record_ir(capture_ir(...))
            ...

    Exactly one row is written per run, at close. Accumulating in memory and
    committing once keeps the row atomic: a reader never sees a half-described
    run, and the append itself is a single locked write.
    """

    def __init__(
        self,
        *,
        run_id: str,
        kind: str,
        root: Path,
        derived_from: "str | None",
        provenance: RunProvenance,
        status: str,
        status_reason: "str | None",
        context: "dict[str, str] | None" = None,
    ) -> None:
        self.run_id = run_id
        self.kind = kind
        self.root = root
        self.derived_from = derived_from
        self.provenance = provenance
        self.context: dict[str, str] = dict(context or {})
        self._status = status
        self._status_reason = status_reason
        self._ir: list[IRRef] = []
        self._closed = False

    @classmethod
    def open(
        cls,
        run_id: str,
        *,
        kind: str = KIND_TRAIN,
        root: "Path | str | None" = None,
        derived_from: "str | None" = None,
        cwd: "Path | None" = None,
        context: "dict[str, str] | None" = None,
    ) -> "RunLedger":
        """Open a ledger entry, failing closed if it cannot be written.

        Provenance is captured here, once, at run start -- never in a per-step
        hook. The git shellouts cost ~10-50 ms, which is irrelevant once per run
        and ruinous per step.

        Under XTRAX_TELEMETRY_OPTOUT the repository is left untouched (no
        pinning, no shellouts beyond the cheap ones) but a row is still written,
        marked ``opted_out`` and therefore non-citable.
        """
        resolved = resolve_root(root)
        _assert_writable(resolved)
        opted_out = telemetry_opted_out()
        provenance = RunProvenance.capture(
            run_id=run_id,
            cwd=cwd,
            pin=not opted_out,
        )
        return cls(
            run_id=run_id,
            kind=kind,
            root=resolved,
            derived_from=derived_from,
            provenance=provenance,
            context=context,
            status=STATUS_OPTED_OUT if opted_out else STATUS_COMPLETE,
            status_reason=("XTRAX_TELEMETRY_OPTOUT was set for this run" if opted_out else None),
        )

    @property
    def opted_out(self) -> bool:
        return self._status == STATUS_OPTED_OUT

    @property
    def closed(self) -> bool:
        """Whether the row has been written.

        Public so a caller managing the ledger's lifetime by hand (``Engine``,
        which owns a try/finally rather than a ``with``) can avoid a double
        close without reaching into private state.
        """
        return self._closed

    @property
    def blob_root(self) -> Path:
        return blobs_dir(self.root)

    def record_ir(self, refs: "IRRef | tuple[IRRef, ...] | list[IRRef]") -> None:
        """Attach captured IR references to this run's row."""
        if isinstance(refs, IRRef):
            self._ir.append(refs)
        else:
            self._ir.extend(refs)

    def set_status(self, status: str, reason: "str | None" = None) -> None:
        """Downgrade (or set) the run's telemetry status.

        An opt-out is never overwritten by a later degrade: the reason a row is
        non-citable should name the caller's own explicit choice, not a
        downstream symptom of it.
        """
        if self._status == STATUS_OPTED_OUT and status != STATUS_FAILED:
            return
        self._status = status
        self._status_reason = reason

    def build_record(self) -> RunLedgerRecord:
        return RunLedgerRecord(
            run_id=self.run_id,
            kind=self.kind,
            derived_from=self.derived_from,
            telemetry_status=self._status,
            status_reason=self._status_reason,
            context=dict(self.context),
            ir=tuple(self._ir),
            provenance=self.provenance,
        )

    def close(self) -> RunLedgerRecord:
        """Write this run's single row.

        NOT idempotent: a second call raises. Exactly one row per run is the
        invariant compaction's dedup and ``find_run``'s last-wins rule both rest
        on, so a silent second write would be worse than a loud refusal. Callers
        managing the lifetime by hand should use :meth:`close_if_open`, which
        handles the already-closed case and contains I/O failures.
        """
        if self._closed:
            raise LedgerRecordError(f"ledger for run_id={self.run_id!r} is already closed")
        record = self.build_record()
        _append_line(_active_segment(self.root), record.to_json_line())
        self._closed = True
        return record

    def close_if_open(self, in_flight_exc: "BaseException | None" = None) -> None:
        """Close unless already closed, containing I/O failures appropriately.

        The single place that decides what a failed *write of the record* means,
        so ``__exit__`` and a hand-rolled try/finally cannot drift apart on it:

        - No exception in flight: the failure is the only thing that went wrong,
          so it is raised as ``LedgerUnavailableError``.
        - An exception already propagating: that one is the more informative of
          the two and must not be masked by bookkeeping. The loss is downgraded
          to a warning, which still makes it visible rather than silent.
        """
        if self._closed:
            return
        try:
            self.close()
        except OSError as close_exc:
            if in_flight_exc is None:
                raise LedgerUnavailableError(
                    f"could not write the ledger row for run_id={self.run_id!r}: {close_exc}"
                ) from close_exc
            warnings.warn(
                f"xtrax: failed to write the ledger row for run_id={self.run_id!r} "
                f"while handling {type(in_flight_exc).__name__}: {close_exc}",
                UserWarning,
                stacklevel=2,
            )

    def __enter__(self) -> "RunLedger":
        return self

    def __exit__(
        self,
        _exc_type: "type[BaseException] | None",
        exc: "BaseException | None",
        _tb: "TracebackType | None",
    ) -> bool:
        """Always write a row, including when the run raised.

        A crashed run is precisely the case where the record matters most, so
        the failure is recorded and the exception is then allowed to propagate
        (returning False). Swallowing it would trade a visible training failure
        for an invisible one.
        """
        # type(exc), not exc_type: they agree at runtime, but only the former is
        # provably non-None to a type checker inside this branch.
        if exc is not None:
            self.set_status(STATUS_FAILED, f"run raised {type(exc).__name__}: {exc}")
        self.close_if_open(exc)
        return False


def iter_rows(
    root: "Path | str | None" = None,
    *,
    include_sealed: bool = True,
    strict: bool = False,
) -> "Iterator[RunLedgerRecord]":
    """Yield every ledger row, sealed rows first, then live segments.

    ``strict=False`` (the default) skips rows this code cannot parse and warns,
    so one corrupt or version-skewed line cannot make an entire history
    unreadable. ``strict=True`` re-raises, which is what a verification pass
    wants. Both are honest; neither silently drops a row without saying so.
    """
    resolved = resolve_root(root)
    sources: list[Path] = []
    if include_sealed:
        sources.extend(sorted((resolved / SEALED_DIRNAME).glob("*.jsonl")))
    sources.extend(sorted(_segments_dir(resolved).glob("*.jsonl")))
    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            if strict:
                raise
            warnings.warn(
                f"xtrax: could not read ledger segment {path}: {exc}",
                UserWarning,
                stacklevel=2,
            )
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield RunLedgerRecord.from_json_line(line)
            except (LedgerRecordError, SchemaVersionMismatchError):
                if strict:
                    raise
                warnings.warn(
                    f"xtrax: skipping unreadable ledger row {path}:{lineno} "
                    "(run `xtrax ledger compact` after upgrading, or read with "
                    "strict=True to see the error)",
                    UserWarning,
                    stacklevel=2,
                )


def find_run(run_id: str, root: "Path | str | None" = None) -> "RunLedgerRecord | None":
    """Return the most recent row for ``run_id``, or None.

    Most recent wins: a run that was reopened (a resume, a retry) is described by
    its latest row, and an earlier ``complete`` row must not out-vote a later
    ``failed`` one.
    """
    found: RunLedgerRecord | None = None
    for record in iter_rows(root):
        if record.run_id == run_id:
            found = record
    return found
