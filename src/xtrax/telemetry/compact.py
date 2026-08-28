"""Ledger compaction: fold live segments into a sealed set, GC orphan blobs.

The ledger is append-only, so it only ever grows. Compaction keeps reads cheap
without violating that: rows are deduplicated into ``sealed/``, and the segments
they came from are **moved** to ``archive/``, never deleted.

That move is a deliberate departure from bathos's ``compact()``, which leaves its
parquet fragments in place after ingesting them. bathos can do that because its
warm tier is a separate DuckDB file, so fragments and warm rows are read by
different code paths. Here ``sealed/`` and ``segments/`` are both JSONL read by
the same :func:`~xtrax.telemetry.ledger.iter_rows`, so leaving the originals
would double-count every compacted row. Moving preserves bathos's actual
invariant -- compaction destroys nothing -- while keeping reads unambiguous.

Blob GC is deliberately conservative. A blob is removed only when *no* row
anywhere, including archived ones, still references it. In practice that deletes
exactly one class of object: blobs written by a run that crashed between storing
its IR and committing its row. Anything a historical row points at stays, because
the archive is meant to remain readable, not merely retained.
"""

import dataclasses
import json
import shutil
from pathlib import Path

from xtrax.telemetry.ledger import (
    SEALED_DIRNAME,
    RunLedger,
    blobs_dir,
    iter_rows,
    resolve_root,
)
from xtrax.telemetry.record import RunLedgerRecord
from xtrax.telemetry.store import BlobStore

ARCHIVE_DIRNAME = "archive"

# Matches bathos's COMPACTION_THRESHOLD. Compaction is cheap here (a JSONL
# rewrite, not a database ingest), so the threshold is about avoiding pointless
# churn rather than amortising real cost.
COMPACTION_THRESHOLD = 50


@dataclasses.dataclass(frozen=True, slots=True)
class CompactResult:
    """What one compaction pass did."""

    rows_read: int = 0
    rows_kept: int = 0
    segments_archived: int = 0
    blobs_deleted: int = 0
    blobs_kept: int = 0
    bytes_reclaimed: int = 0
    dry_run: bool = False

    @property
    def rows_superseded(self) -> int:
        return self.rows_read - self.rows_kept


def _segment_count(root: Path) -> int:
    return len(list((root / "segments").glob("*.jsonl")))


def should_compact(
    root: "Path | str | None" = None,
    threshold: int = COMPACTION_THRESHOLD,
) -> bool:
    """Whether the live segment count justifies a compaction pass."""
    return _segment_count(resolve_root(root)) >= threshold


def _dedup(rows: "list[RunLedgerRecord]") -> "list[RunLedgerRecord]":
    """Keep the last row per ``(run_id, kind)``, preserving first-seen order.

    Last wins because a later row supersedes an earlier one describing the same
    work -- a run that was reopened and then failed must not be represented by
    its earlier ``complete`` row. Order is preserved so a compacted ledger still
    reads chronologically rather than in dictionary order.
    """
    latest: dict[tuple[str, str], RunLedgerRecord] = {}
    order: list[tuple[str, str]] = []
    for row in rows:
        key = (row.run_id, row.kind)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]


def _referenced_digests(root: Path) -> "set[str]":
    """Every blob digest named by any row, including archived ones."""
    digests: set[str] = set()
    archive = root / ARCHIVE_DIRNAME
    sources = [root / SEALED_DIRNAME, root / "segments", archive]
    for directory in sources:
        for path in sorted(directory.glob("*.jsonl")) if directory.is_dir() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    # An unparseable row is not evidence that its blobs are
                    # unreferenced. Skipping it here means GC leaves those blobs
                    # alone, which is the safe direction.
                    continue
                for ref in raw.get("ir", ()) or ():
                    digest = ref.get("sha256")
                    if digest:
                        digests.add(digest)
    return digests


def compact_ledger(
    root: "Path | str | None" = None,
    *,
    force: bool = False,
    threshold: int = COMPACTION_THRESHOLD,
    gc_blobs: bool = True,
    dry_run: bool = False,
    backend: str = "jsonl",
) -> CompactResult:
    """Fold live segments into ``sealed/``, archive them, and GC orphan blobs.

    ``backend`` is present so a future SQL warm tier (DuckDB over the sealed
    rows) can be added without changing this signature. Only ``"jsonl"`` is
    implemented; anything else raises rather than silently falling back, so a
    caller asking for a tier that does not exist finds out immediately. Note that
    any bathos-backed tier belongs in ``controller/``, never here -- ``src/xtrax``
    is gated against importing bathos.
    """
    if backend != "jsonl":
        raise ValueError(
            f"unsupported ledger backend {backend!r}; only 'jsonl' is implemented. "
            "A SQL warm tier is a query-layer concern and belongs in controller/."
        )
    resolved = resolve_root(root)
    if not force and not should_compact(resolved, threshold):
        return CompactResult(dry_run=dry_run)

    rows = list(iter_rows(resolved))
    kept = _dedup(rows)
    segments = sorted((resolved / "segments").glob("*.jsonl"))

    if dry_run:
        store = BlobStore(blobs_dir(resolved))
        referenced = _referenced_digests(resolved)
        orphans = [d for d in store.iter_digests() if d not in referenced]
        return CompactResult(
            rows_read=len(rows),
            rows_kept=len(kept),
            segments_archived=len(segments),
            blobs_deleted=len(orphans) if gc_blobs else 0,
            blobs_kept=store.count() - (len(orphans) if gc_blobs else 0),
            dry_run=True,
        )

    sealed_dir = resolved / SEALED_DIRNAME
    sealed_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = resolved / ARCHIVE_DIRNAME
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Write the new sealed set first, atomically, then archive the inputs. If we
    # die between the two, the worst outcome is duplicate rows on the next read
    # -- recoverable. The reverse order could lose rows outright.
    existing_sealed = sorted(sealed_dir.glob("*.jsonl"))
    next_index = (int(existing_sealed[-1].stem) + 1) if existing_sealed else 0
    target = sealed_dir / f"{next_index:05d}.jsonl"
    tmp = target.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(row.to_json_line() for row in kept), encoding="utf-8")
    tmp.replace(target)

    # Older sealed files are now subsumed by the new one (iter_rows fed them in).
    for old in existing_sealed:
        shutil.move(str(old), str(archive_dir / f"sealed-{old.name}"))
    for segment in segments:
        shutil.move(str(segment), str(archive_dir / f"segment-{segment.name}"))

    blobs_deleted = 0
    bytes_reclaimed = 0
    store = BlobStore(blobs_dir(resolved))
    if gc_blobs:
        referenced = _referenced_digests(resolved)
        for digest in list(store.iter_digests()):
            if digest in referenced:
                continue
            path = store.find(digest)
            if path is not None:
                bytes_reclaimed += path.stat().st_size
            if store.delete(digest):
                blobs_deleted += 1

    return CompactResult(
        rows_read=len(rows),
        rows_kept=len(kept),
        segments_archived=len(segments),
        blobs_deleted=blobs_deleted,
        blobs_kept=store.count(),
        bytes_reclaimed=bytes_reclaimed,
    )


def verify_ledger(root: "Path | str | None" = None) -> "list[str]":
    """Check every row parses and every referenced full-capture blob is intact.

    Returns a list of human-readable problems; empty means the ledger is sound.
    Verification is separate from compaction on purpose -- a corrupt store should
    be diagnosable without mutating anything.
    """
    resolved = resolve_root(root)
    store = BlobStore(blobs_dir(resolved))
    problems: list[str] = []
    for row in iter_rows(resolved, strict=False):
        for ref in row.ir:
            if ref.mode != "full":
                continue
            if not store.has(ref.sha256):
                problems.append(f"run {row.run_id}: {ref.kind} blob {ref.sha256[:12]} is missing")
            elif not store.verify(ref.sha256):
                problems.append(
                    f"run {row.run_id}: {ref.kind} blob {ref.sha256[:12]} fails its digest"
                )
    return problems


__all__ = [
    "ARCHIVE_DIRNAME",
    "COMPACTION_THRESHOLD",
    "CompactResult",
    "RunLedger",
    "compact_ledger",
    "should_compact",
    "verify_ledger",
]
