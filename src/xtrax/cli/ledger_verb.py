"""``xtrax ledger``: inspect, verify, and compact the run ledger.

Four actions, each answering a question an auditor actually asks:

  ``list``     what ran, when, and is it citable?
  ``show``     everything recorded about one run, including its IR digests.
  ``verify``   do the artifacts the rows point at still exist and still hash?
  ``compact``  fold segments into a sealed set and reclaim orphaned blobs.

``compact`` archives the segments it reads rather than deleting them, so the
operation is non-destructive by construction; ``--dry-run`` reports what it
would do without touching anything.

This verb is declared ``ledger_admin`` in audit/telemetry_coverage.toml: it
operates on the ledger itself, so recording its own execution would be circular.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrax.telemetry.compact import compact_ledger, verify_ledger
from xtrax.telemetry.ledger import blobs_dir, find_run, iter_rows, resolve_root
from xtrax.telemetry.store import BlobStore

_ACTIONS = ("list", "show", "verify", "compact")


@dataclass
class LedgerArgs:
    """Arguments for the ``ledger`` verb.

    Args:
        action: One of list, show, verify, compact.
        run_id: Run to describe (required by ``show``).
        root: Ledger root; defaults to XTRAX_LEDGER_ROOT or .xtrax/ledger.
        dry_run: For ``compact``, report without changing anything.
        force: For ``compact``, run even below the segment threshold.
        limit: For ``list``, maximum rows to print (0 means all).
    """

    action: str = "list"
    run_id: "str | None" = None
    root: "str | None" = None
    dry_run: bool = False
    force: bool = False
    limit: int = 50


def _fmt_row(row: Any) -> str:
    flag = " " if row.is_citable else "!"
    ir = ",".join(f"{ref.kind}:{ref.mode}" for ref in row.ir) or "-"
    parent = f" <- {row.derived_from}" if row.derived_from else ""
    return (
        f"{flag} {row.ts}  {row.run_id}  {row.kind:<6} "
        f"{row.telemetry_status:<10} {row.provenance.git_sha[:12]:<12} {ir}{parent}"
    )


def run_ledger(args: LedgerArgs) -> None:
    """Execute the ledger verb."""
    if args.action not in _ACTIONS:
        print(
            f"error: unknown action {args.action!r}; expected one of {', '.join(_ACTIONS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    root = resolve_root(Path(args.root) if args.root else None)

    if args.action == "list":
        rows = list(iter_rows(root))
        if args.limit > 0:
            rows = rows[-args.limit :]
        if not rows:
            print(f"no ledger rows under {root}")
            return
        print(f"{'':1} {'timestamp':<26} {'run_id':<20} {'kind':<6} {'status':<10} sha  ir")
        for row in rows:
            print(_fmt_row(row))
        # '!' marks a row that may not be cited; say so rather than leaving a
        # bare glyph for the reader to decode.
        if any(not r.is_citable for r in rows):
            print("\n! = not citable (degraded, opted out, or failed)")
        return

    if args.action == "show":
        if not args.run_id:
            print("error: --run-id is required for 'show'", file=sys.stderr)
            raise SystemExit(2)
        row = find_run(args.run_id, root)
        if row is None:
            print(f"no ledger row for run_id {args.run_id!r} under {root}", file=sys.stderr)
            raise SystemExit(1)
        print(f"run_id          {row.run_id}")
        print(f"kind            {row.kind}")
        print(f"timestamp       {row.ts}")
        print(f"status          {row.telemetry_status}")
        if row.status_reason:
            print(f"status_reason   {row.status_reason}")
        print(f"citable         {row.is_citable}")
        print(f"derived_from    {row.derived_from or '-'}")
        prov = row.provenance
        print("provenance:")
        for field in (
            "provenance_source",
            "git_sha",
            "git_branch",
            "git_dirty",
            "dirty_content_id",
            "pinned_sha",
            "run_ref",
            "remote_url",
            "submodule_state",
            "jax_version",
            "jaxlib_version",
            "x64_enabled",
            "device_kind",
            "hostname",
            "python_version",
        ):
            print(f"  {field:<18} {getattr(prov, field)}")
        if row.context:
            print("context:")
            for key in sorted(row.context):
                print(f"  {key:<18} {row.context[key]}")
        print("ir:")
        store = BlobStore(blobs_dir(root))
        for ref in row.ir:
            present = "present" if store.has(ref.sha256) else "MISSING"
            suffix = f"  ({ref.reason})" if ref.reason else ""
            print(
                f"  {ref.kind:<14} {ref.mode:<10} {ref.bytes:>10,}B  "
                f"{ref.sha256[:16]}  {present}{suffix}"
            )
        return

    if args.action == "verify":
        problems = verify_ledger(root)
        if not problems:
            print(f"PASS: ledger at {root} is intact")
            return
        print(f"FAIL: {len(problems)} problem(s) in the ledger at {root}", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(1)

    result = compact_ledger(root, force=args.force, dry_run=args.dry_run)
    prefix = "would " if result.dry_run else ""
    print(
        f"{prefix}compact: read {result.rows_read} rows, kept {result.rows_kept} "
        f"({result.rows_superseded} superseded), archived {result.segments_archived} "
        f"segment(s), {prefix}deleted {result.blobs_deleted} orphan blob(s) "
        f"({result.bytes_reclaimed:,}B), {result.blobs_kept} blob(s) retained"
    )
    if result.rows_read == 0 and not result.dry_run:
        print("(below the compaction threshold; pass --force to compact anyway)")


__all__ = ["LedgerArgs", "run_ledger"]
