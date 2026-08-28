"""xtrax.telemetry -- enforced run provenance, IR capture, and an append-only ledger.

Every training and inference run that goes through xtrax writes exactly one
schema-versioned ledger row carrying (a) reconstructable git provenance, (b) an
environment fingerprint, and (c) content-addressed jaxpr/StableHLO. A run that
cannot write that row does not start.

The motivating measurement, from cisternal PR #32: of 345 catalogued bathos runs,
``git_hash`` was populated 345/345, yet only 40.6% still resolved to a commit and
92.2% ran on a dirty tree. Recording a SHA is not the same as being able to
reconstruct what ran -- which is why this package captures the *executed IR* and,
where cisternal is installed, pins it to a durable git ref.

Layout::

    record.py    RunLedgerRecord / RunProvenance / IRRef -- the schema
    store.py     BlobStore -- content-addressed, compressed IR blobs
    ir.py        capture_ir() -- jaxpr + StableHLO at the compile boundary
    ledger.py    RunLedger -- flock'd append-only segments; opens fail-closed
    callback.py  TelemetryCallback -- the Engine 7-hook seam
    compact.py   compact_ledger() -- seal, archive, GC orphan blobs
    migrate.py   forward migration across schema versions

Import-time discipline: this package is stdlib-only apart from
``xtrax.profiling.record`` (a leaf, jax-free at module scope). ``jax`` is
imported lazily inside :mod:`xtrax.telemetry.ir`, and ``cisternal`` lazily inside
:mod:`xtrax.telemetry.record`, so the ledger stays importable for CI gates and
compaction in environments that have neither. It must never import ``bathos``
(LC-02/AC-1b), ``xtrax.devtools``, or ``xtrax.eda``.
"""

from xtrax.telemetry.callback import TelemetryCallback
from xtrax.telemetry.compact import (
    COMPACTION_THRESHOLD,
    CompactResult,
    compact_ledger,
    should_compact,
    verify_ledger,
)
from xtrax.telemetry.ir import (
    DEFAULT_MAX_IR_BYTES,
    IR_KIND_JAXPR,
    IR_KIND_OPTIMIZED_HLO,
    IR_KIND_STABLEHLO,
    IRCaptureMode,
    capture_ir,
    degraded_reason,
    resolve_capture_mode,
)
from xtrax.telemetry.ledger import (
    DEFAULT_LEDGER_ROOT,
    LedgerUnavailableError,
    RunLedger,
    blobs_dir,
    find_run,
    iter_rows,
    resolve_root,
)
from xtrax.telemetry.migrate import migration_chain_is_complete, upgrade_row
from xtrax.telemetry.record import (
    IR_FULL,
    IR_HASH_ONLY,
    IR_SKIPPED,
    KIND_EVAL,
    KIND_EXPORT,
    KIND_TRAIN,
    PROVENANCE_BUILTIN,
    PROVENANCE_CISTERNAL,
    PROVENANCE_DEGRADED,
    SCHEMA_VERSION,
    STATUS_COMPLETE,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_OPTED_OUT,
    IRRef,
    LedgerRecordError,
    RunLedgerRecord,
    RunProvenance,
    SchemaVersionMismatchError,
    telemetry_opted_out,
)
from xtrax.telemetry.store import BlobStore, BlobStoreError, digest_of

__all__ = [
    "COMPACTION_THRESHOLD",
    "DEFAULT_LEDGER_ROOT",
    "DEFAULT_MAX_IR_BYTES",
    "IR_FULL",
    "IR_HASH_ONLY",
    "IR_KIND_JAXPR",
    "IR_KIND_OPTIMIZED_HLO",
    "IR_KIND_STABLEHLO",
    "IR_SKIPPED",
    "KIND_EVAL",
    "KIND_EXPORT",
    "KIND_TRAIN",
    "PROVENANCE_BUILTIN",
    "PROVENANCE_CISTERNAL",
    "PROVENANCE_DEGRADED",
    "SCHEMA_VERSION",
    "STATUS_COMPLETE",
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "STATUS_OPTED_OUT",
    "BlobStore",
    "BlobStoreError",
    "CompactResult",
    "IRCaptureMode",
    "IRRef",
    "LedgerRecordError",
    "LedgerUnavailableError",
    "RunLedger",
    "RunLedgerRecord",
    "RunProvenance",
    "SchemaVersionMismatchError",
    "TelemetryCallback",
    "blobs_dir",
    "capture_ir",
    "compact_ledger",
    "degraded_reason",
    "digest_of",
    "find_run",
    "iter_rows",
    "migration_chain_is_complete",
    "resolve_capture_mode",
    "resolve_root",
    "should_compact",
    "telemetry_opted_out",
    "upgrade_row",
    "verify_ledger",
]
