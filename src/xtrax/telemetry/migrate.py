"""Forward migration of ledger rows across schema versions.

``RunLedgerRecord.from_dict`` fails closed on any version other than the running
one. That strictness is deliberate -- a verdict computed under one schema is not
guaranteed valid under another -- but on its own it would make the ledger's
central promise (an append-only history that outlives the code that wrote it)
self-defeating: the first schema change would render every prior row unreadable.

This module is the escape valve, and it exists at v1 *before* it is needed. The
alternative -- adding migrations once drift has already happened -- means the
first schema bump lands with no upgrade path and a directory of rows nobody can
open. bathos carries 14+ numbered migrations for exactly this reason; the cost of
the seam is one dict, and the cost of omitting it is a corpus of dead history.

Contract for adding a migration, when SCHEMA_VERSION goes from N to N+1:

  1. Write ``def _migrate_vN(row: dict) -> dict`` that takes a row already at
     version N and returns one at N+1, setting ``schema_version`` itself.
  2. Register it in ``_MIGRATIONS[N]``.
  3. Bump ``SCHEMA_VERSION`` in ``record.py``.

``tests/telemetry/test_migrate.py`` enforces that the chain is unbroken from 1 to
SCHEMA_VERSION, so a bump without a migration fails CI rather than at read time
in someone's audit six months later.
"""

from collections.abc import Callable
from typing import Any

from xtrax.telemetry.record import (
    SCHEMA_VERSION,
    LedgerRecordError,
    RunLedgerRecord,
    SchemaVersionMismatchError,
)

# Keyed by the version a migration upgrades *from*. Empty at v1: there is no
# earlier released schema, and inventing a v0 would fabricate history.
_MIGRATIONS: "dict[int, Callable[[dict[str, Any]], dict[str, Any]]]" = {}

# Oldest version this code can still upgrade from.
MIN_SUPPORTED_VERSION = 1


def migration_chain_is_complete() -> bool:
    """Whether every step from MIN_SUPPORTED_VERSION to SCHEMA_VERSION exists."""
    return all(version in _MIGRATIONS for version in range(MIN_SUPPORTED_VERSION, SCHEMA_VERSION))


def upgrade_row(raw: Any) -> "dict[str, Any]":  # noqa: ANN401
    """Upgrade one decoded row to the running schema version.

    Raises :class:`SchemaVersionMismatchError` when no path exists -- a row from
    the future, or one older than ``MIN_SUPPORTED_VERSION``. Refusing is the
    honest outcome: a partially-understood row read as if it were current is
    worse than one that is plainly unreadable.

    ``raw`` is ``Any`` for the same reason as ``RunLedgerRecord.from_dict``: it
    is a decoded JSON value of unproven shape, and the guard below is what
    establishes otherwise.
    """
    if not isinstance(raw, dict):
        raise LedgerRecordError(f"ledger row must be an object, got {type(raw).__name__}")
    version = raw.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise LedgerRecordError(f"ledger row has a non-integer schema_version {version!r}")
    if version > SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"ledger row schema_version={version} is newer than this code's "
            f"{SCHEMA_VERSION}; upgrade xtrax to read it"
        )
    if version < MIN_SUPPORTED_VERSION:
        raise SchemaVersionMismatchError(
            f"ledger row schema_version={version} predates the oldest supported "
            f"version {MIN_SUPPORTED_VERSION}"
        )
    row = dict(raw)
    while row["schema_version"] < SCHEMA_VERSION:
        current = row["schema_version"]
        migration = _MIGRATIONS.get(current)
        if migration is None:
            raise SchemaVersionMismatchError(
                f"no migration registered from schema_version={current} to "
                f"{current + 1}; the migration chain is incomplete"
            )
        row = migration(row)
        if row.get("schema_version") != current + 1:
            raise SchemaVersionMismatchError(
                f"migration from schema_version={current} did not advance the "
                f"version (got {row.get('schema_version')!r})"
            )
    return row


def read_row_upgrading(raw: "dict[str, Any]") -> RunLedgerRecord:
    """Upgrade then construct -- the lenient counterpart to ``from_dict``."""
    return RunLedgerRecord.from_dict(upgrade_row(raw))
