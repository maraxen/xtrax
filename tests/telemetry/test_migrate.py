"""The migration seam.

This suite is the reason the seam exists at v1 rather than being retrofitted:
``test_the_migration_chain_is_unbroken`` fails the moment SCHEMA_VERSION is
bumped without a corresponding migration, so the omission surfaces in CI rather
than at read time in someone's audit six months later.
"""

import pytest

from xtrax.telemetry.migrate import (
    MIN_SUPPORTED_VERSION,
    migration_chain_is_complete,
    read_row_upgrading,
    upgrade_row,
)
from xtrax.telemetry.record import (
    SCHEMA_VERSION,
    LedgerRecordError,
    RunLedgerRecord,
    RunProvenance,
    SchemaVersionMismatchError,
)


def _row() -> dict:
    return RunLedgerRecord(
        run_id="run-1",
        kind="train",
        provenance=RunProvenance(git_sha="a" * 40),
    ).to_dict()


def test_the_migration_chain_is_unbroken():
    """Bumping SCHEMA_VERSION without a migration must fail here, loudly.

    A schema bump with no upgrade path turns every previously written row into
    unreadable history -- the exact failure an append-only ledger exists to
    prevent.
    """
    assert migration_chain_is_complete(), (
        f"SCHEMA_VERSION is {SCHEMA_VERSION} but the migration chain from "
        f"{MIN_SUPPORTED_VERSION} is incomplete. Add the missing _migrate_vN "
        "and register it in _MIGRATIONS before bumping the version."
    )


def test_a_current_row_upgrades_to_itself():
    row = _row()
    assert upgrade_row(row) == row


def test_a_current_row_round_trips_through_the_lenient_reader():
    record = read_row_upgrading(_row())
    assert record.run_id == "run-1"


def test_a_future_row_is_refused_with_an_actionable_message():
    row = _row()
    row["schema_version"] = SCHEMA_VERSION + 5
    with pytest.raises(SchemaVersionMismatchError, match="newer than this code"):
        upgrade_row(row)


def test_a_prehistoric_row_is_refused():
    row = _row()
    row["schema_version"] = MIN_SUPPORTED_VERSION - 1
    with pytest.raises(SchemaVersionMismatchError, match="predates the oldest supported"):
        upgrade_row(row)


def test_a_non_integer_version_is_rejected():
    row = _row()
    row["schema_version"] = "1"
    with pytest.raises(LedgerRecordError, match="non-integer schema_version"):
        upgrade_row(row)


def test_a_boolean_version_is_not_treated_as_an_int():
    row = _row()
    row["schema_version"] = True
    with pytest.raises(LedgerRecordError, match="non-integer schema_version"):
        upgrade_row(row)


def test_a_non_object_row_is_rejected():
    with pytest.raises(LedgerRecordError, match="must be an object"):
        upgrade_row(["not", "a", "row"])


def test_upgrade_does_not_mutate_its_input():
    """Callers iterating a file must not have their data rewritten underneath."""
    row = _row()
    snapshot = dict(row)
    upgrade_row(row)
    assert row == snapshot
