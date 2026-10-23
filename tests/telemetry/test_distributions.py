"""Executing distributions on the run-ledger row (debt #2947, schema v2)."""

import json

import pytest

from xtrax.provenance import DISTRIBUTION_FIELDS, UNKNOWN
from xtrax.telemetry.ledger import RunLedger, find_run, iter_rows
from xtrax.telemetry.migrate import upgrade_row
from xtrax.telemetry.record import (
    LEDGER_DISTRIBUTIONS_ENV,
    SCHEMA_VERSION,
    LedgerRecordError,
    RunLedgerRecord,
    RunProvenance,
    ledger_distributions,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)
    monkeypatch.delenv(LEDGER_DISTRIBUTIONS_ENV, raising=False)


def _entry(**overrides: str) -> "dict[str, str]":
    entry = dict.fromkeys(DISTRIBUTION_FIELDS, UNKNOWN)
    entry["name"] = "pkg"
    entry.update(overrides)
    return entry


def _v1_row() -> dict:
    row = RunLedgerRecord(
        run_id="old-run", kind="train", provenance=RunProvenance(git_sha="b" * 40)
    ).to_dict()
    row["schema_version"] = 1
    del row["provenance"]["distributions"]
    return row


# --- configuration -------------------------------------------------------------


def test_explicit_names_win_over_the_environment(monkeypatch):
    monkeypatch.setenv(LEDGER_DISTRIBUTIONS_ENV, "from-env")
    assert ledger_distributions(["jax", " numpy ", "jax", ""]) == ("jax", "numpy")
    assert ledger_distributions([]) == ()


def test_environment_list_is_read_when_no_explicit_names(monkeypatch):
    monkeypatch.setenv(LEDGER_DISTRIBUTIONS_ENV, "jax, numpy,,jax")
    assert ledger_distributions() == ("jax", "numpy")


@pytest.mark.parametrize("value", ["", "none", "NONE", " "])
def test_empty_or_none_environment_records_nothing(monkeypatch, value):
    monkeypatch.setenv(LEDGER_DISTRIBUTIONS_ENV, value)
    assert ledger_distributions() == ()


def test_unset_environment_records_nothing():
    assert ledger_distributions() == ()


def test_a_bare_string_is_refused_rather_than_split_into_letters():
    with pytest.raises(TypeError, match="not a string"):
        ledger_distributions("jax")  # type: ignore[arg-type]


# --- the field's contract --------------------------------------------------------


def test_capture_records_each_configured_distribution(tmp_path):
    prov = RunProvenance.capture(cwd=tmp_path, distributions=("jax", "xtrax-no-such-dist"))

    assert set(prov.distributions) == {"jax", "xtrax-no-such-dist"}
    assert prov.distributions["jax"]["version"] != UNKNOWN
    assert prov.distributions["xtrax-no-such-dist"]["install_kind"] == "not_installed"


def test_capture_without_names_records_an_empty_mapping(tmp_path):
    assert RunProvenance.capture(cwd=tmp_path).distributions == {}


@pytest.mark.parametrize(
    ("distributions", "match"),
    [
        ({"pkg": {"name": "pkg"}}, "exactly the fields"),
        ({"pkg": {**_entry(), "extra": "x"}}, "exactly the fields"),
        ({"pkg": {**_entry(), "version": 1}}, "not strings"),
        ({"": _entry()}, "non-empty string"),
        ({"pkg": "1.0"}, "exactly the fields"),
        (["pkg"], "must be a dict"),
    ],
)
def test_malformed_distributions_fail_closed(distributions, match):
    with pytest.raises(LedgerRecordError, match=match):
        RunProvenance(distributions=distributions)  # type: ignore[arg-type]


def test_a_row_with_a_malformed_distribution_is_refused_on_read():
    row = RunLedgerRecord(
        run_id="r", kind="train", provenance=RunProvenance(distributions={"pkg": _entry()})
    ).to_dict()
    row["provenance"]["distributions"]["pkg"]["surprise"] = "x"
    with pytest.raises(LedgerRecordError, match="exactly the fields"):
        RunLedgerRecord.from_dict(row)


def test_the_field_round_trips_through_jsonl():
    record = RunLedgerRecord(
        run_id="r",
        kind="train",
        provenance=RunProvenance(distributions={"pkg": _entry(version="1.0", vcs_commit="c")}),
    )
    restored = RunLedgerRecord.from_json_line(record.to_json_line())
    assert restored.provenance.distributions == record.provenance.distributions
    assert restored.schema_version == SCHEMA_VERSION == 2


def test_the_stored_mapping_is_a_copy_of_the_caller_input():
    source = {"pkg": _entry()}
    prov = RunProvenance(distributions=source)
    source["pkg"]["version"] = "mutated"
    assert prov.distributions["pkg"]["version"] == UNKNOWN


# --- ledger integration ----------------------------------------------------------


def test_open_records_explicit_distributions(tmp_path):
    root = tmp_path / "ledger"
    with RunLedger.open("run-d", root=root, cwd=tmp_path, distributions=["jax"]):
        pass
    row = find_run("run-d", root)
    assert row is not None
    assert row.provenance.distributions["jax"]["module"] == "jax"


def test_open_reads_the_environment_list(tmp_path, monkeypatch):
    monkeypatch.setenv(LEDGER_DISTRIBUTIONS_ENV, "jax")
    root = tmp_path / "ledger"
    with RunLedger.open("run-e", root=root, cwd=tmp_path):
        pass
    row = find_run("run-e", root)
    assert row is not None
    assert list(row.provenance.distributions) == ["jax"]


# --- schema v1 -> v2 -------------------------------------------------------------


def test_a_v1_row_upgrades_with_an_empty_mapping():
    upgraded = upgrade_row(_v1_row())
    assert upgraded["schema_version"] == 2
    assert upgraded["provenance"]["distributions"] == {}


def test_a_v1_row_that_claims_distributions_is_refused():
    row = _v1_row()
    row["provenance"]["distributions"] = {}
    with pytest.raises(LedgerRecordError, match="v1 never wrote"):
        upgrade_row(row)


def test_iter_rows_still_reads_rows_written_at_v1(tmp_path):
    """The bump must not hide existing history from find_run or compaction."""
    root = tmp_path / "ledger"
    segments = root / "segments"
    segments.mkdir(parents=True)
    (segments / "00000.jsonl").write_text(json.dumps(_v1_row()) + "\n", encoding="utf-8")

    rows = list(iter_rows(root, strict=True))

    assert [row.run_id for row in rows] == ["old-run"]
    assert rows[0].provenance.distributions == {}
    assert rows[0].schema_version == 2
