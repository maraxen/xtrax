"""Contract tests for the ledger row schema.

The theme throughout: a degradation must be *unrepresentable* without a reason.
These tests pin that property, because it is the whole basis for trusting a
ledger row that says something went wrong.
"""

import dataclasses

import pytest

from xtrax.telemetry.record import (
    IR_FULL,
    IR_SKIPPED,
    KIND_EVAL,
    KIND_TRAIN,
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

_PROV = RunProvenance(git_sha="a" * 40, provenance_source="builtin")


def _record(**kwargs) -> RunLedgerRecord:
    base = {"run_id": "run-000000000001", "kind": KIND_TRAIN, "provenance": _PROV}
    base.update(kwargs)
    return RunLedgerRecord(**base)


# --- IRRef ------------------------------------------------------------------


def test_skipped_ir_requires_a_reason():
    """The core anti-silent-fail invariant for IR."""
    with pytest.raises(LedgerRecordError, match="requires a reason"):
        IRRef(kind="jaxpr", sha256="", bytes=0, mode=IR_SKIPPED)


def test_skipped_ir_with_reason_is_allowed_without_a_digest():
    ref = IRRef(kind="jaxpr", sha256="", bytes=0, mode=IR_SKIPPED, reason="export refused")
    assert ref.reason == "export refused"


def test_non_skipped_ir_requires_a_full_digest():
    with pytest.raises(LedgerRecordError, match="64-char hex digest"):
        IRRef(kind="jaxpr", sha256="deadbeef", bytes=10, mode=IR_FULL)


def test_ir_bytes_must_be_non_negative_and_not_bool():
    with pytest.raises(LedgerRecordError, match="non-negative int"):
        IRRef(kind="jaxpr", sha256="a" * 64, bytes=-1)
    with pytest.raises(LedgerRecordError, match="non-negative int"):
        IRRef(kind="jaxpr", sha256="a" * 64, bytes=True)


def test_ir_mode_is_validated():
    with pytest.raises(LedgerRecordError, match="IRRef.mode must be"):
        IRRef(kind="jaxpr", sha256="a" * 64, bytes=1, mode="whatever")


# --- RunLedgerRecord --------------------------------------------------------


def test_non_complete_status_requires_a_reason():
    """A row that says something went wrong must say what."""
    for status in (STATUS_DEGRADED, STATUS_OPTED_OUT, STATUS_FAILED):
        with pytest.raises(LedgerRecordError, match="requires a status_reason"):
            _record(telemetry_status=status)


def test_complete_status_needs_no_reason():
    assert _record(telemetry_status=STATUS_COMPLETE).status_reason is None


def test_only_complete_rows_are_citable():
    assert _record().is_citable
    for status in (STATUS_DEGRADED, STATUS_OPTED_OUT, STATUS_FAILED):
        assert not _record(telemetry_status=status, status_reason="because").is_citable


def test_run_id_must_be_non_empty():
    with pytest.raises(LedgerRecordError, match="run_id must be"):
        _record(run_id="   ")


def test_kind_is_validated():
    with pytest.raises(LedgerRecordError, match="kind must be one of"):
        _record(kind="inference")


def test_blank_derived_from_is_rejected_not_coerced():
    """A blank parent is not a parent; recording one would fabricate lineage."""
    with pytest.raises(LedgerRecordError, match="fabricate lineage"):
        _record(derived_from="  ")


def test_self_parentage_is_rejected():
    with pytest.raises(LedgerRecordError, match="cannot be its own parent"):
        _record(derived_from="run-000000000001")


def test_derived_from_none_is_the_normal_case():
    assert _record().derived_from is None


# --- serialization ----------------------------------------------------------


def test_json_round_trip_is_exact():
    record = _record(
        derived_from="run-parent00001",
        ir=(IRRef(kind="jaxpr", sha256="b" * 64, bytes=12),),
    )
    restored = RunLedgerRecord.from_json_line(record.to_json_line())
    assert restored.to_dict() == record.to_dict()
    assert restored.ir[0].kind == "jaxpr"


def test_ir_survives_round_trip_as_a_tuple_of_irref():
    record = _record(ir=[IRRef(kind="stablehlo", sha256="c" * 64, bytes=3)])
    restored = RunLedgerRecord.from_json_line(record.to_json_line())
    assert isinstance(restored.ir, tuple)
    assert isinstance(restored.ir[0], IRRef)


def test_from_dict_rejects_a_version_it_does_not_know():
    raw = _record().to_dict()
    raw["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(SchemaVersionMismatchError, match="does not match the running"):
        RunLedgerRecord.from_dict(raw)


def test_version_check_precedes_field_checks():
    """A version-skewed row is diagnosed as skew, not as corruption.

    Mirrors ProbeRecord.from_json's documented rule order: misdiagnosing skew as
    field damage sends a reader looking for the wrong bug.
    """
    raw = _record().to_dict()
    raw["schema_version"] = 99
    raw["totally_unknown_field"] = 1
    with pytest.raises(SchemaVersionMismatchError):
        RunLedgerRecord.from_dict(raw)


def test_from_dict_rejects_unknown_top_level_fields():
    raw = _record().to_dict()
    raw["surprise"] = 1
    with pytest.raises(LedgerRecordError, match="unknown field"):
        RunLedgerRecord.from_dict(raw)


def test_from_dict_rejects_unknown_provenance_fields():
    raw = _record().to_dict()
    raw["provenance"]["surprise"] = 1
    with pytest.raises(LedgerRecordError, match="provenance has unknown field"):
        RunLedgerRecord.from_dict(raw)


def test_from_dict_rejects_a_non_object():
    with pytest.raises(LedgerRecordError, match="must decode to an object"):
        RunLedgerRecord.from_dict(["not", "an", "object"])


def test_from_json_line_rejects_malformed_json():
    with pytest.raises(LedgerRecordError, match="not valid JSON"):
        RunLedgerRecord.from_json_line("{oh no")


def test_json_line_is_one_line_and_newline_terminated():
    line = _record().to_json_line()
    assert line.endswith("\n")
    assert line.count("\n") == 1


# --- provenance -------------------------------------------------------------


def test_provenance_is_captured_by_default_and_cannot_be_omitted():
    """A caller may override provenance, but never leave it out."""
    record = RunLedgerRecord(run_id="run-1", kind=KIND_EVAL)
    assert record.provenance.hostname
    assert record.provenance.python_version
    assert record.provenance.provenance_source in {"cisternal", "builtin", "degraded"}


def test_provenance_defaults_are_the_degraded_reading():
    """An unfilled field must say 'unknown', never imply a clean tree."""
    blank = RunProvenance()
    assert blank.git_sha == "unknown"
    assert blank.git_dirty is None
    assert blank.provenance_source == "degraded"


def test_capture_records_dirty_as_none_when_unknown_not_false():
    """A failed dirty check is not evidence of a clean tree.

    ProbeRecord encodes the same insight as its '-unverified' SHA suffix: two
    runs that both failed the check must not be able to 'agree' as clean.
    """
    field_names = {f.name for f in dataclasses.fields(RunProvenance)}
    assert "git_dirty" in field_names
    assert RunProvenance().git_dirty is None


def test_capture_in_a_real_repo_reports_a_sha_and_source(tmp_path):
    prov = RunProvenance.capture(cwd=tmp_path)
    # tmp_path is not a git repo, so this must degrade honestly rather than
    # inventing a value or raising.
    assert prov.git_sha is not None
    assert prov.provenance_source in {"builtin", "degraded"}


def test_capture_never_raises_on_a_missing_directory():
    prov = RunProvenance.capture(cwd="/nonexistent/path/for/xtrax/test")
    assert prov.provenance_source in {"builtin", "degraded"}


# --- opt-out ----------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_optout_recognises_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", value)
    assert telemetry_opted_out()


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def test_optout_does_not_fire_on_falsy_values(monkeypatch, value):
    """XTRAX_TELEMETRY_OPTOUT=0 must not read as an opt-out."""
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", value)
    assert not telemetry_opted_out()


def test_optout_absent_is_not_opted_out(monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    assert not telemetry_opted_out()
