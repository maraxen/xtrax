"""Claim-time refusal -- the tier that survives an opt-out."""

import pytest

from xtrax.telemetry.claims import (
    UnrecordedRunError,
    assert_run_citable,
    filter_citable,
    run_is_citable,
)
from xtrax.telemetry.ledger import RunLedger


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT", raising=False)
    monkeypatch.delenv("XTRAX_LEDGER_ROOT", raising=False)
    return tmp_path / "ledger"


def test_a_complete_run_is_citable(root):
    with RunLedger.open("run-good", root=root):
        pass
    assert run_is_citable("run-good", root)
    assert assert_run_citable("run-good", root).run_id == "run-good"


def test_assert_returns_the_record_so_provenance_is_to_hand(root):
    with RunLedger.open("run-good", root=root):
        pass
    record = assert_run_citable("run-good", root)
    assert record.provenance.git_sha


def test_an_unknown_run_is_refused(root):
    assert not run_is_citable("run-nope", root)
    with pytest.raises(UnrecordedRunError, match="has no ledger row"):
        assert_run_citable("run-nope", root)


def test_the_unknown_run_error_says_what_to_do(root):
    with pytest.raises(UnrecordedRunError) as excinfo:
        assert_run_citable("run-nope", root)
    assert "XTRAX_LEDGER_ROOT" in str(excinfo.value)


def test_a_failed_run_is_refused(root):
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-crash", root=root):
            raise RuntimeError("training diverged")
    assert not run_is_citable("run-crash", root)
    with pytest.raises(UnrecordedRunError, match="not citable"):
        assert_run_citable("run-crash", root)


def test_the_refusal_quotes_the_recorded_reason(root):
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-crash", root=root):
            raise RuntimeError("training diverged")
    with pytest.raises(UnrecordedRunError) as excinfo:
        assert_run_citable("run-crash", root)
    assert "training diverged" in str(excinfo.value)


def test_an_opted_out_run_is_refused(root, monkeypatch):
    """Opting out of capture is not a route to a citable result."""
    monkeypatch.setenv("XTRAX_TELEMETRY_OPTOUT", "1")
    with RunLedger.open("run-opt", root=root):
        pass
    monkeypatch.delenv("XTRAX_TELEMETRY_OPTOUT")
    assert not run_is_citable("run-opt", root)
    with pytest.raises(UnrecordedRunError, match="opted_out"):
        assert_run_citable("run-opt", root)


def test_a_degraded_run_is_refused(root):
    with RunLedger.open("run-degraded", root=root) as ledger:
        ledger.set_status("degraded", "IR capture incomplete -- jaxpr: export refused")
    with pytest.raises(UnrecordedRunError, match="degraded"):
        assert_run_citable("run-degraded", root)


def test_the_latest_row_decides(root):
    """A later failure must out-vote an earlier success."""
    with RunLedger.open("run-1", root=root):
        pass
    assert run_is_citable("run-1", root)
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-1", root=root):
            raise RuntimeError("retry failed")
    assert not run_is_citable("run-1", root)


# --- batch ------------------------------------------------------------------


def test_filter_citable_splits_and_explains(root):
    with RunLedger.open("run-ok-1", root=root):
        pass
    with RunLedger.open("run-ok-2", root=root):
        pass
    with pytest.raises(RuntimeError):
        with RunLedger.open("run-bad", root=root):
            raise RuntimeError("nope")

    citable, rejected = filter_citable(
        ["run-ok-1", "run-bad", "run-ok-2", "run-missing"], root
    )
    assert citable == ["run-ok-1", "run-ok-2"]
    assert set(rejected) == {"run-bad", "run-missing"}
    # Every rejection carries a reason: silently dropping runs would reintroduce
    # exactly the invisible-gap failure this subsystem exists to prevent.
    assert all(reason for reason in rejected.values())


def test_filter_citable_on_an_empty_list(root):
    assert filter_citable([], root) == ([], {})


def test_filter_citable_preserves_order(root):
    for i in range(3):
        with RunLedger.open(f"run-{i}", root=root):
            pass
    citable, _ = filter_citable(["run-2", "run-0", "run-1"], root)
    assert citable == ["run-2", "run-0", "run-1"]
