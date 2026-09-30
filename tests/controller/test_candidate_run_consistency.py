"""#4584 parts 2 and 4: a CandidateRunResult is self-consistent by construction, and the
adapter turns a bathos envelope whose success flag disagrees with its exit code into a
FAILED run (logged as contract drift) rather than a "successful" non-zero exit."""

from typing import Any

import pytest

from controller.bathos_campaign_adapter import (
    BathosCampaignAdapter,
    BathosMcpToolError,
    CandidateRunResult,
)


def _envelope(**extra: Any) -> dict[str, Any]:
    return {"ok": True, "error_code": None, "error": None, "resolution_hint": None, **extra}


class _Transport:
    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.envelope


@pytest.mark.parametrize(("exit_code", "success"), [(0, True), (1, False), (127, False)])
def test_consistent_results_construct(exit_code: int, success: bool) -> None:
    assert CandidateRunResult(script_path="c.py", exit_code=exit_code, success=success)


@pytest.mark.parametrize(("exit_code", "success"), [(127, True), (0, False)])
def test_contradictory_results_are_unrepresentable(exit_code: int, success: bool) -> None:
    with pytest.raises(ValueError, match="contradicts"):
        CandidateRunResult(script_path="c.py", exit_code=exit_code, success=success)


def test_adapter_turns_success_with_nonzero_exit_into_a_failed_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = BathosCampaignAdapter(
        token="stub-token",
        transport=_Transport(_envelope(script_path="c.py", exit_code=127, success=True)),
    )
    result = adapter.run("c.py", campaign_id="camp-1")
    assert (result.success, result.exit_code) == (False, 127)
    assert "treating the run as failed (#4584)" in caplog.text


def test_adapter_rejects_failure_with_zero_exit_as_contract_drift() -> None:
    """No self-consistent reading exists, so it is refused loudly rather than guessed."""
    adapter = BathosCampaignAdapter(
        token="stub-token",
        transport=_Transport(_envelope(script_path="c.py", exit_code=0, success=False)),
    )
    with pytest.raises(BathosMcpToolError, match="contract drift"):
        adapter.run("c.py", campaign_id="camp-1")


def test_adapter_returns_a_failed_script_run_rather_than_raising() -> None:
    """Part 4: the docstring used to claim a script failure raises BathosMcpToolError."""
    adapter = BathosCampaignAdapter(
        token="stub-token",
        transport=_Transport(_envelope(script_path="c.py", exit_code=2, success=False)),
    )
    result = adapter.run("c.py", campaign_id="camp-1")
    assert (result.success, result.exit_code) == (False, 2)
