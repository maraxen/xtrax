"""Tests for the bathos campaign-sequencing adapter (LC-06, epic #3611, AC-5).

AC-5's own measurable criterion: "the adapter's `run` wrapper accepts `derived_from` as a
first-class parameter." The flagship tests here (`test_run_threads_derived_from_via_injected_
transport` and `test_run_threads_derived_from_end_to_end_over_real_json_rpc_wire`) verify that
claim at two levels: the Python-level call boundary (an injected transport callable) and the
actual JSON-RPC wire format (a tiny stand-in `bth-mcp`-shaped subprocess), so a passing suite
means `derived_from` genuinely reaches the argument name bathos's real `run` MCP tool expects
(`derived_from`, confirmed against `bathos/src/bathos/mcp.py:1608`), not just some adapter-local
field that never makes it onto the wire.

No live bathos infrastructure is required for any test in this module.
"""

import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from controller.bathos_campaign_adapter import (
    BathosCampaignAdapter,
    BathosMcpToolError,
    BathosMcpTransportError,
    BathosTokenMissingError,
    CampaignConclusion,
    CampaignHandle,
    CandidateRunResult,
    _call_mcp_tool,
    _extract_structured_result,
    _read_local_mcp_token,
    _resolve_bth_mcp_command,
    _token_path,
)


def _ok_envelope(**extra: Any) -> dict[str, Any]:
    """A bathos `traced_tool`-shaped success envelope (mirrors
    `bathos/src/bathos/mcp.py:131-140`'s merge: mandatory keys plus tool-specific fields)."""
    return {"ok": True, "error_code": None, "error": None, "resolution_hint": None, **extra}


class _RecordingTransport:
    """A stub transport (this module's injection seam, see `BathosCampaignAdapter.__init__`'s
    `transport` parameter) that records every call and returns a pre-programmed envelope."""

    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, dict(arguments)))
        return self.envelope


# ---------------------------------------------------------------------------
# Flagship AC-5 test: derived_from threading
# ---------------------------------------------------------------------------


def test_run_threads_derived_from_via_injected_transport() -> None:
    """`run`'s `derived_from` kwarg must reach the transport's `arguments["derived_from"]`
    verbatim -- the exact field name bathos's real `run` MCP tool expects."""
    transport = _RecordingTransport(
        _ok_envelope(script_path="candidate.py", exit_code=0, success=True)
    )
    adapter = BathosCampaignAdapter(token="stub-token", transport=transport)

    result = adapter.run("candidate.py", derived_from="parent-run-uuid-123", campaign_id="camp-1")

    assert isinstance(result, CandidateRunResult)
    assert result == CandidateRunResult(script_path="candidate.py", exit_code=0, success=True)
    assert len(transport.calls) == 1
    tool_name, arguments = transport.calls[0]
    assert tool_name == "run"
    assert arguments["derived_from"] == "parent-run-uuid-123"
    assert arguments["campaign_id"] == "camp-1"
    assert arguments["script_path"] == "candidate.py"


def test_run_derived_from_defaults_to_empty_string_not_none() -> None:
    """Bathos's own `run` MCP tool signature types `derived_from: str = ""` (not `str | None`,
    `bathos/src/bathos/mcp.py:1608`) -- the adapter must match that wire type when the caller
    omits lineage, not send `None` or omit the key."""
    transport = _RecordingTransport(
        _ok_envelope(script_path="candidate.py", exit_code=0, success=True)
    )
    adapter = BathosCampaignAdapter(token="stub-token", transport=transport)

    adapter.run("candidate.py")

    _, arguments = transport.calls[0]
    assert arguments["derived_from"] == ""
    assert isinstance(arguments["derived_from"], str)


def test_run_threads_derived_from_end_to_end_over_real_json_rpc_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same claim as above, but through the REAL stdio transport (`_call_mcp_tool`) talking
    to a tiny stand-in `bth-mcp` subprocess over actual JSON-RPC-over-stdio -- proves
    `derived_from` survives serialization onto and off of the wire, not just a Python-level
    call boundary. No real bathos process is spawned.

    The fake server also logs the raw `tools/call` arguments it received to a side-channel
    file (env `FAKE_BTH_MCP_LOG`, inherited by the subprocess since `_call_mcp_tool` does not
    override `env=`) -- `CandidateRunResult` only exposes bathos's own real return contract
    (`script_path`/`exit_code`/`success`), so this is the independent check that
    `derived_from` genuinely reached the wire-level `arguments` dict, not just something the
    adapter computed and then discarded.
    """
    fake_server = _write_fake_bth_mcp(tmp_path)
    log_path = tmp_path / "received_calls.jsonl"
    monkeypatch.setenv("FAKE_BTH_MCP_LOG", str(log_path))
    adapter = BathosCampaignAdapter(command=str(fake_server), token="stub-token", timeout=10.0)

    result = adapter.run("candidate.py", derived_from="parent-run-uuid-456", campaign_id="camp-9")

    assert result == CandidateRunResult(script_path="candidate.py", exit_code=0, success=True)
    logged_calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(logged_calls) == 1
    assert logged_calls[0]["name"] == "run"
    assert logged_calls[0]["arguments"]["derived_from"] == "parent-run-uuid-456"
    assert logged_calls[0]["arguments"]["campaign_id"] == "camp-9"


# ---------------------------------------------------------------------------
# campaign_create / campaign_conclude
# ---------------------------------------------------------------------------


def test_campaign_create_returns_full_campaign_handle() -> None:
    """The MCP tool's return dict carries the real, full campaign_id (`mcp.py:1093`) -- this
    is the concrete reason MCP was chosen over CLI-subprocess (whose own free-text success
    message truncates the id to 8 chars); assert the adapter passes the full id through."""
    transport = _RecordingTransport(
        _ok_envelope(
            campaign_id="11111111-2222-3333-4444-555555555555",
            name="my-campaign",
            mode="exploration",
            status="open",
            started_at="2026-07-16T00:00:00",
        )
    )
    adapter = BathosCampaignAdapter(token="stub-token", transport=transport)

    handle = adapter.campaign_create("my-campaign", mode="exploration")

    assert handle == CampaignHandle(
        campaign_id="11111111-2222-3333-4444-555555555555",
        name="my-campaign",
        mode="exploration",
        status="open",
        started_at="2026-07-16T00:00:00",
    )
    tool_name, arguments = transport.calls[0]
    assert tool_name == "campaign_create"
    assert arguments["name"] == "my-campaign"
    assert arguments["mode"] == "exploration"


def test_campaign_create_forwards_unsupported_mode_verbatim_and_fails_loud() -> None:
    """Finding 2a: the MCP `campaign_create` tool rejects mode="sequential" (CLI supports it,
    MCP doesn't -- `mcp.py:1073-1076`). The epic's own architecture spec accepted this gap
    rather than routing around it; this adapter must forward "sequential" as given and let
    bathos's own rejection surface loudly, not silently coerce or drop it.

    The stubbed envelope below is already `traced_tool`-merged (`ok: False` plus the tool's
    own `error` string) -- a real MCP round-trip always hands the adapter that merged shape
    (`bathos/src/bathos/mcp.py:131-140`), never the bare `{"error": ...}` a bathos tool
    function returns internally before that merge runs.
    """
    transport = _RecordingTransport(
        {
            "ok": False,
            "error_code": None,
            "error": "mode must be 'exploration' or 'confirmation', got 'sequential'",
            "resolution_hint": None,
        }
    )
    adapter = BathosCampaignAdapter(token="stub-token", transport=transport)

    with pytest.raises(BathosMcpToolError) as exc_info:
        adapter.campaign_create("my-campaign", mode="sequential")

    assert "sequential" in str(exc_info.value)
    _, arguments = transport.calls[0]
    assert arguments["mode"] == "sequential"


def test_campaign_conclude_returns_conclusion() -> None:
    transport = _RecordingTransport(
        _ok_envelope(status="concluded", campaign_id="camp-1", outcome_label="success")
    )
    adapter = BathosCampaignAdapter(token="stub-token", transport=transport)

    conclusion = adapter.campaign_conclude("camp-1", "success", conclusion="all gates passed")

    assert conclusion == CampaignConclusion(
        status="concluded", campaign_id="camp-1", outcome_label="success"
    )
    tool_name, arguments = transport.calls[0]
    assert tool_name == "campaign_conclude"
    assert arguments["campaign_id"] == "camp-1"
    assert arguments["outcome_label"] == "success"
    assert arguments["conclusion"] == "all gates passed"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_tool_level_failure_raises_bathos_mcp_tool_error() -> None:
    transport = _RecordingTransport(
        {
            "ok": False,
            "error_code": "campaign_error",
            "error": "Campaign not found: bogus-id",
            "resolution_hint": "",
        }
    )
    adapter = BathosCampaignAdapter(token="stub-token", transport=transport)

    with pytest.raises(BathosMcpToolError) as exc_info:
        adapter.campaign_conclude("bogus-id", "success")

    assert exc_info.value.tool_name == "campaign_conclude"
    assert "Campaign not found: bogus-id" in str(exc_info.value)


def test_every_write_call_carries_the_resolved_token() -> None:
    """All three of campaign_create/run/campaign_conclude are write-verb MCP tools requiring
    `token=` (debt #619, `bathos/src/bathos/mcp.py:168-193`, `require_write_token`)."""
    transport = _RecordingTransport(
        _ok_envelope(campaign_id="c1", name="n", mode="exploration", status="open", started_at="t")
    )
    adapter = BathosCampaignAdapter(token="explicit-token-value", transport=transport)

    adapter.campaign_create("n")

    _, arguments = transport.calls[0]
    assert arguments["token"] == "explicit-token-value"


def test_missing_local_token_file_raises_before_any_transport_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When no explicit token is supplied AND no local token file exists, the adapter must
    fail loud before ever invoking the transport (never silently call with an empty token)."""
    missing_path = tmp_path / "does-not-exist" / "mcp_token"
    monkeypatch.setenv("BTH_MCP_TOKEN_PATH", str(missing_path))
    transport = _RecordingTransport(_ok_envelope())
    adapter = BathosCampaignAdapter(transport=transport)

    with pytest.raises(BathosTokenMissingError):
        adapter.campaign_create("n")
    assert transport.calls == []


def test_token_path_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BTH_MCP_TOKEN_PATH", str(tmp_path / "custom_token"))
    assert _token_path() == tmp_path / "custom_token"


def test_read_local_mcp_token_reads_existing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "mcp_token"
    token_file.write_text("deadbeef" * 8, encoding="utf-8")
    monkeypatch.setenv("BTH_MCP_TOKEN_PATH", str(token_file))

    assert _read_local_mcp_token() == "deadbeef" * 8


def test_read_local_mcp_token_raises_on_empty_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    token_file = tmp_path / "mcp_token"
    token_file.write_text("   \n", encoding="utf-8")
    monkeypatch.setenv("BTH_MCP_TOKEN_PATH", str(token_file))

    with pytest.raises(BathosTokenMissingError):
        _read_local_mcp_token()


# ---------------------------------------------------------------------------
# Low-level transport plumbing
# ---------------------------------------------------------------------------


def test_extract_structured_result_prefers_structured_content() -> None:
    result = {
        "structuredContent": {"ok": True, "campaign_id": "c1"},
        "content": [{"type": "text", "text": '{"ok": false}'}],
    }
    assert _extract_structured_result(result, tool_name="campaign_create") == {
        "ok": True,
        "campaign_id": "c1",
    }


def test_extract_structured_result_falls_back_to_text_content() -> None:
    result = {"content": [{"type": "text", "text": '{"ok": true, "campaign_id": "c1"}'}]}
    assert _extract_structured_result(result, tool_name="campaign_create") == {
        "ok": True,
        "campaign_id": "c1",
    }


def test_extract_structured_result_raises_when_nothing_parseable() -> None:
    result = {"content": [{"type": "text", "text": "not json"}]}
    with pytest.raises(BathosMcpTransportError):
        _extract_structured_result(result, tool_name="run")


def test_resolve_bth_mcp_command_prefers_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "controller.bathos_campaign_adapter.shutil.which",
        lambda name: "/usr/local/bin/bth-mcp" if name == "bth-mcp" else None,
    )
    assert _resolve_bth_mcp_command() == "/usr/local/bin/bth-mcp"


def test_resolve_bth_mcp_command_falls_back_when_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("controller.bathos_campaign_adapter.shutil.which", lambda name: None)
    command = _resolve_bth_mcp_command()
    assert command.endswith("bth-mcp")


def test_call_mcp_tool_raises_transport_error_when_spawn_fails() -> None:
    with pytest.raises(BathosMcpTransportError, match="failed to spawn"):
        _call_mcp_tool(
            "/nonexistent/path/definitely/not/a/real/binary",
            "run",
            {},
            timeout=2.0,
        )


# ---------------------------------------------------------------------------
# Fake bth-mcp-shaped subprocess, for the real-wire end-to-end test above.
# ---------------------------------------------------------------------------

_FAKE_BTH_MCP_SOURCE = """
import json
import os
import sys


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()


_log_path = os.environ.get("FAKE_BTH_MCP_LOG")


def _log_call(name, arguments):
    if not _log_path:
        return
    with open(_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": name, "arguments": arguments}) + "\\n")


for raw_line in sys.stdin:
    line = raw_line.strip()
    if not line:
        continue
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        _send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {"serverInfo": {"name": "fake-bth-mcp", "version": "0.0"}},
            }
        )
    elif method == "notifications/initialized":
        continue
    elif method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})
        _log_call(name, arguments)
        # Mirror bathos's real run_tool return contract (mcp.py:1043-1047) so the adapter's
        # own CandidateRunResult parsing is exercised faithfully. The independently-verified
        # side channel above (not this return payload) is what proves derived_from reached
        # the wire-level arguments dict.
        envelope = {
            "ok": True,
            "error_code": None,
            "error": None,
            "resolution_hint": None,
            "script_path": arguments.get("script_path", ""),
            "exit_code": 0,
            "success": True,
        }
        _send(
            {
                "jsonrpc": "2.0",
                "id": message["id"],
                "result": {
                    "structuredContent": envelope,
                    "content": [{"type": "text", "text": json.dumps(envelope)}],
                },
            }
        )
"""


def _write_fake_bth_mcp(tmp_path: Path) -> Path:
    script_path = tmp_path / "fake_bth_mcp.py"
    script_path.write_text(f"#!{sys.executable}\n{_FAKE_BTH_MCP_SOURCE}", encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script_path
