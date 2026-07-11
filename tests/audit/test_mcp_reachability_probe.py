"""Tests for the MCP-reachability probe (T3-07, #3025, AC-V2)."""

from __future__ import annotations

import shutil

import pytest

from scripts.probe_mcp_reachability import probe_bathos_mcp


def test_unreachable_command_reports_error() -> None:
    outcome = probe_bathos_mcp(command=["this-binary-does-not-exist-xyz"], timeout=1.0)
    assert outcome.reachable is False
    assert outcome.server_info is None
    assert outcome.error


def test_non_responding_process_times_out() -> None:
    outcome = probe_bathos_mcp(
        command=["python3", "-c", "import time; time.sleep(10)"], timeout=0.5
    )
    assert outcome.reachable is False
    assert "no response" in outcome.error


@pytest.mark.skipif(
    shutil.which("bth-mcp") is None, reason="bathos MCP entrypoint (bth-mcp) not installed"
)
def test_bathos_mcp_reachable_from_a_no_claude_node() -> None:
    """The actual AC-V2 certification: a hand-rolled JSON-RPC-over-stdio handshake (no MCP SDK,
    no Claude Code plumbing) against the real bathos MCP entrypoint succeeds -- see
    `.praxia/docs/research/260711_t3-07-mcp-reachability-probe.md` for the recorded verdict.
    """
    outcome = probe_bathos_mcp()
    assert outcome.reachable is True, outcome.error
    assert outcome.server_info["name"] == "bathos"
