#!/usr/bin/env python3
"""T3-07 (#3025, AC-V2): MCP-reachability probe (fork-15/16 gate).

Feeds HITL gate 7 (the phase-2-entry decision in
`.praxia/docs/roadmaps/research-epics/260702_03-dag-plugin-workflows.md`): before any #2181 loop
migrates to a NO-CLAUDE (non-Claude-Code) dispatch backend, this records whether the bathos MCP
server is actually reachable from such a node. A failing probe does not break the loop (#2181's
MVP is authored on Claude-PCW) -- it only decides whether the deferred strict-mode fork (AC-P5,
#3059-adjacent T3-13) must be built, so this script always exits 0 on a completed probe; it never
gates CI.

Deliberately hand-rolls the JSON-RPC-over-stdio `initialize` handshake instead of using an MCP SDK
or any Claude Code plumbing -- that IS the point: simulating what a plain local-model dispatch node
(no Claude Code harness) would need to do itself to prove reachability.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """Result of one reachability probe. Not `xtrax.run.freshness.ProbeResult` -- that
    type's semantics are "did this probe invalidate an attestation" (TTL-attestation-specific);
    this is a standalone recording of reachability, no attestation involved.
    """

    reachable: bool
    server_info: dict | None = None
    error: str = ""


def probe_bathos_mcp(command: Sequence[str] = ("bth-mcp",), timeout: float = 5.0) -> ProbeOutcome:
    """Spawn `command`, send one newline-delimited JSON-RPC 2.0 `initialize` request, and read
    one line back. Returns `reachable=True` with the parsed `serverInfo` on a well-formed
    response; `reachable=False` with `error` set on any spawn failure, timeout, or malformed
    response.
    """
    try:
        proc = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        return ProbeOutcome(reachable=False, error=f"failed to spawn {command!r}: {exc}")

    try:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "xtrax-mcp-reachability-probe", "version": "0.1"},
            },
        }
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

        try:
            outcome = _read_one_response(proc, timeout=timeout)
        except TimeoutError as exc:
            return ProbeOutcome(reachable=False, error=str(exc))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

    return outcome


def _read_one_response(proc: subprocess.Popen, *, timeout: float) -> ProbeOutcome:
    import selectors

    assert proc.stdout is not None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    if not sel.select(timeout=timeout):
        raise TimeoutError(f"no response within {timeout}s")

    line = proc.stdout.readline()
    if not line:
        stderr = proc.stderr.read() if proc.stderr else ""
        return ProbeOutcome(reachable=False, error=f"process closed stdout; stderr={stderr!r}")

    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return ProbeOutcome(reachable=False, error=f"malformed JSON-RPC response: {exc}")

    result = payload.get("result")
    if not isinstance(result, dict) or "serverInfo" not in result:
        return ProbeOutcome(reachable=False, error=f"no serverInfo in response: {payload!r}")

    return ProbeOutcome(reachable=True, server_info=result["serverInfo"])


def main(argv: list[str] | None = None) -> int:
    del argv
    outcome = probe_bathos_mcp()
    if outcome.reachable:
        print(f"PASS: bathos MCP reachable from a NO-CLAUDE node: {outcome.server_info}")
    else:
        print(f"FAIL: bathos MCP not reachable: {outcome.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
