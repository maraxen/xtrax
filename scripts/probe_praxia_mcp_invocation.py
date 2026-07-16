#!/usr/bin/env python3
"""Praxia MCP invocation-mechanics spike (loop-controller epic, #2181 loop-controller gap).

Feeds the "exact praxia CLI/MCP invocation mechanics from a plain Python process" open question
named in `.praxia/docs/roadmaps/loop-controller/260716_00-mandate.md` §5: the future #2181 loop
controller has no Claude Code/MCP-client harness, so it needs its own way to reach
`mcp__praxia__rig_run`. This answers that empirically, the same way T3-07
(`scripts/probe_mcp_reachability.py`, #3025) already answered the identical question for bathos:
hand-roll the JSON-RPC-over-stdio protocol directly, no MCP SDK, no Claude Code plumbing -- that
IS the point, since it simulates exactly what a plain dispatch node would need to do itself.

This is a NEW script, not an extension of `probe_mcp_reachability.py` -- that file's docstring and
test suite are specifically scoped to bathos/T3-07/AC-V2/HITL-gate-7; this targets a different
server for a different, loop-controller-epic-specific question. It deliberately mirrors that
script's hand-rolled JSON-RPC pattern by design (cited, not silently duplicated).

Goes one step further than T3-07 did -- past `initialize` through to `tools/list` -- to confirm
`rig_run` and `dispatch` are genuinely exposed with real schemas from a plain subprocess, not just
that the server accepts a handshake. Deliberately does NOT attempt an actual `tools/call` of
`rig_run`: every flow it can dispatch (recon/reviewer/summarize/research/impl) does real,
side-effecting agent work with no evident dry-run mode -- actually invoking one is a real action
requiring its own deliberate choice of a safe/cheap flow, which belongs to the controller's own
build phase, not a reachability spike. Matches T3-07's own precedent of stopping well short of a
live tool call.

Always exits 0 on a completed probe (a negative finding is itself useful information, matching
T3-07's own stance) -- this is a standalone investigation script, not wired into CI.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

# The live ~/.claude.json mcpServers.praxia entry's command, read directly rather than guessed
# (same technique T3-07 used to find bathos's real bth-mcp entrypoint). Used only as a fallback if
# `praxia-mcp` isn't resolvable via PATH at probe time.
_FALLBACK_COMMAND = "/home/marielle/.cargo/bin/praxia-mcp"

_WATCHED_TOOLS = ("rig_run", "dispatch")


@dataclass(frozen=True, slots=True)
class InvocationProbeOutcome:
    """Result of the invocation-mechanics spike."""

    reachable: bool
    server_info: dict | None = None
    tool_schemas: dict[str, dict] = field(default_factory=dict)
    error: str = ""


def _resolve_command() -> str:
    return shutil.which("praxia-mcp") or _FALLBACK_COMMAND


def _send(proc: subprocess.Popen, payload: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()


def _read_one_response(proc: subprocess.Popen, *, timeout: float) -> dict:
    assert proc.stdout is not None
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    if not sel.select(timeout=timeout):
        raise TimeoutError(f"no response within {timeout}s")

    line = proc.stdout.readline()
    if not line:
        stderr = proc.stderr.read() if proc.stderr else ""
        msg = f"process closed stdout; stderr={stderr!r}"
        raise RuntimeError(msg)

    return json.loads(line)


def probe_praxia_invocation(timeout: float = 10.0) -> InvocationProbeOutcome:
    """Spawn praxia's real MCP entrypoint, complete the initialize handshake, then list tools.

    Reports whether `rig_run`/`dispatch` are genuinely exposed with real wire-level schemas to a
    plain, harness-less client -- not the client-side tool descriptions a Claude Code session
    already sees, which come from a different code path.
    """
    command = _resolve_command()
    env = os.environ.copy()
    env.setdefault("DATABASE_URL", "postgresql:///praxia")

    try:
        proc = subprocess.Popen(
            [command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
    except OSError as exc:
        return InvocationProbeOutcome(reachable=False, error=f"failed to spawn {command!r}: {exc}")

    try:
        _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "xtrax-praxia-invocation-probe", "version": "0.1"},
                },
            },
        )
        try:
            init_response = _read_one_response(proc, timeout=timeout)
        except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            return InvocationProbeOutcome(reachable=False, error=f"initialize failed: {exc}")

        result = init_response.get("result")
        if not isinstance(result, dict) or "serverInfo" not in result:
            return InvocationProbeOutcome(
                reachable=False, error=f"no serverInfo in initialize response: {init_response!r}"
            )
        server_info = result["serverInfo"]

        # MCP requires the initialized notification before tools/list is valid.
        _send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})

        _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        try:
            list_response = _read_one_response(proc, timeout=timeout)
        except (TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            return InvocationProbeOutcome(
                reachable=True,
                server_info=server_info,
                error=f"reachable, but tools/list failed: {exc}",
            )

        tools = (list_response.get("result") or {}).get("tools", [])
        schemas = {
            tool["name"]: tool.get("inputSchema", {})
            for tool in tools
            if tool.get("name") in _WATCHED_TOOLS
        }
        return InvocationProbeOutcome(reachable=True, server_info=server_info, tool_schemas=schemas)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: list[str] | None = None) -> int:
    del argv
    outcome = probe_praxia_invocation()

    if not outcome.reachable:
        print(
            f"FAIL: praxia MCP not reachable from a plain subprocess: {outcome.error}",
            file=sys.stderr,
        )
        return 0

    print(f"PASS: praxia MCP reachable from a plain subprocess: {outcome.server_info}")

    missing = [name for name in _WATCHED_TOOLS if name not in outcome.tool_schemas]
    if missing:
        print(
            f"PARTIAL: reachable, but tools/list did not expose {missing}"
            + (f" ({outcome.error})" if outcome.error else ""),
            file=sys.stderr,
        )
        return 0

    for name in _WATCHED_TOOLS:
        print(
            f"PASS: '{name}' exposed with a real schema: {json.dumps(outcome.tool_schemas[name])}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
