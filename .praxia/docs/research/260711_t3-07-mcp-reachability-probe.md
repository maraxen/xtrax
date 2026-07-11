---
task_id: 260711_t3-07-mcp-reachability-probe
date: 2026-07-11
sources:
  - "empirical probe: scripts/probe_mcp_reachability.py against bth-mcp (hand-rolled JSON-RPC-over-stdio, no MCP SDK, no Claude Code plumbing)"
  - "code recon: /home/marielle/projects/bathos/.praxia/manifest.toml ([plugin.mcp] section)"
  - "empirical check: bth --help (installed CLI subcommand list)"
  - "empirical check: ~/.claude.json's real mcpServers.bathos entry (what Claude Code actually launches)"
verification: direct subprocess probe, not inferred from docs or config
status: resolved — AC-V2 verdict recorded, feeds HITL gate 7
---

# T3-07 / AC-V2 — MCP-reachability probe

Backlog #3025. DAG: `.praxia/docs/roadmaps/research-epics/260702_03-dag-plugin-workflows.md`
(fork-15/16 gate). Feeds **HITL gate 7** (phase-2-entry decision for the cross-repo praxia-side
`rig-run` strict-dispatch items, T3-09..T3-14).

## Question (AC-V2)

From a NO-CLAUDE (non-Claude-Code) local-model dispatch node, is the bathos MCP server actually
reachable, *before* any #2181 loop migrates to a strict backend? Reachable → 15c (Claude-PCW
routing effectful actions through the bathos MCP) extends to strict nodes with no further work;
not reachable → the deferred strict-mode fork (AC-P5 / T3-13, a generic sandboxed
shell/bathos tool registry) becomes a hard prerequisite before that migration.

## Method

`scripts/probe_mcp_reachability.py::probe_bathos_mcp()` hand-rolls the MCP `initialize` handshake
over stdio — one newline-delimited JSON-RPC 2.0 request written to the child process's stdin, one
line read back with a bounded timeout — deliberately without an MCP SDK or any Claude-Code-specific
plumbing. This *is* the NO-CLAUDE scenario: any plain script or local-model dispatch node capable of
spawning a subprocess and speaking line-delimited JSON-RPC can perform the exact same check.

## Result: **reachable**

```
PASS: bathos MCP reachable from a NO-CLAUDE node: {'name': 'bathos', 'version': '3.4.2'}
```

The handshake succeeded on the first attempt, returning a well-formed `initialize` response with
`serverInfo`. Re-run via `tests/audit/test_mcp_reachability_probe.py
::test_bathos_mcp_reachable_from_a_no_claude_node` (skip-guarded on `bth-mcp` being on `PATH`, same
as the existing `praxia`-CLI-based test in `tests/composition/test_port_validation_workflow.py`).

## Verdict

**AC-V2 = PASS.** Per the DAG's own disposition rule: 15c (Claude-PCW routing through bathos MCP)
extends to NO-CLAUDE/strict nodes without modification. **AC-P5 (T3-13, the deferred strict-mode
tool registry) is NOT required** unless a future HITL gate 7 review decides to invest in phase-2
cross-repo dispatch anyway for other reasons — it is not forced by an unreachable MCP.

## Side finding (cross-repo, out of scope for this PR): bathos manifest command mismatch

While locating the real bathos MCP entrypoint to probe, found that bathos's own
`.praxia/manifest.toml` declares:
```toml
[plugin.mcp]
command = ["bth", "serve"]
```
but the installed `bth` CLI has **no `serve` subcommand at all** (confirmed via `bth --help` —
`serve` is absent from its command list). The entrypoint Claude Code actually uses (read from the
real `~/.claude.json` `mcpServers.bathos` entry, not from the manifest) is a separate console
script, `bth-mcp`, installed on `PATH` independently of the `bth` CLI. The manifest's declared
command has apparently never matched the real entrypoint — a genuine drift bug in a different
repo, structurally similar to this session's #570 finding (a documented/declared command that no
longer matches reality). Not fixed here (out of scope: bathos workspace, not xtrax); worth filing
as a debt item in bathos's own backlog in a future session.
