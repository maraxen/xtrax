---
task_id: 260716_praxia-mcp-invocation-probe
date: 2026-07-16
sources:
  - "empirical probe: scripts/probe_praxia_mcp_invocation.py against praxia-mcp (hand-rolled JSON-RPC-over-stdio, no MCP SDK, no Claude Code plumbing)"
  - "code recon: ~/.claude.json's real mcpServers.praxia entry (what Claude Code actually launches)"
  - "empirical check: praxia-mcp confirmed executable on PATH"
verification: direct subprocess probe, not inferred from docs or config
status: resolved — answers the loop-controller epic kickoff doc's §5 invocation-mechanics open question
---

# Loop-controller epic — praxia MCP invocation-mechanics spike

`.praxia/docs/roadmaps/loop-controller/260716_00-mandate.md` (epic kickoff, merged PR #62) named
an explicit open question in §5: the exact mechanics a plain Python process (the future #2181
loop controller, with no Claude Code/MCP-client harness) would need to invoke
`mcp__praxia__rig_run`. This spike answers that question empirically, the same technique T3-07
(`scripts/probe_mcp_reachability.py`, #3025) already used for the analogous bathos question:
hand-roll the JSON-RPC-over-stdio protocol directly, no MCP SDK, no Claude Code plumbing.

## Question

Can a plain Python subprocess — no MCP client library, no Claude Code harness — spawn praxia's
real MCP entrypoint, complete the `initialize` handshake, and get back `rig_run`'s (and
`dispatch`'s) genuine wire-level tool schema? If yes, the pluggable `DispatchBackend` interface's
`PraxiaDispatchBackend` implementation has a concrete, proven invocation recipe to build on.

## Method

Read the live `~/.claude.json` `mcpServers.praxia` entry directly (same technique T3-07 used to
find bathos's real `bth-mcp` entrypoint, rather than trusting a manifest or guessing):

```json
{"type": "stdio", "command": "/home/marielle/.cargo/bin/praxia-mcp", "args": [],
 "env": {"DATABASE_URL": "postgresql:///praxia"}}
```

`scripts/probe_praxia_mcp_invocation.py::probe_praxia_invocation()` spawns `praxia-mcp` (resolved
via `shutil.which`, with the known path as a fallback) with `DATABASE_URL` in its environment,
sends a hand-built `initialize` request, sends the required `notifications/initialized`
notification, then sends `tools/list` — one step further than T3-07's own probe went, since this
question specifically needs the tool schema, not just reachability.

Deliberately stops there. Does **not** attempt a live `tools/call` of `rig_run` or `dispatch` —
every flow `rig_run` can dispatch (recon/reviewer/summarize/research/impl) does real,
side-effecting agent work with no evident dry-run mode; choosing a safe flow to actually invoke is
a decision for the controller's own build phase, not this reachability spike.

## Result: **reachable, with real schemas**

```
PASS: praxia MCP reachable from a plain subprocess: {'name': 'praxia', 'version': '0.1.0'}
PASS: 'rig_run' exposed with a real schema: {"type": "object", "properties": {"action": {"type": "string", "description": "Tool action selector"}, "payload": {"type": "object", "description": "Action-specific payload object", "additionalProperties": true}}}
PASS: 'dispatch' exposed with a real schema: {"type": "object", "properties": {"action": {"type": "string", "description": "Tool action selector"}, "payload": {"type": "object", "description": "Action-specific payload object", "additionalProperties": true}}}
```

The handshake succeeded on both runs (reproducibility checked directly, not assumed from one
lucky pass). `rig_run`'s wire-level schema matches exactly what this session's own `ToolSearch`
call surfaced from inside the Claude Code harness earlier — `{action: string, payload: object}` —
confirming the client-side tool description and the raw MCP wire schema agree; a plain subprocess
sees the same contract a Claude Code session does, with no hidden harness-side transformation.

## Verdict

**The invocation-mechanics open question is resolved: yes, a plain Python process can reach
`rig_run` this way.** The concrete recipe for a future `PraxiaDispatchBackend`:

1. Spawn `praxia-mcp` (via `shutil.which`, `DATABASE_URL` env var set) as a subprocess.
2. Complete the `initialize` → `notifications/initialized` handshake over stdin/stdout.
3. Call `rig_run` via a `tools/call` JSON-RPC request with `{"action": ..., "payload": {...}}` —
   the exact `action`/`payload` shape for a mutation-proposal flow specifically (as opposed to
   recon/reviewer/etc.) is a separate, not-yet-answered design question for the controller's own
   build phase, not this spike.

This does not resolve every open question in the kickoff doc's §5 — the candidate-file hand-off
mechanics and bathos call sequencing remain open, unaffected by this spike.
