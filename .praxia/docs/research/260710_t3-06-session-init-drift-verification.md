---
task_id: 260710_t3-06-session-init-drift-verification
date: 2026-07-10
sources:
  - "praxia code recon: /home/marielle/projects/praxia crates/praxia-workflows/src/session.rs, registry.rs, plugin_cli.rs"
  - "praxia code recon: crates/praxia-cli/src/commands.rs (CLI subcommand wiring)"
  - "praxia code recon: agent_assets/hooks/session-start.sh (SessionStart hook)"
  - "empirical check: ~/.praxia/plugins.toml (global registry, current state)"
verification: direct source read + grep across the full praxia repo for every call site
status: resolved — binary answer recorded, T3-03 disposition flipped
---

# T3-06 / AC-V1 — session-init drift coverage verification

Backlog #3024. Spec: `.praxia/docs/specs/260702_xtrax-workflows-as-a-praxia-plugin-packa.md`
(ORTHOGONAL-3, open Q3). DAG: `.praxia/docs/roadmaps/research-epics/260702_03-dag-plugin-workflows.md`.

## Question (AC-V1)

From an xtrax cwd with no project-local `plugins.toml`, does praxia's `SessionContext::init`
drift-check — which re-exports a plugin's workflow templates when their content hash changes —
fire automatically via the **global** `~/.praxia/plugins.toml` registry path?

This is a gating precondition, not informational: the answer decides whether T3-03 (D3, the
xtrax-side `drift --check` CI gate) is merely belt-and-suspenders or strictly mandatory.

## Method

Read the praxia repo directly (checked out at `/home/marielle/projects/praxia`) rather than
inferring from docs:

1. Confirmed the hash the dirty-check compares against is workflow-template-aware.
2. Found every call site of the function that runs the dirty-check.
3. Traced each call site back to what actually invokes it during normal usage.
4. Checked the SessionStart hook (the one thing that runs on every Claude Code session) for any
   wiring into this mechanism.
5. Checked the live global registry for xtrax's current registration state.

## Evidence

- **The hash mechanism does cover workflow templates.** `compute_composite_hash`
  (`crates/praxia-workflows/src/registry.rs:46-120`) includes every
  `manifest.plugin.workflows[].template_path` file's bytes in the composite hash it compares
  against the stored value — confirmed by its own dedicated test,
  `test_compute_composite_hash_workflow_template_sensitivity` (same file). So *if* the drift-check
  ran, it would correctly detect a workflow-template edit.
- **But `SessionContext::init` — the function that actually runs the dirty-check/auto-heal loop —
  has exactly one call site in the entire praxia codebase:** `crates/praxia-workflows/src/session.rs:80`
  is called only from `crates/praxia-workflows/src/plugin_cli.rs:182`, inside `plugin_cli::export()`.
  Grep across every `.rs` file in the repo (excluding tests) turned up no other call site.
- **`plugin_cli::export()` is wired only to the explicit CLI subcommand** `praxia plugin export`
  (`crates/praxia-cli/src/commands.rs:5217`), alongside `plugin_cli::install()`
  (`commands.rs:5209`) and `plugin_cli::uninstall()` (`commands.rs:5213`). None of these run
  automatically — a human or agent has to explicitly type `praxia plugin export` (or
  install/uninstall).
- **It is not called by the MCP server's `ensure_initialized()`.** That function (12 occurrences
  across `praxia-agent-server`, `praxia-mcp`, `praxia-mcp-tools`, `praxia-cli`,
  `praxia-rig-tools`) is a separate, unrelated initializer (workspace/backlog DB state) — none of
  its implementations touch `SessionContext` or plugin export.
- **It is not called by the `session-start.sh` hook** (`agent_assets/hooks/session-start.sh`,
  praxia hook #1, the one hook that actually fires on every Claude Code session start). Read in
  full: it only logs telemetry (`session_log.jsonl`), appends a telemetry event via
  `praxia transduction log`, and restores handoff/compaction context. No plugin or export logic
  anywhere in it.
- **Empirically, xtrax is not even registered yet.** `~/.praxia/plugins.toml` currently has
  entries for `bathos`, `contemplex`, `jaxlint`, `maraxiom`, `myxcel`, `praxia`, `xperiri` — no
  `[plugins.xtrax]` block at all (that registration is T3-01's job, out of scope here). So today
  there is nothing to drift-check for xtrax regardless of the mechanism's behavior.

## Answer

**No — the free drift-check does not fire automatically from an ordinary xtrax session.** The
mechanism is correctly workflow-template-aware when invoked, but it is invoked only by an
explicit, easy-to-forget `praxia plugin export` (or `install`/`uninstall`) CLI call. Nothing in
the session lifecycle — not the SessionStart hook, not MCP tool initialization, not any other code
path — triggers it.

## Consequence

Per the spec's own stated rule ("Fires → D3 (T3-03) is belt-and-suspenders; does not fire → D3 is
MANDATORY and the spec flips it"): **T3-03 (D3, the xtrax-side `drift --check` gate) is now
MANDATORY, not belt-and-suspenders.** This finding also directly evidences the PM-5 pre-mortem
scenario in the spec ("the drift-check's global-registry path... turned out NOT to fire from xtrax
cwd") — the risk PM-5 anticipated is confirmed real, not merely hypothetical.

T3-03 itself (backlog TBD per the DAG doc) remains out of scope for this session — it is still
`blocked:true`, depending on T3-01 (packaging manifest) and T3-06 (this item). This document
exists to unblock it with the required disposition answer.
