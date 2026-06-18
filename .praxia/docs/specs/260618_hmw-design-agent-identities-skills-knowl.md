---
session_id: b759c73c
topic: HMW: Design agent identities, skills, knowledge bases, and evolvable tooling for JAX/xtrax composition authoring
task_type: architectural
winner: A+D merge: Capability registry TOML (.praxia/composition/capability_registry.toml) with semver agent_identity→skills→mcp_tool_profile→kb_sources, plus graph-native node metadata (MathJax, citations, script_usage, audit_verdict, bathos_sidecar_ref) validated by graph-auditor before JIT lowering; specialist roster (B) encoded as registry entries not skill prose
created_at: 2026-06-18T04:26:31.542391+00:00
---

# Brainstorm: HMW: Design agent identities, skills, knowledge bases, and evolvable tooling for JAX/xtrax composition authoring

## Problem Frame
Fixed: xtrax core stays pure JAX; composition layer is separate; must integrate Praxia MCP (backlog, transduction, knowledge, NLM, dispatch); skills already exist (using-xtrax, orchestration, exporting-jax); bathos binds to nodes not globals; no UI logic in compiler. Negotiable: agent roster granularity, skill packaging (monolith vs per-domain), KB ingestion (NotebookLM vs vector KB), tool evolution (versioned tool profiles vs codegen), validation protocol (agentic audit on graph vs CI-only). Frame confirmed — proceed to divergence.

## Idea Pool
- [ai] A) Capability registry TOML (.praxia/composition/capability_registry.toml): agent_identity → skills[] → mcp_tool_profile → kb_sources[] with semver for evolvable tooling
- [ai] B) Composition-specialist roster dispatched per chain-map node type: composer-orchestrator, jax-purity-reviewer, host-prep-fixer, export-bundle-inferrer, graph-auditor — each binds using-xtrax + orchestration skills
- [ai] C) NLM notebooks + transduction task_id as episodic memory; librarian synthesizes research before contemplex; vector KB for API surface only
- [ai] D) Graph-native metadata on composition nodes (MathJax, citations, script_usage, audit_verdict, bathos_sidecar_ref); validation agents walk graph before JIT lowering
- [user] Competing approaches surfaced: (A) declarative capability registry vs (B) hardcoded composition roster in orchestration skill; (C) NLM-first episodic memory vs vector-KB-first; (D) graph-native node metadata vs post-hoc audit scripts. Trade-off axis: flexibility/evolvability vs operational simplicity. Probing complete — ready for convergence after steelman of runner-up (B hardcoded roster).

## Decision Log

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Pre-mortem: Six months later the capability registry diverged from actual MCP tool names after a Praxia upgrade; graph-auditor agents hallucinated PASS on nodes missing bathos refs; NLM notebooks rotted while vector KB was never ingested; authors ignored registry and used raw orchestration skill anyway. Mitigations: registry discovery via tool_profile_info, empirical graph validation tests (failing pytest per node contract), librarian refresh cadence on epic boundaries, ship v0.1 with 5 identities only.
**AI:** _not recorded_

## Acceptance Criteria
**Given** Fixed: xtrax core stays pure JAX; composition layer is separate; must integrate Praxia MCP (backlog, transduction, knowledge, NLM, dispatch); skills already exist (using-xtrax, orchestration, exporting-jax); bathos binds to nodes not globals; no UI logic in compiler. Negotiable: agent roster granularity, skill packaging (monolith vs per-domain), KB ingestion (NotebookLM vs vector KB), tool evolution (versioned tool profiles vs codegen), validation protocol (agentic audit on graph vs CI-only). Frame confirmed — proceed to divergence.
**When** implementing A+D merge: Capability registry TOML (.praxia/composition/capability_registry.toml) with semver agent_identity→skills→mcp_tool_profile→kb_sources, plus graph-native node metadata (MathJax, citations, script_usage, audit_verdict, bathos_sidecar_ref) validated by graph-auditor before JIT lowering; specialist roster (B) encoded as registry entries not skill prose
**Then**
  - [ ] _add specific measurable criteria_
