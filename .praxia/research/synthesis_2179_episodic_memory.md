---
synthesis_id: 260620_synthesis_2179_episodic_memory
synthesis_version: "1.0"
backlog_id: 2179
parent_epic: 2175
prepared_for: composition-layer contract (#2179)
task_id: 260617_xtrax-composition-mission
spec_ref: .praxia/docs/specs/260618_hmw-design-agent-identities-skills-knowl.md
---

# Research Synthesis: NLM + Transduction Episodic Memory (#2179)

**Prepared by:** composition sprint 18 (2026-06-20)  
**Use:** `prior_context` for composition sessions requiring task_id-threaded recall across agent handoffs  
**Boundary:** Praxia owns transduction MCP implementation; xtrax owns the composition-layer contract + loader tests.

---

## Executive summary

Composition sessions (OODA loops, PCW walks, port-validation waves) lose context at agent handoffs unless recall is **task_id-threaded** and **layered**. The winning brainstorm idea (spec 260618, idea C) merges:

1. **Transduction JSONL** — authoritative episodic log (recon → plan → audit → research → daily)
2. **Vector knowledge base** — API-surface recall (`knowledge.search`, `knowledge.recall`)
3. **NLM notebooks** — deep research synthesis before contemplex brainstorms

This synthesis lands a machine-readable contract at `.praxia/composition/episodic_memory_contract.toml` mirroring the capability registry pattern, with loader validation in `scripts/load_episodic_memory_contract.py`.

---

## 1. Problem frame

**Problem:** Multi-agent composition sessions (composer-orchestrator dispatching specialists, graph-auditor walking lowered graphs) need durable recall keyed by `task_id` across handoffs. Without a declared contract, agents either re-recon from scratch or hallucinate prior decisions.

**Fixed constraints:**
- xtrax core stays pure JAX; memory contracts live in `.praxia/composition/`
- Praxia MCP owns `transduction_log` / `transduction_query` tools
- `capability_registry.toml` already declares `kb_sources` per identity

**Negotiable:** NLM notebook granularity, refresh cadence, staleness thresholds.

---

## 2. Memory layers

| Layer | Authority | Query surface | Append surface | Use case |
|-------|-----------|---------------|----------------|----------|
| Transduction JSONL | **Authoritative** episodic log | `transduction_query` (scope=task\|scan) | `transduction_log` (append_recon/plan/audit/research/daily) | OODA phase boundaries, audit verdicts, recon findings |
| Vector KB | API surface / catalog | `knowledge.search`, `knowledge.recall` | `knowledge.ingest_markdown` | Library docs, skill references, stable API facts |
| NLM notebooks | Deep research synthesis | `nlm_query`, `nlm_research` | `nlm_source.add`, `nlm_research.import` | Literature review, adversarial brainstorm prep |

**Rule:** Agents with `kb_sources = ["transduction"]` (e.g. graph-auditor) must query transduction before asserting PASS. Agents with `kb_sources` including `nlm` (e.g. specification-specialist) refresh NLM at epic boundaries.

---

## 3. Transduction channel map

| Phase | JSONL path | Append action |
|-------|------------|---------------|
| recon | `.praxia/recon.jsonl` | `append_recon` |
| plan | `.praxia/plans.jsonl` | `append_plan` |
| audit | `.praxia/audits.jsonl` | `append_audit` |
| research | `.praxia/research/synthesis.jsonl` | `append_research` |
| daily | `.praxia/daily.jsonl` | `append_daily` |

Cross-phase retrieval: `transduction_query` with `scope=task` and stable `task_id` (e.g. `260617_xtrax-composition-mission`).

---

## 4. Session rules

| Rule | Value | Rationale |
|------|-------|-----------|
| `task_id_format` | `YYMMDD_<slug>` | Matches existing transduction records (e.g. `260617_xtrax-composition-mission`) |
| `handoff_path` | `.praxia/handoffs/` | Structured YAML handoffs for session pickup (handoff skill) |
| `staleness_max_days` | 30 | Reject recall from JSONL records older than threshold without explicit refresh |

**Refresh cadence:**
- **Epic boundaries** — librarian re-synthesizes NLM notebook; vector KB ingest for new specs
- **Session handoff** — write handoff YAML + `append_daily` summary before agent swap

---

## 5. NLM binding

| Field | Value |
|-------|-------|
| `notebook_id` | `2e509f42-31a5-42cd-b6b0-79a09dab6af9` (shared xtrax research notebook) |
| `tag_pattern` | `xtrax-composition` |
| `refresh_policy` | `epic_boundary_or_handoff` |

NLM is **not** the authoritative episodic log — it rots without refresh. Contract encodes refresh policy; loader tests enforce schema.

---

## 6. Integration with capability_registry

| Identity | kb_sources (registry) | Contract identity_defaults |
|----------|----------------------|---------------------------|
| composer-orchestrator | transduction, knowledge | transduction, knowledge |
| graph-auditor | transduction | transduction |

**graph-auditor** walks composition graph nodes and must cross-check `audit_verdict` slots against transduction audit records for the active `task_id`. Contract loader does not duplicate identity tables — it asserts defaults align with registry entries.

---

## 7. Pre-mortem mitigations (from spec 260618)

| Failure mode | Mitigation |
|--------------|------------|
| NLM notebook rot | `refresh_policy = epic_boundary_or_handoff`; librarian synthesis before brainstorm |
| Registry drift from MCP tool names | Contract pins `query_tool` / `append_tool` strings; loader rejects unknown channels |
| Agents skip transduction | `identity_defaults` require transduction for orchestrator + graph-auditor; OODA rules mandate `append_*` at phase boundaries |
| Vector KB never ingested | KB limited to API surface; episodic truth stays in JSONL |
| Hallucinated PASS without audit trail | graph-auditor kb_sources = transduction only; empirical pytest per node (#2178) |

---

## 8. Deliverables (this sprint)

1. `.praxia/composition/episodic_memory_contract.toml` — machine-readable contract
2. `scripts/load_episodic_memory_contract.py` — dataclasses + semver/channel validation
3. `tests/composition/test_episodic_memory_contract.py` — committed contract + invalid fixture rejection
4. `just validate-episodic-memory-contract` — CI-friendly smoke

---

## prior_context (for downstream agents)

```
Backlog #2179 lands episodic memory contract v0.1.
Transduction JSONL is authoritative; NLM is deep-research layer; vector KB is API surface.
task_id threads all phases; handoffs at .praxia/handoffs/.
composer-orchestrator + graph-auditor both require transduction kb_source.
Refresh NLM at epic boundaries and session handoffs.
Contract: .praxia/composition/episodic_memory_contract.toml
Loader: scripts/load_episodic_memory_contract.py
```
