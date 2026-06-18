---
session_id: 073d0395
topic: HMW: Design unified implementation validation pipeline for xtrax (jax-port + audit-fw integration, MVP v0.1)
task_type: architectural
epic_backlog_id: 2180
depends_on: [1573]
depends_on_soft: [1577]
blocks: [2181]
winner: Faction C + in-place translation hybrid: port/ owns oracle (port/reference/), parity tests (port/tests/), emit (port/emit/port_emit.py), bridge stub (port/bridge/composition_map.toml optional until Phase 2), port_target.toml (ad_critical T4 flag). Code translates directly to src/xtrax/ (no port/jax_port/). 6-phase port_validation PCW (P0-ORACLE→P1-SPEC→P1.5-TOPO→P2-STATIC→P3-PARITY→P4-EMIT→P5-ROUTE). Hook-extended subagent-stop with tier_verdict payload. Emit stub-first parallel to N1.1. port/ dev-only extra excluded from wheel.
created_at: 2026-06-18T18:24:25.937648+00:00
invest_pass: true
acceptance_criteria_count: 12
adversarial_review: NEEDS_WORK→reconciled 2026-06-18
design_path: .praxia/docs/designs/260618_2180_port_validation_design.md
---

# Spec: Implementation Validation Pipeline (Epic #2180)

**Backlog:** [#2180](backlog) — Implementation validation pipeline (jax-port + audit-fw integration)  
**Depends on:** [#1573](backlog) audit framework (spec 260611)  
**Soft dependency:** [#1577](backlog) emit envelope N1.1 — stub-first parallel track; delegation optional at runtime  
**Blocks:** [#2181](backlog) autoresearch (immutable evaluator + oracle gates required first)  
**MVP scope:** Phase 1 — **leaf numeric kernels without stochasticity** in `src/xtrax/` (no stateful PRNG, no dynamic-shape padding until v0.2); Phase 2 (#2174) and Phase 3 deferred

## Emit integration (#1577 envelope mapping)

Port records are a **domain extension** of the N1.1 audit emit envelope, not a forked schema.

| Audit-fw field | Port record mapping |
|----------------|---------------------|
| `dim` | `"port"` (stored as `domain: "port"` in JSONL; **hash anchor uses `dim`**) |
| `symbol_qualname` | same |
| `rule_id` | `port_parity_tier` (e.g. `tier_3`) or `jaxlint` for P2-STATIC |
| `finding_id` | `hash(dim + symbol_qualname + rule_id + tolerance_policy)` per #1573 Q4; **`tolerance_policy` is an N1.1 port-domain amendment** (prevents false tombstones on policy change) |
| `label` | `bug` when pytest FAIL on deterministic tier; `observation` for WARN-only static findings |
| `severity` | from `routing.toml` `domain=port` rows (proposed in #2180-c05 PR; interim: `major` on tier FAIL, `minor` on static WARN) |
| `track` | `deterministic` always for port_validation gates |

**Precedence vs dimension audits:** `port_validation` owns pre-merge port-wave gates on symbols in the active `port/manifests/<wave_id>.toml`. Post-merge repo-wide dimension sweeps (#1582 Performance, #1582 JAX-purity) may re-emit only when no `domain=port` record exists for the same `symbol_qualname + rule_id` within the same `task_id` window.

**N1.1 dual validation:** `port_emit.py` feature-detects `audit.emit` serializer when importable; CI `audit-port-emit-contract` runs dual validation (local sketch + shared N1.1 envelope subset) when `audit.emit` is importable. Stub schema fields MUST be a strict subset of the N1.1 envelope. **#1577 is not a hard DAG blocker** — emit stub ships with local contract test; delegation is opportunistic.

**N1.1 amendment (port domain):** Port records extend the base `finding_id = hash(dim + symbol_qualname + rule_id)` with `tolerance_policy` as a fourth hash input. Tombstone dedup and routing use this amended id; dimension audits unchanged.

## jax-port skill deviation (xtrax canonical)

| jax-port skill default | xtrax #2180 variant |
|------------------------|---------------------|
| `port/jax_port/` staging subtree | **Rejected** — in-place `src/xtrax/` (AC-2) |
| Delete `reference/` after parity green | **Rejected** — `port/reference/` is permanently sealed |
| Phases 1–8 full lifecycle | PCW maps phases; MVP v0.1 gates P0–P5 only; PRNG/dynamic-shape/JIT-boundary phases deferred unless kernel is stochastic (out of MVP scope) |

Update `using-jax` / jax-port skill with an **xtrax in-place variant** checklist in backlog tech-debt item (see Decision Log).

## Problem Frame
**Fixed constraints (cannot change):**
- Two-track architecture: agents → deterministic artifacts; CI → same scripts, no LLM
- Reference vendoring as mathematical oracle (Phase 0 gate blocks all downstream)
- Graded parity blocking sequence T1→T2→T3→(T4)→T5
- Immutable evaluator separation for any agent loop (#2181 dependency)
- Extend audit-fw (#1573), don't replace 8 dimensions
- #2181 autoresearch blocked until #2180 MVP ships
- New `port_validation` PCW template (not refactor_with_audit / spec_driven_dev)
- Emit via audits.jsonl with domain="port" (#1577 integration)

**Negotiable (brainstorm scope):**
- Repo layout: `port/reference/` + `port/jax_port/` vs `.praxia/composition/` graph-native paths — need canonical choice or lowering mapping
- T4 in v0.1: per-port-target AD-criticality gate vs blanket defer
- PCW phase granularity: 5 vs 6 phases; hook surface (subagent-stop tier verdicts vs supervisor-only)
- Emit contract timing: sketch now vs wait for N1.1; `port_emit.py` ownership
- Phase 2 entry criteria: all src/xtrax/ green vs only referenced kernels
- port/ subtree: wheel inclusion vs dev-only extra (#1451)
- Paper port info isolation enforcement in PCW

**Frame confirmed.** Ready for divergence on competing architectural factions.

## Idea Pool
- [user] **Faction A — Port-centric monorepo slice:** Canonical tree under `port/` mirroring jax-port skill: `port/reference/` (sealed oracle), `port/jax_port/` (WIP translation), `port/tests/test_parity_*.py`, `port/emit/port_emit.py`. Composition graph paths are derived views via lowering adapter at Phase 2 boundary.
- [user] **Faction B — Graph-native first:** Oracle + parity live under `.praxia/composition/port_nodes/` keyed by capability_registry node IDs. `port/` is a thin symlink/alias layer for jax-port skill compatibility. PCW walks graph nodes, not file topo.
- [user] **Faction C — Dual-root with explicit bridge:** `port/` owns oracle+parity artifacts (ground truth); `.praxia/composition/` owns orchestration metadata. Bridge module `port/bridge/composition_map.toml` maps qualname→node_id. Phase 1 ignores composition; Phase 2 requires bridge completeness for referenced kernels only.
- [user] **Faction D — Audit-dimension overlay:** No separate port tree — parity tests colocated in `tests/parity/src/xtrax/` beside existing dimension tests. Oracle vendored under `tests/fixtures/reference/`. PCW `port_validation` is orchestration-only; artifacts live in standard test layout.
- [user] Probe me on trade-offs — I have positions on layout, emit, hooks, and MVP tier scope.
- [ai] Faction A — Port-centric monorepo slice: canonical `port/reference/` (sealed oracle), `port/jax_port/` (WIP), `port/tests/test_parity_*.py`, `port/emit/port_emit.py`. Composition paths are derived via lowering adapter at Phase 2.
- [ai] Faction C — Dual-root bridge: `port/` owns oracle+parity ground truth; `.praxia/composition/` owns orchestration metadata. `port/bridge/composition_map.toml` maps qualname→node_id. Phase 2 entry requires bridge completeness for referenced kernels only (not all src/xtrax/).
- [user] **SCAMPER Substitute probe on layout:** What if we Substitute `port/jax_port/` with in-place translation directly into `src/xtrax/` (no staging subtree) — oracle stays in `port/reference/`, parity tests in `port/tests/` import from `src/xtrax/`?
- [user] **PEGS probe on emit:** If we gate emit on N1.1 landing first, what's the minimum viable stub — JSONL append-only with schema validation deferred to contract tests, or stdout-only tier verdicts until serializer ships?
- [user] **SCAMPER Combine probe on PCW:** Combine P2-STATIC and P3-PARITY into single "VERIFY" phase with jaxlint as first sub-gate inside parity runner — reduces FSM hops but blurs static vs runtime failure taxonomy in emit.
- [user] **PEGS on hooks:** Extend `subagent-stop` hook payload with `{tier_verdict, port_parity_tier, oracle_id}` vs supervisor-only aggregation — which prevents agent eval hacks better given immutable judge requirement?
- [user] **MVP tier scope probe:** T4 deferred by default but enabled via `port_target.toml` flag `ad_critical: true` per port wave — does this satisfy both MVP speed and training-loss port needs without blanket T4 in v0.1?
- [ai] In-place translation model: no `port/jax_port/` staging — translate directly into `src/xtrax/`. Oracle sealed in `port/reference/`; parity tests in `port/tests/` import from `src/xtrax/`. Reduces subtree duplication; jax-port skill Phase 3 targets production module paths from day one.
- [ai] Emit stub-first strategy: ship `port_emit.py` now as thin wrapper appending domain=port records to audits.jsonl with pydantic/dataclass validation against sketch schema. Contract tests land with N1.1 but port gates don't block on N1.1 merge — parallel tracks.
- [ai] Per-port T4 gate via `port/port_target.toml` flag `ad_critical: true`. Default false (MVP T1+T2+T3+T5). Training-loss ports opt in; EDA/planner ports skip T4 in v0.1. PCW P3-PARITY reads flag to insert T4 between T3 and T5.
- [user] **Answers to probes:**
- [user] **Substitute (in-place translation):** Strong candidate. Eliminates `port/jax_port/` staging drift — translated code lives in `src/xtrax/` from first commit; `port/reference/` is read-only oracle; `port/tests/` is the only port-specific test home. Skill jax-port Phase 3 already targets production paths. Risk: premature promotion before parity green — mitigated by blocking merge on tier gates, not directory location.
- [user] **Emit stub-first:** Prefer parallel track — `port/emit/port_emit.py` writes domain=port JSONL now using sketch schema + local contract test in `tests/contract/test_port_emit_schema.py`. N1.1 hardens shared serializer; port_emit delegates to it when available, falls back to local impl. Ownership: port_validation PCW P4-EMIT phase, fixer maintains wrapper.
- [user] **Combine STATIC+PARITY:** Reject for v0.1 — emit `error_taxonomy_class` needs distinct `compilation_leak` (jaxlint) vs `numeric_drift` (parity). Keep P2-STATIC and P3-PARITY separate; jaxlint runs as pre-flight inside P3 only if P2 already green (no combine).
- [user] **Hooks — subagent-stop tier payload:** Prefer hook extension over supervisor-only. Payload `{tier_verdict, port_parity_tier, oracle_id, pytest_nodeid}` on parity subagent completion enables immutable judge to verify stdout hash matches hook claim. Supervisor aggregates but doesn't originate verdicts. Reduces "tool called but output ignored" pre-mortem failure.
- [user] **T4 per-port flag:** Yes — `port/port_target.toml` with `ad_critical: bool` default false. PCW reads at P0-ORACLE lock-in. Satisfies MVP defer + training-loss needs.
- [user] **New faction on PCW structure:** 6-phase FSM P0-ORACLE → P1-SPEC → P2-STATIC → P3-PARITY → P4-EMIT → P5-ROUTE with P1.5-TOPO (recon produces topo-sorted port wave manifest) inserted between SPEC and STATIC. Topo manifest is artifact, not gate.
- [user] **Paper info isolation:** PCW P1-SPEC enforces mask — specification-specialist context excludes paper results tables; baseline I/O pairs generated from oracle reference only, not author-reported numbers.
- [ai] Hook-extended subagent-stop: parity subagents emit `{tier_verdict, port_parity_tier, oracle_id, pytest_nodeid}` on completion. Immutable judge verifies stdout hash matches hook claim. Supervisor aggregates, does not originate verdicts.
- [ai] 6-phase PCW FSM: P0-ORACLE → P1-SPEC → P1.5-TOPO(manifest) → P2-STATIC → P3-PARITY → P4-EMIT → P5-ROUTE. P1.5 produces topo-sorted port wave manifest (artifact, not gate). P2/P3 remain separate for distinct error_taxonomy in emit.
- [ai] Paper info isolation in P1-SPEC: specification-specialist context excludes paper results tables. Baseline I/O pairs generated from vendored oracle only — never from author-reported numbers. Prevents oracle gap from weak paper specs.
- [ai] port/ dev-only extra (#1451): `port/reference/` and `port/tests/` excluded from wheel; only `src/xtrax/` ships. `pyproject.toml` optional `[dev]` extra includes port validation deps. CI runs port gates on dev install.
- [user] **SCAMPER Eliminate probe:** Can we Eliminate P5-ROUTE as separate PCW phase — fold CC5 routing into P4-EMIT via port_emit.py calling routing.toml directly?
- [user] **PEGS on #1577 integration:** Port emit records share finding_id hash strategy with dimension audits — `hash(port + qualname + port_parity_tier + rule_id)`. Tombstone dedup means re-running T3 on same kernel suppresses duplicate backlog nodes. Does this create false tombstones when tolerance_policy changes between runs?
- [user] **SCAMPER Reverse on Phase 2 entry:** Reverse the dependency — composition integration tests run first as smoke, leaf parity runs only for failing nodes? (Deliberately bad idea to stress-test Phase 1 gate.)
- [user] **Eliminate answer:** Keep P5-ROUTE separate — routing decisions (block-CI vs backlog-node vs found-issues) need supervisor judgment on severity×track matrix; port_emit writes record, supervisor+P5 decides destination. Combining risks auto-routing major findings to tombstone.
- [user] **Tombstone probe answer:** Include `tolerance_policy` in finding_id hash inputs — policy change = new finding_id, no false tombstone.
- [user] **Reverse stress-test:** Confirms Phase 1 gate is non-negotiable — integration cannot isolate leaf math bugs. Reject reverse; Phase 2 entry = referenced kernels green only (Faction C bridge model).
- [user] **Self-debug routing:** NLM suggests `agentic_self_debug` routing action — align with existing fixer dispatch budget via port-repair sub-identity capped at 2 cycles per tier FAIL before human_escalation_reason.
- [user] converge

## Decision Log
- [REJECT] Faction B — Graph-native first (.praxia/composition/port_nodes/): Rejected — couples port validation to composition graph before Phase 1 core parity proven. Topo sort at file level is simpler and matches jax-port skill. Graph-native paths deferred to Phase 2 integration via bridge, not as canonical oracle home.
- [REJECT] Faction D — Audit-dimension overlay (tests/parity/ colocated): Rejected — blurs port lifecycle FSM with repo-wide dimension audits. Oracle vendoring and sealed reference subtree need isolation from tests/fixtures/. port/ subtree provides clear agent sandbox boundary per immutable evaluator requirement.
- [REJECT] Combine P2-STATIC and P3-PARITY into single VERIFY phase: Rejected for v0.1 — emit error_taxonomy_class requires distinct compilation_leak (jaxlint) vs numeric_drift (parity). Separate phases preserve CC5 routing fidelity.
- [DEFER] T4 blanket defer to v0.2: Deferred as default but not absolute — per-port ad_critical flag in port_target.toml enables T4 opt-in for training-loss ports without requiring T4 for all ports in v0.1.
- [REJECT] Emit gated on N1.1 landing: Rejected — port_emit.py stub ships in v0.1 with local contract test; parallel track to N1.1. Port gates must not block on N1.1 merge timeline.
- [REJECT] Supervisor-only tier verdicts (no hook extension): Rejected — subagent-stop hook payload with tier_verdict + pytest_nodeid enables immutable judge stdout hash verification. Prevents tool-called-but-output-ignored pre-mortem failure mode.
- [ACCEPT] Faction C + in-place translation hybrid (winner): port/ owns oracle, parity tests, emit, optional bridge stub; code translates in-place to src/xtrax/ without port/jax_port/ staging. Balances Faction A Phase 1 simplicity with Phase 2 composition bridge. Bridge optional until first #2174 integration port.
- [ACCEPT] 6-phase port_validation PCW (P0-ORACLE→P1-SPEC→P1.5-TOPO→P2-STATIC→P3-PARITY→P4-EMIT→P5-ROUTE): P1.5-TOPO produces versioned topo manifest artifact (not gate). P2/P3 remain separate for distinct error_taxonomy_class in emit (compilation_leak vs numeric_drift). P5-ROUTE separate from P4-EMIT for supervisor severity×track judgment.
- [ACCEPT] Hook-extended subagent-stop with tier_verdict payload: Parity subagents emit {tier_verdict, port_parity_tier, oracle_id, pytest_nodeid}. PCW walker fails dispatch on hook PASS + pytest non-zero or stdout hash mismatch. Immutable judge enforcement per pre-mortem mitigation #5.
- [ACCEPT] Emit stub-first port_emit.py parallel to N1.1 #1577: port/emit/port_emit.py appends domain=port records to audits.jsonl now; delegates to shared serializer when N1.1 lands; local contract test in tests/contract/. finding_id hash includes tolerance_policy to prevent false tombstones.
- [ACCEPT] port/ dev-only extra excluded from wheel (#1451): Only src/xtrax/ ships in wheel. port/reference/ and port/tests/ require [dev] extra. CI dev-install job runs just audit-port on PRs touching src/xtrax/ or port/ per pre-mortem mitigation #7.
- [ACCEPT] T4 per-port opt-in via port_target.toml ad_critical flag (default false): MVP ships T1+T2+T3+T5 by default. ad_critical=true inserts T4 between T3 and T5; requires justification field; T4 pytest marker enforces timeout budget. Prevents blanket T4 CI cost and conftest hack workarounds.
- [ACCEPT] Phase 2 entry: referenced kernels green only (not all src/xtrax/): Faction C bridge model: composition_map.toml maps qualname→node_id; Phase 2 requires bridge completeness for referenced kernels only. Confirms Phase 1 gate non-negotiable per reverse stress-test.
- [ACCEPT] Paper info isolation in P1-SPEC: specification-specialist context excludes paper results tables; baseline I/O pairs generated from vendored oracle only. Prevents oracle gap from weak author-reported numbers.
- [ACCEPT] port-repair self-debug capped at 2 cycles per tier FAIL: agentic_self_debug routing action aligned with fixer dispatch budget; human_escalation_reason after 2 failed repair cycles per tier. Prevents infinite agent loops on numeric drift.

## Assumptions

- Audit-fw foundation gates (N0: import-linter, `__future__` ratchet, jaxlint) remain green; port validation extends, not replaces, dimension audits.
- First port wave targets at least one topo-sorted leaf kernel in `src/xtrax/` with a vendorable reference implementation.
- `capability_registry.toml` semver bump to 0.2.0 is acceptable for adding `reference-vendor`, `specification-specialist`, `test-designer` roles.
- N1.1 (#1577) shared emit serializer will eventually subsume `port_emit.py` local impl; stub-first parallel track is interim-safe.
- Composition graph integration (#2174) does not begin until referenced kernels pass graded parity (Phase 2 entry criteria).
- jax-port skill Phase 3 in-place translation convention aligns with production `src/xtrax/` paths from first commit.

## TBDs

- **Capability registry hook schema:** Normative draft in Appendix A below; formalize in registry 0.2.0 PR ([#2180.6]).
- **P1.5 manifest format:** File path, hash algorithm, and freshness check vs `task_id` — recon to propose `port/manifests/<wave_id>.toml` shape.
- **Bridge lint CI job:** Whether `composition_map.toml` qualname validation runs on every PR or only when file is non-empty (Phase 2+).
- **Hypothesis metamorphic fuzz:** Deferred post-first-port-wave; revisit in v0.2 backlog item.
- **Paper port PDF pipeline (DR-3):** Literature decomposition agents out of MVP v0.1 scope; separate epic candidate.
- **routing.toml CC5 `domain=port` rows:** Who owns severity×track matrix extension — port_validation PCW or audit-fw maintainer?

## Pre-mortem Record
**User:** **6-month pre-mortem (failure scenario):** It is December 2026. Epic #2180 shipped MVP v0.1 but the port_validation pipeline is effectively dead — no port wave has reached T5 green in CI for 4 months. Root causes we hypothesize:

1. **Bridge manifest drift:** `port/bridge/composition_map.toml` was scaffolded but never maintained; Phase 2 slipped while Phase 1 ports renamed `src/xtrax/` qualnames without updating the bridge. Integration tests passed on stale mappings; real boundary bugs surfaced only in production composition graphs (#2174).

2. **Emit stub rot:** `port_emit.py` shipped stub-first parallel to N1.1, but N1.1 landed with a different schema shape. Port records in audits.jsonl became second-class citizens — contract tests only covered stub schema; routing.toml CC5 matrix ignored `domain=port` severity rules. Tier FAILs stopped creating backlog nodes.

3. **Topo vs graph divergence:** P1.5 topo manifest (file call-graph) diverged from composition node dependencies for dynamically composed pipelines. Teams manually curated port waves per kernel but skipped P1.5 artifact updates; parity green on leaves did not imply integration readiness.

4. **T4 flag confusion:** `port_target.toml` `ad_critical` defaults were copy-pasted true for convenience on non-AD ports; finite-diff sweeps blocked CI for 45+ minutes per kernel. Teams disabled T4 globally via conftest hacks (the exact pre-mortem failure mode we tried to prevent).

5. **Hook tier_verdict ignored:** subagent-stop payloads carried tier_verdict PASS but pytest exit code was non-zero; supervisor aggregated hook claims without stdout hash verification. Immutable judge requirement was documented but not enforced in PCW walker.

6. **Oracle seal breach:** Agents gained write access to `port/reference/` via overly broad fixer dispatch scopes; reference subtree was "fixed" to match wrong JAX translation, destroying mathematical ground truth.

7. **Dev-extra install gap:** `port/` dev-only extra meant release CI (wheel-only install) never ran port gates; regressions merged because `just audit-port` was not wired into default CI matrix.

**Mitigations to bake into spec:**
- Bridge optional until Phase 2; CI lint that composition_map.toml qualnames ⊆ live `src/xtrax/` symbols when file exists
- port_emit delegates to N1.1 serializer when present; weekly contract test against shared schema; finding_id includes tolerance_policy
- P1.5 manifest versioned artifact with hash; Phase 2 entry gate checks manifest freshness ≤ port wave task_id
- ad_critical requires explicit justification field in port_target.toml; T4 timeout budget in pytest marker
- PCW walker: FAIL dispatch if hook tier_verdict PASS but pytest_nodeid exit ≠ 0 or stdout hash mismatch
- reference-vendor role has exclusive write to port/reference/; fixer read-only; supervisor red-line on reference diff
- CI matrix: dev extra install job runs `just audit-port` on every PR touching src/xtrax/ or port/

Record these as Pre-mortem Record in final spec. Ready for INVEST gate.
**AI:** _not recorded_

## Acceptance Criteria

Atomic Given-When-Then criteria for MVP v0.1 (Phase 1 core leaf parity).

### AC-1: Sealed oracle vendoring (P0-ORACLE)

**Given** a port wave with `port/port_target.toml` and no vendored reference under `port/reference/`  
**When** the `reference-vendor` agent completes P0-ORACLE  
**Then** `port/reference/` contains a `# REFERENCE: DO NOT MODIFY` sealed subtree, baseline I/O pairs, and `oracle_id` with content hash; fixer dispatch scopes are read-only on `port/reference/`

### AC-2: In-place translation target (no staging subtree)

**Given** a green P0-ORACLE gate  
**When** the port fixer translates reference code  
**Then** production modules land in `src/xtrax/` (no `port/jax_port/` staging); `port/tests/test_parity_*.py` import from `src/xtrax/`

### AC-3: Graded parity blocking sequence (T1→T2→T3→T5)

**Given** `port_target.toml` with `ad_critical = false` (default)  
**When** `pytest port/tests/` runs the parity harness for the first topo-sorted kernel  
**Then** T1 (dtype/shape), T2 (float64), T3 (float32), and T5 (JIT invariance) execute in blocking order; each tier must PASS before the next runs; T4 is skipped

### AC-4: T4 opt-in for AD-critical ports

**Given** `port_target.toml` with `ad_critical = true` and non-empty `ad_critical_justification`  
**When** P3-PARITY runs for that port wave  
**Then** T4 (gradient parity) inserts between T3 and T5; pytest `@pytest.mark.timeout` enforces per-tier budget; FAIL surfaces `error_taxonomy_class: nan_gradient_vulnerability` when applicable

### AC-5: Static and trace-count gates (P2-STATIC)

**Given** a P1.5 topo manifest artifact exists for the port wave  
**When** P2-STATIC executes before P3-PARITY  
**Then** jaxlint reports zero blocking violations on ported `src/xtrax/` modules and `chex.assert_max_traces` passes; failures emit `error_taxonomy_class: compilation_leak`

### AC-6: Port domain emit records (P4-EMIT)

**Given** a parity tier completes with PASS or FAIL  
**When** `port/emit/port_emit.py` appends to `.praxia/audits.jsonl`  
**Then** each record has `domain: "port"`, `port_parity_tier`, `oracle_id`, `tier_verdict` (status, tolerance_policy, error_taxonomy_class, max_discrepancy), and `finding_id` hash includes `tolerance_policy`; `tests/contract/test_port_emit_schema.py` validates schema locally

### AC-7: Hook tier_verdict immutable judge (subagent-stop)

**Given** a parity subagent completes with `subagent-stop` payload `{tier_verdict, port_parity_tier, oracle_id, pytest_nodeid}`  
**When** the PCW walker evaluates the hook claim  
**Then** dispatch FAILs if `tier_verdict: PASS` but pytest exit code ≠ 0 or stdout hash mismatches hook claim; supervisor aggregates verdicts but does not originate them

### AC-8: CI two-track mechanical gate (`just audit-port`)

**Given** a PR touches `src/xtrax/` or `port/`  
**When** CI runs the dev-extra install job  
**Then** `just audit-port` executes the same scripts as local (no LLM); job passes only when oracle seal, parity tiers, jaxlint, trace-count, and emit contract tests all green; `port/` is excluded from wheel build; merge is blocked unless `just audit-port` passes for kernels listed in the PR's active manifest (`port/manifests/<wave_id>.toml` resolved from `port/port_target.toml` `wave_id`) diff vs `main`

### AC-9: N1.1 envelope compatibility (emit integration)

**Given** `audit.emit` from #1577 is importable in the dev environment  
**When** `port_emit.py` appends a port tier record and CI runs `audit-port-emit-contract`  
**Then** the record validates against both `tests/contract/test_port_emit_schema.py` (local sketch) and the shared N1.1 envelope contract test; stub fields are a strict subset of N1.1; when `audit.emit` is absent, local contract test alone suffices

### AC-10: P1-SPEC paper-info isolation + math/pseudocode

**Given** P1-SPEC runs for a port wave  
**When** the specification-specialist agent produces baseline I/O, docstrings, and a **math/pseudocode appendix** tying oracle steps to jaxtyping contracts  
**Then** agent context excludes paper results tables; baseline I/O pairs are generated from vendored oracle only; a `paper_mask_enforced: true` flag is recorded in the port wave manifest; pseudocode references oracle function names, not author-reported numbers

### AC-11: P1.5 topo manifest artifact

**Given** P1-SPEC completes  
**When** recon produces P1.5-TOPO  
**Then** `port/manifests/<wave_id>.toml` exists with topo-sorted `symbol_qualname` list, `manifest_hash` (SHA-256 of canonical TOML), and `task_id`; stale manifest (hash mismatch or `task_id` older than wave) yields WARN at P2 entry, FAIL at Phase 2 integration entry

### AC-12: P5-ROUTE routing artifact + bridge lint

**Given** P4-EMIT completes with tier FAIL or major static finding  
**When** P5-ROUTE executes  
**Then** supervisor produces a routing decision artifact at `.praxia/port/routing/<wave_id>_<finding_id>.toml` (`block_ci` | `backlog_node` | `found_issues`) referencing emit `finding_id`; when `port/bridge/composition_map.toml` is non-empty, CI lint asserts all qualnames ⊆ live `src/xtrax/` symbols

## Appendix A: subagent-stop hook schema (normative v0.1)

**Normative payload** (flat — walker validates this object only):

```json
{
  "tier_verdict": "PASS|FAIL",
  "port_parity_tier": "tier_1|tier_2|tier_3|tier_4|tier_5",
  "oracle_id": "ref:port/reference/<kernel>:v0.1.0:sha256:<hash>",
  "pytest_nodeid": "port/tests/test_parity_<kernel>.py::test_tier_<n>",
  "pytest_exit_code": 0,
  "stdout_sha256": "<sha256 of UTF-8 normalized pytest short summary line>"
}
```

**Optional wrapper metadata** (design/PCW transport only — not validated by walker):

```json
{
  "hook": "subagent-stop",
  "workflow": "port_validation",
  "phase": "P3-PARITY",
  "payload": { "...": "flat payload above" }
}
```

**Walker rule:** FAIL dispatch if `tier_verdict == PASS` and (`pytest_exit_code != 0` OR `stdout_sha256` mismatch). Hash input: final line matching `PASSED|FAILED` for `pytest_nodeid` from captured stdout.

**T4 timeout budget (AC-4):** `@pytest.mark.timeout(120)` per tier on CPU; `port_target.toml` may lower, not raise, without justification.

**Trace-count baseline (AC-5):** `port_target.toml` field `max_traces = 1` default; `chex.assert_max_traces` runs on jitted entrypoint qualname from manifest.

## Non-goals (v0.1)

- Hypothesis metamorphic property tests (audit-fw D1 CC3) — deferred v0.2; leaf example-based parity only
- Paper PDF literature decomposition pipeline — separate epic candidate
- Full jax-port phases 5–8 (PRNG, JIT-boundary) unless kernel explicitly in scope (non-MVP)

## Tech debt (adversarial nitpicks)

| ID | Item | Resolution |
|----|------|------------|
| TD-2180-01 | Rename "6-phase" → "7 executable steps" in docs | Doc clarity only; P1.5 remains non-gate artifact |
| TD-2180-02 | jax-port skill xtrax in-place variant doc | Child backlog after MVP scaffold |
| TD-2180-03 | `routing.toml` `domain=port` severity rows ownership | #2180-c05 proposes rows; #1579 maintainer reviews |
| TD-2180-04 | `port_target.toml` `[capabilities]` flags | v0.1 enforces `stochastic=false`, `dynamic_shape=false`; v0.2 revisits PRNG/dynamic-shape kernels |
