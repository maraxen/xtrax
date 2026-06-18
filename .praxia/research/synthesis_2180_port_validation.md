---
synthesis_id: 260618_synthesis_2180_port_validation
synthesis_version: "2.0"
epic_backlog_id: 2180
notebook_id: 2e509f42-31a5-42cd-b6b0-79a09dab6af9
notebook_title: "xtrax Research: Port Validation & Autoresearch"
sources: "NLM deep research DR-1..DR-6 (imported); NLM queries Q1–Q9 (2026-06-18)"
prepared_for: contemplex brainstorm (#2180)
task_id: 260617_xtrax-composition-mission
---

# Research Synthesis v2: Implementation Validation Pipeline (#2180)

**Prepared by:** librarian pass (NLM queries + local triangulation, 2026-06-18)  
**Use:** `prior_context` block at end for contemplex `brainstorm_start` on epic #2180  
**Note:** NLM answers synthesize DR-1..6 deep-research imports — treat as **knowledge-store synthesis**, not independently verified papers unless cited to primary sources.

---

## Executive summary

Epic **#2180** extends the settled xtrax audit framework (#1573, spec 260611) into a **vertical slice** for the full **jax-port lifecycle** (phases 0–8): reference vendoring → graded parity → training-boundary gates. The slice must **not** duplicate mechanical CI — agents produce deterministic pytest artifacts and scripts; CI runs them without LLM involvement (two-track model, aligned with audit-fw).

**MVP v0.1** should ship: oracle vendoring gate, T1–T3 + T5 parity tiers (defer T4 gradients to v0.2 unless the port target is explicitly differentiable), jaxlint + chex trace-count foundation gates, and a **`port_validation` PCW template** (new template — do not overload `refactor_with_audit` or `spec_driven_dev`). Scope is **phased**: `src/xtrax/` core leaf parity first, then composition-layer integration (#2174), then lowered-graph JIT invariance — never test lowered graphs before core passes graded parity.

**Emit integration (#1577):** port findings join the contract-tested `audits.jsonl` envelope as `domain: "port"` records with `port_parity_tier`, `oracle_id`, `tier_verdict`, and CC5 routing — reusing tombstone dedup and severity×track matrix from audit spec 260611. **#2181 autoresearch** remains blocked until immutable evaluator + oracle gates exist.

**Out of scope:** Praxia agent episodic memory (#2179).

---

## 1. Problem frame (locked)

xtrax needs an **implementation validation** track that extends repo **audit framework** (#1573) into the full **jax-port lifecycle** (vendor → decompose → pure functions → JIT → graded parity → training boundaries). Today jax-port is skill-only; audit-fw covers repo-wide dimensions but not algorithm port orchestration, literature decomposition, or vendored-reference governance.

**Sibling epic #2181** (autoresearch) must not run until validation gates exist — greedy ratchet loops reward-hack without immutable judges and reference oracles.

---

## 2. Evidence table (NLM query log)

| # | Question focus | Status | Summary answer |
|---|----------------|--------|----------------|
| Q1 | Non-negotiable port requirements | OK (prior) | Reference vendoring, topo port order, five parity tiers, double-where, stateless PRNG, jaxtyping contracts, recompilation gates, immutable evaluator |
| Q2 | Two-track agentic + CI architecture | OK (prior) | Agentic local track produces falsifiable artifacts; CI runs standalone scripts only; no LLM in CI |
| Q3 | Subagent roles + phase gates | OK (prior) | Literature → spec → oracle → port → parity → audit emit FSM; supervisor red-lines eval tampering |
| Q4 | **Pre-mortem failure modes** | **OK** | Recompilation storms (dynamic fn recreation, shape churn, loop unroll); numeric drift (double-where NaN grads, XLA fusion); oracle gaps (underspecified papers, weak test suites); agent eval hacks (conftest tampering, fabricated results, cache spoofing); judgment-track failures (unanimity bias, same-family correlated errors, tool-output ignored) |
| Q5 | Autoresearch vs validation (#2181) | OK (prior) | Karpathy three-file immutable judge; block #2181 until #2180 MVP |
| Q6 | **MVP v0.1 parity tiers** | **OK** | Blocking sequence T1→T2→T3→T4→T5; MVP adds jaxlint static gate + chex `assert_max_traces`; T4 needs finite-diff or IFT for implicit solvers |
| Q7 | **Emit envelope schema (#1577)** | **OK** | `port_parity_tier` enum (tier_1..5), `oracle_id` (hash to vendored artifact), `tier_verdict` {status, tolerance_policy, error_taxonomy_class, max_discrepancy}, `routing` {action, repair_context, human_escalation_reason} |
| Q8 | **Scope: core vs composition (#2174)** | **OK** | Phased: Phase 1 `src/xtrax/` leaf parity (topo sort); Phase 2 composition integration (boundary shape/index errors); Phase 3 lowered-graph JIT invariance + trace gotchas — do not skip Phase 1 |
| Q9 | **PCW template pattern** | **OK** | New `port_validation` template (not extend `refactor_with_audit` / `spec_driven_dev`); five template phases: oracle → topo translation → static/runtime QA → graded parity → self-debug loop |

**Source:** all Q4/Q6–Q9 via `mcp_praxia_nlm_query` on notebook `2e509f42-31a5-42cd-b6b0-79a09dab6af9` (knowledge-store; DR-1..6 corpus). Q1–Q3/Q5 from prior librarian pass (same notebook).

---

## 3. Non-negotiable architectural decisions

1. **Reference vendoring as mathematical oracle** — freeze verified legacy implementation (`# REFERENCE: DO NOT MODIFY`); tests import reference only; delete vendor subtree only after all parity tiers pass (jax-port skill Phase 0).

2. **Topological port order** — static call-graph analysis; port and verify leaves before orchestration layers.

3. **Graded parity hierarchy (blocking sequence):**
   - Tier 1: dtype + shape parity (zero tolerance)
   - Tier 2: float64 grounding (`jax_enable_x64` at test module scope)
   - Tier 3: float32 convergence (document XLA fusion / TF32 tolerances)
   - Tier 4: gradient parity vs finite differences (~1e-4 rtol) or IFT for implicit solvers
   - Tier 5: JIT invariance (`jit(fn)` vs `disable_jit()`)

4. **Double-where gradient safety** — mandatory for masked domain ops (sqrt/log at zero).

5. **Stateless PRNG** — no `np.random` inside traced code; explicit key threading.

6. **Trace-time contracts** — jaxtyping + beartype at boundaries.

7. **Recompilation gates** — jaxlint AST sensors + `chex.assert_max_traces`; block dynamic-shape recompile storms in CI.

8. **Immutable evaluator separation** — for any agent loop: judge read-only; agent sandbox cannot mutate metrics, oracle artifacts, or test data.

---

## 4. Two-track validation architecture (no CI duplication)

| Track | Where | What | Output artifacts |
|-------|-------|------|------------------|
| **Track 1 — Agentic (local)** | Pre-commit / dispatch | Refute-or-Promote critic on diff; adversarial spec review; agent-authored **deterministic** pytests + Hypothesis properties; phoenix2pytest-style regression capture | Code + test files + observation log |
| **Track 2 — Mechanical (CI)** | Remote CI | Same **standalone scripts** as local (`just audit-*`, `pytest tests/parity/`); **no LLM in CI** | Pass/fail only |

**Handoff rule:** Agents produce falsifiable artifacts; CI only runs scripts.

**Integration with xtrax audit-fw:** Port validation is a **vertical slice** atop foundation gates (import-linter, `__future__` ratchet, jaxlint — N0 done). Judgment track uses Refute-or-Promote → `observation`; **`bug` label** requires failing pytest (pull-based, budget-capped) — per audit spec 260611 winner.

---

## 5. MVP recommendation (v0.1)

### Ship in v0.1

| Gate | Rationale |
|------|-----------|
| **Oracle vendoring** (Phase 0) | Blocks all downstream work without ground truth |
| **T1 dtype/shape** | Cheapest failure detector; catches promotion bugs |
| **T2 float64** | Isolates math translation from fp32 noise |
| **T3 float32** | Production-dtype validation with documented tolerances |
| **T5 JIT invariance** | Catches tracing side-effects before integration |
| **jaxlint** (static) | Pre-execution trace-leak / static-arg detection |
| **chex trace-count** | Recompilation storm gate (Performance dimension overlap) |
| **`port_validation` PCW template** | Dedicated FSM; reuses audit emit substrate |
| **Scope: `src/xtrax/` core only** | Phase 1 leaf parity per topo sort |

### Defer to v0.2

| Gate | Rationale |
|------|-----------|
| **T4 gradient parity** | Higher cost (finite-diff sweeps); defer unless port epic is explicitly AD-critical |
| **Composition-layer lowered graphs** (#2174) | Requires core green; boundary errors need integration fixtures |
| **Paper-to-code literature pipeline** (DR-3) | Spec decomposition agents; not blocking first numeric port |
| **Hypothesis metamorphic fuzz** (full) | Start with tier gates; expand after first port wave |
| **Full emit envelope contract tests** (#1577) | Sketch in v0.1; harden when N1.1 lands |

### Blocking sequence (v0.1)

```
ORACLE → T1 → T2 → T3 → T5 → jaxlint → trace-count → AUDIT_EMIT
```

T4 inserts between T3 and T5 when enabled: `… → T3 → T4 → T5 → …`

---

## 6. Pre-mortem failure modes (expanded, Q4)

| Category | Failure mode | Mitigation |
|----------|--------------|------------|
| **Recompilation** | Dynamic fn recreation (`lambda` in jit, new `id()` each call) | jaxlint + chex `assert_max_traces`; ban lambda-in-jit pattern in reviewer checklist |
| **Recompilation** | Dynamic shapes without bucketing | Host-side padding; static bucket policy in port spec |
| **Recompilation** | Python loops inside jit → unroll hang | `lax.scan`/`fori_loop` gate in jax-purity-reviewer |
| **Numeric drift** | Double-where NaN gradients | Mandatory pattern in jax-port skill; `error_taxonomy_class: nan_gradient_vulnerability` in emit |
| **Numeric drift** | XLA fusion reordering | Tiered tolerances; f64 before f32; document expected fusion drift |
| **Numeric drift** | Finite-diff false failures on compiled code | Use tier-appropriate rtol; IFT for implicit solvers (NLM Q6) |
| **Oracle gaps** | Underspecified paper / missing preprocessing | Spec gate blocks port until pseudocode + data contract complete |
| **Oracle gaps** | Developer tests pass but port wrong | Graded parity against vendored reference, not author tests alone |
| **Agent eval hacks** | Malicious `conftest.py` auto-pass | Immutable judge sandbox; supervisor red-line; no agent write to `tests/conftest.py` |
| **Agent eval hacks** | Fabricated results / dummy files | Require pytest exit code + artifact hash in emit; evaluator reads oracle not agent logs |
| **Agent eval hacks** | Cache / reference leakage | Seal oracle artifacts; agent context excludes `port/reference/` |
| **Judgment track** | Unanimous hallucination (80+ agents) | Empirical oracle only ground truth (audit spec 260611); Refute-or-Promote emits `observation` not `bug` without repro |
| **Judgment track** | Same-family correlated blind spots | Admit all-Claude ceiling; do not engineer persona theater as proof |
| **Judgment track** | Tool called but output ignored in artifact | End-to-end gate: emit must cite verifier stdout or FAIL the dispatch |
| **Process** | Autoresearch before validation | Block #2181 on #2180 MVP (loop_priorities.toml) |
| **Process** | Audit-fw duplication | Port track produces tests/scripts; audit-fw consumes emit envelope |

---

## 7. Emit schema sketch (#1577 integration)

Port findings extend the audit emit envelope — **same `audits.jsonl` stream**, new `domain` value. Reuse CC5 routing (`audit/routing.toml`), tombstone ledger, and `finding_id` strategy from spec 260611.

### Record shape (sketch)

```json
{
  "audit_id": "260618_port_attention_t3",
  "task_id": "260617_xtrax-composition-mission",
  "domain": "port",
  "dim": "port",
  "track": "deterministic",
  "finding_id": "hash(port + qualname + port_parity_tier + rule_id)",
  "symbol_qualname": "xtrax.sparse.safe_norm",
  "rule_id": "parity_tier_3_float32",
  "label": "observation",
  "severity": "major",
  "port_parity_tier": "tier_3_float32_convergence",
  "oracle_id": "ref:port/reference/safe_norm:v0.2.0:sha256:8f3a…",
  "tier_verdict": {
    "status": "FAIL",
    "tolerance_policy": "rtol=1e-4, atol=1e-4, matmul_precision=highest",
    "error_taxonomy_class": "numeric_drift",
    "max_discrepancy": 0.0023
  },
  "evidence": {
    "test_path": "port/tests/test_parity_safe_norm.py::test_parity_float32",
    "pytest_nodeid": "…",
    "traceback_excerpt": "…"
  },
  "routing": {
    "destination": "backlog-node",
    "executable_as_failing_test": true,
    "repair_context": null,
    "human_escalation_reason": null
  }
}
```

### Field semantics

| Field | Values / notes |
|-------|----------------|
| `port_parity_tier` | `tier_1_dtype_shape` \| `tier_2_float64_grounding` \| `tier_3_float32_convergence` \| `tier_4_gradient_ad` \| `tier_5_jit_invariance` |
| `oracle_id` | URI to vendored reference + version pin + content hash; links baseline I/O pairs |
| `tier_verdict.status` | `PASS` \| `FAIL` \| `SKIP` \| `TIMEOUT` |
| `tier_verdict.error_taxonomy_class` | `shape_mismatch` \| `numeric_drift` \| `nan_gradient_vulnerability` \| `jit_invariance_violation` \| `compilation_leak` \| `oracle_missing` |
| `label` | Default `observation`; promote to `bug` only with failing pytest (audit-fw rule) |
| `routing.destination` | CC5 matrix: `block-CI` \| `found-issues.md` \| `backlog-node` \| `tombstone-eligible` |

**Integration point:** N1.1 emit envelope (#1577) should add `domain` enum including `"port"` and contract-test the sketch above. Port gate pytest modules emit via a thin `port_emit.py` wrapper calling the same serializer as dimension audits.

---

## 8. Scope boundaries: core vs composition (#2174)

| Phase | Scope | Gates | Depends on |
|-------|-------|-------|------------|
| **1 — Core leaf parity** | `src/xtrax/**` pure kernels | T1–T5 per topo-sorted module | Oracle vendoring |
| **2 — Composition integration** | Host prep + chain-map lowered graph nodes | Boundary contract tests; `graph-auditor` walks node metadata (`audit_verdict` slot per capability_registry.toml) | Phase 1 green for referenced kernels |
| **3 — Lowered graphs** | `jax.export` / inspected JIT artifacts | T5 JIT invariance on full graph; trace-count on composed call | Phase 2 integration fixtures |

**Do not** validate composition lowered graphs before `src/xtrax/` passes graded parity — integration tests cannot isolate leaf math bugs from boundary wiring errors (NLM Q8).

**Repo layout tension (open):** jax-port skill uses `port/reference/` + `port/jax_port/`; composition uses `.praxia/composition/` graph-native paths. Brainstorm should pick one canonical layout or define a lowering mapping.

---

## 9. PCW workflow template: `port_validation` (Q9)

**Recommendation:** New **`port_validation`** template — not an extension of `refactor_with_audit` (repo-wide refactor lacks oracle/parity FSM) or `spec_driven_dev` (spec challenge/defend lacks numerical tiers). Prior art: `spec_driven_dev` rejected for Sprint 6 Track A as overkill (handoff 260608); port validation is similarly specialized.

### Template phases (map to PCW D4 FSM)

| PCW phase | Gate | Primary agents | Artifact |
|-----------|------|----------------|----------|
| **P0 — ORACLE** | Vendored reference + baseline I/O | reference-vendor, recon | `port/reference/*`, baseline pickle/json |
| **P1 — SPEC** | Math/pseudocode/jaxtyping contracts | specification-specialist | Docstring Math/Pseudocode blocks |
| **P2 — STATIC** | jaxlint clean | jax-purity-reviewer | JL report |
| **P3 — PARITY** | T1→T3→(T4)→T5 blocking | test-designer, fixer (port) | `port/tests/test_parity_*.py` |
| **P4 — EMIT** | audits.jsonl port domain records | graph-auditor / auditor | `.praxia/audits.jsonl` entries |
| **P5 — ROUTE** | CC5 severity×track | supervisor | backlog / found-issues |

**Capability registry additions (future semver bump):** `reference-vendor`, `specification-specialist`, `test-designer` (port); extend `jax-purity-reviewer` and `graph-auditor` hooks for tier verdicts.

**Hook surface:** `subagent-stop` may carry `tier_verdict: PASS|FAIL` for parity phases — brainstorm to confirm vs central supervisor only.

---

## 10. Subagent roster (proposed)

| Role | Phase | Responsibility |
|------|-------|----------------|
| **recon / literature** | 0 | Paper PDF, citations, locate reference impl, topological file map |
| **specification-specialist** | 1 | Math → pseudocode; evaluation rubric from paper tables |
| **reference-vendor** | 1 | Vendor oracle; pin version; baseline I/O pairs |
| **planner** | 2 | Port waves by topo sort; file anchors; task_id threading |
| **fixer (port)** | 3 | JAX translation; Equinox if needed; one-file dispatches |
| **test-designer / test-writer** | 3–4 | Graded parity tiers; Hypothesis properties; RED before GREEN |
| **jax-purity-reviewer** | 4 | JL rules, PRNG, double-where, io_callback placement |
| **reviewer / auditor** | 5 | Tier gate verdicts; numerical evidence |
| **supervisor** | all | Red-line: no metric tampering, no conftest hacks, no eval leakage |

Maps to existing registry identities (`jax-purity-reviewer`, `graph-auditor`) plus new port-specific entries.

---

## 11. Foundational context: xtrax audit framework

xtrax audit-fw (spec 260611, epic #1573): **8 dimensions** + **foundation gates**. Two tracks: deterministic CI + judgment agents. Refute-or-Promote; failing-pytest promotes observation→bug. Baseline JSON ratchets; tombstone dedup; severity×track routing. N0.1–N0.4 implemented; **N1.1 emit envelope (#1577) next**.

**#2180 extends** audit-fw with **algorithm port lifecycle** — does not replace dimensions. Port tier failures on `src/xtrax/` kernels may also surface as Correctness or JAX-purity dimension findings when the same code is audited repo-wide.

---

## 12. Open questions for contemplex brainstorm

1. **Repo layout** — `port/reference/` + `port/jax_port/` (jax-port skill) vs `.praxia/composition/` graph-native paths; lowering mapping between them?
2. **T4 in v0.1?** — NLM recommends full T1–T5; MVP table defers T4 unless AD-critical — pick per first port target (e.g. training loss = yes, EDA planner = no).
3. **Emit contract timing** — sketch now vs wait for N1.1 #1577 land; who owns `port_emit.py`?
4. **Hook surface** — `subagent-stop` tier PASS/FAIL extensions vs supervisor-only verdict?
5. **Composition Phase 2 entry criteria** — all of `src/xtrax/` green or only kernels referenced by active graph nodes?
6. **Paper ports** — information isolation (mask author results tables during validation): how enforced in PCW template?
7. **Routing repair loop** — NLM suggests `agentic_self_debug` action; align with existing fixer dispatch budget or separate port-repair identity?
8. **Relation to distribution epic #1451** — does `port/` subtree ship in wheel or dev-only extra?

---

## 13. Suggested brainstorm topic string

```
HMW: Design a unified implementation validation pipeline for xtrax that extends the audit framework into jax-port (reference vendoring, graded parity, agentic pre-CI track) without duplicating mechanical gates — MVP v0.1 scoped to src/xtrax core with port_validation PCW template and emit envelope port domain?
```

---

## prior_context (for contemplex brainstorm_start)

```text
EPIC #2180 — Implementation validation pipeline (jax-port + audit integration). Extends settled xtrax audit-fw (#1573, spec 260611): 8 dimensions, two tracks (agentic local → deterministic CI), Refute-or-Promote, bug label needs failing pytest. N0 foundation gates done; N1.1 emit envelope #1577 next.

PROBLEM: jax-port is skill-only today. Need vertical slice for phases 0-8: vendor reference oracle, topo port order, graded parity T1(dtype/shape)→T2(f64)→T3(f32)→T4(grads)→T5(JIT), double-where, stateless PRNG, jaxtyping, jaxlint+chex trace gates, immutable evaluator for any agent loop. #2181 autoresearch BLOCKED until #2180 MVP.

MVP v0.1 (NLM Q6): ship oracle vendoring, T1+T2+T3+T5 (defer T4 unless AD-critical), jaxlint+chex, new port_validation PCW template (NOT refactor_with_audit or spec_driven_dev). Scope Phase 1: src/xtrax/ core leaf parity only. Phase 2: composition #2174 integration. Phase 3: lowered-graph JIT invariance. Never test lowered graphs before core passes parity.

TWO-TRACK: agents write deterministic pytest+scripts; CI runs same scripts, no LLM. No duplicate mechanical gates.

PRE-MORTEM (NLM Q4): recompilation storms (dynamic fn id, shapes, loop unroll); numeric drift (double-where NaN, XLA fusion); oracle gaps (underspecified papers, weak tests); agent hacks (conftest tampering, fake results); judgment failures (unanimity bias, same-family errors, ignored verifier output). Mitigations: supervisor red-lines, sealed oracle, empirical pytest as only bug ground truth.

EMIT (#1577 sketch, NLM Q7): domain=port records in audits.jsonl with port_parity_tier, oracle_id (vendored ref hash), tier_verdict{status,tolerance_policy,error_taxonomy_class,max_discrepancy}, CC5 routing. Reuse tombstone+finding_id from audit spec.

PCW TEMPLATE (NLM Q9): port_validation phases P0-ORACLE→P1-SPEC→P2-STATIC(jaxlint)→P3-PARITY(T1..T5)→P4-EMIT→P5-ROUTE. Agents: reference-vendor, specification-specialist, test-designer, jax-purity-reviewer, graph-auditor, supervisor. Extend capability_registry.toml.

OPEN: repo layout port/ vs composition graph paths; T4 in v0.1?; hook tier verdicts; Phase 2 entry criteria; paper info isolation; port/ in wheel vs dev-only.

BRAINSTORM TOPIC: HMW unified validation pipeline extending audit-fw into jax-port without CI duplication — MVP core-only + port_validation template + emit port domain?
```
