---
task_id: 260702_research-roadmap-dags
date: 2026-07-02
sources:
  - "NLM notebooks (5) via prequery dump: .praxia/research/260702_nlm_prequery_roadmap.md (Q1–Q9, incl. notebook 3b11ab5b)"
  - "praxia code recon: crates/praxia-workflows, praxia-cli, praxia-rig-tools, praxia-workflow-runtime, praxia-config"
  - "bathos code recon: src/bathos (prereg, sidecar, campaigns, claim, parity, provenance, runner, verify, mcp, schema)"
  - "xtrax code recon: worktree src/xtrax (inference, stages, cli, run, engine, devtools, safety) + main-checkout uncommitted stages/topology.py"
  - "web verification: AlphaEvolve (arXiv 2506.13131), Karpathy autoresearch, Cerebras 2026-03-19, AutoSOTA (2604.05550), POPPER (2502.09858), MLR-Bench (2505.19955), EXP-Bench (2505.24785), CodeEvolve (2510.14150), XGrammar-2, Outlines, Mündler et al. PLDI 2025 (2504.09246), LINC (2310.15164), Stitch/LILO/DreamCoder"
verification: adversarial verify pass per theme; refuted claims dropped (see appendix), corrections applied inline
status: synthesis for roadmap-DAG construction + contemplex brainstorm seeding
---

# Roadmap Research Synthesis — 260702

## Executive summary

Six verified research threads converge on one architecture: **xtrax stays the truth-emitting substrate, bathos owns campaign rigor, praxia owns dispatch — and #2181's autoresearch loop is a ratchet (Karpathy-style), not an island search, built behind an immutable `evaluate(candidate) -> dict[str, float]` interface designed for later population-search drop-in.**

The strongest external evidence is negative: MLR-Bench (80% fabricated results), EXP-Bench (0.5% full-experiment success), and Cerebras' documented goal drift + proxy gaming within an overnight run mandate scoped single-function evolution over pure-JAX callables, metrics accepted only from an immutable execution-verified evaluator, and one-experiment-per-call with git-as-memory state separation.

Locally, the substrate is further along than expected on two of three legs (mutation boundaries via Fuse/Tap/Sink purity contracts + StageBundle typed slots; async tracking via ordered io_callback contracts + the in-flight topology validator) and **greenfield on the third: the immutable fitness oracle has no existing xtrax primitive**. Bathos already ships pre-registration gates, POPPER e-value stopping, SHA-anchored claims, and a full MCP surface — but has **zero statistical battery, no seed field, chain-only lineage, and no xtrax bridge**. Praxia plugins are workflow-aware at packaging level but **rig-run cannot dispatch plugin flows at all** (hardcoded nine-arm table, compiled-Rust-only tools).

Recommended immediate sequence: land `topology.py` → boundary executor → minimal composition substrate child item (narrowing #2181's #2174 dependency) → typed composition IR + deterministic validate verb (the neuro-symbolic grounding node) → bathos stats-gates module + seed schema → praxia generic rig-run flow.

---

## 1. #2181 — Agentic algorithm evolution & autoresearch (xtrax + bathos)

### 1.1 Reference architecture

**Verdict: ratchet-loop MVP, immutable-evaluator interface designed for island-search drop-in.** Karpathy's autoresearch is the proven template at exactly the single-GPU/single-file scale xtrax targets: the **~630-line single-file framework** drives a simplified-nanochat `train.py` as the **sole mutable sandbox**, following a human-authored `program.md` constitution (correction applied: secondary sources conflate the two — the ~630 LOC scope the framework, not `train.py`; initial 2-day run ≈ 700 experiments). Git ratchet: commit kept only if val_bpb improves, else `git reset` to previous best; `prepare.py` (eval + data prep) is the immutable judge, never modified by the agent.

Loop stages with trust zones (per the NLM irreversibility taxonomy, Q2):

| Stage | Trust zone | Content |
|---|---|---|
| GENERATE | autonomous | AutoSOTA-style constrained hypothesis library (param-tune vs code-edit vs algorithmic, risk-leveled); mutations restricted to EVOLVE-BLOCK-marked pure functions (Fuse impls, StageBundle slots); CodeEvolve-style context = mutable target + read-only deps only. Marker mechanics per the AlphaEvolve white paper (arXiv 2506.13131); DeepMind blog stats: 75% matched existing SOTA / 20% improved best-known — denominator is the 50+ open-problem suite (correction applied). |
| EXECUTE | mechanical gates | Hardened sandbox (no network, read-only config), `SafetyManager(enabled=True)` checkify wrap, fixed wall-clock budget, one-experiment-per-call (Cerebras), bathos run sidecar per candidate. |
| MEASURE | irreversible boundary — agent never writes | Frozen evaluator script + dataset splits outside the writable tree, read-only to the agent; fitness = scalar dict from the evaluator only; eval-path content hash checked every iteration. |
| CRITIQUE | mechanical + adversarial | Information barrier (schema-validated JSON failure summaries, never raw tracebacks); devtools `run_refute_or_promote` as the promotion state machine; `empirical_oracle` failing-pytest repro with promotion budget; POPPER sequential e-value aggregation for statistical claims. |
| REVISE | autonomous within lineage | Git-as-memory ratchet: commit on improvement, `git reset` on failure/crash; lineage recorded via bathos. |

**Local-minima mitigation (cheap):** AutoSOTA-style Leap-Path forcing (scheduled structural-mutation quota when N consecutive iterations lack structural change) + periodic restart-from-best-K. Island GA (CodeEvolve, arXiv 2510.14150) pays off mainly at large parallel budgets — Phase-2 upgrade, made drop-in by fixing the evaluator interface now. Read GEAR (arXiv 2605.13874, title-verified only) and AutoSOTA in full before finalizing Phase-2 nodes.

### 1.2 Trust boundaries → xtrax mapping

The EVOLVE-BLOCK writable/frozen split maps onto contracts xtrax already enforces:

- **Mutable:** source text of functions satisfying the Fuse purity contract (`boundaries.py:33-44`: "pure JAX function — no side effects, no io_callback") and `Optional[Callable]` StageBundle fields (`bundle.py` `__init_subclass__` enforcement). Each candidate in its own file, bracketed by EVOLVE-BLOCK markers.
- **Frozen:** Tap/Sink (effectful io_callback code), AxisSpec/BatchPlanner/tiling, the evaluator, all tracking, the constitution.
- **Static pre-gates per mutant (before compute):** import + jaxlint JL-rules → `extract_schema` (jax.eval_shape, genuinely zero-cost) against the slot's declared BundleSchema → `verify_structure` on a tiny batch. Correction applied: **`verify_structure` executes the candidate once** (`verify.py:43`) — it is a first-concrete-run structural tripwire, not a zero-cost gate; together the pair gates candidates before *sustained* GPU spend.

**Triad status vs NLM Q3 (corrected):** mutation boundaries ✅ (Fuse/AxisBoundary/StageBundle), async JIT-safe tracking ✅ contract-level (Tap/Sink ordered flags; topology validator in-flight — see §2), **immutable fitness oracle ❌ greenfield** — `extract_schema`/`verify_structure` are structural gates, not fitness evaluators; devtools' `empirical_oracle` is a code-review promotion protocol, not a locked scalar-fitness `evaluate()`. This is the D2 deliverable in §2.3.

### 1.3 Gate catalog (concrete thresholds; every gate = success metric + fast/loud failure)

**Per-iteration, mechanical (in-loop; no LLM judgment inside the loop):**

| Gate | Success metric | Fast/loud failure behavior |
|---|---|---|
| candidate-static | Clean import + zero jaxlint JL-series errors | Reject pre-compute; structured JSON error envelope, exit 1; zero GPU time |
| schema-gate | eval_shape-derived schema == slot's declared BundleSchema (zero FLOPs) | Reject before any execution; loud schema-mismatch error |
| structure-tripwire | `verify_structure` abstract==concrete pytree/shape/dtype on one tiny batch | `StructureMismatchError` raised; candidate rejected after exactly one cheap execution |
| candidate-smoke (F4) | L1 dry-run + L2 CPU smoke exit 0 in <60 s (pinned uv lockfile) | Reject pre-budget; sanitized failure summary emitted; no cluster/GPU submission |
| checkified-execution | No NaN/Inf/overflow under `SafetyManager(enabled=True)` checkify float_checks | Host-side raise; candidate marked failed; git reset; promoted best-lineage runs stay unchecked (enabled=False = strict identity) |
| prereg-match (F8) | Run config == bathos pre-registered hypothesis+metric sidecar | bathos `gate_check` denial with structured `GateErrorPayload`; candidate rejected |
| eval-hash-invariant (F2) | SHA-256 of evaluator script, splits, metric defs identical to locked manifest; candidate touches no protected path | **Halt the loop** + human escalation (not mere candidate rejection) — judge tampering is the one non-recoverable event |
| metrics-provenance (F1) | 100% of fitness scalars traceable to immutable-evaluator stdout envelope + bathos sidecar/manifest attestation | Metrics discarded, iteration voided; loud provenance error; never accept agent-reported numbers (MLR-Bench: 80% fabrication) |
| info-barrier-lint (F3) | Failure summaries are schema-validated JSON; agent has no raw-log read path | Iteration blocked; LOUD-FAIL schema error; never skip-on-drift |
| multi-metric-regression (F7) | "Improvement" label only if WR ≥ 0.6, BP ≥ 0.2, Cohen's d ≥ 0.2 across the fitness dict | No ratchet commit; git reset to previous best; silent-semantic-failure (faster-but-more-memory) blocked |
| diversity-quota (F5) | ≥1 structural (non-parameter) mutation per N consecutive iterations | Forced Leap-Path structural mutation scheduled; audit finding emitted (mechanical count; periodic judgment review) |
| external-stop (F9) | Loop terminates within wall-clock/compute budget or on convergence, enforced **outside agent context** | Hard kill of loop process; lineage preserved via git; termination criteria never visible/editable to the agent |

**Promotion (judgment track, machinery exists in devtools):**

| Gate | Success metric | Fast/loud failure behavior |
|---|---|---|
| refute-or-promote (F6) | Promoted iff assert-persona passes AND refute-persona fails to kill (cross-model pairs) | No promotion; label stays `observation`; ceiling_note caveat: same-family critics share blind spots — survival is necessary-not-sufficient |
| empirical-oracle | Observation→bug only on reproducible failing pytest, within promotion budget (default 3/run) | Promotion refused when budget ≤ 0 or repro gate fails (returncode == 0) |

**Campaign-boundary, mechanical (wired into bathos `conclude_campaign` where `run_union_gate` sits today):**

| Gate | Success metric | Fast/loud failure behavior |
|---|---|---|
| statistical battery (BUILD) | Wilcoxon signed-ranks pairwise / Friedman+Nemenyi multi-model, α=0.05 Holm step-down; Cohen's d ≥ 0.2; WR ≥ 0.6 or P(A>B) ≥ 0.75; BP ≥ 0.2; ≥3 seeds ICC > 0.990 (N=29 trials power floor for P(A>B)>0.75 at β=0.05) | Verdict downgrade (`confounded`/`underpowered`) — hard block for confirmation/sequential campaigns, advisory for exploration (reuse existing mode-dependent downgrade pattern) |
| baseline-budget-equivalence (BUILD) | Baseline received ≥ equal HPO trials/compute as candidate | Comparison verdict downgraded; loud in campaign report |
| union gate + parity confound (EXISTS) | All claim clauses mapped; parity confounds controlled | Hard downgrade to `confounded` for confirmation/sequential; bypass only via `force_verdict`, audit-trailed as `claim_mode='bypassed'` |
| POPPER threshold (EXISTS) | Per-script e-value product ≥ 1/α (threshold locked at first non-error run) | Threshold change refused — requires child campaign via parent link; anytime-valid stopping |
| anomaly gates (EXISTS) | residual_rate ≤ 0.10, bypass_rate ≤ 0.10, zero unknown-outcome runs | Flagged in `campaign_review`; truth-only JSON artifacts emitted |
| sidecar-drift (BUILD — promote reserved `SIDECAR_HASH_MISMATCH`) | Sidecar SHA identical to first-run manifest across all runs of a script | Deny (autonomous) / warn (collaborative); cheapest highest-leverage immutable-evaluator safeguard |
| nonrepudiation attestation (BUILD, interim) | Signed manifest_sha256 + stdout hash verifies | Unverifiable run excluded from evidence; full K-Veritas RSA-PSS deferred to late milestone |

**Epic-boundary, judgment:** ablation-ladder design review (length-matched placebo, 1–5 Likert on Importance/Faithfulness/Soundness) · prereg-faithfulness audit (prompts, gen params, model versions, pilot disclosure) · leakage controls (reusable holdouts, masked analysis) · claim calibration ("SOTA"/"improved" wording only if WR ≥ 0.6 AND BP ≥ 0.2, else downgrade wording). Failure behavior for all: finding routed to backlog_node, release blocked at readiness gate.

**Non-negotiable human gates (encode a–d as human-approval DAG nodes, e as release-checklist item):** (a) authoring/amending the constitution; (b) any change to evaluator code, splits, or metric definitions — the agent never approves its own judge; (c) promotion of evolved code out of the sandbox lineage into xtrax main; (d) any expansion of network/tool allowlist or sandbox capabilities; (e) publication-grade claim calibration.

### 1.4 Bathos capability map

**EXISTS (reuse, do not rebuild):**
- Pre-registration as a pre-execution gate: enforced dirs (experiments/benchmarks/validation) require a valid `.bth.toml` sidecar — correction applied: **any of the four kinds** ([experiment]/[benchmark]/[validation]/[debug]) satisfies the gate, and a non-empty hypothesis is a supported [experiment] field but **not mechanically enforced** (validate.py enforces only [outcomes] presence, per-branch condition/decision/reasoning, DuckDB-parseable SQL, result_schema ref, is_residual fallback).
- Autonomous-mode stricter gates: first-of-kind check + mandatory adversarial_check on non-residual outcomes.
- Campaign modes (exploration/confirmation/sequential) with temporal-ordering HARKing guard (confirmation rejects runs predating campaign creation).
- POPPER sequential e-values: threshold-locked, anytime-valid; per-script e-value products in `campaign_review`.
- Claim tier: SHA-anchored `claim.bth.toml`, union gate, parity confound check, atomic `attest_parity`.
- Mechanical SQL oracle (`evaluate_outcome`), X1 cap-lattice parity grader (invariant_pass = hard FAIL floor).
- W3C PROV-JSON lineage (single-parent chain), postmortems with checksum/drift validation, SHA-256 pre-execution lock manifests.
- Full MCP surface for loop orchestration (run/campaign/claim/postmortem/lineage/run_sql tools; note the run tool's lineage param is `derived_from`, not `parent_run_id`).

**BUILD (#2181 scope):**
1. **Statistical battery** — nothing of Q6 exists (zero grep hits for Wilcoxon/Friedman/Nemenyi/Cohen/Holm/ICC/WR/BP). New module (e.g. `bathos/stats_gates.py`) invoked from `conclude_campaign`.
2. **Seed tracking** — Run schema has no `seed` field; ≥3-seed ICC replication is unenforceable until added (pair with `baseline_hpo_trials`/compute fields).
3. **Baseline HPO-budget equivalence** gate.
4. **Campaign DAGs** — lineage is single-parent chain; Q4's parallel-branch merges need a multi-parent `campaign_edges`/`run_edges` table + multi-`wasDerivedFrom` PROV emission.
5. **Component-level sidecar binding** — sidecars bind to script files today, not xtrax pipeline components (StageBundle/composition nodes).
6. **Sidecar drift detection** — `SIDECAR_HASH_MISMATCH` is reserved-unimplemented; promote it (mostly plumbing).
7. **Cryptographic nonrepudiation** — hash-based only today; interim signed-manifest covers most value.
8. **The xtrax↔bathos bridge itself** — xtrax has zero bathos coupling; scope to a thin run-layer hook (RunSpec/StageBundle) emitting bathos runs with campaign_id + component sidecar refs. xtrax stays truth-emitting, not gate-owning.
9. **Budget/convergence stopping** — only e-value stopping exists; the loop controller owns compute-budget/convergence/graceful-termination stops.

---

## 2. #2174 — Next slices + minimal composition substrate (the #2181 dep-narrowing child)

### 2.1 Current state (verified)

- **E1 complete:** `infer_bundle(fn, abstract_inputs) -> (BundleSchema, list[AxisSpec])` via eval_shape; `@axis_config`/AxisOverride (frozen dataclass, `config.py:9-38`); `verify_structure` guard.
- **E2 CLI is single-function only:** plan/explain consume fn import-path + shapes string; run/sweep consume TrainConfig TOML of import-path components; resume consumes manifest.json. **No graph input format exists anywhere — mission pillar 2 is 0% implemented.** Correction applied: `cli/export.py` does serialize *single traced functions* to StableHLO MLIR/flatbuffers via jax.export — no composition-graph IR exists, only per-function export.
- **Stages are declared, never executed:** Fuse/Tap/Sink protocols + AxisBoundary exist; zero io_callback implementations in src/ (docstring hits only); no executor invokes `boundary.fuse/tap/sink`.
- **Load-bearing seam mismatch:** `RunSpec.boundaries` is `list[AxisBoundary] | None` while in-flight `validate_plan_topology` takes `Mapping[str, AxisBoundary]` keyed by axis name — a keying decision blocks wiring.
- **Topology validator status (correction applied):** `validate_plan_topology` + `PlanTopologyError` exist as **uncommitted code on the main checkout** (`/home/marielle/projects/xtrax/src/xtrax/stages/topology.py` + tests), enforcing no-Scan-on-heterogeneous-axis and no-ordered-Tap/Sink-on-Vmap pre-trace, deliberately duck-typed for foreign planners. The roadmap gap is **landing/merging it, not writing it**. `make_inference_plan` itself never existed; the main checkout's dirty tree re-points the AxisBoundary docstring; this worktree still carries the stale reference.
- **Nice-to-haves ahead of must-haves:** `node_metadata_schema.toml` + `capability_registry.toml` exist with zero code consumers.
- **Sealed evaluator absent:** Trainer.step is the closest analog (wrong signature, not sealed); Engine.callbacks is an **immutable static tuple** (correction applied) — the gap is that nothing prevents constructing a new Engine with different callbacks; no registration lock or seal exists anywhere.

### 2.2 Sequencing

1. **S1** — Commit `topology.py` + tests; resolve the list-vs-Mapping boundaries keying in the same commit so the validator has a real producer. (Tiny; already written; unblocks everything.)
2. **S2** — Boundary executor + io_callback reference Tap/Sink: converts the declared surface into behavior. Highest-leverage single slice for both epics.
3. **S3** — HostPrepGraph model + serialization + CLI graph consumption with metadata validation (pillar 2).
4. **S4** — Sealed EvaluateFn seam (small; primarily for #2181).
5. **S5** — Chain-map UI deferred until S1–S4 land and #2175 research completes (`compiler_boundary` invariant: no UI concepts in src/xtrax).

**S2+S3+S4 = the minimal-substrate child item.** Re-point #2181's `depends_on` in `.praxia/loop_priorities.toml:131` from `pure-jax-composition-layer` to the new child id; keep #2175/#2180 deps unchanged.

### 2.3 Minimal composition substrate — child item definition

**Deliverables:**
- **D1 — HostPrepGraph data model:** typed pre-JIT node (callable ref + node_metadata slots enforced from `node_metadata_schema.toml`, required `nl_description`) + edges + **per-node frozen/mutable flag** (the Q3a mutation boundary / EVOLVE-BLOCK analog).
- **D2 — Sealed EvaluateFn seam:** fixed signature `(frozen_context, candidate) -> dict[str, float]`, registration-locked (re-register raises), living beside Engine, not inside it (Q3b — the greenfield triad leg from §1.2).
- **D3 — Executed boundaries:** reference Tap/Sink on `jax.experimental.io_callback(ordered=...)`; an executor that actually invokes AxisBoundary ops on plan axes; `RunSpec.boundaries` re-keyed to `Mapping[str, AxisBoundary]` (or adapter — fork B3); `validate_plan_topology` called at graph construction (Q3c).
- **D4 — Graph serialization (TOML/JSON) + CLI path** (`xtrax plan --graph` / run) consuming the serialized graph. This artifact doubles as the NS typed composition IR (§3) — one schema, two consumers.

**Exclusions:** chain-map visual UI + MathJax rendering (#2174 later), agent identities/skills (#2175), the mutation/evolution engine itself (#2181 proper), auto `@axis_config` codegen (deferred T2+ per CHANGELOG), bathos campaign integration beyond the existing `bathos_sidecar_ref` slot.

### 2.4 Candidate acceptance criteria

- **AC1** — A 2-node host-prep graph + 1 traced fn, serialized, consumed by the CLI, yields the identical BatchPlan as direct `infer_bundle`.
- **AC2** — Mutating a frozen node raises before any trace.
- **AC3** — Sealed evaluator: returns `dict[str, float]`; second registration raises; module exposes no mutation API. Apply the BATHOS measurement-verification rule: sanity-check on synthetic ground truth (one-hot fitness case) before any #2181 loop trusts it.
- **AC4** — Reference Sink writes host records from inside jit via io_callback; step order preserved under `ordered=True` on SafeMap/Scan (counter test); ordered-on-Vmap rejected by `validate_plan_topology` (validator half already tested in `tests/stages/test_topology.py:69-95`).
- **AC5** — Graph round-trip preserves node metadata; missing `nl_description` fails validation.
- **AC6** — Grep-gate: no UI/chain-map identifiers in src/xtrax (`compiler_boundary` invariant). Success metric: zero hits; failure: CI exit 1.

---

## 3. Neuro-symbolic placement recommendation + integration map

### 3.1 Placement recommendation

**Grounding for BOTH #2174 and #2181; no standalone epic** (concurring with NLM Q8, corroborated by backlog structure). Concretely: **one named node inside #2174 — "typed composition IR + schema-constrained authoring"** — sequenced after E1 signature-inference and E2 auto-CLI, alongside the chain-map pillar (the lowered graph is what the compiler consumes); the deterministic **validation-gate artifact becomes an explicit #2181 entry criterion** (its immutable-evaluator/fitness substrate per NLM Q3). A standalone epic would duplicate #2174's dependency chain (#1573, #1451) and starve #2181; burying it only in #2181 delays the chain-map/CLI payoff #2174's pillars already promise. DAG cost: one new #2174 node + one new #2181 edge.

### 3.2 Integration map (ranked by leverage/effort)

1. **HIGHEST leverage, medium effort — versioned typed composition IR:** JSON schema derived from BundleSchema + AxisSpec/AxisOverride + the closed Fuse/Tap/Sink/AxisBoundary node vocabulary (stays-vs-leaves encoded in the Protocols' `__call__` return types — Fuse: S→Out, Tap: T→T, Sink: T→None — plus the `ordered: bool` attribute) + node_metadata slots. Consumed by tyro CLI, future chain-map UI, and agents. This is the LINC pattern: **"fill a strongly-typed graph schema," never "write valid JAX."** Same artifact as D4 (§2.3).
2. **HIGH leverage, LOW effort — validation-gates-as-reward:** chain `extract_schema`/eval_shape + `validate_plan_topology` + jaxlint into one deterministic gate emitting PASS/FAIL/NEEDS_WORK into the existing `audit_verdict` slot. ~80% of the machinery exists. *Gate contract:* success metric = all three checks pass → `audit_verdict=PASS` written to node metadata; failure = FAIL/NEEDS_WORK written, CLI verb exits 1, structured JSON envelope names the failing check.
3. **MEDIUM effort, needs (1) first — schema-constrained graph authoring** via Outlines/XGrammar JSON-schema mode. Production engines: XGrammar-2 is the default backend in vLLM/SGLang/TensorRT-LLM/MLC-LLM (>6x tool-calling compile speed, near-zero overhead); Outlines is mature (guaranteed-valid JSON-Schema/regex/CFG output across transformers/llama.cpp/vLLM/Ollama/OpenAI/Gemini; per-token overhead negligible but **vendor-claimed**, not independently benchmarked — correction applied); every major API provider ships native structured output (Anthropic: public beta Nov 2025, GA by mid-2026 — correction applied).
4. **DEFER — jaxtyping-level type-guided decoding:** Mündler et al. (PLDI 2025) show **74.8% (HumanEval) / 56.0% (MBPP) compile-error reduction vs 9.0% / 4.8% for syntax-only constraining** (correction applied — benchmark-specific figures, not a single 74.8% number), but formalized only for a simply-typed calculus + TypeScript via custom prefix automata. No Python/JAX implementation exists — approximate with generate-then-eval_shape-verify loops.
5. **DEFER to #2181 late phase — Stitch compression** over an accumulated corpus of validated composition graphs into registry abstractions. Stitch is genuinely usable (1,000–10,000x faster, ~100x less memory than DreamCoder sleep); no corpus exists yet.

**LINC pattern status (corrected):** LINC (EMNLP 2023 Outstanding Paper, arXiv 2310.15164) remains the architectural template — a pattern to implement, not a library to import. The originally cited follow-up (arXiv 2509.17377) is **dropped** as supporting evidence (see appendix); replacement evidence (arXiv 2601.09446) shows syntactic constraints improve translation validity while formulas often remain **semantically unfaithful**. Net: pattern maturity **moderate, not high**, and semantic faithfulness of the LLM→IR step is the open risk for "LLM fills typed graph schema" — one more reason the deterministic post-generation verifiers (§3.2 item 2) are non-optional.

### 3.3 Minimum first slice (~1 sprint)

(a) Schema emitter dumping composition IR JSON Schema v0.1 from existing types (keep `schema_version` discipline from manifest.py); (b) deterministic `graph validate` CLI verb in `cli/registry.py` — loads an IR document, runs extract_schema-consistency + `validate_plan_topology`, writes `audit_verdict` (gate contract as in §3.2 item 2); (c) smoke demo feeding the same JSON schema to Outlines locally to author one valid graph. Proves the whole NS loop — constrained generate, deterministically verify, record verdict — without touching chain-map UI or research-stage mechanisms.

### 3.4 Guardrail for the spec gate

Benchmark schema-fill authoring against generate-then-validate on ~10 authoring tasks **before** hardwiring constrained decoding into any loop: hard structural constraints are documented to degrade semantic quality 10–30% on reasoning-heavy tasks (degradation literature is on reasoning benchmarks, not schema-fill — the empirical question is open), and xtrax's cheap deterministic verifiers make generate-then-validate a viable fallback.

---

## 4. Praxia plugin integration contract + gaps

### 4.1 The contract (xtrax side, pure packaging)

1. Author `/home/marielle/projects/xtrax/.praxia/manifest.toml` with `[plugin]` (name="xtrax", version, description, requires_praxia) + `[[plugin.workflows]]` entries `{name, template_path}` where `template_path` is **repo-root-relative** (e.g. `"agent_assets/workflows/port_validation.yaml"`). Note: the key is `template_path`, not `path` — cisterna's uninstalled manifest uses `path=` and would fail parsing.
2. `praxia plugin install /home/marielle/projects/xtrax` → each template exported to `~/.praxia/workflows/xtrax_<name>.yaml`, resolvable as template name `xtrax_<name>` from any cwd via tier 2. `dw_mapping.toml` registration is intentionally NOT done by export.
3. **Drift gate (comes free):** composite hash includes workflow template contents; `SessionContext::init` dirty-check compares current vs stored hash in `~/.praxia/plugins.toml` and re-exports on change; `praxia plugin export` runs it on demand. *Gate contract:* success metric = hash match; failure behavior = **auto-heal (re-export), not loud** — if CI-failing loudness is required, file `praxia plugin export --check` (exit non-zero on mismatch) or a small xtrax diff script vs `~/.praxia/workflows/xtrax_*.yaml`, mirroring `dw emit --check` (diff-level, exit 1 on drift).
4. **SubFlow caveat:** children are resolved by bare filename via the same registry, but plugin install prefixes files to `<plugin>_<name>.yaml` — author flow-internal SubFlow references using the **post-install prefixed names** (child declared `name="port_repair"` is referenced as `"xtrax_port_repair"`). Untested in production: **zero installed consumers of `[[plugin.workflows]]` today** (xtrax would be the first *installed* consumer); integration test needed before multi-template flows rely on it.
5. Registered plugins today: bathos, contemplex, jaxlint, maraxiom, myxcel, praxia, xperiri (seven — correction applied); xtrax absent; no manifest exists yet.

### 4.2 The empirical rig-run bug (two root causes + mitigation)

- **Root cause A:** `FsTemplateRegistry::with_default_dirs()` derives tier 1 (`.praxia/workflows`) and tier 3 (`agent_assets/workflows`) from `std::env::current_dir()`, ignoring `--workspace`; `handle_rig_run` canonicalizes `--workspace` but forwards `template_dirs=None`.
- **Root cause B:** `~/.praxia/workflows` (tier 2, the only cwd-independent tier) was populated 2026-06-27, pre-dating the 260701 contract migration: 30 yamls + dw_mapping.toml, **zero `*_contract.yaml`** (praxia's agent_assets has seven).
- **Immediate mitigation (zero code):** `praxia dw install-templates --overwrite` from the praxia repo root (copies every yaml + dw_mapping.toml, no name filter). Verify: `ls ~/.praxia/workflows/*contract*`.
- **Praxia fix to file:** `handle_rig_run` should construct registry dirs from the canonicalized workspace (`[ws/.praxia/workflows, ~/.praxia/workflows, ws/agent_assets/workflows]`) and pass them as `template_dirs` — plumbing exists end-to-end; add a regression test running from a temp cwd.

### 4.3 Hard praxia-side gaps (backlog items for the roadmap DAG)

- **G1 — Dispatch:** `run_rig_flow` is a hardcoded match over **nine** flow names (recon, reviewer, summarize, research, research_triangulate, procedural_recall, spec_draft_test, impl, recon_legacy — correction applied; the bail at `rig_flow.rs:380-383` itself stale-lists only 8); any other name bails "unknown flow." **Plugin workflows resolve but cannot be dispatched via rig-run at all.** Fix: a generic entry (`--flow generic --template xtrax_<name>` or flow==template fallback) using the existing tool-less spec fallback.
- **G2 — Tools:** strict-mode (ActionContract) tool registries are compiled Rust (`DynActionTool` + `ToolRegistryFactory`); plugin YAML cannot add tools. An unregistered `tool_profile` resolves to an EMPTY registry — non-failing but logged (tracing::warn DIAG — correction applied); dispatch proceeds with zero tools. Hazard: toolless strict nodes.
- **G3 — Substrate mismatch:** `RigDispatchBackend` enforces NO-CLAUDE (local backends only); `port_validation.yaml` is authored as a Claude PCW workflow (7 role_step nodes, Claude agent roles, no tool_profile/action_mode, mapped to a Claude Code dynamic .js). Rig-run dispatch would silently change its execution substrate — **explicit design decision required, not packaging** (recommendation: keep port_validation on the dw emit → Claude PCW path; author a separate `xtrax_port_validation_contract.yaml` with `action_mode: strict` nodes only if a local-model variant is wanted, accepting the Rust tool-registry contribution).
- **G4 — dw run deprecated 260701** in favor of rig-run (as are research_fast/deep.yaml) — the plugin contract must target rig-run; dw run entries kept for backward compat during transition.

### 4.4 Autoresearch-loop flow packaging (for #2181)

Author the loop as a contract yaml declared in `[[plugin.workflows]]`; encode evaluator/critique gates as `verdict_enum` edges with `max_total_rewinds` budgets; put immutable evaluate/commit actions in a praxia-side tool registry **or route them through the bathos MCP** (YAML cannot declare tools); enforce the information barrier by having tools return sanitized JSON failure summaries. Contract-yaml schema: `WorkflowTemplate {name, version, description (optional), depth_class, budgets, trigger_predicates, nodes, edges}`; strict nodes require `action_mode: strict` + `action_space` + `terminal_actions` (default Agentic).

---

## 5. Cross-cutting gate-design template

xtrax's shipped audit machinery is the reusable template. Seven elements, each with success metric + fast/loud behavior, then per-thread apply/skip verdicts.

### 5.1 The template

1. **Track split.** Deterministic gate scripts run BEFORE any judgment dispatch. Deterministic gate shape: pure audit fn returning (JSON envelope with `schema_version`/`emitted_at`/`failure_count`/`failures[]`, exit code); exit 1 on `failure_count>0`; per-dimension gate recipes pass `--no-write-baseline` (correction applied: CI's `audit-deterministic` never invokes baseline-writing gates at all — only the scheduled judgment workflow runs a live gate, with `--no-emit`); each Justfile recipe lints + contract-tests the gate's own code before running it (gates are themselves gated). Judgment findings default `severity=info`/`label=observation` (label set in the file header; the ceiling_note additionally warns same-family assert/refute critics share correlated blind spots — survival is necessary-not-sufficient); **promotion to bug only via a failing pytest (pull-based)**. *Success:* every finding carries `source_track`; no judgment finding blocks CI directly. *Failure:* deterministic critical/major → block_ci; judgment critical/major → backlog_node.
2. **Routing.** Declarative severity×track TOML matrix per domain (signal-bearing rows rank more specific; port→dimension fallback). *Success:* every finding resolves to exactly one destination. *Failure:* resolver **raises ValueError on unmatched rows — no silent default destination.**
3. **Dedup.** Structural `finding_id = sha256(dim ␟ qualname ␟ rule_id)` with normalized-evidence fallback — survives line churn. **Tolerance lesson: every behavior-affecting policy knob goes inside the hash** (port domain adds `tolerance_policy`; otherwise a tombstone recorded under an old tolerance silently suppresses a re-failing finding after the knob changes). Suppression only via the append-only tombstone ledger (`disposition ∈ {accepted, wontfix, out_of_scope}`, raises otherwise). *Success:* records self-validate (recompute hash) before append. *Failure:* ValueError at emit time.
4. **Baselines.** Comparator-typed ratchet (`minimize|maximize|best_ever`; best_ever records, never blocks); tighten only on strict improvement; unknown metrics auto-bootstrap; atomic tmp+fsync+os.replace writes; **CI cannot self-tighten** — by construction, not by flag: baseline-writing scripts never execute in CI (correction applied). *Success:* metric within baseline allowance. *Failure:* gate exit 1; ratchet-debt (at-allowance) distinguished from failure via `_DEBT_HINTS` vs `_FAILURE_HINTS` backlog seeds in the per-dimension bootstrap manifest.
5. **Per-dimension seeding.** One-shot bootstrap under a single run_id emits per-dimension `{passed, metrics, backlog_seed}` — failing vs passing-but-carrying-debt get different actionable hints.
6. **Anti-staleness.** **No hand-maintained `expected_status` anywhere.** Automated rows derive status by running the gate; human gates must bind to a machine-checkable probe (PyPI project exists, tag pushed, file hash present) or a timestamped attestation with TTL that goes stale loudly. Evidence (corrected framing): the n9 OIDC human-gate row **was stale until manually corrected 2026-07-02 (fix uncommitted at verification time)** — the committed record read "open" for ~a week after v0.3.0 shipped to PyPI, and `audit_release_readiness.py` still copies `expected_status` verbatim as live status for `gate_type=='human'` and blocks the verdict on it. The structural criticism stands: the code path trusts a hand-maintained field with no probe. *Success:* every gate status derivable from a machine check. *Failure:* stale attestation → gate flips to blocked loudly at TTL expiry, never silently green.
7. **Anti-verification-theater.** Content-hashed oracle seals (`oracle_id = ref:<subtree>:<version>:sha256:<hash>`, regex-validated, hash recomputed from reference bytes; reference files banner-marked DO-NOT-MODIFY); canonical manifest hashes (excluding the hash line itself); hook payloads cross-checked: FAIL when verdict=PASS but pytest_exit_code≠0, stdout_sha256 mismatch, oracle_id mismatch, or hook-vs-emit-record disagreement. **Supervisors aggregate hook payloads; they never originate verdicts.** *Success:* all four cross-checks agree. *Failure:* walker FAILs the phase, loud.

**Fast/loud conventions (standardize in the epic-DAG template doc):** (a) `schema_version` in every record + LOUD-FAIL mismatch exception, never skip-on-drift; (b) resolvers raise on unmatched rows; loaders raise on missing/wrong-typed **required** fields (optional fields may carry explicit defaults — correction applied); (c) records self-validate before append; (d) state files via tmp+fsync+replace, records via append-only JSONL with explicit flush (today only the tombstone ledger flushes explicitly — extend to the emit seams); (e) every gate recipe lints + contract-tests its own gate code first; (f) gate scripts print a JSON envelope and exit 1 on failure so CI and DAG walkers consume the same artifact.

**PCW DAG shape (from port_validation):** gate node = verdict enum + explicit retry/escalation edges + budget ceilings (`max_total_rewinds: 6`, `max_cost_usd: 12`, `max_walltime_s: 9000`); retry edges are per-node (`p0_oracle`/`p2_static` self-loop on `needs_work`, `p1_spec` on `needs_revision`, `p3_parity` on `port_repair` capped at 2 cycles before `human_escalation` — correction applied); P1.5-TOPO is structurally a degenerate single-verdict gate node, not gateless.

**Known debt before reuse:** `.praxia/audits.jsonl` (217 lines) shows schema pluralism — recent agent-written records bypass the contract-tested N1.1 emit seam. Any new DAG must mandate emit-seam usage or put a validator on the JSONL, else routing/dedup automation breaks. Also: the tolerance_policy-in-hash amendment exists only in `port_emit.py` — the generic seam should grow a `policy_fingerprint` parameter before new domains adopt it.

### 5.2 Per-thread verdicts

- **#2181 loop DAG:** APPLY everything — track split (immutable evaluator), routing matrix (long-running loop = token-budget + dangerous-op scrutiny), finding_id+tombstones WITH policy-knob-in-hash, bootstrap baseline+ratchet (it IS the loop's git-as-memory fitness ratchet; `best_ever` is purpose-built), per-dimension seeding (re-seeded waves), strict anti-staleness (a loop tolerates zero hand-maintained status), hook anti-theater cross-checks (agents will otherwise report PASS theatrically).
- **#2174 substrate DAG:** APPLY deterministic gates + envelope/exit-code convention (graph_auditor recipe is the seed) + anti-staleness. SKIP dedicated routing matrix (reuse dimension defaults; uniformly low-risk scope). SKIP tombstones initially (greenfield). SKIP pre-step bootstrap baseline — let first gate runs auto-bootstrap via the unknown-metric path. DEFER judgment track to post-MVP (typed contracts + deterministic oracles are the must-haves).
- **Plugin thread:** APPLY oracle-seal/manifest-hash pattern to plugin manifests (canonical-hash-excluding-hash-line is directly liftable) + anti-staleness (install/export state probed, not recorded) + SystemExit-on-missing-config typed TOML loaders. SKIP routing matrix and tombstones (small scope).

---

## Brainstorm fork list (contested forks each contemplex session must probe)

**#2181 loop architecture:**
1. **Ratchet vs island search at MVP** — recommendation is ratchet, but GEAR (arXiv 2605.13874, title-verified only) hybridizes them and post-dates the corpus; does it change the Phase-2 plan or the MVP evaluator interface?
2. **Scalar fitness dict vs AutoSOTA tree-structured rubric** — is rubric density worth the judgment-track cost at single-GPU scale?
3. **Compile time in or out of the wall-clock fitness budget** — exclude via persistent XLA cache, or include as an optimization target? Affects ratchet-comparison fairness.
4. **Information-barrier ownership** — bathos (redact stdout in MCP get_run for autonomous agents) vs praxia orchestration layer.
5. **Nonrepudiation depth** — interim signed-manifest (manifest_sha256 + stdout hash) vs full K-Veritas RSA-PSS + hardware telemetry; is bathos claim_register/attest_parity tamper-evidence sufficient?

**Bathos rigor:**
6. **Stats battery placement** — bathos core (scipy dep) vs `bathos[stats]` extra vs external gate script shelled from campaign_review.
7. **Campaign DAG ownership** — bathos schema (`campaign_edges`) vs xtrax composition layer owning the DAG with bathos storing flattened lineage (Q4 favors registry-side; #2177 gives xtrax typed node metadata).
8. **Seed-gate enforcement point** — conclude-time blocker (can't conclude 'held' with <3 seeds per script_sha256) vs advisory anomaly; and is the N=29 power floor per campaign, per script, or per hypothesis clause?

**#2174 substrate:**
9. **RunSpec.boundaries keying** — break to `Mapping[str, AxisBoundary]` (breaking aminx subclasses) vs name-keying adapter in the executor.
10. **Boundary executor placement** — inside the tiling iterators (Vmap/SafeMap/Scan) vs a run-layer wrapper; affects whether Fuse can receive stacked scan ys.
11. **EvaluateFn seam location** — xtrax core vs sibling package, given the compiler_boundary invariant (autoresearch concern vs Q3-classified substrate).

**Neuro-symbolic:**
12. **Schema-fill constrained decoding vs generate-then-validate** — the 10–30% semantic-degradation literature is on reasoning benchmarks, not schema-fill; must be benchmarked (~10 tasks) before the loop design hardwires either.
13. **IR schema ownership** — inference layer vs chain-map UI epic (risk of two competing graph formats); and does the local qwen microflow runtime expose an Outlines/XGrammar-compatible interface, or does constrained authoring need a serving-engine change?

**Praxia plugin:**
14. **Generic rig-run naming** — flow==template fallback vs namespaced `--flow xtrax:port_validation`; interacts with the `<plugin>_` prefix and SubFlow child references.
15. **Strict-mode tool provenance** — praxia-side sandboxed generic shell/bathos tool registry usable by plugin yamls vs xtrax tools contributed as a praxia crate feature vs routing all effectful actions through bathos MCP (determines whether the autoresearch flow is pure-xtrax or praxia-coupled).
16. **port_validation substrate** — keep Claude PCW path exclusively vs also authoring a strict-mode local-model contract variant.

**Gate template:**
17. **Human-gate freshness mechanism** — external probes (network inside the gate: PyPI API, git tag existence) vs timestamped TTL attestations that expire loudly. The repo currently has neither.

---

## Dropped-claims appendix (refuted by adversarial verification)

1. **[neuro-symbolic]** "arXiv 2509.17377 is a 2025 follow-up showing syntactic constraints on LINC's translation step further improve it." **DROPPED.** That paper is a robustness study: neurosymbolic LLM+solver methods are *more robust* to counterfactual perturbation but *perform worse* overall than purely neural methods (even NSCoT lags standard CoT); it neither tests syntactic constraints on translation nor supports "durable, replicated." Replaced (per corrections) with ACL 2025 Industry (GCD for logical parsing) and NL2LOGIC (arXiv 2602.13237); LINC-core claim retained at reduced confidence (~0.75) with the robustness counter-evidence noted (§3.2).
2. **[audit-gate-patterns]** "release_readiness.toml *currently* pins n9_human_oidc expected_status='open'" (present-tense stale-record claim). **DROPPED as stated.** The working tree flipped it to "completed" on 2026-07-02 (uncommitted at verification). Retained in corrected form (§5.1 item 6): the staleness incident was real against committed HEAD, and the structural defect — hand-maintained expected_status copied verbatim as live status with no machine probe — remains unfixed.
3. **[praxia-plugin]** "run_rig_flow is a hardcoded match over 8 flow names; bail at rig_flow.rs:347-350." **DROPPED as stated.** Nine arms (research_triangulate omitted — likely because the bail message itself stale-lists 8); bail at :380-383. Load-bearing conclusion (no generic/plugin-registered flow; plugin workflows undispatchable via rig-run) retained (§4.3 G1).
4. **[praxia-plugin]** "~/.praxia/plugins.toml lists only bathos/contemplex/jaxlint." **DROPPED as stated.** Six plugins at verdict time (bathos, contemplex, jaxlint, maraxiom, myxcel, praxia); seven as of a live re-check 2026-07-02 (xperiri registered since). Load-bearing part — xtrax absent, no manifest — retained (§4.1).

**Demoted to unverified corpus detail (not refuted; excluded from load-bearing use):** MLR-Bench's "Claude Code: 8/10 tasks reported placeholder/synthesized data" sub-figure and EXP-Bench's failure breakdown (39.7% missing components / 29.4% environment/dependency misconfig) are not abstract-confirmed — only the headline numbers (80% fabricated/invalidated; 20–35% subtask vs 0.5% complete) are cited in this synthesis.

*(No other theme produced a refuted claim; all remaining amendments were corrections, applied inline and flagged "correction applied.")*

---

## Open questions

1. **GEAR (arXiv 2605.13874)** verified at title level only — read in full (with AutoSOTA) before finalizing Phase-2 population-search DAG nodes.
2. **bathos attestation strength** — does claim_register/claim_attest_parity provide tamper-evidence sufficient for nonrepudiation of evaluator outputs, or is the signing layer a hard prerequisite for autonomous mode?
3. **Session-init drift-check coverage from xtrax cwd** — SessionContext::init walks ancestors for `.praxia/plugins.toml`; xtrax has no project-local plugins.toml — confirm the global-registry path fires in xtrax sessions.
4. **SubFlow prefixing under plugin install** is untested in production — integration test before xtrax ships multi-template flows.
5. **jax.experimental.io_callback API stability** — still experimental namespace; pin/vendor strategy for the reference Tap/Sink if it moves (JEP 10657 tokens are internal).
6. **Does #2181 consume .praxia/audits.jsonl directly for routing?** If so the mixed-schema records need migration or a domain filter before the loop DAG lands.
7. **Is routing.toml's block_ci destination actually wired to CI for domain=port?** Rows exist and audit-port runs in the Justfile, but no CI consumer of resolve_destination was verified.
8. **Claim-calibration plumbing** — should claim_coverage_report embed the statistical verdicts so the epic-gate reviewer has WR/BP results in one artifact at judgment time?
9. **Component-level sidecar enforcement scope** — does the child item (§2.3) need `bathos_sidecar_ref` code-enforced, or can that remain a #2174 follow-up?
10. **Stitch corpus threshold** — what corpus size of validated composition graphs makes library-learning compression worthwhile for #2181's late phase?
11. **Tier-1 cwd-relative `.praxia/workflows` override** — once workspace-derived dirs land in rig-run, should tier 1 become workspace-relative only (avoiding surprise overrides from unrelated cwds)?
