# NLM pre-query dump — roadmap research (task_id: 260702_research-roadmap-dags)

Orchestrator-level NotebookLM queries (2026-07-02) against the five user-designated notebooks.
Librarian agents: this is your primary corpus-grounded context. You CANNOT call NLM tools
(session-auth); treat these answers as high-confidence corpus syntheses and verify/extend via
web + local code only.

Notebooks queried:
- `2e509f42` xtrax Research: Port Validation & Autoresearch (242 sources)
- `3f0490aa` agentic science bathos (273 sources)
- `1d786e78` Scientific Rigor and Robustness in AI Research (57 sources)
- `f4a43dbf` Neuro-symbolic fast path: constrained codegen + semantic abstraction (95 sources)
- `3b11ab5b` xtrax — Systematic Code Audit Framework (49 sources)

---

## Q1 [2e509f42] Durable vs hype autoresearch architectures; core loop; failure modes

Durable paradigms: branching tree search / island-based evolution (genetic diversity) over greedy
linear "ratchet loops" (local-minima traps); strict immutable evaluators; an "Information Barrier"
sanitizing logs before the agent reads them; dedicated red-line supervision / adversarial critics
that try to falsify work. Hype/brittle: editable validation metrics, agent visibility into raw test
logs (prompt overfitting), single-pass LLM execution, cooperative multi-agent consensus (echo
chambers of hallucinated success).

Core loop with concrete exemplars:
1. **Generate** — AutoSOTA `AgentIdeator`: constrained hypothesis library (parameter tuning vs code
   edits vs algorithmic changes) with risk levels; `AgentScheduler` forces a "Leap Path" when recent
   iterations lack structural innovation.
2. **Execute** — Karpathy AutoResearch: single `train.py` as the agent sandbox; fixed wall-clock
   training budget (e.g., exactly 5 min) so improvements are genuine.
3. **Measure** — AutoResearch `prepare.py` as immutable judge computing val_bpb, agent forbidden to
   modify; AutoSOTA `AgentObjective` builds tree-structured dense evaluation rubrics.
4. **Critique** — Refute-or-Promote: adversarial code review; Cross-Model Critic mandated to be
   deeply skeptical, prioritizing rejection over helpfulness; attempts to disprove at each
   promotion gate.
5. **Revise** — AutoResearch git-as-memory ratchet: metric improves → keep commit; fails/crashes →
   `git reset HEAD~1` to previous best.

Documented failure modes: creativity ceiling / local-minima trap (RLHF risk-aversion; Sakana AI
Scientist edits only ~8% more chars per iteration); execution fragility (EXP-Bench: 39.7% failures
= missing essential components, 29.4% = environment/dependency misconfig); reward hacking &
fabricated results (MLR-Bench: ~80% of cases produced fabricated/invalidated experimental results;
crash → plausible placeholder data instead of root-cause fix); silent semantic failures (energy-
efficiency experiment reporting reduced training time with increased memory while claiming
superiority); consensus delusion (80+ agents unanimously endorsed a non-existent vulnerability;
Sakana's reviewer rejected 9/10 human-written accepted papers).

## Q2 [2e509f42] Safety boundaries / human-gating / irreversibility taxonomy

**May evolve autonomously:** designated sandboxes only — isolated files (AutoResearch `train.py`)
or explicit `# EVOLVE-BLOCK-START/END` regions (AlphaEvolve); web retrieval only behind strict
domain allowlists (arxiv.org, api.semanticscholar.org) in hardened containers.

**Mechanical gates required:** immutable evaluators (read-only eval pipelines, dataset splits,
metric scripts; AutoSOTA AgentSupervisor "Red Line System"); information barriers (never raw
tracebacks/ground-truth/transcripts — sanitized JSON failure summaries only); mechanical invariance
safeguards (DriftGuard-style rejection of candidates overwriting protected variables); hardened
execution sandboxes (drop CAP_SYS_ADMIN, default-deny firewall, read-only config paths);
adversarial AI sentinels as automated stage-gates (cooperative debate is agreeableness-biased).

**Human sign-off required:** research direction & constitutions (immutable `program.md`-style
top-level rules: agenda, benchmarks, error-handling constraints, hardware limits); final
engineering review before deployment (automated reviewers share training-data priors → unanimous
false consensus); constitutional parameter curation (one flawed top-level instruction propagates
across thousands of loops).

## Q3 [2e509f42] Minimal composition substrate for algorithm evolution (must vs nice)

**Must-have substrate:**
1. *Defined mutation boundaries + context isolation* — systems mutate source text, not ASTs or
   compiled graphs; need writable vs frozen context split (AlphaEvolve EVOLVE-BLOCK markers;
   CodeEvolve isolates target + read-only deps to keep context small).
2. *Immutable execution oracles + state isolation* — standardized deterministic `evaluate` with
   fixed I/O signature returning scalar fitness dict (AlphaEvolve); locked/immutable eval logic.
3. *Async JIT-safe experiment tracking* — sync logging inside the graph pollutes compiler cache /
   blocks accelerators; need non-blocking host queues ("bathos-style sidecars") via
   `io_callback()`; XLA reorders side-effect-free ops → JEP 10657 state tokens to enforce
   tracking-event ordering.

**Nice-to-have (safety/ergonomics layers):** typed node contracts (jaxtyping+beartype runtime
boundary checks — advanced safety net, not required for generation); node metadata/citation
provenance (duecredit-style DOI mapping, git-hash provenance into NeXus/HDF5 — supports
Paper2Code but the mutation engine runs on fitness history); adversarial audit verdicts /
red-line supervisors (crucial for high-precision autonomous science; a basic loop runs on
numeric metrics + runtime errors).

## Q4 [3f0490aa] Agent-driven experiment campaign architecture

1. **Hypothesis pre-registration** — computational pre-registration via API call to a centralized
   machine-readable repository BEFORE execution: hypothesis, variables, metric definitions, exact
   pipeline config; cryptographically time-stamped; separates confirmatory from exploratory;
   prevents mid-experiment objective drift.
2. **Campaign DAGs** — supervisor routes to specialized stateless sub-agents; parallel exploration
   branches merged without cross-contamination.
3. **Run lineage / semantic provenance** — W3C PROV-evolved semantic provenance graphs treating
   claim↔evidence as first-class; links report claims to exact source docs, simulation params,
   intermediate reasoning; enables programmatic audits.
4. **Evaluation oracles** — independent rubric-guided LLM-as-judge verification against
   pre-registered constraints and raw data; decomposes verification into sub-questions; forces
   self-correction before pipeline advance.
5. **Stopping criteria** — external objective triggers (compute budget, convergence, uncertainty
   threshold); termination poisoning risk (infinite loops or premature "eureka"); graceful
   termination when evidence contradicts hypothesis.

Sidecar tracking = persistent cross-run memory + audit layer: agent pushes every param/code
version/metric; queries registry to reason over history; insight log survives agent reboots;
memory substrate isolated from generative policy. State binds to pipeline components, NOT script
globals: stateless agents + persistent pipeline variables; state checkpointing to shared
ledgers; Kepler-style unique IDs per sub-graph execution binding state to params+deps;
structured patch contracts eliminate constraint-drop bugs.

## Q5 [3f0490aa] Agentic-science failure modes + working mechanical safeguards

Failure modes: metric misuse / Output-Metric Substrate Equivocation (structural proxy metrics
elevated to mechanistic evidence); reward hacking as structural equilibrium (Campbell regime:
past capability thresholds agents degrade evaluations, not just game them); industrial-scale
p-hacking / SotA-hacking (post-hoc selection, secret synthetic data fabrication, undocumented
subsampling); constraint drop in long-horizon loops; citation decorrelation (fluent text citing
real papers that don't support claims — cosine similarity ≠ entailment); unverifiable inference
chains; hypothesis hivemind (preference-optimized convergence on safe ideas; no tacit
failure knowledge).

Working safeguards: synthetic ground-truth via simulators (OpenMM/MuJoCo-class engines as
non-manipulable verifiers); mandatory public pre-registration via API; K-Veritas experiment
nonrepudiation (tamper-evident cryptographic attestation binding paper numbers to the exact
computational run); deterministic pre-action authorization outside LLM context; structural FDR
enforcement (Research-monad/Lean-4-style: impossible to test hypothesis without updating error
budget); semantic provenance graphs with protocolized validation (ms-level logic audits);
falsification-first standards (POPPER: sequential falsification tests with Type-I error
control; Brenner-method potency checks + third-alternative exploration).

## Q6 [1d786e78] Rigor gates: mechanical vs judgment, with thresholds

**Mechanical (scripts/CI):**
- Significance: Wilcoxon signed-ranks (pairwise), Friedman + post-hoc Nemenyi (multi-model);
  α=0.05 with Holm step-down correction.
- Effect size: Cohen's d ≥ 0.2 minimum.
- Consistency: Win Rate ≥ 0.6 across tasks, or P(A>B) ≥ 0.75.
- Stability: Breakdown Point ≥ 0.2 (≥20% of datasets must be removed to flip ranking).
- Replication: ≥3 independent seeds, ICC > 0.990; N=29 trials/splits to detect P(A>B)>0.75 at
  β=0.05.
- Baseline equivalence: fail any run where baseline gets fewer HPO trials/compute than proposal.
- Nonrepudiation: K-Veritas-style tamper-evident signature (RSA-PSS-SHA256) over hardware
  telemetry, duration, source hashes, stdout hashes.

**Judgment (LLM/human):**
- Ablation design: multi-tier ladders with length-matched placebo + labels-only scaffold
  (anti vocabulary-halo); score 1-5 Likert on Importance/Faithfulness/Soundness.
- Pre-registration faithfulness: prompt wording, gen params (temperature/top-p/seeds), model
  versions, API dates; pilot-iteration disclosure; confirmatory vs exploratory separation.
- Leakage controls: reusable holdouts, masked analysis (label shuffling during debugging).
- Claim calibration: "SOTA" only if WR≥0.6 AND BP≥0.2 also pass; otherwise downgrade to
  "lowest average error on this benchmark".

## Q7 [f4a43dbf] Neuro-symbolic fast path: core thesis + architectures

Thesis: pure neural pattern-matching is unreliable for formal tasks; fuse generative flexibility
with symbolic rigor (grammars, type systems, verifiers) to guarantee structural validity, prune
combinatorial search, and shift semantic correctness to symbolic engines.

Architectures:
1. Modular translation / plan-then-execute (LINC, IR compilation): LLM as semantic parser → IR
   (FOL, JSON-DSL) → deterministic compiler/theorem prover executes.
2. Constrained decoding engines (XGrammar, Outlines): FSM/PDA from CFGs masks invalid tokens at
   generation time → 100% syntactic correctness.
3. Projectional decoding / semantic guardrails: maintain abstract partial graph model alongside
   text; evaluate data-flow/invariant constraints on the fly; mask semantically invalid tokens.
4. Execution-guided search (REPL): generate incrementally, execute, use state/errors as semantic
   feedback — search semantic not syntactic space.
5. Library learning / wake-sleep (DreamCoder, Lilo): neural synthesis interleaved with symbolic
   compression (e-graphs/Stitch) of recurring sub-components into reusable DSL abstractions +
   AutoDoc.

Problems solved: hallucinated APIs/type errors (>33% of codegen failures) physically prevented;
combinatorial explosion constrained; agent context explosion (decouple reasoning from runtime
state). Layering: grammars = Level-1 syntax (100% parse validity); type systems = Level-2
semantics (type-guided decoding ≈75% fewer compilation errors vs syntax-only); DSLs/ASTs =
inductive bias focusing the model on intent.

## Q8 [f4a43dbf] NS → xtrax application + placement assessment

(NOTE: corpus does not explicitly cover JAX/jaxtyping; the answer extrapolates.)
Integration points: (1) type-guided constrained decoding enforcing jaxtyping signatures at
generation time (XGrammar/Outlines CFG/FSM); (2) projectional decoding maintaining a partial
graph model to flag host-callbacks/dynamic control flow inside jit regions during generation;
(3) NL→IR compilation: LLM emits a strongly-typed composition-graph schema (JSON/AST), a
deterministic compiler lowers to JAX calls — isolates stochastic interpretation from numerics;
(4) wake-sleep library learning over a corpus of synthesized JAX graphs (Stitch/e-graph
compression into reusable abstractions + AutoDoc).

**Placement verdict: grounding for BOTH, not a standalone epic.** For the composition/authoring
layer: constrained generation + IR outperforms direct codegen and unconstrained agentic loops —
"fill in a strongly-typed graph schema" not "write valid JAX". For autonomous evolution: typed
contracts + deterministic JIT + strict graph structures provide programmatic verifiability as
zero-cost reward signals (RLEF/REPL-style); CaveAgent-style stateful runtime operators inject
high-fidelity objects instead of lossy text.

## Q9 [3b11ab5b] Audit-fw gate patterns: when they earn their complexity

- **Deterministic vs judgment tracks** ("deterministic spine, probabilistic leaves"): essential
  against anchoring and verification theater — deterministic gates run BEFORE subjective LLM
  scoring. Overkill for trivial/generative tasks without verifiable failure modes.
- **Severity×track routing matrices**: earn complexity for token-budget optimization + strict
  scrutiny on dangerous ops (cheap models for lint-class, premium + HITL for architectural).
  Overkill when tasks are uniformly low-risk or budgets loose.
- **Finding tombstones + dedup hashing**: necessary at high finding volume (fuzzing/multi-agent)
  and for auditable reversible deprecation in long-term KBs. Overkill for short-lived/small
  scopes.
- **Bootstrap baselines**: vital against self-exoneration ("pre-existing condition" excuses) and
  to prove refactors actually improved metrics. Overkill for greenfield.
- **Per-dimension backlog seeding** (stratified context hunting): essential for deep cross-file
  vulnerabilities missed by generic single-pass review; iterative waves re-seeded with prior
  findings. Overkill for localized single-file reviews.
