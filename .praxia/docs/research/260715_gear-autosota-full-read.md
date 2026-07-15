---
task_id: 260702_research-roadmap-dags
date: 2026-07-15
sources:
  - "GEAR: Genetic AutoResearch for Agentic Code Evolution (arXiv 2605.13874) — Jeddi, Le, Karaimer, Derpanis, Taati — full read via arxiv.org/html"
  - "AutoSOTA: An End-to-End Automated Research System for State-of-the-Art AI Model Discovery (arXiv 2604.05550) — Li, Shao, Liu, Zhao, Liu, Su, Chen, Yang, Xu, Fang, Zeng, Li, Xu, Xu, Li, Liu — full read via arxiv.org/html"
  - "code recon: src/xtrax/stages/evaluate.py (sealed EvaluateFn seam, T1-07)"
  - ".praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md (T2-03, T2-33, P4-gated section)"
  - ".praxia/docs/specs/260702_design-the-2181-agentic-algorithm-evolut.md (Fork 1, Fork 2)"
  - ".praxia/docs/research/260702_roadmap-research-synthesis.md (prior title-verified-only GEAR/AutoSOTA mentions)"
verification: primary-source full read of both papers (not title/abstract only); cross-checked against the actual sealed-seam source code, not just its docstring claims
status: resolved — T2-03 gate satisfied; P4 (T2-33) unblocked to proceed on its existing design sketch
---

# T2-03 — GEAR + AutoSOTA full read (Phase-2 unblocker)

Backlog: #2181 DAG `.praxia/docs/roadmaps/research-epics/260702_02-dag-2181-autoresearch.md`, item
T2-03 (P0, research, no ACs — planning artifact for P4). This note is the "full read" the DAG
required before finalizing any island-search node; it supersedes the title-verified-only status
recorded in `260702_roadmap-research-synthesis.md` (lines 9, 38, 44, 253, 298) for GEAR, and
resolves the previously-unidentified "AutoSOTA" reference to arXiv 2604.05550.

## 1. GEAR (arXiv 2605.13874) — what it actually does

GEAR is **not** a classic island-GA with periodic migration between isolated demes (the CodeEvolve
model the roadmap's Fork 1 discussion assumed as the comparison point). It is a single **bounded
elite frontier** — a fixed-size pool of "research state" nodes — searched via UCB-style parent
selection, combined with a literal, never-discarded **best-slot ratchet**. Concretely:

- **Research state (a frontier node `eᵢ`)** carries: a commit of the training code, its measured
  scalar metric (bits-per-byte, "bpb"), parameter count, peak VRAM, a pointer to its parent(s), a
  short natural-language description of what was tried, and *node statistics* — how many times the
  node has served as a parent, the mean improvement its children achieved over it, and the step it
  was last used. After every step the agent also writes a reflection (parent/child metrics,
  improvement delta, promotion decision).
- **Parent selection** (which node the next mutation is derived from) uses a composite score:
  productivity (a UCB exploration/exploitation term over `ḡₑ`, the mean child-improvement, and
  `nₑ`, times used as parent) + novelty (Jaccard distance between a candidate's description tokens
  and its most similar recent sibling) + coverage. **This composite score is a population-level
  bookkeeping computation that lives entirely outside the evaluator** — it's computed from the
  frontier's accumulated history, not from any single evaluate() call.
- **Promotion** (whether a newly-scored child enters the frontier) is separate from parent
  selection and uses simple rule-based thresholds: new global best → fills the reserved best slot;
  beats the weakest current elite → replaces it; otherwise discarded. Each child is evaluated
  **independently** — GEAR never compares two children pairwise or scores a batch/generation at
  once.
- **The ratchet.** One frontier slot is permanently reserved as "best (lowest bpb)." It is never
  evicted; improvements are strictly additive over the run. This is the "ratchet" half of the DAG's
  "ratchet/island hybrid" framing — GEAR literally keeps a monotonically-improving best node
  alongside the exploratory frontier, which is closer in spirit to xtrax's own MVP
  (restart-from-best-K + Leap-Path) than to a pure island model.
- **Three variants**: GEAR-Prompt (an LLM agent executes the whole loop from a natural-language
  description of the rules — soft constraints only), GEAR-Fixed (a deterministic external module
  computes parent selection, mutation/crossover choice, and promotion; the agent only edits code),
  and GEAR-Evolve (extends GEAR-Fixed by letting the agent occasionally edit the *search-policy
  controller itself*, with a guard requiring justification after 5 consecutive experiment steps
  without a controller edit). All three beat the single-path baseline; GEAR-Evolve made 6 actual
  controller edits over the run, targeting observed crossover-degeneration failures.

**Evaluation interface, precisely.** Each experiment trains under a fixed 5-minute H100 budget with
the data pipeline, tokenizer, and evaluator harness held constant. The call GEAR actually makes to
score a candidate is, in effect:

```
(code_string) -> {bpb: float, vram_gb: float, params_m: int, success: bool}
```

— **single candidate in, one scalar-keyed metric dict out**, invoked sequentially, once per child.
Everything population/frontier-shaped (UCB parent scores, novelty, promotion thresholds, node
statistics) is orchestration-layer bookkeeping wrapped *around* repeated calls to this
single-candidate evaluator — not a change to the evaluator's own signature. GEAR never asks its
evaluator to see, rank, or compare multiple candidates in one call.

## 2. AutoSOTA (arXiv 2604.05550) — what it is and its relevant mechanism

**Confirmed identity**: "AutoSOTA: An End-to-End Automated Research System for State-of-the-Art AI
Model Discovery" (Li, Shao, Liu, Zhao, Liu, Su, Chen, Yang, Xu, Fang, Zeng, Li, Xu, Xu, Li, Liu).
An eight-agent system that grounds published papers to reproducible code+dependencies, then
iteratively tries to beat the paper's own reported numbers; the study reports 105 new
state-of-the-art results discovered across recent conference papers at ~5 hours/paper.

Relevant mechanism for this question:

- **Search strategy is single-path, best-first — explicitly not population-based.** The paper
  contrasts itself directly with evolutionary approaches: "Compared with evolutionary optimization
  methods such as AlphaEvolve … the present system operates on a fundamentally more complex problem
  space," and runs one sequential iterative-optimization loop (Phase 3) per paper, not a population.
- **Anti-stagnation via a Leap-Path bifurcation**, not diversity islands: if recent iterations were
  all parameter-type (PARAM) tweaks, the next iteration is *forced* onto a "Leap Path" requiring a
  structurally novel idea rather than another parameter nudge. This is mechanically the same shape
  as xtrax's own T2-18 diversity-quota-semantic Leap-Path (forced structural mutation after N
  cosmetic-only iterations) — AutoSOTA is independent corroborating evidence for that design choice,
  not a competing architecture.
- **Fitness representation**: a tree-structured rubric exists at the *planning* layer (recursive
  decomposition of a replication objective into graded sub-tasks — this is the "tree-structured
  AutoSOTA rubric fitness" the roadmap synthesis flagged as a deferred fork-2 option), but the
  actual optimization loop that decides accept/reject per iteration operates on a **single primary
  metric g\*** per paper plus a metrics dict logged to `scores.jsonl` per iteration — i.e., the
  rubric is a planning/scoping artifact, not the loop's live comparison signal. The live
  accept/reject check is `eval(ℛ*) > eval(ℛ_rep)` (candidate result beats the reproduced baseline),
  a scalar-keyed comparison, evaluated one candidate at a time.
- **Evaluation-integrity governance** ("AgentSupervisor" red-line rule R2: evaluation script, score
  aggregation, and metric computation code must not be modified by the agent) is the same
  fitness-monopoly principle as xtrax's AC-7/AC-8 (sealed evaluator closure hash + provenance) —
  another independent corroboration, not a new requirement.

## 3. Does either system require changing xtrax's frozen `evaluate()` seam?

**No — the seam assumption holds.** Both systems' live scoring calls are single-candidate →
scalar-keyed-metric-dict, exactly `(frozen_context, candidate) -> dict[str, float]`
(`src/xtrax/stages/evaluate.py`, T1-07). Specifically:

- **GEAR's actual evaluator call** — `(code_string) -> {bpb, vram_gb, params_m, success}` — is
  structurally identical in shape to xtrax's sealed `EvaluateFn`: one candidate in, one flat dict of
  floats out, no batch/population argument. Everything that *looks* population-shaped in GEAR
  (frontier membership, UCB parent-selection scores, per-node novelty/coverage statistics,
  promotion-vs-weakest-elite comparisons) is computed by an **orchestration layer that sits above
  the evaluator and accumulates history across repeated single-candidate calls** — it is loop-
  controller state, not evaluator-seam state. Nothing in GEAR asks the evaluator itself to receive
  more than one candidate or to emit a relative/pairwise judgment; all comparison (parent-selection
  UCB score, promotion thresholds) is computed *outside* the evaluate() call, by the controller,
  from history the controller itself maintains.
- **AutoSOTA's live loop** never departs from single-candidate, scalar-dominant scoring either; its
  tree-rubric is a pre-loop planning artifact (already the deferred, documented fork in the spec —
  "tree-structured AutoSOTA rubric fitness … nested-dict extension keeps the scalar-leaf contract" —
  and this read confirms that deferral is safe: the rubric never needs to reach the sealed seam).
- xtrax's `evaluate.py` docstring already anticipates exactly this shape of extension: `frozen_context`
  and `candidate` are "intentionally opaque, caller-supplied types," and "the concrete fitness
  registry and any #2181 population representation are a documented extension seam (fork 11), not
  defined here." This read is empirical confirmation that the *opaque* half of the interface
  (`frozen_context`) is where population/frontier bookkeeping would live if ever needed — the
  two-argument, dict-return shape itself never has to change. No mechanism found in either paper
  requires a signal the current seam structurally cannot carry (no within-generation comparison
  reaches the evaluator boundary in either system; no batch argument; no relative/rank-only return
  type).

**One correction to the DAG's framing, not a blocker.** The DAG describes GEAR as a "ratchet/island
hybrid." Having read the paper, GEAR is more precisely a **ratchet + bounded-elite-frontier-with-UCB-
selection** hybrid, not a classical multi-island model with periodic inter-island migration (that
description fits CodeEvolve, the other system cited in the roadmap's Fork 1, arXiv 2510.14150, not
GEAR). This doesn't change the seam-compatibility conclusion, but it does change what "island
upgrade delta" should mean in practice — see §4.

## 4. Implication for T2-33's design

T2-33 (`P4-gated`, currently sketched as: "batch/generation eval, population state, migration
hooks, per-candidate resource accounting" as a config flip over the frozen scalar seam) should be
adjusted in one respect once it's actually scoped, though nothing here forces action before then:

- **"Batch/generation eval" is not evidenced as necessary.** Neither GEAR nor AutoSOTA ever calls
  an evaluator with more than one candidate at a time; T2-33's eventual design should not assume it
  needs a batched evaluate() call. If per-candidate resource accounting (GPU/wall-clock budgeting
  across concurrently-running candidates) is wanted, that is a **scheduler/controller concern**
  (how many single-candidate evaluate() calls run concurrently and how their results are folded into
  frontier state), not a change to the evaluate() call shape itself.
- **"Migration hooks" should probably be renamed/rescoped to "frontier promotion + parent-selection
  bookkeeping" rather than literal island-migration.** GEAR's actual mechanism — a bounded elite
  pool, UCB-style parent selection over productivity/novelty/coverage, and rule-based promotion
  against a reserved best-slot — is a closer, cheaper, more directly-portable model for xtrax's
  Phase-2 delta than CodeEvolve-style island migration would be, and it composes naturally with the
  MVP's *already-chosen* restart-from-best-K + diversity-quota Leap-Path (T2-17/T2-18): GEAR's
  best-slot ratchet is structurally the same idea as xtrax's restart-from-best-K, and GEAR's novelty
  term is structurally the same idea as xtrax's Leap-Path quota. **When T2-33 is actually designed,
  it's worth explicitly evaluating "extend the existing ratchet-with-restart-from-best-K into a
  bounded elite frontier (GEAR-style)" against "adopt CodeEvolve-style islands with migration"** —
  this read suggests the former is both cheaper and closer to what's already built, but this note
  doesn't force that choice; it's a design-input for whoever scopes T2-33, not a re-litigation of
  Fork 1's MVP decision (ratchet-with-restart-from-best-K stands, unchanged, for the MVP).
- **Population state** (per-node parent-usage counts, mean-child-improvement, novelty descriptions)
  is real, non-zero-cost state a Phase-2 upgrade would need to add — consistent with the DAG's own
  framing ("a documented 'island upgrade delta' … rather than promising zero-cost drop-in"). This
  read doesn't reduce that cost; it just localizes where the cost lands (controller/`frozen_context`
  bookkeeping) and confirms it never touches the evaluate() call signature.

**Bottom line for the P4 gate**: T2-03's gate condition is satisfied. T2-33 may be finalized on
its existing sketch (config-flip over the frozen scalar seam); the one recommended adjustment is
scoping "migration hooks" as GEAR-style bounded-frontier/UCB-parent-selection bookkeeping rather
than literal multi-island migration, which is cheaper and more consistent with the MVP already
shipped in P1/P2.
