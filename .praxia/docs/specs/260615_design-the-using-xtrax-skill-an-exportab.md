---
session_id: f910dd23
topic: Design the using-xtrax skill — an exportable reference guide for agents working with the xtrax JAX library. Must cover: core primitives (AxisSpec, BatchPlanner, BatchPlan, strategies), composition model (Fuse/Tap/Sink/AxisBoundary, TransformFn/RollingFn, StageBundle), the run layer (RunSpec, CarrySpec, FeatureBatch, InputResolver, RuntimeBundle, SinkSpec), iterators (VmapIterator, SafeMapIterator, JaxScanIterator, BucketIterator), JAX integration (PyTree, vmap, JIT, PRNG, eqx.Module), EDA features, sparsification, training loop (Trainer/Engine/ResumableState), and documented gaps between the current implementation and the pure-functional composition vision.
task_type: constrained-technical
winner: Faction D (Hybrid) — critic-revised: self-contained tier-1 + versioned anchors + cross-cutting invariants section.

Structure:
- TIER-1 (self-contained mandatory read, must function without tier-2):
  (a) Skill frontmatter: xtrax_version: 0.3.0, trigger conditions, pre-flight compatibility assertion
  (b) Pre-flight checklist: version check, import verification, extras check for EDA
  (c) JAX discipline for domain library authors: static vs dynamic fields, PyTree invariants for AxisBoundary (all-static, empty leaves), JIT boundaries (sparsify_model outside, Fuse inside, Tap/Sink via io_callback), eqx.filter_jit vs jax.jit
  (d) Primitive → problem decision tree: which xtrax primitive for which task (flowchart in prose, merged from Faction B)
  (e) Workflow index: task-oriented anchors to tier-2 sections (build domain library / tiled inference / training loop / EDA planning audit / sparsification)
  (f) Minimal working pattern: AxisSpec → BatchPlanner → BatchPlan → make_axis_dispatch → iterator (fully self-contained, no tier-2 needed)

- TIER-2 (deep reference, organized by workflow proximity, not module hierarchy):
  (a) Tiling layer (40%): AxisSpec, BatchPlanner, strategies, dispatch, iterators, CarrySpec, DedupSpec, bucketize
  (b) Run layer (20%): RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec, AxisBoundary, Fuse/Tap/Sink
  (c) Training layer (25%): ResumableState, Trainer, SafetyTrainStep, Engine, callbacks, Optax integration
  (d) EDA (10%): extract_plan_stats, explain_plan, render, plan_to_dataframe — including EDA-as-planning-audit workflow
  (e) Sparse/Distributed/Checkpoint (5%): pointer pattern + brief API surface

- TIER-3 (inline with tier-2, not appended): 
  Each section carries inline gap callouts (⚠ GAP), HiTL investigation stops (🔬 HiTL), and enforcement-backed NOT callouts (🚫 citing file:line of enforcement). No soft-convention callouts.

Key design constraints resolved from critic:
- Tier-1 is fully self-contained: no dangling anchors, navigation usable without tier-2
- All code examples cite file:line (source is authoritative; skill is a pointer)  
- Version frontmatter + compatibility assertion in pre-flight
- "What NOT to do" callouts restricted to enforcement-backed contracts with file:line citation
- JAX discipline section serves domain library authors who don't navigate by workflow
- HiTL stops: (1) DedupSpec k_bucket at k>256, (2) CarrySpec init static-shape verification, (3) topology validator absent (ordered tap/sink + Vmap: runtime-only detection), (4) bucket_boundaries recompilation vs. padding-waste tradeoff
created_at: 2026-06-15T23:50:28.201904+00:00
---

# Brainstorm: Design the using-xtrax skill — an exportable reference guide for agents working with the xtrax JAX library. Must cover: core primitives (AxisSpec, BatchPlanner, BatchPlan, strategies), composition model (Fuse/Tap/Sink/AxisBoundary, TransformFn/RollingFn, StageBundle), the run layer (RunSpec, CarrySpec, FeatureBatch, InputResolver, RuntimeBundle, SinkSpec), iterators (VmapIterator, SafeMapIterator, JaxScanIterator, BucketIterator), JAX integration (PyTree, vmap, JIT, PRNG, eqx.Module), EDA features, sparsification, training loop (Trainer/Engine/ResumableState), and documented gaps between the current implementation and the pure-functional composition vision.

## Problem Frame
Fixed constraints:
1. All content must be verified against actual code — no inferring behavior from names or specs that may have diverged from implementation.
2. The skill must be exportable (usable outside the xtrax project context), so it cannot rely on local CLAUDE.md or project-level context.
3. Accuracy over completeness — flag gaps and TBDs explicitly rather than filling them with guesses.
4. The skill covers xtrax v0.3.0 (AxisSpec.default_batch_size / tile_granularity post-rename).
5. The skill must trigger on correct conditions: agents writing JAX pipelines, building domain libraries on xtrax, or analyzing batching plans via EDA.
6. Code examples must be verified-working patterns from tests, not invented.

Negotiable:
- Structure (sections, ordering, depth of coverage per area)
- Whether to split into a single file or a multi-section skill with sub-skill pointers
- Whether gap documentation lives inline or in a separate "tech-debt" section
- Level of detail on distributed/checkpoint (less-explored surfaces)
- Whether to include a "how to extend xtrax" section for domain library authors
- Whether EDA gets its own sub-skill or is folded inline

## Idea Pool
- [user] PEGS decomposition:
- [user] Processes: batching plan construction (AxisSpec → BatchPlanner → BatchPlan → strategy selection), axis dispatch (BatchPlan → make_axis_dispatch → iterator), domain library construction (RunSpec subclass + InputResolver + StageBundle + AxisBoundary), training step execution (ResumableState → Trainer.step → new state), EDA analysis (BatchPlan → extract_plan_stats → render), sparsification workflow (model → sparsify_model → sparse forward fn → sparse_filter_jit), carry threading (CarrySpec → Scan strategy → JaxScanIterator → carry-bearing iteration), deduplication (dedup_eligible=True + DedupSpec → DedupGather strategy)
- [user] Events: AxisSpec construction (with or without bucket_boundaries/dedup_eligible), BatchPlanner.plan() trigger (reads CarrySpec/DedupSpec in Phase 0, then applies priority rules), make_axis_dispatch() call (converts strategy to typed iterator, rejects DedupGather and Bucket), Fuse/Tap/Sink invocation (post-stacking, at each step, terminal), Engine.fit() / fit_sync() (async loop, callback chain), sparsify_model() call (must be outside jit — trace guard enforced), render() call (requires xtrax[eda] extras)
- [user] Goals: write a domain library (define RunSpec subclass, InputResolver, StageBundle subclass, AxisBoundary per axis); run a tiled inference loop without recompilation; track training state resumably; analyze a batching plan before committing to it; apply structured sparsity at inference time; visualize why the planner chose each strategy
- [user] States: BatchPlan (frozen tuple of AxisDecisions), ResumableState (step + key + model + opt_state, immutable eqx.Module), SparsePolicy (mask state, Python-side mutable), AxisBoundary (static fuse/tap/sink callables, no dynamic leaves)
- [user] Generating idea dimensions using SCAMPER + assumption-reversal for skill structure:
- [user] SCAMPER on skill structure:
- [user] Substitute: Instead of one monolithic skill file, split into a trigger-gated multi-section format (tiling section, training section, EDA section) that each has its own trigger conditions
- [user] Combine: Merge the gap catalog and the tech-debt section so gaps are co-located with the feature they relate to rather than appended at end
- [user] Adapt: Borrow the "Quick Reference Index" pattern from using-naurmalade spec (COMPOSITE A winner there) — task-to-section anchors so agent can load once and navigate without re-reading
- [user] Modify: Add "What NOT to do" callout boxes at each major API surface (e.g. "Do not call sparsify_model inside jit", "Do not use deprecated .batch_size")
- [user] Put to other use: The PlanStatsDict output of explain_plan can serve as an agent self-audit tool — add an "EDA as planning audit" workflow section
- [user] Eliminate: Drop deep coverage of distributed/checkpoint since those surfaces are less-explored; use pointer pattern (reference to orbax docs)
- [user] Reverse: Structure the skill "decision-first": start with the decision tree (which xtrax primitive for which problem) rather than bottom-up (here is every class)
- [user] Assumption-reversal:
- [user] Assumption: "Agents should understand all of xtrax before writing code" → Reverse: "Agents should start with a single working pattern (AxisSpec → BatchPlanner → plan) and expand from there"
- [user] Assumption: "Gap catalog belongs in a separate section" → Reverse: "Inline gap callouts are more actionable than an appendix"
- [user] Assumption: "The skill covers all API surfaces equally" → Reverse: "Weight coverage by actual consumer frequency: tiling 40%, training 25%, run layer 20%, EDA 10%, sparse/distributed/checkpoint 5% each"
- [user] Assumption: "HiTL hooks are optional" → Reverse: "Make HiTL investigation callouts first-class so the agent knows exactly when to pause and ask the human"
- [user] Competing factions:
- [user] Faction A (Essentials-First): tight trigger block → 10-line pre-flight checklist → quick reference index → deep reference by section; borrows from naurmalade composite winner
- [user] Faction B (Decision-Tree-First): open with "which primitive for which problem" flowchart in prose → then per-primitive reference → then composition patterns; better for agents starting cold
- [user] Faction C (Workflow-First): organize by workflow (build a domain library / run a tiled inference / analyze a plan / train a model) rather than by module; most aligned with agent task orientation
- [user] Faction D (Hybrid): tier-1 (trigger + pre-flight + workflow index) → tier-2 (per-module deep reference) → tier-3 (gaps + HiTL hooks); all three previous factions as layers
- [user] Two more dimensions before converging:
- [user] Pure-functional composition patterns (vision vs. reality):
- [user] Current state: xtrax has no compose() or pipeline() operator by design; domain libraries manually wire RunSpec + InputResolver + StageBundle + Fuse/Tap/Sink
- [user] What the vision requires: agents need to understand the two-layer composition model: (1) Python-level topology (which stages are active, which AxisBoundary applies) via StageBundle.active_stages() and AxisBoundary; (2) JAX-level execution (iterator maps fn over axis, Fuse post-stacks, Tap/Sink fire via io_callback)
- [user] Gap: make_inference_plan validator referenced in AxisBoundary docstring does not exist in src — topology validation is absent from current implementation
- [user] Gap: no explicit rule about whether Fuse runs inside or outside jit — the docstring says "pure JAX" but the boundary between static fields and dynamic execution needs explicit documentation
- [user] HiTL investigation callouts (where human judgment is required):
- [user] When DedupSpec is first set up: k_bucket choice and the large-k regime TODO require judgment
- [user] When carry shape is non-obvious: init shape in CarrySpec must be static at trace time — agents may need human review to verify
- [user] When topology validator is absent: make_inference_plan does not exist, so ordered tap/sink + Vmap conflicts are not caught at plan-build — runtime errors only
- [user] When bucket_boundaries are chosen: boundary choice affects recompilation frequency vs. padding waste trade-off
- [user] Ready to converge.

## Decision Log
- [DEFER] Faction D (Hybrid): three-tier structure with tier-1 trigger/pre-flight/workflow-index: [CRITIC] idea=faction-d-hybrid lens=feasibility finding=The three-tier structure has no defined loading contract — there is no mechanism guaranteeing an agent receives all three tiers before invoking tier-2 anchors, and a partial load (tier-1 only) leaves the agent with a navigation index pointing at content it cannot see, which is actively worse than no index. severity=FATAL
- [DEFER] Faction D tier-1 workflow navigation anchors pointing into tier-2 module deep reference: [CRITIC] idea=faction-d-tier1-anchors lens=edge_cases finding=When xtrax updates and renames an API (as it just did with AxisSpec.batch_size → default_batch_size and granularity → tile_granularity in v0.3.0), tier-2 module references go stale but tier-1 navigation anchors continue to look syntactically valid — an agent will follow the anchor, find plausible-seeming but wrong content, and not know to distrust it; the skill becomes a confident misinformation source. severity=FATAL
- [REJECT] Task-oriented workflow index as navigation layer for domain library authors: [CRITIC] idea=task-oriented-index lens=user_impact finding=A domain library author building on xtrax (RunSpec subclass, InputResolver, StageBundle, AxisBoundary) does not navigate by task — they need cross-cutting invariants: PyTree discipline, which fields must be static at trace time, eqx.Module structural requirements, and JIT boundary rules; task-oriented indexing buries these under workflow headings where they cannot be found until the author hits a trace-time error, at which point the skill has already failed. severity=MAJOR
- [DEFER] What NOT to do callout boxes at each major API surface: [CRITIC] idea=what-not-to-do-callouts lens=implementation_cost finding=Negative-contract callouts (e.g. "do not call sparsify_model inside jit") require ongoing sync with actual behavior: they are the first to become false when implementation changes (the jit trace guard could be lifted, the deprecation shim could be removed), they have no automated test coverage, and no owner is identified in the design — making them high-maintenance content that will silently rot and produce harder-to-debug agent errors than having no callout at all. severity=MAJOR
- [DEFER] Faction D exported as a shared skill dependency for multiple agents: [CRITIC] idea=faction-d-exportable-dependency lens=reversibility finding=Once Faction D is exported and consumed by multiple agents, refactoring the tier structure — which will be necessary when xtrax v0.4 introduces breaking changes — requires simultaneously updating every consumer's load contract; there is no versioning mechanism proposed for the skill itself, so a breaking xtrax change forces either a flag-day migration across all agents or maintaining a diverged skill file per version, both of which are more costly than the three-tier structure saves. severity=MAJOR
- [REJECT] Faction A (Essentials-First) — tight trigger + checklist + deep reference: Too module-oriented; agents starting with a task ("build a domain library") must manually cross-reference sections rather than following a task-oriented path. Navigation becomes "find the module" not "find the task."
- [MERGE] Faction B (Decision-Tree-First) — flowchart in prose → per-primitive reference → patterns: The "which primitive for which problem" flowchart is a strong component — merge it into tier-1 of Faction D as the primary navigation mechanism. Works better as a section than a standalone approach.
- [MERGE] Faction C (Workflow-First) — organized by workflow, not module: Workflow-orientation is correct for agent task framing. Merged into Faction D via task-oriented navigation anchors in tier-1. Pure workflow-first without a reference tier leaves domain library authors without cross-cutting concern coverage.
- [REJECT] Gap catalog as appendix: Inline gap callouts co-located with the feature are more actionable — the agent encounters the gap at the point of use, not after reading the full doc. Appendix pattern puts gaps out of sight.
- [DEFER] Deep coverage of distributed/checkpoint surfaces: These surfaces are less-explored in the current codebase and less-frequently used by domain library authors. Pointer pattern (reference to orbax docs for checkpoint, brief API surface listing for distributed) is appropriate. Full coverage deferred to a separate skill or future revision.
- [REJECT] EDA as a separate sub-skill: EDA is lightweight enough (explain_plan + render) to fold inline. The key workflow — "use explain_plan as a planning audit before committing to a batching strategy" — is a first-class workflow that belongs in tier-1. Full EDA reference folds into tier-2.
- [ACCEPT] Coverage weighting: tiling 40%, training 25%, run layer 20%, EDA 10%, sparse/dist/checkpoint 5% each: Reflects actual consumer frequency based on codebase analysis. Tiling and composition are the unique value of xtrax — training patterns are better documented by Equinox/Optax directly. Run layer is the integration seam for domain libraries, so 20% is appropriate.
- [ACCEPT] HiTL investigation callouts as first-class annotations: Four identified HiTL gates: (1) DedupSpec k_bucket choice at k>256, (2) CarrySpec init shape static-at-trace-time verification, (3) topology validator absent — runtime-only detection of ordered tap/sink + Vmap conflicts, (4) bucket_boundaries recompilation vs. padding-waste tradeoff. These require human judgment and should be explicit STOP points in the skill.
- [REJECT] Three-tier structure with no minimum-loadable unit (FATAL from critic): FATAL: tier-1-only loads create navigation anchors pointing at invisible tier-2 content — actively worse than no skill. Resolution: redefine tier-1 as a complete, self-contained standalone. Tier-1 must include the full pre-flight checklist, decision-tree for primitive selection, and enough to complete the most common call (AxisSpec → BatchPlanner → plan) without tier-2. Tier-2 is a depth-extension, not required for basic operation.
- [REJECT] High-trust navigation anchors pointing at low-trust tier-2 content (confident-staleness FATAL): FATAL: tier-1 anchors survive API renames while tier-2 content goes stale, producing confident misinformation. Resolution: (a) tier-1 anchors must describe behavioral invariants enforced by code (trace guards, ValueError) rather than API surface details; (b) all code examples must cite file:line — source is authoritative, skill is a pointer; (c) explicit skill version header (xtrax v0.3.0) + compatibility assertion (assert xtrax.__version__.startswith("0.3")) in pre-flight.
- [MERGE] Task-oriented workflow index that buries cross-cutting concerns for domain library authors (MAJOR): MAJOR resolved: add a "JAX discipline for domain library authors" section as a mandatory first-read in tier-1, parallel to the workflow index. Covers: static vs dynamic eqx.Module fields, PyTree invariants for AxisBoundary (all-static, empty leaves), JIT boundary rules (sparsify_model outside jit, Fuse inside, Tap/Sink via io_callback), eqx.filter_jit vs jax.jit. This is a non-workflow, invariant-organized section alongside the workflow index — not replacing it.
- [MERGE] What NOT to do callouts with no enforcement owner (MAJOR): MAJOR resolved: restrict callouts to enforcement-backed contracts only. Each callout must cite: (1) the enforcement mechanism (e.g., RuntimeError from assert_not_tracing, ValueError from AxisSpec.__post_init__, DeprecationWarning from __getattr__ shim), and (2) the file:line where the enforcement lives. Soft conventions excluded. When enforcement is removed in a future version, the file:line reference goes stale in a visible, discoverable way.
- [MERGE] No skill versioning for exported multi-agent dependency (MAJOR): MAJOR resolved: add explicit version frontmatter (xtrax_version: 0.3.0) and a compatibility assertion in the pre-flight checklist. Skill exports include the version tag so consuming agents know what version they were written for. This does not eliminate flag-day migrations but makes them visible rather than silent.

## Assumptions

## TBDs

## Pre-mortem Record
**User:** Pre-mortem: The skill shipped and worked well for 3 months. Then xtrax v0.4.0 was released. The DeprecationWarning shim for .batch_size was removed (as documented as planned). Several consuming agents had been loading tier-1 only (context pressure) and were running the minimal working pattern correctly. But one agent doing DedupSpec work had internalized the k_bucket large-k TODO note and was successfully routing to HiTL for those cases.

The failure: the tier-2 tiling section still described the old DedupGather constructor signature after k_bucket changed from powers-of-2 to geometric buckets. An agent loaded tier-2 (full load, rare but it happened), constructed DedupGather with the old power-of-2 k_bucket formula, got wrong results silently (no ValueError, just suboptimal bucketing). The file:line anchor in the skill pointed to tiling/dedup.py:29 — but the resolution at that line changed without the skill being updated. The agent trusted the anchor, read the stale skill text rather than the live source, and did not re-read the source because the anchor looked authoritative.

What specifically went wrong: agents were trained to trust file:line anchors as pointers to authoritative source, but some agents (not all) read the skill text alongside the anchor rather than navigating to the source. The skill had become the authority rather than the pointer.

Mitigation built in: the pre-flight compatibility assertion catches major version bumps. For this case, the mitigation is to add explicit "READ THE SOURCE AT THIS LOCATION" language to every anchor — not "as documented at tiling/dedup.py:29" but "verify current behavior at tiling/dedup.py:29 — this skill is a map, not the territory."
**AI:** _not recorded_

## Acceptance Criteria
**Given** Fixed constraints:
1. All content must be verified against actual code — no inferring behavior from names or specs that may have diverged from implementation.
2. The skill must be exportable (usable outside the xtrax project context), so it cannot rely on local CLAUDE.md or project-level context.
3. Accuracy over completeness — flag gaps and TBDs explicitly rather than filling them with guesses.
4. The skill covers xtrax v0.3.0 (AxisSpec.default_batch_size / tile_granularity post-rename).
5. The skill must trigger on correct conditions: agents writing JAX pipelines, building domain libraries on xtrax, or analyzing batching plans via EDA.
6. Code examples must be verified-working patterns from tests, not invented.

Negotiable:
- Structure (sections, ordering, depth of coverage per area)
- Whether to split into a single file or a multi-section skill with sub-skill pointers
- Whether gap documentation lives inline or in a separate "tech-debt" section
- Level of detail on distributed/checkpoint (less-explored surfaces)
- Whether to include a "how to extend xtrax" section for domain library authors
- Whether EDA gets its own sub-skill or is folded inline
**When** implementing Faction D (Hybrid) — critic-revised: self-contained tier-1 + versioned anchors + cross-cutting invariants section.

Structure:
- TIER-1 (self-contained mandatory read, must function without tier-2):
  (a) Skill frontmatter: xtrax_version: 0.3.0, trigger conditions, pre-flight compatibility assertion
  (b) Pre-flight checklist: version check, import verification, extras check for EDA
  (c) JAX discipline for domain library authors: static vs dynamic fields, PyTree invariants for AxisBoundary (all-static, empty leaves), JIT boundaries (sparsify_model outside, Fuse inside, Tap/Sink via io_callback), eqx.filter_jit vs jax.jit
  (d) Primitive → problem decision tree: which xtrax primitive for which task (flowchart in prose, merged from Faction B)
  (e) Workflow index: task-oriented anchors to tier-2 sections (build domain library / tiled inference / training loop / EDA planning audit / sparsification)
  (f) Minimal working pattern: AxisSpec → BatchPlanner → BatchPlan → make_axis_dispatch → iterator (fully self-contained, no tier-2 needed)

- TIER-2 (deep reference, organized by workflow proximity, not module hierarchy):
  (a) Tiling layer (40%): AxisSpec, BatchPlanner, strategies, dispatch, iterators, CarrySpec, DedupSpec, bucketize
  (b) Run layer (20%): RunSpec, InputResolver, RuntimeBundle, FeatureBatch, SinkSpec, AxisBoundary, Fuse/Tap/Sink
  (c) Training layer (25%): ResumableState, Trainer, SafetyTrainStep, Engine, callbacks, Optax integration
  (d) EDA (10%): extract_plan_stats, explain_plan, render, plan_to_dataframe — including EDA-as-planning-audit workflow
  (e) Sparse/Distributed/Checkpoint (5%): pointer pattern + brief API surface

- TIER-3 (inline with tier-2, not appended): 
  Each section carries inline gap callouts (⚠ GAP), HiTL investigation stops (🔬 HiTL), and enforcement-backed NOT callouts (🚫 citing file:line of enforcement). No soft-convention callouts.

Key design constraints resolved from critic:
- Tier-1 is fully self-contained: no dangling anchors, navigation usable without tier-2
- All code examples cite file:line (source is authoritative; skill is a pointer)  
- Version frontmatter + compatibility assertion in pre-flight
- "What NOT to do" callouts restricted to enforcement-backed contracts with file:line citation
- JAX discipline section serves domain library authors who don't navigate by workflow
- HiTL stops: (1) DedupSpec k_bucket at k>256, (2) CarrySpec init static-shape verification, (3) topology validator absent (ordered tap/sink + Vmap: runtime-only detection), (4) bucket_boundaries recompilation vs. padding-waste tradeoff
**Then**
  - [ ] AC-1: Loading tier-1 in isolation, an agent can complete the minimal working pattern (`AxisSpec → BatchPlanner → BatchPlan → make_axis_dispatch → iterator`) without dereferencing any anchor into tier-2. No tier-2 symbol is required to reach a constructed iterator object.
  - [ ] AC-2: The skill file carries `xtrax_version: 0.3.0` frontmatter and a pre-flight compatibility assertion (`assert xtrax.__version__ == "0.3.0"`) with a note that the assertion is blind to forks; an additional source-verify step reads `src/xtrax/__init__.py:1`.
  - [ ] AC-3: Every code example in tier-2 carries a `# verify: <file>:<line>` annotation. The annotation points to the live source, not to the skill as authority. Annotation text reads "verify current behavior at <path> — this skill is a map, not the territory."
  - [ ] AC-4: Every `🚫 HALTS` callout cites a halting enforcement mechanism (RuntimeError or ValueError) with its file:line. Every `⚠ WARN` callout cites a non-halting mechanism (DeprecationWarning) with its file:line. The two classes are never mixed under a single 🚫 symbol.
  - [ ] AC-5: The `.batch_size` deprecation callout is scoped to `AxisSpec` explicitly. It does not apply to `AxisDecision.batch_size`, `SafeMap.batch_size`, `AxisStatsEntry["batch_size"]`, or `safe_map(batch_size=...)` — all of which remain live fields in xtrax v0.3.0.
  - [ ] AC-6: All four HiTL investigation stops define the concrete agent action: pause execution, present the specific question to the user (question text included in the stop), and do not proceed until the human confirms. Stop text does not use the bare verb "stop" without an action clause.
  - [ ] AC-7: The skill deliverable is a SKILL.md at `~/.claude/skills/using-xtrax/SKILL.md` with YAML frontmatter (`name`, `description`, `triggers`, `xtrax_version`) following the jaxlint plugin export convention (see `~/.claude/skills/exporting-jax/SKILL.md` for format reference).

## Amendments Applied (post-adversarial-review)

### C6 — `.batch_size` deprecation scoping (MAJOR, challenger-verified)
`batch_size` is deprecated **only on `AxisSpec`** (shim at `src/xtrax/tiling/plan.py:76-83`). It remains a live, correct field on:
- `AxisDecision.batch_size` (`src/xtrax/tiling/plan.py:101`)
- `SafeMap.batch_size` (`src/xtrax/tiling/strategy.py:53`)
- `AxisStatsEntry["batch_size"]` (`src/xtrax/eda/types.py:27`)
- `safe_map(batch_size=...)` parameter (`src/xtrax/transforms/map.py:8`)
A blanket "do not use `.batch_size`" would be misinformation. Callouts must be type-scoped to `AxisSpec` only.

### C5 — Enforcement taxonomy split (MAJOR)
Revised two-class taxonomy for 🚫/⚠ callouts:
- **🚫 HALTS** — execution stops. Sources: `RuntimeError` from `assert_not_tracing` (`src/xtrax/sparse/inference.py:44`), `ValueError` from `AxisSpec.__post_init__` (`src/xtrax/tiling/plan.py:51-74`), `DispatchRejected` from `make_axis_dispatch` (`src/xtrax/tiling/dispatch.py:78`).
- **⚠ WARN** — execution continues with signal. Source: `DeprecationWarning` from `AxisSpec.__getattr__` shim (`src/xtrax/tiling/plan.py:76-91`).
Both are in-scope but agent risk model differs — HALTS is mandatory stop, WARN is advisory.

### C7 — HiTL action definition (MAJOR)
Each HiTL stop includes: (1) trigger, (2) specific question to user, (3) consequence of proceeding without confirmation. Four stops with full text:
1. **DedupSpec k_bucket at k>256**: Trigger: `k > 256`. Question: "k={k} exceeds 256 — power-of-2 bucketing wastes up to 2× compute here (see `src/xtrax/tiling/dedup.py:29` TODO). Proceed with powers-of-2 or define custom bucket boundaries?" Block until confirmed.
2. **CarrySpec init static shape**: Trigger: `CarrySpec.init` contains shapes not known at Python-side. Question: "Verify `init` shape is static at JAX trace time before `BatchPlanner.plan()`. Dynamic shapes fail at `jax.lax.scan` compilation. Is this shape static: {shape}?" Block until confirmed.
3. **Topology validator absent**: Trigger: `AxisBoundary(tap=..., ordered=True)` or `AxisBoundary(sink=..., ordered=True)` paired with `Vmap` strategy. Question: "`make_inference_plan` validator (referenced at `src/xtrax/stages/boundaries.py:79`) does not exist in src. Ordered tap/sink + Vmap conflict will only be caught at runtime. Manually verify topology before running?" Block until confirmed.
4. **bucket_boundaries tradeoff**: Trigger: setting `bucket_boundaries` on an AxisSpec. Question: "Boundaries {boundaries} → {n_buckets} compiled XLA programs. Each adds compilation latency; each gap adds padding waste. Review tradeoff and confirm boundaries?" Block until confirmed.

### C3 — Version assertion strengthening (MAJOR)
Replace `startswith("0.3")` with `== "0.3.0"` (exact). Add: read `src/xtrax/__init__.py:1` to verify `__version__ = "0.3.0"` in the live source tree — catches forks maintaining the same version string without a code bump.

### C2 — Export deliverable path resolved (BLOCKING → resolved)
Deliverable: `~/.claude/skills/using-xtrax/SKILL.md`. Format: jaxlint convention at `~/.claude/skills/exporting-jax/SKILL.md` is the template reference.
