---
title: using-xtrax skill — Backlog DAG
task_id: 260615_using-xtrax-skill
spec: .praxia/docs/specs/260615_design-the-using-xtrax-skill-an-exportab.md
date: 260615
status: draft
---

# Backlog DAG: using-xtrax Skill

## Delivery target
`~/.claude/skills/using-xtrax/SKILL.md` — exportable, self-contained, xtrax v0.3.0.

---

## Phase 1 — Tier-1 Core (self-contained, blocks everything else)

### T1-A: Scaffold SKILL.md with frontmatter and pre-flight
**Input:** spec AC-2, AC-7  
**Output:** `~/.claude/skills/using-xtrax/SKILL.md` with YAML frontmatter (`name`, `description`, `triggers`, `xtrax_version: 0.3.0`) and pre-flight block (version exact-match assertion + source-verify step for `src/xtrax/__init__.py:1`).  
**Depends on:** nothing (first task)  
**Gate:** frontmatter parses as valid YAML; `assert xtrax.__version__ == "0.3.0"` code snippet is present and correct.

### T1-B: JAX discipline section (domain library author mandatory read)
**Input:** verified source at `src/xtrax/stages/boundaries.py`, `src/xtrax/sparse/inference.py`, `src/xtrax/tiling/plan.py`  
**Output:** Section in tier-1 covering: static vs dynamic `eqx.Module` fields; `AxisBoundary` PyTree invariant (all-static, empty leaves — verify `jax.tree_util.tree_flatten(AxisBoundary()).leaves == []`); JIT boundary rules (🚫 HALTS: `sparsify_model` outside jit enforced at `inference.py:44`; Fuse inside jit; Tap/Sink via `io_callback`); `eqx.filter_jit` vs `jax.jit`.  
**Depends on:** T1-A (file must exist)  
**Gate:** AC-4 enforced — each 🚫 HALTS and ⚠ WARN carries its file:line citation; no DeprecationWarning appears under 🚫 HALTS.

### T1-C: Primitive-to-problem decision tree
**Input:** `BatchPlanner` priority rules (spec: `plan.py:123+`), strategy sealed union (`strategy.py:109`)  
**Output:** Prose flowchart in tier-1: "If axis needs variable-length bucketing → Bucket. If axis has repeated elements → DedupGather. If cardinality ≤ default_batch_size → Vmap. If cardinality > batch_size AND divisible → SafeMap. Else → SafeMap + deferred-failure warning."  
**Depends on:** T1-A  
**Gate:** Each branch is source-annotated with `# verify: tiling/plan.py:<line>`.

### T1-D: Minimal working pattern (fully self-contained)
**Input:** Tests at `tests/tiling/test_plan.py`, `tests/tiling/test_dispatch.py`  
**Output:** A complete, runnable code snippet in tier-1 showing `AxisSpec → BatchPlanner.plan() → BatchPlan → make_axis_dispatch() → VmapIterator`. Zero tier-2 references required to understand it.  
**Depends on:** T1-A  
**Gate:** AC-1 — the pattern uses only `xtrax.tiling` symbols, all importable from `xtrax.tiling.__init__`; no `xtrax.run` or `xtrax.stages` imports in the tier-1 pattern.

### T1-E: Workflow index (task-oriented anchors to tier-2)
**Input:** Five workflows: build domain library / tiled inference / training loop / EDA planning audit / sparsification  
**Output:** Indexed list in tier-1 with anchor text (not module names) linking to tier-2 sections.  
**Depends on:** T1-A, T2 sections must exist  
**Gate:** Each anchor resolves to a real tier-2 section heading; no dead links.

---

## Phase 2 — Tier-2 Deep Reference (40/25/20/10/5% weighting)

### T2-A: Tiling layer (40% of tier-2 depth)
Covers: `AxisSpec` (all fields, deprecation scoping per AC-5), `BatchPlanner` (selection rules, memory_estimator, carry_specs, dedup_specs), strategies (`Vmap/SafeMap/Scan/DedupGather/Bucket`), `make_axis_dispatch` (rejection rules for DedupGather/Bucket), iterators (`VmapIterator/SafeMapIterator/JaxScanIterator/BucketIterator`), `CarrySpec`, `DedupSpec/get_k_bucket`, `select_bucket/bucketize`.  
**Inline annotations (tier-3):**
- ⚠ WARN `.batch_size` on AxisSpec only (`plan.py:76-83`, `DeprecationWarning`)
- 🚫 HALTS empty `bucket_boundaries` (`plan.py:51-74`, `ValueError`)
- 🚫 HALTS DedupGather passed to `make_axis_dispatch` (`dispatch.py:78`, `DispatchRejected`)
- 🔬 HiTL: DedupSpec k>256 (full text from spec C7 amendment)
- 🔬 HiTL: CarrySpec init static shape (full text from spec C7 amendment)
- 🔬 HiTL: bucket_boundaries tradeoff (full text from spec C7 amendment)
- ⚠ GAP: DedupGather large-k regime — powers-of-2 bucketing wastes up to 2× at k>256 (`tiling/dedup.py:29` TODO; no fix in v0.3.0)

**Gate:** AC-3 (all code examples carry `# verify: <file>:<line>`), AC-5 (`.batch_size` scoped to AxisSpec).

### T2-B: Run layer (20% of tier-2 depth)
Covers: `RunSpec` (eqx.Module, fields, `from_spec()` identity), `InputResolver` (Protocol, singledispatch convention), `RuntimeBundle` (iterator union + model), `FeatureBatch` (NewType), `SinkSpec` (formats: jsonl/h5/none), `AxisBoundary` (all-static, empty PyTree leaves), `Fuse/Tap/Sink` protocols.  
**Inline annotations:**
- ⚠ GAP: `xtrax.run` not in top-level `xtrax.__init__` exports — import as `from xtrax.run import RunSpec` not `from xtrax import RunSpec`
- ⚠ GAP: `AxisBoundary/Fuse/Tap/Sink` not in top-level exports — import as `from xtrax.stages import ...`
- ⚠ GAP: `make_inference_plan` validator referenced in `boundaries.py:79` does not exist in src; topology validation is runtime-only
- 🔬 HiTL: ordered tap/sink + Vmap conflict (full text from spec C7 amendment)
- 🚫 HALTS: (none in this layer — no halting enforcement on run layer in v0.3.0)

**Gate:** AC-3, AC-4.

### T2-C: Training layer (25% of tier-2 depth)
Covers: `ResumableState` (fields, immutability, `eqx.tree_at` update pattern), `Trainer` (`filter_jit` step, `filter_value_and_grad`, `apply_updates`), `SafetyTrainStep/create_train_step`, `Engine` (`async fit`, `fit_sync`, callback chain, checkpoint), `Callback` protocol (7 hooks), `LossFunction`, `WeightedLoss/MultiTaskLoss`, `make_optimizer/adamw_with_schedule`.  
**Inline annotations:**
- ⚠ NOTE: `Engine.fit` is async; use `fit_sync()` for blocking usage
- ⚠ NOTE: Callback hooks run Python-side outside JAX traces; mutating state in callbacks has no effect on training

**Gate:** AC-3.

### T2-D: EDA layer (10% of tier-2 depth)
Covers: `extract_plan_stats` (stdlib+numpy, no extras), `explain_plan` (guaranteed non-empty reasoning), `analyze_dedup/analyze_bucket`, `render` (requires `pip install xtrax[eda]`), `plan_to_dataframe`, `PlanStatsDict/PlanLogger/PanelName`. Includes EDA-as-planning-audit workflow: "call `explain_plan(plan)` before committing to a batching strategy to inspect per-axis reasoning before the first JIT compilation."  
**Inline annotations:**
- ⚠ WARN: `render()` requires `xtrax[eda]` extras; import is lazy — no ImportError at module load, fails at call time

**Gate:** AC-3.

### T2-E: Sparse/Distributed/Checkpoint (5% of tier-2 depth — pointer pattern)
Brief API surface only:
- Sparse: `sparsify_model(model, policy, leaf_filter) → eqx.Module` with BCOO leaves. Canonical: `eqx.nn.inference_mode(sparsify_model(model, policy))`. 🚫 HALTS inside jit (`inference.py:44`, `RuntimeError`).
- Distributed: `init_dist()`, `is_distributed()`, `LogicalMesh`, `with_manual_axes`. Pointer: see `src/xtrax/distributed/` for full surface.
- Checkpoint: `save_checkpoint(state, dir)`, `load_checkpoint(dir)`. Pointer: see orbax docs for full checkpoint manager API.

**Gate:** AC-3 for the sparse section (which has testable behavior); pointer sections are reference-only.

---

## Phase 3 — Tech Debt Investigation Items (HiTL, not sprint tasks)

These are NOT implementation tasks. They require human-in-the-loop investigation before any spec or sprint can be written for them.

### TD-1: make_inference_plan topology validator (missing)
**Gap:** `src/xtrax/stages/boundaries.py:79` references a `make_inference_plan` validator that does not exist in `src/`. Ordered tap/sink + Vmap conflicts are currently caught at runtime only.  
**HiTL question:** Should this be implemented? If yes, what surface — a standalone function in `stages/`, or integrated into `BatchPlanner`? Investigate and open a backlog item if confirmed.

### TD-2: Fuse inside-vs-outside-jit documentation gap
**Gap:** `Fuse` is documented as "pure JAX, no side effects" but the spec for how Fuse invocation interacts with static fields on `AxisBoundary` is not written. Is Fuse called inside the jit-traced loop or outside? The answer changes whether Fuse can use Python-level conditionals.  
**HiTL question:** Read the engine/trainer codepath and confirm where Fuse is invoked. Document the invariant in xtrax source and add it to the skill.

### TD-3: DedupSpec large-k regime (k > 256)
**Gap:** `src/xtrax/tiling/dedup.py:29` has an explicit TODO: powers-of-2 bucketing wastes up to 2× compute for k > 256.  
**HiTL question:** Prioritize? If yes, what geometric step ratio to use (1.5×? mixed strategy?), and what's the boundary between pure powers-of-2 and finer steps?

### TD-4: top-level __init__.py export gaps
**Gap:** `xtrax.run`, `xtrax.stages.Fuse/Tap/Sink/AxisBoundary`, `CarrySpec`, `DedupSpec/DedupGather` are not in the top-level `xtrax.__init__.__all__`. Domain library authors must use submodule imports.  
**HiTL question:** Intentional design (keep top-level minimal) or tech debt to address? If promoted, which symbols should go to top-level, and what's the deprecation/migration path for consumers already using submodule imports?

### TD-5: Pure-functional composition vision gap
**Gap:** No `compose()` or `pipeline()` operator exists in xtrax by design. Domain libraries (aminx, prolix) manually wire `RunSpec + InputResolver + StageBundle + AxisBoundary`. The user's vision is for xtrax to handle PyTree management of model weights, axis/batching, and IO optimally so consumers write purely functional pipelines.  
**HiTL question (brainstorming session recommended):** What would a first-class functional composition layer look like? Should xtrax provide a `make_pipeline(stages, axes, boundaries)` factory? Or is the current explicit-wiring model intentional? This is a design decision requiring a full brainstorm cycle before implementation.

---

## Sprint sequencing

```
T1-A (scaffold)
├── T1-B (JAX discipline) ──┐
├── T1-C (decision tree) ───┤
├── T1-D (minimal pattern)  ├── T1-E (workflow index) → PUBLISH
│   T2-A (tiling) ──────────┤
│   T2-B (run layer) ────────┤
│   T2-C (training) ─────────┤
│   T2-D (EDA) ──────────────┘
└── T2-E (sparse/dist/ckpt)

Tech debt items: TD-1 through TD-5 are parallel HiTL investigations
triggered after PUBLISH, not blocked by it.
```

**Sprint 1** (small-to-medium): T1-A + T1-B + T1-C + T1-D — delivers a self-contained tier-1 that satisfies AC-1 through AC-4 on its own.  
**Sprint 2** (medium): T2-A through T2-E — delivers the full deep reference. T2-A (tiling) is the largest section; T2-E (sparse/dist/ckpt) is pointer-only and trivially small.  
**Sprint 3** (small): T1-E — workflow index linking tier-1 to completed tier-2 sections. Closes the skill and triggers PUBLISH.  
**HiTL queue**: TD-1 through TD-5 routed to the ideas/debt backlog for human-scheduled investigation.
