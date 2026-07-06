# xtrax Internal Docs

## Specs
- [260706_joint-budget-batch-planner](specs/260706_joint-budget-batch-planner.md) — joint-budget mode for BatchPlanner (MemoryBudget greedy demotion, order-as-priority contract, fail-loud infeasibility, native memory_analysis/memory_stats estimator hooks); 13 ACs; motivating consumer: aminx planner migration
- [260702_design-the-2181-agentic-algorithm-evolut](specs/260702_design-the-2181-agentic-algorithm-evolut.md) — #2181 autoresearch loop spec (contemplex 129ecc73, INVEST PASS): evaluator-first ratchet MVP, sealed SHA-closure evaluator monopoly, pure-xtrax + bathos-MCP dispatch, bathos owns campaign_edges, 30 ACs (walking skeleton {E2,7,8,9,11,13,14}), island = evidence-gated Phase-2
- [260702_design-2174-next-slices-minimal-composit](specs/260702_design-2174-next-slices-minimal-composit.md) — #2174 substrate spec (contemplex 87f306f8, INVEST PASS): executor-first minimal-breakage, two-tier boundary executor, name-keying adapter (RunSpec.boundaries stays list), core-sealed EvaluateFn seam, IR inference-layer-owned, generate-then-validate + benchmark gate; 11 ACs, 2 slices
- [260702_xtrax-workflows-as-a-praxia-plugin-packa](specs/260702_xtrax-workflows-as-a-praxia-plugin-packa.md) — workflows-as-plugin spec (contemplex 8ded6692, INVEST PASS S-overridden): S-C phased dual-track — phase-1 xtrax packaging + Claude-PCW, phase-2 praxia dispatch gated on MCP-reachability probe; 15 ACs partitioned xtrax/praxia/probes; TTL+probe human-gate freshness
- [260623_e3-run-verb](specs/260623_e3-run-verb-trainconfig-driven-cli.md) — E3-MVP `xtrax run` spec (contemplex architectural brainstorm f300363c): winner A+H+I+J+M — `[data]` factory returns a dataset run always wraps (no duck-type branch); always-write `.xtrax/runs/<id>/manifest.json` (non-optional model-path, run-id=config-hash+uuid-fallback, checkpoint_dir independent of run-id); `TrainConfig` orthogonal to E1 `RunSpec`; raw-TOML + mandatory schema_version; `init_state` promoted to `xtrax.training`; 11 ACs
- [260623_e2-auto-cli](specs/260623_e2-auto-cli-for-xtrax-how-to-scope-and-d.md) — E2-MVP auto-CLI spec (contemplex architectural brainstorm c519ff24): C-AMENDED progressive scaffold — tyro.extras.subcommand_cli_from_dict + plan/explain verbs + lazy import-path loader (lesson #145); export/run/sweep/resume deferred (roadmap); 8 ACs
- [260623_e1-signature-inference-core](specs/260623_e1-signature-inference-core-how-should-x.md) — E1-MVP spec (contemplex architectural brainstorm, session 4dfc27aa): derive BundleSchema + list[AxisSpec] from a typed pure fn via eval_shape; fail-loud UNKNOWN roles (not batch); @axis_config Tier-1; 8 ACs; Tier-2 jaxtyping/codegen/CarrySpec deferred (TBDs)
- [260615_xtrax-eda-api-revised](specs/260615_xtrax-eda-api-revised.md) — EDA visualization API spec (post-critic + adversarial review): render(), PlanStatsDict, PlanLogger, PanelName, 12 acceptance criteria + 7 amendments
- [260615_using-xtrax-skill](specs/260615_design-the-using-xtrax-skill-an-exportab.md) — using-xtrax exportable skill spec (post-adversarial-review): Faction D Hybrid, 7 ACs, 5 amendments, 4 HiTL stops, enforcement taxonomy split
- [260604_xtrax-spec](specs/260604_xtrax-spec.md) — Full xtrax v0.2.0 specification (Phases 0–7, Sprints 1–4)
- [260608_xtrax-s5-sparse](specs/260608_xtrax-s5-sparse.md) — Sprint 5: coverage completion, benchmarks, sparse infrastructure (Phase 8–10)
- [260608_inference-time-sparsification](specs/260608_inference-time-sparsification-in-xtrax-h.md) — Sprint 7: sparsify_model functional API, BucketIterator minimal impl, jit-trace guard (inference-time sparsification)

## Plans
- [260623_e3-run-verb-backlog-dag](plans/260623_e3-run-verb-backlog-dag.md) — E3-MVP `xtrax run` DAG (staff): 11 tasks T0–T9 + Z, single L keystone T7 `run_from_config`; corrects two under-specified anchors (`fit_sync` is `Engine` method; `DataModule` arity), maps all 4 pre-mortem invariants to tests, 3 judgment flags for Cursor hand-off
- [260623_e2-mvp-backlog-dag](plans/260623_e2-mvp-backlog-dag.md) — E2-MVP xtrax.cli DAG (staff → adversarial plan-audit NEEDS_WORK, 7 fixes folded): 9 tasks T0–T8 + roadmap node; caught the import-linter-lazy-import trap + AC4-unreachable-without-decorated-fixture; critical path through the explain+emit long pole
- [260623_e1-mvp-backlog-dag](plans/260623_e1-mvp-backlog-dag.md) — E1-MVP implementation DAG (staff → adversarial plan-audit NEEDS_WORK, all 8 fixes folded in): 12 tasks incl. split E1.3a/E1.3b keystone, w1.5 decision gate (explicit AxisRole field; in-test InputResolver adapter), AC2/AC8 ownership fixes
- [260615_using-xtrax-skill-backlog-dag](plans/260615_using-xtrax-skill-backlog-dag.md) — Backlog DAG for the using-xtrax exportable skill: 5 tier-1 tasks, 5 tier-2 tasks, 5 tech-debt HiTL items, sprint sequencing
## Roadmaps
- [research-epics/260702_00-mandate](roadmaps/research-epics/260702_00-mandate.md) — roadmap mandate: 0.3.0 baseline, locked decisions, T1/T2/T3 dependency spine, 7 HITL gates, cross-repo edges (bathos ×9, praxia ×6)
- [research-epics/260702_01-dag-2174-substrate](roadmaps/research-epics/260702_01-dag-2174-substrate.md) — T1 backlog DAG: 14 items, 2 slices, 7 marked #2181 entry-edge items, all 11 spec ACs mapped
- [research-epics/260702_02-dag-2181-autoresearch](roadmaps/research-epics/260702_02-dag-2181-autoresearch.md) — T2 backlog DAG: 33 xtrax items (P0-P4 + 5 human gates + CC invariants) + 9 bathos cross-repo items; walking skeleton unblocked from bathos lane
- [research-epics/260702_03-dag-plugin-workflows](roadmaps/research-epics/260702_03-dag-plugin-workflows.md) — T3 backlog DAG: 5 phase-1 xtrax + 3 gating probes + 6 praxia cross-repo edges; strict-mode registry double-gated
- [composition/260622_e1-signature-inference-epic](roadmaps/composition/260622_e1-signature-inference-epic.md) — E1 keystone epic: derive Bundle pytree + AxisSpec roles from typed pure JAX functions (eval_shape inference + jaxtyping dim-names → roles; Tyro-backed CLI). Grounded, pre-brainstorm.

## Handoffs
- [260623_e1-e2-composition-layer](handoffs/260623_e1-e2-composition-layer.md) — session handoff: E1-MVP (xtrax.inference) + #2561 hardening + E2-MVP (xtrax.cli plan/explain/export) all merged to main (341 tests); deferred run/sweep/resume verbs + push pending

## Audits
## Research
- [260706_xtrax-assessment-tiling-vs-xla-jit](research/260706_xtrax-assessment-tiling-vs-xla-jit.md) — niche assessment: why tiling lives above the trace boundary (jit = shape-specialized AOT, not adaptive JIT; static buffer assignment; recompile-penalized adaptation), differentiated (tiling/inference/eda) vs commodity (trainer/checkpoint/distributed) split, graveyard/MaxText economics, absorption watch list (`lax.map(batch_size=)`, ragged shapes)
- [260702_roadmap-research-synthesis](research/260702_roadmap-research-synthesis.md) — roadmap-cycle research synthesis (task 260702_research-roadmap-dags): 6 adversarially-verified themes → #2181 autoresearch ratchet-loop architecture + gate catalog + bathos capability map, #2174 minimal composition substrate child item (D1–D4, AC1–AC6), neuro-symbolic placement (grounding node in #2174, entry criterion for #2181), praxia plugin contract + rig-run dispatch gaps, cross-cutting gate-design template; brainstorm fork list + dropped-claims appendix

## Misc

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
