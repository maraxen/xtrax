# xtrax Internal Docs

## Specs
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
- [composition/260622_e1-signature-inference-epic](roadmaps/composition/260622_e1-signature-inference-epic.md) — E1 keystone epic: derive Bundle pytree + AxisSpec roles from typed pure JAX functions (eval_shape inference + jaxtyping dim-names → roles; Tyro-backed CLI). Grounded, pre-brainstorm.

## Handoffs
- [260623_e1-e2-composition-layer](handoffs/260623_e1-e2-composition-layer.md) — session handoff: E1-MVP (xtrax.inference) + #2561 hardening + E2-MVP (xtrax.cli plan/explain/export) all merged to main (341 tests); deferred run/sweep/resume verbs + push pending

## Audits
## Research
- [260702_roadmap-research-synthesis](research/260702_roadmap-research-synthesis.md) — roadmap-cycle research synthesis (task 260702_research-roadmap-dags): 6 adversarially-verified themes → #2181 autoresearch ratchet-loop architecture + gate catalog + bathos capability map, #2174 minimal composition substrate child item (D1–D4, AC1–AC6), neuro-symbolic placement (grounding node in #2174, entry criterion for #2181), praxia plugin contract + rig-run dispatch gaps, cross-cutting gate-design template; brainstorm fork list + dropped-claims appendix

## Misc

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
