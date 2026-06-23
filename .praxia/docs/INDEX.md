# xtrax Internal Docs

## Specs
- [260623_e1-signature-inference-core](specs/260623_e1-signature-inference-core-how-should-x.md) — E1-MVP spec (contemplex architectural brainstorm, session 4dfc27aa): derive BundleSchema + list[AxisSpec] from a typed pure fn via eval_shape; fail-loud UNKNOWN roles (not batch); @axis_config Tier-1; 8 ACs; Tier-2 jaxtyping/codegen/CarrySpec deferred (TBDs)
- [260615_xtrax-eda-api-revised](specs/260615_xtrax-eda-api-revised.md) — EDA visualization API spec (post-critic + adversarial review): render(), PlanStatsDict, PlanLogger, PanelName, 12 acceptance criteria + 7 amendments
- [260615_using-xtrax-skill](specs/260615_design-the-using-xtrax-skill-an-exportab.md) — using-xtrax exportable skill spec (post-adversarial-review): Faction D Hybrid, 7 ACs, 5 amendments, 4 HiTL stops, enforcement taxonomy split
- [260604_xtrax-spec](specs/260604_xtrax-spec.md) — Full xtrax v0.2.0 specification (Phases 0–7, Sprints 1–4)
- [260608_xtrax-s5-sparse](specs/260608_xtrax-s5-sparse.md) — Sprint 5: coverage completion, benchmarks, sparse infrastructure (Phase 8–10)
- [260608_inference-time-sparsification](specs/260608_inference-time-sparsification-in-xtrax-h.md) — Sprint 7: sparsify_model functional API, BucketIterator minimal impl, jit-trace guard (inference-time sparsification)

## Plans
- [260615_using-xtrax-skill-backlog-dag](plans/260615_using-xtrax-skill-backlog-dag.md) — Backlog DAG for the using-xtrax exportable skill: 5 tier-1 tasks, 5 tier-2 tasks, 5 tech-debt HiTL items, sprint sequencing
## Roadmaps
- [composition/260622_e1-signature-inference-epic](roadmaps/composition/260622_e1-signature-inference-epic.md) — E1 keystone epic: derive Bundle pytree + AxisSpec roles from typed pure JAX functions (eval_shape inference + jaxtyping dim-names → roles; Tyro-backed CLI). Grounded, pre-brainstorm.

## Handoffs
## Audits
## Research
## Misc

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
