# Session Handoff (checkpoint) — 260825_jax_optimizing_skill

> 2026-08-25T20:25Z. Status: **complete** (initiative delivered; no open
> agent-owned work). Scope doc section 7b is the authoritative record.

## Metadata
- workspace: xtrax | branch: `fix/wheel-freshness-not-devtools` (shared; my
  commits touch only new files + two small doc edits)
- commits: 7de4ead (scope) → d8ae444 (survey fix) → 20bdc62 (P1 skill +
  one-hot probes) → 346daef (P2 Tier-1/2 drivers + refs + cross-links) →
  3ad9bcd (tests + changelog) → a5c391b (scope marked DELIVERED)
- origin task: Marielle 2026-08-25 "kernel/jax-ops optimization utilities…
  separate tap and host side… prefetching… composition changes like on-the-fly
  one-hot… grounding everything with measurements akin to bathos discipline"

## What exists now
- Skill `agent_assets/skills/xtrax-optimizing/`: SKILL.md (three-tier taxonomy:
  T1 host-boundary mechanics / T2 data movement / T3 composition) +
  references/{measurement-protocol,tier1-host-boundary,tier2-data-movement,
  tier3-composition}.md. Every measured claim cites a real driver run.
- Drivers (all emit claim-valid ProbeRecords, all pinned by
  tests/scripts/test_prof_optimizing_drivers.py):
  - prof_stage0_onehot_cost.py — never-execute cost_analysis, one record per
    encoding variant.
  - prof_stage1_onehot_micro.py — single-executable named-scope attribution;
    parity gate BEFORE measurement (refuses to emit on divergence).
  - prof_stage1_host_boundary.py — none/unordered/ordered sink under Scan;
    correctness gate before timing; dispatch counts from short separate traces.
  - prof_stage1_feed_overlap.py — sequential vs async_indexed_stream; carries
    isolated_step_seconds regime guard in-record.

## Measured first results (CPU-only jaxlib machine, treat as structural only)
- One-hot: on-the-fly path owns ~2x fused thunks of dense-fed (stable); wall
  ratios swung 0.5x–1.9x across runs → NO directional verdict at this scale.
- Host boundary: ANY per-step callback ~1000–1450x the boundary-free scan at
  64 steps; ordered-vs-unordered ratio scheduler-dependent, flagged non-citable.
- Feed overlap: 0.70x (SLOWER) on sub-ms CPU steps — async machinery outweighs
  hidden latency. Negative result preserved deliberately as the regime lesson.

## If a future session picks this up
1. Stage-2 GPU re-runs (needs GPU jaxlib): rerun all three stage-1 drivers
   with platform="gpu" wiring — records then demand device_kind (auto-captured);
   TERM_RANKING pairs become available via claims.paired_configs unanimity.
   NOTE: drivers currently hardcode platform="cpu" — flipping that per-run is
   the first code change needed.
2. Gate tripwires: DELIBERATELY not wired. House test pins repo targets to
   no-dispatch-config (test_repo_targets_have_no_dispatch_config). Enabling =
   policy change = Marielle's call.
3. xtrax.perf package: only if driver count grows; leaf-purity rules like
   xtrax/profiling would apply.

## Verification state at handoff
- 180 passed (tests/profiling + tests/scripts + tests/audit/test_performance_gate).
- ruff clean on all touched files; audit_profiling_contract OK; traces stay
  gitignored under outputs/profiling/**/_traces/.
- Branch also carries unrelated in-flight work NOT mine (27 modified files,
  .praxia locks) — do not sweep-commit those with anything above.
