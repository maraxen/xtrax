---
name: xtrax-optimizing
description: This skill should be used when the user asks to "optimize a JAX function", "make this faster", "why is my scan/vmap slow", "reduce host sync or device round trips", "should this be computed on-the-fly instead of precomputed", "prefetch batches", "donate buffers", "compare encoding strategies", "benchmark before and after a perf change", or mentions optimization tiers, tap/sink cost accounting, data-movement tuning, composition-level rewrites, ProbeRecord-gated keep/revert decisions, prof_stage0_onehot_cost, prof_stage1_onehot_micro, prof_stage1_host_boundary, or prof_stage1_feed_overlap. Covers the three-tier taxonomy separating host-boundary mechanics, data movement, and program composition changes, plus the preregister-and-probe measurement protocol every optimization claim must pass.
xtrax_version: 0.4.0a5
triggers:
  - optimize / speed up / why slow (JAX programs)
  - tap / sink / io_callback round-trip cost
  - prefetch / double-buffer / H2D transfer
  - on-the-fly encoding / one-hot in-graph vs materialized
  - donate_argnums / remat / strategy swap
  - Tier-1 / Tier-2 / Tier-3 optimization
  - prof_stage0_onehot_cost / prof_stage1_onehot_micro
  - prof_stage1_host_boundary / prof_stage1_feed_overlap
---

# xtrax-optimizing

## Purpose

Turn performance hypotheses into **classified, probe-gated changes**: every
optimization is assigned a tier BEFORE touching code, the tier determines which
probe must run, and no change is kept without a supporting ProbeRecord. The
sibling skill `xtrax-probing` covers how measurements are made and what they
may be cited for; this skill covers what to change and what evidence each class
of change owes.

Verify-paths (house convention): every rule below cites the module that owns
it. When this skill and the code disagree, the code wins -- then update this
skill.

Scope doc: `.praxia/docs/specs/260825_jax-optimizing-skill-scope.md`.

## Non-Negotiables

1. Classify first, change second. Every optimization is Tier 1, 2, or 3 before
   any edit; never mix tiers in one measured candidate.
2. Baseline record first, candidate record second, same git sha, same machine.
   No baseline = nothing to compare = nothing to keep.
3. A change without a supporting record is reverted, not argued for.
   `ClaimValidityError` on an unsupported keep-claim is the contract working
   (`xtrax.profiling.claims.assert_claim_supported`).
4. Never hand-write record JSON; emit via `emit_probe_record`
   (`src/xtrax/profiling/emitters.py`). Never widen a guard to make a claim
   pass.
5. Tier-3 candidates that could alter numerics carry a parity metric in-record;
   the stage-1 one-hot driver gates emission on it outright.

## Tier Taxonomy (classify here, then follow the row)

| Tier | Change surface | Examples | Required evidence | Claim ceiling |
|---|---|---|---|---|
| 1 -- Host boundary mechanics | How host callbacks fire; never the math | ordered vs unordered Tap/Sink, per-step sink batching via ZarrStagingSink, io_callback round-trip accounting | Stage-1 micro probe isolating callback cost; DISPATCH_COUNT must not regress | DISPATCH_COUNT |
| 2 -- Data movement | When/how arrays reach the device; still not the math | async_indexed_stream buffer sizing, double-buffering input iteration, dtype coercion at load vs in-graph, bucket selection to bound recompiles | paired Stage-1 wall probes (feed-starved vs fed); show overlap hides latency, not shifts it | TERM_RANKING (same platform); END_TO_END only via scale guard |
| 3 -- Composition | The program itself | Vmap/SafeMap/DedupGather/Bucket swaps, on-the-fly one-hot/categorical encode, fusion-friendly refactors, donate/remat tradeoffs | Stage-0 cost probe FIRST, then paired Stage-1/2; ranking claims need TERM_RANKING floors; parity metric when numerics could shift | TERM_RANKING / END_TO_END |

Tier-1 facts already established in code (verify-paths):
- An ordered Tap/Sink inside a Scan of N steps costs N serialized host round
  trips (`src/xtrax/stages/executor.py` module docstring).
- Vmap cannot lower an ordered io_callback; topology rejects the plan
  (`src/xtrax/stages/topology.py`, `validate_plan_topology`).
- Only set `ordered=True` when correctness depends on host-observed order.

## Quick Start (worked example: on-the-fly vs materialized one-hot)

```bash
# Structural cost signal first (never executes):
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage0_onehot_cost.py
# Then micro-execution + attribution + parity gate:
XTRAX_GIT_SHA=$(git rev-parse HEAD) \
    uv run python scripts/prof_stage1_onehot_micro.py
```

Records land in `outputs/profiling/stage{0,1}/`; raw traces stay alongside,
gitignored. The stage-1 driver FAILS LOUD if the two encodings disagree beyond
parity tolerance -- a numerics regression blocks measurement rather than
producing a citable-but-wrong record.

Known first results (CPU, jax 0.10.2, 256x32): the STABLE structural finding
is that the on-the-fly path consistently owns ~2x the fused thunks of the
dense-fed path (higher attributed occurrence count, every run). Wall-clock
ratios between the variants fluctuated ~0.5x-1.9x across repeated runs -- CPU
micro walls at this scale are too noisy for a directional verdict, which is
itself the lesson: repeat paired runs before believing a sign, and re-run at
stage>=2 on GPU for anything rankable. H2D transfer savings of the compact
feed are NOT measured inside either driver -- do not extrapolate past the
platform guard.

## Measurement Protocol

Preregistration -> baseline -> single-tier candidate -> paired probe ->
claim-gated keep/revert. Full procedure, probe recipes per tier, and the
keep/revert decision rule: see `references/measurement-protocol.md`.

Tier exemplars, each shipped WITH its measured driver:

- Tier-1: `scripts/prof_stage1_host_boundary.py` -- ordered vs unordered vs
  no sink under Scan; correctness gate before timing.
- Tier-2: `scripts/prof_stage1_feed_overlap.py` -- sequential vs
  `async_indexed_stream` overlapped feeding; regime guard in-record.
- Tier-3: `prof_stage0_onehot_cost.py` + `prof_stage1_onehot_micro.py` --
  materialized vs on-the-fly one-hot with parity gating.

## Status / Roadmap

- P1 (shipped): taxonomy, measurement protocol, Tier-3 one-hot probe pair.
- P2 (shipped): Tier-1 host-boundary driver + Tier-2 feed-overlap driver,
  each with a reference page written from its verified first results
  (including negative results -- see tier2's 0.70x finding).
- P3 (shipped): cross-references from `using-xtrax` and `xtrax-probing`.
- Open (needs user/GPU): Stage-2 GPU re-runs of all three probes for
  rankable TERM_RANKING pairs; performance-gate rubric entries if any driver
  earns a permanent tripwire; possible `xtrax.perf` package if shared driver
  scaffolding justifies it.

## Additional Resources

Load as needed -- do not read all up front:

- **`references/measurement-protocol.md`** -- preregistration format,
  paired-probe recipes per tier, claim-class mapping, keep/revert procedure.
- **`references/tier1-host-boundary.md`** -- callback cost accounting,
  ordering rules, batching-the-payload pattern (measured).
- **`references/tier2-data-movement.md`** -- prefetch/regime checks,
  bucketing, dtype placement (measured, incl. a negative result).
- **`references/tier3-composition.md`** -- strategy swaps, on-the-fly
  encoding, donate/remat, parity gating.
