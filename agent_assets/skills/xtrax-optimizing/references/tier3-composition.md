# Tier 3: Composition Playbook

Composition changes rewrite the program: strategy swaps, encoding decisions,
fusion-friendly refactors, donate/remat tradeoffs. Highest reward, highest
risk -- they can alter numerics or sharding behavior, so their evidence bar is
the highest of the three tiers.

## Order of operations

1. Stage-0 cost probe FIRST (never executes). A losing idea often shows up as
   flops/bytes regression before you spend compile+run budget. Pattern:
   `scripts/prof_stage0_onehot_cost.py` (one lowered program per variant,
   `cost_analysis()` only, STRUCTURAL-only records).
2. Stage-1 micro-exec with attribution + parity gate. Single executable
   hosting all variants under disjoint `jax.named_scope` labels (why: two-input
   attribution joins compiled-HLO op_name paths to trace thunk names within
   ONE executable's namespace -- `src/xtrax/profiling/trace.py`; cross-
   executable thunk-name collisions are real). Pattern:
   `scripts/prof_stage1_onehot_micro.py`.
3. Only then consider Stage-2+ GPU runs for rankable pairs (record construction
   demands platform="gpu" + device_kind at stage>=2).

## Strategy swaps (tiling)

Vmap / SafeMap / DedupGather / Bucket are the closed strategy vocabulary
(`src/xtrax/tiling/strategy.py`, dispatched via `src/xtrax/tiling/dispatch.py`).
Constraints that are already enforced, do not rediscover them:

- Scan/WhileCarry rejected on heterogeneous axes (static carry shape)
  (`dispatch.py::make_axis_dispatch`).
- Ordered Tap/Sink cannot ride a Vmap axis (`topology.py`,
  `validate_plan_topology`).
- Bucket bounds recompiles for variable-length axes (`src/xtrax/tiling/bucket.py`)
  -- prefer it over unbounded jit re-tracing when shapes vary.

Existing cost/attribution coverage of these strategies lives in
`scripts/prof_stage0_tiling_cost.py` + `scripts/prof_stage1_tiling_micro.py`;
extend those rather than duplicating when the kernel is tiling-shaped.

## On-the-fly encoding (one-hot exemplar)

Decision shape: dense materialized input vs compact indices + in-graph encode.

First measured results (CPU jax 0.10.2, rows=256 classes=32 cols=16): the
STABLE finding is structural -- the on-the-fly path owns ~2x the fused thunks
of the dense-fed path across every run; attributed wall ratios fluctuated
~0.5x-1.9x between runs, so CPU micro walls at this scale carry no directional
verdict on their own. AND the dense feed pays 8*K extra bytes/row on H2D
transfer, which neither the stage-0 analysis nor the stage-1 traced region
sees. Honest conclusion at this scale: no sign yet -- repeat paired runs,
re-run at stage>=2 on GPU for anything rankable, and measure end-to-end walls
(Tier-2 style) before switching a real pipeline's feed format.

Parity rule: whenever a composition change could alter results, capture
max-abs-diff against the incumbent in-record and gate emission on tolerance
(see PARITY_TOLERANCE in the stage-1 one-hot driver). A record whose parity
metric launders a numerics regression violates Non-Negotiable 5.

## donate / remat

- Donation plumbing exists today only in `xtrax.sparse.inference.sparse_filter_jit`
  (donate= passthrough). Generalizing donation = new library surface; per the
  scope doc, build utilities only when probes justify them (>=2 drivers sharing
  scaffolding).
- Compile-vs-runtime split is already measured by
  `xtrax.loop.compile_time_clock.measure_two_phase_timing` (AC-27): use it when
  a candidate trades compile time for steady-state speed; never let a persistent
  XLA cache launder compile cost into "free".
