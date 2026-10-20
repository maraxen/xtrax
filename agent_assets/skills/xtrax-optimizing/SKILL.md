---
name: xtrax-optimizing
description: "Use when a JAX scan or vmap is slow, host or device round trips dominate, a table should be computed in the graph instead of materialized, batches should be prefetched, or a speed change needs a before-and-after measurement before it is kept. Covers host-boundary cost, data movement, and composition rewrites."
when_to_use: 'Triggers: scan or vmap is slow, host sync or device round trip dominates a step, prefetch batches or overlap host-to-device transfer, compute a table in the graph instead of materializing it, donate buffers or rematerialize an activation, measure a speed change before keeping it.'
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
| 3 -- Composition | The program itself | Vmap/ChunkedMap/DedupGather/Bucket swaps, on-the-fly one-hot/categorical encode, fusion-friendly refactors, donate/remat tradeoffs | Stage-0 cost probe FIRST, then paired Stage-1/2; ranking claims need TERM_RANKING floors; parity metric when numerics could shift | TERM_RANKING / END_TO_END |

Tier-1 facts already established in code (verify-paths):
- An ordered Tap/Sink inside a Scan of N steps costs N serialized host round
  trips (`src/xtrax/stages/executor.py` module docstring).
- Vmap cannot lower an ordered io_callback; topology rejects the plan
  (`src/xtrax/stages/topology.py`, `validate_plan_topology`).
- Only set `ordered=True` when correctness depends on host-observed order.

## Vectorize-First Discipline (replicates, draws, pairs, cells, frames)

A Python loop over replicates, bootstrap draws, pairs, cells or frames is a
defect until shown otherwise. Incident (downstream coverage run, 2026-10-16):
a host-side loop of B=200 fits per target per replicate took 30-98 s per
replicate at production size.

- Batch the leading axis in the first design: `jax.vmap`, or xtrax's
  `Vmap` / `ChunkedMap` strategies via `axis_dispatch` (`src/xtrax/tiling/dispatch.py`).
  Chunk with `ChunkedMap` when memory bounds the batch; a ragged final chunk is fine.
- Test batched == looped output (same keys, same inputs) before trusting the
  batched path; keep the looped version only as the test oracle.
- Profile one unit at production shapes (real K, N, B, pair count), not toy
  shapes, before any run expected to take more than a few minutes. Read
  `exec/element` from `xtrax profile`.
- Count recompiles: enable `jax.config.update("jax_log_compiles", True)` (or
  `JAX_LOG_COMPILES=1`) and expect zero compiles at steady state; `xtrax profile`
  reports `steady compiles`.
- Put the unit's wall-time estimate in the run's pre-registration sidecar.

## Hoist shared work out of a mapped axis

A stage that reads only shared inputs (a structure encoder, a table) should run
once per call, not once per element. Use `xtrax.inference.hoist`
(`src/xtrax/inference/hoist.py`):

```python
from xtrax.inference import MemoPolicy, hoisted_map, make_hoisted

hm = make_hoisted(shared, per_element, batch_size=4, cache=MemoPolicy())  # hold hm across calls
ys = hm({"coords": coords}, keys)          # shared(**args) once; per_element(s, x) mapped over keys
ys = hoisted_map(shared, per_element, shared_args, xs)  # one-shot: builds a fresh instance
```

- `shared(**shared_args)` returns `s`; `per_element(s, x)` maps over the leading axis of `xs`.
- False dependence is the real waste. A per-element PRNG key that feeds the encoder
  makes the encoder read a varying input, so the jaxpr batches it over N even when
  its value does not depend on the key (`backbone_noise == 0`). Keep per-element keys
  in `xs`, not in `shared_args`. Passing `AxisSpec(..., varying_inputs=(...))` to
  `make_hoisted(axis=...)` refuses a shared input with a declared varying name
  (`HoistVaryingInputError`).
- `check_elements=k, reference=monolithic` runs a parity probe on the first k
  elements; a mismatch raises `HoistParityError`. Use it while moving code across.
- `cache=MemoPolicy(...)` memoizes `shared` on the instance (content-keyed, see
  `src/xtrax/inference/memo.py`). Its entries can hold large arrays; bound them with
  `max_entries`. Under an enclosing `jit` the cache is bypassed (`stats()["traced_bypass"]`).
- Probe on this machine (CPU, jax 0.11.1, throwaway script, 261020): XLA already hoists a
  truly invariant subgraph out of `lax.map`, chunked or not, including a size-inflating
  `(L,3)->(L,L,C)` encoder. Hand-hoisting it gave no measurable change. The gain from
  `hoisted_map` is removing false dependence and cross-call recomputation. GPU and
  IREE/StableHLO behaviour is unverified.
- `hoisted_map` builds fresh jit wrappers on every call, so the mapped stage recompiles
  each time. A caller that repeats a call should hold a `HoistedMap` from `make_hoisted`.
- Tests: `tests/inference/test_hoist.py`, including the negative and invariant controls.

## Remediation Playbook (keyed by `xtrax profile` flags)

Measure the unit with `xtrax profile` first (`xtrax-probing` covers the report).
Each flag names a likely cause and a first change. Classify the change
(tier table above) and preregister the re-measure before editing.

| `xtrax profile` says | Likely cause | First change | Re-measure |
|---|---|---|---|
| `exec/element` >= 0.5 | Host loop over the leading axis: one jitted step per element | Vmap the leading axis (`jax.vmap`). If memory bounds the batch, chunk it with a jitted chunk kernel (`ChunkedMap`). Test batched == looped output before keeping it. | exec/element < 0.5 at the same shape |
| steady compiles > 0 | `jax.jit` built inside the callable, shape or dtype drift between calls, unhashable static args | Hoist the jit to module scope or a cached factory. Hold shapes and dtypes fixed (bucket them if they vary). Make static args hashable. | steady compiles = 0 |
| host share dominates (`host s / unit` >> `device s / unit`) | Host-side work between dispatches: Python-side preprocessing, NumPy round trips, a sync (`.item()`, `block_until_ready`) per iteration, host callbacks | Move the work into the graph, or batch it so one dispatch does many steps. Callback mechanics are Tier 1; data placement is Tier 2. | host share falls; wall per unit falls |
| `fits: no` | The compiled program exceeds device memory at this extent | Measure smaller with `--target-extent N` (the estimate is extrapolated, or withheld). For the production run, scan with a carry instead of materializing every step, or checkpoint/remat the body (Tier 3 donate/remat tradeoff). | `fits: yes` at the production extent, or a chunked design that fits |

Rule: a flag is a reason to measure again, not a verdict on the code. Keep a
change only with a paired record that shows the flag cleared and the wall
moved (see Non-Negotiables 2 and 3).

General JAX patterns (loop placement, jit boundaries, shape polymorphism,
scatter-gather) belong to the jaxlint skills: `jaxlint:using-jax` and
`jaxlint:using-jaxlint`. This skill covers what xtrax implements: the tier
mapping, the probe pairs, and the keep/revert rule.

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
