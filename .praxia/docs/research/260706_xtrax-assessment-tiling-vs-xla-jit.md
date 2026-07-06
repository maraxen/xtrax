# xtrax niche assessment: does tiling make sense given jax.jit/XLA, and ecosystem positioning

**Task:** 260706_xtrax_assess · **Date:** 2026-07-06 · **Status:** analysis, no code changes

Two questions from the author, answered honestly:

1. Does the tiling layer even make sense given how jax.jit and XLA work? ("I thought the whole point of jit is adapting to runtime constraints.")
2. Does xtrax serve a useful role at all, given MaxText and the graveyard of archived JAX projects — and why hasn't this been done before?

---

## 1. Why tiling makes sense: jax.jit is not an adaptive JIT

The premise "jit adapts to runtime constraints" is the misconception. `jax.jit` is a
**shape-specialized ahead-of-time compiler with a trace cache**, not a runtime-adaptive
JIT. The name collides with V8/HotSpot-style JITs, which genuinely do adapt (profile hot
paths → speculate → deoptimize on miss). JAX has none of that machinery. Three facts pin
this down:

### 1a. Trace-time freezing

`jax.jit` traces the Python function with abstract values (`ShapedArray`: shape + dtype
only, no data) and lowers to a **closed HLO program in which every shape is a
compile-time constant**. Anything data- or environment-dependent — free device memory,
actual batch size, sequence-length distribution — is invisible to the compiler unless
encoded in the shape/static-arg signature. The compiler optimizes *the program it was
given*; it cannot ask whether a different program should have been given.

### 1b. Static buffer assignment

XLA computes buffer assignment (liveness, allocation, reuse) **at compile time** — a
fixed memory plan for the whole program. Peak memory is a function of program
*structure*. `vmap(f)` over N widens every intermediate by N; if the resulting live set
exceeds device memory, XLA fails — it does not degrade gracefully, chunk the batch, or
convert data-parallelism into sequentialism. Fusion shrinks some intermediates and
`jax.checkpoint`/remat trades recompute for activation memory, but **neither restructures
the batching topology**. Turning "vmap over 100k" into "scan over 100 chunks of 1k" is a
source-level rewrite that must happen in Python, *before* tracing. XLA is structurally
incapable of making that decision — it happens on the wrong side of the trace boundary.

### 1c. Adaptation is penalized, not free

The one adaptive behavior jit has — recompile on a new shape signature — costs seconds to
minutes per signature plus unbounded cache growth. So runtime shape variation is exactly
what a JAX program must *suppress*, by padding/bucketing to a small canonical shape set.
Naive "adapt batch size at runtime" produces compile storms. (Aside: XLA autotuning does
run-and-measure at compile time, but only to select kernel implementations for fixed ops
— never to change program topology. GPU memory preallocation further hides memory state
from the program.)

### Consequence

OOM avoidance under jit is **necessarily a pre-trace planning problem**: choose chunk
sizes, choose vmap-vs-scan, choose padding buckets, dedup redundant work — all functions
of (data statistics × device memory), neither of which the traced program can observe.
The cross-project patterns that became `xtrax.tiling` were not workarounds for a missing
XLA feature that will eventually ship; they are the layer's job that XLA's design
*assigns to the user*. Static shapes + whole-program compilation is the deliberate trade
that buys fusion and TPU performance; its cost is exactly that someone above the trace
boundary must decide the tiling. Today that someone is ~300 vendored lines in every
scientific-JAX repo, or a planner.

**Verdict: tiling makes sense not despite how jit works, but because of how jit works.**

Mapping to xtrax strategies:

| Compiler gap | xtrax answer |
|---|---|
| Cannot narrow a vmap that OOMs | `SafeMap` / `Scan` (chunking, sequentialization) |
| Recompiles per distinct shape; ragged data → compile storm | `Bucket` (pad to boundaries, bound signatures) |
| Cannot see that N inputs contain K unique elements | `DedupGather` + power-of-2 k-bucketing (O(log k) variants) |
| Decision inputs (data stats, memory) invisible post-trace | `AxisSpec`/`BatchPlanner` declarative planning + `plan`/`explain` EDA |

---

## 2. Ecosystem positioning: the honest evaluation

### Module-by-module split

**Differentiated (invest):** `tiling` (~1.4k LOC), `inference` (signature inference,
fail-loud axis roles), `eda` (introspectable *reasons* for strategy choices — no known
equivalent anywhere). The combination — declarative axis planning with dedup/bucketing
for heterogeneous scientific workloads, on Equinox, with introspection — is unclaimed
territory.

**Commodity, graveyard-shaped (keep thin):** `training`/`engine` competes with elegy,
treex, objax, ciclo — all dead — and with Equinox's explicit "the training loop is 20
lines you should own" philosophy. `checkpoint` is a thin orbax wrapper (absorbs orbax API
churn — cost, not value). `distributed` overlaps with native jit auto-sharding +
`shard_map`. `safety` is fine but tiny.

**Meta:** `devtools` is ~4.5k LOC (37% of src) of praxia-loop infrastructure (emit gates,
judgment rubrics, tombstones), not library surface. User-facing footprint is ~½ of raw
src LOC.

### Why hasn't this been done before? It has — three ways, each with a death mechanism

1. **Google's attempts died organizationally, not intellectually.** `xmap` — the closest
   ancestor to AxisSpec-style named-axis thinking — was deprecated and removed; the core
   team retrenched to minimal `shard_map`. Trax, Objax, jraph archived; Haiku maintenance
   mode since 2023; T5X/Flaxformer frozen. Pattern: general middleware over a fast-moving
   core needs permanent staffing, and Google funds models, not middleware.
2. **MaxText is the anti-abstraction conclusion Google drew.** Deliberately flat,
   fork-and-modify reference code for LLM pretraining on TPU — Google saying "we no
   longer believe in general JAX training frameworks; we ship forkable examples." It has
   nothing to say about ragged, dedup-heavy scientific inference. Different niche.
3. **The closest living relative proves the niche is real but unserved.**
   Levanter/Haliax (Stanford CRFM) is maintained, Equinox-based, named-axis — but shaped
   for fixed-shape LLM pretraining with FSDP. No bucketing planner, no dedup, no
   heterogeneous-axis story. The actual customers for xtrax's niche (comp bio/chem labs)
   each vendor their own padding/bucketing code and never extract it: low career payoff,
   high maintenance cost. **The niche is empty for economic reasons, not technical ones.**

### Absorption risks (watch list)

- `jax.lax.map(..., batch_size=)` covers SafeMap's core (uniform chunking) in JAX core.
  Verified 260706: `safe_map` (`src/xtrax/transforms/map.py`) already delegates to it —
  no duplicate implementation exists. Residual work filed as **backlog #3120**: drop the
  stricter-than-core `n % batch_size` ValueError (core lax.map handles remainders), pin
  the delegation with a test, reassess the vmap-when-it-fits fast path.
- If XLA/JAX ever ships first-class ragged/dynamic shapes, the bucketing half shrinks —
  though bounded-dynamism memory planning is still worst-case-static, and this has been
  "coming" for years.
- `jax.checkpoint`/remat covers "activations too big within a fixed structure";
  donation covers buffer reuse. Neither threatens the planner.

### Strategic conclusions

- The graveyard pattern kills breadth-value projects (frameworks needing adoption to
  justify maintenance). It is kind to depth-for-the-author projects. xtrax's README
  already frames it correctly: personal research infrastructure, best-effort support.
  It pays for itself at zero external users as the shared substrate across the author's
  pipelines (aminx, naurmalade — ragged, repetitive protein workloads are exactly what
  dedup/bucket planning is for).
- Invest in tiling/inference/EDA; keep trainer/engine/checkpoint deliberately thin and
  boring (every LOC there is churn liability against optax/orbax drift).
- Don't measure success by adoption. Realistic external ceiling: a handful of
  scientific-JAX labs find the planner and stop vendoring padding code. That's a good
  outcome.

**Grounding caveat:** ecosystem status (MaxText active under AI-Hypercomputer,
Levanter/Haliax active, xmap removed, Haiku/T5X frozen) is as of early 2026; no fresh
web sweep was done for entrants newer than that.
