# Why xtrax exists

Put simply (with no LLM-assistance): I kept having to re-implement practices used
throughout this library per-project to take advantage of JAX to run massively parallel
in silico experiments but avoid OOMs, manage the host-accelerator boundary, and keep
track of telemetry/metadata. After I did this enough times, I decided it was worth it
to just get a common base library that keeps my whole JAX-oriented stack maintained in
one place. As stated in the README, I hope others operating under similar constraints
can find this useful and I intend to give my best effort to maintain this. That said,
this project resolves a specific set of problems around maintaining consistent
performance and tracking in my JAX stack.

---

xtrax packages the decisions a JAX program makes *before* tracing: how to batch, how
to pad, what to deduplicate, and how to keep memory bounded. XLA optimizes the program
it is given; xtrax is about deciding which program to give it.

## The pre-trace layer

Three properties of `jax.jit`/XLA define where that layer sits:

- **Shapes freeze at trace time.** Functions are traced with shape and dtype only, so
  data statistics and device memory are invisible to the compiler.
- **Memory is planned statically.** Buffer assignment happens at compile time. `vmap`
  over an axis of size N widens every intermediate by N, and nothing in the stack will
  rewrite "vmap over 100k" into "scan over 100 chunks of 1k" — that restructuring
  happens in Python, before tracing.
- **Each new shape signature costs a compile.** Ragged data fed naively triggers a
  recompile per distinct length, so shapes must be padded to a bounded canonical set.

Batching topology, padding policy, and memory budget are therefore *inputs* to the
compiler, decided host-side from exactly the two things the compiled program cannot
see: your data's statistics and your device's memory.

## What xtrax does about it

`xtrax.tiling` turns those decisions into declared, inspectable objects. Describe each
axis with an `AxisSpec` (cardinality, batch size, heterogeneity, dedup eligibility,
bucket boundaries); `BatchPlanner` selects a strategy per axis and returns a
`BatchPlan` you can execute — or interrogate with `xtrax plan` and `xtrax explain`.

| Decision the compiler can't make | xtrax strategy |
| --- | --- |
| Narrow a `vmap` that exceeds memory | `SafeMap` (chunked; delegates to `jax.lax.map(batch_size=)`) or `Scan` |
| Bound recompiles on ragged axes | `Bucket` — pad to configured boundaries |
| Skip duplicate work when N inputs hold K unique elements | `DedupGather` — compute on unique values, scatter back; power-of-2 padding bounds variants at O(log K) |
| Explain the batching topology it was given | `xtrax plan` / `xtrax explain` — per-axis decisions, with reasons |

The rest of the library is conveniences around the same workflows: `Trainer`/`Engine`
for Equinox training loops, checkify-based NaN/Inf guards, structured sparsity with
fixed compile shapes, sharding helpers, and orbax-backed checkpointing.

## What xtrax is not

- **Not a framework.** The tiling layer works without `Trainer` or `Engine` if you
  prefer to own your training loop.
- **Not an LLM pretraining stack.** MaxText and Levanter target fixed-shape
  transformer training at scale; xtrax targets ragged, heterogeneous, repetitive axes
  — the scientific batch-workload regime.
- **Not a replacement for JAX's own machinery.** Sharding is `jit` auto-sharding and
  `shard_map` under thin helpers; checkpointing is orbax; data loading is grain. Where
  JAX core grows an equivalent — as `jax.lax.map(batch_size=)` did for uniform
  chunking — xtrax delegates to it rather than maintaining a parallel implementation.
