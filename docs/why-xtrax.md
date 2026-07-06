# Why xtrax exists

xtrax is a set of building blocks for JAX/Equinox pipelines, extracted from research
code for batched scientific inference and training. Its core bet is that the hard,
recurring work in a JAX project is not inside the compiled function — XLA handles that
well — but in the decisions you must make *before* tracing: how to batch, how to pad,
what to deduplicate, and how to keep memory bounded. This page explains why that layer
has to exist at all, and where xtrax draws its boundaries.

## `jax.jit` specializes; it does not adapt

The name "JIT" suggests a compiler that watches your program run and adjusts — the way
V8 or HotSpot profile hot paths and re-optimize. `jax.jit` is not that. It is closer to
a shape-specialized ahead-of-time compiler with a cache:

1. **Shapes freeze at trace time.** `jit` traces your function with abstract values —
   shape and dtype, no data — and lowers it to a program in which every shape is a
   compile-time constant. Free device memory, the composition of the current batch, and
   your sequence-length distribution are all invisible to the compiler.

2. **Memory is planned statically.** XLA computes buffer assignment for the whole
   program at compile time. `vmap` over an axis of size N widens every intermediate by
   N; if the result exceeds device memory, compilation or execution fails — there is no
   graceful degradation. XLA fuses operations, and `jax.checkpoint` trades recompute for
   activation memory, but nothing in the stack will rewrite "vmap over 100k" into "scan
   over 100 chunks of 1k". That is a source-level restructuring, and it has to happen in
   Python, before tracing.

3. **Shape variation is expensive.** The one adaptive behavior `jit` has — recompiling
   when it sees a new shape signature — costs seconds to minutes per signature. Ragged
   data fed naively triggers a compile per distinct length. The fix is to *suppress*
   runtime variation by padding to a bounded set of canonical shapes.

The consequence: staying inside a memory budget while getting good utilization out of
`jit` is a planning problem that lives entirely above the trace boundary, driven by two
things the compiled program can never observe — your data's statistics and your
device's memory.

## What xtrax does about it

`xtrax.tiling` turns those pre-trace decisions into declared, inspectable objects. You
describe each axis with an `AxisSpec` (cardinality, batch size, heterogeneity, dedup
eligibility, bucket boundaries); `BatchPlanner` selects a strategy per axis and returns
a `BatchPlan` that you can execute — or interrogate with `xtrax plan` and
`xtrax explain`.

| The compiler cannot… | xtrax strategy |
| --- | --- |
| Narrow a `vmap` that exceeds memory | `SafeMap` (chunked; delegates to `jax.lax.map(batch_size=)`) or `Scan` |
| Avoid one recompile per distinct shape on ragged axes | `Bucket` — pad to configured boundaries, bounding the number of compiled variants |
| See that N inputs contain only K unique elements | `DedupGather` — compute on unique values and scatter back; power-of-2 padding keeps compiled variants at O(log K) |
| Explain the batching topology it was given | `xtrax plan` / `xtrax explain` — per-axis strategy decisions, with reasons |

The rest of the library is conveniences around the same workflows: a `Trainer`/`Engine`
pair for Equinox training loops, checkify-based NaN/Inf guards, structured sparsity
with fixed compile shapes, sharding helpers, and orbax-backed checkpointing.

## What xtrax is not

- **Not a framework.** If you prefer to write your own training loop — the Equinox
  house style — the tiling layer works without `Trainer` or `Engine`.
- **Not an LLM pretraining stack.** MaxText and Levanter target fixed-shape transformer
  training at datacenter scale. xtrax targets the opposite regime: ragged,
  heterogeneous, repetitive axes, as found in scientific batch workloads.
- **Not a replacement for JAX's own machinery.** Sharding is `jit` auto-sharding and
  `shard_map` under thin helpers; checkpointing is orbax; data loading is grain. Where
  JAX core grows an equivalent capability — as `jax.lax.map(batch_size=)` did for
  uniform chunking — xtrax's policy is to delegate to it rather than maintain a
  parallel implementation (`safe_map` already does).

## Status

xtrax is alpha software, built first for the author's own research. The tiling,
inference, and EDA layers are where the design effort goes; the training conveniences
are deliberately thin. APIs may change without notice before 1.0.
