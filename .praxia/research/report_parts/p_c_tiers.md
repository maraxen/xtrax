## (c) Kernel technique inventory mapped to OUR tiers

### Tier 1 — host-boundary mechanics (taps, sinks, callbacks)

Direct literature coverage of *host-callback* mechanics is thin in this corpus — the sources are device-centric. What maps:

- **Producer/consumer role separation is the device-side analogue of our host-boundary concern**: FA-3 splits warps into dedicated producers (memory issue) and consumers (compute) precisely so neither stalls the other, and adds a dedicated dQ-writer so gradient accumulation doesn't contend with compute warps [FA-3]. Our ordered-tap-inside-scan problem (N serialized host round trips) is the same decoupling problem at the host boundary; the corpus says the fix pattern is "separate the mover from the computer," which is exactly what `async_indexed_stream` + staging sinks do.
- **Serialization tax on ordered operations**: both GPUDirect Storage and PagedAttention exist because a staging path serializes work that could overlap — GDS removes CPU bounce-buffer round trips between storage and GPU [GPUDirect Storage], and vLLM pages the KV cache so per-step work stops paying fragmentation taxes [PagedAttention/vLLM]. Same shape as our tap/sink problem: move bytes off the critical path or batch them.

### Tier 2 — data movement & prefetch

- **Fusion as IO elimination**: FlashAttention's entire win is refusing to round-trip intermediates through HBM — tiling into SRAM and fusing matmul+softmax+mask+dropout into one kernel [FlashAttention]. This grounds our Tier-3 fusion-friendly refactors AND Tier-2 dtype/placement choices: every byte that never leaves the chip is free.
- **Overlap via pipelining**: FA-3's s-stage circular SMEM buffers overlap stage-j compute with stage-j+1/j+2 loads; ablation shows removing it costs 661→582 TFLOPS (~12%) [FA-3]. This is the device-side twin of our double-buffered input iterator; it also warns that overlap gains are real but modest-sized even in expert hands.
- **Async copy machinery**: Ampere's `cp.async` global→shared copies and Hopper's TMA (bulk, hardware-driven tensor copies) exist to keep loads off the critical path [FA-3]. JAX/XLA users don't emit these directly, but they predict where XLA has headroom we can't reach by hand — relevant to how we set expectations for Stage-1 probes.
- **Bucketing ↔ capacity equations**: FA derives tile shapes from SRAM size [FlashAttention]; Triton exposes tile shapes as first-class tunables and JIT-autotunes them per target [Triton]. Our shape-bucketing (`tiling.bucket`) is the same idea one level up: pick shapes the compiler can tile well instead of recompiling per length.

### Tier 3 — composition changes

- **Recomputation over materialization**: FA's backward recomputes attention probabilities from tiled inputs instead of storing the N×N matrix — trading FLOPs (cheap, abundant) for bytes (expensive, scarce) [FlashAttention]. Directly parallels our remat/checkpoint tradeoff item and the on-the-fly one-hot question.
- **Online aggregation enables fusion of "impossible" ops**: softmax couples an entire row, yet online softmax computes it block-incrementally with running max/sum statistics and rescaling [FlashAttention]. Lesson for us: a reduction that *looks* like a global barrier can often be restructured as a streaming accumulation — the same trick scan-based gradient accumulation exploits.
- **Algorithm/schedule separation as the compiler answer to hand-tuning**: Halide fully separates what from how, letting identical math retarget CPU vector code or CUDA by editing schedule lines [Halide]; Triton declines full separation, keeping schedules implicit and autotuned via parametric tiles [Triton]; Ansor searches schedule spaces automatically, beating hand-written kernels in some regimes [Ansor]. XLA sits closer to the Triton end: implicit scheduling with limited user levers (donate, remat, shard_map) — which is why our taxonomy treats composition changes as the primary lever and documents expectations accordingly.
- **Low precision as a data-movement optimization**: FP16→FP8 halves bytes moved per activation; FA-3's FP8 path reaches 1.2 PFLOPs/s vs 740 TFLOPs/s FP16 [FA-3]. In our terms, dtype choice is legitimately Tier-2/Tier-3 boundary territory with numerics-parity obligations: FP16 safety requires FP32 master weights, FP32 accumulation, and FP32 reductions in sensitive reductions [Micikevicius]; FP8 assigns E4M3 to weights/activations and E5M2 where range matters [FP8 formats].

