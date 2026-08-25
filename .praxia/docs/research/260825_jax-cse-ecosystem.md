# Research Memo: Prior Art & Correctness Hazards for Runtime Compute Reuse in JAX

- Task: `260825_xtrax-cse-runtime-opt`
- Date: 2026-08-25
- Scope: prior art + correctness hazards for (a) jaxpr-level duplicate-op detection, (b) content-keyed memoization around jitted callables, (c) data-dependent dedup-gather of duplicated batch rows.
- Method note: general search engines (DDG/Bing) were blocked from this machine; all findings below come from direct fetches of primary sources (docs.jax.dev, github.com raw sources, GitHub REST API, llvm.org, kidger.site, dask/joblib RTD). Anything not backed by a fetched source is marked **UNVERIFIED**. Local established facts (jaxprs not CSE'd; XLA removes intra-program duplicates post-opt; hand-dedup wall-clock delta ~1% noise; jit cache keyed by structure/shapes/dtype; ClosedJaxpr hashable in 0.10.2) are treated as ground truth and referenced as [LOCAL].

---

## Q1. Does JAX expose an official/public API for CSE on jaxprs?

**Finding: No. As of JAX 0.9-0.10 there is no public jaxpr-level CSE pass. The only public API that touches CSE is `jax.lax.optimization_barrier`, which *prevents* it. All actual CSE happens inside XLA during compilation.**

1. `jax.lax.optimization_barrier(operand)` documents: "An optimization barrier prevents common subexpression elimination. This is used by JAX to implement rematerialization," and "Optimization barriers have no effect outside a compiled function."
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
   -> The one public knob is negative (block CSE locally), not positive (perform CSE).
2. Open feature request (2025-10, still open): "Option to prevent common subexpression elimination for part of a graph" — author states they "looked very hard" for any way to control CSE and found none; confirms users expect XLA, not JAX, to own CSE decisions.
   Source: https://github.com/jax-ml/jax/issues/32544
3. The CSE implementation lives in XLA: `xla/service/hlo_cse.cc` combines identical constants (keyed by literal + shape) and merges instructions with equal opcode + shape + operand ids (order-insensitive for commutative binary ops).
   Source: https://raw.githubusercontent.com/openxla/xla/main/xla/service/hlo_cse.cc
4. MLIR (which JAX lowers through for StableHLO export) ships a generic `-cse` pass ("Eliminate common sub-expressions", driven by the Memory SideEffect interface) plus a `cse-between-iterations` option in `-canonicalize`; StableHLO pipelines do not run it on JAX's pre-lowering IR.
   Source: https://mlir.llvm.org/docs/Passes/
5. JAX's persistent compilation cache keys on the **non-optimized** HLO ("The computation performed by the function captured by the non-optimized HLO ... hashed", plus jaxlib version, flags, device config) — i.e., upstream deliberately treats pre-optimization IR as the identity of a program; duplicate ops survive into that key.
   Source: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
6. `jax.jit(..., compiler_options=dict)` passes options to XLA CompileOptions, but no documented option toggles CSE specifically.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html ; existence of an `--xla_disable_hlo_passes=cse` style flag is **UNVERIFIED** (flag-listing page could not be fetched; grep.app rate-limited).
7. Implication for (a): a duplicate-op detector must be built at the jaxpr/StableHLO layer ourselves. Building blocks: `ClosedJaxpr` is hashable [LOCAL, matches use as cache key], `make_jaxpr`, `jax.jit(...).lower(...).as_text()` for pre/post-opt diffing (workflow demonstrated in jax issue #32544 using `hlo_module_from_text`/dot graphs).

## Q2. Prior art for value-caching jitted callables

**Finding: Nothing in core JAX caches *values*; everything documented caches *compilation*. Closest external precedents: Equinox's internal compile caches and joblib.Memory's content-hashed disk memoization.**

1. Core JAX: `jit` maintains a compilation cache keyed by argument structure/shapes/dtypes plus static args; "Static arguments are included as part of a compilation cache key, which is why hash and equality operators must be defined"; JAX holds only a weak reference to `fun`.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html [consistent with LOCAL fact]
2. Persistent compilation cache (`jax_compilation_cache_dir`): disk store of compiled executables; explicitly warns the cache is trusted code — "Sharing a compilation cache is equivalent to allowing anyone who can write to the cache directory to run code on your machine." Any value-memo store we build inherits the same trust model.
   Source: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
3. Equinox: `filter_jit` traces arrays dynamically, treats all other leaves statically, and exposes `donate='none'|'all'|...` with default off; `equinox.clear_caches()` exists to "Clear internal Equinox caches ... Best used before calling `jax.clear_caches()`" — i.e., even wrapper libraries accumulate structure-keyed caches that need explicit lifecycle management.
   Sources: https://docs.kidger.site/equinox/api/transformations/ , https://docs.kidger.site/equinox/api/caches/
4. joblib.Memory: disk memoization designed around numpy arrays, using `NumpyHasher` ("special case for fast hashing of numpy arrays", md5 over pickled stream + array bytes, `coerce_mmap` option); supports memmapped reloads and `call_and_shelve` references. Documented pitfalls map 1:1 onto our risks: pickle-trust warning; hash values not stable across joblib/numpy upgrades (cache-wide invalidation); function-name collisions across sessions; objects with non-deterministic pickles silently never hit (they call out pytorch.Tensor).
   Sources: https://joblib.readthedocs.io/en/stable/memory.html , https://raw.githubusercontent.com/joblib/joblib/main/joblib/hashing.py
5. `functools.lru_cache` over hashable leaves: workable only because `ClosedJaxpr`/pytrees-with-static-leaves are hashable [LOCAL]; raw `jax.Array` leaves are unhashable by design, forcing byte-hashing detours (see Q3). Specific lru_cache-on-jax guidance pages could not be fetched (**UNVERIFIED** beyond first principles).
6. `jax.experimental.host_callback`: the module file no longer exists on jax main (404 fetching `jax/experimental/host_callback.py`) — it was removed after deprecation, so it is not usable as a caching/callback substrate. Replacement-era callback limits (serialization overhead, no direct buffer reuse) are **UNVERIFIED** here.
7. Documented donation pitfall (applies to any cache returning retained buffers): "You should not reuse buffers that you donate to a computation; JAX will raise an error if you try to." A cached output passed back into `donate_argnums` code would be donated out from under the cache.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html

## Q3. Hashing JAX arrays on host: cost & practice

**Finding: Established practice (joblib, dask) is: normalize metadata (dtype/shape/endian) + hash contiguous bytes with a fast hash. Device->host transfer, not hashing, dominates cost. Bit-exactness caveats (-0.0, NaN payloads) apply.**

1. joblib's `NumpyHasher` hashes `arr.tobytes()`-style payload under an md5 stream including dtype/shape context, and treats memmaps coercibly — the canonical "content-hash a numpy array" recipe since 2009.
   Source: https://raw.githubusercontent.com/joblib/joblib/main/joblib/hashing.py
2. Dask formalizes the same idea as a protocol: `__dask_tokenize__()` must return "value that fully represents the object", consumed for deterministic keys/graph merging — precedent for exposing content-keying as an opt-in protocol rather than a hidden override.
   Source: https://docs.dask.org/en/stable/custom-collections.html
3. BLAKE3 README (official): "Much faster than MD5, SHA-1, SHA-2, SHA-3, and BLAKE2"; SIMD (SSE2..AVX-512, NEON) with runtime detection; Merkle-tree parallelism; `b3sum` is "an order of magnitude faster than e.g. sha256sum". Exact GB/s figures live in an image/paper, not fetched text — concrete throughput numbers **UNVERIFIED**, but blake3/xxhash-class speed >> hashlib-md5 on large inputs is well-supported qualitatively.
   Source: https://raw.githubusercontent.com/BLAKE3-team/BLAKE3/master/README.md
4. Cost shape: outputs of jitted calls are device-resident futures; any host hash must first force readiness/transfer. Async-dispatch doc shows host materialization (`np.asarray`) is the expensive step relative to enqueueing.
   Source: https://docs.jax.dev/en/latest/async_dispatch.html
5. Practice recommendation supported by sources: hash `(shape, dtype, weak_type?, endianness)` + bytes via blake3/xxhash; treat NaN payloads and ±0.0 bit patterns as distinct unless canonicalized (bit-pattern hashing makes them distinct; canonicalization policy is ours — **no fetched source prescribes this**).
6. Content-addressed *device-array stores*: no library found that stores jax.Array buffers content-addressed on device; joblib/dask operate host-side. Gap confirmed by absence in fetched ecosystems (**UNVERIFIED exhaustively**).

## Q4. Data-dependent dedup-gather precedent

**Finding: JAX-native precedent exists in spirit (BCOO duplicate coalescing, unique+inverse gather), but always with statically-bounded output shapes. Op-level CSE analogues exist in MLIR/torch.compile; nobody does runtime value-dedup inside compiled programs.**

1. `jnp.unique(ar, ..., size=..., fill_value=...)`: output of unique is data-dependent so it is incompatible with jit *unless a static `size` is given*; padded/truncated deterministically. With `return_inverse=True` it returns indices such that `values[inverse] == ar` (i.e., gather-reconstruct) — exactly the dedup-gather shape, but only callable outside compiled regions or with bounded size.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.unique.html
2. BCOO sparse: constructor takes `indices_sorted=False, unique_indices=False` hints, indices attribute documents "Duplicate entries will be summed", and `sum_duplicates(nse, remove_zeros)` returns "a copy of the array with duplicate indices summed" — an in-JAX, transform-compatible dedup-by-key (sort + segment-sum) precedent with explicit static-nse bounds. Module flagged experimental/not actively developed.
   Sources: https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.BCOO.html , https://docs.jax.dev/en/latest/jax.experimental.sparse.html
3. torch.compile/Inductor added a graph-level CSE pass (config `torch._inductor.config.inductor_cse`): folds nodes with "identical target + args + kwargs", explicitly skipping "random ops, mutable ops, higher-order ops"; motivating examples had 24-32 identical kernel launches collapsed to 1. This is *static op* dedup pre-codegen, not runtime value dedup.
   Sources: https://github.com/pytorch/pytorch/pull/184991 , https://github.com/pytorch/pytorch/issues/180957
4. MLIR `-cse` likewise eliminates common sub-expressions at IR level using Memory SideEffect safety info.
   Source: https://mlir.llvm.org/docs/Passes/
5. Molecular-simulation neighbor dedup in JAX-MD (segment_sum-based pair dedup) and retrieval-augmented batching precedents: could not be fetched within timebox — **UNVERIFIED**, treat as open literature check.
6. Design consequence: our dedup-gather must follow the JAX convention of static worst-case sizing (like `unique(size=...)`/BCOO `nse`): compute `unique`-style inverse indices with a static upper bound on group count, then `take`/gather unique rows and scatter-add results back.

## Q5. Correctness hazards catalog for value memoization

1. **PRNG keys as cache inputs — semantically sound.** Docs guarantee: "feeding the same key object to a random function will always result in the same sample being generated" (keys are pure, immutable inputs). So key-bytes -> result caching is correct *for a fixed JAX version/RNG implementation*; the same page notes keys carry a PRNG-implementation dtype (`key<fry>`) and points to JEP 9263 (typed keys) / JEP 263 (threefry design) — historical RNG/key-representation changes mean cache entries should embed the key dtype/JAX version or be invalidated on upgrade.
   Source: https://docs.jax.dev/en/latest/random-numbers.html
2. **Float nondeterminism across compiles/runs/devices.** XLA documents two independent GPU nondeterminism sources: (i) compile-time autotuning picks different kernels run-to-run (different reduction orders => different floats); (ii) runtime atomics in scatter/reductions vary scheduling per execution; mitigations `--xla_gpu_exclude_nondeterministic_ops`, `--xla_gpu_autotune_level=0`, persistent autotune cache — each with documented slowdowns or hard failures. Consequence: a cached tensor may bitwise-diverge from a fresh recomputation even for identical inputs; equality must be defined numerically or determinism flags required.
   Source: https://raw.githubusercontent.com/openxla/xla/main/docs/determinism.md
3. **Weak types.** jax.Arrays carry a `weak_type` flag (visible e.g. as `Array(0., dtype=float32, weak_type=True)` in official optimization_barrier example output). Two values with equal dtype+bytes but different weak_type are not interchangeable under type promotion. Cache key must include weak_type alongside dtype/shape. Dedicated weak-type doc page could not be located (404s) — full semantics **UNVERIFIED** beyond the flag's existence.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
4. **Donated buffers invalidate cached inputs/outputs.** Donating passes ownership; reuse raises errors (see Q2.7). A memo layer must either refuse to cache donated arguments or copy before handing to donating call sites; conversely cached outputs returned to users can be donated downstream, corrupting the cache.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
5. **Async dispatch: retention is safe, reading is blocking, memory is real.** Outputs are futures ("a value that will be produced ... isn't necessarily available immediately"); they can be stored and passed to further computations freely; only host inspection blocks. Retaining them (as a cache does) pins device memory indefinitely — eviction policy is a correctness-adjacent requirement, not just perf.
   Source: https://docs.jax.dev/en/latest/async_dispatch.html
6. **Sharding provenance.** Compiled input/output shardings are inferred from the argument shardings when unspecified; a cached output therefore carries whatever sharding existed at compute time and may mismatch a consumer expecting a different layout/device.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html
7. **Cross-version/bit-identity drift** (composite of 1+2): joblib documents that hashes invalidate wholesale across dependency upgrades; the analogous rule for us: version-stamp cache keys with (jax, jlib, backend, device kind) — mirrors what JAX itself does for compilation-cache keys.
   Sources: https://joblib.readthedocs.io/en/stable/memory.html , https://docs.jax.dev/en/latest/persistent_compilation_cache.html

## Q6. Tools that report "wasted duplicate compute"

**Finding: None found that report jaxpr-level duplication. Profilers see post-CSE kernels; no jaxpr linter for duplicate ops surfaced.**

1. Official instrumentation is `jax.profiler.trace` (CPU/GPU/TPU activity incl. Python + on-device ops; TensorBoard/Perfetto viewing). Because XLA CSE runs before codegen (Q1.3), duplicated-but-elided ops are invisible post-compilation; only cross-call repeated kernels appear.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html
2. Ecosystem-gap corroboration: PyTorch needed a dedicated 2025-26 effort for missing CSE (issue titled "Missing CSE in TorchInductor"), i.e., even adjacent ecosystems lack routine wasted-compute reporting.
   Source: https://github.com/pytorch/pytorch/issues/180957
3. `optimization_barrier`'s docs explicitly tie CSE-blocking to rematerialization — the closest shipped "recompute vs remember" diagnostic surface.
   Source: https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
4. Existence/status of third-party "jaxlint"-style tools on PyPI: could not be verified (PyPI blocked, search engines down) — **UNVERIFIED**. Practical detector approach remains bespoke: walk `make_jaxpr` output / pre-opt StableHLO, count structurally-identical primitive applications with equal operands (the same equivalence XLA uses per hlo_cse.cc).

---

## Implications for spec (design-shaping facts)

1. Build duplicate-op detection at jaxpr/StableHLO level ourselves; no public JAX API performs CSE, and `optimization_barrier` is the only sanctioned interaction point ([Q1]).
2. Pre-opt IR is a stable identity concept upstream (persistent cache hashes non-optimized HLO): our duplicate reports should key on the same pre-opt representation so reports stay stable across XLA versions ([Q1.5]).
3. Memoize *compilation-shaped* metadata the way JAX does (structure/shapes/dtype/static-hash + jaxlib/backend/device stamp); extend with content-hash only at explicitly opted-in boundaries ([Q2.1, Q5.7]).
4. Any disk-backed memo store is arbitrary-code-execution-trusted (pickle/executables) per both JAX and joblib warnings; spec needs a trust/permission story ([Q2.2, Q2.4]).
5. Hash recipe: (shape, dtype, weak_type, endianness) + blake3/xxhash over transferred bytes; canonicalize or document -0.0/NaN-bit behavior; expect D2H transfer to dominate cost ([Q3]).
6. Same-PRNG-key memoization is semantically correct but must be version-stamped (typed-keys/threefry history) ([Q5.1]).
7. Bitwise reuse guarantees require opting into XLA determinism flags and fixing autotune choices; otherwise define hit-equality numerically ([Q5.2]).
8. Never retain donated buffers; copy-on-cache or refuse donation; assume cached outputs may be donated downstream ([Q5.4]).
9. Dedup-gather must use static worst-case sizing (`unique(size=)`, BCOO `nse` conventions) with sort+inverse-gather+scatter-add; dynamic-sized dedup cannot cross jit ([Q4.1, Q4.2]).
10. There is no ecosystem tool for wasted-compute reporting; a jaxpr duplicate counter would be novel and cheaply implementable with hlo_cse.cc-style equivalence (opcode+shape+operand-ids, commutativity-insensitive) ([Q6, Q1.3]).

## Unresolved / follow-up

- Exact XLA flag(s) to disable HLO CSE (`--xla_disable_hlo_passes=...`) — flag list page unfetchable this session.
- BLAKE3/xxhash numeric throughput on typical hosts — figures exist only in image benchmarks/paper.
- JAX-MD neighbor-pair dedup internals; retrieval-augmented batching dedup literature.
- Third-party jaxpr linters / "jaxlint" ecosystem status; exhaustive confirmation that no content-addressed device-array store exists.
