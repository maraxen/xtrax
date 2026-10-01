# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`xtrax-io-on-shared-filesystems` skill** (`agent_assets/skills/`). Reference for
  choosing a read pattern on NFS/Lustre/GPFS: cost is requests x latency, not bytes;
  what `md.iterload(stride=k)` and an offset-table seek reader actually read; the
  CPU-bound check first, then `nfsiostat` per-op RTT/kB-op (not `/proc/<pid>/io`
  call size, which stdio buffering makes small even for good sequential reads) to
  separate latency-bound from bandwidth-bound; stage-by-large-copy, reduce-once/read-many and
  filesystem-keyed reader selection; and the same-filesystem A/B gate before an
  I/O-path change ships. Baseline agents without it misstated what a strided
  `iterload` reads and proposed adding parallel readers to a bandwidth-bound pool.

### Removed

- **The deprecated `SafeMap` / `SafeMapIterator` / `safe_map` aliases** (#5680). They
  shipped for one release (0.4.0a11) as `DeprecationWarning` aliases of `ChunkedMap` /
  `ChunkedMapIterator` / `chunked_map` (#3644) and now raise `AttributeError` /
  `ImportError`. **Breaking.** Migrate with the ast-grep rules in
  `codemods/safemap-to-chunkedmap/`. Name-based strategy matching (`stages.topology`,
  `export.composer`) still accepts the class name `"SafeMap"`, for consumers' own
  duck-typed classes of that name (aminx's, until its deprecation). The port wave now
  targets `xtrax.transforms.map.chunked_map` and is re-sealed (manifest hash recomputed
  with `scripts/audit_port_oracle_seal.py`; the reference oracle is unchanged).

## [0.4.0a11] - 2026-10-01

### Added

- **`xtrax.export.divergence` / `xtrax.export.rings`: divergence mapping** (#5095, #156).
  Locates *where* a compiled artifact departs from production JAX, instead of
  `parity.compare`'s single scalar. `compare_pytree` reports per-leaf `LeafDivergence`
  (ULP distance for floats; exact match for integers and bools, with no budget, ever),
  and `classify_probes` labels each named probe with a `DivergenceClass` in precedence
  order (`UNCOMPARABLE`, `DISCRETE_FLIP`, `INJECTED`, `AMPLIFIED`, `ATTENUATED`, `CLEAN`)
  against fusion-noise budgets measured on the host (`budget_leaf`, `budget_key`). The
  ring runners `r0_replay_gate`, `r1_target_isa`, `r2a_fusion`, `r2b_lowering`, `r3_probe`
  and `run_ladder` separate replay, target-ISA, fusion and lowering effects. Ring
  execution needs the `export` extra; `divergence` itself imports no IREE.

- **`xtrax.export.ONNX`: an ONNX export target, verified on ONNX Runtime** (new `onnx`
  extra: `jax2onnx>=0.17.0,<0.18`, `onnx`, `onnxruntime`; no IREE). `export_pipeline(...,
  targets=(ONNX,))` converts the composed callable with jax2onnx at opset 23 and runs the
  `.onnx` file on ORT's CPU execution provider against `reference_fn`. The target is
  `EXECUTED`. Parity is per output leaf (`LeafParityResult`): integer and bool leaves must
  match exactly and every leaf must keep its dtype. `ExportResult.onnx_census` records where
  int64 occurs inside the graph (from `TopK`/`ArgMax`); graph I/O keeps JAX's dtypes.
  `Target.backend` (`Backend.IREE` / `Backend.ONNX`) now selects the export-safety op
  rules. The four IREE rules apply to IREE targets only, because stable sorts and
  `lax.top_k` were measured exact on ORT. `ONNX` adds `"onnx-in-graph-rng"`
  (unsuppressible), because jax2onnx lowers `jax.random` draws to key-ignoring ONNX
  `RandomUniform`. Conversion refuses `jax_enable_x64`, and it undoes the `jnp.cumsum`
  patch that a process's first jax2onnx call leaves behind. `IREE_TARGETS` is the five
  IREE targets; `ALL_TARGETS` adds `ONNX`; `compile_for_target` refuses a non-IREE target.
  `ExportResult.artifact_bytes` is a backend-neutral alias of `vmfb_bytes`. Measured in
  `.praxia/docs/research/260930_onnx-route-spike.md`. The verification covers ORT CPU
  only, not ORT Web.

- **`xtrax.tiling.dedup_synthesis.verify_dedup_outputs(spec, fn, xs, *, rtol, atol)`**
  (#5217): checks claim (ii), that dispatching `fn` through the dedup path
  (vmap over canonical rows, then gather) reproduces per-row `jax.vmap(fn)(xs)`. The
  comparison is numeric (`allclose`, NaN equal to NaN; integer and bool outputs exact),
  never bitwise: XLA fuses the two programs differently, and float32 outputs legitimately
  differ (3.3e-6 measured at N=2000, K=7). It raises `DedupOutputMismatchError` with
  the worst error and first bad row. Both paths are jitted by default (`jit=True`), as
  production dispatch is, and equal infinities compare equal. It compares on device and
  moves one mask through the module's single `_to_host` route. It complements `verify_dedup_spec`,
  which checks input-row identity.

- **`MemoPolicy(execute_screened=True)`** (`xtrax.inference`, #5233): on a cache miss,
  run the jitted **screened program** of that call signature instead of the function
  eagerly, so a path the trace never took (`isinstance(x, jax.core.Tracer)`, identity
  checks, a caught `ConcretizationTypeError`) cannot be what gets cached. The runner
  is built from the same closed jaxpr the screen inspected; if it has to be rebuilt,
  a re-trace whose program differs from the screened one raises `MemoStalenessError`.
  Spot checks under this policy re-run the screened program, not the function
  eagerly. Opt-in; the default is unchanged. Costs one compile per signature, and a
  miss returns fresh output buffers, never an argument's.

- **`xtrax.profiling.loop_scaling`: flag loop bodies whose per-iteration cost grows with
  the loop's own extent** (debt #1983). `loop_bodies(fn, *args)` traces `fn` (nothing
  is compiled or run) and reports every `scan`/`while` body at any depth with ONE
  iteration's work and the trip count -- the multiplication XLA's `cost_analysis`
  hides by counting a `while` body once. `extent_scaling_report(fn, make_args, n)`
  traces at `n` and `2n` and flags bodies whose per-iteration work grows (ratio near
  2.0 for full-extent recompute, near 1.0 for incremental), the O(L^2)
  autoregressive-sampler shape that correctness tests and small-L wall clocks miss.
  In-place `dynamic_update_slice`/`scatter` are costed by their update, so a correct
  incremental body writing its row into an `(L, ...)` buffer is not a false positive.
  Work is a ratio-oriented proxy (dot_general FLOPs + output elements), not a
  FLOP-exact model. The debt's third check (consumption ratio) is not implemented.

- **`xtrax.tiling.dedup_synthesis.verify_dedup_spec`**: checks a `DedupSpec`'s
  row-identity claim (spec §4.3/§10.2-10.3, backlog #5172). Every row is
  compared **bitwise, per leaf, in native byte layout** against its claimed
  canonical row — the same byte definition `synthesize_dedup_spec` uses, so
  the two functions cannot drift. A mismatch in *any* leaf fails the row
  (union semantics); `DedupSpecVerificationError.first_bad_row`/`n_bad`/
  `leaf_index` describe the first failure. Structural checks (array type,
  integer dtype, `k`/length/bounds) run first, entirely on host, before any
  device→host transfer; the row check then moves exactly one `(N, L)` boolean
  mismatch mask (`N * L` bytes). Checks row-equality only (claim i); the
  numeric compute-equivalence claim (ii, "does re-running `fn` on the deduped
  rows reproduce `fn`'s full output") needs `fn` and is follow-up #5217.
- **`xtrax.inference` donation rejection, both directions**: `memoize_jaxpr`
  now rejects, at admission, any wrapped function whose traced jaxpr carries a
  donation marker on any equation at any nesting depth — `jit`/`pjit`
  `donate_argnums`/`donate_argnames`, and `device_put(..., donate=True)` via
  its `copy_semantics` operand. Previously only the output side was half-built
  and the input side was entirely unchecked (`test_donation_rejected_at_wrap`
  asserted an unrelated policy field and never exercised donation). New
  `MemoDonationError(MemoImpurityError)`, exported from
  `xtrax.inference.errors.__all__` and `xtrax.inference`, carries a structured
  `.sites` tuple naming each offending equation, its carrier
  (`"donated_invars"`/`"copy_semantics"`), and — for top-level equations —
  which flattened input leaf it aliases. Two known fail-open admission paths
  are not fixed (the screen runs once, keyed off the first call's shape, and
  never traces kwargs) and are pinned with strict `xfail` markers citing
  #5214/#5215.

### Fixed

- **`ZarrStagingSink.drain` errors name the payload that failed** (`xtrax.run`, #5552).
  A zarr write error now carries the staged key, group path, array name, shape and
  dtype in its message, on the same exception object (type unchanged), and says the
  pending buffer was not cleared. The context is in the message itself because JAX's
  `io_callback` re-renders only the original exception's message line, not notes or
  chained causes; previously a drain from the activation-capture path surfaced as an
  opaque `CpuCallback error` naming only zarr internals.

- **CLI shape specs: the documented form works and errors show a real example**
  (`xtrax.cli`, #5174). `<dtype>` in the grammar was a placeholder but read as literal
  syntax, so `x=(4,3)<float32>` was rejected. The parser now accepts the long aliases
  `float32`/`float64`/`int32` and an optional `<...>` around the dtype, reports a
  missing dtype specifically, and every error message shows `e.g. x=(4,3)f32`.

- **`BatchPlanner.plan()`: duplicate `DedupSpec`s raise; dedup never makes a budget
  plan infeasible** (`xtrax.tiling`, #5175). Two `DedupSpec`s for one axis now raise
  `DedupSpecCollisionError` (plan() routes through `merge_dedup_specs`) instead of
  silently keeping the last. **Behaviour change** for callers passing duplicates. In
  joint-budget mode a DedupGather axis was a fixed decision, so adding a `DedupSpec`
  could turn a plan that fit into `BudgetInfeasibleError`. If every ordinary demotion
  still leaves the plan over budget, dedup axes are now planned without dedup as a
  last resort, one at a time in spec order until the plan fits, each with a
  `RuntimeWarning` and `"DedupSpec dropped"` in the decision's `reasoning`. The error is raised only if that fails too.

- **`memoize_jaxpr` purity screen audited against the primitives JAX registers**
  (`xtrax.inference`, #5234). Five of the eleven banned names were registered by
  neither JAX 0.10.2 nor 0.11.1 (the "stateful" list banned nothing at all). The screen
  now rejects `debug_print` and `debug_callback` (a hit would skip the side effect),
  `random_gamma`, `rng_uniform`, `threefry2x32`/`threefry4x32` and `philox2x32`/`philox4x32`,
  and **a function that
  closes over a mutable `jax.Ref`**, which previously crashed in the program digest
  with `ValueError: Out of bound indexer`. Key plumbing on a key argument
  (`random_split`, `random_fold_in`, `random_wrap`, `random_unwrap`, `random_clone`)
  is admitted explicitly. Tests fail if any listed name stops being registered, or if
  a registered random/rng/callback/debug-family primitive is left unclassified.
  **Behaviour change:** a memoized function that calls `jax.debug.print` is now rejected.

- **Chunked mapping handles a ragged final chunk** (`xtrax.transforms`,
  `xtrax.tiling`, #5565). An axis whose cardinality is not a multiple of its batch
  size used to plan as SafeMap with a `RuntimeWarning`, then raise `ValueError` at
  dispatch. Any data-driven axis could hit this; an MSA depth of 50,713 = 13·47·83 has
  no usable divisor. `jax.lax.map(batch_size=...)` already runs the remainder as one
  smaller vmapped chunk, with no padding and peak memory still bounded by the batch
  size. xtrax's own divisibility check was the only obstacle, and it is removed (verified
  equal to `vmap` on JAX 0.10.2 and 0.11.1, including n=50,713 with batch 512). The
  planner no longer warns, and the decision's reasoning notes the ragged remainder.
  **Behaviour change:** a non-divisible axis whose `memory_estimator` says it fits now
  plans as `Vmap`, like a divisible one. The old Rule 5 forced chunking there only
  because dispatch was going to fail.

- **`ZarrStagingSink.drain` stores 0-d and zero-length payloads** (`xtrax.run`, #161).
  Chunks are now rank-matched (`tuple(max(d, 1) for d in shape)`); previously a
  0-d per-step scalar, or any array with a zero-length dimension such as
  `(0, 3)`, raised inside zarr -- surfacing from `io_callback` as an opaque
  `JaxRuntimeError: INTERNAL: CpuCallback error calling callback`.
- **`synthesize_dedup_spec` row identity is now exact native bytes per leaf**,
  not numpy-float equality over a promoted, dtype-concatenated view (spec
  §4.3/§5, backlog #5172). The predecessor's `jnp.concatenate` of raw leaves
  silently merged rows that are not the same bits:
  - `+0.0`/`-0.0` compared equal (`np.unique(axis=0)` merges signed zeros) —
    they are now distinct, so `k` may rise, possibly past `max_unique_k`
    (turning a `"synthesized"` result into `"k_over_limit"`).
  - Mixed-dtype leaves (e.g. an int32 leaf beside a float32 leaf) were
    promoted to a common floating dtype before comparison, silently
    collapsing distinct integers (`2**24` and `2**24+1` compared equal).
  - A numpy int64/float64 leaf was truncated to int32/float32 by the implicit
    `jnp.moveaxis` conversion under x64-off, silently merging wide values.
  - `axis != 0` read the wrong dimension for `N` (`stacked.shape[axis]` on an
    already batch-first array), producing a spurious `ValueError` or a
    garbage duplication ratio; `axis != 0` now works correctly.
  - `transfer_bytes_spent`/`k_bucket_bytes` now report true per-leaf byte
    widths instead of the promoted-dtype width (e.g. an int8 leaf beside a
    float32 leaf now counts 5 bytes/row, not the promoted 8).
  - Bitwise-identical NaN rows now deduplicate (`k` may fall), which is also
    now sound rather than an artifact of promotion.
  - Typed PRNG key leaves are now refused with `DedupSynthesisUnsupportedError`
    naming `jax.random.key_data`, where they used to raise a raw `TypeError`
    from deep inside JAX. Sub-byte integer leaves (int2/int4/uint2/uint4) are
    compared exactly after a lossless widening `astype`; sub-byte float leaves
    (float4/float6) are refused rather than silently widened through a lossy
    numeric conversion.
  Every change is in the sound direction (see risk table, spec §12); none is a
  regression. `tests/tiling/test_dedup_synthesis.py` (PR #104) passes
  unmodified against the new implementation.
- **`added-types-diff` gate no longer trips on nested `def`s** (backlog #5205,
  verified live on PR #156's `visit` closure inside `_topological_order`):
  `_PublicFunctionCollector.visit_FunctionDef` now stops descending into
  function bodies after recording a public-named def, so a nested def is
  never mis-reported as "unable to locate callable", and no longer silently
  overwrites a same-named top-level def's entry in the gate's base/head maps
  — previously this could either spuriously flag an unrelated top-level def
  as changed, or, more seriously, hide a real signature change to it (both
  base and head maps held the unchanged nested def under that key). A
  factory pattern (`foo = _make_foo()` with a nested `def foo`) is now
  invisible to the gate rather than accidentally loud; covering assigned
  callables would be a gate feature, not this fix.

### Changed

- **Divergence rungs R1 and R2b report per-probe divergence; `UNCOMPARABLE` class**
  (`xtrax.export`, #5210, #5209, #157). Both rungs used to return `passed=True` with
  `probes=()` for arbitrarily large value divergence, so `all(r.passed for r in
  results)` read an integer index flip as "no divergence". `passed` stays structural
  (the precondition held and every leaf was comparable). `r1_target_isa` and
  `r2b_lowering` now take `probe_deps=` (forwarded by `run_ladder`) and populate
  `probes` with `classify_probes` against R2a's budgets; check `probes`, not `passed`,
  for value divergence. A probe with a leaf that could not be compared (shape/dtype
  mismatch) is now `UNCOMPARABLE` rather than `AMPLIFIED`.

- **`memoize_jaxpr` cache hits are cheaper** (`xtrax.inference`, #5241, #160): arguments
  are flattened once per call (was three times), the purity and donation screens walk
  the jaxpr once, and integer bounds are cached. Keys and screening results are
  unchanged (pinned against the previous implementation).

- **`export_pipeline` runs every target's safety gate first, then evaluates `reference_fn`
  once, then compiles**, rather than gating, compiling and calling `reference_fn` per target. That makes the oracle independent of
  toolchain side effects on `jax.numpy`.
- **`[tool.uv] override-dependencies` lifts jax2onnx's `orbax-checkpoint<0.11.37` cap** in
  xtrax's own lock, so the `onnx` extra adds packages without pulling every environment
  down to orbax 0.11.x (the lock keeps 0.12.1). The ONNX and checkpoint suites pass on orbax
  0.12.x. A `pip install xtrax[onnx]` from PyPI still gets jax2onnx's own cap.

- **`SafeMap` → `ChunkedMap`, `SafeMapIterator` → `ChunkedMapIterator`, `safe_map` →
  `chunked_map`** (#3644). `safe_map` already means something else in JAX
  (`jax._src.util.safe_map`, a length-checked map), and "Safe" suggested the
  `xtrax.safety` subsystem; the strategy is memory-bounded chunking. **The old names
  import for one release as deprecated aliases** (each access raises
  `DeprecationWarning`); they are removed in the release after. Migrate mechanically
  with `codemods/safemap-to-chunkedmap/`: ast-grep rules for Python plus a Markdown
  script, with a fixture pair showing the exact transformation (see its README).
  **Consumers that dispatch on the strategy's class *name*** (aminx's
  `kernel_dispatch.py` compares `type(strategy).__name__ == "SafeMap"`) must accept
  `"ChunkedMap"` before upgrading. The alias constructs the new class, and the codemod
  rewrites that comparison. xtrax's own name-based matching accepts both names for
  this release. Strategy-name strings in EDA output (`strategy_counts` keys, the
  `strategy` column) are now `"ChunkedMap"`, and `RuntimeBundle.iterator`'s annotation
  names `ChunkedMapIterator`. The sealed `port/` apparatus still
  references `xtrax.transforms.map.safe_map` through the alias, so it must be re-sealed
  before the aliases are removed.

- **Dependency floors raised to versions that actually work** (debt #738). With
  jax at its own floor (0.10.2), `equinox>=0.11.0` and `orbax-checkpoint>=0.6.0`
  could not `import xtrax` at all (`jax.core.Primitive`, `jax.sharding.PositionalSharding`
  and `jax.experimental.layout.DeviceLocalLayout` were removed from jax). New floors,
  each the lowest release that installs on Python 3.13 and passes the
  dependency-sensitive suite at jax 0.10.2: `equinox>=0.13.1`,
  `orbax-checkpoint>=0.11.17`, `numpy>=2.1` (already forced by jax), and in the
  extras `zarr>=3.0.8` and `tyro>=0.9.1`. `optax>=0.2.3` is unchanged and verified.
  A new `dependency-floors` CI job resolves core + io + cli with
  `--resolution lowest-direct`, fails if any floor is not exactly what installed
  (`scripts/check_dependency_floors.py`), and runs the tests there.
- **`xtrax.checkpoint` moved off orbax's deprecated `CheckpointManager` kwargs**
  (debt #739). `max_to_keep`/`keep_period` become an `AnyPreservationPolicy` of
  `LatestN` + `EveryNSteps` (same retention: the latest N plus every multiple of
  the period), `item_handlers=` becomes a `handler_registry`, and `items=`
  save/restore becomes `args=`. The public signatures and on-disk layout are
  unchanged, and checkpoints written by earlier releases restore as before
  (pinned by a test). orbax's v1 API is not adopted: as of orbax 0.12.1 it is
  still `orbax.checkpoint.experimental.v1`.
- **`ZarrStagingSink` enforces `extension_schema` `required` fields at `drain()`,
  not per `stage()` call** (`xtrax.run`, debt #1540). Per-call enforcement
  contradicted the documented merge-on-repeat contract: a key whose required
  fields arrived in a later `stage()` call was rejected. Value-type checks stay
  fail-fast at `stage()`. `drain()` checks every key with staged attrs against
  its on-disk attrs merged with pending ones, raises before writing anything,
  and leaves the buffer intact. An auto-flush (`flush_every`) is a drain, so a
  split must complete within `flush_every` calls.

## [0.4.0a10] - 2026-09-13

Release theme: the export boundary now refuses, at plan time, the constructs
measured to break through IREE — rather than letting them compile and diverge
or abort. Four of the five rules below were found by measuring a real model's
exported artifact against eager JAX, and two of them produce **no compile
error and no exception**: the artifact runs and returns wrong integers, which a
float-tolerance parity check reports as `max_abs_diff 0.0` and passes.

### Added

- **Export-safety rule `"unlegalizable-op"`**: refuses `jax.lax.top_k`, which
  IREE's StableHLO importer rejects outright on every target (it lowers to a
  `stablehlo.composite` wrapping `chlo.top_k`, marked explicitly illegal). This
  rule is **not** suppressible — no caller can accept a risk that is a hard
  compile failure. The suggested replacement,
  `jnp.argsort(-x, axis=-1, stable=True)[..., :k]`, itself trips the next rule.
- **Export-safety rule `"sort-stability"`**: flags a stable sort/argsort, whose
  tie order IREE does not preserve relative to XLA. Measured: an integer
  `sort_key_val` with 64 slots and 4 distinct keys differs at **45 of 64**
  positions. Nothing crashes and no float leaf changes, so a parity gate built
  on float magnitude cannot see it at any tolerance. Fold an explicit index
  tiebreak into the sort key rather than relying on backend stability.
- **Export-safety rule `"random-permutation"`**: flags `jax.random.permutation`,
  which IREE compiles into a valid but **wrong** permutation when the key is a
  split half and the sibling half is also consumed — 248 of 256 positions
  differ. Suppressible via `acknowledged`. It is **necessary but not
  sufficient**: the underlying defect is the split-derived key, not the
  permutation, so a clean run of this rule does not certify a model's
  randomness. Reported upstream as
  [iree-org/iree#24927](https://github.com/iree-org/iree/issues/24927).
- **Export-safety rule `"unbatched-threefry-key"`**: flags threefry driven by a
  key that arrives as a **runtime input** with at most one lane. Unlike the
  rules above this is not a wrong answer but a **hard abort at invocation** —
  IREE's HAL rejects the command buffer with a bogus `2**39`-ish length against
  a 192-byte binding. The boundary is the lane count, not the rank of the key
  input and not `vmap`-versus-not: an un-batched draw aborts from `(2,)`,
  `(1,2)` and `(4,2)` inputs alike, one lane aborts, two or more is bit-exact.
  A **constant** key is genuinely exempt (it stays exact under
  `--iree-opt-const-eval=false`), so the rule tests input provenance rather
  than matching the primitive alone. Two measured workarounds, both named in
  the blocker text: give the key two or more lanes, or set
  `jax_threefry_partitionable=False`. Reported upstream as
  [iree-org/iree#24929](https://github.com/iree-org/iree/issues/24929).
- **`xtrax.export.targets.NATIVE_PORTABLE`**: a fifth export target, `"native-portable"`,
  `EXECUTED` like `NATIVE` but compiled with a fixed `--iree-llvmcpu-target-cpu=x86-64-v2`
  baseline instead of `=host`, so the artifact runs on a recipient's CPU rather than only
  the one that compiled it (any x86-64 CPU from roughly 2013 onward — not "any CPU since
  2009"). It still requires an IREE runtime on the recipient's machine; it is not a
  standalone binary. `NATIVE` is unchanged and remains host-tuned, for use as a parity
  oracle only.
- **`export-runtime` extra**: `iree-base-runtime`, `safetensors` and `huggingface_hub` —
  what a consumer needs to *load and run* an exported artifact. The existing `export`
  extra now aliases it and adds `iree-base-compiler`, which is what xtrax needs to
  *build* one. The compiler is 349 MB installed against ~7 MB for the runtime, so a
  downstream package that only executes artifacts no longer pays for a toolchain it
  never invokes.

### Fixed

- **Pytree inputs are flattened at vmfb invocation.** `export_pipeline`'s
  EXECUTED path passed a pytree straight to the runtime, which accepts only
  flat arrays, so any model whose entry point takes a structured input failed
  at invocation rather than at export — late, and with an error naming the
  runtime instead of the call site.
- **A materializing inner sink is refused on the literal-`vmap` route.**
  `compose_vmap_of_scan`'s literal-`vmap` route discarded the materialized
  `ys` and returned only the final carry, so a sink declared `materialize=True`
  silently lost every value but the last. It now refuses rather than
  under-reporting.
- **`zarr_content_digest` now excludes provenance attrs by default**: run-ID,
  git SHA/branch/dirty, and creation timestamp are excluded from the digest
  unless explicitly included via the new `include_provenance=True` parameter
  (same parameter added to `update_zarr_node_digest` and `run_repro_floor`, the
  latter so a caller holding a digest pinned before this change has a way to
  reproduce it). This restores the
  intended contract that digest values are unaffected by which process or
  session wrote the store. **Digest values computed before this change do not
  match values computed after it for any sink-written store** — stored done-marker
  digests will mismatch and must be recomputed.
- **`export-toolchain-tests` now fails on a skipped test.** The job ran a bare
  `pytest tests/export/ -q`, which was exactly as green with every export test skipped
  as with all of them run — so the "tested against real IREE 3.11" claim it exists to
  make had nothing enforcing it. A silently-failed toolchain install now fails the job
  instead of passing it.
- **Corrected the 0.4.0a8 entry below**, which described `WASM32` as `EXECUTED`. It has
  always been registered `CODEGEN_ONLY` (`targets.py`); executing a wasm artifact needs
  an emsdk-built IREE runtime that no published package provides. The code was right and
  the changelog was wrong, in the direction that overstates what ships.

### Changed

- **`grain` and `pytest-asyncio` are no longer runtime dependencies.** Both were
  declared in `[project].dependencies` but imported nowhere under `src/`.
  `grain` moves to a new `data` extra — consumers who need it must now install
  `xtrax[data]`. `pytest-asyncio` is removed entirely from runtime deps; it
  remains in the `dev` extra, where a test-only plugin belongs. **This is
  consumer-visible:** anyone relying on `pip install xtrax` to pull `grain` or
  `pytest-asyncio` must now ask for them explicitly.
- **The `dev` and `eda` dependency-groups are now thin aliases of their
  matching extras** (`dev = ["xtrax[dev]"]`). Previously each group duplicated
  its extra's contents and had drifted apart, so `uv sync --group <x>` kept the
  group and dropped the extra — silently uninstalling `beartype`, `chex`,
  `interrogate`, `jaxlint` and `libcst`. An alias has no content of its own and
  so cannot diverge. A new contract enforces the shape for any name declared in
  both tables.

## [0.4.0a9] - 2026-09-07

### Added

- **`derive_sink_spec` and `new_run_id` on the public API**: both names now
  resolve from the `xtrax` root and are listed in `distribution/public_api.toml`
  as tier-1 exports. Backlog #4457 item (2) deliberately held them back until a
  real driver consumed the seam, so that the surface would be shaped by a caller
  rather than by speculation; the run CLI now builds its sink exclusively
  through `derive_sink_spec`/`make_sink`, which discharges that condition. They
  are promoted through the lazy-export path the contract requires, so
  `xtrax.derive_sink_spec` still imports `xtrax.run` only on first attribute
  access rather than at package import.

### Changed

- **The `controller` extra is capped at `bathos<0.14`** (was an uncapped
  `>=0.13.0a1`). Once CI depended on that extra, an uncapped alpha meant an
  unrelated bathos release could turn the board red without a commit here.


## [0.4.0a8] - 2026-09-02

### Added

- **`xtrax.export`**: compiles a `BatchPlan` to a standalone artifact via
  StableHLO and IREE. `export_pipeline` folds a plan into one traceable
  callable, exports it, compiles it for one or more targets, and verifies
  numerical parity against the original JAX callable where the target can be
  executed. Four targets ship — `NATIVE` (`EXECUTED`), `WASM32`,
  `VULKAN_SPIRV`, and `METAL_SPIRV` (all `CODEGEN_ONLY`) — with `VerificationLevel`
  recording how far each one is actually checked, so a `CODEGEN_ONLY` artifact
  never reads as verified. SPIR-V shaders are extracted from `vulkan-spirv`
  builds, magic-filtered so `metal-spirv`'s MSL dump is rejected rather than
  mistaken for a shader. The IREE toolchain is imported lazily behind the
  `export` extra: a missing toolchain surfaces at compile time with a clear
  error, never as an ImportError at module load.
  (spec: `.praxia/docs/specs/260901_xtrax-export-webgpu.md`)
- **Multi-axis export composition**: `build_traceable_callable` handles an outer
  `Vmap` axis wrapping an inner `Scan` axis, following the composition recipe
  certified by `tests/stages/test_nested_ordering.py` literally rather than
  routing through the executor's scan helper. A plan whose shape forces a
  literal `jax.vmap` around a lane-dependent ordered `Tap`/`Sink` is refused
  with `MultiAxisCompositionError`, carrying the executor's own guidance rather
  than a second paraphrase of it. Previously any plan with more than one axis
  was refused outright.
- **Export-safety gate** (`check_export_safety`, `validate_export_safe`): blocks
  BCOO leaves and out-of-envelope dtypes before any compile runs, reporting
  every blocker at once. Leaves reachable through `fn`'s closure are scanned
  alongside `abstract_inputs`, which is what catches a weight held in an
  Equinox module — the commonest shape, since weights are never arguments.
- **`materialize` boundary kind** (`xtrax.stages`): marks a sink whose values
  the export should carry as real outputs, stripped to a no-op inside the
  exported artifact so the compiled graph stays pure.
- **`load_hf_weights`**: loads safetensors checkpoints, casting any dtype the
  chosen target cannot verify and reporting every cast leaf untruncated. `f64`
  raises rather than casting silently.

### Changed

- The export dtype envelope is measured against IREE 3.11.0 rather than derived
  from a backend's published numeric model. Two results are worth stating
  because both contradict the obvious assumption: `f64` is rejected on **every**
  target including `NATIVE`, because IREE does not refuse it — `ConvertTypesPass`
  demotes it to `f32` and rewrites the entry point's public signature with a
  warning, which on a `CODEGEN_ONLY` target is never surfaced at all. And `bf16`
  splits by verification level rather than by backend: it compiles everywhere
  with its signature untouched, and fails only in IREE's runtime
  buffer-to-numpy mapping, so a `CODEGEN_ONLY` target carries it while an
  `EXECUTED` target cannot verify it.
- `huggingface_hub` pinned to `>=1,<2`. The previous `>=0.24,<2` spanned the
  1.0 break and was resolving across API generations.

### Fixed

- The release-readiness gate no longer requires markers for a staging index that
  was retired. `distribution/release_readiness.toml` asserted that
  `publish.yml` contained `publish-testpypi` and `test.pypi.org/legacy`, but the
  TestPyPI job was deliberately dropped on 2026-07-02 ("publishing to PyPI only
  by decision"). The gate had therefore reported `BLOCKED_AUTOMATED` for two
  months on a condition no longer wanted, with every other check passing.
  `just audit-release-readiness` was the only place that failure could appear,
  and nothing in CI ran it, so it surfaced only when someone went to cut a
  release. Its hermetic half now runs inside `audit-deterministic` as
  `audit-release-readiness-contract`, alongside every other
  `tests/distribution` gate, and the test asserts against the config's own
  marker list rather than restating it — the duplication is what let the two
  drift apart.

### Note

- No WebGPU target ships. IREE's Vulkan HAL passes dispatch parameters through
  push constants, which are not a WebGPU capability, so naga rejects every
  SPIR-V module IREE emits and no flag removes them. The question is tracked as
  research rather than left implied by a `VALIDATED` target that nothing
  registers.

## [0.4.0a7] - 2026-08-27

### Added

- **CSE runtime-optimization layer** (`xtrax.inference`, `xtrax.tiling`): `analyze_cse`
  detects structurally duplicate jaxpr equations via a union-find fixpoint pass
  (opcode + params + operand identity, matching XLA's `hlo_cse.cc` equivalence rule),
  exposed via a new `xtrax explain --report cse` CLI path. Dedup-spec synthesis
  (`xtrax.tiling.dedup_synthesis`) adds two-stage sample-then-exact duplicate
  detection for tiling batches, avoiding an O(N) device-to-host transfer unless
  sampled duplication justifies the exact pass. `xtrax.inference.memo` gains a
  `MemoizedCallable` Protocol and a dual-signature `@overload` for `memoize_jaxpr`.
  Jury-audited over two review rounds. (spec: `.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md`)
- **`xtrax.profiling` core module + `xtrax-probing` skill**: `ProbeRecord` schema
  with a fail-closed claim-validity contract (an unsupported claim raises at
  construction, never silently passes) upstreamed from prolix's measurement
  tooling, plus HLO-as-text + real Perfetto runtime-trace parsing (`trace.py`),
  a pytest-benchmark-to-`ProbeRecord` bridge (`bench.py`), and report generation.
  New `xtrax-probing` skill documents the contract, stage-0/1/2 probe drivers,
  and gate/controller integration.
- **`RunSpec.run_id` + `derive_sink_spec` seam** (`xtrax.run`): `RunSpec` gains
  an optional `run_id` (defaults to `None`) and a `new_run_id()` helper; the new
  `derive_sink_spec` seam threads run identity from `RunSpec` through to sink
  construction. The `xtrax run` CLI verb now persists this provenance
  automatically via the same seam.
- First real GPU stage-2 `ProbeRecord`s captured against real hardware (aminx
  L40S dogfood) — validates the stage-0/1/2 taxonomy end-to-end for the first
  time; coverage is currently single-GPU/single-vendor, broader multi-hardware
  campaign automation is tracked separately.
- **`xtrax-optimizing` skill + Tier-gated probe drivers** (`agent_assets/skills/xtrax-optimizing`,
  `scripts/prof_stage{0,1}_*.py`): three-tier taxonomy separating host-boundary
  mechanics (ordered/unordered Tap/Sink cost), data movement
  (`async_indexed_stream` prefetch overlap), and composition-level changes
  (on-the-fly vs materialized one-hot), each exemplified by a driver that emits
  claim-valid ProbeRecords: `prof_stage0_onehot_cost` (never-execute cost
  analysis), `prof_stage1_onehot_micro` (named-scope attribution +
  parity gate before measurement), `prof_stage1_host_boundary` (correctness
  gate + per-variant dispatch counts), `prof_stage1_feed_overlap` (regime
  guard in-record). Smoke/unit pins in `tests/scripts/test_prof_optimizing_drivers.py`.
  (scope: `.praxia/docs/specs/260825_jax-optimizing-skill-scope.md`)
- **Sink provenance tracking** (`xtrax.run`): `ZarrStagingSink` now auto-captures
  static run provenance for downstream consumers. `SinkSpec` gains a required
  `run_id` and an optional JSON-Schema-style `extension_schema`. The store's
  root group receives the full record (`git_sha`, `git_branch`, `git_dirty`,
  `run_id`, `created_at` as ISO-8601 UTC); each drained key's group gets a
  minimal `run_id`/`git_sha` pointer. Git capture never raises (falls back to
  `git_sha="unknown"` with a `UserWarning`). Core field names are reserved
  against caller attrs; schema validation happens at `stage()` time. New
  `finalize()` method consolidates store metadata exactly once (refusing to
  run while staged payloads are undrained) and locks the sink; opening a
  second sink on the same `output_dir` with a different
  `run_id` now raises. (spec: #96, task 260824_default-sink-provenance-tracking)

### Changed

- **Breaking**: `SinkSpec.run_id` is now a required constructor argument.

## [0.4.0a6] - 2026-08-20

### Added

- **`WhileCarry`** (`xtrax.tiling`): `lax.while_loop`-backed `AxisStrategy`
  for carry-only loops (no input sequence, no per-step `ys`).
  `CarrySpec(collect_outputs=False)` pre-demotes to `WhileCarry`;
  `make_axis_dispatch` returns `WhileLoopIterator`. Not reverse-mode AD
  safe. (#81 spec, #83 implementation)

## [0.4.0a5] - 2026-07-08

### Added

- **`xtrax.run.zarr_integrity`** (`xtrax.run`): content-digest and durability
  primitives for Zarr directory stores, hoisted out of aminx's
  `host/campaign.py` -- they were fully generic (no domain logic) and
  belong in the shared layer. `zarr_content_digest(path)` computes a
  deterministic sha256 over a store's full logical content (paths, attrs,
  array data), unaffected by filesystem metadata or which process wrote
  it; `fsync_tree(path)` durabilizes a directory-of-many-files store
  bottom-up before that digest is trusted. Also exports the lower-level
  building blocks (`canonical_json_bytes`, `normalize_json_value`,
  `update_array_digest`, `update_zarr_node_digest`, `fsync_file`,
  `fsync_directory`) for callers assembling their own verification/durability
  logic on top. `zarr_content_digest`/`update_zarr_node_digest` require the
  optional `zarr` dependency at call time only (`pip install xtrax[io]`);
  importing `xtrax.run` never requires zarr installed.
  (`src/xtrax/run/zarr_integrity.py`, `tests/run/test_zarr_integrity.py`)

## [0.4.0a4] - 2026-07-07

### Added

- **`ZarrStagingSink.stage()` gains `attrs`** (`xtrax.run`): optional
  `dict[str, Any]` of JSON-safe scalar metadata written to the Zarr group's
  `.attrs` on drain, merging the same way staged arrays merge across
  repeated `stage()` calls for the same key. `take()` discards pending
  attrs (it returns the in-memory payload without persisting). Fills the
  gap for consumers whose payloads mix array data with small provenance
  metadata that has no array-shaped equivalent.
  (`src/xtrax/run/zarr_sink.py`, `tests/run/test_zarr_sink.py`)

## [0.4.0a3] - 2026-07-07

### Added

- **`xtrax.run.ZarrStagingSink`** (`xtrax.run`): a keyed staging buffer for
  JAX `io_callback`-driven streaming output, draining into nested Zarr
  groups. `SinkSpec.format` gains `"zarr"`; a new `make_sink(spec)` factory
  dispatches to the real implementation (`zarr`/`none` today; `jsonl`/`h5`
  remain routing-only stubs pending their own writers). Generalizes the
  keyed-staging-then-drain pattern used by consumers with per-chunk tensor
  payloads (sequences, logits, encoder intermediates, accumulated tensors) —
  domain-specific `io_callback` dispatch stays with the caller; this module
  owns staging and Zarr storage only.

  Zarr is a new optional extra (`pip install xtrax[io]`), imported lazily
  inside `ZarrStagingSink.__init__` — `xtrax.run` remains fully importable
  without it; only constructing a Zarr sink requires the dependency, with a
  clear `ImportError` pointing at the install command otherwise.
  (`src/xtrax/run/sink.py`, `src/xtrax/run/zarr_sink.py`,
  `tests/run/test_sink.py`, `tests/run/test_zarr_sink.py`)

## [0.4.0a2] - 2026-07-07

### Fixed

- **`StageBundle.__init_subclass__` validator** (`xtrax.stages`): three limitations
  that blocked domain code from adopting `StageBundle` were fixed.
  - **PEP 563 blindness**: modules using `from __future__ import annotations`
    left field annotations as unevaluated strings; the validator now resolves
    them via `typing.get_type_hints(cls, include_extras=True)` and raises a
    clear `TypeError` (naming the unresolved annotation) instead of silently
    misclassifying fields.
  - **Structural-callable `Protocol`s rejected**: fields typed as a
    `typing.Protocol` whose only member is `__call__` are now accepted as
    callable-shaped, alongside plain `Callable`.
  - **Union check hardcoded to exactly two args**: `X | None` unions now
    validate for any arity — N-args-with-exactly-one-`None`, not just the
    2-arg case — so e.g. `Callable | SomeProtocol | None` fields validate
    correctly instead of raising.
  (`src/xtrax/stages/bundle.py`, `tests/stages/test_bundle.py`)

## [0.4.0a1] - 2026-07-06

### Added

- **Joint-budget planning mode for `BatchPlanner`** (`xtrax.tiling`):
  `BatchPlanner(budget=MemoryBudget(bytes=..., estimate=...))` replaces the
  independent per-axis rules with whole-plan greedy demotion — every eligible
  axis starts at `Vmap`, then axes with `cardinality > default_batch_size` are
  demoted to `SafeMap` in the order specs were given until the joint estimate
  fits the budget. Callers express demotion priority by spec order. Strict by
  design: mutually exclusive with the per-axis `memory_estimator`, estimator
  exceptions propagate, and an unfittable plan raises `BudgetInfeasibleError`.
  Carry/dedup/bucket decisions stay fixed but participate in the estimate;
  budget-mode reasoning strings carry the byte numbers for `xtrax explain`.

  Native-tooling estimator building blocks (`xtrax.tiling.estimators`):
  - `device_memory_budget(fraction=0.9, device=None)` — budget bytes from the
    XLA allocator's `Device.memory_stats()["bytes_limit"]`; fails loud when
    the backend reports no stats.
  - `lowered_memory_estimate(fn, *abstract_args)` — AOT-compiles from
    `ShapeDtypeStruct`s and returns XLA's own buffer-assignment bytes
    (argument + output + temp) via `Compiled.memory_analysis()`.

  Exports are tiling-level (`xtrax.tiling`), same tier as `CarrySpec` — no
  root public-API change. Spec:
  `.praxia/docs/specs/260706_joint-budget-batch-planner.md`.
  (`src/xtrax/tiling/budget.py`, `src/xtrax/tiling/estimators.py`,
  `tests/tiling/test_budget_plan.py`, `tests/tiling/test_estimators.py`)

### Fixed

- **CI recovery**: install `just` via `uv tool` (runner image stopped shipping
  it); publish-OIDC gate aligned with the no-TestPyPI decision; coverage DAG
  gate now prints the pytest output tail on failure (failures were previously
  undiagnosable from CI logs); praxia-CLI emit smoke test skips when the
  binary is absent; CITATION/README version metadata synced to `__version__`.

### Changed

- **Docs positioning**: new `docs/why-xtrax.md` (why the tiling layer lives
  above the JIT boundary); README "Why xtrax?" and docs index rewritten to
  match; `.claude/workflows/port-validation.js` now tracked in-repo.

## [0.3.1] - 2026-07-02

### Added

- **Plan topology validator** (`xtrax.stages`): `PlanTopologyError` raised on
  invalid stage-plan topologies.

### Changed

- **Publish workflow**: straight to PyPI via OIDC Trusted Publishing;
  TestPyPI staging dropped by decision.

## [0.3.0] - 2026-07-02

### Added

- **`xtrax.eda` — EDA visualization subpackage** (optional extras: `pip install xtrax[eda]`):
  A two-layer exploratory data analysis interface for inspecting `BatchPlan` outputs from
  the tiling subsystem.

  _Stats layer_ (stdlib + numpy only, no extras required):
  - `extract_plan_stats(plan: BatchPlan) -> PlanStatsDict` — extracts strategy distribution,
    axis metadata, dedup/bucket statistics, and memory warnings.
  - `explain_plan(plan: BatchPlan) -> PlanStatsDict` — like `extract_plan_stats` with
    guaranteed non-empty `reasoning` strings per axis.
  - `analyze_dedup(decision: AxisDecision) -> DedupStatsEntry` — dedup ratio, padding waste,
    unique vs padded counts.
  - `analyze_bucket(decision: AxisDecision) -> BucketStatsEntry` — bucket boundaries and count.

  _Viz layer_ (requires `pip install xtrax[eda]`):
  - `render(plan, fmt, path, stats_transform, metadata, logger, panels) -> bytes | str | None`
    — single entry point for PNG (bytes), SVG (bytes), HTML (str) output. Seaborn/matplotlib
    backend; headless via `Agg`. Supports post-stats transform hook, JSON metadata sidecar,
    panel filtering, and a structural `PlanLogger` protocol for wandb/tensorboard adapters.
  - `plan_to_dataframe(stats: PlanStatsDict) -> pd.DataFrame` — one row per axis.

  _Types_ (no extras):
  - `PlanStatsDict` — fully-typed `TypedDict` for the stats surface.
  - `PlanLogger` — structural `Protocol`; xtrax never imports wandb or tensorboard.
  - `PanelName` — `Literal["strategy","cardinality","dedup","bucket","memory","reasoning"]`.

  (`src/xtrax/eda/`, `tests/eda/`, `docs/advanced/eda-guide.md`, `docs/api/eda.md`)

- **`xtrax.inference` — Signature inference subpackage** (Tier-1 MVP, E1):
  Zero-config axis detection and fail-loud semantics for batched JAX computations. Enables
  automatic extraction of output schemas and input axis specifications, with explicit role
  assignment (KNOWN via `@axis_config` or UNKNOWN with fail-loud guards).

  _Public API_:
  - `infer_bundle(fn, abstract_inputs, *, verify_against=None) -> (BundleSchema, list[AxisSpec])`
    — main entrypoint; infers output schema and axis specs from abstract inputs.
  - `@axis_config(*AxisOverride(...))` — decorator for Tier-1 axis resolution; attaches
    overrides positionally to leading axes; each override specifies `name` and required
    `default_batch_size` (Assumption A3: not inferable from shape alone).
  - `AxisOverride` — dataclass for single-axis configuration; fields include `name`,
    `default_batch_size`, `cardinality`, `tile_granularity`, `heterogeneous`, `dedup_eligible`,
    `bucket_boundaries`.
  - `BundleSchema` — output structure mapping field names to `ShapeDtypeStruct`; carries
    optional `carry_specs` list (deferred for T2+, always None in MVP).
  - `AxisRole` — Enum with KNOWN (axis resolved) and UNKNOWN (axis ambiguous, fail-loud);
    future tiers extend with concrete roles (BATCH, SEQUENCE, etc.).
  - `AmbiguousAxisError` — raised by `BatchPlanner.plan()` when axis role is UNKNOWN.
  - `StructureMismatchError` — raised when `verify_against` outputs diverge from abstract-traced.
  - `synthesize_axes(abstract_inputs, overrides=None) -> list[AxisSpec]` — lower-level factory
    that synthesizes AxisSpec with explicit role assignment.

  _Deferred (T2+)_:
  - Concrete axis roles (BATCH, SEQUENCE, FEATURE) with domain-specific planner behavior.
  - jaxtyping dimension-name adapter for role inference without decorators.
  - CarrySpec auto-derivation for RNN-like stateful axes.
  - LibCST Bundle codegen for boilerplate `@axis_config` and dataclass generation.

  (`src/xtrax/inference/`, `tests/inference/`, `docs/api/inference.md`)

- **`xtrax run` / `xtrax resume` / `xtrax sweep` — CLI training verbs** (E3):
  TOML-driven training, checkpoint resume, and local grid-search sweep.
  - `xtrax run config.toml` — resolve model/optimizer/loss/data from import paths, write
    manifest, train with orbax checkpointing.
  - `xtrax resume <run-id> --epochs N` — read manifest, reconstruct state from latest
    checkpoint, train for N additional epochs into a new sibling run dir.
  - `xtrax sweep sweep_config.toml` — sequential in-process grid search with atomic
    sweep manifest, JAX compilation cache reuse, and per-run fault tolerance.

  (`src/xtrax/cli/`, `tests/cli/`)

### Changed

- Distribution readiness: tiered coverage gates (`tier1_core` 90/80, `tier2_eda` 90/75),
  docs plumbing gate (`just audit-docs-build`), narrative docs gate (`just audit-narrative-docs`),
  output-sink docs gate (`just audit-output-sink-docs`), publish OIDC gate
  (`just audit-publish-oidc`), release readiness convergence audit
  (`just audit-release-readiness`), LibCST added-types diff gate, and deterministic audit
  track expansion (`just audit-deterministic`).
- `pyproject.toml`: Added `eda` to both `[project.optional-dependencies]` and
  `[dependency-groups]` (`pandas>=2.0`, `matplotlib>=3.8`, `seaborn>=0.13`).

## [0.2.1] - 2026-06-14

### Added

- **`CarrySpec`, `CarryShape`, `DedupSpec`** ported from `aminx.tiling` (T2.2-2.3): Three
  new types for declaring pre-committed axis strategies before the `BatchPlanner` budget
  loop.
  - `CarrySpec(axis_name, init, transition, ordered_sinks)` — declares an axis as a
    `jax.lax.scan` carry; `__post_init__` guards against heterogeneous axis names.
  - `CarryShape(name, shape, dtype)` — typed carry-buffer descriptor; `materialize()`
    returns a zero-initialized buffer.
  - `DedupSpec` + `get_k_bucket` — declares an axis for dedup-gather; `get_k_bucket`
    rounds cardinality up to the next power of 2.
  ([`src/xtrax/tiling/carry.py`](src/xtrax/tiling/carry.py),
   [`src/xtrax/tiling/carry_shape.py`](src/xtrax/tiling/carry_shape.py),
   [`src/xtrax/tiling/dedup.py`](src/xtrax/tiling/dedup.py))

- **`BatchPlanner` Phase 0 and Phase 0b** (T2.2-2.3): `BatchPlanner.plan()` now accepts
  `carry_specs: list[CarrySpec] | None` and `dedup_specs: list[DedupSpec] | None`.
  Phase 0 pre-commits declared carry axes to `Scan` before the cardinality/budget loop;
  Phase 0b pre-commits dedup-eligible axes to `DedupGather`. Remaining axes proceed
  through the existing budget rules unchanged.
  ([`src/xtrax/tiling/plan.py`](src/xtrax/tiling/plan.py))

- **Factory `make_axis_dispatch` + iterator types** (T2.4): `make_axis_dispatch(strategy)`
  is now a pure factory — it takes a strategy and returns an iterator object, not a result.
  Three iterator types ported from `aminx.tiling`:
  - `VmapIterator` — wraps `jax.vmap`
  - `SafeMapIterator` — chunk-order-stable chunked map
  - `JaxScanIterator` — `jax.lax.scan`; returns `(final_carry, stacked_outputs)`
  - `MapIterator` — eager Python-level map
  `DispatchRejected` raised for `DedupGather` (handled upstream by `BatchPlanner`).
  Backward-compat shim `axis_dispatch(strategy, fn, xs, init=None)` preserves the prior
  eager 4-arg call.
  ([`src/xtrax/tiling/dispatch.py`](src/xtrax/tiling/dispatch.py),
   [`src/xtrax/tiling/iterator.py`](src/xtrax/tiling/iterator.py))

- **`Scan.init` field** (T2.1): `Scan` strategy now carries an optional `init: Any | None`
  for configurable carry initialization. Default `None` (zero-init, backward-compatible).
  ([`src/xtrax/tiling/strategy.py`](src/xtrax/tiling/strategy.py))

### Exports

All new types and exceptions exported from `xtrax.tiling`:
`CarrySpec`, `CarryShape`, `DedupSpec`, `get_k_bucket`,
`VmapIterator`, `SafeMapIterator`, `JaxScanIterator`, `MapIterator`,
`DispatchRejected`, `axis_dispatch`.

## [0.2.0] - 2026-06-10

### Added

- **Distribution readiness**: Apache-2.0 license, PyPI metadata, py.typed marker for type checking support
- **Lazy public API**: 43 curated top-level imports (Trainer, Engine, AxisSpec, BatchPlan, etc.) via PEP 562 lazy loading; bare `import xtrax` overhead <1ms
- **Documentation**: Sphinx-powered docs hosted on RTD with furo theme, quickstart guide, concepts, and architecture diagrams
- **CI/CD**: GitHub Actions workflow for lint (ruff), type-check (pyright), test (pytest, 414 tests at 96.5% coverage), with 90% coverage gate
- **Publish pipeline**: Trusted publishing via OIDC to TestPyPI and PyPI; automated on git tags matching v*

### Fixed

- **Version reconciliation**: Single-sourced version 0.2.0 via hatchling; removed version duplication across files
