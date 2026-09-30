# Capture Mechanics Reference

Owner modules: `src/xtrax/stages/boundaries.py` (Fuse/Tap/Sink/AxisBoundary),
`src/xtrax/stages/executor.py` (when each fires, and what happens to its
return value), `src/xtrax/stages/topology.py` (plan-time rejection),
`src/xtrax/stages/_callback.py` (the pinned `io_callback`),
`src/xtrax/run/sink.py` (`SinkSpec` / `make_sink` / `derive_sink_spec`),
`src/xtrax/run/zarr_sink.py` (`ZarrStagingSink`),
`src/xtrax/run/zarr_integrity.py` (digests + fsync). When this document and
the code disagree, the code wins.

Verified against xtrax **0.4.0a7**, jax **0.10.2**, zarr **3.2.1**.

## Why a boundary op rather than a modified forward pass

`Tap[T]` is declared `T -> T` -- identity plus a side effect -- and `Sink[T]` is
declared `T -> None` (`boundaries.py`). That is the whole reason to capture
here: the observation rides alongside the computation instead of changing its
signature. Editing a forward pass to return extra tensors changes the thing
you are measuring, and it cannot be left in place afterwards.

**But `Tap`'s identity property is contractual, not enforced.** The executor
uses a tap's return value as the step output:

```python
# src/xtrax/stages/executor.py, _wrap_step
y = fn(x)
if boundary.tap is not None:
    y = boundary.tap(y)          # <-- tap's return value REPLACES the step output
if boundary.sink is not None:
    boundary.sink(y)             # <-- sink's return value is discarded
return y
```

Pinned both ways in `tests/stages/test_executor.py::TestTapContinuesSinkDiscards`:
`tap=lambda x: x * 10` changes the stacked output, while a deliberately
misbehaving sink returning `x * 999` does not.

**Consequence: use `sink=` for capture.** A capture sink that accidentally
returns the `io_callback` result, or returns `None` where a tap was expected,
is structurally unable to corrupt the computation. The same mistake in a tap
silently perturbs every downstream value -- the exact failure this whole
approach exists to avoid.

`Fuse` is not a capture point: it must be a pure JAX function, no side
effects, no `io_callback` (`boundaries.py`, `Fuse` docstring).

## `AxisBoundary` must flatten to zero dynamic leaves

All three fields are `eqx.field(static=True)` (`boundaries.py`). Assert it on
any boundary you build, because a capture object holding a JAX array as
instance state would silently become a traced leaf:

```python
import jax.tree_util
assert jax.tree_util.tree_leaves(boundary) == [], "AxisBoundary must have no dynamic leaves"
```

Keep capture state host-side (a plain Python counter, a list, the sink handle)
-- never a `jax.Array` attribute.

## The `io_callback` shim

Import from `xtrax.stages._callback`, never `jax.experimental` directly. The
shim runs two checks at **module import** time, so signature or version drift
is a loud `IoCallbackSignatureError` (an `ImportError` subclass) rather than an
opaque traceback inside a dispatch loop: `_check_jax_version` against
`PINNED_JAX_RANGE` (currently `((0, 10, 2), (0, 11, 0))`) and `_check_signature`
against the expected parameter tuple plus `ordered`'s `False` default.

Notes for callers:

- The module is **private** and deliberately not re-exported from
  `xtrax.stages.__all__`. `from xtrax.stages._callback import io_callback` is
  the sanctioned form -- it is what `tests/stages/test_executor.py` and
  `tests/stages/test_nested_ordering.py` use.
- At 0.4.0a7 the shim is present and `io_callback is jax.experimental.io_callback`
  (pinned by `tests/stages/test_callback.py::test_shim_reexports_the_real_io_callback`).
  No fallback path is needed at this version. Older 0.4.0aN wheels predate it;
  if `ImportError` names the module rather than jax, fall back to
  `from jax.experimental import io_callback` and record that you did.
- `io_callback` is the effectful callback. Use it, not `pure_callback`, for
  capture: a capture whose result is unused must still run.

## Ordering: what halts, and what silently degrades

| combination | outcome | owner |
|---|---|---|
| `ordered=True` tap/sink on a **`Vmap`** axis | **HALTS** `PlanTopologyError` at plan construction | `topology.py`, `validate_plan_topology` rule 2 |
| same, reaching the executor directly | **HALTS** `ExecutorError` (defense in depth) | `executor.py`, `execute_map_axis` |
| an ordered capture nested inside a `Vmap` axis's `fn` | **HALTS** `ExecutorError`, re-raised from JAX's own `ValueError` | `executor.py` |
| `ordered=True` tap/sink on a **`ChunkedMap`** axis | **runs, silently ignoring `strategy.batch_size`** -- one element at a time, unconditionally, at any batch size | `executor.py`, `execute_map_axis` ChunkedMap branch |
| `Scan` strategy on a **heterogeneous** axis | **HALTS** `PlanTopologyError` | `topology.py` rule 1 |

The underlying JAX error string the executor matches on is
`Cannot \`vmap\` ordered IO callback` -- **with backticks around `vmap`**
(`executor.py`). Prose elsewhere in the codebase writes it without them; match
on the backticked form if you ever need to catch it yourself.

Cost, not just correctness (`executor.py` module docstring, verified against
JEP-10657): `ordered=True` threads an XLA token as a real data dependency
between consecutive calls, so XLA cannot reorder, overlap or pipeline them.
Under a `Scan` of N steps an ordered capture is N strictly serialized host
round trips. **Do not set `ordered=True` by default** -- set it only when the
capture's meaning depends on host-observed order (per-step indices assigned by
a host-side counter, as in the Quick Start, is such a case; capturing whole
pre-keyed tensors is not).

Validate the plan before tracing: `validate_plan_topology(plan.decisions,
axis_boundaries_by_name(run_spec.axes, run_spec.boundaries))`. `boundaries` is
positional, one entry per axis; a length mismatch or a duplicate axis name is
itself a `PlanTopologyError` rather than a silently dropped boundary.

## `SinkSpec` and `ZarrStagingSink`

`ZarrStagingSink` (`zarr_sink.py`) stages keyed numpy payloads and drains them
into a chunked Zarr store. Its documented purpose explicitly includes encoder
intermediates, which is exactly the activation-capture case. Key components are
stringified and joined with `/` into the Zarr group path, so key by
`(side, tensor_name, step)` and the store's layout reads as the trace itself.
Repeated `stage()` on one key **merges** (later array names overwrite
same-named earlier ones, new names accumulate). It auto-drains once
`spec.flush_every` stage calls have accumulated.

Traps, in the order they bite:

1. **`SinkSpec.format` defaults to `"jsonl"`, which has no writer.**
   `make_sink` raises `NotImplementedError` for `"jsonl"` and `"h5"` -- both are
   routing-only stub values (`sink.py`). Set `format="zarr"` explicitly, or go
   through `derive_sink_spec`, which pins `"zarr"`. That divergence between the
   two defaults is deliberate (`sink.py` docstring).
2. **`run_id` is required and must be non-blank**, rejected at sink
   construction. `derive_sink_spec` is the canonical seam and resolves it by
   precedence: explicit `run_id=` -> `run_spec.run_id` -> `new_run_id()`.
3. **One `output_dir` cannot host two run ids.** Opening a directory that
   already carries a different `run_id` raises, because the earlier run's
   per-key provenance pointers would be orphaned. **Give each side of a parity
   comparison its own `output_dir`.**
4. **Reserved attrs names.** `stage(attrs=...)` raises on any of `git_sha`,
   `git_branch`, `git_dirty`, `run_id`, `created_at` (`_CORE_PROVENANCE_FIELDS`)
   -- the sink owns those. When `spec.extension_schema` is set, attr value
   types are checked at `stage()` time; its `required` fields are enforced at
   `drain()` against the key's merged view (on-disk + pending attrs), so a
   capture may add required metadata over several `stage()` calls. With
   `flush_every=1` every `stage()` drains, so each call must be complete.
5. **0-d and zero-length payloads are stored as-is.** `drain` chunks each
   array as one rank-matched chunk (`chunks=tuple(max(d, 1) for d in shape)`),
   so per-step scalars keep `shape == ()` and `(0,)`/`(0, 3)` round-trip
   (`tests/run/test_zarr_sink.py::test_drain_round_trips_degenerate_shapes`).
   Before xtrax #161 these crashed `drain()`; do not reintroduce an
   `np.atleast_1d` wrap -- it changes the stored shape both sides compare on.
6. **Errors raised inside a capture callback surface as
   `jax.errors.JaxRuntimeError: INTERNAL: CpuCallback error calling callback`**,
   with the real exception's frames embedded in that message's text. When a
   capture harness dies opaquely, read past the JAX frames.
7. **`finalize()` refuses while payloads are pending** and may run only once;
   `stage()`/`drain()` after it raise. Call `drain()` then `finalize()` -- it
   consolidates store metadata and will not strand buffered payloads silently.
8. **`zarr` is an optional extra** (`xtrax[io]`), imported lazily inside
   `ZarrStagingSink.__init__` and inside the zarr-touching digest functions.
   Importing `xtrax.run` never requires it; constructing a sink does, and the
   `ImportError` names the install command.

## Integrity: what the digest proves, and what it does not

`zarr_content_digest(path)` folds every node's **path**, **attrs**, and (for
arrays) dtype/shape/little-endian-canonicalized bytes into one sha256
(`zarr_integrity.py`). It is unaffected by filesystem metadata -- mtimes, chunk
file layout -- or by which process wrote the store.

**Sink provenance is excluded by default** (`include_provenance=False`, xtrax
#5013, `update_zarr_node_digest`): the root group skips all five core fields
(`git_sha`, `git_branch`, `git_dirty`, `run_id`, `created_at`) and each non-root
group skips its `run_id`/`git_sha` pointer. So two stores with identical data,
identical key paths and identical caller attrs digest **equal** even under
different `run_id`s and wall-clock `created_at`s
(`tests/run/test_zarr_integrity.py::test_digest_equal_across_different_run_ids_by_default`).
Before 0.4.0a10 this was false -- `created_at` alone made every store unique.

**It is still not a cross-side equality oracle for a parity trace.** It folds
node **paths** and every non-provenance attr, so keying by
`(side, tensor_name, step)` -- the layout recommended above -- makes the two
sides' digests differ by construction, and a caller attr that differs between
sides (a tolerance, a source tag) does the same. And a whole-store verdict
cannot localize: "the digests differ" says nothing about *which* tensor.

So:

- **Use `zarr_content_digest` for self-verification**: record it in a done
  marker and re-compute later to prove a capture has not drifted or been
  partially rewritten. Re-reading an unchanged store reproduces it exactly.
  Pass `include_provenance=True` when you want the run identity sealed in too.
- **Use `update_array_digest` for cross-side content comparison.** It folds
  `dtype | shape | canonicalized bytes` and **nothing else** -- no path, no
  attrs. Fold a fixed name ordering into your own `hashlib.sha256()` and you
  have a content-only fingerprint that matches across independently written
  stores, per tensor -- which is what a first-divergence table needs.
- `update_zarr_node_digest(digest, node, path)` is public if you want a
  subtree rather than the whole store; it applies the same default exclusions
  (the root is matched on the stripped path, so `root.path == ""` works).

**`fsync_tree(path)` before trusting any digest.** A Zarr store is a directory
of many chunk and metadata files, unlike a single-file HDF5; content
verification is only meaningful once every file's bytes are actually on disk.
`fsync_tree` durabilizes files first, then directories bottom-up, ending with
the root. `fsync_file` and `fsync_directory` are exported for narrower use.
