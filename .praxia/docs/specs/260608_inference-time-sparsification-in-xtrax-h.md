---
session_id: 0b53c429
topic: Inference-time sparsification in xtrax: how should SparsePolicy/SparseMaskManager integrate with the inference path, tiling/bucket dispatch, and BCOO format to enable effective weight sparsification at inference time while avoiding XLA recompilations
task_type: constrained-technical
winner: "Composite: (A) sparsify_model(model, policy, leaf_filter=eqx.is_array) -> eqx.Module — a pure Python-side functional transform that traverses the model pytree via jax.tree_util.tree_map_with_path, applies policy.apply_mask to each leaf passing leaf_filter, and returns a new eqx.Module with BCOO leaves (fixed nse_budget). (B) Composition convention: callers write eqx.nn.inference_mode(sparsify_model(model, policy)) — sparsify_model runs first, inference_mode second. Canonically documented. (C) jit-trace guard: assert_not_tracing() in sparsify_model's body checks isinstance(leaf, jax.core.Tracer) on all leaves — raises RuntimeError with descriptive message if called inside jit. (D) BucketIterator minimal implementation: pads inputs to nearest static bucket size (128, 256, 512) using jnp.pad; raises ValueError on inputs exceeding max bucket; yields (padded_batch, original_length_mask) tuples. Merged into winner: freeze_masks (internal impl detail), make_sparse_forward_fn (helper utility), SparseSafeMap (= BucketIterator in bucket mode)."
runner_up: "SparseEngine subclass of Engine that pre-applies masks in eval() before dispatching to the tiling/bucket path"
created_at: 2026-06-08T18:34:33.127737+00:00
---

# Brainstorm Spec: Inference-Time Sparsification in xtrax

## Problem Frame

**Fixed constraints — these cannot change:**

1. `SparsePolicy.apply_mask` is not jit-safe. It must execute Python-side, before any jit boundary.
2. `SparseMaskManager` holds mutable Python-side state. It cannot be treated as a static pytree leaf or passed through jit.
3. BCOO with a fixed `nse_budget` gives static shape. The `nse_budget` must be set before compilation and must not change between calls to the same compiled function.
4. xtrax does not own sparse matmul. The caller is responsible for the actual sparse computation; xtrax only owns the mask application and model preparation steps.
5. `BucketIterator` is currently a stub — any design that depends on it being complete must account for this gap.
6. Sprint scope is one medium sprint. Designs that require multi-sprint infrastructure work are out of scope.

**Negotiable:**
- The exact API surface for mask application (functional transform vs. subclass vs. wrapper object).
- Whether `BucketIterator` gets a minimal implementation in this sprint or remains stubbed with a clear contract.
- Whether the sparse model is a new `eqx.Module` subtype or an unmodified pytree with BCOO leaves.
- The granularity of `nse_budget` (per-model global vs. per-layer).
- Serialization story for sparse checkpoints (can be deferred).

---

## Winner

**Composite design (Ideas 0 + 9 + 11 + 3):**

### Part A — `sparsify_model` functional transform
```python
def sparsify_model(
    model: eqx.Module,
    policy: SparsePolicy,
    leaf_filter: Callable[[Any], bool] = eqx.is_array,
) -> eqx.Module:
    ...
```
- Pure Python-side transform: no jit involvement.
- Traverses the model pytree via `jax.tree_util.tree_map_with_path`.
- Applies `policy.apply_mask` to each leaf passing `leaf_filter`.
- Returns a new `eqx.Module` with BCOO leaves in place of filtered dense arrays.
- Internally uses `_freeze_masks()` as an implementation detail (Idea 2, merged).

### Part B — Composition convention
Canonical usage:
```python
inference_model = eqx.nn.inference_mode(sparsify_model(model, policy))
```
`sparsify_model` runs first (replaces dense leaves with BCOO), then `eqx.nn.inference_mode` runs (disables dropout/BN). This order is canonical and must be documented. Reversed order is technically safe but inconsistent.

### Part C — jit-trace guard
`assert_not_tracing(leaves)` is called at the top of `sparsify_model`, before any traversal. Checks `isinstance(leaf, jax.core.Tracer)` across all leaves. Raises `RuntimeError` with message `"sparsify_model cannot be called inside jax.jit"` if any leaf is traced.

### Part D — `BucketIterator` minimal implementation
Implements `BucketIterator.__iter__` (currently a stub):
- Uses `bisect.bisect_right(boundaries, seq_len)` to find the target bucket.
- Pads input to the bucket size with `jnp.pad`.
- Yields `(padded_batch, original_length_mask)` tuples — `original_length_mask` is a boolean array so callers can mask attention keys/values.
- Raises `ValueError` (preceded by `warnings.warn`) on inputs exceeding the largest bucket.

### Helper — `make_sparse_forward_fn`
```python
def make_sparse_forward_fn(
    fn: Callable,
    sparse_model: eqx.Module,
) -> Callable:
    ...
```
Closes over `sparse_model`; returns a function taking `(inputs,)` only, suitable for direct `jax.jit`. Minimal implementation (5–8 lines).

---

## Runner-Up

**Idea 1 — `SparseEngine` subclass:**
Strongest argument: enforces sparsification structurally — a caller using `SparseEngine` cannot run eval without going through pre-sparsification. No call-site discipline required.

**Why runner-up lost:** The subclass must hold a reference to `SparseMaskManager` (mutable Python-side state, Constraint 2), coupling the Engine lifecycle to the manager lifecycle. This is a maintenance trap and a reversibility failure — removing `SparseEngine` later requires touching all call sites. The functional approach leaves Engine untouched.

---

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| Idea 0 — `sparsify_model` functional transform | ACCEPT | Pure Python-side, no jit. Traversal via `tree_map_with_path`. Low implementation cost. Decouples sparse model from SparseMaskManager. |
| Idea 1 — `SparseEngine` subclass | REJECT | Holds mutable manager reference (Constraint 2 violation). Creates parallel API surface. Low reversibility. |
| Idea 2 — `freeze_masks` public API | MERGE | Merge into `sparsify_model` as internal `_freeze_masks` implementation detail. No value as public API if `sparsify_model` already returns a self-contained sparse pytree. |
| Idea 3 — BucketIterator minimal impl | ACCEPT | Stub → minimal impl is 1–2 days of work, within sprint scope. Overflow policy must error loudly. Padding length must be returned as mask. |
| Idea 4 — `SparseInferenceState` carrier dataclass | DEFER | Redundant if `sparsify_model` already returns a fully self-contained sparse pytree. No concrete `policy_metadata` use case in Sprint 7. Scope creep at Engine.eval boundary. |
| Idea 5 — Per-layer heterogeneous nse_budget | DEFER | Requires dict[str,int] traversal + naming convention — at least one sprint of infrastructure. Tied-weights ambiguity unresolved. Prove global budget first. |
| Idea 6 — Auto-budget (nse_budget = ceil(nnz * 1.05)) | DEFER | Auto-budget changes when policy threshold changes, breaking static-shape retrace guarantee (Constraint 3 violation). Arbitrary 1.05 headroom. |
| Idea 7 — `make_sparse_forward_fn` | MERGE | Merge into winner as helper utility. Requires Idea 0 first. 5–8 line implementation. |
| Idea 8 — Mask caching with content hash | DEFER | DtH transfer cost for hashing large tensors. Benefit only materializes for repeated `sparsify_model` calls on unchanged checkpoints — an anti-pattern good API should eliminate. |
| Idea 9 — Compose with `eqx.nn.inference_mode` | ACCEPT | Zero implementation cost — this is a composition convention. Order (sparsify before inference_mode) must be canonically documented. |
| Idea 10 — SparseCheckpoint (BCOO serialization) | DEFER | New checkpoint schema + BCOO deserialization path = multi-sprint scope. Dense/sparse checkpoint format collision risk. |
| Idea 11 — Runtime jit-trace guard | ACCEPT | 3–5 line implementation. Guard must live at `sparsify_model` public API boundary (not inside SparseMaskManager internals) for testability. |
| Idea 12 — SparseSafeMap | MERGE | Functionally identical to Idea 3. Merge into BucketIterator implementation. |
| Idea 13 — Two-phase mask (structural + magnitude) | DEFER | Submask constraint non-trivial to enforce. Marginal latency benefit since fine-mask update also runs Python-side. |
| Idea 19 — BCOO baked at checkpoint save time | DEFER | Legitimate optimization path (eliminates Python-side `apply_mask` at inference), but moves problem to checkpoint writer. Future sprint. |

---

## Assumptions

| # | Assumption | Risk if false |
|---|-----------|---------------|
| A1 | BCOO leaves in an eqx.Module pytree are valid JAX pytree leaves with static shape — JAX does not retrace when the same BCOO structure (fixed `nse_budget`) is passed to a compiled function. | Retrace on every forward call; entire design breaks. Validate with a two-call trace-count test before impl. |
| A2 | `jax.tree_util.tree_map_with_path` visits all array leaves in an eqx.Module pytree stably and bijectively. | Non-deterministic leaf assignment breaks mask-to-weight alignment. |
| A3 | xtrax models at inference have no tied weights (two pytree paths pointing to the same underlying array). | Double-sparsification of tied weights produces two separate BCOO leaves, breaking weight sharing. Documented as out-of-scope for Sprint 7; caller must use path-based filter. |
| A4 | Bucket sizes 128/256/512 cover typical inference sequence length distribution for xtrax users. | Inputs >512 hit the hard-error path; users with longer sequences must configure custom boundaries. |
| A5 | `eqx.nn.inference_mode` does not mutate or re-wrap BCOO leaves — it only sets boolean flags on stochastic layers. | `eqx.nn.inference_mode(sparsify_model(...))` would corrupt BCOO structure if inference_mode replaces array leaves. |

---

## TBDs

| # | TBD | Owner | Target sprint |
|---|-----|-------|---------------|
| T1 | Tied-weights edge case: how should `sparsify_model` handle pytree paths that alias the same underlying array? Policy: error, skip, or apply once? | Spec author | Sprint 8 |
| T2 | Per-layer nse_budget: API shape for heterogeneous budgets (dict[str, int] keyed by leaf path, or Callable[..., int]). Deferred — global budget must be proven first. | Spec author | Sprint 8+ |
| T3 | BCOO checkpoint serialization: schema for storing BCOO alongside dense orbax checkpoint. Multi-sprint scope. | Spec author | Sprint 9+ |
| T4 | Interaction between BucketIterator and attention masking: does `original_length_mask` need to be a JAX array or a Python int? Contract for callers who pad attention keys/values. | Fixer | Sprint 7 (resolve during implementation) |
| T5 | nse_budget waste ratio: recommended upper bound on `nse_budget / actual_nnz` beyond which BCOO throughput degrades. Empirically measure with benchmark. | Spec author | Sprint 9 |

---

## Pre-mortem Record

**Six-month failure scenarios identified during convergence:**

1. **Double-sparsification**: Callers pass `eqx.is_array` without realizing BCOO leaves from a prior call are also arrays — re-running `sparsify_model` on an already-sparse model silently produces doubly-sparse garbage.
   - Mitigation: `sparsify_model` asserts at entry that no leaf is already a `jax.experimental.sparse.BCOO`. Raises `ValueError("model already contains BCOO leaves; call sparsify_model on the dense checkpoint")`.

2. **Silenced BucketIterator overflow**: `ValueError` on sequences >512 is caught by caller's blanket `except Exception`, causing silent fallback to unbucketed dispatch with latency spikes and no error signal.
   - Mitigation: `BucketIterator` emits `warnings.warn(...)` with `stacklevel=2` **before** raising `ValueError`, ensuring the warning is visible even if the exception is caught.

3. **Reversed composition order**: Contributor calls `sparsify_model(eqx.nn.inference_mode(model), policy)`. Technically valid but inconsistent with the canonical idiom.
   - Mitigation: Document canonical order (`sparsify_model` before `inference_mode`) in module docstring with an explicit example. Emit `warnings.warn` inside `sparsify_model` if inference_mode has already been applied.

---

## Acceptance Criteria

### AC-1: Basic sparsification
**Given** a dense `eqx.Module` model and a `SparsePolicy(config=SparseConfig(nse_budget=N, update_schedule=...))`,
**When** `sparsify_model(model, policy)` is called outside any jit boundary,
**Then** returns a new `eqx.Module` where every leaf passing `eqx.is_array` has been replaced by a `jax.experimental.sparse.BCOO` with `nse == nse_budget`, and `.todense()` on each BCOO leaf equals `original_leaf * mask` within float32 precision.

### AC-2: jit-trace guard fires
**Given** a dense model and policy,
**When** `sparsify_model(model, policy)` is called inside a `jax.jit`-compiled function (any model leaf is a `jax.core.Tracer`),
**Then** raises `RuntimeError` with a message containing "sparsify_model" and "jit", before any mask application occurs.

### AC-3: Double-sparsification guard fires
**Given** a model whose pytree already contains at least one `BCOO` leaf,
**When** `sparsify_model` is called on that model again,
**Then** raises `ValueError` with a message containing "already contains BCOO leaves", before any traversal.

### AC-4: BucketIterator pads to nearest bucket
**Given** a `BucketIterator(boundaries=[128, 256, 512], batch_sizes=[16, 8, 4, 2], fn=identity, xs=batch_of_length_300)`,
**When** `__iter__` is called,
**Then** yields exactly one tuple `(padded_batch, original_length_mask)` where `padded_batch` has leading-axis length 512 and `original_length_mask` is a boolean array of shape `(512,)` with exactly 300 `True` values.

### AC-5: BucketIterator overflow raises with warning
**Given** a `BucketIterator(boundaries=[128, 256, 512], ...)` and input with sequence length 1024 (> max bucket 512),
**When** `__iter__` is called,
**Then** emits `warnings.warn(...)` and raises `ValueError` containing "exceeds maximum bucket size", before yielding any item.

### AC-6: No retrace across calls with same bucket shape
**Given** a `sparsify_model`-transformed model wrapped in `jax.jit`,
**When** the function is called twice with inputs of the same bucket-padded shape,
**Then** `jax.jit` does not retrace on the second call (verify via `jax.make_jaxpr` or XLA compilation count).

### AC-7: make_sparse_forward_fn helper
**Given** a sparse model returned by `sparsify_model` and a forward function `fn(model, inputs)`,
**When** `jit_fn = make_sparse_forward_fn(fn, sparse_model)` is called and then `jit_fn(inputs)` is called,
**Then** returns the same result as `fn(sparse_model, inputs)`, with model closed over (callable with `(inputs,)` only).

### AC-8: leaf_filter excludes non-target leaves
**Given** a model with both 2-D weight matrices and 1-D bias vectors,
**When** `sparsify_model(model, policy, leaf_filter=lambda x: x.ndim >= 2)` is called,
**Then** 2-D leaves are replaced with BCOO and 1-D bias leaves remain as dense `jax.Array`.

### AC-9: Composition with inference_mode
**Given** the canonical idiom `eqx.nn.inference_mode(sparsify_model(model, policy))`,
**When** this composed model is used in `Engine.eval()`,
**Then** the model has both dropout disabled and sparse BCOO weights, with no XLA retrace across eval calls of the same bucketed input shape.

---

## Phase Gates

**Sprint 7 gate**: `uv run pytest tests/sparse/ tests/tiling/ -v` all pass. `uv run pytest tests/ -v` full suite passes (no regressions). ruff clean on all new code. Coverage ≥ 95% on `src/xtrax/sparse/inference.py` and modified `src/xtrax/tiling/iterator.py`.

---

## New Module Layout

```
src/xtrax/sparse/
  __init__.py      # add exports: sparsify_model, make_sparse_forward_fn
  inference.py     # NEW: sparsify_model, make_sparse_forward_fn, assert_not_tracing

src/xtrax/tiling/
  iterator.py      # MODIFY: implement BucketIterator.__iter__ (stub → minimal impl)

tests/sparse/
  test_inference.py  # NEW: AC-1 through AC-8

tests/tiling/
  test_iterator.py   # MODIFY: add AC-4, AC-5 for BucketIterator
```

---

## Sprint 7 Decomposition

**Track A — `src/xtrax/sparse/inference.py`** (primary, ~3 days):
1. `assert_not_tracing(leaves)` — check all leaves for `jax.core.Tracer`, raise `RuntimeError` (feeds AC-2)
2. `sparsify_model(model, policy, leaf_filter=eqx.is_array)` — BCOO guard (AC-3), traversal via `tree_map_with_path`, `policy.apply_mask` per leaf (AC-1, AC-8)
3. `make_sparse_forward_fn(fn, sparse_model)` — closure helper (AC-7)
4. `__init__.py` exports update
5. Tests in `tests/sparse/test_inference.py` covering AC-1 through AC-8

**Track B — `src/xtrax/tiling/iterator.py`** (~2 days):
1. Implement `BucketIterator.__iter__`: find bucket via `bisect.bisect_right`, pad with `jnp.pad`, build `original_length_mask`, yield tuple (AC-4)
2. Overflow: `warnings.warn` then `ValueError` (AC-5)
3. Tests in `tests/tiling/test_iterator.py` covering AC-4, AC-5

**Integration** (after both tracks, ~1 day):
- AC-6: no-retrace test with `jax.make_jaxpr` compilation-count assertion
- AC-9: `Engine.eval` with `sparsify_model` + `inference_mode` composition end-to-end test

---

*Next step: promote from staging to backlog, then route through `spec_driven_dev` workflow starting at `spec_challenge`.*
