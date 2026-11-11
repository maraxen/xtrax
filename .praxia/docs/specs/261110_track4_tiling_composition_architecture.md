---
task_id: 261110_xtrax-debt-burn-down-and-consumer-adoption
track: 4
track_name: Tiling, Composition & Architecture
debts: [3194, 3193, 3192, 3116, 3044, 3043, 3042, 2543]
created: 2026-11-10
status: draft
---

# Track 4 Engineering Specification: Tiling, Composition & Architecture

## Executive Summary & Scope

Track 4 of the `xtrax` technical debt burndown addresses the architectural foundation of `xtrax`: host-side and device-side tiling runtime, loop iteration strategies, kernel autodiff composition, packaging boundaries, declarative planning decisions (sparsity, multi-device sharding, numerical safety), and loop carry contract execution.

This track resolves eight interconnected technical debts:
1. **#3194** (Architecture): Host `BucketDispatcher`, `vmap_any_cond` for periodic amortised work, and `BatchedCapacityState` in `xtrax.tiling`.
2. **#3193** (Architecture): Dynamic scalar `n_steps` (tracer-compatible), cached JIT runner, and carry-leaf canonicalization in `WhileCarry`.
3. **#3192** (Performance): `symbolic_zeros=True` active-tangent VJP, primal+grad fusion, and 2D Pallas grid batching with stride-0 shared topology in `xtrax.op`.
4. **#3116** (Packaging): Decouple training-only dependencies (`optax`, `orbax-checkpoint`) into an optional extra `[project.optional-dependencies].training` in `pyproject.toml` while preserving backward-compatible `[all]` and `[dev]` extras.
5. **#3044** (Feature): Re-allow opt-in sparsity specification as an explicit, result-changing option in `xtrax` planning and execution contexts.
6. **#3043** (Feature): Re-build `ShardingPolicy` as an active multi-device placement planner.
7. **#3042** (Feature): Re-build `NaN`/`Inf` safety checking as a declarative, `Context`-driven compiler decision replacing deprecated `SafetyManager`/`SafetyTrainStep`.
8. **#2543** (Architecture): End-to-end execution contract verification for `CarrySpec.transition` through `BatchPlanner` and `axis_dispatch`.

---

## Architectural Context & Invariants

### System Component Seams

```
+-----------------------------------------------------------------------------------+
|                               Caller / Consumer                                  |
|                 (aminx, demistify, prolix, downstream models)                     |
+-----------------------------------------------------------------------------------+
       |                                       |                               |
       v                                       v                               v
+------------------------+      +-------------------------------+    +--------------------+
|  xtrax.intent.Context  |      |   xtrax.tiling.BatchPlanner   |    |      xtrax.op      |
|  - memory_budget       | ---> |   - AxisSpec (with sparsity)  |    |  - reference       |
|  - sparsity (Opt-in)   |      |   - CarrySpec (scan / while)  |    |  - kernels (Pallas)|
|  - mesh & sharding     |      |   - Tiling strategy decision  |    |  - custom_vjp with |
|  - safety (checkified) |      +-------------------------------+    |    symbolic_zeros  |
+------------------------+                     |                     +--------------------+
       |                                       v                               |
       |                         +---------------------------+                 |
       |                         |    Tiling Dispatch Tier   |                 |
       |                         |  - Vmap / ChunkedMap      | <---------------+
       |                         |  - Scan / WhileCarry      |
       |                         |  - Host BucketDispatcher  |
       +-----------------------> +---------------------------+
                                               |
                                               v
                                 +---------------------------+
                                 |   XLA Device Execution    |
                                 |  (GPU / TPU / CPU Pallas) |
                                 +---------------------------+
```

### Core Invariants

1. **Host-Boundary Invariant**: Dynamic shape manipulation (e.g. bucketing, padding, ragged item grouping) MUST occur on the host before entering the `@jax.jit` boundary. Inside `@jax.jit`, array shapes and PyTree structures MUST be static.
2. **Reverse-Mode AD Safety Invariant**: Carry-bearing loops requiring reverse-mode automatic differentiation MUST lower to `jax.lax.scan`. Carry-bearing loops with data-dependent termination (`lax.while_loop` / `WhileCarry`) MUST be restricted to inference, analysis, or forward-only execution, and MUST raise an explicit error if subjected to reverse-mode gradient transformations.
3. **Purity & Integrity Invariant**: No mock implementations, dummy returns, or hardcoded test results. Every dispatcher, runner, and planner MUST maintain genuine state, enforce shape contracts, and execute actual computations.

---

## Detailed Architectural Designs

---

### 1. Debt #3194: Host `BucketDispatcher`, `vmap_any_cond`, and `BatchedCapacityState`

#### 1.1 Motivation & Root Cause
In `src/xtrax/tiling/dispatch.py:205-213`, `axis_dispatch` unconditionally raises `TypeError` when encountering `Bucket`, noting that `Bucket` is a host-side strategy. While `src/xtrax/tiling/bucket.py` provides low-level utility functions (`select_bucket`, `select_rung`, `bucketize`, `pad_axis`, `trim_axis`), there is no host-level batch coordinator. Callers with variable-length sequences are forced to write ad-hoc padding loops, risking shape explosion and backend recompilations.

Furthermore, molecular dynamics and graph models (e.g., prolix, denxity) frequently execute amortized periodic maintenance (such as neighbor-list rebuilds or capacity expansions). Currently, executing this on a batch forces either full-batch recomputation on every step or divergent per-item control flow.

#### 1.2 `BucketDispatcher` Architectural Specification
`BucketDispatcher` is a host-side coordinator that groups inputs by length bucket, executes cached JIT-compiled executables per bucket rung, unpads results, and restores original ordering.

```python
# src/xtrax/tiling/bucket.py (new additions)

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

class BucketDispatcher:
    """Host-side batch coordinator for variable-length inputs across fixed bucket rungs.
    
    Attributes:
        boundaries: Strictly ascending sequence of bucket capacity thresholds.
        fn: Core device computation to apply to each bucketed batch.
        axis: The variable-length axis to bucket (default: 0).
        pad_value: Value used to pad unused slots (default: 0).
    """
    boundaries: tuple[int, ...]
    fn: Callable[[Any], Any]
    axis: int
    pad_value: Any
    _cached_runners: dict[int, Callable[[Any], Any]]

    def __init__(
        self,
        fn: Callable[[Any], Any],
        boundaries: Sequence[int] = BUCKET_LADDER,
        axis: int = 0,
        pad_value: Any = 0,
    ) -> None:
        if not boundaries:
            raise ValueError("BucketDispatcher: boundaries must not be empty.")
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError(f"BucketDispatcher: boundaries must be strictly ascending, got {boundaries}")
        self.boundaries = tuple(boundaries)
        self.fn = fn
        self.axis = axis
        self.pad_value = pad_value
        self._cached_runners = {}

    def _get_runner(self, bucket_size: int) -> Callable[[Any], Any]:
        """Retrieve or compile the JIT runner for a specific bucket size."""
        if bucket_size not in self._cached_runners:
            self._cached_runners[bucket_size] = jax.jit(jax.vmap(self.fn))
        return self._cached_runners[bucket_size]

    def dispatch_batch(self, items: Sequence[Any]) -> list[Any]:
        """Dispatch a heterogeneous sequence of PyTree items through bucketed device execution.
        
        Args:
            items: Sequence of PyTree inputs, each sharing leaf structure but differing in length.
            
        Returns:
            List of unpadded PyTree results in exact 1:1 correspondence with `items`.
            
        Raises:
            ValueError: If any item exceeds the largest bucket boundary.
        """
        if not items:
            return []

        # 1. Determine lengths and assign bucket indices
        item_lengths = []
        bucket_assignments: dict[int, list[int]] = {b: [] for b in self.boundaries}

        for idx, item in enumerate(items):
            leaves = jax.tree_util.tree_leaves(item)
            if not leaves:
                raise ValueError(f"BucketDispatcher: item {idx} contains no array leaves.")
            length = int(leaves[0].shape[self.axis])
            item_lengths.append(length)
            
            # Select bucket (raises ValueError if length > max(boundaries))
            bucket = select_bucket(length, self.boundaries)
            bucket_assignments[bucket].append(idx)

        results: list[Any | None] = [None] * len(items)

        # 2. Process each non-empty bucket
        for bucket, indices in bucket_assignments.items():
            if not indices:
                continue

            runner = self._get_runner(bucket)

            # Pad each item in this bucket and stack along a new batch dimension
            bucket_items = [items[i] for i in indices]
            padded_batch = []

            for item in bucket_items:
                # Pad leaves along self.axis up to bucket size using pad_axis (Debt #3194 Finding C4)
                padded = jax.tree_util.tree_map(
                    lambda leaf: pad_axis(np.asarray(leaf), bucket, axis=self.axis, fill=self.pad_value),
                    item,
                )
                padded_batch.append(padded)

            # Stack along batch axis (axis 0 of batch)
            # Use tree_map to stack leaves across the list of items
            stacked_batch = jax.tree_util.tree_map(
                lambda *leaves: np.stack(leaves, axis=0),
                *padded_batch
            )

            # 3. Execute device JIT runner (batched) via cached runner (Debt #3194 Finding C3)
            device_out = runner(stacked_batch)

            # 4. Unstack, trim padding, and reassign to original index
            for local_idx, global_idx in enumerate(indices):
                orig_len = item_lengths[global_idx]
                # Slice item out of batch
                single_out = jax.tree_util.tree_map(
                    lambda leaf, l_idx=local_idx: leaf[l_idx],
                    device_out
                )
                # Trim back to original length only on leaves whose rank exceeds self.axis
                # and whose dimension matches the bucket size (preserving unpadded/reduced leaves, Debt #3194 Finding C4)
                def _trim_leaf(leaf: Any, l: int = orig_len) -> Any:
                    if hasattr(leaf, "ndim") and leaf.ndim > self.axis and leaf.shape[self.axis] == bucket:
                        return trim_axis(leaf, l, axis=self.axis)
                    return leaf

                trimmed_out = jax.tree_util.tree_map(_trim_leaf, single_out)
                results[global_idx] = trimmed_out

        return [r for r in results if r is not None]
```

#### 1.3 `vmap_any_cond` Architectural Specification
For amortized operations across batched items:

```python
# src/xtrax/tiling/dispatch.py

def vmap_any_cond(
    pred_fn: Callable[[Any], jax.Array],
    work_fn: Callable[[Any], Any],
    default_fn: Callable[[Any], Any],
    xs: Any,
) -> Any:
    """Execute expensive work_fn across xs if ANY item in xs satisfies pred_fn.
    
    If jnp.any(vmap(pred_fn)(xs)) is True, executes vmap(work_fn)(xs).
    Otherwise, executes vmap(default_fn)(xs).
    
    This amortizes expensive synchronization/recomputation (e.g. neighbor-list
    updates, allocation checks) without branch divergence across batch lanes.
    
    Args:
        pred_fn: Callable(x) -> scalar bool JAX array.
        work_fn: Callable(x) -> y, executed when predicate fires for any item.
        default_fn: Callable(x) -> y, executed when predicate is false for all items.
        xs: Batched PyTree input.
        
    Returns:
        Batched PyTree result matching the structure of work_fn/default_fn outputs.
    """
    # 1. Evaluate predicates in parallel across batch
    preds = jax.vmap(pred_fn)(xs)  # Shape: (B,)
    need_work = jnp.any(preds)     # Scalar bool

    # 2. Dynamic branch via jax.lax.cond
    return jax.lax.cond(
        need_work,
        lambda operands: jax.vmap(work_fn)(operands),
        lambda operands: jax.vmap(default_fn)(operands),
        xs,
    )
```

#### 1.4 `BatchedCapacityState` Architectural Specification
Tracks dynamic capacity allocation, active occupancy, and headroom across batched operations.

```python
# src/xtrax/tiling/bucket.py

@dataclass(frozen=True)
class BatchedCapacityState:
    """Immutable state tracking capacity and occupancy for batched memory structures.
    
    Attributes:
        allocated_capacity: Static integer capacity currently allocated per batch lane.
        active_counts: JAX array of shape (B,) with integer active element counts.
        headroom_threshold: Minimum headroom before reallocation is required.
    """
    allocated_capacity: int
    active_counts: jax.Array
    headroom_threshold: int = 0

    @property
    def headroom(self) -> jax.Array:
        """Remaining capacity per batch item: (allocated_capacity - active_counts)."""
        return self.allocated_capacity - self.active_counts

    @property
    def needs_reallocation(self) -> jax.Array:
        """Scalar boolean indicating if any batch item has headroom <= threshold."""
        return jnp.any(self.headroom <= self.headroom_threshold)

    def suggest_next_rung(self, ladder: Sequence[int] = BUCKET_LADDER) -> int:
        """Compute the next bucket rung from a ladder that accommodates max occupancy.
        
        Host-Side Allocation Coordinator (Debt #3194 Finding C5):
            This method concretizes active occupancy via `int(jnp.max(self.active_counts))`
            to select the next discrete capacity rung from the ladder. It MUST be called
            outside the `@jax.jit` boundary (or after transferring active counts to the host
            via `np.asarray`), whereas `headroom` and `needs_reallocation` are pure JAX array
            properties fully compatible with device tracing and `vmap_any_cond`.
        """
        max_active = int(jnp.max(self.active_counts))
        return select_bucket(max_active + 1, ladder)
```

**Host/Device Separation Note (Debt #3194 Finding C5)**:
`BatchedCapacityState` enforces a strict separation between host-side coordination and device-side evaluation. Properties `headroom` and `needs_reallocation` evaluate pure JAX array operations, enabling them to execute inside `@jax.jit` and drive device-side conditional triggers like `vmap_any_cond`. Conversely, `suggest_next_rung` is an explicit host-side allocation coordinator: it casts the device max active count to a Python integer via `int(jnp.max(self.active_counts))` to select a discrete capacity rung. Attempting to call `suggest_next_rung` inside `@jax.jit` raises a `jax.errors.ConcretizationTypeError`. Callers must trigger capacity reallocation host-side when `needs_reallocation` evaluates to True.

---

### 2. Debt #3193: Dynamic Scalar `n_steps`, Cached JIT Runner, and Carry-Leaf Canonicalization in `WhileCarry`

#### 2.1 Motivation & Root Cause
In `src/xtrax/tiling/strategy.py:154-165`:
```python
def fixed_step_count_cond(n_steps: int) -> WhileCondFn:
    def _cond(carry: Any) -> Any:
        step_i = carry[0] if isinstance(carry, tuple) else carry.step_i
        return step_i < n_steps
    return _cond
```
1. `n_steps: int` accepts only a Python `int`. When passed a dynamic scalar (`jax.Array` or tracer from another JAX stage), static type checkers and tracing pipelines fail.
2. Inside `axis_dispatch(WhileCarry, ...)` and `WhileLoopIterator`, `jax.lax.while_loop(cond, body, init)` is invoked repeatedly without compilation caching across repeated invocations with the same carry structure.
3. `jax.lax.while_loop` requires that the carry structure, shapes, and dtypes remain strictly invariant across iterations. Discrepancies (e.g. promoting `int32` step counters to `int64` or `float32` state to `float64`) trigger runtime errors with opaque tracebacks.

#### 2.2 Specification for Dynamic Scalar `n_steps`
`fixed_step_count_cond` is generalized to accept `int | jax.Array | Any`:

```python
# src/xtrax/tiling/strategy.py

def fixed_step_count_cond(n_steps: int | jax.Array | Any) -> WhileCondFn:
    """Continuation predicate for running exactly n_steps iterations.
    
    Accepts either a static Python integer or a dynamic JAX scalar array/tracer.
    Assumes carry is a tuple where element 0 is step_i, or carry exposes a .step_i field.
    """
    # Enforce 0-D scalar rank via jnp.squeeze to prevent 1-D bool arrays (Debt #3193 Finding M3)
    scalar_steps = jnp.squeeze(n_steps) if hasattr(n_steps, "shape") else n_steps

    def _cond(carry: Any) -> Any:
        step_i = carry[0] if isinstance(carry, tuple) else carry.step_i
        # jnp.less is tracer-safe and promotes correctly across static int and dynamic array;
        # scalar rank is strictly guaranteed via scalar_steps to prevent bool[1] lax.while_loop errors
        return jnp.less(step_i, scalar_steps)

    return _cond
```

**Scalar Rank Enforcement (Debt #3193 Finding M3)**:
`jax.lax.while_loop` requires that `cond_fun` return a boolean scalar (`bool` or `jax.Array` with shape `()`). If a 1-D array (such as shape `(1,)`) is passed as `n_steps`, `jnp.less` produces a 1-D boolean array (`bool[1]`), triggering `TypeError: cond_fun must return a boolean scalar`. Applying `jnp.squeeze(n_steps)` guarantees scalar rank regardless of whether `n_steps` is provided as a Python scalar, a 0-D array, or a squeezed 1-D tensor.
```

#### 2.3 Carry-Leaf Canonicalization Specification
To protect against shape and dtype drift across loop iterations:

```python
# src/xtrax/tiling/carry.py

class CarryCanonicalizationError(TypeError):
    """Raised when carry leaves deviate in structure, shape, or dtype."""
    pass

def canonicalize_carry(carry: Any) -> Any:
    """Canonicalize all array-like leaves of a carry PyTree to immutable JAX arrays.
    
    Ensures leaves are jnp.asarray instances with preserved dtypes and static shapes.
    """
    def _canonicalize_leaf(leaf: Any) -> Any:
        if isinstance(leaf, (int, float, bool, np.ndarray, np.number)):
            return jnp.asarray(leaf)
        return leaf

    return jax.tree_util.tree_map(_canonicalize_leaf, carry)

def validate_carry_transition(init_carry: Any, next_carry: Any) -> None:
    """Validate that next_carry matches init_carry in PyTree structure, shape, and dtype.
    
    Raises:
        CarryCanonicalizationError: On structure or leaf shape/dtype divergence.
    """
    init_treedef = jax.tree_util.tree_structure(init_carry)
    next_treedef = jax.tree_util.tree_structure(next_carry)
    
    if init_treedef != next_treedef:
        raise CarryCanonicalizationError(
            f"WhileCarry structure mismatch: init treedef {init_treedef} != "
            f"body return treedef {next_treedef}"
        )

    init_leaves = jax.tree_util.tree_leaves(init_carry)
    next_leaves = jax.tree_util.tree_leaves(next_carry)

    for i, (in_leaf, out_leaf) in enumerate(zip(init_leaves, next_leaves, strict=True)):
        if hasattr(in_leaf, "shape") and hasattr(out_leaf, "shape"):
            if in_leaf.shape != out_leaf.shape:
                raise CarryCanonicalizationError(
                    f"WhileCarry leaf {i} shape mismatch: init shape {in_leaf.shape} != "
                    f"returned shape {out_leaf.shape}"
                )
            if in_leaf.dtype != out_leaf.dtype:
                raise CarryCanonicalizationError(
                    f"WhileCarry leaf {i} dtype mismatch: init dtype {in_leaf.dtype} != "
                    f"returned dtype {out_leaf.dtype}"
                )
```

#### 2.4 Cached JIT Runner for WhileCarry
`WhileLoopIterator` maintains an execution cache keyed on the loop function identities and carry signatures:

```python
# src/xtrax/tiling/iterator.py

class WhileLoopIterator(Struct):
    """Iterate via jax.lax.while_loop with carry-leaf canonicalization and JIT runner caching."""

    def __call__(self, cond: Any, body: Any, init: Any) -> Any:
        canonical_init = canonicalize_carry(init)

        # Wrap body to ensure canonicalization and strict invariant checking
        def safe_body(carry: Any) -> Any:
            new_carry = body(carry)
            validate_carry_transition(carry, new_carry)
            return new_carry

        return jax.lax.while_loop(cond, safe_body, canonical_init)
```

---

### 3. Debt #3192: `symbolic_zeros=True` Active-Tangent VJP, Primal+Grad Fusion, and 2D Pallas Grid Batching

#### 3.1 Motivation & Root Cause
In `src/xtrax/kernel_seam.py:566-582`:
1. `fn = jax.custom_vjp(fn)` calls `fn.defvjp(fn_fwd, fn_bwd)`. Without `symbolic_zeros=True`, JAX allocates explicit zero-filled buffers for all inactive output cotangents, wasting device HBM and memory bandwidth.
2. In training paths, forward residual caching and backward passes are evaluated independently without primal+grad fusion.
3. 2D Pallas kernels tiling over batch $B$ and length/state tiles $K$ lack support for stride-0 broadcast topology when one operand is batched (e.g. sequences `(B, N, D)`) and another is shared across the batch (e.g. weights `(N, D)` or graph adjacency `(N, N)`).

#### 3.2 `symbolic_zeros=True` in Custom VJP
In `Op._build_vjp_form(self, vk: VjpKernel)`:

```python
# src/xtrax/kernel_seam.py

def _build_vjp_form(self, vk: VjpKernel) -> Callable[..., Any]:
    """custom_vjp with symbolic_zeros=True to eliminate dense zero cotangent allocations."""
    reference = self.reference

    @jax.custom_vjp
    def fn(*args):
        return vk.fwd(*args)[0]

    def fn_fwd(*args):
        out, residuals = vk.fwd(*args)
        return out, (args, residuals)

    def fn_bwd(saved, ct):
        args, residuals = saved
        
        # Check for symbolic zeros across cotangents
        # If ct is SymbolicZero, or all leaves in ct are SymbolicZero,
        # return SymbolicZero for input cotangents without kernel dispatch
        leaves = jax.tree_util.tree_leaves(ct)
        if all(isinstance(leaf, jax.custom_derivatives.SymbolicZero) for leaf in leaves):
            return tuple(
                jax.custom_derivatives.SymbolicZero(jax.core.ShapedArray(a.shape, a.dtype)) 
                if hasattr(a, "shape") else None
                for a in args
            )

        # Coerce any partial SymbolicZero leaves to zero-arrays directly from leaf.aval
        # without redundant re-evaluation of fn_fwd(*args) on device (Debt #3192 Finding C2)
        def _resolve_ct_leaf(leaf):
            if isinstance(leaf, jax.custom_derivatives.SymbolicZero):
                return jnp.zeros(leaf.aval.shape, leaf.aval.dtype)
            return leaf

        resolved_ct = jax.tree_util.tree_map(_resolve_ct_leaf, ct)
        return _normalize_cotangents(args, vk.bwd(residuals, resolved_ct))

    # Enable symbolic_zeros=True in VJP registration
    fn.defvjp(fn_fwd, fn_bwd, symbolic_zeros=True)
    return fn
```

**Autodiff Purity Note (Debt #3192 Finding C2)**:
Resolving `SymbolicZero` leaves inside `fn_bwd` MUST NOT call `fn_fwd(*args)`. An instance of `jax.custom_derivatives.SymbolicZero` retains its abstract value in `leaf.aval` (providing `.shape` and `.dtype`). Querying `leaf.aval` allows instantaneous creation of matching zero-tensors via `jnp.zeros(leaf.aval.shape, leaf.aval.dtype)` without executing redundant forward kernel passes or doubling device FLOPs during backpropagation. Retaining `@jax.custom_jvp` on `fwd_pair` and `bwd_apply` ensures second-order autodiff correctly falls back to reference execution.
```

#### 3.3 Primal+Grad Fusion
Introduce `fused_primal_and_grad` on `Op` and `VjpKernel`:
- Combines the evaluation of $f(x)$ and $\nabla f(x)$ into a single pass when residuals can be kept entirely in device SRAM/registers (via Pallas kernel fusion).
- Avoids writing intermediate forward activation tensors to HBM.

#### 3.4 2D Pallas Grid Batching with Stride-0 Shared Topology
Provide a grid and `BlockSpec` builder for batched 2D Pallas execution where batch items share common parameter or adjacency matrices:

```python
# src/xtrax/op/grid.py (new module)

import jax
from jax.experimental import pallas as pl

def make_2d_grid_specs(
    batch_size: int,
    total_elements: int,
    block_size: int,
) -> tuple[tuple[int, int], pl.BlockSpec, pl.BlockSpec]:
    """Construct 2D Pallas grid and BlockSpecs for batched inputs with shared topology.
    
    Grid shape: (batch_size, num_tiles)
    
    Returns:
        (grid, batched_block_spec, shared_block_spec)
        - batched_block_spec: maps (b, i) -> (b, i * block_size)
        - shared_block_spec: maps (b, i) -> (0, i * block_size) [stride-0 along batch]
    """
    if total_elements % block_size != 0:
        raise ValueError(f"total_elements ({total_elements}) must be divisible by block_size ({block_size})")
        
    num_tiles = total_elements // block_size
    grid = (batch_size, num_tiles)

    # Batched operand: indexing depends on both batch index b and tile index i
    # Signature: pl.BlockSpec(block_shape, index_map, ...) (Debt #3192 Finding C1)
    batched_block_spec = pl.BlockSpec(
        block_shape=(1, block_size),
        index_map=lambda b, i: (b, i * block_size),
    )

    # Shared operand: index along batch axis is fixed to 0 (stride-0 shared topology)
    shared_block_spec = pl.BlockSpec(
        block_shape=(1, block_size),
        index_map=lambda b, i: (0, i * block_size),
    )

    return grid, batched_block_spec, shared_block_spec
```

**BlockSpec Constructor Invariant (Debt #3192 Finding C1)**:
In JAX Pallas, `pl.BlockSpec` constructor signature is `BlockSpec(block_shape, index_map, ...)`. Passing positional arguments in reverse order (`index_map, block_shape`) erroneously binds the mapping function to `block_shape`, causing JAX's internal `to_block_mapping()` to attempt iterating over the callable and crashing with `TypeError: 'function' object is not iterable`. Implementations MUST pass keyword arguments (`block_shape=..., index_map=...`) or positional arguments strictly ordered as `(block_shape, index_map)`.

---

### 4. Debt #3116: Split Training Dependencies in `pyproject.toml`

#### 4.1 Motivation & Root Cause
In `pyproject.toml:7`:
```toml
dependencies = [
    "jax>=0.10.2,<0.12",
    "jaxlib>=0.10.2,<0.12",
    "jaxtyping>=0.3.3",
    "optax>=0.2.3",
    "orbax-checkpoint>=0.11.17",
    "numpy>=2.1",
]
```
`optax` and `orbax-checkpoint` are declared as required core dependencies. However, `xtrax` consumers (`aminx`, `demistify`) and lightweight deployment targets (IREE, ONNX, WebGPU, CLI packing) only require JAX array manipulation, tiling, and campaign coordination.

#### 4.2 Dependency Decoupling Architecture
1. Move `optax` and `orbax-checkpoint` from `project.dependencies` to `[project.optional-dependencies].training`:
   ```toml
   # pyproject.toml
   dependencies = [
       "jax>=0.10.2,<0.12",
       "jaxlib>=0.10.2,<0.12",
       "jaxtyping>=0.3.3",
       "numpy>=2.1",
   ]

   [project.optional-dependencies]
   training = [
       "optax>=0.2.3",
       "orbax-checkpoint>=0.11.17",
   ]
   ```
2. Update composite extras:
   - `dev` extra MUST include `"xtrax[training]"`.
   - Add/update `all` extra to include `"xtrax[training]"`.
   - Update `[dependency-groups].dev = ["xtrax[dev]"]` to satisfy the alias contract in `tests/distribution/test_dependency_group_alias.py`.
3. Guarded imports in `src/xtrax/training/` and `src/xtrax/checkpoint/`:
   ```python
   # src/xtrax/training/optim.py
   try:
       import optax
   except ImportError as err:
       raise ImportError(
           "xtrax.training requires 'optax'. Install it with: pip install 'xtrax[training]'"
       ) from err

   # src/xtrax/checkpoint/orbax.py
   try:
       import orbax.checkpoint as ocp
   except ImportError as err:
       raise ImportError(
           "xtrax.checkpoint requires 'orbax-checkpoint'. Install it with: pip install 'xtrax[training]'"
       ) from err
   ```

---

### 5. Debt #3044: Opt-In Sparsity Specification in Planning & Execution

#### 5.1 Motivation & Root Cause
During the thin-wrapper audit (#6107), `src/xtrax/sparse/` was deleted because `sparse_filter_jit` was merely a re-export of `eqx.filter_jit`. However, declarative sparsity specification is essential: when processing large biological or physical interaction graphs, density is often $< 5\%$. Planning based on dense tensor dimensions causes `BatchPlanner` to over-estimate memory by 20x to 100x, triggering unnecessary tiling or refusing execution.

#### 5.2 Declarative Sparsity Design
1. Add `SparseSpec` dataclass:
   ```python
   # src/xtrax/tiling/sparse_spec.py (or in src/xtrax/tiling/strategy.py)

   @dataclass(frozen=True)
   class SparseSpec:
       """Declarative specification of tensor sparsity.
       
       Attributes:
           density: Expected proportion of non-zero elements in (0.0, 1.0].
           format: Layout format ("bcoo", "coo", "csr").
           index_dtype: Dtype of sparse indices (default: jnp.int32).
       """
       density: float
       format: Literal["bcoo", "coo", "csr"] = "bcoo"
       index_dtype: Any = jnp.int32

       def __post_init__(self) -> None:
           if not (0.0 < self.density <= 1.0):
               raise ValueError(f"SparseSpec: density must be in (0.0, 1.0], got {self.density}")
   ```
2. Extend `Context` and `AxisSpec`:
   - In `src/xtrax/intent/context.py`: Add `sparsity: float | SparseSpec | None = None` to `Context`.
   - In `src/xtrax/tiling/plan.py`: Add `sparsity: float | SparseSpec | None = None` to `AxisSpec`.
3. Result-Changing Planner Logic in `BatchPlanner.plan()`:
   - When computing memory footprint:
     ```python
     effective_density = 1.0
     if spec.sparsity is not None:
         effective_density = spec.sparsity.density if isinstance(spec.sparsity, SparseSpec) else float(spec.sparsity)
     memory_bytes = int(dense_bytes * effective_density)
     ```
   - Strategy Selection:
     - When `effective_density < 0.2`, the planner records `is_sparse=True` in `AxisDecision` and selects higher chunk sizes or specialized sparse mapping kernels.

---

### 6. Debt #3043: Re-build `ShardingPolicy` as an Active Multi-Device Placement Planner

#### 6.1 Motivation & Root Cause
`ShardingPolicy` was deprecated in commit `f7766c0` because it was a passive regex lookup table over `PartitionSpec`. Multi-device execution across meshes requires an active planner that checks array dimensions against mesh dimensions, ensures divisibility, avoids illegal shardings, and produces concrete `NamedSharding` instances.

#### 6.2 `ShardingPlanner` Architecture
```python
# src/xtrax/distributed/sharding.py

class ShardingPlannerError(ValueError):
    """Raised when tensor shapes are incompatible with device mesh placement."""
    pass

class ShardingPlanner:
    """Active placement planner resolving PartitionSpecs into NamedShardings across a device Mesh.
    
    Attributes:
        mesh: Active jax.sharding.Mesh instance.
        policy: ShardingPolicy containing path matching rules.
    """
    mesh: jax.sharding.Mesh
    policy: ShardingPolicy

    def __init__(self, mesh: jax.sharding.Mesh, policy: ShardingPolicy) -> None:
        self.mesh = mesh
        self.policy = policy

    def plan_sharding(self, pytree: Any) -> Any:
        """Derive concrete NamedSharding for each PyTree leaf, verifying divisibility.
        
        Raises:
            ShardingPlannerError: If a tensor dimension is not divisible by the target mesh axis size.
        """
        def _plan_leaf(path: tuple, leaf: Any) -> jax.sharding.NamedSharding:
            if not hasattr(leaf, "shape"):
                return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

            path_str = self.policy._path_to_string(path)
            pspec = self.policy.get_partition_spec(path_str)

            # Validate rank compatibility (Debt #3043 Finding M2)
            if len(pspec) > len(leaf.shape):
                raise ShardingPlannerError(
                    f"ShardingPlanner: PartitionSpec for '{path_str}' has rank {len(pspec)} "
                    f"exceeding leaf rank {len(leaf.shape)} (shape {leaf.shape})."
                )

            # Validate divisibility across single and tuple mesh axes (Debt #3043 Finding M2)
            mesh_shape_dict = dict(zip(self.mesh.axis_names, self.mesh.shape, strict=True))
            for dim_idx, axis_name in enumerate(pspec):
                if axis_name is None:
                    continue
                dim_size = leaf.shape[dim_idx]
                if isinstance(axis_name, (tuple, list)):
                    total_axis_size = 1
                    for sub_axis in axis_name:
                        if sub_axis not in mesh_shape_dict:
                            raise ShardingPlannerError(
                                f"ShardingPlanner: PartitionSpec for '{path_str}' references unknown "
                                f"mesh axis '{sub_axis}'. Available: {list(mesh_shape_dict.keys())}"
                            )
                        total_axis_size *= mesh_shape_dict[sub_axis]
                    if dim_size % total_axis_size != 0:
                        raise ShardingPlannerError(
                            f"ShardingPlanner: Incompatible placement for '{path_str}'. "
                            f"Dimension {dim_idx} (size {dim_size}) is not divisible by "
                            f"combined mesh axes {axis_name} (product {total_axis_size})."
                        )
                else:
                    if axis_name not in mesh_shape_dict:
                        raise ShardingPlannerError(
                            f"ShardingPlanner: PartitionSpec for '{path_str}' references unknown "
                            f"mesh axis '{axis_name}'. Available: {list(mesh_shape_dict.keys())}"
                        )
                    axis_size = mesh_shape_dict[axis_name]
                    if dim_size % axis_size != 0:
                        raise ShardingPlannerError(
                            f"ShardingPlanner: Incompatible placement for '{path_str}'. "
                            f"Dimension {dim_idx} (size {dim_size}) is not divisible by "
                            f"mesh axis '{axis_name}' (size {axis_size})."
                        )

            return jax.sharding.NamedSharding(self.mesh, pspec)

        return jax.tree_util.tree_map_with_path(_plan_leaf, pytree)
```

**Tuple Mesh Axis and Rank Validation (Debt #3043 Finding M2)**:
JAX `PartitionSpec` allows a tensor dimension to be mapped to a tuple of mesh axes (e.g. `P(('data', 'model'), None)`), in which case the tensor dimension is split across the product of the named mesh axes. `ShardingPlanner` validates that all constituent sub-axes exist in `mesh.axis_names` and that their product divides the dimension size. Furthermore, `len(pspec)` is verified not to exceed `len(leaf.shape)` to prevent unhandled `IndexError` on higher-rank partition specs.

---

### 7. Debt #3042: Context-Driven `NaN`/`Inf` Safety Checking

#### 7.1 Motivation & Root Cause
`SafetyManager` was deprecated in commit `f7766c0` as a thin wrapper around `checkify`. However, runtime numerical safety cannot be an ambient or separate wrapper class. It must be an integrated compilation decision specified in `Context`.

#### 7.2 Context-Driven Safety Decision
1. Extend `Context` (`src/xtrax/intent/context.py`):
   ```python
   @dataclass(frozen=True)
   class Context:
       # Existing fields...
       safety: bool = False
       safety_checks: tuple[str, ...] = ("nan", "div")
   ```
2. Compiler Integration (`src/xtrax/intent/_compile.py`, Debt #3042 Finding M1):
   - When `context.safety is True`:
     - Builds error flags:
       ```python
       import jax.experimental.checkify as checkify
       errors = frozenset()
       if "nan" in context.safety_checks:
           errors |= checkify.nan_checks
       if "div" in context.safety_checks:
           errors |= checkify.div_checks
       if "inf" in context.safety_checks:
           errors |= (checkify.nan_checks | checkify.div_checks)
       ```
     - Instruments the compiled callable:
       ```python
       checked_fn = checkify.checkify(fn, errors=errors)
       @functools.wraps(fn)
       def safe_execution(*args, **kwargs):
           err, result = checked_fn(*args, **kwargs)
           err.throw()  # Outside JIT: raises Python exception on host
           return result
       ```
     - Emits `safety="checkified"` in the compiled `SiteDecision` record.
   - When `context.safety is False`:
     - Compiles directly with zero wrapping overhead.

---

### 8. Debt #2543: `CarrySpec.transition` End-to-End Execution Contract

#### 8.1 Motivation & Root Cause
`src/xtrax/tiling/carry.py` defines `CarrySpec(axis_name, init, transition, collect_outputs, cond)`. While unit tests in `tests/tiling/test_carry.py` test class fields and immutability, there is no contract test proving that a user-provided `transition` executes end-to-end through `BatchPlanner.plan()`, `axis_dispatch`, and `safe_scan` / `lax.while_loop`.

#### 8.2 Execution Contract Specification
The contract requires:
1. `collect_outputs=True` (Scan path):
   - `BatchPlanner` creates `AxisDecision(strategy=Scan(init=cs.init, transition=cs.transition))`.
   - `axis_dispatch` executes `safe_scan(strategy.transition, carry_init, xs)`.
   - The returned tuple `(final_carry, ys)` matches exact mathematical recurrence:
     $c_{t} = \text{transition}(c_{t-1}, x_t)[0]$, $y_t = \text{transition}(c_{t-1}, x_t)[1]$.
2. `collect_outputs=False` (WhileCarry path):
   - `BatchPlanner` creates `AxisDecision(strategy=WhileCarry(init=cs.init, body=cs.transition, cond=cs.cond))`.
   - `axis_dispatch` executes `jax.lax.while_loop(strategy.cond, strategy.body, carry_init)`.
   - Returns final carry after condition terminates.

---

## Verification Controls & Matrix

| Debt ID | Feature / Component | Positive Verification Control | Negative Verification Control | Test File |
|---|---|---|---|---|
| **#3194** | `BucketDispatcher` | Routes variable-length batches to correct bucket rungs; reassembles in original order with exact values. | Length $> \max(\text{boundaries})$ raises `ValueError`. Empty boundaries raises `ValueError`. | `tests/tiling/test_bucket_dispatcher.py` |
| **#3194** | `vmap_any_cond` | Executes `work_fn` across batch when any item satisfies `pred_fn`; executes `default_fn` when none do. | Predicate evaluation returning non-boolean shape raises `TypeError`. | `tests/tiling/test_vmap_any_cond.py` |
| **#3194** | `BatchedCapacityState` | Accurately calculates headroom; signals `needs_reallocation=True` when capacity is breached. | Capacity $< 0$ raises `ValueError`. | `tests/tiling/test_batched_capacity.py` |
| **#3193** | `WhileCarry` Dynamic Steps | Executes loop to completion with dynamic scalar `n_steps` (`jax.Array` tracer). | Negative `n_steps` terminates immediately without error. | `tests/tiling/test_while_dynamic_steps.py` |
| **#3193** | Carry-Leaf Canonicalization | Canonicalizes NumPy/scalar inputs into uniform JAX array leaves. | Mismatched return PyTree or altered shape/dtype raises `CarryCanonicalizationError`. | `tests/tiling/test_while_canonicalization.py` |
| **#3192** | `symbolic_zeros=True` VJP | Un-differentiated outputs receive `SymbolicZero`; zero cotangent allocation avoided. | Inactive output cotangent access does not corrupt active gradient values. | `tests/op/test_symbolic_zeros.py` |
| **#3192** | 2D Pallas Grid Batching | 2D grid `(B, num_tiles)` runs kernel with stride-0 shared operand. | Non-divisible element count raises `ValueError`. | `tests/op/test_pallas_grid_batching.py` |
| **#3116** | Packaging Decoupling | `import xtrax.tiling` succeeds in clean environment without `optax`. `pip install .[training]` installs `optax`. | Accessing `xtrax.training.optim` without extra raises actionable `ImportError`. | `tests/distribution/test_training_extra.py` |
| **#3044** | Opt-In Sparsity | `Context(sparsity=0.05)` reduces estimated memory footprint; planner selects higher batch size. | Density outside $(0.0, 1.0]$ raises `ValueError`. | `tests/planning/test_opt_in_sparsity.py` |
| **#3043** | `ShardingPlanner` | Valid mesh placement (including tuple mesh axes) generates valid `NamedSharding` across leaves. | Incompatible tensor shape not divisible by mesh axis, unknown mesh axis, or PartitionSpec rank exceeding tensor rank raises `ShardingPlannerError`. | `tests/distributed/test_sharding_planner.py` |
| **#3042** | Context-Driven Safety | `Context(safety=True)` detects NaN/Inf division and raises Python exception on host. | Unsafe input when `safety=False` returns NaN without raising. | `tests/intent/test_context_safety.py` |
| **#2543** | `CarrySpec.transition` Contract | End-to-end execution of `CarrySpec` through planner and dispatch produces verified mathematical outputs. | Heterogeneous axis in `CarrySpec` raises `ValueError` during planning. | `tests/tiling/test_carry_execution_contract.py` |

---

## TDD Implementation & Verification Plan

### Test Commands
All test suites are executed via `uv run pytest`:

```bash
# 1. Tiling & Bucketing (#3194, #3193, #2543)
uv run pytest tests/tiling/test_bucket_dispatcher.py tests/tiling/test_vmap_any_cond.py tests/tiling/test_batched_capacity.py tests/tiling/test_while_dynamic_steps.py tests/tiling/test_while_canonicalization.py tests/tiling/test_carry_execution_contract.py -v --no-cov

# 2. Kernel Autodiff & Grid Batching (#3192)
uv run pytest tests/op/test_symbolic_zeros.py tests/op/test_pallas_grid_batching.py -v --no-cov

# 3. Packaging & Dependencies (#3116)
uv run pytest tests/distribution/test_training_extra.py tests/distribution/test_dependency_group_alias.py -v --no-cov

# 4. Planning, Sharding & Safety (#3044, #3043, #3042)
uv run pytest tests/planning/test_opt_in_sparsity.py tests/distributed/test_sharding_planner.py tests/intent/test_context_safety.py -v --no-cov

# 5. Full Static & Import Linter Gates
uv run ruff check .
uv run ruff format --check .
uv run ty check src/
uv run lint-imports
```

---

## File Mutation Inventory

### Modified Files
1. `pyproject.toml` — Move `optax` and `orbax-checkpoint` to `[project.optional-dependencies].training`, update `all` and `dev` extras.
2. `src/xtrax/tiling/bucket.py` — Add `BucketDispatcher`, `BatchedCapacityState`.
3. `src/xtrax/tiling/dispatch.py` — Add `vmap_any_cond`.
4. `src/xtrax/tiling/strategy.py` — Generalize `fixed_step_count_cond` to accept dynamic JAX arrays.
5. `src/xtrax/tiling/iterator.py` — Add carry-leaf canonicalization and runner caching in `WhileLoopIterator`.
6. `src/xtrax/tiling/carry.py` — Add `canonicalize_carry` and `validate_carry_transition`.
7. `src/xtrax/kernel_seam.py` — Update `custom_vjp` to set `symbolic_zeros=True` and handle `SymbolicZero` cotangents.
8. `src/xtrax/intent/context.py` — Add `sparsity`, `mesh`, `sharding_policy`, `safety`, and `safety_checks` fields.
9. `src/xtrax/intent/_compile.py` — Wire `Context.safety` to automatic checkification (Debt #3042 Finding M1).
10. `src/xtrax/distributed/sharding.py` — Add `ShardingPlanner` and `ShardingPlannerError`.
11. `src/xtrax/tiling/plan.py` — Support `sparsity` in `AxisSpec` and memory footprint calculation.

### New Modules
1. `src/xtrax/op/grid.py` — 2D Pallas grid and BlockSpec construction with stride-0 shared topology.
2. `src/xtrax/tiling/sparse_spec.py` — Declarative `SparseSpec` class.
3. `changelog.d/3194.added.md`, `changelog.d/3193.added.md`, `changelog.d/3192.added.md`, `changelog.d/3116.changed.md`, `changelog.d/3044.added.md`, `changelog.d/3043.added.md`, `changelog.d/3042.added.md`, `changelog.d/2543.added.md` — Fragment files for all modifications.

### New Test Suites
1. `tests/tiling/test_bucket_dispatcher.py`
2. `tests/tiling/test_vmap_any_cond.py`
3. `tests/tiling/test_batched_capacity.py`
4. `tests/tiling/test_while_dynamic_steps.py`
5. `tests/tiling/test_while_canonicalization.py`
6. `tests/tiling/test_carry_execution_contract.py`
7. `tests/op/test_symbolic_zeros.py`
8. `tests/op/test_pallas_grid_batching.py`
9. `tests/distribution/test_training_extra.py`
10. `tests/planning/test_opt_in_sparsity.py`
11. `tests/distributed/test_sharding_planner.py`
12. `tests/intent/test_context_safety.py`
