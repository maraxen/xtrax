---
task_id: 261110_xtrax-debt-burn-down-and-consumer-adoption
track: 3
track_name: Profiling, Testing & Observability
debts: [3197, 3196, 3195]
created: 2026-11-10
status: draft
---

# Track 3 Engineering Specification: Profiling, Testing & Observability

**Task ID**: `261110_xtrax-debt-burn-down-and-consumer-adoption`  
**Track**: 3 (Profiling, Testing & Observability)  
**Debts**: #3197, #3196, #3195  
**Date**: 2026-11-10  
**Status**: Draft  

---

## 1. Executive Summary & Scope

Track 3 addresses foundational capabilities in hardware-independent kernel verification, compiler observability, and profiling instrumentation across `xtrax`:

1. **Debt #3197 (Testing & Linting)**:
   - Provide a CPU-only offline CUDA Triton lowering verifier (`verify_triton_lowering`) in `src/xtrax/testing/triton_lowering.py` (exported via `src/xtrax/testing/__init__.py`). It leverages JAX abstract CUDA meshes (`AbstractDevice(platform='cuda')`, `use_abstract_mesh`) to prove that Pallas kernels lower to `@__gpu$xla.gpu.triton` custom calls on CPU without physical NVIDIA GPU hardware or CUDA drivers.
   - Enforce `JL-PALLAS` static lint rules in `scripts/lint_pallas.py` and `src/xtrax/jaxlint_runner.py` / `scripts/audit_jaxlint_json.py` to disallow `interpret=True` in production, require explicit `compiler_params`, ban Python control flow in kernel bodies, and require power-of-2 block dimensions.
2. **Debt #3196 (Observability & Profiling)**:
   - In `src/xtrax/profiling/loop_scaling.py`, detect and flag `vmap`-induced `scan` loops over `pallas_call` or custom calls that cause serialized execution degradation on accelerator streams.
   - In `src/xtrax/profiling/trace.py`, auto-attribute GPU custom-call kernels (including Triton custom calls) within execution traces and HLO instruction maps, eliminating unattributed thunk fallbacks.
   - In `src/xtrax/profiling/record.py`, extend `ProbeRecord` with multi-replica and bucketing metadata (`num_replicas`, `replica_id`, `bucket_index`, `bucket_capacity`) with strict invariant checking and schema serialization.
3. **Debt #3195 (Compiler Observability)**:
   - In `src/xtrax/profiling/compile_count.py`, add `explain_cache_miss` and `diff_jit_signature` diagnostic utilities. When `assert_no_recompile_after` fails, it pinpoints the exact cause of recompilation by diffing PyTree structures, leaf shapes, dtypes, and static parameter variations between warmup and failing iterations.

---

## 2. Debt #3197: Offline CUDA Triton Lowering Verifier & JL-PALLAS Lint Rules

### 2.1 Problem Analysis

In `xtrax`, Pallas kernels (such as custom matrix multiplications, binned sums, and histogram primitives in `tests/op/_kernels.py`) are primarily executed using `interpret=True` during CPU unit testing. However:
- `interpret=True` executes kernels via Python loops on CPU arrays, bypassing compiler transformations, block index computations, memory layouts, and backend lowering passes.
- Running full GPU execution tests requires physical NVIDIA accelerators and CUDA drivers, which are absent in standard CI environments, local CPU development workstations, and multi-platform packaging runners.
- Bugs in grid indexing, unsupported dialect operations, invalid block dimensions, or backend parameter mismatches go undetected until late deployment on GPU clusters.

### 2.2 Mechanism: Abstract CUDA Mesh Lowering on CPU

JAX's compilation pipeline allows abstract evaluation without hardware execution. By binding an abstract CUDA device mesh using `jax._src.mesh.use_abstract_mesh`, JAX's compiler (`jax.jit(...).lower(...)`) lowers Pallas kernels through the GPU lowering route, invoking Triton MLIR generation and emitting StableHLO custom calls on CPU.

#### Verified Working Mechanism
```python
from jax._src import mesh as mesh_lib
from jax.experimental.pallas import triton as plt
import jax

# Configure an abstract CUDA device and mesh
device = mesh_lib.AbstractDevice(device_kind="NVIDIA A100", num_cores=None, platform="cuda")
mesh = mesh_lib.AbstractMesh((1,), ("x",), abstract_device=device)

with mesh_lib.use_abstract_mesh(mesh):
    lowered = jax.jit(kernel_fn).lower(*args)
    hlo_text = lowered.as_text()
```

When successfully lowered to CUDA Triton, the resulting StableHLO module contains:
```mlir
%0 = stablehlo.custom_call @__gpu$xla.gpu.triton(%arg0, ...) {
    backend_config = "",
    mhlo.backend_config = {
        debug = false,
        grid_x = 1 : i32,
        grid_y = 1 : i32,
        grid_z = 1 : i32,
        ir = "ML\EFR\0DMLIR...",
        name = "...",
        num_stages = 3 : i32,
        num_warps = 4 : i32
    }
}
```

### 2.3 Architectural Design: `src/xtrax/testing/triton_lowering.py`

#### 2.3.1 Data Structures & Error Classes

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import re

class TritonLoweringError(AssertionError):
    """Raised when a Pallas kernel fails to lower to valid CUDA Triton HLO."""

@dataclass(frozen=True, slots=True)
class TritonLoweringResult:
    """Diagnostic outcome of an offline CUDA Triton lowering verification."""
    hlo_text: str
    grid: tuple[int, int, int]
    num_warps: int | None
    num_stages: int | None
    kernel_name: str | None
    raw_backend_config: str
```

#### 2.3.2 Verification Function Signature & Implementation

```python
def verify_triton_lowering(
    kernel_fn: Callable[..., Any],
    *args: Any,
    compiler_params: Any | None = None,
    expected_grid: tuple[int, int, int] | None = None,
    expected_num_warps: int | None = None,
    expected_num_stages: int | None = None,
    device_kind: str = "NVIDIA A100",
    **kwargs: Any,
) -> TritonLoweringResult:
    """Verify that `kernel_fn(*args)` lowers to CUDA Triton on CPU offline.

    Uses an abstract CUDA mesh to drive JAX/XLA lowering without GPU hardware.
    Confirms that StableHLO emits `@__gpu$xla.gpu.triton` and validates grid
    and compiler configurations when specified.

    Args:
        kernel_fn: Pallas kernel callable or function wrapping `pl.pallas_call`.
        *args: Example inputs (arrays or `jax.ShapeDtypeStruct` instances).
        compiler_params: Optional Triton compiler parameters (`plt.CompilerParams`).
        expected_grid: Expected (grid_x, grid_y, grid_z) tuple, if asserted.
        expected_num_warps: Expected num_warps setting, if asserted.
        expected_num_stages: Expected num_stages setting, if asserted.
        device_kind: Abstract device model name (defaults to "NVIDIA A100").
        **kwargs: Additional keyword arguments passed to `kernel_fn`.

    Returns:
        A `TritonLoweringResult` containing the extracted HLO text, grid,
        warps, stages, and kernel metadata.

    Raises:
        TritonLoweringError: If lowering fails, emits Mosaic GPU or generic HLO
            instead of Triton, or mismatches expected grid/warp/stage constraints.
    """
```

#### 2.3.3 Backend Config Extraction Logic
The verifier parses `mhlo.backend_config` from the lowered text:
- `grid_x = (\d+) : i32`, `grid_y = (\d+) : i32`, `grid_z = (\d+) : i32`
- `num_warps = (\d+) : i32`
- `num_stages = (\d+) : i32`
- `name = "([^"]+)"`
- Custom call target `@__gpu$xla.gpu.triton` validation:
  If the HLO text lacks `@__gpu$xla.gpu.triton`, the verifier identifies whether it lowered to Mosaic GPU (`mosaic_gpu`) or plain StableHLO primitives, and raises `TritonLoweringError` detailing the fallback.

#### 2.3.4 Re-export
Export `verify_triton_lowering`, `TritonLoweringResult`, and `TritonLoweringError` in `src/xtrax/testing/__init__.py`.

---

### 2.4 JL-PALLAS Static Lint Rules

Static lint rules are added to enforce Pallas coding standards across the codebase. These rules are integrated into `scripts/lint_pallas.py` and invoked via `src/xtrax/jaxlint_runner.py` / `scripts/audit_jaxlint_json.py`.

| Rule ID | Name | Severity | Description | Target Files |
|---|---|---|---|---|
| **JL-PALLAS01** | `no-interpret-in-src` | Major / Error | Disallow `interpret=True` in production code under `src/xtrax/`. Allowed only in `tests/`. | `src/xtrax/**/*.py` |
| **JL-PALLAS02** | `explicit-compiler-params` | Minor / Warning | GPU-targeted `pallas_call` must supply `compiler_params=plt.CompilerParams(...)` to avoid ambiguous backend selection (e.g. Mosaic fallback). | `src/xtrax/**/*.py` |
| **JL-PALLAS03** | `power-of-two-block-spec` | Minor / Warning | Block sizes in `pl.BlockSpec` must be powers of 2 (or statically bounded constants) for memory bus alignment. | `src/xtrax/**/*.py`, `tests/` |
| **JL-PALLAS04** | `no-python-control-in-kernel` | Major / Error | Python control flow (`if`, `while`, `for`) inside Pallas kernel body functions is prohibited; must use `pl.when`, `pl.loop`. | All Pallas kernel definitions |

#### Implementation in `scripts/lint_pallas.py`:
The script parses Python AST (`ast.parse`) across candidate files:
1. **JL-PALLAS01**: Searches for `Call` nodes where `func` is `pallas_call` (or imported aliases). If keyword argument `interpret` has `value=Constant(value=True)` and the file path is under `src/xtrax/`, emit JL-PALLAS01 finding.
2. **JL-PALLAS02**: If `pallas_call` lacks `compiler_params` keyword argument when `interpret` is not `True`, emit JL-PALLAS02 finding.
3. **JL-PALLAS03**: Inspects `BlockSpec` calls (both positional argument 0 `block_shape` and keyword `block_shape=...`, Debt #3197 Finding M4).
   - For each dimension in the `block_shape` sequence:
     - If the dimension node is an `ast.Constant` with an integer value: checks whether `value > 0 and (value & (value - 1)) == 0`. Emits a `JL-PALLAS03` warning if the integer literal is not a power of 2.
     - If the dimension node is a non-literal symbolic identifier, attribute access, or arithmetic expression (e.g., `ast.Name(id='BLOCK_SIZE')`, `ast.Attribute`, or `ast.BinOp(op=ast.FloorDiv)`), the visitor safely inspects `ast.Constant.value` only when available and ignores unresolved symbolic identifiers rather than raising an unhandled AST attribute or indexing exception.
     - If the dimension node is `ast.Constant(value=None)`, it represents an unblocked/full axis and is accepted as valid.
4. **JL-PALLAS04**: Analyzes the function definition passed as `kernel_fn` body to `pallas_call`; flags any `ast.If`, `ast.For`, or `ast.While` nodes within the body.

---

## 3. Debt #3196: Profiling Observability (vmap over pallas_call, auto-attribution, multi-replica)

### 3.1 Sub-Task A: Flagging `vmap`-Induced `scan` over `pallas_call` in `loop_scaling.py`

#### Problem
When `jax.vmap` is applied over functions containing `pl.pallas_call` that lack an analytical vectorization rule or whose grid cannot be automatically expanded, JAX or user-space fallbacks may transform the vectorization into an iterative `lax.scan` or sequential loop. In GPU execution, launching a Pallas kernel sequentially inside a scan incurs catastrophic launch latency (`O(L)` host/device synchronizations), defeating the purpose of hardware acceleration.

#### Architectural Design in `src/xtrax/profiling/loop_scaling.py`:
1. **Jaxpr Inspection**:
   Extend `_jaxpr_work(jaxpr, path, found)` to recursively analyze loop bodies:
   ```python
   def _contains_pallas_call(jaxpr: Any) -> bool:
       """Check if a jaxpr or any of its nested sub-jaxprs contains a pallas_call."""
       for eqn in jaxpr.eqns:
           if eqn.primitive.name in ("pallas_call", "custom_call"):
               return True
           for sub in sub_jaxprs(eqn.params):
               if _contains_pallas_call(sub):
                   return True
       return False
   ```
2. **Dataclass Extensions**:
   - In `LoopBody`:
     ```python
     has_pallas_call: bool = False
     ```
   - In `ScalingFinding`:
     ```python
     has_pallas_call: bool = False
     vmap_induced_pallas_scan: bool = False
     ```
   - In `ExtentScalingReport`:
     Add property:
     ```python
     @property
     def pallas_scan_findings(self) -> tuple[ScalingFinding, ...]:
         """Findings where a scan or while loop wraps a Pallas or custom call."""
         return tuple(f for f in self.findings if f.has_pallas_call)
     ```
3. **Diagnostic Warning & Flagging**:
   When `body.primitive == "scan"` and `has_pallas_call` is `True`, mark `vmap_induced_pallas_scan = True`. The report highlights these findings with warning diagnostics:
   `"WARNING: Loop at '{path}' is a scan over a pallas_call (likely vmap-induced). Serialized kernel launches will severely degrade stream throughput."`

---

### 3.2 Sub-Task B: Auto-Attribution of Custom-Call Kernels in `trace.py`

#### Problem
In JAX execution traces, instructions lowered as custom calls (such as Triton GPU launches or foreign C/CUDA calls) frequently appear with generic instruction thunk names (e.g. `custom-call.1`, `command_buffer_14`) or without standard `op_name` metadata. As a result, `scope_map_from_hlo_text` fails to resolve them, leaving substantial portions of GPU runtime unattributed in `ProbeRecord.scopes`.

#### Architectural Design in `src/xtrax/profiling/trace.py`:
1. **Custom-Call HLO Pattern Matching**:
   Add regular expressions and parsers for custom call instructions in HLO text:
   ```python
   _CUSTOM_CALL_RE = re.compile(
       r'^\s*(%[\w.\-]+)\s*=\s*[^=]*\bcustom-call\s*\([^)]*\)\s*,'
       r'\s*custom_call_target="([^"]+)"'
   )
   _BACKEND_CONFIG_NAME_RE = re.compile(r'name\s*=\s*"([^"]+)"')
   ```
2. **Attribute Resolution Pipeline**:
   Update `scope_map_from_hlo_text`:
   - Inspect custom call instructions.
   - Extract the custom call target (`custom_call_target`).
   - If target is `"__gpu$xla.gpu.triton"`, parse `backend_config` to extract the kernel `name`.
   - If `metadata={op_name="..."}` exists, resolve via `_deepest_known_label` against `known_labels`.
   - If `op_name` is absent but the extracted kernel name matches or contains a `known_labels` token, attribute the instruction directly to that known label.
   - If no known label matches, map the instruction to the sanitized kernel name (e.g. `"triton:matmul_kernel"`), ensuring it is attributed rather than lost.

---

### 3.3 Sub-Task C: Multi-Replica Bucket Fields in `ProbeRecord` (`record.py`)

#### Problem
`ProbeRecord` currently captures single-device profiling metrics (`stage`, `n_atoms`, `platform`, `device_kind`, `metrics`, `scopes`, `config`). For SPMD distributed scaling, multi-replica execution, and bucketing memory planners, benchmark records cannot specify which replica was probed or which bucket capacity was executed, preventing automated validation of scaling efficiency.

#### Architectural Design in `src/xtrax/profiling/record.py`:
1. **Field Definitions**:
   Add four multi-replica bucket fields to `ProbeRecord`:
   ```python
   @dataclasses.dataclass(frozen=True, slots=True)
   class ProbeRecord:
       probe_id: str
       stage: int
       n_atoms: int
       platform: str  # "cpu" | "gpu"
       ...
       # Multi-replica & Bucketing Observability (#3196)
       num_replicas: int = 1
       replica_id: int | None = None
       bucket_index: int | None = None
       bucket_capacity: int | None = None
   ```
2. **Invariants Enforced in `__post_init__`**:
   - `num_replicas`: Must be an integer `int >= 1`. Reject `bool` (following `stage` and `n_atoms` convention).
   - `replica_id`: If not `None`:
     - Must be an integer `int` (reject `bool`).
     - Must satisfy `0 <= replica_id < num_replicas`.
     - When `num_replicas == 1`, `replica_id` must be either `None` or `0`.
   - `bucket_index`: If not `None`:
     - Must be an integer `int >= 0` (reject `bool`).
   - `bucket_capacity`: If not `None`:
     - Must be an integer `int >= 1` (reject `bool`).
   - Co-dependency invariant:
     - If both `bucket_index` and `bucket_capacity` are provided, enforce `bucket_index <= bucket_capacity` (or index within ladder bounds).
3. **Serialization & Round-trip Compatibility**:
   - Update `to_dict()`, `from_dict()`, `to_json()`, `from_json()`.
   - In `from_json()` / `from_dict()`:
     - Allow optional fields to default gracefully (`num_replicas=1`, others `None`) when deserializing legacy JSON records.
     - Reject unknown fields to preserve strict schema protection against laundering.

---

## 4. Debt #3195: JIT Recompile Diagnostics (`explain_cache_miss` / `diff_jit_signature`)

### 4.1 Problem Analysis

In `src/xtrax/profiling/compile_count.py`, `assert_no_recompile_after` verifies that a stepped callable (e.g. `jax.jit` or `eqx.filter_jit`) does not recompile after warmup:
```python
if recorded.count != 0:
    raise AssertionError(
        f"{recorded.count} backend compile(s) ({recorded.seconds:.3f}s) "
        f"across {n_steps} step(s) after {warmup} warmup call(s)"
    )
```
When this fails, developers and CI test suites receive only the count of compilations. The error provides zero information about:
- Which execution step triggered the recompile.
- Which argument or PyTree leaf caused the cache miss.
- Whether the mismatch was caused by array shape changes, dtype promotion (e.g. float32 to float64), PyTree structure differences, or static argument variations.

### 4.2 Architectural Design in `src/xtrax/profiling/compile_count.py`

#### 4.2.1 Argument Signature Representation
We define an internal, lightweight signature extractor that captures JIT cache keys:
```python
@dataclass(frozen=True, slots=True)
class LeafSignature:
    path: str
    is_array: bool
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    static_value: Any = None
    type_name: str = ""

@dataclass(frozen=True, slots=True)
class CallSignature:
    treedef: Any
    leaves: tuple[LeafSignature, ...]
```

#### 4.2.2 Signature Extraction Helper
```python
def extract_signature(args: tuple[Any, ...]) -> CallSignature:
    """Extract PyTree structure, leaf shapes, dtypes, and static values."""
    import jax
    leaves, treedef = jax.tree_util.tree_flatten_with_path(args)
    leaf_sigs = []
    for path, leaf in leaves:
        path_str = jax.tree_util.keystr(path)
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            leaf_sigs.append(
                LeafSignature(
                    path=path_str,
                    is_array=True,
                    shape=tuple(leaf.shape),
                    dtype=str(leaf.dtype),
                    type_name=type(leaf).__name__,
                )
            )
        else:
            leaf_sigs.append(
                LeafSignature(
                    path=path_str,
                    is_array=False,
                    static_value=leaf,
                    type_name=type(leaf).__name__,
                )
            )
    return CallSignature(treedef=treedef, leaves=tuple(leaf_sigs))
```

#### 4.2.3 Signature Diffing & Explanation Functions

```python
def diff_jit_signature(
    warmup_args: tuple[Any, ...],
    failing_args: tuple[Any, ...],
) -> list[str]:
    """Diff two argument tuples and report discrepancies that trigger XLA recompile.

    Returns:
        A list of human-readable divergence descriptions.
    """
    sig_warmup = extract_signature(warmup_args)
    sig_failing = extract_signature(failing_args)
    diffs: list[str] = []

    # 1. PyTree Structure Check
    if sig_warmup.treedef != sig_failing.treedef:
        diffs.append(
            f"PyTree structure mismatch: warmup={sig_warmup.treedef} vs current={sig_failing.treedef}"
        )
        return diffs

    # 2. Leaf-by-Leaf Comparison
    for idx, (lw, lf) in enumerate(zip(sig_warmup.leaves, sig_failing.leaves, strict=True)):
        if lw.is_array != lf.is_array:
            diffs.append(
                f"Leaf [{idx}] '{lw.path}': array status changed from is_array={lw.is_array} "
                f"({lw.type_name}) to is_array={lf.is_array} ({lf.type_name})"
            )
            continue

        if lw.is_array:
            if lw.shape != lf.shape:
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': shape changed from {lw.shape} to {lf.shape}"
                )
            if lw.dtype != lf.dtype:
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': dtype changed from {lw.dtype} to {lf.dtype}"
                )
        else:
            # Guard against uncomparable objects or array comparisons (Debt #3195 Finding C6)
            try:
                is_diff = bool(lw.static_value != lf.static_value)
            except Exception:
                is_diff = repr(lw.static_value) != repr(lf.static_value)

            if is_diff:
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': static value changed from "
                    f"{lw.static_value!r} to {lf.static_value!r}"
                )
            elif type(lw.static_value) is not type(lf.static_value):
                diffs.append(
                    f"Leaf [{idx}] '{lw.path}': static type changed from "
                    f"{lw.type_name} to {lf.type_name}"
                )
    return diffs

def explain_cache_miss(
    warmup_args: tuple[Any, ...],
    failing_args: tuple[Any, ...],
    step_index: int | None = None,
) -> str:
    """Format an actionable diagnostic explanation of an XLA cache miss."""
    diffs = diff_jit_signature(warmup_args, failing_args)
    step_prefix = f" at step {step_index}" if step_index is not None else ""
    if not diffs:
        return (
            f"XLA cache miss detected{step_prefix}, but no PyTree, shape, dtype, "
            "or static leaf mismatch was found. Possible cause: internal compilation "
            "cache eviction, non-deterministic hash, or closure variable mutation."
        )
    formatted_diffs = "\n  - ".join(diffs)
    return f"XLA cache miss detected{step_prefix} due to signature divergence:\n  - {formatted_diffs}"
```

#### 4.2.4 Upgraded `assert_no_recompile_after` Implementation
```python
def assert_no_recompile_after(
    step_fn: Callable[..., Any],
    args_iter: Iterable[tuple[Any, ...]],
    warmup: int = 1,
) -> None:
    """Assert step_fn does not backend-compile after warmup calls, with rich diagnostics."""
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    import jax

    iterator = iter(args_iter)
    last_warmup_args: tuple[Any, ...] | None = None
    for i in range(warmup):
        try:
            args = next(iterator)
        except StopIteration:
            raise ValueError(f"args_iter ended during warmup ({i} of {warmup} calls)") from None
        last_warmup_args = args
        jax.block_until_ready(step_fn(*args))

    step_idx = 0
    total_recompiles = 0
    total_seconds = 0.0
    first_args: tuple[Any, ...] | None = None
    first_failure_step: int | None = None
    first_failure_args: tuple[Any, ...] | None = None

    for args in iterator:
        step_idx += 1
        if first_args is None:
            first_args = args
        with count_backend_compiles() as cc:
            jax.block_until_ready(step_fn(*args))
        if cc.count > 0:
            total_recompiles += cc.count
            total_seconds += cc.seconds
            if first_failure_step is None:
                first_failure_step = step_idx
                first_failure_args = args

    if total_recompiles != 0:
        diag = ""
        # Determine baseline for comparison: last warmup args, or first executed step args (Debt #3195 Finding C6)
        baseline_args = last_warmup_args if warmup > 0 else first_args
        if baseline_args is not None and first_failure_args is not None:
            if warmup == 0 and first_failure_step == 1:
                diag = (
                    f"\nFunction compiled at step 1 with warmup=0 (expected pre-warmed cache). "
                    f"Initial invocation signature: {extract_jit_signature(first_failure_args)}"
                )
            else:
                diag = "\n" + explain_cache_miss(
                    baseline_args, first_failure_args, step_index=first_failure_step
                )
        elif warmup == 0 and first_failure_step is not None:
            diag = f"\nFunction compiled at step {first_failure_step} with warmup=0 (expected pre-warmed cache)."
        raise AssertionError(
            f"{total_recompiles} backend compile(s) ({total_seconds:.3f}s) "
            f"across {step_idx} step(s) after {warmup} warmup call(s).{diag}"
        )
```

**Warmup=0 Diagnostic Invariant (Debt #3195 Finding C6)**:
When `warmup=0` (e.g. verifying that an AOT-compiled executable or pre-warmed execution cache incurs zero compilation overhead), `last_warmup_args` remains `None`. In this scenario, `assert_no_recompile_after` captures `first_args` from the first executed step. If compilation occurs on the initial invocation (`first_failure_step == 1`), it outputs an explicit diagnostic indicating compilation occurred at step 1 with `warmup=0` alongside the extracted JIT signature. If compilation occurs on subsequent steps, it compares `first_args` against `first_failure_args` via `explain_cache_miss` to identify the triggering signature drift.
```

---

## 5. Verification Controls

To ensure absolute integrity and compliance with Praxia/Teamwork mandates, Track 3 implements strict positive and negative verification controls:

### 5.1 Positive Verification Controls

| Control ID | Description | Exact Verification Target | Expected Result |
|---|---|---|---|
| **POS-3197-01** | Offline CUDA Triton Lowering on CPU | Run `verify_triton_lowering` on standard Pallas matrix-multiply and scale kernels using abstract CUDA A100 mesh on CPU. | Lowering succeeds, `@__gpu$xla.gpu.triton` is verified in HLO text, grid dimensions match, and returns `TritonLoweringResult`. |
| **POS-3197-02** | JL-PALLAS Clean Lint | Run `scripts/lint_pallas.py` against all production modules in `src/xtrax/`. | Zero violations detected; exit code 0. |
| **POS-3196-01** | Custom-Call Auto-Attribution | Parse HLO text containing `@__gpu$xla.gpu.triton` and custom call targets with `scope_map_from_hlo_text`. | Instructions are attributed to specific kernel names rather than unmapped `None`. |
| **POS-3196-02** | Multi-Replica Bucket Record Serialization | Construct `ProbeRecord` with `num_replicas=4`, `replica_id=2`, `bucket_index=1`, `bucket_capacity=256`; serialize to JSON and deserialize via `from_json()`. | All 4 fields preserved exactly, round-trip equality asserted. |
| **POS-3195-01** | Stable JIT Recompile Pass | Run `assert_no_recompile_after` with identical shapes and static args across multiple steps on `@jax.jit` or `@eqx.filter_jit`. | Function succeeds with 0 recompiles, no exception raised. |

### 5.2 Negative Verification Controls

| Control ID | Description | Exact Test Setup / Fault Injection | Expected Failure Mode |
|---|---|---|---|
| **NEG-3197-01** | Non-Triton Lowering Rejection | Pass a plain JAX elementwise function (`lambda x: x + 1.0`) to `verify_triton_lowering`. | Raises `TritonLoweringError` stating that `@__gpu$xla.gpu.triton` was not emitted. |
| **NEG-3197-02** | JL-PALLAS01 Interpret Flag Rejection | Introduce `pl.pallas_call(..., interpret=True)` into a mock production module under `src/xtrax/`. | `scripts/lint_pallas.py` catches violation `JL-PALLAS01` with non-zero exit. |
| **NEG-3197-03** | JL-PALLAS03 Non-Power-of-Two Literal Rejection | Pass AST with `pl.BlockSpec(block_shape=(1, 30))` alongside symbolic expressions `(1, BLOCK_SIZE)`. | Flags `JL-PALLAS03` warning for 30; safely ignores `BLOCK_SIZE` without raising exceptions (Finding M4). |
| **NEG-3196-01** | `vmap` over `pallas_call` Detection | Trace a `lax.scan` whose body calls a `pallas_call` kernel using `loop_bodies` and `extent_scaling_report`. | `has_pallas_call` is `True`, `vmap_induced_pallas_scan` is flagged, diagnostic warning emitted. |
| **NEG-3196-02** | `ProbeRecord` Invariant Enforcement | Construct `ProbeRecord` with invalid replica configurations (`replica_id=4` with `num_replicas=2`, or `num_replicas=0`, or `replica_id=True`). | Raises `ClaimValidityError` specifying the violated replica invariant. |
| **NEG-3195-01** | JIT Recompile Shape Mutation Diff | Run `assert_no_recompile_after` where step 2 mutates input shape from `(8,)` to `(16,)`. | Raises `AssertionError` matching `"backend compile(s)"` AND containing `"[0] 'x': shape changed from (8,) to (16,)"`. |
| **NEG-3195-02** | JIT Recompile Dtype Promotion Diff | Run `assert_no_recompile_after` where step 2 changes input dtype from `float32` to `float64`. | Raises `AssertionError` containing `"dtype changed from float32 to float64"`. |
| **NEG-3195-03** | JIT Recompile Static Parameter Diff | Run `assert_no_recompile_after` where step 2 changes a static parameter (e.g. `mode='train'` to `mode='eval'`). | Raises `AssertionError` containing `"static value changed from 'train' to 'eval'"`. |
| **NEG-3195-04** | JIT Recompile with warmup=0 Diagnostic | Run `assert_no_recompile_after` with `warmup=0` where backend compilation occurs at step 1. | Raises `AssertionError` containing `"Function compiled at step 1 with warmup=0 (expected pre-warmed cache)"` (Finding C6). |

---

## 6. Test-Driven Development (TDD) Implementation Plan

Implementation follows a strict test-first methodology:

### Step 1: Tests First
1. Write `tests/testing/test_triton_lowering.py`:
   - Positive test: `test_verify_triton_lowering_matmul()` using abstract CUDA mesh.
   - Positive test: `test_verify_triton_lowering_grid_and_warps()`.
   - Negative test: `test_verify_triton_lowering_rejects_non_triton_primitive()`.
   - Negative test: `test_verify_triton_lowering_catches_invalid_grid_assertion()`.
2. Write `tests/audit/test_jl_pallas_lint.py`:
   - Positive test: Verify clean code passes `lint_pallas.py`.
   - Negative tests: Mock ASTs triggering `JL-PALLAS01`, `JL-PALLAS02`, `JL-PALLAS03`, `JL-PALLAS04`.
3. Write `tests/profiling/test_loop_scaling_pallas.py`:
   - Positive test: Normal scans without Pallas kernels have `has_pallas_call=False`.
   - Negative test: Scans containing `pallas_call` are flagged with `has_pallas_call=True` and `vmap_induced_pallas_scan=True`.
4. Write `tests/profiling/test_trace_custom_calls.py`:
   - Positive test: HLO text with Triton custom calls attributes kernel name in `scope_map_from_hlo_text`.
5. Write `tests/profiling/test_record_multi_replica.py`:
   - Positive tests: Multi-replica record creation, serialization, deserialization round-trip.
   - Negative tests: Reject invalid `num_replicas`, out-of-range `replica_id`, boolean values, negative bucket indices.
6. Write `tests/profiling/test_recompile_diagnostics.py`:
   - Positive test: `diff_jit_signature` returns empty list on identical arguments.
   - Negative tests: `diff_jit_signature` and `explain_cache_miss` on shape divergence, dtype divergence, PyTree structure mismatch, and static arg divergence.
   - Verification of enhanced `assert_no_recompile_after` error message.

### Step 2: Implementation of Source Modules
1. Implement `src/xtrax/testing/triton_lowering.py` and export in `src/xtrax/testing/__init__.py`.
2. Implement `scripts/lint_pallas.py` and connect to `src/xtrax/jaxlint_runner.py`.
3. Enhance `src/xtrax/profiling/loop_scaling.py` with `_contains_pallas_call` and `vmap_induced_pallas_scan`.
4. Enhance `src/xtrax/profiling/trace.py` with custom call attribution.
5. Enhance `src/xtrax/profiling/record.py` with multi-replica fields, validators, and serialization.
6. Enhance `src/xtrax/profiling/compile_count.py` with signature extraction, diffing, and enriched error messages.

### Step 3: Verification & Static Checks
Run the exact test commands:
```bash
# 1. Verification of Track 3 test suites
uv run pytest tests/testing/test_triton_lowering.py tests/testing/test_parity.py -v --no-cov
uv run pytest tests/audit/test_jl_pallas_lint.py -v --no-cov
uv run pytest tests/profiling/test_compile_count.py tests/profiling/test_recompile_diagnostics.py -v --no-cov
uv run pytest tests/profiling/test_loop_scaling.py tests/profiling/test_loop_scaling_pallas.py -v --no-cov
uv run pytest tests/profiling/test_trace_parse.py tests/profiling/test_trace_custom_calls.py -v --no-cov
uv run pytest tests/profiling/test_bench_records.py tests/profiling/test_record_multi_replica.py -v --no-cov

# 2. Entire xtrax test suite verification
uv run pytest -v --no-cov

# 3. Static linting & code formatting
uv run ruff check .
uv run ruff format --check .
uv run python3 scripts/lint_pallas.py src/xtrax
```

---

## 7. Migration, Compatibility & Integrity Assessment

1. **Backwards Compatibility**:
   - `assert_no_recompile_after`: Retains exact signature `(step_fn, args_iter, warmup=1)`. Existing callers continue to work without modification; error diagnostics are strictly additive.
   - `ProbeRecord`: New fields have defaults (`num_replicas=1`, `replica_id=None`, `bucket_index=None`, `bucket_capacity=None`). Deserialization of prior records remains valid.
   - `verify_triton_lowering`: Purely additive export under `xtrax.testing`.
   - `loop_scaling`: Retains existing report fields; additions are non-breaking attributes and properties.
2. **Integrity Mandate Compliance**:
   - Zero hardcoding of test outputs or expected MLIR strings. All lowering checks execute the real JAX/Pallas abstract compiler pipeline.
   - Diagnostics in `compile_count` genuinely compute and diff live argument PyTrees.
   - Record validation enforces genuine dataclass invariants, rejecting fabrications or boolean coercions.
