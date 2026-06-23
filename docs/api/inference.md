# Signature Inference

Signature inference provides automatic detection and validation of axis signatures for batched JAX computations. It enables zero-config operation with fail-loud semantics, helping you catch ambiguities early rather than silently propagating incorrect assumptions into the tiling and execution layers.

## Overview

The `xtrax.inference` module answers these questions about a JAX function:

1. **What is the output structure?** Extracts shape and dtype information via `jax.eval_shape`.
2. **What are the input axes?** Infers leading dimensions from abstract inputs.
3. **Which axes are resolved?** Determines whether each axis has an explicit role (KNOWN) or requires annotation (UNKNOWN).
4. **Is the traced structure valid?** Optionally verifies that abstract-traced outputs match concrete batch results.

## Core Concepts

### The Fail-Loud Model

In the MVP, axes follow a **fail-loud** strategy:

- **UNKNOWN axes** (no explicit override) trigger `AmbiguousAxisError` at `BatchPlanner.plan()` time.
- **KNOWN axes** (provided via `@axis_config` decorator) proceed normally through the planner.

This prevents silent misinterpretations of ambiguous dimensions. You explicitly opt in to resolution, not out of failure.

### AxisRole: KNOWN vs UNKNOWN

```python
from xtrax.inference import AxisRole

# AxisRole is an Enum with two values in the MVP:
AxisRole.KNOWN    # Axis is resolved, planner proceeds
AxisRole.UNKNOWN  # Axis is ambiguous, planner raises AmbiguousAxisError
```

Future tiers (T2+) will extend this with concrete role names (e.g., `BATCH`, `SEQUENCE`, `FEATURE`), each with domain-specific semantics.

### The @axis_config Decorator

The `@axis_config` decorator is Tier-1 override mechanism. It attaches metadata to a function without changing its behavior:

```python
from xtrax.inference import AxisOverride, axis_config

@axis_config(
    AxisOverride(name="batch", default_batch_size=32),
    AxisOverride(name="sequence", default_batch_size=128),
)
def my_model(x, y):
    """
    x: (batch_size, seq_len, d_model) → output
    y: (batch_size, d_model) → output
    """
    ...
```

The decorator stores overrides positionally: override index *i* applies to the *i*-th qualifying axis (ndim >= 1) in the input pytree, in tree-leaf order.

**Key constraint (Assumption A3):** `default_batch_size` is **required** and **not inferable from shape alone**. You must specify it explicitly.

### BundleSchema: Output Structure

`BundleSchema` captures the output structure and shape information:

```python
from xtrax.inference import BundleSchema
from jax import ShapeDtypeStruct

# Returned by infer_bundle():
schema: BundleSchema = ...

# Access field information:
schema.fields  # dict[str, ShapeDtypeStruct]
schema.fields["logits"]  # ShapeDtypeStruct with .shape and .dtype

# Optional carry specifications (deferred for future tiers):
schema.carry_specs  # None in MVP
```

Fields are named via introspection:
- Dataclass/Equinox module fields: field names (e.g., `logits`, `embeddings`)
- Dict keys: string representations of keys
- Sequence indices: positional fallback (e.g., `out_0`, `out_1`)

## Example 1: Zero-Config Inference (Fail-Loud)

When no `@axis_config` is provided, all axes are UNKNOWN:

```python
import jax
import jax.numpy as jnp
from xtrax.inference import infer_bundle

def my_fn(x):
    """Process a batch of sequences."""
    return jnp.sum(x, axis=-1)

# Abstract inputs: one ShapeDtypeStruct per argument
abstract_x = jax.ShapeDtypeStruct((None, 10, 256), jnp.float32)

schema, axes = infer_bundle(my_fn, [abstract_x])

# schema.fields contains output structure:
print(schema.fields["out_0"].shape)  # TBD (output shape depends on computation)

# axes contains one AxisSpec for the leading (batch) dimension:
print(axes[0].name)      # "axis_0" (positional default)
print(axes[0].role)      # AxisRole.UNKNOWN
print(axes[0].cardinality)  # 10 (inferred from abstract_x.shape[0])

# Later, when planning:
from xtrax.tiling import BatchPlanner

planner = BatchPlanner()
try:
    plan = planner.plan(axes)
except AmbiguousAxisError as e:
    print(f"Ambiguous axis: {e}")
    # Resolve with @axis_config decorator
```

## Example 2: Resolved Axes with @axis_config

Using the decorator to mark axes as KNOWN:

```python
import jax
import jax.numpy as jnp
from xtrax.inference import (
    AxisOverride,
    axis_config,
    infer_bundle,
)

@axis_config(
    AxisOverride(name="batch", default_batch_size=32, cardinality=1024),
)
def my_fn(x):
    """Process a batch of sequences."""
    return jnp.sum(x, axis=-1)

abstract_x = jax.ShapeDtypeStruct((1024, 10, 256), jnp.float32)

schema, axes = infer_bundle(my_fn, [abstract_x])

# axes now contains one KNOWN AxisSpec:
print(axes[0].name)      # "batch" (from override)
print(axes[0].role)      # AxisRole.KNOWN
print(axes[0].default_batch_size)  # 32 (from override, required!)

# Planning now succeeds:
from xtrax.tiling import BatchPlanner

planner = BatchPlanner()
plan = planner.plan(axes)  # No error
print(plan.decisions[0].strategy)  # e.g., "SafeMap"
```

## Worked Examples (Detailed)

### Example 2a: Multi-Input Model with Mixed Overrides

```python
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from xtrax.inference import AxisOverride, axis_config, infer_bundle

@dataclass
class ModelOutput:
    logits: jax.Array
    embeddings: jax.Array

@axis_config(
    AxisOverride(name="batch", default_batch_size=32),
    # No override for the second axis (sequence) -> will be UNKNOWN
)
def encoder_model(x, positions):
    """
    x: (batch, seq_len, d_model)
    positions: (seq_len,)  # No leading batch dim, skipped
    -> ModelOutput with (batch, d_model) logits and embeddings
    """
    # Process x with position embeddings
    logits = jnp.sum(x, axis=1)  # (batch, d_model)
    embeddings = logits * 2
    return ModelOutput(logits=logits, embeddings=embeddings)

# Define abstract inputs
abstract_x = jax.ShapeDtypeStruct((None, 512, 768), jnp.float32)
abstract_positions = jax.ShapeDtypeStruct((512,), jnp.int32)

schema, axes = infer_bundle(encoder_model, [abstract_x, abstract_positions])

# schema.fields contains:
print(schema.fields.keys())  # {"logits", "embeddings"}
print(schema.fields["logits"].shape)   # (None, 768)
print(schema.fields["embeddings"].shape)  # (None, 768)

# axes contains TWO AxisSpec objects:
print(len(axes))  # 2 (positions is 1-D, skipped; two axes from abstract_x)
print(axes[0].name)  # "batch" (KNOWN, from override)
print(axes[0].role)  # AxisRole.KNOWN
print(axes[1].name)  # "axis_1" (UNKNOWN, no override)
print(axes[1].role)  # AxisRole.UNKNOWN

# Planning will fail on axes[1]:
from xtrax.tiling import BatchPlanner
from xtrax.inference import AmbiguousAxisError

planner = BatchPlanner()
try:
    plan = planner.plan(axes)
except AmbiguousAxisError as e:
    print(f"Must resolve: {axes[1].name}")
    # Add another override or accept the error
```

### Example 2b: Structure Verification with verify_against

```python
from xtrax.inference import infer_bundle, StructureMismatchError
import jax
import jax.numpy as jnp

def conditional_fn(x):
    """A function with control flow."""
    if x.shape[0] > 10:
        return jnp.sum(x)  # Scalar output
    else:
        return x  # Array output

# Abstract input
abstract_x = jax.ShapeDtypeStruct((20, 5), jnp.float32)

# Concrete input that triggers the other branch
concrete_x = jnp.ones((5, 5), dtype=jnp.float32)

try:
    schema, axes = infer_bundle(
        conditional_fn,
        [abstract_x],
        verify_against=[concrete_x],
    )
except StructureMismatchError as e:
    print(f"Structure mismatch: abstract traced differently than concrete!")
    print(f"Error: {e}")
```

## API Reference

```{automodule} xtrax.inference
:members:
:undoc-members:
:show-inheritance:
```

### Key Types

**`infer_bundle(fn, abstract_inputs, *, verify_against=None) -> (BundleSchema, list[AxisSpec])`**

Main entrypoint. Infers output schema and axis specifications from a JAX function and abstract inputs.

- **fn**: A pure, traceable JAX function
- **abstract_inputs**: Sequence of `ShapeDtypeStruct` or `(shape, dtype)` tuples
- **verify_against**: Optional concrete inputs for structure validation
- **Returns**: Tuple of `(BundleSchema, list[AxisSpec])` where each AxisSpec has an explicit role (KNOWN or UNKNOWN)
- **Raises**: `StructureMismatchError` if `verify_against` outputs diverge from abstract-traced outputs

**`@axis_config(*AxisOverride(...)) -> Callable`**

Decorator factory for Tier-1 axis resolution. Attaches overrides to a function's `__xtrax_axis_config__` attribute. Overrides are applied positionally to leading axes (ndim >= 1) in tree-leaf order.

**`AxisOverride`** (dataclass)

Configuration for a single axis:
- **name** (required): Human-readable axis name (e.g., "batch", "sequence")
- **default_batch_size** (required): Default batch size; **not inferable from shape, must be explicit**
- **cardinality**: Override cardinality; if None, inferred from leading dimension
- **tile_granularity**: Alignment granularity (default 1)
- **heterogeneous**: Whether elements have variable shapes (default False)
- **dedup_eligible**: Eligible for deduplication (default False)
- **bucket_boundaries**: Sorted bucket sizes for length-based padding (optional)

**`BundleSchema`** (dataclass)

Output structure from signature inference:
- **fields**: `dict[str, ShapeDtypeStruct]` mapping field names to shape/dtype specs
- **carry_specs**: Optional list of carry specifications (deferred for T2+, always None in MVP)

**`AxisRole`** (Enum)

Sentinel values for axis resolution status:
- **KNOWN**: Axis is resolved; planner proceeds
- **UNKNOWN**: Axis is ambiguous; planner raises `AmbiguousAxisError`

**`AmbiguousAxisError`** (Exception)

Raised by `BatchPlanner.plan()` when it encounters an axis with `role == AxisRole.UNKNOWN`. Indicates that an axis could not be automatically resolved and requires explicit annotation via `@axis_config`.

**`StructureMismatchError`** (Exception)

Raised when `verify_against` is provided and the abstract-traced output structure differs from the concrete output. Indicates control flow or branching that produces different tree structures depending on runtime values.

**`synthesize_axes(abstract_inputs, overrides=None) -> list[AxisSpec]`**

Lower-level function that synthesizes `AxisSpec` objects from abstract inputs. Called internally by `infer_bundle()`. Returns a list of AxisSpec with explicit role assignments:
- Axes with overrides: role = KNOWN
- Axes without overrides: role = UNKNOWN

## Deferred and TBD Features

### Tier-2 (Future): Concrete Axis Roles

The `AxisRole` enum will be extended with domain-specific values (e.g., `BATCH`, `SEQUENCE`, `FEATURE`) and associated semantics. Each concrete role will carry planner behavior and validation rules.

### Tier-2 (Future): jaxtyping Dimension-Name Adapter

Support for inferring axes from jaxtyping dimension names (e.g., `Float32["batch seq d_model"]`). This enables roles to be resolved without explicit decorators when functions use jaxtyping annotations.

### Tier-2 (Future): CarrySpec Auto-Derivation

Automatic detection of axes suitable for `jax.lax.scan` carry semantics (e.g., stateful RNN axes). Currently deferred to explicit `CarrySpec` in the tiling layer.

### Tier-2 (Future): LibCST Bundle Codegen

Automated generation of boilerplate `@axis_config` decorators and bundle dataclass definitions from concrete function signatures. Will reduce manual annotation burden for large model families.

## When To Use Signature Inference

Use `infer_bundle()` and `@axis_config` when:

- You have JAX functions (models, loss functions, custom training steps) whose axis structure must be determined before tracing
- You want to catch axis ambiguities early (at decoration time) rather than silently propagating wrong assumptions
- You plan to use `BatchPlanner` to automatically select tiling strategies

You do **not** need signature inference if:

- Your model always uses fixed, pre-specified batch sizes
- You manually manage axis strategies via `AxisSpec` and `BatchDecision`
- You use only single-axis computations (no multi-axis tiling)
