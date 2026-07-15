> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Signature Inference (xtrax.inference) — derive AxisSpecs + BundleSchema from a typed function

> **Availability**: shipped in the 0.3.0 release (Tier-1 MVP, E1).  
> Import paths: `from xtrax.inference import ...` (all 8 public symbols re-exported from `__init__`).  
> `AxisRole` lives canonically in `xtrax.tiling.roles` (zero xtrax deps) and is re-exported by `xtrax.inference` for convenience. Verify: `src/xtrax/inference/errors.py:12`, `src/xtrax/tiling/roles.py:14`.

#### `infer_bundle`: The Entrypoint

```python
from xtrax.inference import infer_bundle, BundleSchema, AxisOverride, axis_config
from xtrax.tiling.plan import AxisSpec

schema, axes = infer_bundle(
    fn,                         # pure, traceable JAX function  — verify: src/xtrax/inference/api.py:15
    abstract_inputs,            # Sequence[ShapeDtypeStruct | (shape, dtype)]
    verify_against=None,        # optional Sequence of concrete inputs
)
# returns: tuple[BundleSchema, list[AxisSpec]]  — verify: src/xtrax/inference/api.py:20
```

Exact signature (verify: `src/xtrax/inference/api.py:15-20`):
```python
def infer_bundle(
    fn: Any,
    abstract_inputs: Sequence[Any],
    *,
    verify_against: Sequence[Any] | None = None,
) -> tuple[BundleSchema, list[AxisSpec]]:
```

Internally: calls `jax.eval_shape` (zero FLOPs) to extract the output schema, reads any `@axis_config` sidecar on `fn`, synthesizes one `AxisSpec` per qualifying input leaf (ndim >= 1), and optionally calls `verify_structure`.  
Verify: `src/xtrax/inference/api.py:58-78`

> **CLI cross-link**: `xtrax plan` and `xtrax explain` both call `infer_bundle` internally (load `--fn` import path + parse `--shapes`, then plan). See TIER-2: CLI Layer (E2/E3). `explain` adds `explain_plan` + format emission (`json`/`text`/`html`/`png`). Verify: `src/xtrax/cli/plan.py:31-39`, `src/xtrax/cli/explain.py:52-60`

#### Fail-Loud Model: `AxisRole.KNOWN` vs `AxisRole.UNKNOWN`

Every synthesized `AxisSpec` carries a `role` field (verify: `src/xtrax/tiling/plan.py:50`):

- **`AxisRole.KNOWN`** (default for hand-written specs) — planner proceeds normally.
- **`AxisRole.UNKNOWN`** — sentinel set on every axis that `infer_bundle` cannot resolve (i.e., bare functions with no `@axis_config`).

```python
# AxisRole is an enum with two MVP members:
class AxisRole(enum.Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
# verify: src/xtrax/tiling/roles.py:14-26
```

🚫 HALTS: `BatchPlanner.plan()` raises `AmbiguousAxisError` for any `AxisSpec` with `role == AxisRole.UNKNOWN`.  
Enforcement: `src/xtrax/tiling/plan.py:259-266`  
Message: `"axis '<name>' has an unresolved role; declare it with @axis_config or provide an override before planning."`

This is intentional: `infer_bundle` on a zero-config function produces UNKNOWN axes, and the planner never silently proceeds with ambiguous axes. You must resolve every axis before planning.

#### `@axis_config`: Tier-1 Override (Resolve UNKNOWN → KNOWN)

```python
from xtrax.inference import axis_config, AxisOverride

@axis_config(
    AxisOverride(name="batch", default_batch_size=32),   # positional: axis 0
    AxisOverride(name="seq",   default_batch_size=128),  # positional: axis 1
)
def my_fn(x, y):
    ...
# verify: src/xtrax/inference/config.py:44-74
```

`AxisOverride` fields (verify: `src/xtrax/inference/config.py:9-38`):

```python
@dataclass(frozen=True)
class AxisOverride:
    name: str                              # Required. Human-readable axis name.
    default_batch_size: int                # Required. Batch size (NOT inferrable from shape).
    cardinality: int | None = None         # Override leading-dim cardinality (None = infer).
    tile_granularity: int = 1              # Alignment granularity (default 1).
    heterogeneous: bool = False            # Variable-shape elements?
    dedup_eligible: bool = False           # Eligible for deduplication?
    bucket_boundaries: tuple[int,...] | None = None  # Variable-length bucketing?
```

`@axis_config` stores overrides as `__xtrax_axis_config__` on the decorated function (zero call-path overhead) and returns the function unchanged. Resolving an axis via override sets its `AxisSpec.role` to `KNOWN`. `default_batch_size` is REQUIRED because batch size is never inferrable from shape alone (Assumption A3 of the inference design). Verify: `src/xtrax/inference/config.py:18-19`

#### `BundleSchema`: Output Schema

```python
from xtrax.inference import BundleSchema
# verify: src/xtrax/inference/schema.py:12-27

@dataclass
class BundleSchema:
    fields: dict[str, ShapeDtypeStruct]  # leaf name -> ShapeDtypeStruct (from eval_shape output)
    carry_specs: list[Any] | None = None  # passthrough seam; always None in MVP
```

Field names are recovered from the eval_shape output pytree's key_path:
- `GetAttrKey.name` — dataclass / eqx.Module field names
- `DictKey.key` — dict keys
- Positional fallback `out_{i}` for SequenceKey or bare leaves

Verify: `src/xtrax/inference/schema.py:30-93`

#### `verify_structure` / `verify_against=`: Purity Guard

`verify_structure` runs `fn` concretely on `concrete_inputs`, compares the pytree structure, leaf shapes, and dtypes against the abstract eval_shape output, and raises `StructureMismatchError` on any divergence.

```python
from xtrax.inference.verify import verify_structure  # verify: src/xtrax/inference/verify.py:19-96
from xtrax.inference import StructureMismatchError

# Via infer_bundle:
schema, axes = infer_bundle(fn, abstract_inputs, verify_against=concrete_inputs)

# Direct call:
verify_structure(fn, abstract_inputs, concrete_inputs)  # returns None or raises
```

`StructureMismatchError` is raised when `jax.eval_shape`'s abstract output structure diverges from actual execution — e.g., due to data-dependent control flow.  
Verify: `src/xtrax/inference/errors.py:28-41`

#### jaxtyping Note

jaxtyping is **optional**: the inference layer never hard-imports it. The Tier-2 jaxtyping dim-name role adapter (which would map annotated dim names to concrete `AxisRole` values) is deferred and not part of the MVP.

#### Minimal Working Example

```python
import jax
from jax import ShapeDtypeStruct
import numpy as np
from xtrax.inference import infer_bundle, axis_config, AxisOverride, AxisRole
from xtrax.tiling.plan import BatchPlanner

def encode(x, y):
    """Two-input function: (batch, feat), (batch, feat) -> (batch, feat)."""
    return x + y

# --- Zero-config: UNKNOWN axes, planner will fail loud ---
abstract = [ShapeDtypeStruct((32, 128), np.float32),
            ShapeDtypeStruct((32, 128), np.float32)]
schema, axes = infer_bundle(encode, abstract)
print(schema.fields)        # {"out_0": ShapeDtypeStruct(shape=(32, 128), dtype=float32)}
print(axes[0].role)         # AxisRole.UNKNOWN — not annotated, cannot plan yet

planner = BatchPlanner()
# planner.plan(axes)        # 🚫 HALTS: AmbiguousAxisError — axis 'axis_0' has unresolved role

# --- With @axis_config: KNOWN axes, planner proceeds ---
@axis_config(
    AxisOverride(name="batch", default_batch_size=32),
    AxisOverride(name="batch", default_batch_size=32),  # one override per input
)
def encode_annotated(x, y):
    return x + y

schema2, axes2 = infer_bundle(encode_annotated, abstract)
print(axes2[0].role)        # AxisRole.KNOWN
print(axes2[0].name)        # "batch"

plan = planner.plan(axes2)  # succeeds — all axes KNOWN
print(plan.decisions[0].reasoning)  # "cardinality <= batch_size → Vmap"
```

#### Deferred (Not in MVP)

- **Tier-2 jaxtyping dim-name adapter**: maps annotated dimension names to concrete `AxisRole` values; would eliminate `@axis_config` for jaxtyping-annotated functions.
- **libcst Bundle codegen**: generate `BundleSchema` as typed Python source from inferred schema.
- **CarrySpec auto-derivation**: automatically derive `CarrySpec` from function return type annotations.
