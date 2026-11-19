> Part of the `using-xtrax` skill (`agent_assets/skills/using-xtrax/SKILL.md`) — TIER-2 deep reference.

# Categorical Layer (`xtrax.categorical`: exact co-occurrence counts)

A generic primitive for exact integer weighted contingency tables over compact categorical
codes. It holds no domain content: no entropy, no mutual information. Those finishes belong
to the caller. Spec: `.praxia/docs/specs/261015_exact-categorical-cooccurrence.md`.

#### Public API

```python
from xtrax.categorical import (  # verify: src/xtrax/categorical/__init__.py
    CategoricalCodes,
    CooccurrenceCounts,
    code_dtype,
    cooccurrence_counts,
    encode_codes,
    iter_cooccurrence_row_blocks,
    weight_dtype,
)
from xtrax.categorical.plan import plan_cooccurrence, select_method  # verify: src/xtrax/categorical/plan.py
```

- `encode_codes(codes, n_categories)` casts to the narrowest `code_dtype` and raises `ValueError`
  on concrete out-of-range codes. It never relabels. (`src/xtrax/categorical/codes.py:96`)
- `cooccurrence_counts(codes_a, codes_b=None, *, weights, shifts_a, shifts_b, pairs, strategy,
  differentiable, method, byte_budget, plan, check, mesh)` returns a `CooccurrenceCounts` with
  `joint`, `marginal_a`, `marginal_b`, `total`, plus static `layout`, `method` and `method_rule`.
  (`src/xtrax/categorical/counts.py:699`)
- `iter_cooccurrence_row_blocks(codes_a, codes_b=None, *, row_block, ...)` is a host generator
  that yields `(row slice, device_get'd counts)`. Use it when the dense `(Pa, Pb, Ka, Kb)` grid
  does not fit in memory. (`src/xtrax/categorical/counts.py:942`)

#### Dtype contract

- `differentiable=False`: all four outputs are **int32** and exact for `total < 2**31`. No
  float touches the counts. Totals above `2**24` stay exact.
- `differentiable=True`: all four outputs are **float32**, covered by a `custom_vjp` over float32
  weights. Exact only while `total < 2**24`, which `check=True` asserts. Reverse mode only; forward
  mode raises `TypeError`.
- `strategy="float"`: float32 accumulation. Exact for hard codes with integer weights below
  `2**24`. Soft assignments are approximate.

#### Method selection (`select_method`, `src/xtrax/categorical/plan.py:68`)

- CPU, `method="auto"`: `matmul` when `Ka*Kb <= KK_CROSSOVER_CPU` (256), else `scatter`.
  `method_rule="cpu_measured"`. The constant comes from bathos run
  `e1f92621-7f8f-41d9-85f2-a16209f3345f`; the earlier provisional 64 was an unregistered spike.
- Pairs layout with `n_pairs < Pa*Pb`: `scatter`, `method_rule="cpu_provisional"`. Not measured.
- GPU: provisional and conservative (`matmul` only for dense layouts with `Ka*Kb <= 16`).
  Re-derive it with the Stage-1b GPU probe before relying on it.
- Both methods are exact. The rule only affects speed and memory. `out.method` records the
  method that actually ran.

#### Shifts and layouts

- Shifts follow `np.roll`: variable `a` in replicate `r` reads `x_a[(t - s[r, a]) mod T]`.
  Any integer shift is valid.
- Self layout (`codes_b is None`): the diagonal block equals `diag(marginal_a)`.
- Pairs layout: pass `pairs` as `(n_pairs, 2)` rows of `(a, b)`.

#### Checks

`check=True` (the default) evaluates code range, non-negative weights, integer-valued float
weights on the differentiable path, total bounds and the pair range on device, with no host
callback, into `CooccurrenceCounts.status` (int32 per replicate, `STATUS_*` bits; 0 means valid).

- Eager call (no traced input): a failure raises `ValueError` before the kernel runs.
- Inside `jit`, `vmap` or `grad`: nothing raises. Return `status` (as an aux output under
  `grad`) and check it, or call `counts.raise_if_invalid()` outside the trace. A replicate
  with a nonzero status has undefined counts.

`check=False` skips the checks (`status` is all 0). Use it only for benchmarks, and only with
inputs already known to be valid.

#### Memory

`byte_budget=None` uses unbudgeted defaults and never queries the device. An int budget runs a
deterministic tile search and raises `BudgetInfeasibleError` when nothing fits. `"device"` reads
device memory and raises if the backend exposes no `memory_stats`. Use an int in that case.
`plan_cooccurrence` returns the tile sizes the kernel will execute.

#### Gotchas

- The import-linter contract forbids `xtrax.categorical` from importing `xtrax.inference`,
  `xtrax.export`, `xtrax.eda` or `xtrax.devtools`. Put domain finishes in the caller.
- Ka and Kb may differ. Ragged per-variable alphabets are not supported.
- Static guards: `Pa*Ka`, `Pb*Kb` and `Ka*Kb` must each stay below `2**31`.
