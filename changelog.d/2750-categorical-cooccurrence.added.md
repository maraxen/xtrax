- **`xtrax.categorical`: exact categorical co-occurrence counts** (Stage 1 of
  spec `261015_exact-categorical-cooccurrence`, debt #2750). `cooccurrence_counts`
  returns exact int32 weighted contingency tables (joint, marginals, total) over
  compact categorical codes, with batched replicates (`(R, T)` weights), circular
  per-variable shifts that follow `np.roll` for any integer shift, a dense, pairs
  or self layout, and frame-tiled `matmul` and `scatter` methods. Totals above
  2**24 stay exact on the integer path. `differentiable=True` adds a `custom_vjp`
  over float32 weights, with reverse mode only. `strategy="float"` is the float
  path. `check=True` evaluates code range, weight sign and integrality, total
  bounds and pair range on device into `CooccurrenceCounts.status` (`STATUS_*`
  bits, no host callback): an eager call raises `ValueError`, and under
  `jit`/`vmap`/`grad` the bits are returned and `raise_if_invalid()` raises outside
  the trace. `plan_cooccurrence` returns the tile plan the kernel executes; a
  scatter budget is judged feasible at `pair_tile=1`. `iter_cooccurrence_row_blocks`
  streams row blocks. The package imports only jax, jaxtyping, numpy,
  `xtrax._tree` and `xtrax.tiling.budget` (no equinox); an import-linter contract
  forbids inference, export, eda and devtools imports. Method selection on CPU is
  measured: `auto` picks matmul when `Ka*Kb <= 256` and scatter above it
  (`KK_CROSSOVER_CPU = 256`, from bathos run
  `e1f92621-7f8f-41d9-85f2-a16209f3345f`, which replaced the provisional value 64).
  GPU selection is still provisional.
