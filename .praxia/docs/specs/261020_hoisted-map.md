---
title: 'hoisted_map: run a shared stage once per distinct input, map the per-element stage'
description: 'Explicit shared/per-element split for debt #2600; structural varying-input check, optional parity probe, opt-in memoize_jaxpr cache across calls'
status: draft
task_id: 261020_hoist-invariant-subgraphs
date: '261020'
backlog_ids: 'debt #2600'
adversarial_review: 'spec-challenger (sonnet) 261020: NEEDS_WORK, 2 blockers + 4 major addressed in revision'
---
# hoisted_map: run a shared stage once, map the per-element stage

## Purpose

Composed pipelines recompute shared work (structure encoding) once per element of a
mapped axis, and sometimes once per Python-level call on the same input. Debt #2600
asks for execution that runs that work once per distinct input.

## What exists (tags: V = verified at file:line or by probe, U = unverified)

- V `chunked_map(fn, xs, batch_size)` — `src/xtrax/transforms/map.py:72`. vmap,
  `lax.map(batch_size=)`, size-1 guard (#2520).
- V `memoize_jaxpr(fn, *, policy=MemoPolicy())` — `src/xtrax/inference/memo.py:1173`.
  Content-keyed; typed PRNG keys digest `key_data` (`memo.py:1-30`); purity screen per
  call signature; stats via `.memo_get_stats()`.
- V `verify_axis_invariance(fn, example_inputs, specs, claimed_invariant_outputs)` —
  `src/xtrax/inference/invariance.py:62`; raises `InputInvarianceError`.
- V `AxisSpec.varying_inputs`, `declare_varying_inputs` — `src/xtrax/tiling/plan.py:79,151`.
- V Import contract "tiling must not import inference" (`pyproject.toml`), and
  `xtrax.tiling` imports `xtrax.transforms` (`tiling/dispatch.py:18`). A transforms module
  that imports `xtrax.inference` therefore violates the contract transitively. **The
  primitive lives in `xtrax.inference.hoist`**, re-exported from `xtrax.inference`.
- V (probe 261020, CPU, jax 0.11.1, throwaway): XLA already hoists a truly invariant
  subgraph out of `lax.map` (chunked and unchunked, including a size-inflating
  `(L,3)->(L,L,C)` encoder): naive vs hand-hoisted within noise, one `sine` op in the
  optimized HLO either way. U on GPU and on IREE/StableHLO export.
- V The real within-call waste is **false dependence**: the per-element PRNG key flows
  into feature extraction, so the encoder genuinely reads a varying input in the jaxpr
  even when `backbone_noise=0` makes the value key-independent.
  tev_design `scripts/generate_tev_conditionals.py:378-392` (`jit(vmap(one))` over keys,
  `one` calls aminx `sample(key, model, coords, ...)`); tev_design
  `scripts/run_design_grid.py:578-691` hand-hoists the encoder behind
  `assert backbone_noise == 0.0`, citing "~24 GB encoder-context replication B times".
  No compiler pass can remove this; only the caller can, by not passing the key.
- V Cross-call waste: aminx `src/aminx/host/runner.py:1655-1698` rebuilds the bundle and
  re-encodes the same structure for every inspection feature.

## API (revised after adversarial review, 261020)

```python
# xtrax/inference/hoist.py, re-exported from xtrax.inference
class HoistedMap:            # returned by make_hoisted
    def __call__(self, shared_args: Mapping[str, Any], xs: X, *,
                 reference: Callable[[Mapping[str, Any], X], Y] | None = None,
                 check_elements: int = 0, rtol: float = 1e-5, atol: float = 1e-8) -> Y: ...
    def stats(self) -> dict[str, int]:   # {"shared_calls", "traced_bypass", + memo stats if cached}

def make_hoisted(shared: Callable[..., S], per_element: Callable[[S, X], Y], *,
                 batch_size: int | None = None, axis: AxisSpec | None = None,
                 cache: MemoPolicy | None = None) -> HoistedMap: ...

def hoisted_map(shared, per_element, shared_args, xs, *, batch_size=None, axis=None,
                reference=None, check_elements=0, rtol=1e-5, atol=1e-8) -> Y:
    """One-shot: make_hoisted(...)(...). Repeated calls should hold a HoistedMap."""
```

Execution (per `HoistedMap.__call__`):

1. Validate: `xs` must have at least one leaf and leading dim N >= 1, else `ValueError`
   (avoids `chunked_map`'s IndexError on an empty pytree, `transforms/map.py:96`).
2. **Structural check** (when `axis` given): any key of `shared_args` that appears in
   `axis.varying_inputs` raises `HoistVaryingInputError(axis.name, name)` (new, subclass of
   `XtraxInferenceError` in `inference/errors.py`). Name check only. No
   `verify_axis_invariance` call: `shared` never receives `xs`, so a jaxpr check on it is
   vacuous, and `verify_axis_invariance` raises `ValueError` when a varying name (normally an
   `xs` name) is not one of its example inputs (`invariance.py:109-114`).
3. `s = shared_fn(**shared_args)`, `shared_calls += 1`. `shared_fn` is, owned by the instance:
   - uncached: `jax.jit(shared)`;
   - cached: `memoize_jaxpr(shared, policy=cache)` built once at `make_hoisted` (raises
     `MemoMultiDeviceError` on multi-device hosts, `memo.py:713`: documented, not caught).
     If any `shared_args` leaf is a `jax.core.Tracer` (an enclosing `jit`), the memo key
     cannot be built (`np.asarray` on a tracer, `memo.py:182,192`), so call the uncached
     `jax.jit(shared)` instead and `traced_bypass += 1`. Never raise for this.
4. `y = mapped(s, xs)` where `mapped = jax.jit(lambda s, xs: chunked_map(lambda x:
   per_element(s, x), xs, batch_size))` is built once per instance: `s` is a jit argument,
   not a fresh closure, so repeated calls with the same shapes do not recompile.
5. **Parity probe** (when `check_elements > 0`): `reference` is required (else `ValueError`);
   leaves of `xs` must be concrete (else `ValueError`: the probe needs values). For
   `i in range(min(check_elements, N))`, `x_i = tree_map(lambda l: l[i], xs)`; compare
   `tree_map(lambda l: l[i], y)` with `reference(shared_args, x_i)`: float/complex leaves by
   `allclose(rtol, atol)`, all others exact; structures must match. `xs` carries every
   per-element input, PRNG keys included, so the reference sees what the monolithic
   function saw. Mismatch: `HoistParityError(index, leaf_path, max_abs_diff)`.
6. Return `y` (pytree with leading axis N, `chunked_map`'s output structure).

## Acceptance

Tests in `tests/inference/test_hoist.py` (tier-1 counts it; `tests/audit` would not).

- A1 Runs once per call: N=16 over vmap, `batch_size=4` with N=10 (ragged), and N=1;
  `stats()["shared_calls"] == 1` per call. Cached: 3 calls with equal `shared_args` give
  `memo_get_stats()["misses"] == 1, hits == 2`; a changed arg gives one more miss.
  (Python counters inside `shared` count traces, and `debug_callback` is refused by the
  memo purity screen (`memo.py:478`), so neither is used.)
- A2 Parity: hoisted output `allclose` to `jax.vmap(lambda x: reference(args, x))(xs)` on
  all three paths in A1.
- A3 Key in the cache key: `shared` takes a typed PRNG key in `shared_args`; two calls with
  different keys give `misses == 2` and different outputs. Negative control in the same
  file: monkeypatch `memo._leaf_digest` to skip typed-key leaves and assert the second call
  becomes a hit — proves A3's assertion goes red when the key is dropped.
- A4 `axis=AxisSpec(..., varying_inputs=("key",))` with `"key" in shared_args` raises
  `HoistVaryingInputError` naming `key`; the same with `"key"` only in `xs` passes.
- A5 Parity red: `shared` drops a per-element noise term whose scale is passed as an array
  (`noise=jnp.float32(0.1)`, not a Python float, so it cannot fold); `check_elements=2`
  raises `HoistParityError`. With `noise=0.0` the probe passes.
- A6 False dependence is real and removed: a per-element key reaches an encoder
  `(L,3)->(L,L,C)`. In `jax.make_jaxpr` of the monolithic `vmap`, the encoder's output
  var has shape `(N,L,L,C)`; in the hoisted version it has shape `(L,L,C)`. Control: a
  truly invariant encoder (no key) has shape `(L,L,C)` in both, so the test distinguishes
  false dependence from real invariance and cannot pass vacuously.
- A7 Under an enclosing `jax.jit`: uncached and cached instances both return correct
  values; cached reports `traced_bypass == 1`.
- A8 Errors: N=0 / empty `xs` → `ValueError`; `check_elements>0` without `reference` →
  `ValueError`; probe on traced `xs` → `ValueError`.
- Exports added to `xtrax.inference.__all__`; import-linter contracts pass (no cycle:
  `transforms` imports neither `inference` nor `tiling`); `just audit-structure-complexity-gate`
  stays at or under baseline; ruff + `ruff format --check` + ty clean; CHANGELOG
  `[Unreleased]` "Added" entry.
- Docs: new section "Hoist shared work out of a mapped axis" in
  `agent_assets/skills/xtrax-optimizing/SKILL.md`: false dependence, the API, the CPU probe
  result (XLA already hoists truly invariant work on CPU; GPU/IREE unverified). Regenerate
  `src/xtrax/agent_plugin.json` with `cisternal assets snapshot` (checked by
  `tests/cli/test_plugin_snapshot.py`).

## Out of scope

Consumer migrations (tev run_design_grid, aminx runner, aminx sample): separate PRs.
Automatic jaxpr partitioning. Stages/executor integration. GPU/IREE re-probe.

## Risks

| Risk | Mitigation |
|---|---|
| Caller moves a varying input into `shared` | A4 name check + A5 parity probe |
| Per-call cache gives zero reuse | only `make_hoisted` owns a cache |
| Large `s` held by the cache (tev cites ~24 GB) | opt-in; `MemoPolicy.max_entries`; doc the cost |
| Counter test passes vacuously | A3 negative control; A6 invariant control |
| Fresh closures recompile every call | `s` passed as a jit argument; jit built once per instance |
