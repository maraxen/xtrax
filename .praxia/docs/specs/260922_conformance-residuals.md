---
title: "Conformance residuals of epic #5169 (memo donation, verify_dedup_spec, exact synthesizer row identity) + added-types-diff nested-def fix"
description: "Two items the converged 260825 spec requires but PR #104 did not ship, the synthesizer row-identity unsoundness found while specifying them, and the verified CI-gate bug #5205"
task_id: 260922_conformance-residuals
status: FINAL r4 — adversarially converged, challenger round 3 verdict ACCEPT (7 MINOR conditions, binding in §14)
revision: 4 (260922; round-3 ACCEPT conditions folded in as §14; evidence moved into the repo)
---

# Conformance residuals: memo donation, `verify_dedup_spec`, synthesizer row identity, added-types-diff nested defs

## Revision history

| Rev | Date | Change |
|---|---|---|
| r1 | 260922 | Initial draft, written without a shell; P1-P3 framed as probes still to run. |
| r2 | 260922 | Challenger returned REVISE (1 BLOCKER, 6 MAJOR, 7 MINOR); all dispositions in §11.<br>Measured probes P1-P7 replace the "probe first" framing (§1.1).<br>**T4 added** (orchestrator decision D1): synthesizer row identity is now exact per leaf.<br>T1 now screens both donation carriers (D3). Its fail-open paths are stated and pinned with xfail tests (D4).<br>An amendment to 260825 is recorded here (D5, §9). One PR, fixed task order (D6, §8). |
| r3 | 260922 | Challenger round 2 returned REVISE (2 MAJOR, 7 MINOR); dispositions in §13. Probes P8-P12 added (§1.1).<br>**Transfer oracle rebuilt as structural** (§7.1): the `_value` spy is deleted; in its place, the `_to_host` spy checks argument types, device-only helpers are pinned by an AST allowlist, and module-wide `np.*` call sites are pinned.<br>**Sub-byte dtypes** (§5.1): integer sub-byte dtypes are widened before bitcast; sub-byte floats are refused.<br>**Zero-width leaves** need no special case (P12, AC-44).<br>**Test fixtures** hardened: ×50 repetition, explicit threshold, and a stage assertion before any `k` check.<br>**Strict-xfail tests** now pin `raises=pytest.fail.Exception`, verified against the installed pytest 9.1.1.<br>**Fixers** run sequentially (T3 → T1 → T2+T4) in the one sprint worktree, with a per-commit ownership check (§8). |
| r4 | 260922 | Challenger round 3 returned **ACCEPT** with 7 MINOR conditions. They are recorded verbatim as §14 and are binding on the T2+T4 fixer (conditions 1-6) and the orchestrator (condition 7, done).<br>Evidence moved out of a temporary job directory into the repo: all probes are consolidated in the tracked `scripts/measure_conformance_residuals_260922.py`, and its raw output is in `.praxia/docs/research/260922_conformance-residuals-probes.md`.<br>T3 landed first as `3c5f315`. The orchestrator independently confirmed its red-before-green: 7 tests fail against the old collector, and all 18 pass with the fix. |

## 1. Context

Epic #5169 (xtrax runtime compute-reuse layer) shipped in PR #104 (squash `abcce47`,
released `0.4.0a7`). It implements `.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md`
(FINAL v4.1, ACCEPT@0.85). That spec was committed in PR #101, 26 minutes before #104 merged,
so #104 was built against it. A conformance check finds two required items missing, and PR
#104's body records no deferral of either:

1. **§4.2 item 6: "Donation, both directions".** Neither half conforms (§3.1).
   `tests/inference/test_memo.py:244-255` (`test_donation_rejected_at_wrap`) is **vacuous**: it
   asserts `core.policy.block_on_miss is True` and never exercises donation.
2. **§4.3 `verify_dedup_spec(spec, leaves)`**, which appendix §10.2/§10.3 also require. It does
   not exist anywhere in the codebase.

While specifying (2), the probes showed that the shipped synthesizer's row identity is
**unsound** (P2-P4, plus an axis bug). A verifier built on the same row definition would share
the defect. That fix is T4 and is in scope (D1).

Separately, **backlog #5205** is a verified bug in the added-types-diff CI gate. It fired for
real on PR #156: a closure named `visit` inside `_topological_order`.

### 1.1 Measured evidence

Source: `.praxia/docs/research/260922_conformance-residuals-probes.md`, the raw output of one
run of the tracked script `scripts/measure_conformance_residuals_260922.py`, which reproduces
every row below (P0-P12). Re-run it after a JAX upgrade. Environment: jax 0.11.1, numpy 2.5.1,
CPU, x64 off.

| Probe | Result | Used by |
|---|---|---|
| Donation visibility | `jit` eqn `donated_invars=(True, False)` for `donate_argnums=0`. `(False, True)` for `donate_argnames='y'`. A nested donating `jit` is visible. The jitted callable exposes no donate attribute. | T1 |
| **P1** CPU honours donation | **YES**: `x.is_deleted()` after `jit(..., donate_argnums=0)(x)`. | T1 ACs use `is_deleted` as a real discriminator. |
| **P1b** nested donation invalidates the caller's buffer under eager `fn` | **YES**: `outer(y)` → `y.is_deleted()`. | T1 case (c) is live. |
| **P2** `np.unique(axis=0)` merges `-0.0`/`+0.0` | **YES**. Identical-NaN rows stay 2 uniques. | T4 |
| **P3** mixed-dtype `jnp.concatenate` collapses distinct ints | **YES**: int32 `2**24` and `2**24+1` with a float32 leaf compare equal. The collapse happens in shipped `_stack_batch_leaves` (`dedup_synthesis.py:348`). | T4, AC-28 |
| **P4** `jnp.moveaxis` on a numpy int64 leaf, x64 off | **Truncates to int32** (wraps mod 2**32). float64 rounds to float32. Site: `dedup_synthesis.py:334`. | T4 |
| **P5** `device_put(x, donate=True)` | `device_put` eqn with `copy_semantics=(ArrayCopySemantics.DONATE_INPUT,)`, which is **not** in `donated_invars`. | T1 (second carrier) |
| **P5b** eager `device_put(donate=True)` on CPU | Caller buffer **not** deleted. Latent on CPU, possibly live on accelerators. | T1 AC-6 asserts rejection only. |
| **P6** on-device `lax.bitcast_convert_type(·, uint8)` | complex64 → **TypeError**. bfloat16 → ok, `(3,2)→(3,2,2)`. bool: `astype(uint8)` ok. | T4 byte recipe |
| **P7** `np.asarray(jax.random.split(key(0), 4))` | **TypeError**. | T4, AC-23 |
| Axis bug (code reading) | `dedup_synthesis.py:156-157`: the stacked array is batch-first, yet `N = stacked.shape[axis]` reads the feature width when `axis != 0`. The stale `axis` is also passed on. | T4, AC-31 |
| **P8** | The `ArrayImpl._value` getter does **not** observe `np.asarray`, `np.array` or `np.ascontiguousarray`. | §7.1: the `_value` spy is deleted |
| **P9** | The complex real/imag split preserves `±0` in the imaginary part. The split is injective. | T4 |
| **P10** | `jax.transfer_guard` never fires on CPU, so it cannot serve as the oracle. | §7.1 |
| **P11** | int4 `(4,3)` bitcast to uint8 raises **ValueError**. int4 `(4,2)` bitcast **packs** within the row to `(4,)`, which is lossless. `astype(int8)` is exact (`[-8, 7]` round-trips), after which the bitcast leaves `(4,3)` as `(4,3)`. | T4 sub-byte widening, AC-22 |
| **P12** | `np.unique` over uint8 `(5,0)` gives 1 unique row. `(5,0).reshape(5,-1)` stays `(5,0)`. A float32 `(5,0)` bitcast gives `(5,0,4)`, which reshapes to `(5,0)`. Concatenating `(5,0)` with `(5,2)` gives `(5,2)`. | Zero-width leaves need no special case (AC-44) |

Additional evidence, read directly from source:

- **The strict-xfail exception is `pytest.fail.Exception`.** Installed pytest is 9.1.1. An
  unmet `pytest.raises` calls `_pytest.outcomes.fail` (`_pytest/raises.py:22`, `:697-702`),
  which raises `Failed` (`_pytest/outcomes.py:59`); that is `pytest.fail.Exception`
  (`outcomes.py:158`). `xfail(raises=...)` matches by `isinstance` (`_pytest/skipping.py:290-294`).
  `Failed` is not `XFailed` (`outcomes.py:76`: `XFailed` subclasses `Failed`, not the reverse),
  so the imperative-xfail branch at `skipping.py:284` does not intercept it.
- **Sub-byte detection must use `jax.dtypes.itemsize_bits`.** It is public
  (`jax/dtypes.py:20`). `dtype.itemsize` is wrong for sub-byte types: see the comment at
  `jax/_src/dtypes.py:282-285`, "we cannot use dtype.itemsize here because this is incorrect
  for sub-byte integer types". The sub-byte types JAX exposes are:
  - integers: int2, uint2, int4, uint4, and int1/uint1 when available (`_intn_dtypes`,
    `:201-216`);
  - floats: float4_e2m1fn, float6_e2m3fn, float6_e3m2fn (`:121-181`).

## 2. Repo constraints (binding on every task)

- **`from __future__ import annotations`.** Every `src/` file touched here already carries it,
  and no new `src/` file is created. Do not add it anywhere new, and do not remove it (that
  would be a separate migration).
- **No public-named nested functions** in new or changed `src/` code. Until T3 lands, the gate
  fails on them. All new helpers are module-level `_private` functions.
- **Fail-loud typed errors.** New error classes are exported from their module's `__all__`.
  For `xtrax.inference` they are also exported from `xtrax/inference/__init__.py`.
- **Import-linter:** `xtrax.tiling` must not import `xtrax.inference` (`errors.py:76-79`).
- **Docs.** `docs/api/inference.md` is hand-written and has no `memoize_jaxpr` section.
  `docs/api/tiling.md` is `automodule xtrax.tiling`, which renders only `xtrax.tiling.__all__`;
  `dedup_synthesis` is not in that list.
- **Narrow runs only** (§8). Never run a whole suite on this machine.

## 3. T1: `memoize_jaxpr` donation, both directions

### 3.1 Finding

Spec §4.2 item 6 reads: "`donate_argnums` users rejected at wrap (input invalidation);
consumers warned not to pass cached outputs into donation sites; `copy_on_return` offers
defensive copies (default off)".

- **Input direction: absent.** `rg donat src/xtrax/inference` finds nothing.
- **Output direction: half-built.**
  - The store is protected: `copy_on_return` copies at store (`memo.py:367`).
  - What comes back to the caller is not. The cached object itself is returned on the miss
    (`:375`), on every hit (`_finalize_hit`, `:383`), and on the spot-check hit path (`:357`).
    A consumer that donates or `.delete()`s a returned value therefore destroys the entry.
  - `_maybe_copy` calls `.copy()` on every leaf (`:481`), so a Python-scalar output raises
    `AttributeError`.
  - No documentation warns consumers.

### 3.2 How memo executes `fn`

- **Screening.** Admission is trace-only. `_ensure_program` runs
  `jax.make_jaxpr(probe)(*args)` (`:277`) at wrap time for zero-parameter callables (`:535-539`),
  and otherwise on the first call, before any concrete execution. Errors latch (`:326-330`).
- **Misses and spot-checks run `fn` eagerly** (`:361`, `:423`). Donation in top-level
  equations therefore acts on real buffers: P1 for direct donation, P1b for pass-through.

### 3.3 Donation-hazard rule (unchanged from r1) and fail-open paths (new)

**Rule: conservative.** Reject when any equation anywhere in the recursively walked jaxpr
carries a donation marker. There are two carriers (D3, P5):

1. any `True` in a `donated_invars` param;
2. any element of a `copy_semantics` param whose `.name == "DONATE_INPUT"`. This is
   duck-typed; do not import the private `ArrayCopySemantics` type.

| Case | Runtime effect under memo | Rejected? |
|---|---|---|
| (a) top-level input donated | The caller's buffer is invalidated on a miss and on a spot-check hit, but not on a plain hit. Invalidation becomes cache-state-dependent. | Yes (hazard) |
| (c) pass-through or closure constant donated | Same as (a). For a closure constant, the first miss deletes it, and whether later calls work depends on cache state. P1b confirms this. | Yes (hazard) |
| (b) intermediate donated | The donated buffer is invisible to the caller. | Yes (**over-rejection, by design**) |

**Why not provenance tracing.** It would need a per-primitive map from sub-jaxpr invars to
outer vars, and every one of those maps can fail open. It would also miss constvars unless
they were special-cased. Trace-time identity also diverges from eager identity: for example,
`jnp.asarray(x)` returns `x` itself and emits no equation.

**Consequence.** Donating an intermediate is not memoizable in v1, and there is no escape hatch.

**Fail-open paths (D4; stated here and in the user docs, pinned by strict-xfail tests):**

- **#5214: the screen runs once.** It checks the first call's abstract signature only
  (`program_digest` latches at `:271`). A function whose donation depends on shape, or on a
  Python branch over static structure, passes the screen on its first call.
- **#5215: the screen never traces kwargs.** It traces `probe(*args)` only (`:273-277`), so
  `fn(x, *, fast=False)` that donates only when `fast=True` passes.

Neither is fixed in this sprint: fixing either would change the key or admission machinery.

### 3.4 T1 contract

**Error type.** Add `MemoDonationError(MemoImpurityError)` to `xtrax/inference/errors.py`
and export it from `errors.__all__` and `xtrax.inference`.
- **Why a subclass:** the latch and `memo_rewrap()` keep working, existing
  `except MemoImpurityError` callers keep working, and the type stays distinguishable.
- **Structured attribute:**
  `sites: tuple[tuple[str, str, tuple[int, ...], tuple[int, ...]], ...]`. Each entry is
  `(path, carrier, eqn_operand_indices, wrapped_input_leaf_indices)`:
  - `carrier` is `"donated_invars"` or `"copy_semantics"`.
  - `eqn_operand_indices` are **flattened operand positions of that equation**. They are not
    argnums.
  - `wrapped_input_leaf_indices` is filled for top-level equations only. It lists each `j`
    such that the operand var `is closed.jaxpr.invars[j]`: an index into the flattened pytree
    leaves of the positional args. This is an identity lookup, not provenance tracing. For
    nested equations it is `()`.
- **Message** lists each site and the remedy: "remove donate_argnums/donate_argnames/
  device_put(donate=True) from functions wrapped by memoize_jaxpr (spec §4.2 item 6)".

**Traversal: `_iter_subjaxprs(value)`.**
- Yields any value that has `.eqns`, and **recurses into `tuple`/`list` values**.
- It must **not** reuse `_screen_jaxpr`'s `getattr(param, "eqns")` walk, which skips the tuple
  `cond.branches` and returns silently at depth > 8 (**#5216**).
- It walks with an explicit stack and has **no depth cap**.
- It covers all param values generically. Known carriers are exercised by tests: jit/pjit
  `jaxpr`, cond `branches`, scan `jaxpr`, while `cond_jaxpr`/`body_jaxpr`, and
  custom_jvp/vjp `call_jaxpr`.

**Screening.**
- `_screen_donation(closed)` collects **all** sites, then raises.
- `_ensure_program` calls it right after `_screen_jaxpr(closed)` and before `program_digest`
  is set.
- The zero-param wrap path (`:536-539`) must re-raise with a **bare `raise`** so the exception
  type is preserved.

**Output direction (OBJ-R1-01, OBJ-R1-13).** When `copy_on_return=True`:
- **Store:** store `_copy_array_leaves(raw_out)`. This keeps today's store-time protection, so
  the cache never holds a buffer that `fn` merely passed through from the caller.
- **Miss:** return `raw_out` itself. It is not the cached object. It may alias the caller's
  argument, exactly as the unwrapped `fn` would.
- **Hit and spot-check hit:** return `_copy_array_leaves(entry.value)`.
- **`_copy_array_leaves`** copies `jax.Array` and `np.ndarray` leaves only. Other leaves are
  immutable Python scalars or strings and are returned as-is.
- **Default (`False`):** unchanged. A hit returns the cached object by identity, and `fn`'s
  pass-through outputs alias caller arguments.

**Docs.** Add a "Value memoization (`memoize_jaxpr`)" section to `docs/api/inference.md`
covering:
- the import;
- donation is rejected, including the stated over-rejection;
- the two fail-open paths, citing #5214 and #5215;
- the consumer warning: "unless `copy_on_return=True`, do not donate or `.delete()` returned
  values, nor arguments whose values `fn` may return unchanged (e.g. `lambda x: x`)".

**Not in T1:**
- per-signature or kwargs re-screening (#5214, #5215);
- fixing `_screen_jaxpr`'s own traversal (#5216);
- an `allow_donation` escape hatch;
- any change to key derivation.

## 4. T2: `verify_dedup_spec`

### 4.1 Which claim verify checks (unchanged decision; §9 records the amendment)

- **Claim (i), row-equality, is checked bitwise per leaf, using the literal §4.3 signature.**
  A tolerance on *inputs* would accept merging `±0` or `x`/`nextafter(x)`, which is the
  silent-wrong-result class F3 targets.
- **Claim (ii), compute-equivalence, is not checked.** It needs `fn`. The appendix's "numeric,
  never bitwise" rule belongs to it and is deferred to **#5217**.
- **Scope of the check: soundness only.** F3's ascending first-occurrence form is not required.

### 4.2 Contract

Everything lives in `src/xtrax/tiling/dedup_synthesis.py` and is exported from that module's
`__all__`. It is **not** exported from `xtrax.tiling.__init__`.

```python
class DedupSpecVerificationError(ValueError):
    check: str                # index_type | index_dtype | k_mismatch | index_map_length |
                              # unique_indices_bounds | index_map_bounds | row_mismatch
    first_bad_row: int | None # row_mismatch: min row index bad in ANY leaf (union)
    n_bad: int                # row_mismatch: count of rows bad in ANY leaf (union)
    leaf_index: int | None    # row_mismatch: lowest leaf index mismatching at first_bad_row

@dataclass(frozen=True)
class DedupVerificationResult:
    n_rows: int
    k: int
    transfer_bytes_spent: int  # device->host bytes moved via _to_host (see below)

def verify_dedup_spec(spec: DedupSpec, leaves: Sequence[Any], *, axis: int = 0
                      ) -> DedupVerificationResult: ...
```

**Leaf validation.** Leaves go through T4's shared `_validate_batch_leaves`, so both functions
accept identical inputs and raise identical errors. Unsupported leaves (object dtype, ragged,
typed PRNG keys, dtypes the byte recipe cannot handle) raise `DedupSynthesisUnsupportedError`.
Any `N` mismatch or bad `axis` raises `ValueError`.

**Structural checks** live in `_check_spec_structure(spec, N)`. They run in this order on the
spec's host arrays and complete **before any device→host transfer**. The first failure raises.
1. `index_type`: `spec.unique_indices` or `spec.index_map` is not an `np.ndarray`. A
   `jax.Array` or list would make the host checks below perform hidden transfers or
   conversions.
2. `index_dtype`: not a 1-D integer array (`np.issubdtype(dt, np.integer)`).
3. `k_mismatch`: `len(unique_indices) != k`.
4. `index_map_length`: `len(index_map) != N`.
5. `unique_indices_bounds`: any value outside `[0, N)`. JAX clamps out-of-range indices and
   numpy wraps negative ones (F3b).
6. `index_map_bounds`: any value outside `[0, k)`.

All checks re-run because `DedupSpec`'s arrays are mutable after construction.

**Row check.**
1. `canon = unique_indices[index_map]` is computed on host, after the bounds checks pass.
2. For each leaf `l`: `b_l = _leaf_row_bytes(leaf_l, axis, N)` gives an `(N, B_l)` uint8 array
   on device (T4).
3. The device-only helper `_mismatch_mask_device(blocks, canon)` builds the `(N, L)` bool mask:
   `jnp.stack([jnp.any(jnp.not_equal(b, jnp.take(b, canon, axis=0)), axis=1) for b in blocks], axis=1)`.
   Move it to host with **exactly one `_to_host` call**.
4. Union semantics apply as defined above.
5. `transfer_bytes_spent = N * L` (the mask). NumPy leaves count the same way, because their
   byte rows pass through the device.

**Transfer route.**
- `_to_host(x: jax.Array) -> np.ndarray` is the **only** device→host route in
  `dedup_synthesis.py`, used by both verify and synthesize.
- It begins with `if not isinstance(x, jax.Array): raise TypeError(...)` and then returns
  `np.asarray(x)`.
- §7.1 gives the structural oracle that enforces this route.

**Docs.** Append `automodule xtrax.tiling.dedup_synthesis` (`:members:`) to
`docs/api/tiling.md`. The docstring states:
- the (i)/(ii) split and #5217;
- that the comparison is bitwise, and why;
- the O(N) device compute and N·L-byte transfer;
- the union semantics.

**Not in T2:** claim (ii) (#5217), an `fn=` parameter, and wiring verify into the planner.

## 5. T4: synthesizer row identity is exact per leaf in native bytes (D1)

### 5.1 Contract

T4 lives in the same file as T2 and **shares its helpers**, so the two definitions cannot drift.

**`_validate_batch_leaves(batch_leaves, axis) -> tuple[list[np.ndarray | jax.Array], int]`**
- It is the existing ragged/object validation loop (`:285-326`), unchanged in its
  no-materialization behaviour: `jax.Array` leaves are skipped, `np.ndarray` gets a
  metadata-only dtype check, and other inputs go through `np.asarray`.
- It adds a **typed-key rejection** (P7). Test
  `jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key)` and raise
  `DedupSynthesisUnsupportedError` with the remedy "pass `jax.random.key_data(keys)`".
- It checks `axis` range and equal `N` per leaf, using metadata only.

**`_row_byte_kind(dtype) -> str`** (host-side glue, dtype metadata only). It returns one of
five kinds, checked in this order:

| Order | Condition | Result |
|---|---|---|
| 1 | typed key dtype | already refused in `_validate_batch_leaves` |
| 2 | `bool` | `"bool"` |
| 3 | complex | `"complex"` |
| 4 | `jax.dtypes.itemsize_bits(dtype) < 8` and integer | `"subbyte_signed"` or `"subbyte_unsigned"`, by signedness |
| 5 | `jax.dtypes.itemsize_bits(dtype) < 8` and floating | raise `DedupSynthesisUnsupportedError` |
| 6 | otherwise | `"plain"` |

- **Why sub-byte floats are refused, not widened.** Widening via `astype(int8)` would be a
  numeric conversion, not a lossless one. A float32 widening is plausible but unmeasured, so
  this sprint refuses instead of guessing.
- **Why `itemsize_bits`, not `itemsize * 8`.** `itemsize` is 1 for int4 (§1.1). Using it would
  never detect sub-byte types.

**`_leaf_row_bytes(leaf, axis, N) -> jax.Array`** is host-side glue. It returns the leaf's
rows as an `(N, B_l)` uint8 array on device.
- It computes `kind = _row_byte_kind(leaf.dtype)`, then dispatches.
- **NumPy leaf** (`isinstance(leaf, np.ndarray)`):
  `jnp.asarray(_host_leaf_row_bytes(leaf, axis, N, kind))`.
- **`jax.Array` leaf:** `m = _device_rows(leaf, axis, N)`, then the kind's device helper:

  | Kind | Device helper | Body |
  |---|---|---|
  | `bool` | `_device_bytes_bool(m)` | `m.astype(jnp.uint8)` |
  | `complex` | `_device_bytes_complex(m)` | `jnp.concatenate([_device_bytes_plain(jnp.real(m)), _device_bytes_plain(jnp.imag(m))], axis=1)` |
  | `subbyte_signed` | `_device_bytes_subbyte_signed(m)` | `_device_bytes_plain(m.astype(jnp.int8))`; exact per P11 |
  | `subbyte_unsigned` | `_device_bytes_subbyte_unsigned(m)` | `m.astype(jnp.uint8)` |
  | `plain` | `_device_bytes_plain(m)` | `lax.bitcast_convert_type(m, jnp.uint8).reshape(m.shape[0], -1)` |

- **Unsupported dtypes.** The glue wraps the dispatch in `except (TypeError, ValueError)` and
  re-raises `DedupSynthesisUnsupportedError` naming the dtype.

**`_host_leaf_row_bytes(a, axis, N, kind) -> np.ndarray`** is host-only.
1. It starts with a guard: `if not isinstance(a, np.ndarray): raise TypeError`.
2. For the sub-byte kinds, it first sets `a = a.astype(np.int8 or np.uint8)`.
3. It returns `np.ascontiguousarray(np.moveaxis(a, axis, 0)).reshape(N, -1).view(np.uint8)`.

Properties of this host path:
- It uses host-native bytes. `np.moveaxis` avoids `jnp.moveaxis`'s x64-off truncation (P4).
- `ascontiguousarray` makes transposed and strided leaves viewable.
- Complex and bool leaves view natively on host.
- Non-canonical numpy bool bytes compare as distinct, which is the safe direction and is
  documented.

**D2, complex: split, not reject.** Device bitcast of complex is unsupported (P6). Splitting
into real and imaginary halves is injective, and P9 shows it preserves `±0` in the imaginary
part. Rejecting would newly refuse complex inputs that the synthesizer accepted before T4.

**Zero-width rows** need no special case (P12). All-zero-width input correctly yields `k == 1`.

**`_stack_batch_leaves(batch_leaves, axis) -> jax.Array`** is kept, with a new meaning:
1. `leaves, N = _validate_batch_leaves(...)`
2. `_concat_row_bytes([_leaf_row_bytes(l, axis, N) for l in leaves])`, where
   `_concat_row_bytes` is `jnp.concatenate(blocks, axis=1)`.

The result is an `(N, B)` uint8 array. Concatenating uint8 arrays cannot promote dtypes (P3).
PR #104's `TestJaxArrayNoMaterialization` (`test_dedup_synthesis.py:424-484`) runs against it
**unmodified**.

**Sample-stage gather.** The device-only helper `_take_rows_device(stacked, idx)` computes
`jnp.take(stacked, jnp.asarray(idx), axis=0)`. `_sample_stage` then calls
`_to_host(_take_rows_device(stacked, idx))`, so the gather happens **on device, before the
transfer**. `_exact_stage` calls `_to_host(stacked)`, which is the intended full transfer. All
gather indices are proven to lie in `[0, N)` on host before use, so `jnp.take`'s
out-of-bounds mode never applies.

**`synthesize_dedup_spec`:**
- **Axis bug:** `N = stacked.shape[0]`. Drop the stale `axis` argument to
  `_sample_stage`/`_exact_stage`. Both stages already index the batch-first array.
- **Transfers:** both stages go through `_to_host`. The sample stage transfers only the rows
  gathered on device by `_take_rows_device`.
- **Row identity:** `np.unique(uint8 rows, axis=0)` is now exact byte identity. This fixes P2,
  and identical-NaN rows now deduplicate.
- **Accounting:** `_element_width_bytes(stacked)` now reports the true bytes per row, `Σ B_l`.

### 5.2 Behaviour changes

Document these in the `synthesize_dedup_spec` docstring and in the CHANGELOG. Each is a
soundness fix; none is a regression.

- **Signed zeros:** `±0` rows are now distinct, so `k` may rise. It can cross `max_unique_k`,
  turning a former `"synthesized"` result into `"k_over_limit"`.
- **NaN rows:** bitwise-identical NaN rows now deduplicate, so `k` may fall.
- **Wide numpy leaves:** numpy int64/float64 leaves are no longer truncated under x64-off, so
  `k` may rise.
- **Mixed-dtype accounting:** `transfer_bytes_spent` and `k_bucket_bytes` now use true byte
  widths. For example, int8 + float32 counts 5 bytes per row, not the promoted 8.
- **Non-zero `axis`:** `axis != 0` now works. Before, `N` was wrong, which led to a spurious
  `ValueError` or a garbage ratio.
- **Typed PRNG key leaves:** now refused with a typed error, where they used to raise a raw
  `TypeError`.
- **Sub-byte dtypes:** sub-byte integer leaves (int2/int4/uint2/uint4 …) are compared exactly
  after widening. Sub-byte float leaves (float4/float6) are refused with
  `DedupSynthesisUnsupportedError`.

**Not in T4:** heterogeneous-shape axes, and any change to the sampling policy or thresholds.

## 6. T3: added-types-diff, nested public-named defs (#5205)

**The fix, confirmed in r1:** `visit_FunctionDef` returns `False` unconditionally after
`_record`.

- **Methods are still collected.** `visit_ClassDef` returns `True` for public classes (`:41`).
- **Two pre-fix defects are fixed.**
  1. "Unable to locate callable" on nested public-named defs.
  2. The dict-overwrite collision at `:52`, in **both** directions:
     - A change to the nested def spuriously flags the top-level def.
     - More seriously, a real signature change to the top-level `helper` is **silent**
       (OBJ-R1-11), because both the base and head maps hold the nested def.
- **Classes defined inside functions are no longer collected.** This is correct: they are not
  importable, and lookup cannot resolve them.

**Accepted residual (OBJ-R1-11).** Factory-built public API, such as `foo = _make_foo()` with
a nested `def foo`, moves from accidentally loud ("unable to locate") to silent. The gate
audits `def` statements, and an assigned callable has no top-level `def` to audit. Covering
assigned callables would be a gate feature, not this bug fix.

**Residual scope mismatches, not fixed:** nested classes (lookup uses `parts[0:2]`) and
module-level `if`/`try` defs. Neither occurs in `src/xtrax` today; the AC-43 sweep makes a
future instance fail loudly.

## 7. Acceptance criteria (GWT)

**Negative controls.** Every control must be shown **red before green**: run it against the
pre-change code, or with the new check disabled, and record the result in the PR body.

**Concrete spy (T1).** Counts calls whose argument is not a `jax.core.Tracer`.

### 7.1 Transfer oracle (T2/T4), defined only here

A runtime-only spy cannot see every transfer route on CPU:
- the `ArrayImpl._value` getter misses `np.asarray`, `np.array` and `np.ascontiguousarray`
  (P8);
- `jax.transfer_guard` never fires on CPU (P10).

The oracle is therefore **structural plus a typed runtime spy**. **The `_value` spy is deleted.**

1. **`_to_host` spy (runtime).** `monkeypatch.setattr(dedup_synthesis, "_to_host", wrapper)`.
   The wrapper records `(type(arg), arg.shape)`, then calls the original. Every test that uses
   it asserts that **every recorded argument is a `jax.Array` instance**, which catches any
   conversion made before the call. It also asserts the expected call count and shapes.
   `_to_host` itself raises `TypeError` on a non-`jax.Array` argument (AC-48).
2. **Device-only helper set `D`**, in `dedup_synthesis.py`: `_device_rows`,
   `_device_bytes_plain`, `_device_bytes_bool`, `_device_bytes_complex`,
   `_device_bytes_subbyte_signed`, `_device_bytes_subbyte_unsigned`, `_concat_row_bytes`,
   `_take_rows_device`, `_mismatch_mask_device`.
   - Host-only counterpart: `_host_leaf_row_bytes`, guarded by `isinstance(np.ndarray)`.
   - Host-side glue: `_row_byte_kind`, `_leaf_row_bytes`, `_validate_batch_leaves`,
     `_stack_batch_leaves`, `_sample_stage`, `_exact_stage`, `_check_spec_structure`,
     `verify_dedup_spec`, `synthesize_dedup_spec`.
3. **AST allowlist over `D` (AC-45).** The test parses `dedup_synthesis.py` with `ast` and
   asserts that every name in `D` is defined at module level. Inside each helper:
   - Every `ast.Call.func` must be one of:
     - an attribute chain rooted at the name `jnp` or `lax`;
     - an attribute whose final name is in `{"reshape", "astype", "real", "imag"}`;
     - the name `_to_host`;
     - a name in `D`.
   - These node types must not appear: `If`, `While`, `Assert`, `Compare`, `BoolOp`,
     `JoinedStr`, `IfExp`.
   - Widening the method allowlist requires a spec revision.
4. **Module-wide `np.*` pin (AC-46).** Collect every `Call` whose func is an attribute chain
   rooted at `np`, as a pair `(enclosing function name, dotted attribute)`. Assert that the set
   **equals** the pinned literal below; any new site fails until reviewed. Expected final set:
   - `(_validate_batch_leaves, asarray)`: list inputs only;
   - `(_host_leaf_row_bytes, ascontiguousarray)` and `(_host_leaf_row_bytes, moveaxis)`;
   - `(_to_host, asarray)`;
   - `(_sample_stage, linspace)` and `(_sample_stage, unique)`;
   - `(_exact_stage, …)` for each of `unique`, `where`, `array`, `argsort`, `empty`, `arange`;
   - `(_estimate_duplication_ratio, unique)`;
   - `(_check_spec_structure, issubdtype)`.

   `_row_byte_kind` uses `jnp.issubdtype` and `jax.dtypes.itemsize_bits`, and so adds no `np`
   site. A deviation from this list is allowed only with a PR-body justification, and the test
   literal must match the final code.
5. **Checker controls (AC-47).** Fed synthetic source, the allowlist checker must flag each of
   these:
   - `np.asarray(stacked)` inside a function named `_take_rows_device`;
   - `np.unique(blocks)` inside `_concat_row_bytes`;
   - an `if` inside `_device_rows`.

   The pin checker must flag a synthetic extra `np.array(...)` site. In each case the red
   result is printed in the PR.
6. **Mutation control for AC-33.** Replace `_sample_stage`'s gather with
   `_to_host(stacked)[idx]`, a host-side slice after a full transfer. AC-33's shape assertion
   must go red. Record this in the PR.

### T1 (`tests/inference/test_memo.py`)

| ID | Given | When | Then | Negative control / red |
|---|---|---|---|---|
| AC-1 | `memoize_jaxpr(jax.jit(f, donate_argnums=0))`, f(x,y)=x*2+y | first call | `MemoDonationError` (also `MemoImpurityError`); `memo_get_stats()["misses"] == 0`; `x.is_deleted() is False` | no donation → `misses == 1`, correct value. Pre-fix red: `x.is_deleted()` (P1) |
| AC-2 | f(p, y) with `p=(a, b)`, `jit(f, donate_argnums=1)`; separately `donate_argnames='y'`; separately g(d) with dict `d={'a','b'}`, `donate_argnums=0` | first call | `sites[0]` has `wrapped_input_leaf_indices == (2,)`, `(2,)`, and `(0, 1)` respectively: flattened leaves, not argnums | — |
| AC-3 | Python `outer(x, y)` with a concrete spy, calling `jax.jit(f, donate_argnums=1)(x, y) + 1.0` | first call | `MemoDonationError`; spy == 0; `y.is_deleted() is False`; the path names the inner `jit` | no donation → spy == 1. Pre-fix red: `y.is_deleted()` (P1b) |
| AC-4 | donating `jit` inside a `lax.cond` branch (#5216) | first call | `MemoDonationError`; path contains `branches[` | same with no donation → admitted. Red if the walk uses `getattr(param, "eqns")`: the fixer shows this by swapping the walker |
| AC-5 | donating `jit` in a `lax.scan` body; separately 10 nested `jit` levels deep | first call | both rejected | no donation → admitted. Red for any depth cap ≤ 9 |
| AC-6 | `outer(x) = jax.device_put(x, donate=True) * 2.0` | first call | `MemoDonationError`; carrier `"copy_semantics"` | `device_put(x)` without donate → admitted. No `is_deleted` assertion (P5b) |
| AC-7 | intermediate donation `jit(g, donate_argnums=0)(x * 2)` | first call | rejected. Test name `test_intermediate_donation_rejected_by_design`, citing §3.3(b) | — |
| AC-8 | AC-1's function | 2nd call; `memo_rewrap()`; 3rd call | 2nd raises from the latch (`make_jaxpr` spy unchanged); 3rd re-screens and raises | — |
| AC-9 | zero-param fn donating a closure constant | `memoize_jaxpr(fn)` | `type(exc) is MemoDonationError` at wrap | pre-fix `:539` downcasts → red |
| AC-10 | canaries | `make_jaxpr` of `jit(f, donate_argnums=0)` and of `device_put(x, donate=True)` | a `donated_invars` param contains `True`; a `copy_semantics` element has `.name == "DONATE_INPUT"` | on failure, the message says the screen would silently pass |
| AC-11 | `memoize_jaxpr(lambda x: x, policy=MemoPolicy(copy_on_return=True))`; `x2 = x.copy()` | `f(x)`; `x.delete()`; `r = f(x2)` (hit) | `r` is usable and `allclose(r, x2)` | red on r1's store-raw design (the fixer shows this by storing `raw_out`) |
| AC-12a | `copy_on_return=True`; `r1 = f(x)` (miss) | `r1.delete()`; `r2 = f(x)` (hit) | `r2` usable, allclose, hits == 1 | `copy_on_return=False` → `r2 is r1` and `r2.is_deleted()` (current code red: same) |
| AC-12b | `copy_on_return=True`; `f(x)`; `r2 = f(x)` (hit) | `r2.delete()`; `r3 = f(x)` (hit) | `r3` usable, `r3 is not r2` | `copy_on_return=False` → `r3 is r2` and `r3.is_deleted()` (current code red: same) |
| AC-12c | `copy_on_return=True, spot_check_every=1`; `r1 = f(x)` (miss) | `r1.delete()`; `r2 = f(x)` (spot-checked hit) | no error; `r2` usable; `spot_check_mismatches == 0` | `copy_on_return=False` → `MemoStalenessError`: the spot-check compares against the deleted cached buffer, `_numeric_equal` falls back to `a is b` (`:474-475`) (current code red: same) |
| AC-13 | `copy_on_return=True`, fn returns `(x * 2, 3.0)` | miss, then hit | both succeed; the float leaf is returned as-is | current code red: `AttributeError` (`:481`) |
| AC-14 | default policy | two equal calls | `f(x) is f(x)` | — |
| AC-15 | fn donating only when `x.shape[0] > 4`; first call shape (4,), second call shape (8,) | second call | `with pytest.raises(MemoDonationError)`. Marker: **`xfail(strict=True, raises=pytest.fail.Exception, reason="#5214")`** (§1.1 evidence). Only an unmet `pytest.raises` counts as the expected failure. Any other exception is a real FAIL. | the strict XPASS forces marker removal once fixed |
| AC-16 | `fn(x, *, fast=False)` donating only when `fast=True` | `f(x, fast=True)` | same form, with `reason="#5215"` | as AC-15 |
| AC-17 | docs | `rg -n "memoize_jaxpr\|donat\|#5214\|#5215" docs/api/inference.md` | all present, with the consumer warning | — |

The vacuous `test_donation_rejected_at_wrap` is deleted. Mutation check: disabling the
`_screen_donation` call turns AC-1 through AC-9 red.

### T2 (`tests/tiling/test_verify_dedup_spec.py`, new)

| ID | Given | When | Then | Negative control / red |
|---|---|---|---|---|
| AC-18 | spec from `synthesize_dedup_spec` on the AC7 fixture (N=10000, ~30 uniques, one leaf) | verify | `DedupVerificationResult(10000, spec.k, 10000)` | — |
| AC-19 | hand-built sound, non-F3 spec (last-occurrence canonicals, unsorted) | verify | passes | — |
| AC-20 | two leaves; row 5 differs from its canonical in leaf 1 only, row 9 in leaf 0 only | verify | `row_mismatch`, `first_bad_row == 5`, `n_bad == 2`, `leaf_index == 1` | same spec with rows 5 and 9 repaired → passes |
| AC-21 | float32 rows `+0.0`/`-0.0`, and separately `1.0`/`nextafter(1.0)`, mapped together | verify | both raise `row_mismatch` | identical-NaN-payload rows mapped together → pass |
| AC-22 | dtype matrix; each row pair differs only in the named way | verify **and** synthesize | distinct: `bfloat16` (`jax.Array`); `complex64` (imag only, `jax.Array` and numpy); `bool` (`jax.Array`); numpy int64 `5` vs `2**32+5` (x64 off); numpy float64 `1.0` vs `1.0+2**-40`; numpy transposed/strided leaf with `axis=1`; **int4 `(N,3)`** (`jax.Array`); **uint4 `(N,3)`** (`jax.Array`); uint4 `(N,2)` (`jax.Array`) | the same pairs made identical → merged / pass. **Red on a no-widen implementation:** int4 `(N,3)` and uint4 `(N,3)` hit the bitcast `ValueError` (P11), which becomes `DedupSynthesisUnsupportedError` instead of a result. The uint4 `(N,2)` case is a **consistency case, not a red control**: without widening it packs within the row, which is lossless (P11), so it passes either way. The fixer shows the red by disabling the sub-byte dispatch. |
| AC-23 | typed key leaf `jax.random.split(key(0), 4)`; separately a `float4_e2m1fn` leaf | verify; synthesize | all raise `DedupSynthesisUnsupportedError`. The key message names `key_data`; the float4 message names the dtype | `jax.random.key_data(...)` leaf → accepted |
| AC-24 | spec fields mutated after construction: `unique_indices` replaced by a `jax.Array` (`index_type`); float `index_map` (`index_dtype`); `index_map` of length N-1; `unique_indices` ∋ N; ∋ -1; `index_map` ∋ k | verify under the §7.1 `_to_host` spy | the matching `check` name each time; **zero** `_to_host` calls | AC-25 |
| AC-25 | valid spec, two `jax.Array` leaves | verify under the **same** §7.1 spy | exactly one `_to_host` call, with argument a `jax.Array` of shape `(N, 2)` | this is the control proving AC-24's spy observes transfers |
| AC-26 | public surface + docs | import the three names; `rg -n "dedup_synthesis" docs/api/tiling.md` | imports succeed; `issubclass(DedupSpecVerificationError, ValueError)`; the automodule block is present | — |

### T4 (`tests/tiling/test_dedup_row_identity.py`, new)

**Fixture rule for AC-27 through AC-32.** Every fixture follows this form, so no `k` assertion
can pass vacuously through an early exit:
- two named row patterns, each repeated **×50** (N = 100);
- an explicit `threshold=0.1`;
- `assert result.stage == "synthesized"` **before** any `k` assertion.

| ID | Given | When | Then | Pre-T4 red |
|---|---|---|---|---|
| AC-27 | float32 patterns `[0., 1.]` and `[-0., 1.]` | synthesize | stage synthesized; `spec.k == 2`; the two patterns map to different slots; `verify(spec)` passes | `k == 1` (P2) |
| AC-28 | int32 leaf patterns `[2**24]` and `[2**24+1]`, beside a float32 leaf `[1.]` in both | synthesize; separately, verify a hand-built spec merging the two patterns | stage synthesized; `k == 2`; verifies clean; the hand-built merge raises `row_mismatch` (OBJ-R1-10) | `k == 1` (P3) |
| AC-29 | numpy int64 patterns `[5]` and `[2**32+5]`, x64 off | synthesize | stage synthesized; `k == 2` | `k == 1` (P4) |
| AC-30 | a bitwise-identical NaN pattern and `[1.]` | synthesize | stage synthesized; `k == 2` | `k == 51` |
| AC-31 | leaf of shape `(3, 100)` with `axis=1`, whose columns are two patterns ×50 | synthesize | stage synthesized; `len(spec.index_map) == 100`; `k == 2`; verifies clean | wrong `N` → `ValueError` |
| AC-32 | int8 leaf patterns `[1]`/`[2]` beside float32 `[1.]`/`[2.]` | synthesize | stage synthesized; `k_bucket_bytes == get_k_bucket(2) * 5 == 10`; `transfer_bytes_spent == 5 * (100 + 100) == 1000` (100 sampled rows + 100 exact rows) | promoted width 8 → 1600 |
| AC-33 | all-unique `jax.Array` batch of N = 1000, below threshold, `max_sample_rows=64` | synthesize under the §7.1 `_to_host` spy | stage `no_duplication`; exactly one `_to_host` call; its argument is a `jax.Array` with `shape[0] == len(sample idx) ≤ 64` | red under the §7.1(6) mutation (full transfer, then host slice) |
| AC-34 | `_leaf_row_bytes` wrapped with a counter | call verify and synthesize | both call it once per leaf (one shared definition) | — |
| AC-35 | regression | `git diff origin/main -- tests/tiling/test_dedup_synthesis.py` and run that file | the diff is empty; all tests pass, including `TestJaxArrayNoMaterialization` | — |
| AC-44 | leaf A: float32 `(100,1)`, two patterns ×50. Leaf Z: float32 `(100,0)` | synthesize `[A, Z]` and `[A]`; synthesize `[Z]`; verify each | `[A, Z]` and `[A]` give equal `unique_indices` and `index_map`. `[Z]` gives stage synthesized, `k == 1`, and `verify(spec, [Z])` passes (P12) | — |
| AC-45 | `dedup_synthesis.py` source | §7.1(3) AST allowlist test | every helper in `D` exists; no forbidden call or node | AC-47 |
| AC-46 | `dedup_synthesis.py` source | §7.1(4) `np.*` pin test | the collected `(function, attr)` set equals the pinned literal | AC-47 |
| AC-47 | synthetic sources from §7.1(5) | run the checkers from AC-45 and AC-46 | all four planted violations are flagged | this is the control for AC-45/46 |
| AC-48 | guards | `_to_host(np.zeros(3))`; `_host_leaf_row_bytes(jnp.zeros((3, 1)), 0, 3, "plain")` | both raise `TypeError` | — |

### T3 (`tests/audit/test_added_types_diff.py`)

| ID | Given | When | Then | Pre-fix red |
|---|---|---|---|---|
| AC-36 | typed public `build()` + private `_topo()` containing `def visit(n)` | `diff_callables_to_audit(None, head)` → `audit_changed_callables` | touched == `{"build"}`; zero violations | "unable to locate callable \`visit\`" |
| AC-37 | public-named closure in a public fn and in the public method `Cls.meth` | same | closures not collected; `build` and `Cls.meth` are | `Cls.inner` unlocatable |
| AC-38 | top-level `helper(x: int) -> int`, then `outer()` with a nested `helper(y)`; head changes only the **nested** helper's params | `diff_callables_to_audit(base, head)` | `set()` | `{"helper"}` |
| AC-39 | same file; head changes only the **top-level** helper's params | same | `{"helper"}` | `set()`: **silent gate** (OBJ-R1-11) |
| AC-40 | the same file also adds an untyped `bad(x)` and an untyped method `Cls.worse(self, y)` | `run_added_types_diff_gate` on a tmp repo | `status == "fail"`; violations name exactly `bad` and `Cls.worse` | — (control: the fix does not pass by collecting nothing) |
| AC-41 | class defined inside a public fn, with a public method | collect | not collected | `Local.m` unlocatable |
| AC-42 | factory pattern `foo = _make_foo()` with a nested `def foo` | collect | not collected (accepted residual, §6, asserted so it is visible) | "unable to locate" |
| AC-43 | every `*.py` under `src/xtrax` **except `__init__.py`** (mirrors `added_types_diff.py:164`) | the sweep | 0 unresolvable qualnames. The fixer records the pre-fix count in the PR body | pre-fix count recorded, not asserted |

## 8. Tasks, order, ownership, verification

**One PR (D6). Execution is sequential, one fixer at a time (OBJ-R2-08, corrected):**

1. **T3.** Already in progress. The gate runs on every diff, and later tasks add code.
2. **T1.**
3. **T2+T4**, as one fixer, because the two tasks share helpers and files.

**Where fixers run.** All fixers run in the single sprint worktree
`/home/marielle/projects/xtrax/.claude/worktrees/ring-probes`, on branch
`feat/conformance-residuals`. Parallel worktrees are **not executable** here: a dispatched
fixer inherits the session's worktree isolation and cannot write to another worktree
(measured by the orchestrator). Running one fixer at a time removes the `index.lock` race and
any chance of one fixer's commit sweeping another's working-tree files.

**Between fixers**, before the next one starts, the orchestrator:
- checks each new commit's file list (`git show --stat <sha>`) against the ownership table
  below. A commit touching a file its task does not own is repaired first.
- repairs any hook auto-commit ("auto: fixer output") before pushing.

**`.praxia/` is off-limits to fixers.** Fixers never stage anything under `.praxia/`. The
orchestrator commits this spec file to the PR once the spec converges.

| Task | ACs | Files it owns (sole editor) | Est. LOC |
|---|---|---|---|
| T3 | AC-36 to AC-43 | `src/xtrax/devtools/gates/added_types_diff.py`, `tests/audit/test_added_types_diff.py` | ~5 src + ~170 test |
| T1 | AC-1 to AC-17 (AC-12 has three variants: a, b, c) | `src/xtrax/inference/memo.py`, **`src/xtrax/inference/errors.py`**, **`src/xtrax/inference/__init__.py`**, **`docs/api/inference.md`**, `tests/inference/test_memo.py` | ~120 src/docs + ~280 test |
| T2+T4 | AC-18 to AC-35, AC-44 to AC-48 | `src/xtrax/tiling/dedup_synthesis.py`, **`docs/api/tiling.md`**, **`CHANGELOG.md`** (writes the entries for **all four** tasks, from §3.4, §5.2 and §6), `tests/tiling/test_verify_dedup_spec.py` (new), `tests/tiling/test_dedup_row_identity.py` (new; also holds the AC-45 to AC-47 oracle tests) | ~240 src + ~420 test |
| none | — | `src/xtrax/tiling/__init__.py` is **unchanged** by every task | — |

**Gates.** Narrow runs only. Never a whole suite.

```bash
uv run --extra dev pytest tests/audit/test_added_types_diff.py -q                                    # T3
JAX_PLATFORMS=cpu uv run --extra dev pytest tests/inference/test_memo.py -q                          # T1
JAX_PLATFORMS=cpu uv run --extra dev pytest tests/tiling/test_verify_dedup_spec.py \
    tests/tiling/test_dedup_row_identity.py tests/tiling/test_dedup_synthesis.py -q                  # T2+T4 (+ #104 regression file)
uv run --extra dev ruff check src/xtrax/inference src/xtrax/tiling src/xtrax/devtools/gates tests/inference tests/tiling tests/audit
uv run --extra dev ruff format --check src tests
uv run --extra dev ty check src/
uv run python scripts/audit_added_types_diff.py --repo-root . --base origin/main --no-emit             # gate dogfood on this PR: PASS
rg -n "device_get|\.tolist\(|\.item\(" src/xtrax/tiling/dedup_synthesis.py                           # expect no matches (transfer-route rule)
```

## 9. Amendment to 260825 (D5, recorded here; the 260825 file is not edited)

- **§4.3 / §10.2 / §10.3 (`verify_dedup_spec`).**
  - Appendix l.399-401 and l.427-428 say "compare numerically, never bitwise". That rule is
    **re-scoped** to output compute-equivalence (claim ii).
  - `verify_dedup_spec(spec, leaves)` checks input row identity (claim i), **bitwise per leaf**.
    A tolerance there is unsound (§4.1).
  - The numeric output check is follow-up **#5217**.
- **§4.3 exact stage.** "Exact" now means **byte identity per leaf in native dtype** (T4), not
  numpy float equality over a promoted concatenation.
- **§4.2 item 6.**
  - "Rejected at wrap" means **at admission**: at wrap for zero-param callables, otherwise on
    the first call before concrete execution.
  - Donation carriers are `donated_invars` and `device_put` `copy_semantics`.
  - `copy_on_return` protects both the store and every returned hit.
  - Admission has two known fail-open paths, **#5214** and **#5215**.

## 10. Out of scope

- The rest of epic #5169, and #5202.
- Any change to `memoize_jaxpr` key derivation.
- #5214, #5215, #5216 and #5217 (filed; §3.3, §3.4, §9).
- `allow_donation`.
- N-level class paths and `if`/`try` defs in `_lookup_ast_callable`.
- Heterogeneous-shape dedup axes.
- Two vacuity notes, not fixed here:
  - `test_memo.py:105`'s `or True` (`:107-108` assert strictly);
  - `test_dedup_synthesis.py:531/549`'s `if result.spec is not None:` guards, which can pass
    vacuously. That file must stay unmodified (AC-35), so this goes to a follow-up.

## 11. Round-1 dispositions

| OBJ | Sev | Disposition | What changed / why |
|---|---|---|---|
| R1-01 | BLOCKER | ACCEPT (mechanism refined) | The store-time copy is kept. Hits and spot-check hits return copies. The miss returns `raw_out`, which is not the cached object, so no second copy is made. AC-11 is the challenger's case; the docs warning was added. |
| R1-02 | MAJOR | ACCEPT | AC-1 now uses `misses == 0` and `is_deleted`; P1 shows this is non-vacuous. The spy moved to an outer Python fn (AC-3). |
| R1-03 | MAJOR | ACCEPT via D4 | Both paths are stated in §3.3 and the docs, pinned by strict-xfail AC-15/16, and filed as #5214 and #5215. |
| R1-04 | MAJOR | ACCEPT via D3 | Second carrier `copy_semantics` DONATE_INPUT (P5); AC-6 and AC-10. P5b is stated. |
| R1-05 | MINOR | ACCEPT | Structured `sites`; operand indices vs wrapped-input flattened-leaf indices; pytree cases in AC-2. |
| R1-06 | MINOR | ACCEPT, one part REBUT | The recipe is pinned (§5.1): host recipe for numpy leaves, device bitcast for `jax.Array` (keeps sample-stage transfers minimal). Dtype, strided and axis=1 cases are in AC-22. **Rebut on error class:** typed keys raise the existing `DedupSynthesisUnsupportedError` (260825 §4.3 l.252, the typed refusal for unsupported input), not a new TypeError-class error. It is typed, fail-loud, and adds no API. |
| R1-07 | MINOR | ACCEPT | `transfer_bytes_spent` means device→host bytes via `_to_host`, and numpy leaves count the same. Union semantics are defined. AC-20 pins `n_bad == 2`. |
| R1-08 | MINOR | ACCEPT (**superseded in r3**; see §13, R2-01) | r2 used a `_to_host` spy plus an `ArrayImpl._value` spy. P8 showed the `_value` spy misses `np.asarray`, `np.array` and `np.ascontiguousarray`. r3 replaces it with the §7.1 structural oracle. |
| R1-09 | MAJOR | ACCEPT via D5 | §9 amendment; #5217. |
| R1-10 | MINOR | ACCEPT | AC-28. |
| R1-11 | MAJOR | ACCEPT | AC-39 covers the silent direction. The factory-API residual is accepted and asserted (AC-42). |
| R1-12 | MINOR | ACCEPT | AC-43 asserts a post-fix count of 0, with the pre-fix count recorded in the PR. `__init__.py` is skipped, mirroring `:164`. |
| R1-13 | MINOR | ACCEPT | `_copy_array_leaves` (§3.4); AC-13. |
| R1-14 | MINOR | ACCEPT via D6 | §8 gives the order and sole ownership of shared files. |

## 12. Risks

| Risk | Mitigation / rollback |
|---|---|
| Over-rejection breaks an existing `memoize_jaxpr` user | Alpha-only release. Grep aminx, tev_design and plegadx for `memoize_jaxpr` before merge. The message states the remedy. Rollback: remove the `_screen_donation` call. |
| JAX renames `donated_invars`/`copy_semantics` | AC-10 canaries fail loudly. The traversal is generic. |
| T4 changes synthesized `k` for real data (±0, NaN, wide numpy dtypes) | Every change is in the sound direction, listed in §5.2 and the CHANGELOG. The verify ACs prove the new specs are exact. |
| `lax.bitcast_convert_type` inside `_stack_batch_leaves` materializes, breaking the unmodified #104 test | AC-35 catches it. If it trips, the path has a real transfer that must be removed. Do not edit the test. |
| The AST allowlist (AC-45) is too strict for a legitimate future device-side helper | Widening it requires a spec revision. That friction is intended: a device-only helper is where a hidden transfer would hide. |
| The `np.*` pin literal (AC-46) churns on refactors | Each new site needs a one-line PR justification. That review is the point of the pin. |
| The `jax.dtypes.itemsize_bits` API changes | The int4/uint4 cases in AC-22 fail loudly if sub-byte detection stops working. |
| `copy_on_return=True` now copies on every hit | Opt-in; the cost is documented. The previous behaviour protected nothing on return. |
| The T4 device byte matrix doubles device memory during synthesis | Same footprint as the old promoted concat. Documented. |
| This PR trips the unfixed gate | T3 goes first. The §8 dogfood command runs the gate on this PR. |

## 13. Round-2 dispositions

The challenger verified these as SOUND:
- byte identity in the exact stage (`dedup_synthesis.py:391-414`);
- the complex split (P9);
- `bool` via `astype`;
- the R1-06 error-class rebut;
- the #104 test staying green.

| OBJ | Sev | Disposition | What changed / why |
|---|---|---|---|
| R2-01 | MAJOR | ACCEPT | §7.1 has been rebuilt as a structural oracle in one place:<br>- The `_value` spy is **deleted** (P8, P10).<br>- The `_to_host` spy checks argument types.<br>- There is a named device-only helper set `D`, and a guarded host-only `_host_leaf_row_bytes`.<br>- An AST allowlist covers `D` (AC-45) and a module-wide `np.*` pin covers the rest (AC-46).<br>- The checker controls are AC-47, and the AC-33 mutation control is §7.1(6).<br>AC-24, AC-25 and AC-33 reference §7.1. The r2 `rg` line is kept as a cheap extra check. |
| R2-02 | MAJOR | ACCEPT, three parts REBUT | Integer sub-byte dtypes are widened via `astype(int8/uint8)` before the bitcast (P11). Unsupported dtypes are caught as `(TypeError, ValueError)` and re-raised as `DedupSynthesisUnsupportedError`. int4 and uint4 `(N,3)` added to AC-22. **Rebut (a), detection:** `itemsize*8 < 8` never fires, because `itemsize` is 1 for int4 (`jax/_src/dtypes.py:282-285`). The spec uses the public `jax.dtypes.itemsize_bits` (`jax/dtypes.py:20`). **Rebut (b), sub-byte floats:** "widen by signedness" would put float4/float6 through an integer `astype`, which is a lossy numeric conversion. They are refused instead (AC-23). **Rebut (c), uint4 `(N,2)` is not red on a no-widen implementation:** the bitcast packs within the row, which is lossless (P11), so row identity is preserved. The red control for unsigned is uint4 `(N,3)`, and `(N,2)` stays as a consistency case. |
| R2-03 | MINOR | ACCEPT | Fixture rule for AC-27 to AC-32: ×50 repetition, `threshold=0.1`, and a stage assertion before `k`. Expected byte counts are pinned in AC-32. |
| R2-04 | MINOR | ACCEPT | AC-15/16 use `xfail(strict=True, raises=pytest.fail.Exception)`. Verified against pytest 9.1.1 source (§1.1): an unmet `pytest.raises` raises `Failed` (`pytest.fail.Exception`), and `skipping.py:290-294` matches it by `isinstance`. |
| R2-05 | MINOR | ACCEPT | AC-12 split into variants a, b and c, each with its own `copy_on_return=False` control. In the spot-check control (c), the expected red is `MemoStalenessError`. |
| R2-06 | MINOR | ACCEPT | New structural check `index_type`: the spec's fields must be `np.ndarray`. It runs first and is covered in AC-24. |
| R2-07 | MINOR | RESOLVED by P12 | No special case for zero-width. AC-44 covers a zero-width leaf beside a normal one, and an all-zero-width input (`k == 1`, verify passes). |
| R2-08 | MINOR | ACCEPT (corrected per orchestrator) | Fixers run **sequentially** (T3 → T1 → T2+T4) in the one sprint worktree, with a per-commit `git show --stat` ownership check and repair of any auto-commit. Parallel worktrees were measured as not executable. Fixers never stage `.praxia/`. |
| R2-09 | MINOR | ACCEPT | AC-33's red is the §7.1(6) mutation. The row now says so. |

**Design decisions left to a fixer: none.** Every helper name, dispatch rule, error class,
check order, transfer route, fixture, expected value and file owner is fixed above. The only
fixer-produced artifacts are measurements the spec asks them to record:
- the AC-43 pre-fix count;
- the red results of each negative control.

The AC-46 pin literal is also fixed: §7.1(4) lists it, and any deviation needs a PR
justification.

## 14. Round-3 ACCEPT conditions (binding)

Challenger round 3 returned **VERDICT: ACCEPT**. Its round-2 table marks R2-02 to R2-09 as
RESOLVED and R2-01 as PARTIAL. The partial items are closed by conditions 1-4 below. The
challenger verified:
- rebuts (a) to (c) of R2-02;
- that each AC-47 plant goes red under §7.1(3)/(4);
- the pre-T4 reds of AC-12a/b/c, AC-29, AC-30 and AC-31;
- the expected values in AC-18 and AC-32.

The conditions below are the challenger's wording and **override any conflicting text above**.
Conditions 1-6 bind the T2+T4 fixer; condition 7 bound the orchestrator and is done.

1. **(OBJ-R3-01) Checker shape and the meaning of "rooted at `jnp`".**
   - **Functions:** implement the checkers as `check_device_allowlist(source: str) -> list[str]`
     and `collect_np_sites(source: str) -> set[tuple[str, str]]`.
   - **Inputs:** AC-45/46 call them on the real file text, and AC-47 on synthetic strings.
   - **Walk:** walk every `D` function with `ast.walk` over the whole `FunctionDef`, including
     nested lambdas, comprehensions and decorators.
   - **An allowed `Call.func` is exactly one of:**
     - (a) an `ast.Attribute` whose `.value` chain contains only `ast.Attribute` nodes and ends in
       an `ast.Name` with id in {`jnp`, `lax`};
     - (b) an `ast.Attribute` with `attr` in {`reshape`, `astype`, `real`, `imag`}, on any
       receiver;
     - (c) an `ast.Name` with id in `D ∪ {"_to_host"}`.
   - **Disqualifier:** any `Call` or `Subscript` inside the chain disqualifies rule (a). So
     `jnp.sum(m).item()` and `jnp.asarray(m).tolist()` are rejected.
2. **(OBJ-R3-02) Forbidden nodes inside `D`.** In addition to §7.1(3), these are forbidden:
   - `ast.UnaryOp` with op `ast.Not`;
   - `ast.Match`;
   - any `ast.comprehension` with a non-empty `.ifs`;
   - any `ast.Subscript` whose `.value` is not an `ast.Attribute` with `attr == "shape"`.

   AC-47 gets a plant for each:
   - `[b for b in blocks if jnp.any(b)]` in `_mismatch_mask_device`;
   - `not jnp.any(m)` in `_device_rows`;
   - `m[mask]` in `_take_rows_device`;
   - `jnp.sum(m).item()` in `_device_bytes_plain`.
3. **(OBJ-R3-03) Builtin conversions.**
   - **Collect:** AC-46 also collects, module-wide, every `Call` whose func is an `ast.Name` in
     {`int`, `float`, `bool`, `complex`, `bytes`, `bytearray`, `memoryview`, `list`, `tuple`},
     as (function, name) pairs.
   - **Assert:** the collected set equals a literal derived from the final code. Justify each
     site in one line in the PR body.
   - **Extra control:** plant `memoryview(stacked)` in `_sample_stage` as an AC-47 control.
   - **Replace the `rg` line:** its patterns (`device_get`, `.tolist`, `.item`, `.__array__`)
     move into this AST check.
4. **(OBJ-R3-04) The AC-46 literal.**
   - **Also includes** `(_sample_stage, round)` and `(_exact_stage, asarray)`.
   - **`verify_dedup_spec` contains no `np.*` calls.** Compute the union, `first_bad_row`,
     `n_bad` and `leaf_index` with ndarray methods only: `.any(axis=1)`, `.nonzero()`,
     `.argmax()`, `.sum()`.
5. **(OBJ-R3-05) Dtype kinds.**
   - **Scope of the wrapper:** call `_row_byte_kind` inside the same
     `except (TypeError, ValueError)` wrapper.
   - **Allowed kinds:** bool, complex, integer and floating, tested with `jnp.issubdtype`
     against `np.bool_`, `np.complexfloating`, `np.integer` and `np.floating`.
   - **Everything else** raises `DedupSynthesisUnsupportedError`. That covers numpy `U`, `S`,
     `datetime64`, `timedelta64` and structured dtypes.
   - **AC-23 additions:** a numpy `'U4'` leaf. AC-23's float4 red control: disabling the
     sub-byte-float refusal makes it return a result.
6. **(OBJ-R3-06) Error messages.** `_validate_batch_leaves` keeps the existing `ValueError`
   messages byte-for-byte: `'empty'`, `'batch axis {axis} has length 0'`,
   `'has batch dimension'` and `'out of range'`. It also keeps the `N == 0` check. AC-35 runs
   the unmodified test file.
7. **(OBJ-R3-07) Evidence traceability. Done by the orchestrator:** P11/P12 raw output and every
   probe are now in the repo (§1.1).

## References

- `.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md`: §4.2 item 6 (l.190-192), §4.3
  (l.252, l.268), §10.2-10.4 (l.391-457), AC3, AC20.
- PR #104 / `abcce47`; PR #101; PR #156; backlog #5205; follow-ups #5214, #5215, #5216, #5217;
  epic #5169.
- `.praxia/docs/research/260922_conformance-residuals-probes.md` (P0-P12 raw output and the
  axis bug), produced by `scripts/measure_conformance_residuals_260922.py`.
- Installed sources: `.venv/.../_pytest/{raises,outcomes,skipping}.py` (pytest 9.1.1);
  `.venv/.../jax/dtypes.py`, `jax/_src/dtypes.py` (jax 0.11.1).
