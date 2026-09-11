---
title: Divergence mapping for exported artifacts
description: A ring ladder and a reusable fixture that localizes where a compiled artifact departs from production JAX, instead of reporting one scalar
task_id: 260911_export-divergence-map
status: draft
---

# Divergence mapping for exported artifacts

## 1. The problem, stated as a measurement failure

`xtrax.export.parity.compare` reduces an entire comparison to one scalar,
`max_abs_diff`, over one array (`parity.py:62-104`). When the aminx ProteinMPNN
artifact diverged (#5093) it reported `score 4.938e-02`, `logits 2.160e+00`,
`decoding order not exact`. Those three numbers are the complete diagnostic
output of the current system. They do not say which stage, which operation,
which input class, or even whether the cause is the model, the export, or the
backend.

Localization was therefore done by hand: four throwaway scripts
(`topk_equiv_control.py`, `prng_parity.py`, `iree_divergence.py`,
`divergence_scope.py`), each hypothesis hand-coded, each result discarded. That
is the process this spec replaces.

**This is not a debugging convenience.** A scalar tolerance check is
*structurally blind* to the bug class that actually bit us, demonstrated in §2:
a divergence carried entirely by integer indices, with every float leaf
bit-identical. `compare()` on the float output returns `max_abs_diff = 0.0` and
passes. The instrument is not merely coarse; for this class it reads zero.

## 2. The motivating measurement (260911, this session)

Measured against `iree-base-compiler 3.11.0rc20260316`, `jax 0.10.2`, on
`llvm-cpu` with `--iree-llvmcpu-target-cpu=host`. Reproduce with:

```
uv run --extra export --extra export-runtime python \
    scripts/measure_iree_sort_stability.py
```

| # | Question | Result |
|---|---|---|
| M1 | Does a multi-output pytree survive `jax.export` → IREE → runtime? | **Yes.** Returns a flat `tuple`. Dict keys are NOT preserved — flattened-pytree order only. |
| M2 | Is `ireert.Config("local-task")` (multi-threaded) run-to-run deterministic? | **Yes**, bit-identical across 20 runs. |
| M3 | Does IREE reproduce XLA's *integer* `sort_key_val` tie-break? | **No.** 64 slots / 4 distinct keys: XLA `[0 4 8 12 …]`, IREE `[48 36 20 28 …]`, differing at **45 of 64** positions. |
| M4 | Does IREE honour `jnp.argsort(..., stable=True)` on a tie-heavy float row? | **No.** Indices diverge; values bit-identical (`max|diff| = 0.000e+00`). A distinct-valued control is exact. |

M4 is the load-bearing one. The construct tested is verbatim the replacement
shipped in aminx PR #155:

```python
order = jnp.argsort(-x, axis=-1, stable=True)[..., :k]
```

`jnp.argsort` is stable *by JAX's documented default*, and PR #155's commit
message calls that stability "load-bearing, not cosmetic" — correctly, because
`jax.lax.top_k` breaks ties toward the lower index. **IREE does not honour it.**
Masked or padded distance matrices are saturated with exactly-equal entries, so
the tie path is the common case, not an edge case.

On a row whose 20 masked slots sit at indices 0–19, top-8 selection gives:

| | selected indices |
|---|---|
| XLA (stable — first tied wins) | `[0 1 2 3 4 5 6 7]` |
| IREE | `[12 13 14 15 16 17 18 19]` |

**8 of 8 differ — the selected sets are disjoint**, while `max|value diff|`
stays `0.000e+00` because every one of those entries holds the same masked
sentinel. A distinct-valued control on the identical function is exact, so the
divergence is attributable to tie-breaking specifically and not to the function
or the pipeline.

The causal chain to #5093 is then complete and consistent with every earlier
negative result:

1. Tied masked distances → IREE selects a different, equally-distant kNN index set.
2. Values compare bit-identical, because tied entries are equal by definition —
   which is exactly why the whole-model *eager* control in PR #155 was clean and
   proved nothing about the compiled path.
3. The differing **indices** feed the neighbour graph, so all downstream message
   passing runs on a different graph → `score 4.9e-02`, `logits 2.16`.

This also **retires the standing hypothesis** recorded in #5093. FMA/reassociation
under `-march=native` was the leading suspect; `jax.random.permutation` is
implemented as `lax.sort_key_val` over uint32 random bits
(`jax._src.random.core._shuffle`), so no floating-point arithmetic participates
and FMA cannot perturb it. The mechanism is sort-stability, not FMA.

### 2.1 Scope beyond aminx

Any JAX model relying on documented stable-sort tie-breaking diverges silently
when compiled through IREE. That is an export-safety class, not one model's bug.
Filed consequence in §8.

## 3. What "self-consistency ring" means here

The term is not currently defined anywhere in xtrax or aminx (verified: zero
hits in source or docs of either repo). This spec fixes it, and the definition
is offered explicitly for challenge rather than assumed:

> A **ring** is a closed comparison loop in which exactly one axis of the system
> is varied while everything else is held fixed. It is *self*-consistency because
> both sides are the same computation; a disagreement therefore localizes to the
> varied axis and to nothing else.

A ring is not a test. It is a controlled experiment with one independent
variable. Rings compose into a **ladder**, ordered by cost and — critically — by
whether they perturb the artifact under study.

| Ring | Varied axis | Perturbing | Isolates |
|---|---|---|---|
| **R0 Replay** | nothing — same artifact, same inputs, twice | no | runtime nondeterminism |
| **R1 Target** | target CPU: `host` / `x86-64-v2` / `wasm32-generic` | no | ISA-dependent codegen (FMA, vector reductions) |
| **R2 Stack** | eager JAX / `jit` XLA / IREE | no | model fusion-sensitivity vs export lowering |
| **R3 Probe** | cut depth — named intermediates as extra outputs | **yes** | spatial onset inside the model |
| **R4 Input** | input population: ties, masks, magnitudes, lengths | no | which input *class* triggers it |

**The ladder is the strategy the sprint exists to deliver.** Run the cheap,
non-perturbing rings first; escalate to R3 only when R0–R2 fail to explain the
divergence. R3 is last because it is the only ring that changes the thing being
measured (§5.3).

Worked against #5093, the ladder reaches the answer without a single hand-written
script: R0 passes (M2, deterministic), R1 passes (tie-break is not ISA-dependent),
R2 shows jit-XLA and IREE disagreeing on an **int** leaf while all float leaves
match — which is already the diagnosis, before R3 is ever needed. R4 then
confirms it by showing the divergence appears only in the tie-bearing input class.

### 3.1 Why R2 needs three legs, not two

Comparing eager JAX against IREE conflates two independent things: whether the
*model* is fusion-sensitive, and whether the *export* lowers faithfully. Eager
JAX is op-by-op XLA; `jit` is fused XLA. So:

- `eager ≠ jit` — the model is fusion-sensitive. A property of the model. Not an
  export bug.
- `jit ≠ IREE` — the export/lowering differs. This is xtrax's problem.

This yields a **calibrated tolerance**: a model whose own `eager`-vs-`jit`
divergence is already 1e-2 has no standing to demand 1e-5 of IREE. Ring R2
derives the budget from the model's measured self-sensitivity instead of an
arbitrary `atol`, which is what makes the ring "self"-consistent rather than
merely a comparison against a constant.

## 4. Non-goals

- Not a replacement for `parity.compare`. Parity answers *is it correct*;
  divergence mapping answers *where did it stop being correct*. Both ship.
- Not automatic root-causing. The fixture localizes and classifies; a human reads
  the report.
- Not a fix for the IREE sort-stability gap itself. That is upstream (§8).
- Not multi-axis plan support. Unchanged from today's composer limits.

## 5. Design

### 5.1 Probe mechanism — ordinary pytree outputs

Recon established three facts that kill the obvious design and dictate the real
one:

- `Tap` is contractually an `io_callback` wrapper (`stages/boundaries.py:47-63`,
  "Implementations must use io_callback internally"), and
  `topology.py:201-209` rejects **any** Tap at the export boundary
  unconditionally. Taps cannot become outputs.
- `materialize` is a deliberate single-slot mechanism: one sink, one axis
  (`MultipleMaterializeAxesError`, `topology.py:65-71`), never a Tap, and the
  outer axis of the only certified multi-axis shape may carry no boundary at all
  (`composer.py:236-248`).
- The composer/executor layer is **pytree-transparent** — `ys` is whatever the
  step fn's `y` is, and nothing inspects its shape.

Therefore: **a probe is a named entry in the step function's ordinary return
pytree.** No changes to `topology.py`, `pipeline.py`, or the boundary protocol.
The consumer writes an instrumented variant of its step fn returning
`(primary, {probe_name: value})`; everything downstream already works.

M1 confirms this survives compilation, with one trap: **the runtime returns a
flat tuple with no key information.** Probe names MUST be recovered from the
exported `out_tree`, never from positional assumption. An implementation that
zips names onto the tuple in declaration order is wrong whenever a dict is
involved, because JAX flattens dict keys in sorted order.

### 5.2 Dtype-aware metrics

A single `max_abs_diff` is wrong for non-float leaves — M4 is exactly the case
where the float metric reads 0.0 while the result is wrong. Per-leaf, by dtype:

| dtype | metrics |
|---|---|
| floating | `max_abs_diff`, `max_rel_diff`, `max_ulp_diff`, `n_nonfinite_mismatch` |
| integer | `exact_match_fraction`, `first_mismatch_index`, `n_mismatched` |
| bool | `hamming_distance`, `first_mismatch_index` |

`max_ulp_diff` is what separates "last-bit fp noise" from "a real lowering
change" without a hand-tuned `atol`.

### 5.3 Injection vs amplification — the actual diagnostic payload

Reporting only "the earliest probe that exceeds tolerance" is tolerance-dependent
and therefore misleading: a 1-ULP difference at probe 1 that grows to 1e-1 by
probe 9 has its "onset" wherever the threshold happens to sit. The report must
instead classify each probe against its predecessor:

- **CLEAN** — within the R2-calibrated budget.
- **AMPLIFIED** — divergence was already nonzero upstream and grew smoothly.
  Ordinary floating-point behaviour meeting a sensitive function.
- **INJECTED** — divergence appears where the immediately-upstream probe was
  bit-identical. **This is the signal.** It names an operation that changed
  semantics under lowering.
- **DISCRETE_FLIP** — an integer/bool leaf mismatched while all upstream float
  leaves were bit-identical. The M4 signature, and its own category because the
  float metrics read zero.

The primary output is the ordered probe sequence with these labels, not a scalar.

### 5.4 Instrumentation fidelity — the Heisenberg guard

**The most serious threat to R3's validity.** Adding outputs changes DCE and
fusion decisions in both XLA and IREE. A probe that forces an intermediate to be
materialized can suppress the very fusion that caused the divergence, so the
instrumented artifact may not diverge at all.

R3 is therefore **invalid unless it first reproduces what it is explaining.**
Before any probe is interpreted, the instrumented artifact's *primary* output
must reproduce the uninstrumented artifact's divergence within a stated
tolerance. If it does not, the fixture MUST report
`INSTRUMENTATION_CHANGED_RESULT` and refuse to emit a probe map — never silently
report "no divergence found". A vanishing divergence is itself the finding: it
says the cause is fusion-dependent.

This is why the ladder puts the three non-perturbing rings first.

### 5.5 Probe coverage

The user asked for coverage checks. Two distinct senses, both required:

**(a) Code coverage.** `tier1_core` measures `coverage_packages = ["xtrax"]` at
90% line / 80% branch enforced, but syncs only `["dev", "io"]` — **no export
extra** (`distribution/coverage_dag.toml`). Anything needing a live IREE
toolchain is unreachable there. This forces the module's architecture: pure
comparison/classification/coverage logic in one module with no toolchain import,
execution in another. The pure half is the large half and is fully testable with
no IREE, matching the existing fake pattern in `tests/export/conftest.py`.

**(b) Probe coverage — the one that matters.** A probe map is only as good as its
resolution. Two probes on a forty-stage model localize nothing, yet the report
would look identical in structure to a thorough one. The fixture MUST therefore
report, and the gate MUST enforce, the *resolution* of its own localization:

- `n_probes`, and the StableHLO op count between consecutive probes;
- `largest_unprobed_span` — the maximum op count between adjacent probes, which
  is the true upper bound on localization precision;
- `unprobed_head` / `unprobed_tail` — ops before the first and after the last
  probe.

A divergence localized to a 4,000-op span is not localized. Reporting
`largest_unprobed_span` alongside every verdict prevents the fixture from
overclaiming — the failure mode of a diagnostic tool is false confidence, not
false negatives.

## 6. Module layout

```
src/xtrax/export/divergence.py       # pure: metrics, classification, coverage
src/xtrax/export/rings.py            # execution: R0-R4 runners
tests/export/test_divergence.py      # pure-logic tests, no toolchain
tests/export/test_rings.py           # toolchain-gated + fake-injected
```

Public surface added to `xtrax/export/__init__.py`:
`LeafDivergence`, `ProbeReport`, `DivergenceClass`, `RingResult`, `RingLadder`,
`compare_pytree`, `classify_probes`, `probe_coverage`, `run_ring_ladder`.

`compare_pytree` generalizes `compare`; `compare` stays, unchanged, for the
single-array parity path.

## 7. Tasks

| # | Task | Deliverable |
|---|---|---|
| T1 | `compare_pytree` + dtype-aware `LeafDivergence` | `divergence.py` |
| T2 | `classify_probes` — CLEAN/AMPLIFIED/INJECTED/DISCRETE_FLIP | `divergence.py` |
| T3 | `probe_coverage` — op counts from the StableHLO module | `divergence.py` |
| T4 | R0 Replay + R1 Target + R2 Stack runners | `rings.py` |
| T5 | R3 Probe runner **with** the §5.4 fidelity precondition | `rings.py` |
| T6 | R4 Input-class runner | `rings.py` |
| T7 | `out_tree` name recovery (M1 trap) | `rings.py` |
| T8 | Pure-logic tests, no toolchain, ≥90% line / ≥80% branch | `test_divergence.py` |
| T9 | Toolchain tests + fakes | `test_rings.py` |
| T10 | `docs/api/export.md` — the ladder, and when to escalate | docs |
| T11 | aminx dogfood (§9) | aminx PR |

## 8. Consequences to file separately

- **`check_export_safety` should block or warn on stable-sort reliance.** M3/M4
  show `jnp.argsort(stable=True)`, `jnp.sort`, and `lax.sort_key_val` do not
  preserve XLA tie-breaking through IREE. This belongs with the unlegalizable-op
  work already filed as #5092.
- **aminx PR #155's `top_k` replacement has a latent divergence.** The fix was
  still necessary — `jax.lax.top_k` is uncompilable — but its correctness
  argument rests on stability that IREE does not provide. Needs a follow-up that
  makes tie-breaking explicit rather than inherited (e.g. a deterministic
  index-tiebreak composed into the sort key).
- **#5093 should be updated**: FMA is retired as the hypothesis; sort stability
  is the identified mechanism.

## 9. Dogfood on aminx

aminx supplies both the probe points and the schema precedent, so the dogfood
requires no model changes:

- `features.py:109` `forward_edge_stages()` already returns
  `ProteinEdgeStageTensors` — seven named intermediates (`neighbor_indices`,
  `rbf`, `encoded_positions`, `edges_concat`, `after_w_e`, `after_norm`, `final`),
  documented verbatim as *"Intermediate edge tensors for parity diagnosis"*.
  Production `__call__` calls it and discards all but four. This is exactly the
  §5.1 shape, already written.
- Encoder/decoder layers are plain accessible lists (`model.encoder.layers[i]`),
  so per-layer probes need only an external re-drive of the loop, no model edit.
- `src/aminx/parity/evidence.py` already defines `EvidenceMetricRecord` /
  `EvidencePointRecord`. The fixture emits into that schema rather than
  inventing one.

**Acceptance for the dogfood:** the ladder, run on the real
`proteinmpnn_v_48_020` artifact, must independently reproduce §2's conclusion —
`DISCRETE_FLIP` on `neighbor_indices` with upstream float leaves bit-identical —
without any hypothesis supplied by the operator. If it cannot rediscover a
divergence whose cause is already known, it will not find an unknown one.

## 10. Acceptance criteria

- **AC-1** `compare_pytree` returns one `LeafDivergence` per leaf with
  dtype-appropriate metrics; a structure mismatch fails loudly rather than
  broadcasting (inherits `compare`'s existing rule).
- **AC-2** Probe names are recovered from the exported `out_tree`. A regression
  test uses a **dict** probe map whose sorted key order differs from declaration
  order, and asserts names bind to the right arrays. Without this the test is
  vacuous.
- **AC-3** `classify_probes` labels a synthetic sequence containing one injected
  step, one amplified tail, and one discrete flip, correctly and in order.
- **AC-4** `DISCRETE_FLIP` is raised when an int leaf mismatches while every
  upstream float leaf is bit-identical — asserted on the M4 construct
  specifically.
- **AC-5** R3 refuses to emit a probe map when the instrumented artifact does
  not reproduce the uninstrumented divergence, reporting
  `INSTRUMENTATION_CHANGED_RESULT`.
- **AC-6** `probe_coverage` reports `largest_unprobed_span`, and every ring
  verdict carries it.
- **AC-7** R2 reports all three legs separately, and the derived tolerance
  budget is a function of the measured `eager`-vs-`jit` divergence, not a
  constant.
- **AC-8** `divergence.py` imports nothing from `iree` — asserted by a test —
  and reaches ≥90% line / ≥80% branch under `tier1_core` with no export extra.
- **AC-9** `just audit-deterministic` exits 0 **and** its log contains no
  `FAILED` (per the known audit-masking trap in this repo).
- **AC-10** The aminx dogfood rediscovers §9's divergence unaided.
- **AC-11** R0/R1/R2 run without instrumenting the model — asserted structurally,
  since the ladder's whole ordering argument depends on it.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Instrumentation suppresses the divergence | §5.4 fidelity precondition; AC-5 |
| Probe map looks authoritative but is low-resolution | `largest_unprobed_span` on every verdict; AC-6 |
| New module tanks tier1 coverage (no IREE there) | Pure/execution split; AC-8 |
| Multi-output IREE return assumed rather than measured | Already measured (M1); AC-2 guards the naming trap |
| Ladder ordering degrades into "run everything" | AC-11 pins non-perturbation of R0–R2 |
| `max_ulp_diff` on non-IEEE or nonfinite values | Explicit `n_nonfinite_mismatch`; ULP computed only on finite pairs |
