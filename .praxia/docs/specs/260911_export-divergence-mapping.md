---
title: Divergence mapping for exported artifacts
description: A comparison ladder and a reusable fixture that localizes where a compiled artifact departs from production JAX, instead of reporting one scalar
task_id: 260911_export-divergence-map
status: draft
revision: 3 (post-defense)
---

# Divergence mapping for exported artifacts

> **Revision 2.** Revised against an adversarial challenge (9 BLOCKER / 13 MAJOR).
> Changes of substance: the ring taxonomy lost two of its five rows, which were
> not rings; the calibrated tolerance gained a formula, a floor, and a scope
> limit; probe ordering became an explicitly declared DAG rather than an
> inferred sequence; the coverage metric became a backward-slice difference and
> is no longer claimed as a bound; the fidelity precondition stopped gating on
> the one quantity that reads zero for the target bug class. §2's headline claim
> was narrowed after measurement — see §2.3.
>
> **Revision 3.** Adjudicated against the defense, which upheld both of revision
> 2's rejections but found six blocking items, four of them introduced *by*
> revision 2. Fixed: the load-bearing measurement moved from a regime the
> artifact refuses to an in-contract one (M6); the aminx probe DAG was corrected
> and given slice-subset validation, revision 2 having linearized a join in the
> paragraph arguing against linearizing joins; the tolerance floor was
> redenominated in ULP, which its own rationale required; treedef and leaf
> mismatch were separated; `DISCRETE_FLIP` stopped relabelling inherited index
> flips as fresh injections; and every task now maps to a criterion.

## 1. The problem, stated as a measurement failure

`xtrax.export.parity.compare` reduces an entire comparison to one scalar,
`max_abs_diff`, over one array (`parity.py:62-104`). When the aminx ProteinMPNN
artifact diverged (#5093) it reported `score 4.938e-02`, `logits 2.160e+00`,
`decoding order not exact`. Those three numbers are the complete diagnostic
output of the current system. They name no stage, no operation, and no input
class.

Localization was therefore done by hand across four throwaway scripts, each
hypothesis hand-coded and each result discarded. That is the process this spec
replaces.

**This is not a debugging convenience.** A scalar float tolerance is
*structurally blind* to the bug class demonstrated in §2: a divergence carried
entirely by integer indices, with every float leaf bit-identical. `compare()`
returns `max_abs_diff = 0.0` and passes. For this class the instrument does not
merely lose resolution — it reads zero.

## 2. The motivating measurement

Measured 260911 against `iree-base-compiler 3.11.0rc20260316`, `jax 0.10.2`,
`llvm-cpu`, `--iree-llvmcpu-target-cpu=host`. Reproduce with:

```
uv run --extra export --extra export-runtime python \
    scripts/measure_iree_sort_stability.py
```

| # | Question | Result |
|---|---|---|
| M1 | Does a multi-output pytree survive `jax.export` → IREE → runtime? | **Yes.** Returns a flat `tuple`. Keys are NOT carried. |
| M2 | Is `ireert.Config("local-task")` (multi-threaded) deterministic? | **Yes**, bit-identical across 20 runs. |
| M3 | Does IREE reproduce XLA's *integer* `sort_key_val` tie-break? | **No.** 64 slots / 4 distinct keys differ at **45 of 64** positions. |
| M4 | Does IREE honour `jnp.argsort(stable=True)` on a tie-heavy float row? | **No.** Indices diverge, values bit-identical (`0.000e+00`); all-distinct control exact. |
| M5 | Does the divergence reach a **live** (`mask==1`) row? | **Only under exact ties.** See §2.2. |
| M6 | Does an **in-contract** input (no padding, `L >= k`) diverge? | **Yes.** Ideal α-helix: 58 of 64 live rows, 846 of 3072 index slots. Irregular control: 0. |

### 2.1 The construct measured

M4's construct is verbatim `aminx/model/features.py:66` on branch
`feat/xtrax-export-dogfood` (PR #155, **unmerged**; `main` still has the
`jax.lax.top_k` wrapper):

```python
order = jnp.argsort(-x, axis=-1, stable=True)[..., :k]
```

The branch matters and is stated because the two trees differ. `jax.lax.top_k`
lowers to a `stablehlo.composite` that IREE rejects outright, so **among
artifacts that compute their own kNN**, the argsort form is the only one that
compiles.

Revision 2 said "necessarily", which was too strong. There is a third path:
`features.py:152` branches on `if rbf_features is not None`, setting
`distances = None` (`:158-162`), and the sole `top_k` call at `:202` is gated on
`if distances is not None` (`:182`). An artifact exported in precomputed-features
mode contains **no sort at all** and compiles regardless. That matters twice —
it bounds this section's claim, and it is a live alternative explanation for
#5093 (§2.3).

`jnp.argsort` is stable by JAX's documented default, and PR #155 calls that
stability "load-bearing, not cosmetic" — correctly, since `lax.top_k` breaks
ties toward the lower index. **IREE does not honour it.** On a row whose 20
masked slots sit at indices 0–19, top-8 selection gives:

| | selected indices |
|---|---|
| XLA (stable — first tied wins) | `[0 1 2 3 4 5 6 7]` |
| IREE | `[12 13 14 15 16 17 18 19]` |

Disjoint sets, with `max|value diff|` at `0.000e+00` because every entry holds
the same sentinel.

### 2.2 When it reaches a live row — M5

aminx masks by `jnp.where(mask[:,None]*mask[None,:], distances, jnp.inf)`
(`features.py:182-189`), so masked pairs become `+inf` and sort **last**. They
are selected only when a row holds fewer than `k_neighbors` finite entries.
Measured at `L=64, k=48`:

| regime | finite per live row | ties selected | rows differing | **live** rows differing |
|---|---|---|---|---|
| no padding | 64 | no | 0 | **0** |
| sub-`k` padding | 40 | yes | 40 | **40** |

Two conclusions, and the first corrects this spec's revision 1:

1. **With every row at or above `k` finite entries there is no divergence at
   all.** Revision 1 claimed "the tie path is the common case, not an edge
   case". That was overreach and is withdrawn. Euclidean distances between
   distinct float32 coordinates are not tied in general.
2. **Below `k`, the divergence lands squarely on live rows** — all 40 of them. A
   live row with fewer than `k` valid neighbours is *forced* to select tied
   `-inf` padding. The tie-saturated rows are emphatically not the dead rows.

**But neither regime above carries the claim, and M5 alone cannot.** The sub-`k`
regime is precisely the one the artifact *refuses*: the bucket-ladder decision
makes padding score-preserving iff `L >= k_neighbors`, and the artifact declines
below the clamp. Demonstrating a divergence in a regime that is never served
proves nothing about served inputs.

The in-contract route is **exact geometric ties at `L >= k` with no padding at
all**, and it is measured as M6. An ideal α-helix has constant rise and turn, so
`d(i,j)` depends only on `|i-j|` and every pair at equal separation is exactly
equidistant:

| input (L=64, k=48, mask all ones) | tied rows | live rows diverging | index slots diverging |
|---|---|---|---|
| ideal α-helix | 60 / 64 | **58 / 64** | **846 / 3072 (27.5%)** |
| irregular backbone (control) | 0 / 64 | 0 / 64 | 0 |

No padding, no sub-`k` row, no refusal — over a quarter of the kNN graph
differs. Idealised and symmetric backbones are not a curiosity for ProteinMPNN;
they are the standard input of de novo design. **This, not M5, is the load-bearing
measurement.**

(Revision 2 found this route by accident — a constant 3.8 Å backbone step makes
`d(i,i-1)` and `d(i,i+1)` exactly equal — and mistook it for a generator
artefact to be jittered away. It was the in-contract case.)

**Propagation to the score.** M5 and M6 establish that differing rows are live;
liveness alone is not propagation. The two sites that carry it:
`encoder.py:239-249` updates edge features with no mask applied, and
`decoder.py:144` gates attention as `attention_mask = ar_mask[neighbor_indices]`
then `mask_bw = mask[:,None] * attention_mask` — the row's own validity times the
*neighbour's autoregressive state*, never `mask[neighbour]`. So a changed
neighbour set reaches the score through both, and padding-slot edge features are
not neutralised downstream.

### 2.3 What this does and does not establish about #5093

**Established:** IREE does not preserve XLA's sort tie-breaking (M3, M4); under
exact ties this changes the kNN graph on live rows, both out of contract (M5)
and **in** contract (M6); and the changed graph reaches the score through an
unmasked encoder edge update and a neighbour-indexed decoder gate (§2.2).

**Not established:** that #5093's specific artifact met that condition. Whether
its inputs carried tied geometry has not been measured, and there is a competing
explanation that does not involve sorting at all — if the artifact was exported
in precomputed-features mode (§2.1) it contains no sort, and #5093's divergence
must then come from somewhere else entirely. Two candidate causes, neither
eliminated.

This is deliberately left open rather than hand-resolved: distinguishing them is
exactly what the fixture is for, and §9 makes it an acceptance criterion. A spec
that guessed here would be asserting the thing its own instrument exists to
measure.

**Partially retired:** the FMA hypothesis. `jax.random.permutation` is
`lax.sort_key_val` over uint32 random bits (`jax._src.random.core._shuffle`), so
no floating-point arithmetic participates and reassociation cannot perturb it.
That retires FMA **for the decoding-order leg only**. #5093's `score` and
`logits` legs are float outputs of message passing where reassociation under
`-march=native` remains entirely live and independent of the sort question.
R1 (§3) settles that leg directly and should be run first.

### 2.4 Scope beyond aminx

Any JAX model relying on documented stable-sort tie-breaking diverges silently
through IREE, whenever its inputs produce exact ties. That is an export-safety
class, not one model's bug (§8).

## 3. The comparison ladder

"Self-consistency ring" is not defined anywhere in xtrax or aminx (verified:
zero hits in source or docs of either). Revision 1 proposed a definition that
its own table then violated in two rows. The corrected definition:

> A **ring** is a controlled comparison with exactly one independent variable
> and a shared reference input. Both sides are the same computation, so a
> disagreement is attributable to the varied axis and nothing else.

The word "ring" is retained as the user's name for the construct. Revision 1's
"closed comparison loop" imagery is dropped — nothing here is closed or a loop,
and that undischarged metaphor is what admitted two non-rings.

**R0 is not a ring.** Replaying one artifact on one input varies *nothing*. It
is the ladder's **validity gate**: if the system is nondeterministic, no ring's
disagreement localizes to anything. It runs first and gates everything.

| | Independent variable | Edits program outputs | Isolates |
|---|---|---|---|
| **R0** *(gate, not a ring)* | — replay, same artifact | no | runtime nondeterminism |
| **R1 Target** | target CPU: `native` vs `native-portable` | no | ISA-dependent codegen (FMA, vector reductions) |
| **R2a Fusion** | eager vs `jit` — same compiler, different graph | no | the model's own fusion sensitivity |
| **R2b Lowering** | `jit` XLA vs IREE — same graph, different backend | no | export/lowering fidelity |
| **R3 Probe** | cut depth — named intermediates as extra outputs | **yes** | spatial onset inside the model |

**R4 was removed.** Input class is not an independent variable *of the system* —
it varies the input, and its two sides are two invocations of some other ring.
It is a **stratification variable** applied to R1/R2/R3, not a peer rung. §2.2 is
R2b stratified by padding regime, and naming that honestly is what made the
result legible. Input-class generators are specified in §6.2.

**R1 has two executable legs, not three.** `targets.py:15` records that `wasm32`
"is compiled only. Executing it needs an emsdk-built IREE runtime." A ring needs
two *runnable* sides, so wasm32 cannot be a leg. R1 must also assert its legs
genuinely differ: `targets.py:155-165` documents that on a non-x86-64 host, LLVM
warns-and-ignores `x86-64-v2` and `native-portable` silently falls back to
host-tuned codegen, which would report "no ISA divergence" for the wrong reason.
`test_parity_multi_size.py` already reads back real `cpu_features` via
`iree-dump-module`; R1 reuses that check as a precondition.

### 3.1 Ordering

Run R0, then the non-editing rings, then R3.

The justification is **interpretability, not cost.** Revision 1 claimed cost
ordering; that was false on this spec's own facts, since §9's aminx probes
already exist and R3 there costs one export while R1 costs a recompile per
target. The real argument is that **R3's result is uninterpretable until §5.4
passes**, because instrumenting a program changes its fusion decisions. A ring
that edits the program's outputs must come after the ones that do not.

Note the column heading: *edits program outputs*, not "perturbing". R1 emits a
different binary and R2a changes the fusion graph — both perturb in the ordinary
sense. The property that singles out R3 is that it requires modifying what the
program returns.

### 3.2 The tolerance budget

Revision 1 asserted the budget should derive from measured fusion sensitivity
but gave no formula, no floor, and no scope. All three are now fixed, and the
scope limit is the important one.

Let `m_leaf` be R2a's measured `eager`-vs-`jit` divergence for a float leaf.

```
budget_leaf = max(ULP_FLOOR, SLACK * m_leaf)   ULP_FLOOR = 4 ULP, SLACK = 4.0
```

**The floor is denominated in ULP, not absolute.** Revision 2 used
`FLOOR = 1e-6` absolute, which contradicts its own justification: 1e-6 is
*tighter* than one ULP for float32 magnitudes above about 8.4, so the floor
meant to tolerate last-bit noise would have rejected it on any leaf with
ordinary-sized values. `max_ulp_diff` is already in the metric set (§5.2) and is
the correct denomination; `SLACK * m_leaf` is likewise compared in ULP.

- **The floor is mandatory and is not a defeat.** `eager == jit` bit-identical is
  the *normal* case for graphs where fusion is a no-op. Without a floor such a
  model gets budget 0 and fails on a 1-ULP difference, defeating §5.2's own
  rationale for `max_ulp_diff`. The floor is what makes the mechanism usable;
  the calibration is what adapts it upward for genuinely sensitive models.
  Consequently the budget is **flat below `ULP_FLOOR / SLACK`** — two models with
  different but tiny sensitivities get the same budget. That is intended, and
  AC-7 tests monotonicity only above the floor.
- **It is a heuristic, not an error bound.** A bound on fusion sensitivity is not
  a bound on lowering fidelity; they are different failure surfaces measured in
  the same units. A model whose length-2048 reduction measures `m ≈ 1e-2` from
  reassociation alone would be granted a budget that could mask a genuine 1e-3
  mis-lowering. R2a and R2b are therefore always **reported separately**, and
  the budget is advisory on R2b rather than authoritative. Stated plainly so
  nobody mistakes it for soundness.
- **`m_leaf` is per `(model, input_class)`** and must be re-derived under each
  stratification, since fusion sensitivity is input-dependent.

**Calibration applies to float leaves only.** For integer and bool leaves the
criterion is absolute — `exact_match_fraction == 1.0` — and no measurement
loosens it. This is not a limitation to apologise for: §2's entire bug class
lives on integer leaves, so the budget is *inert* for the case this document was
written about. Revision 1 claimed calibration was "what makes the ring
self-consistent"; that claim is withdrawn. Calibration is a convenience for
float comparisons. The discrete criterion is what catches the bug.

## 4. Non-goals

- Not a replacement for `parity.compare`. Parity asks *is it correct*;
  divergence mapping asks *where did it stop being correct*. Both ship.
- Not automatic root-causing. The fixture localizes and classifies; a human reads
  the report.
- Not a fix for the IREE sort-stability gap (§8).
- Not multi-axis plan support.

## 5. Design

### 5.1 Probe mechanism — ordinary pytree outputs

Recon established three facts that kill the obvious design and dictate the real
one:

- `Tap` is contractually an `io_callback` wrapper (`stages/boundaries.py:47-63`)
  and `topology.py:201-209` rejects **any** Tap at the export boundary
  unconditionally. Taps cannot become outputs.
- `materialize` is a single-slot mechanism: one sink, one axis
  (`MultipleMaterializeAxesError`), never a Tap.
- The composer/executor layer is **pytree-transparent**.

Therefore **a probe is a named entry in the step function's ordinary return
pytree.** No change to `topology.py`, `pipeline.py`, or the boundary protocol.

M1 confirms this survives compilation, with one trap: the runtime returns a flat
tuple carrying no key information. Probe names MUST be recovered from the
exported `out_tree`. Binding names positionally in declaration order is silently
wrong for a `dict`, which JAX flattens in **sorted key** order, and silently
right for a `NamedTuple`, which flattens in **field** order. Both appear in
practice (§9 uses a NamedTuple), so both are tested (AC-2).

### 5.2 Dtype-aware metrics

| dtype | metrics |
|---|---|
| floating | `max_abs_diff`, `max_rel_diff`, `max_ulp_diff`, `n_nonfinite_mismatch` |
| integer | `exact_match_fraction`, `first_mismatch_index`, `n_mismatched` |
| bool | `hamming_distance`, `first_mismatch_index` |

`max_ulp_diff` is computed on finite pairs only; non-finite disagreements are
counted separately rather than folded into a magnitude.

Two mismatch kinds must be distinguished; revision 2 conflated them under
"structure mismatch" and left the harder one undefined.

- **Leaf mismatch** — treedefs agree, a leaf's shape or dtype does not. Leaves
  still correspond pairwise, so `compare_pytree` **records a per-leaf failure and
  continues**. A pytree walk is a report, not an assertion, and aborting on leaf
  1 would discard the evidence that localizes the problem.
- **Treedef mismatch** — the structures themselves differ. There is no leaf
  correspondence at all, so "one `LeafDivergence` per leaf" is not even
  well-formed. `compare_pytree` **raises** `ProbeStructureError` naming both
  treedefs. Pairing by position across differing structures would silently
  compare unrelated arrays, which is worse than failing.

This is not hypothetical for the dogfood: `ProteinEdgeStageTensors.node_features_out`
is typed `jax.Array | None` (`features.py:79`), and JAX flattens `None` as an
empty node — so the output **arity is configuration-dependent** and the two sides
of a comparison can legitimately differ in structure. Detecting that loudly is
the point.

This differs from `compare`, which returns a single failed `ParityResult` for a
shape mismatch; the divergence path is deliberately stricter rather than
inheriting that behaviour.

### 5.3 Probe dependency and classification

**Probe order must be declared, not inferred.** Revision 1 said "the
immediately-upstream probe", which presupposes a chain. Two independent problems
made that unimplementable:

- Pytree flatten order is not dataflow order. Under a sorted-dict flatten,
  `final` would become the predecessor of `neighbor_indices` — suppressing the
  discrete flip and failing the dogfood on its own worked example.
- The real graph is a DAG. aminx's edge stage (`features.py:212-247`) is:

  ```
  neighbor_indices ──┬──────────────► rbf ──────────┐
                     │                              ├──► edges_concat
                     └──► encoded_positions ────────┘         │
                                                              ▼
                                        final ◄── after_norm ◄── after_w_e
  ```

  `encoded_positions` derives from `neighbor_indices` through
  `neighbor_offsets`/`edge_chains_neighbors` (`:216,222-240`) and **never from
  `rbf`**; `edges_concat = concatenate([encoded_positions, rbf])` (`:242`) is the
  join. Revision 2 of this spec wrote that chain as a straight line — committing,
  in the paragraph introducing declared dependencies, precisely the error
  declared dependencies exist to prevent. Which is the argument for the
  validation below: a human transcribing a DAG by hand gets it wrong.
- Real models are DAGs, not chains. aminx's encoder consumes *and* produces both
  `h_V` and `h_E` (`mpnn.py:165-169`), so a per-layer probe is a join with two
  predecessors. Classifying against the wrong one manufactures a false
  `INJECTED` at exactly the points where divergence merges.

The consumer therefore supplies `probe_deps: Mapping[str, tuple[str, ...]]`
naming each probe's immediate predecessors. Classification is against the
**worst predecessor** — `max` over the predecessor set — never an arbitrary one.
Dataflow edges cannot be recovered from a flat output tuple, so there is no
inference fallback; an undeclared dependency is an error, not a default.

**A declared graph must be validated, or the fixture produces confident
nonsense.** A fabricated edge is invisible to every other check. The validation
is free, because §5.5b already computes backward slices: for a declared edge
`p → q`, `slice(p)` must be a **subset** of `slice(q)` — if `q` truly depends on
`p`, every op needed for `p` is needed for `q`. A declared edge failing that
subset test is rejected with both slices named. This catches the exact error
revision 2 made: `slice(rbf) ⊄ slice(encoded_positions)`, because
`encoded_positions` does not consume `rbf`.

Classes:

- **CLEAN** — within the §3.2 budget (float) or exact (discrete).
- **AMPLIFIED** — some predecessor already diverged, and this probe's divergence
  is `>= ` that of every predecessor.
- **ATTENUATED** — some predecessor diverged and this probe's is strictly
  smaller. Renormalizing operations do this routinely; aminx has a `LayerNorm`
  between `after_w_e` and `after_norm`. Revision 1 had no class for it and its
  classifier was undefined on that input.
- **INJECTED** — **every** predecessor was bit-identical and this probe is not.
  This is the signal: it names an operation that changed semantics under
  lowering.
- **DISCRETE_FLIP** — an integer or bool leaf mismatched while every
  predecessor was bit-identical **on all leaves, discrete included**. Its own
  class because the float metrics read zero. A probe with an **empty**
  predecessor set and a mismatched discrete leaf is `DISCRETE_FLIP` by rule,
  stated explicitly rather than falling out of quantification over an empty set
  — this is the aminx case, since `neighbor_indices` is the first thing computed.

  The "discrete included" clause is load-bearing. Revision 2 quantified only over
  predecessors' *float* leaves, so a downstream probe merely **inheriting** an
  already-flipped index set would be relabelled `DISCRETE_FLIP` and announced as
  "an operation that changed semantics under lowering" — readmitting, through the
  discrete branch, exactly the propagation-as-injection error that declared
  dependencies were introduced to stop. An index mismatch whose predecessor also
  had one is `AMPLIFIED`.

**Precedence.** A probe may qualify under more than one class — typically float
divergence beyond budget *and* a discrete mismatch. Evaluate in this order and
take the first match: `DISCRETE_FLIP`, `INJECTED`, `AMPLIFIED`, `ATTENUATED`,
`CLEAN`. Discrete outranks float because a flipped index is a semantic change
while a float excursion may be tolerable, and because the discrete signal is the
one a float-only instrument cannot see.

"Amplified" is defined by the `>=` comparison above, not by the word "smoothly",
which revision 1 left with no operational meaning and which AC-3 could not test.

### 5.4 Instrumentation fidelity

**The most serious threat to R3's validity.** Adding outputs changes DCE and
fusion in both XLA and IREE. A probe that forces materialization can suppress
the very fusion that caused the divergence.

R3 is invalid unless it first reproduces what it is explaining. Before any probe
is interpreted:

1. Compute the **full §5.2 metric set over every primary leaf** for both the
   instrumented and uninstrumented artifacts. Revision 1 gated on the primary
   *float* divergence — the one quantity §1 establishes reads `0.0` for this bug
   class, so the guard's blind spot was identical to the blind spot the document
   exists to fix.
2. Float leaves must agree within `budget` (§3.2). **Discrete leaves must match
   as exact sets**: the divergence is an index set, and M2's determinism
   (bit-identical across 20 runs) makes exact equality achievable, so nothing
   weaker is justified. If R0 fails, exact equality is unachievable by
   construction and R3 must not run at all.
3. The check is **two-sided**. Instrumentation that *creates* a divergence — a
   probe forcing materialization of a value XLA would otherwise have held in
   wider precision — is as disqualifying as one that suppresses it.

On failure the fixture reports `INSTRUMENTATION_CHANGED_RESULT` and refuses to
emit a probe map. It must never report "no divergence found": a vanishing
divergence is itself the finding, and says the cause is fusion-dependent.

### 5.5 Coverage

**(a) Code coverage.** `tier1_core` measures `coverage_packages = ["xtrax"]` at
90% line / 80% branch enforced, syncing only `["dev","io"]`
(`distribution/coverage_dag.toml:16,25`). Anything importing IREE is unreachable
there. This forces the module split in §6. Note `rings.py` is *also* in package
`xtrax` and is genuinely uncoverable without a toolchain, so it carries an
explicit `coverage_omit` entry — the pure/execution split alone does not solve
this, it only moves the problem to the half that must be omitted by name.

**(b) Probe resolution.** A probe map is only as good as its resolution, and a
two-probe map on a forty-stage model is structurally indistinguishable from a
thorough one. Revision 1 proposed "StableHLO op count between consecutive
probes", which is not computable: a probe is a *return operand*, not a position,
and the op list is one arbitrary topological linearization of a DAG. There is no
"between".

The computable quantity is the **backward-slice difference**: for probes `p` and
`q` with `q` depending on `p`, `|slice(q) \ slice(p)|` — the ops needed for `q`
that were not needed for `p` — obtained by def-use walk over
`exported.mlir_module(serialized=False)`. The argument is load-bearing: it
defaults to `True` and returns a **string** (`jax/_src/export/_export.py:216`),
while `False` returns a live `ir.Module` the walk can traverse. jaxlib ships its
own `mlir/dialects/stablehlo.py`, so this does not depend on the missing IREE
stablehlo binding noted at `compile.py:12-18`.

Reported as `slice_delta` per declared edge, plus `unattributed_ops` for ops in
no slice difference. The same slices power the §5.3 edge validation, so the
metric pays for itself twice.

Three honest caveats, all of which revision 1 got wrong:

- Sibling slices **overlap** (shared subexpressions belong to both), so slice
  deltas do not partition the module and do not sum to the total.
- Counts are **static**. A 10-op `scan` body executing L times counts as 10
  (`composer.py:274`), so the most heavily executed region can look like the
  best covered.
- It measures the **instrumented** module's StableHLO, *before* IREE's own
  fusion and DCE — which is where the divergence lives.

It is therefore a **heuristic proxy for localization resolution, not a bound**.
Revision 1 called it "the true upper bound on localization precision" in a
section whose entire argument was about not overclaiming.

## 6. Module layout and interfaces

```
src/xtrax/export/divergence.py    # pure: metrics, classification, slices
src/xtrax/export/rings.py         # execution: R0-R3 runners  (coverage_omit)
tests/export/test_divergence.py   # pure-logic tests, no toolchain
tests/export/test_rings.py        # toolchain-gated + fake-injected
```

Public surface: `LeafDivergence`, `ProbeReport`, `DivergenceClass`,
`RingResult`, `compare_pytree`, `classify_probes`, `probe_resolution`,
`run_ladder`. `compare` stays unchanged for the single-array parity path.

### 6.1 Probe declaration

```python
run_ladder(
    fn, plan, abstract_inputs, concrete_inputs,
    probe_deps={"rbf": ("neighbor_indices",), "final": ("after_norm",), ...},
    input_classes=(...),
)
```

### 6.2 Input-class generators

Revision 1 listed "ties, masks, magnitudes, lengths" — a list of words, not a
specification. Concretely, each generator returns inputs plus a label:

| class | generator | in contract? | rationale |
|---|---|---|---|
| `nominal` | the artifact's own reference input | yes | baseline |
| `symmetric_geometry` | ideal α-helix / idealised coordinates | **yes** | M6 — the load-bearing tie route |
| `magnitude_extremes` | values near float32 overflow and denormal | yes | reassociation sensitivity |
| `sub_k_neighbours` | mask leaving fewer than `k_neighbors` valid | **no — refused** | M5; diagnostic only |

`symmetric_geometry` is the primary class, not `sub_k_neighbours`. Revision 2 had
that backwards. The artifact is bucket-aligned and **refuses** below the clamp
(bucket ladder `(64,128,256,512,1024,1536,2048)`, padding score-preserving iff
`L >= k_neighbors`), so `sub_k_neighbours` exercises a regime that is never
served. It is retained because a diagnostic tool should be able to characterise
inputs the artifact rejects, but it must be **labelled out-of-contract in the
report** so no verdict rests on it alone.

**Length is not a free axis** either — lengths are drawn from the ladder, not
swept.

## 7. Tasks

| # | Task | Deliverable |
|---|---|---|
| T1 | `compare_pytree` + dtype-aware `LeafDivergence`, per-leaf failure records | `divergence.py` |
| T2 | `classify_probes` over a declared DAG, max-over-predecessors, 5 classes | `divergence.py` |
| T3 | `probe_resolution` — backward-slice deltas from `mlir_module(serialized=False)` | `divergence.py` |
| T3b | `validate_probe_deps` — slice-subset check on every declared edge (§5.3) | `divergence.py` |
| T4 | R0 gate + R1 (two legs, `cpu_features` precondition) | `rings.py` |
| T5 | R2a/R2b runners + §3.2 budget derivation | `rings.py` |
| T6 | R3 runner **with** the §5.4 two-sided fidelity precondition | `rings.py` |
| T7 | `out_tree` name recovery; dict and NamedTuple | `rings.py` |
| T8 | §6.2 input-class generators | `rings.py` |
| T9 | Pure-logic tests, no toolchain, ≥90% line / ≥80% branch | `test_divergence.py` |
| T10 | Toolchain tests + fakes + `coverage_omit` entry | `test_rings.py` |
| T11 | `docs/api/export.md` — the ladder and when to escalate | docs |
| T12 | aminx dogfood (§9) | aminx PR |

Every task maps to at least one acceptance criterion and every criterion to a
task: T1→AC-1/AC-13, T2→AC-3/4/5/14, T3→AC-9, T3b→AC-15, T4→AC-16, T5→AC-7/8,
T6→AC-6, T7→AC-2, T8→AC-17, T9/T10→AC-12, T11→docs, T12→AC-10, and AC-11 is
T4/T5's API surface. Revision 2 left T4 — the R0 gate and R1's `cpu_features`
precondition, both normative — with no criterion at all, and AC-11 with no task.

## 8. Consequences to file separately

- **`check_export_safety` should flag stable-sort reliance.** M3/M4 show
  `jnp.argsort(stable=True)`, `jnp.sort` and `lax.sort_key_val` do not preserve
  XLA tie-breaking through IREE. Pairs with the unlegalizable-op work in #5092.
- **aminx PR #155's `top_k` needs an explicit tiebreak.** The fix was necessary —
  `lax.top_k` is uncompilable — but its correctness argument rests on stability
  IREE does not provide. A deterministic index tiebreak folded into the sort key
  (e.g. sort on `(-x, index)` lexicographically) removes the dependency entirely.
  This is also the negative control AC-10 requires.
- **#5093 should be updated**, not closed: FMA is retired for the decoding-order
  leg only (§2.3); sort stability is a demonstrated mechanism whose applicability
  to that artifact is still unmeasured.

## 9. Dogfood on aminx

aminx supplies the probes and the schema, so no model changes are needed:

- `features.py:109` `forward_edge_stages()` returns `ProteinEdgeStageTensors`,
  documented verbatim as *"Intermediate edge tensors for parity diagnosis"*.
  Production `__call__` calls it and discards all but four. It has **nine**
  fields — seven edge-stage tensors plus `node_features_out` and `prng_key` — of
  which the seven are the probe set. Revision 2 called it "seven intermediates",
  conflating the probe set with the tuple.
- `node_features_out` is typed `jax.Array | None` (`features.py:79`) and JAX
  flattens `None` as an empty node, so the **output arity is
  configuration-dependent**. This is the concrete reason §5.2 must separate
  treedef mismatch from leaf mismatch, and it is reachable in the dogfood rather
  than theoretical.
- It is a `NamedTuple`, so it flattens in field order, which *coincides* with a
  topological order of the real DAG — a coincidence the fixture must not rely on
  (the DAG is not a chain; see §5.3), which is why declared dependencies are
  required regardless.
- Encoder/decoder layers are accessible lists (`model.encoder.layers[i]`), so
  per-layer probes need only an external re-drive of the loop.
- `src/aminx/parity/evidence.py` already defines `EvidenceMetricRecord` /
  `EvidencePointRecord`; the fixture emits into that schema.

**Acceptance:** the ladder, run on the real `proteinmpnn_v_48_020` artifact under
the §6.2 input classes, must report `DISCRETE_FLIP` on `neighbor_indices` for
`symmetric_geometry` — the in-contract class — and **not** for `nominal`. It must
additionally *answer §2.3's open question*: whether the exported artifact
computes its own kNN at all (§2.1's precomputed-features path contains no sort),
and if it does, whether its reference input carries ties. Those are the two
competing explanations for #5093, and the dogfood is what distinguishes them.

## 10. Acceptance criteria

- **AC-1** `compare_pytree` returns one `LeafDivergence` per leaf with
  dtype-appropriate metrics; a structure mismatch yields per-leaf failure records
  and continues (§5.2), asserted on a pytree with one mismatched leaf among
  matching ones.
- **AC-2** Probe names are recovered from `out_tree`. Two tests: a **dict** whose
  sorted key order differs from declaration order, and a **NamedTuple** whose
  field order differs from sorted order. Both assert names bind to the right
  arrays.
- **AC-3** `classify_probes` labels a DAG containing a join with one clean and
  one diverged predecessor as `AMPLIFIED` (max-over-predecessors), **not**
  `INJECTED`. This is the OBJ-15 regression and is the reason the ordering is
  declared.
- **AC-4** `ATTENUATED` is produced for a probe whose divergence is strictly
  smaller than its predecessor's.
- **AC-5** `DISCRETE_FLIP` is raised for a mismatched int leaf with
  bit-identical predecessor floats, **and** for the empty-predecessor case.
- **AC-6** R3 reports `INSTRUMENTATION_CHANGED_RESULT` in both directions —
  suppression and creation — and emits no probe map in either.
- **AC-7** `budget_leaf` satisfies §3.2: two fixtures with sensitivities
  `ULP_FLOOR/SLACK < m₁ < m₂` yield `budget₁ < budget₂`, and a fixture with
  `m = 0` yields exactly `ULP_FLOOR`. Monotonicity is asserted **only above the
  floor** — revision 2 demanded it unconditionally, which `max(FLOOR, SLACK*m)`
  violates for any two sensitivities below `ULP_FLOOR/SLACK`, so the criterion
  contradicted its own formula. (Revision 1's "is a function of, not a constant"
  was separately unfalsifiable — `1e-5 + 0*m` satisfied it.)
- **AC-8** Integer leaves are judged by exact match regardless of any budget —
  asserted by running a calibrated ladder with a deliberately huge float budget
  and confirming a single flipped index still fails.
- **AC-9** `probe_resolution` reports `slice_delta` per declared edge and
  `unattributed_ops`, and its docstring states the three §5.5b caveats.
- **AC-10** aminx dogfood: `DISCRETE_FLIP` on `neighbor_indices` under
  `symmetric_geometry` (in-contract; revision 2 named the refused
  `sub_k_neighbours` class here), **no** `DISCRETE_FLIP` under `nominal`, and —
  the **negative control**, without which this AC is unfalsifiable — **no**
  `DISCRETE_FLIP` under `symmetric_geometry` after applying §8's explicit index
  tiebreak to the same model. A fixture that hardcodes the expected answer passes
  the first clause and fails the other two.
- **AC-11** R0/R1/R2 runners accept no probe argument — an **API-surface** claim,
  not a guarantee. Since a probe is an ordinary pytree entry (§5.1), nothing
  type-level distinguishes an instrumented callable from a plain one, so a caller
  can still pass one. Stated at its true strength rather than implying more.
- **AC-12** `divergence.py` imports nothing from `iree` (asserted by test) and
  reaches ≥90% line / ≥80% branch under `tier1_core` with no export extra.
- **AC-13** A **treedef** mismatch raises `ProbeStructureError` naming both
  treedefs, while a **leaf** mismatch under a matching treedef produces a
  per-leaf failure record and continues (§5.2). Both asserted, including the
  `None`-valued-field case that makes aminx's output arity vary.
- **AC-14** A probe whose predecessor *also* had a discrete mismatch is
  `AMPLIFIED`, **not** `DISCRETE_FLIP` — the propagation-through-the-discrete-branch
  regression. Plus: a probe qualifying under two classes resolves by the §5.3
  precedence order.
- **AC-15** `validate_probe_deps` rejects a declared edge whose slice-subset
  relation fails, naming both slices. Asserted on this spec's own revision-2
  error — a declared `rbf → encoded_positions` edge for the aminx graph, which
  must be rejected because `encoded_positions` does not consume `rbf`.
- **AC-16** R0 reports non-determinism as a hard gate (a deliberately
  nondeterministic fake artifact prevents any ring from running), and R1 refuses
  to interpret its legs when the `cpu_features` readback shows them identical —
  the silent-fallback case `targets.py:155-165` documents.
- **AC-17** Each §6.2 generator is labelled in the report with its
  in-contract/out-of-contract status, and `sub_k_neighbours` results are marked
  out-of-contract so no verdict rests on them alone.

**Merge gate, not an acceptance criterion:** `just audit-deterministic` exits 0
*and* its log contains no `FAILED` (the known audit-masking trap). It is
satisfied or broken by unrelated work and carries no information about whether
this feature works, so it is listed here rather than above.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Instrumentation suppresses or creates a divergence | §5.4 two-sided precondition; AC-6 |
| Probe map looks authoritative but is low-resolution | `slice_delta` + stated caveats; AC-9 |
| `rings.py` drags the tier1 aggregate | Explicit `coverage_omit`; AC-12 scoped to `divergence.py` |
| Calibrated budget masks a real mis-lowering | R2a/R2b always reported separately; budget advisory on R2b; discrete criterion absolute (§3.2) |
| R1 legs silently identical on a non-x86-64 host | `cpu_features` readback precondition (§3) |
| Declared probe DAG is wrong or incomplete | `unattributed_ops` surfaces unreachable regions; an undeclared dep is an error, not a default |
| §2.3's open question quietly forgotten | §9 acceptance requires the dogfood to answer it |
