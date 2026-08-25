# Spec-challenger memo r1 — 260825_xtrax-cse-runtime-opt (DRAFT v2)

Reviewer: SPEC-CHALLENGER round 1. Every code-grounded claim below was verified against the
working tree this session. Anchors cited as file:line. Spec citations are to
`.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md`.

---

## Blockers

### OBJ-R1-01 — BLOCKER — §4.3 Component C construction semantics
**Objection:** "constructs DedupSpec with sampled unique_indices/index_map" is either incorrect
or self-defeating, and no third reading is offered. Literal reading: `index_map` built from the
sample covers only sampled positions, but `DedupGather` scatter requires `(N,)` coverage
(`src/xtrax/tiling/dedup.py:54`); `DedupSpec.__post_init__` (`src/xtrax/tiling/dedup.py:73-90`)
never checks `len(index_map) == N`, so the malformed spec passes AC7's stated check and fails
silently at dispatch. Charitable reading (unsampled rows become singleton slots): k ≥
N − max_sample_rows, so for any N ≳ 4350 the k>256 guard returns None regardless of true
duplication — the component never fires in the large-batch regime that motivates it. Worse,
sample-based k UNDER-estimates population k, so the guard under-fires precisely when padding
waste is worst. Either way D4's rationale ("full hash O(N) transfer defeats purpose") is
contradicted: exact spec construction requires touching all N rows, i.e., the rejected O(N)
pass is unavoidable whenever synthesis succeeds.
**Evidence:** spec.md:179-181, 187-189, 201-204; src/xtrax/tiling/dedup.py:54,73-90;
decision table D4 spec.md:41.
**Confidence:** 0.85

### OBJ-R1-02 — BLOCKER — §4.3 collision policy vs specified signature
**Objection:** The collision policy requires the synthesizer to know the caller-declared
DedupSpec set, but `synthesize_dedup_spec(batch_leaves, *, axis, threshold,
max_sample_rows)` carries no such parameter, so `DedupSynthesisCollisionError` is
unimplementable against the specified API and AC11 has no implementable When-step. As written,
the only entity positioned to compare is the caller, which reduces the "fail-loud" guarantee to
documentation. The signature must gain an `existing_specs`-style parameter (or the collision
check must move into a merge helper the spec defines).
**Evidence:** spec.md:169-176 (signature), 195-198 (collision policy), 220 (AC11).
**Confidence:** 0.9

## Majors

### OBJ-R1-03 — MAJOR — §4.3 collision policy vs plan.py silent last-wins
**Objection:** Even with OBJ-R1-02 fixed, the design creates two different collision behaviors
depending on entry path: synthesized-vs-caller raises, while caller-vs-caller duplicates still
silently last-win inside `plan()` (`src/xtrax/tiling/plan.py:215`, untouched per N3). A caller
who merges spec lists manually (bypassing the synthesizer's check) regresses to silent
last-wins with no diagnostic. This contradicts G4's "fail-loud conventions" claim: the loudest
hazard path (duplicate axis_name anywhere) stays silent for the pre-existing entry path while
the new path errors — inconsistent failure semantics inside one subsystem.
**Evidence:** spec.md:30 (N3), 195-198; src/xtrax/tiling/plan.py:215; recon memo lines 77-80, 186-187.
**Confidence:** 0.85

### OBJ-R1-04 — MAJOR — §4.2.8 expected hit regime contradicts gate architecture
**Objection:** The claimed profitable "within-pass" regime does not exist as described:
candidate_smoke executes the candidate in a fresh subprocess under `JAX_PLATFORMS=cpu`
(`src/xtrax/loop/candidate_smoke.py:107-111,158-166`), checkified_execution resolves a NEW
callable object from `handoff.path` and wraps it under SafetyManager
(`src/xtrax/loop/checkified_execution.py:92-97`), and guarded_evaluate invokes a distinct
in-process evaluator (`src/xtrax/loop/closure_lock.py:233-262`). One in-process wrapper cannot
observe calls across a subprocess boundary or a path-reloaded module object; N2 declines the
cross-process persistence that would be needed; and the smoke gate's cpu-only environment stamp
can never match the main backend stamp by the spec's own §4.2.2 rule. The surviving hit regime
is only unchanged-candidate retries, which halves the advertised value of component B and
should be restated honestly.
**Evidence:** spec.md:157-163 (§4.2.8), 29 (N2); files above; recon lines 57-62 repeat the
same wrong premise ("smoke/checkified steps execute the SAME callable").
**Confidence:** 0.8

### OBJ-R1-05 — MAJOR — §4.2.2/D5/A1 load-bearing unknown: is hash(ClosedJaxpr) const-value-sensitive?
**Objection:** The entire staleness story hinges on an unstated property: if closure constants
literalize into the traced jaxpr and `ClosedJaxpr.__hash__` is value-sensitive, then a mutated
closure produces a DIFFERENT key (a miss, never a stale hit), making AC6 unconstructible as
written and the mandatory-salt ceremony a solution to a non-problem; if it is
value-insensitive, salt plus spot-check are the only defense, yet `spot_check_every` defaults
to 0 and `salt` defaults to "" with documentation-only enforcement. A1/R2 verify hash existence
and stability, never value sensitivity. P1 must empirically pin this property before the salt
design or AC6 can be finalized; as drafted the centerpiece staleness AC may be untestable or
redundant.
**Evidence:** spec.md:110-119 (admission/key), 41 (D5), 214-215 (AC6), 251 (A1), 235-236 (R2).
**Confidence:** 0.7

### OBJ-R1-06 — MAJOR — §4.2.2/N5 environment stamp does not address sharding provenance
**Objection:** Research Q5.6 raised that a cached array carries its compute-time sharding; the
stamp `(jax.__version__, jaxlib, backend kind, device kind)` permits hits across device
INSTANCES of the same kind, so in a multi-GPU process a buffer resident on device A is handed
to a consumer compiling for device B — hidden cross-device copies or device-mismatch/donation
errors. N5 declares "single-device semantics only" but nothing in the key or wrapper enforces
that scope (no device-index in the stamp, no `len(jax.local_devices())==1` assertion), and
output sharding inferred from compute-time argument shardings can mismatch a consumer using
different input pspecs even on one device. The hazard the research flagged is neither mitigated
nor scoped out with a mechanism.
**Evidence:** spec.md:130-134 (stamp), 32 (N5); research memo Q5.6 (lines 88-89);
docs.jax.dev jit (sharding inference), per research citation.
**Confidence:** 0.75

### OBJ-R1-07 — MAJOR — §4.2.4/D3 advisory gate: unimplementable on decorator path and blind to the slowdown it warns about
**Objection:** The wrap-time cost ratio requires abstract inputs, but the decorator form
`memoize_jaxpr(fn)` has none, and `lowered_memory_estimate` is shape-driven
(`src/xtrax/tiling/estimators.py:60-97`), so the advisory silently cannot fire in exactly the
ergonomic path most likely to be misused. In the adversarial case (per-call hash transfer cost
persistently exceeds op cost) the user gets slower code while believing the opposite, and the
specified telemetry cannot reveal it: `.memo_stats` fields (hits/misses/evictions/bytes_cached/
last_hit_age/spot_check_mismatches) include NO wall-time or hash-cost counters, contradicting
D3/C2's promise of "observed hit-rate/wall-time telemetry". Once-per-site RuntimeWarning is
also trivially suppressed by standard filters. R1 floats a hard floor but leaves the honesty
gap open.
**Evidence:** spec.md:107, 138-139 (§4.2.4), 140-141 (§4.2.5 field list), 40 (D3), 233-234 (R1);
src/xtrax/tiling/estimators.py:27-59,60-97.
**Confidence:** 0.85

### OBJ-R1-08 — MAJOR — §4.1 equivalence algorithm undercounts vs its own XLA-correspondence claim
**Objection:** Hashing eqns as `(primitive, params, invar ids)` and grouping with union-find
over those hashes cannot reproduce "what XLA actually merges post-lowering": XLA's hlo_cse
iterates to fixpoint, so merging duplicate `sin` eqns makes downstream `mul` eqns
operand-identical and mergeable too, whereas a single pass over raw hashes reports only the
first generation (operand ids of transitive duplicates still differ pre-merge). AC1's own
example demonstrates the undercount: XLA would eliminate both the duplicated sins AND the two
`*2.0` muls, but the specified algorithm and AC1 expect one class of two eqns and stay silent
on the muls. Fixpoint rewriting of operand ids after each merge round is required and absent.
**Evidence:** spec.md:74-78 (algorithm + correspondence claim), 210 (AC1); research memo Q1.3
(hlo_cse.cc semantics, line 19).
**Confidence:** 0.7

### OBJ-R1-09 — MAJOR — §4.2.1/D2 B3′ screen: detectability account is technically wrong and AC3 is untestable as worded
**Objection:** The screen's stated false-negative classes misdescribe tracing: custom_jvp/vjp
PRIMAL bodies are inlined into `make_jaxpr` output (only derivative/backward rules escape, and
those never execute in forward value-memoization), while host callbacks appear as named
callback primitives that ARE structurally screenable — so D2's justification for rejecting
static-only admission rests on a wrong premise and the detectable/undetectable split in §4.2.1
is drawn in the wrong places (what actually escapes is Python-side state consumed outside the
trace, time/I/O, and objects with unstable traced representations). Separately, AC3's "function
never called" is false for any wrapped fn because `make_jaxpr` executes the Python body once
under tracers at wrap time; the AC must be reworded to "never concretely executed" with a spy
on concrete dispatch, else the implementer must fight the assertion.
**Evidence:** spec.md:115-119 (§4.2.1), 39 (D2), 212 (AC3); JAX tracing semantics (body
executes under make_jaxpr), per docs.jax.dev referenced in research memo.
**Confidence:** 0.7

## Minors

### OBJ-R1-10 — MINOR — §4.2.2 weak_type vs mandated reuse of `update_array_digest`
**Objection:** The key spec requires `(shape, dtype, weak_type, byte-content)` per leaf but
mandates reuse of `update_array_digest`, which folds dtype.name + shape + bytes and knows
nothing of `weak_type` (an `np.ndarray`-typed parameter; a jax.Array must first be
host-materialized). Composing the two requires an unspecified extension (folding weak_type
into the digest stream separately), so "MUST reuse rather than inventing" oversells the fit.
**Evidence:** spec.md:120-127; src/xtrax/run/zarr_integrity.py:65-72.
**Confidence:** 0.8

### OBJ-R1-11 — MINOR — §6 phased delivery orphans AC13/AC14; AC13 needs an unstated seam
**Objection:** P1 schedules only AC3–AC6 and AC9, leaving component-B behaviors AC13
(cross-stamp isolation) and AC14 (spot-check corruption) assigned to no phase. AC13 also
presumes two backends/stamps exist in CI or that the stamp is injectable for tests; neither a
matrix requirement nor an injection seam is specified, so the AC risks permanent skip.
**Evidence:** spec.md:226-229 (phases), 222 (AC13), 223 (AC14).
**Confidence:** 0.85

### OBJ-R1-12 — MINOR — AC9 is unfalsifiable as written
**Objection:** "No test flake attributable to async dispatch" names no observable Then-state;
absence of flake cannot be asserted by a single test run. Reword to a positive invariant
(e.g., stored entries are always ready-buffer identities and a forced immediate second call
returns equal values) so CI can actually evaluate it.
**Evidence:** spec.md:218 (AC9).
**Confidence:** 0.9

### OBJ-R1-13 — MINOR — AC12 lint test is vacuous today and alias-blind
**Objection:** No production code applies `memoize_jaxpr` yet (component B is greenfield), so a
grep over controller wiring passes trivially forever until someone wires it, at which point a
grep for the wrapper name misses aliased imports (`from xtrax.inference.memo import
memoize_jaxpr as cache`). Documentation-plus-vacuous-lint is weaker enforcement than §4.2.7
implies for a rule the spec itself labels correctness-critical.
**Evidence:** spec.md:152-157 (§4.2.7), 221 (AC12); verified negative: no memo imports exist in
controller/.
**Confidence:** 0.75

### OBJ-R1-14 — MINOR — §4.2.5 spot-check mismatch lifecycle unspecified
**Objection:** "spot_check_mismatches (must remain 0; nonzero → raise MemoStalenessError)"
does not say WHEN the raise happens (at detection? on next call?), whether the corrupted entry
is evicted, or whether every subsequent Kth call re-raises on a poisoned counter. AC14's
deterministic error depends on these choices. Operational meaning of "forced value corruption"
is likewise undefined (tests must reach into the private LRU and swap an entry — acceptable,
but say so).
**Evidence:** spec.md:140-145, 223 (AC14).
**Confidence:** 0.8

### OBJ-R1-15 — MINOR — §4.2.3 block_until_ready-before-store needlessly serializes misses
**Objection:** Per the async-dispatch contract cited by the research (Q5.5), retaining
not-yet-ready futures is safe; blocking before insertion converts every miss into a
synchronous call and destroys pipelining for callers who never inspect outputs eagerly. The
spec presents blocking as required for safety ("correctness-adjacent"), which its own citation
does not support; it is a performance-policy choice that belongs behind the cost advisory, not
a correctness rule.
**Evidence:** spec.md:135-137; research memo Q5.5 (lines 86-87).
**Confidence:** 0.65

### OBJ-R1-16 — MINOR — §4.3 heterogeneous refusal narrows the component below its motivating gap
**Objection:** DedupSpec's own docstring motivates dedup for heterogeneous axes
(`src/xtrax/tiling/dedup.py:2-4`), and `AxisSpec.heterogeneous`/`dedup_eligible` are
independent flags (`src/xtrax/tiling/plan.py:39-40,51-52`), so the Rule-2 gap the recon targets
(`plan.py:433-438`) may be dominated by exactly the ragged rows v1 refuses. Combined with
OBJ-R1-01's large-N emptiness, the honest profitable envelope of component C is narrow and the
spec never states it.
**Evidence:** spec.md:203-204; src/xtrax/tiling/dedup.py:2-4; src/xtrax/tiling/plan.py:39-40,51-52,433-438.
**Confidence:** 0.6

---

## Summary judgment

The spec's strongest sections are the research-grounded key design (weak_type, bit-exactness,
donation bidirectional hazard, numeric spot-check) and honest R-item accounting. The fatal
concentration is Component C: sample-to-spec construction (OBJ-R1-01) and the collision API
(OBJ-R1-02/03) need redesign before planning. Component B needs one empirical fact pinned
(ClosedJaxpr const-value sensitivity) and an honest restatement of its reachable hit regime;
Component A needs fixpoint equivalence or a weaker correspondence claim.
