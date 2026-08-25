# Spec-challenger memo r2 — 260825_xtrax-cse-runtime-opt (DRAFT v3)

Reviewer: SPEC-CHALLENGER round 2. Method: every v3 code anchor re-read this session
(dedup.py, plan.py, candidate_smoke.py, checkified_execution.py, closure_lock.py,
zarr_integrity.py — all cited anchors verified accurate). JAX behavior claims re-tested
empirically in `.venv/bin/python` (JAX 0.10.2); experiment outputs cited inline as [E*].
Spec citations are to `.praxia/docs/specs/260825_xtrax-cse-runtime-opt-spec.md`.

---

## Part 1 — Disposition audit (OBJ-R1-01..16)

| OBJ | Verdict | One-line reason |
|-----|---------|-----------------|
| R1-01 | RESOLVED | Two-stage redesign is coherent: sample decides WHETHER to pay O(N), exact pass builds full `(N,)` coverage; self-assert is genuinely needed (`__post_init__` checks k/index_map range but not `len(index_map)==N`, dedup.py:73-90) and present (spec:199-202). Residual None-overload ambiguity charged to OBJ-R2-07, not to this fix. |
| R1-02 | RESOLVED | `existing_specs` parameter added to the signature (spec:191); AC11 has a concrete When-step; caller-spec-untouched guarantee stated. |
| R1-03 | PARTIAL | `merge_dedup_specs` gives ONE loud semantic for paths that USE it, but `plan()`'s silent last-wins dict stays untouched until a post-v1 backlog item (spec:215-219); the inconsistent-failure-mechanism objection is scoped, not resolved. Helper's raised error type is also unnamed. |
| R1-04 | RESOLVED | §4.2.8 restates the hit envelope honestly and matches source: cpu-only subprocess smoke (candidate_smoke.py:107-111,158-166), path-reloaded callable under SafetyManager (checkified_execution.py:92-97); A2 corrected. |
| R1-05 | PARTIAL | The demanded experiment was really run and reproduces ([E1]: same-fn retrace returns the SAME ClosedJaxpr object; 3 identical-text traces → 3 distinct hashes; mutated closure visible only via fresh callable). But the design conclusion (D6 text-digest identity) overreaches the evidence: `str()` is deterministic yet NON-INJECTIVE (see OBJ-R2-02), so the identity mechanism is defective even though the pinned facts are correct. |
| R1-06 | RESOLVED | Scope now mechanically enforced: wrap-time `len(jax.local_devices())==1` assertion (N5, spec:40-43) plus device index in the stamp (spec:142); sharding provenance is out of scope by construction. |
| R1-07 | PARTIAL | Decorator-path concern genuinely fixed (advisory is measured, works without abstract inputs; counters + AC16 added). But the measured numbers are mis-attributed under async dispatch (see OBJ-R2-03), so the honesty objective is reintroduced as a measurement-validity defect. |
| R1-08 | PARTIAL | Fixpoint iteration is now specified and mirrors hlo_cse semantics (spec:84-88), but the algorithm still cannot achieve AC1's OWN example: the two `mul` eqns carry distinct Literal objects (verified, OBJ-R2-01), so operand-id rewriting alone never merges them. Letter advanced, mechanism incomplete. |
| R1-09 | RESOLVED | Blind spots corrected to the right triple (out-of-trace state, time/I/O, unstable representations); derivative-rule clarification is right for forward memo; AC3 reworded to "never concretely executed" with dispatch spy. |
| R1-10 | RESOLVED | Explicit extension folds weak_type + dtype name into the digest stream (spec:139-141); verified `update_array_digest` alone lacks both (zarr_integrity.py:65-72). |
| R1-11 | RESOLVED | AC13/AC14 assigned to P1 with `_stamp_override` named as the seam (spec:250-251); injection mechanism now exists. |
| R1-12 | RESOLVED | AC9 rewritten as positive invariant (ready buffer + immediate synchronous second call allclose-equal, spec:238). |
| R1-13 | RESOLVED | Alias-resolving AST lint specified; vacuous-green-until-wiring documented in the test itself (spec:166-170). Transitive-wrap detection remains AST-hard, acceptable at stated scope. |
| R1-14 | RESOLVED | Lifecycle fully specified: immediate raise, entry evicted, counter poisoned until `.memo_reset()`, AC14 states the private-LRU operational definition (spec:157-162, 243). |
| R1-15 | RESOLVED | `block_on_miss` made an explicit performance policy; correctness claim withdrawn; retention-safety citation retained (spec:145-149). New mode interactions charged to OBJ-R2-05. |
| R1-16 | RESOLVED | Typed refusal + honest narrow envelope stated in §4.3 (spec:207-210); heterogeneity gap recorded as R3 with the padding-normalization unlock. |

Tally: 12 RESOLVED, 4 PARTIAL (R1-03, R1-05, R1-07, R1-08), 0 FAILED.

---

## Part 2 — Fresh attack on v3 new content

### OBJ-R2-01 — BLOCKER — §4.1/AC1 fixpoint cannot merge the mul pair: Literal operands are not unified by value
**Objection:** The fixpoint hashes eqns as `(primitive.name, params, operand-class representatives)`
and rewrites VAR ids through union-find (spec:84-88). In an actual jaxpr, each occurrence of a
literal constant is a DISTINCT Literal object: for `sin(x)*2.0` duplicated, `mul[0].invars[1]` and
`mul[1].invars[1]` are two separate Literals holding equal `TypedNdArray(2.)` [E1: object ids
139123550988240 vs 139123550988960]. Var-id union-find therefore never unifies them, the two muls
keep distinct fingerprints through EVERY fixpoint round, and AC1 ("TWO duplicate classes reported",
spec:230) is unachievable by the specified algorithm. Fix: canonicalize Literal operands into
classes keyed by (dtype, value-bytes) before fingerprinting — one sentence, but absent.
**Evidence:** spec.md:84-91, 230; experiment [E1] this session (invar inspection of
`make_jaxpr` output, JAX 0.10.2).
**Confidence:** 0.9

### OBJ-R2-02 — BLOCKER — D6/A1′ program identity is non-injective: `str(ClosedJaxpr)` elides ARRAY constant values
**Objection:** A1′ generalizes from "three equal-text traces → identical `str()`" to text-digest
identity (D6, spec:54, 135-137, 270-273). Empirically the inference direction is invalid:
`str(ClosedJaxpr)` prints array constants as TYPE-ONLY free vars — `{ lambda a:f32[64]; b:f32[64].
let c:f32[64] = add b a in (c,) }` — so two programs whose closures hold DIFFERENT arrays
(zeros-with-spike at index 0 vs index 63; two distinct 32×32 arrays) produce BYTE-IDENTICAL text
[E2]. Scalar consts DO inline (`add a 1.0:f32[]` vs `2.0`) [E3], so the A1′ mutation experiment
happened to use the only const kind the text can see. Consequence: `program_digest` cannot
distinguish any two wrapped callables whose captured arrays differ, and NO AC pins
const-sensitivity of the key. Today this is masked only by an UNSTATED assumption that the LRU is
strictly per-wrapper-instance; any later shared/module-level cache turns it into silent
wrong-result hits. Fix: fold `ClosedJaxpr.consts` values (via the leaf-digest primitive) into
`program_digest`, and amend A1′ to state determinism WITHOUT injectivity.
**Evidence:** spec.md:54, 134-137, 270-273; experiments [E2]/[E3] this session.
**Confidence:** 0.85

### OBJ-R2-03 — MAJOR — §4.2.4 measured advisory mis-attributes time under async dispatch
**Objection:** `cum_op_seconds` = "underlying call wall time" (spec:150-151). Under JAX async
dispatch that wall time is dispatch overhead, not compute; completion sync lands LATER — often
inside the NEXT call's key-build, whose input digests require device→host transfers that drain the
queue. Effects: (i) in `block_on_miss=False` mode op-time collapses toward ~0 and
`slow_ratio_warn` fires spuriously, telling a well-configured user caching makes them slower;
(ii) in default mode the ratio depends on whether the pre-store `block_until_ready` sits inside or
outside the timed region — unspecified; (iii) producer-stall latency caused by OTHER layers
surfaces as inflated `cum_hash_seconds`. AC16 (spec:245) therefore validates the plumbing, not the
attribution. Spec must define timer brackets: op time includes readiness-blocking (or device-event
timing); hash time excludes sync waits attributable to previously enqueued work.
**Evidence:** spec.md:150-154, 155-156, 245.
**Confidence:** 0.75

### OBJ-R2-04 — MAJOR — §4.2.2 key completeness for non-array pytree leaves is undefined
**Objection:** The key is `(program_digest, pytree structure, per-leaf digests, salt, stamp)`
(spec:134) but the only defined digest primitive takes `np.ndarray` (zarr_integrity.py:65-72);
the weak_type extension (spec:139-141) is likewise array-scoped. Meanwhile Python scalars are
LEGAL and COMMON arguments: `make_jaxpr(f)(x, 3)` converts the int to a TRACED input
(`b:i32[]`), NOT a literal [E4], so the program text does NOT discriminate scalar values. A
natural implementation digests only ndarray leaves → two calls differing solely in a Python
float/int/bool/str argument share structure and array digests → silent wrong-result HIT, the
exact failure class this component exists to prevent. Fix: define canonicalization for
non-array leaves (type tag + repr/value bytes; strings NFC per house convention) or restrict
admission to array-only pytrees with a typed error.
**Evidence:** spec.md:134-141; experiment [E4]; zarr_integrity.py:65-72.
**Confidence:** 0.75

### OBJ-R2-05 — MAJOR — §4.2.3/§4.2.5 pipelining mode breaks the spot-check contract and the "eviction bounds memory either way" claim
**Objection:** With `block_on_miss=False`, stored entries are unresolved futures. Then: (i) the
Kth-call spot-check (spec:157-161, AC6 spec:235) must numerically compare a cached value that may
not be ready — awaiting it inside spot-check reintroduces exactly the blocking the mode exists to
avoid, and is unspecified; (ii) evicting a pending future drops the reference while compute is
still in flight, making `bytes_cached` and `max_entries` meaningless as memory measures — directly
contradicting §4.2.3's "eviction bounds memory either way (max_entries vs
device_memory_budget())" (spec:147-149); (iii) original execution and spot-check recompute can run
CONCURRENTLY, doubling in-flight working set with no bound stated. AC9 covers only the blocking
mode (spec:238). Define: await-on-compare policy, pending-entry eviction semantics, and delete or
qualify the "either way" memory claim.
**Evidence:** spec.md:145-149, 155-162, 235, 238.
**Confidence:** 0.7

### OBJ-R2-06 — MINOR — `_stamp_override` has no production containment
**Objection:** The test seam is a public-constructor field of a frozen dataclass distinguished
only by its underscore name (spec:118, 242). Nothing prevents a config-driven policy builder from
shipping an injected stamp, after which unrelated environments share keys and cross-stamp
collisions become silent stale hits — the failure AC13 guards against, reintroduced by config.
Require a guard (raise unless an explicit debug/test env var is set) or strip the field in any
documented production-policy factory.
**Evidence:** spec.md:118, 242, 250-251.
**Confidence:** 0.7

### OBJ-R2-07 — MINOR — None-return overload erases the difference between "no duplication", "too costly", and "below threshold"; metadata channel undefined
**Objection:** Three outcomes return None (below threshold spec:196-198; exact k > max_unique_k
spec:202-203; all-unique AC8 spec:237) with no observable distinction, yet AC10 requires "O(N)
cost documented in RETURNED METADATA" (spec:239) — bare `DedupSpec | None` (spec:192) has no
metadata slot, so AC10 is untestable as written; §4.3's budget bullet cites the same phantom
"return metadata" (spec:220-222). Separately the sample stage never specifies WHICH rows are
sampled; under the stated contiguous-row envelope, tail-concentrated duplication is invisible to
a prefix sample, a false-negative direction the docstring should state. Define a result/exception
payload carrying stage + spent-transfer bytes, and pin the sampling rule.
**Evidence:** spec.md:184-203, 220-222, 237, 239.
**Confidence:** 0.75

### OBJ-R2-08 — MINOR — Deferred first-call screening: unlatched failure repeats traced side effects mid-loop
**Objection:** On the decorator-without-shapes path the screen runs at FIRST REAL CALL
(spec:131-133). The abort contract is stated ("aborts the call ... before any caching begins"),
but: (i) the traced execution itself runs the Python body with real shapes BEFORE the purity
verdict — for precisely the stateful/I/O functions the screen targets, that is one extra host-side
effect occurring mid-user-loop rather than at wrap time, never acknowledged; (ii) whether a failed
screen LATCHES is unspecified — if each subsequent call re-screens, an impure fn accumulates
traced executions per call. Specify: poison-after-first-`MemoImpurityError` until rewrap, and
document the extra traced execution on the deferred path.
**Evidence:** spec.md:130-133, 232.
**Confidence:** 0.65

---

## Summary judgment

The v3 revision is substantially responsive: 12/16 clean resolutions, the two-stage synthesis and
the honest hit-envelope rewrite are genuine fixes, and A1′'s experiments replicate exactly [E1].
But both blockers concentrate in the NEW content, and both are empirically demonstrated on the
repo's own venv: the fixpoint algorithm cannot pass AC1's flagship example without literal-value
canonicalization (OBJ-R2-01), and the D6 text-digest identity is provably blind to array
constants (OBJ-R2-02) — the load-bearing empirical fact A1′ missed is that `str()` determinism
says nothing about injectivity. Component B additionally needs async-honest timing brackets
(OBJ-R2-03) and non-array-leaf key completeness (OBJ-R2-04) before P1 planning. Targeted,
verifiable revisions; no structural retreat required.

Experiment log (this session, `.venv/bin/python`, JAX 0.10.2):
[E1] duplicated sin/mul jaxpr: mul invars carry two distinct Literal objects, equal value.
[E2] distinct 64-elem / 32×32 array consts → byte-identical `str(ClosedJaxpr)`.
[E3] scalar closure consts inline in text (1.0 vs 2.0 distinguishable).
[E4] Python int arg becomes traced input `b:i32[]`; program text identical for values 3 and 4.
[E1-supplement] A1′ replication: same-fn retrace returns same object; 3 identical-text traces →
3 distinct `hash()`es; mutated closure visible only through a fresh callable.
