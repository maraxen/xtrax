---
title: memoize_jaxpr screen hardening
description: 'Per-signature purity/donation screening with static non-array leaves and kwargs, uncapped sub-jaxpr traversal for the impurity screen, and kwarg-faithful spot-check replay (#5214, #5215, #5216)'
status: draft
task_id: 260922_memo-screen-hardening
date: '260922'
backlog_ids: '5214, 5215, 5216, 5231'
adversarial_review: 'r1 REVISE (1 BLOCKER, 2 MAJOR in scope + 1 MAJOR, 4 MINOR) — dispositions §8'
revision: 2
---
# memoize_jaxpr screen hardening

## Revision history

| Rev | Date | Change |
|---|---|---|
| r1 | 260922 | Initial draft. Probes P1-P4 measured before drafting. |
| r2 | 260922 | Challenger round 1 returned REVISE; every objection was accepted (§8). The main changes: <br>- The ABSTRACT→STATIC fallback retries on **any** exception when a traced scalar is present (the BLOCKER). The r1 exception-taxonomy premise is false in JAX 0.11.1. <br>- `bool` scalars and scalars whose type is not exactly `int`/`float` are always held static. <br>- String and bytes keys are exact; the NFC normalization, which caused collisions, is removed. <br>- The leaf container type enters the token and the key. <br>- G1 is reworded to claim only trace-visible paths. <br>- N5 is filed as #5233 and N3 as #5234. Probes P5-P11 are added. |

## 1. Context

`memoize_jaxpr` (`src/xtrax/inference/memo.py`) admits a function only after two static
screens pass on its traced jaxpr:
- `_screen_jaxpr`, the purity screen (stateful, callback and random primitives);
- `_screen_donation`, the donation screen (added in PR #158).

The adversarial review of spec `260922_conformance-residuals` found three ways a function
reaches eager execution without being screened. They are pinned in
`tests/inference/test_memo.py::TestDonationFailOpen` with strict xfails:

- **#5214, one screen per wrapper.** `_MemoCore._ensure_program` runs once, guarded by
  `if self.program_digest is None`, using the first call's arguments. A function whose trace
  depends on shape is admitted by a small first call. A later call with a new shape then runs
  the unscreened path eagerly on a cache miss.
- **#5215, kwargs are never traced.** The probe traces `self.fn(*a)` with no kwargs.
- **#5216, the purity walk is incomplete.** `_screen_jaxpr._walk` finds sub-jaxprs with
  `getattr(param_val, "eqns", None)`, which misses tuple-valued params such as `lax.cond`'s
  `branches`. It also silently stops past depth 8.

Writing the spec also found a fourth defect in the same function:

- **#5231 (S-4), spot-check replay drops kwargs and races.** `_maybe_spot_check_unlocked`
  recomputes with `self.fn(*self._last_args)`.
  - It drops kwargs, so a kwarg-dependent function is recomputed on the wrong path. That
    produces a *false* `MemoStalenessError` and poisons the wrapper.
  - `_last_args` is a single slot on the core, written by `_wrapped` without the lock. A
    concurrent call can overwrite it between a hit and that hit's replay.

### 1.1 Measured probes (JAX 0.11.1, CPU)

- P1-P4: `/tmp/claude/memo_probe.py`, run by the orchestrator.
- P5-P9: the challenger's `/tmp/claude/challenger_p1..p5.py`.
- P5, P6, P8 and P9 were re-measured by the orchestrator in `/tmp/claude/verify_r1.py`.

These scripts are ephemeral. Every load-bearing result becomes a tracked test in §4.

| # | Question | Result |
|---|---|---|
| P1 | Can a Python-bool kwarg be traced abstractly? | **No.** `if fast:` raises `TracerBoolConversionError`. |
| P2 | Can a positional Python int that the function branches on be traced abstractly? | **No**, same error. Such functions cannot be memoized on main. |
| P3 | What is the shape of `lax.cond`'s sub-jaxpr param? | `branches` is a tuple of jaxprs. A random primitive inside one branch is absent from the top level. |
| P4 | Can a `str` positional arg be traced? | **No**, `TypeError`. |
| P5 | Which exceptions do scalar idioms raise when `n` is traced abstractly? | `if n` → `TracerBoolConversionError`, which **is** a `ConcretizationTypeError` subclass. `range(n)` → `TracerIntegerConversionError`, which is **not** a subclass. `jnp.zeros(n)` → `TypeError`. `x[:n]` → `IndexError`. `x + 2**40` → `OverflowError`. |
| P6 | Does a successful ABSTRACT trace prove that no Python branch consulted the value? | **No.** `if flag is True`, `isinstance(s, float)` and a caught `ConcretizationTypeError` all trace cleanly, on the pure side. On `b85e1e5`, `memoize_jaxpr(f)(x, True)` with `if flag is True: <random>` was admitted and cached a random output. |
| P7 | Which carriers does `_iter_subjaxprs` reach? | All that were probed: `cond.branches`, `while.cond_jaxpr/body_jaxpr`, `scan.jaxpr`, `custom_jvp_call.call_jaxpr`, `custom_vjp_call.call_jaxpr`, `remat2.jaxpr`, `jit.jaxpr`, `custom_linear_solve.jaxprs`. The only callable-valued params seen were `custom_vjp_call.out_trees` and `jit.ctx_mesh`; neither runs user code. |
| P8 | Do NFC and NFD forms of "é" collide under `_leaf_digest`? | **Yes.** `nfc == nfd` is False, yet the digests are equal. |
| P9 | Can `_leaf_digest` digest `b"\xff"`? | **No**, `UnicodeDecodeError`. |
| P10 | Are an `np.ndarray` leaf and an equal `jax.Array` leaf distinct keys? | **No.** The `jax.Array` call is served from the entry keyed by the numpy call. |
| P11 | Which banned primitives do `jax.random` draws emit? | With a constant key, `uniform`/`normal` emit `random_seed` and `random_bits`. With a key argument, `bits(k)` emits `random_bits`. With a key argument, `split`/`fold_in` emit only `random_split`/`random_fold_in`, which are **not banned** (#5234). |

## 2. Goals and non-goals

**Goals**
- G1. Every eager execution of the user function on a cache miss is preceded by a successful
  purity and donation screen of the trace **for that call's signature**, including kwargs.
  The screen covers every **trace-visible** path of that signature. N5 lists the paths a trace
  cannot see.
- G2. The purity screen visits every sub-jaxpr that the donation screen visits: no depth cap,
  and tuple/list params included.
- G3. Spot-check replay recomputes with exactly the `(args, kwargs)` of the call being
  checked, with no shared mutable slot.
- G4. Retracing stays bounded for the common case: a function that takes exact-`int`/`float`
  scalars and never branches on them in Python traces once per array signature, not once
  per scalar value.

**Non-goals**
- N1. Out-of-trace impurity (closure state, time, I/O) remains a documented blind spot.
- N2. Custom-rule bodies that are Python callables rather than jaxprs, namely `custom_vjp`
  `bwd` and `custom_jvp` `jvp`, remain a documented blind spot. P7 found no callable param
  that runs in the primal.
- N3. Auditing the banned-primitive name lists is out of scope and filed as **#5234**. The
  ACs below name their draw, so they cannot pass by accident of the list.
- N4. Multi-device support.
- N5. **Trace/eager divergence**, filed as **#5233**. A miss runs `fn` eagerly, but the screen
  saw its trace. These idioms can take an eager path the trace never took:
  - identity checks (`x is True`, `mode is Mode.A`);
  - `isinstance`/`type` checks, including `Tracer` vs concrete and `np.ndarray` vs `jax.Array`;
  - catching `ConcretizationTypeError`.

  This hole exists on main for positional scalars and arrays, and this sprint does not widen
  it. Kwargs move from *untraced* to traced, and positional scalars are traced abstractly
  exactly as on main. The sprint narrows N5 with D-2 (§3.1) and documents the rest (AC-16).
  The closing design, running the screened program on a miss, is #5233. It changes the
  miss-return aliasing contract from #158 and costs a compile per signature, so it is not
  done here.

## 3. Design

### 3.1 Leaf classes, signature and trace modes (#5214, #5215)

Flatten `(args, kwargs)` with `jax.tree_util.tree_flatten`. Classify each leaf in this order:

1. **Array leaf:** has `.shape` and `.dtype`. This includes numpy scalars such as
   `np.float32(1)`, as in `_leaf_digest`. Its descriptor is
   `("arr", type(leaf).__qualname__, shape, dtype.name, weak_type)`. The container type is
   included per D-4.
2. **Traceable scalar:** `type(leaf) is int` or `type(leaf) is float`, using an **exact** type
   check. Its descriptor in ABSTRACT mode is `("dyn", "int" | "float")`.
3. **Static scalar:** every other instance of `bool`, `int`, `float`, `str` or `bytes`. That
   covers `bool`, `IntEnum`, other `int`/`float` subclasses, `str` and `bytes`. The leaf is
   **always** held static (D-2), with descriptor `("static", type(leaf).__qualname__,
   exact_value_token)`. `exact_value_token` is:
   - `repr(leaf)` for numbers;
   - `leaf.encode("utf-8", "surrogatepass")` for `str`;
   - the raw bytes for `bytes`.

   There is **no** Unicode normalization (D-3).
4. Anything else raises `MemoKeyUnsupportedLeafError` **before tracing**, as today.

**ABSTRACT mode** traces array and traceable-scalar leaves as abstract values and holds static
scalars static. Its token is `("A", treedef, descriptors)`.

**STATIC mode** traces array leaves only; traceable scalars are also held static. Its token is
`("S", treedef, descriptors)`, where each traceable scalar's descriptor becomes
`("static", "int" | "float", repr(value))`.

**What ABSTRACT mode guarantees (restated per OBJ-R1-02).** The guarantee is narrow: a Python
branch that consults a traced value's *truthiness, integer value or concrete array value*
cannot silently take one side, because it raises. It does **not** cover identity checks, type
checks, or caught concretization errors (P6). Those are the N5 idioms.

D-2 closes N5 for `bool` and enum scalars (`flag is True`, `mode is Mode.A`), because those
leaves are never traced. The cost is one trace per distinct static value, which is at most two
per `bool` and so leaves G4 intact.

**Fallback rule (per OBJ-R1-01).** The ABSTRACT trace can raise.
- If it raises **any** `Exception` **and** the signature has at least one traceable-scalar
  leaf, retry in STATIC mode.
- If there are no traceable scalars, STATIC mode would be identical, so classify the
  ABSTRACT exception directly under the rule below.

**Classification of a failed final trace** (STATIC mode, or ABSTRACT mode when there are no
traceable scalars):
- `jax.errors.ConcretizationTypeError` (including its subclasses),
  `jax.errors.TracerIntegerConversionError`, `jax.errors.TracerArrayConversionError` or
  `IndexError` (which covers `NonConcreteBooleanIndexError`):
  - raise `MemoKeyUnsupportedLeafError`, chained with `from exc`;
  - the message must say the function consults the **value of an array argument** in Python
    (branching, indexing or conversion), which cannot be screened.
- Any other `TypeError`: raise `MemoKeyUnsupportedLeafError` with today's "cannot be traced"
  message, chained with `from exc`.
- Any other exception propagates **unchanged**. It is the user function's own error, and it
  would be raised eagerly too.

The trace calls a probe that takes only the traced leaves. The probe re-inserts the static
leaves at their flat positions, unflattens with the recorded treedef, and calls
`fn(*args, **kwargs)`.

### 3.2 Screened-signature table (per OBJ-R1-06)

`_MemoCore` replaces `program_digest: str | None` with `_screened: OrderedDict[token,
str | _NeedsStatic]`. The resolution and screening function is
`_ensure_screened(args, kwargs) -> str`.

1. Build the ABSTRACT token. Look it up under `self.lock`, and **return the digest you read**.
   Never re-read the table later, since a concurrent eviction could remove the entry.
   - If the value is a digest, return it.
   - If the value is `_NeedsStatic`, build the STATIC token and look that up. If it is present,
     return its digest; otherwise go to step 3.
2. On an ABSTRACT miss, trace in ABSTRACT mode outside the lock.
   - On success, run both screens. If they pass, insert `abstract_token → digest` and return
     the digest.
   - On failure, apply the §3.1 fallback rule.
3. Trace in STATIC mode outside the lock and run both screens. **Only if both pass**, insert
   `static_token → digest` and, if we came from step 2's fallback, `abstract_token →
   _NeedsStatic`. Both inserts happen under one lock acquisition.
   - Rule: **nothing is inserted until a screen has passed for some mode.**
4. A screen failure (`MemoImpurityError` or `MemoDonationError`) inserts nothing and latches
   (§3.4).
5. A classification failure (`MemoKeyUnsupportedLeafError`) inserts nothing and does **not**
   latch. The same signature re-traces on every call, which is accepted explicitly: it is an
   error path, and the user must change the call to proceed.

- The table is bounded by the module constant `_MAX_SCREENED_SIGNATURES = 1024`. It is read
  **at use time** as a module global, never captured in `__init__` or a default argument, so
  that AC-13 can monkeypatch it. Eviction is LRU, and costs only a re-trace.
- Tracing never holds the lock. Two threads racing on one new signature both trace, which is
  idempotent.
- `build_key(digest, args, kwargs)` hashes **the digest returned by `_ensure_screened` for
  this call**, then the structure, the leaf digests, the salt and the stamp, as today.
- The `program_digest` attribute is removed. Grep finds no reference outside `memo.py`.

### 3.3 Leaf digest (per OBJ-R1-03 and OBJ-R1-04)

`_leaf_digest` changes in two ways. The cache is in-memory only, so no persisted keys
change.
- **Arrays** fold `type(leaf).__qualname__` into the digest, before the dtype.
- **`str`** digests `b"str:" + leaf.encode("utf-8", "surrogatepass")`, and **`bytes`**
  digests `b"bytes:" + leaf`. Both are exact and unnormalized; the NFC call is deleted.

`_program_digest` passes `np.asarray(const)` for every const. Its behaviour is unchanged,
because every const is an `ndarray` there.

### 3.4 Purity walk (#5216)

`_screen_jaxpr` is rewritten to traverse the way `_screen_donation` does:
- an explicit stack over `closed.jaxpr`;
- for every eqn, every param value, and every `_iter_subjaxprs(param_val, name)`;
- no depth cap;
- all offenders collected before raising once.

The exception type is unchanged. The message gains the offender **paths**, built with
`_eqn_label` as in the donation screen, e.g. `jaxpr.cond.branches[1].random_bits`.

### 3.5 Latch semantics (unchanged)

A screen failure on **any** signature sets `screen_latched_error`. Every later call raises it,
including calls with signatures admitted earlier, until `memo_rewrap()`. Wrapping attests
that the *function* is pure, and one detected impure path falsifies that attestation.
`MemoDonationError` keeps its type through the latch.

### 3.6 Spot-check replay (#5231)

- `call()` passes its own `args` and `kwargs` to
  `_maybe_spot_check_unlocked(key, args, kwargs)`, which recomputes with
  `self.fn(*args, **kwargs)`.
- `_last_args`, both the class attribute and the `_wrapped` write, is deleted.

### 3.7 Zero-arg wrap-time screening

This is unchanged in behaviour: `_ensure_screened((), {})` runs at wrap time for a function
with no parameters.

## 4. Acceptance criteria

All tests go in `tests/inference/test_memo.py`.
- Every impurity AC **must** draw with `jax.random.uniform(jax.random.key(0), shape)`, which
  emits `random_seed` and `random_bits` (P11). Split/fold_in-only bodies are forbidden (N3).
- Every donation AC uses `jax.jit(..., donate_argnums=0)`.
- "Fails on main" means: fails when run against `b85e1e5`'s `memo.py`.

| AC | Criterion | Red control |
|---|---|---|
| AC-1 | #5214: `test_ac15_shape_dependent_donation_not_rescreened` has its strict xfail removed and passes: a (4,) call is admitted, then an (8,) call on the donating branch raises `MemoDonationError`. | xfail strict on main |
| AC-2 | #5214 purity twin: a shape-dependent `uniform` draw on the second shape raises `MemoImpurityError`, and it is exactly that type, not `MemoDonationError`. | fails on main |
| AC-3 | #5215: `test_ac16_kwarg_dependent_donation_not_traced` has its xfail removed and passes. | xfail strict on main |
| AC-4 | #5215 purity twin: `f(x, *, fast=False)` whose `fast=True` path draws. `f(x)` is admitted and hits on repeat. `f(x, fast=True)` raises `MemoImpurityError`. Afterwards `f(x)` raises the latched error. | fails on main |
| AC-5 | STATIC fallback: each of these is memoizable with `n` a Python `int` (same `n` hits, different `n` misses), and each fails on main with `MemoKeyUnsupportedLeafError`: (a) `if n > 1`; (b) `sum(range(n))`; (c) `jnp.zeros(n)`; (d) `x[:n]`. | fails on main (P5) |
| AC-5e | `f(x, n)` computing `x + n` with `n = 2**40` raises `OverflowError` unchanged, the same exception the eager call raises (measured 260922: eager and STATIC-traced both overflow). It is not relabelled as `MemoKeyUnsupportedLeafError` and not swallowed. OBJ-R1-01 proposed "memoizable" here; that is false, since the eager call itself overflows. | pin (§3.1 "propagates unchanged") |
| AC-5f | An impure branch selected in STATIC mode by an int value is rejected: `if n > 3: <uniform draw>` with `n=5` raises `MemoImpurityError`. | fails on main |
| AC-6 | `str`: a positional string argument is memoizable. The NFC `"é"` and NFD `"é"` forms are **distinct** keys, so both calls miss. | fails on main (P4, P8) |
| AC-6b | `bytes`: `b"\xff"` as an argument is memoizable. | fails on main (P9) |
| AC-7 | G4: a function `x * s` with float `s`, called with 5 distinct float values, traces **once**. Count traces with a tracer-only counter, as in `_counting_fn`. | mutation: route every scalar to STATIC → 5 traces |
| AC-7b | D-2: a `bool` leaf is held static. `f(x, flag)` using `if flag is True: <uniform draw>` raises `MemoImpurityError` for `flag=True`, while `flag=False` is admitted. | fails on main (P6) |
| AC-7c | D-2: an `IntEnum` leaf is held static. `mode is Mode.B` selecting a draw raises `MemoImpurityError`. | fails on main |
| AC-8 | Array-value refusal: `if x[0] > 0`, `x[x > 0]` and `np.asarray(x)` inside the function each raise `MemoKeyUnsupportedLeafError`, and the message contains `value of an array argument`. | message assertion (new wording) |
| AC-8b | A non-classified exception propagates unchanged: a function that raises `ZeroDivisionError` during tracing surfaces `ZeroDivisionError`. | pin |
| AC-9 | #5216: a draw inside a `lax.cond` branch raises `MemoImpurityError`, and the message contains `branches[`. | fails on main |
| AC-10 | #5216: a draw nested in 10 `jax.jit` levels raises `MemoImpurityError`. | fails on main (P: `jit^9` and `jit^10` admitted) |
| AC-11 | #5216: a draw inside a `while_loop` body and inside a `scan` body each raise. | regression pin (passes on main) |
| AC-12 | Latch across signatures: admit signature A, reject B; then A raises the latched error. After `memo_rewrap()`, A is admitted again. | mutation: per-signature latch |
| AC-13 | Table bound: with `_MAX_SCREENED_SIGNATURES` monkeypatched to 2, cycling 3 shapes never grows the table past 2, and an evicted shape re-traces (counter). | mutation: no eviction |
| AC-13b | Container type: an `np.ndarray` leaf, then an equal `jax.Array` leaf: both miss. | fails on main (P10) |
| AC-14 | #5231: `f(x, *, scale=1.0)` returning `x * scale`, with `spot_check_every=1`, and two calls `f(x, scale=3.0)`: no `MemoStalenessError`, and the hit returns `x * 3`. | fails on main (false staleness) |
| AC-15 | #5231 race, behavioural and deterministic (no sleeps). Monkeypatch the instance's `build_key` to block on a `threading.Event` for inputs A. Thread 1 calls `f(A)`, a spot-checked hit, and blocks. Thread 2 calls `f(B)` to completion. Release thread 1. Assert thread 1 gets no `MemoStalenessError` and returns A's value. With a timeout join, the test must not hang on failure. | fails on main (shared `_last_args`) |
| AC-16 | Docs (`docs/api/inference.md`): the "fail-open admission paths" block becomes a "Fixed" note naming #5214, #5215, #5216 and #5231. The docs describe: <br>- ABSTRACT/STATIC modes and D-2; <br>- the STATIC retrace cost; <br>- the array-value refusal; <br>- exact string keys; <br>- blind spots N1, N2 and N5, with N5's idioms listed and #5233 named. <br>`test_ac17` also requires `#5216`, `#5233` and `STATIC`. | doc test |
| AC-17 | No regressions: `uv run --extra dev pytest tests/inference -q` passes. This is a narrow run; the whole suite is never run locally. | — |

## 5. Tasks (sequential fixers, one worktree)

| Task | Scope | Files | Backlog |
|---|---|---|---|
| T1 | §3.4 purity walk; AC-9, 10, 11 | `memo.py` (`_screen_jaxpr`), tests | #5216 |
| T2 | §3.1, 3.2, 3.3, 3.5, 3.7; AC-1..8b, 12, 13, 13b | `memo.py`, tests | #5214, #5215 |
| T3 | §3.6; AC-14, 15 | `memo.py`, tests | #5231 |
| T4 | AC-16 | `docs/api/inference.md`, `test_ac17` | — |

Order: T1 → T2 → T3 → T4, one commit each. After each commit the orchestrator runs
`git show --stat` against this table, then checks that task's tests red-before-green against
`b85e1e5`.

## 6. Risks

| Risk | Mitigation |
|---|---|
| STATIC mode retraces once per distinct scalar value. | It is only paid by functions whose ABSTRACT trace fails, and by bool/enum/str leaves. Those functions were refused outright on main. The cost is documented, and the 1024 cap bounds memory. |
| Retrying on *any* exception runs the user function's trace twice for a genuine bug. | Only when a traceable scalar is present; the second failure is propagated or classified. Tracing has no concrete side effects beyond N1. |
| Container type now enters the key, so `np` and `jax` inputs no longer share entries. | Intended (P10 showed eager behaviour can differ). |
| A later-signature rejection latches and breaks earlier-working calls. | Intended (§3.5), documented. |

## 7. Evidence discipline

- Each "fails on main" control is demonstrated by running the new tests against a
  `git show b85e1e5:src/xtrax/inference/memo.py` copy. The auditor re-runs a sample.
- The mutation controls for AC-7, 12 and 13 are run once by the auditor; they are not
  shipped.

## 8. Adversarial review dispositions

### Round 1 (challenger, Opus): REVISE

| Obj | Sev | Disposition |
|---|---|---|
| OBJ-R1-01 | BLOCKER | **Accepted.** The fallback retries on any exception when traceable scalars exist, and only the final trace is classified (§3.1). P5 is re-measured, and AC-5 is extended with (b)-(f). The objection's claim that `2**40` becomes memoizable in STATIC mode is **rejected**: eager `x + 2**40` also raises `OverflowError` (`/tmp/claude/verify_r2.py`), so AC-5e pins propagation instead. |
| OBJ-R1-02 | MAJOR | **Accepted.** The guarantee is restated narrowly. D-2: `bool` and non-exact scalars are always static. AC-7b and AC-7c added. |
| OBJ-R1-03 | MAJOR | **Accepted.** Exact str/bytes digests, no NFC (§3.3). AC-6 is inverted and AC-6b added. |
| OBJ-R1-04 | MINOR | **Accepted.** Container type in the descriptor and the digest (§3.1, §3.3). AC-13b added. |
| OBJ-R1-05 | MINOR | **Accepted.** The ACs mandate a `uniform` draw. The dead/unbanned names are filed as #5234. |
| OBJ-R1-06 | MINOR | **Accepted.** Insert-after-screen rule, returned digest, use-time constant, and the non-latching classification failure made explicit (§3.2). |
| OBJ-R1-07 | MINOR | **Accepted.** AC-15 is now a deterministic behavioural race test. |
| OBJ-R1-08 | MINOR | **Accepted.** The N5 wording is corrected. |
| OD-1 | ruling | **Carve-out accepted on the challenger's conditions:** (a) G1 reworded; (b) D-2 adopted; (c) AC-16 lists the N5 idioms; (d) filed as #5233. |

## 10. Follow-ups filed

- #5231: S-4, in scope (T3).
- #5233: N5 trace/eager divergence.
- #5234: N3 banned-primitive list audit.
