---
title: memoize_jaxpr screen hardening
description: 'Per-signature purity/donation screening with static non-array leaves and kwargs, uncapped sub-jaxpr traversal for the impurity screen, and kwarg-faithful spot-check replay (#5214, #5215, #5216)'
status: draft
task_id: 260922_memo-screen-hardening
date: '260922'
backlog_ids: '5214, 5215, 5216'
adversarial_review: ''
revision: 1
---
# memoize_jaxpr screen hardening

## Revision history

| Rev | Date | Change |
|---|---|---|
| r1 | 260922 | Initial draft. Probes P1-P4 measured before drafting (§1.1). |

## 1. Context

`memoize_jaxpr` (`src/xtrax/inference/memo.py`) admits a function only after two static
screens pass on its traced jaxpr:
- `_screen_jaxpr`, the purity screen (stateful, callback and random primitives);
- `_screen_donation`, the donation screen (added in PR #158).

The adversarial review of spec `260922_conformance-residuals` found three ways a function
reaches eager execution without being screened. They were filed as backlog items and are
pinned in `tests/inference/test_memo.py::TestDonationFailOpen` with strict xfails:

- **#5214, one screen per wrapper.** `_MemoCore._ensure_program` runs once, guarded by
  `if self.program_digest is None`, using the first call's arguments. A function whose trace
  depends on shape is admitted by a small first call. A later call with a new shape then runs
  the unscreened path eagerly on a cache miss.
- **#5215, kwargs are never traced.** The probe traces `self.fn(*a)` with no kwargs. So a path
  selected by a kwarg, such as `f(x, fast=True)`, is never screened.
- **#5216, the purity walk is incomplete.** `_screen_jaxpr._walk` finds sub-jaxprs with
  `getattr(param_val, "eqns", None)`. That misses tuple-valued params, such as `lax.cond`'s
  `branches`. It also silently stops past depth 8. Both failures admit rather than refuse.

This sprint closes all three. Writing the spec also surfaced a fourth defect in the same
function, **S-4**, which is filed with the sprint:

- **S-4, spot-check replay drops kwargs and races.** `_maybe_spot_check_unlocked` recomputes
  with `self.fn(*self._last_args)`, which drops kwargs. A kwarg-dependent function is
  therefore recomputed on the wrong path. The allclose comparison fails, and the wrapper
  poisons itself with `MemoStalenessError`: a false staleness report.
  `_last_args` is also a single slot on the core, written by `_wrapped` without the lock. A
  concurrent call can overwrite it between a hit and its replay, so the replay compares key
  A's cached value against call B's inputs.

### 1.1 Measured probes

Script: `/tmp/claude/memo_probe.py`. It is ephemeral inspection; §7 moves anything
load-bearing into tracked tests. Run with JAX from the repo lock, CPU backend.

| # | Question | Result |
|---|---|---|
| P1 | Can a Python-bool kwarg be traced abstractly (`make_jaxpr(f)(x, fast=True)`)? | **No.** `if fast:` raises `TracerBoolConversionError`, which is a `TypeError` subclass. Today's `except TypeError` would relabel it as `MemoKeyUnsupportedLeafError`. |
| P2 | Can a positional Python int that the function branches on be traced abstractly? | **No**, same error. So today such a function cannot be memoized at all. This is a usability bug, not a safety bug. |
| P3 | What is the shape of `lax.cond`'s sub-jaxpr param? | `branches` is a tuple of 2 jaxpr objects, each exposing `.eqns`. A `random_seed`/`random_bits` inside one branch is present in that branch's eqns and absent from the top level. |
| P4 | Can a `str` positional arg be traced? | **No**, `TypeError`. `_leaf_digest` supports strings, yet any string argument fails at trace time. |

## 2. Goals and non-goals

**Goals**
- G1. Every eager execution of the user function on a cache miss is preceded by a successful
  purity and donation screen of the trace **for that call's signature**, including kwargs.
- G2. The purity screen visits every sub-jaxpr that the donation screen visits: no depth cap,
  and tuple/list params included.
- G3. Spot-check replay recomputes with exactly the `(args, kwargs)` of the call being
  checked.
- G4. Retracing stays bounded for the common case: a function that never branches in Python
  on a scalar argument traces once per array shape/dtype signature, not once per scalar value.

**Non-goals**
- N1. Out-of-trace impurity: closure state, time, I/O. This remains a documented blind spot.
- N2. Custom-rule bodies that are not jaxprs: `custom_vjp` `bwd` and `custom_jvp` `jvp`
  callables. They are Python callables, not sub-jaxprs, and do not run in the primal
  computation this cache stores. Documented as a blind spot.
- N3. Reviewing the banned-primitive name lists (`_STATEFUL_PRIMITIVES` etc.) for coverage
  against current JAX primitive names. A separate item is filed (§10).
- N4. Multi-device support.
- N5. **Trace/eager divergence via Python introspection**, recorded as open decision OD-1.
  - *The hole.* The screen inspects the traced jaxpr, but a miss runs `self.fn(*args,
    **kwargs)` **eagerly**. A function that branches on `isinstance(x, jax.core.Tracer)`,
    `type(n) is int`, or `isinstance(x, np.ndarray)` can take a path eagerly that its trace
    never took. This hole exists on `main` for array leaves today, and ABSTRACT mode extends it
    to scalars.
  - *The alternative that closes it.* On a miss, execute the screened program itself (e.g.
    `jax.jit(jax.core.jaxpr_as_fun(closed))` with the static leaves bound), not `fn`.
  - *What that costs.* A compile per signature. The miss/aliasing semantics that #158's
    AC-13/§3.4 fixed would change. Behaviour would differ for functions that jit internally.
  - *r1 position.* Out of scope: documented blind spot, backlog item filed. The challenger is
    asked to rule on this explicitly.

## 3. Design

### 3.1 Signature and trace modes (#5214, #5215)

Leaves are classified from `jax.tree_util.tree_flatten((args, kwargs))`:
- **Array leaves:** anything with `.shape` and `.dtype`, as `_leaf_digest` defines them.
- **Scalar leaves:** `bool`, `int`, `float`, `str`, `bytes`. Check `bool` before `int`.
- Any other leaf raises `MemoKeyUnsupportedLeafError` **before tracing**, using the same
  message family as `_leaf_digest`.

Two trace modes exist:

- **ABSTRACT mode** is tried first.
  - Array leaves and `bool`/`int`/`float` leaves are traced as abstract values.
  - `str`/`bytes` leaves are always held static, because P4 shows they cannot be traced.
  - The signature token is `("A", treedef, per-leaf descriptors)`. An array leaf's descriptor
    is `(shape, dtype.name, weak_type)`. A traced scalar's descriptor is its Python type name.
    A static string's descriptor is `("S", type name, NFC-normalized value)`.
- **STATIC mode** is the fallback.
  - Array leaves are traced abstractly. Every scalar leaf is held static.
  - The token is `("S", treedef, descriptors)`. Each static scalar's descriptor is
    `("S", type name, repr(value))`, NFC-normalized for strings.
  - The type name keeps `True` and `1` distinct.

**Soundness argument for ABSTRACT mode.** Under abstract tracing, a Python-level branch on a
traced value cannot silently pick one side: it raises `ConcretizationTypeError` or one of its
subclasses (`TracerBoolConversionError`, `TracerIntegerConversionError`). So when an ABSTRACT
trace *succeeds*, every value-dependent control path is in the jaxpr as `cond`/`switch`/
`while`/`select`, and §3.3's walk visits all of them. Shape and dtype are part of the token,
so any shape-dependent Python branch is re-screened per shape.

**Fallback rule.** If the ABSTRACT trace raises `jax.errors.ConcretizationTypeError` or a
subclass, retry in STATIC mode. The table (§3.2) records the ABSTRACT token as needing STATIC
mode, so later calls with that abstract signature go straight to STATIC. If the STATIC trace
*also* raises `ConcretizationTypeError`, the function branches in Python on an **array
value**. That cannot be screened, so raise `MemoKeyUnsupportedLeafError` with a message
naming that cause. The exception type is kept for compatibility.

Any other `TypeError` from tracing raises `MemoKeyUnsupportedLeafError`, as today.

The trace wraps the user function in a probe that takes only the traced leaves. The probe
re-inserts the static leaves at their flat positions, unflattens with the recorded treedef,
and calls `fn(*args, **kwargs)`.

### 3.2 Screened-signature table

`_MemoCore` replaces `program_digest: str | None` with
`_screened: OrderedDict[token, str | _NeedsStatic]`, an LRU table that maps a token to its
program digest. `_NeedsStatic` is a sentinel stored only under ABSTRACT tokens.

- It is bounded by the module constant `_MAX_SCREENED_SIGNATURES = 1024`. Eviction costs only
  a re-trace and re-screen, which is safe.
- Lookup and insertion happen under `self.lock`. Tracing and screening run **outside** the
  lock. Two threads racing on the same new signature both trace and screen. That is
  idempotent, and the second insert overwrites with an identical digest.
- A token is inserted **only after both screens pass**. A failing signature is never
  recorded, so `memo_rewrap()` followed by a retry re-screens it and fails again. That is the
  existing contract.
- `build_key` hashes **the current call's** signature digest, not a per-wrapper digest.
  Scalar and string leaf *values* still enter the key through `_leaf_digest`, as today.

The `program_digest` attribute is removed. No test or src reference exists outside
`memo.py`; this was verified by grep in §1.1's session.

### 3.3 Purity walk (#5216)

`_screen_jaxpr` is rewritten to traverse the same way `_screen_donation` does:
- an explicit stack over `closed.jaxpr`;
- for every eqn, every param value, and every `_iter_subjaxprs(param_val, name)`;
- no depth cap;
- all offenders collected before raising once.

The message and exception type are unchanged, but the message gains the offending
**paths**, built by `_eqn_label` as in the donation screen. That tells users which cond branch
to fix.

### 3.4 Latch semantics (unchanged, restated)

A screen failure on **any** signature sets `screen_latched_error`. Every later call raises it,
including calls with signatures admitted earlier, until `memo_rewrap()`. This is deliberate:
wrapping is a purity attestation about the *function*, and one detected impure path falsifies
that attestation for every call. `MemoDonationError` keeps its type through the latch.

### 3.5 Spot-check replay (S-4)

- `_maybe_spot_check_unlocked(key, args, kwargs)` receives the current call's arguments
  explicitly from `call()` and recomputes with `self.fn(*args, **kwargs)`.
- `_last_args`, both the class attribute and the `_wrapped` write, is deleted.
- Replay runs only on a hit for this call's own key. Because of how keys are built
  (§3.2), `(args, kwargs)` are by construction the inputs that produced the cached value, up
  to digest equality.

### 3.6 Zero-arg wrap-time screening

This is unchanged in behaviour. `_ensure_program(())` becomes
`_ensure_screened((), {})`, which records the empty signature's token.

## 4. Acceptance criteria

All tests go in `tests/inference/test_memo.py`. Each AC names a red control: a check that
fails on `origin/main` (`b85e1e5`), or a mutation that must turn the test red.

| AC | Criterion | Red control |
|---|---|---|
| AC-1 | #5214: `test_ac15_shape_dependent_donation_not_rescreened` has its strict xfail **removed** and passes. A (4,) call is admitted, then an (8,) call on the donating branch raises `MemoDonationError`. | xfail strict today = fails on main |
| AC-2 | #5214 purity twin: shape-dependent `jax.random` on the second shape raises `MemoImpurityError`, not a subclass. | fails on main (admitted) |
| AC-3 | #5215: `test_ac16_kwarg_dependent_donation_not_traced` has its xfail removed and passes. | xfail strict today |
| AC-4 | #5215 purity twin: a kwarg-gated random path raises `MemoImpurityError`. The same wrapper with `fast=False` first is admitted and caches. After the rejection, **both** signatures raise (latch, §3.4). | fails on main |
| AC-5 | A positional Python int that the function branches on is memoizable: two calls with the same int hit, and different ints miss. It is rejected when the int value selects an impure branch. | on main the first call raises `MemoKeyUnsupportedLeafError` (P2) |
| AC-6 | A `str` positional argument is memoizable, and keys differ by NFC-normalized value. | on main raises (P4) |
| AC-7 | Bounded retrace (G4): a function with a float scalar and no Python branch on it, called with 5 distinct float values, traces **once**. Count via a tracer-only counter, as in `_counting_fn`. | mutation: force STATIC mode, then count == 5 |
| AC-8 | A Python branch on an **array value** (`if x[0] > 0`) raises `MemoKeyUnsupportedLeafError`, and the message names the Python-branch cause. | message assertion |
| AC-9 | #5216: a random primitive inside a `lax.cond` branch raises `MemoImpurityError`; the message names a `branches[...]` path. | fails on main |
| AC-10 | #5216: a random primitive nested 10 `jax.jit` levels deep raises `MemoImpurityError`. | fails on main (depth > 8 silently admitted) |
| AC-11 | #5216: a random primitive inside a `lax.while_loop` body and inside a `lax.scan` body each raise. | regression pin |
| AC-12 | Latch across signatures (§3.4): admit signature A, reject signature B, then signature A raises the latched error; after `memo_rewrap()` A is admitted again. | mutation: per-signature latch |
| AC-13 | Table bound: with `_MAX_SCREENED_SIGNATURES` monkeypatched to 2, three shapes cycle. The table length never exceeds 2, and an evicted shape re-traces. | mutation: remove eviction |
| AC-14 | S-4: a kwarg-dependent function with `spot_check_every=1` and calls `f(x, scale=3.0)` twice gives **no** `MemoStalenessError`, and the hit returns the right value. | fails on main (false staleness) |
| AC-15 | S-4 structural: `_MemoCore` has no `_last_args` attribute, and `_maybe_spot_check_unlocked` accepts `args` and `kwargs`. | fails on main |
| AC-16 | Docs: in `docs/api/inference.md`, the "fail-open admission paths" block becomes a "fixed in 0.4.0a?" note naming #5214, #5215, #5216. The docs also describe ABSTRACT/STATIC modes, the retrace cost of STATIC mode (one trace per distinct scalar value), the array-value-branch refusal, and blind spots N1/N2. `test_ac17` is updated to also require `#5216` and `STATIC`. | doc test |
| AC-17 | No regressions: `uv run --extra dev pytest tests/inference -q` passes (narrow run; the whole suite is not run locally). | — |

## 5. Tasks (sequential fixers, one worktree)

| Task | Scope | Files | Backlog |
|---|---|---|---|
| T1 | §3.3 purity walk + AC-9/10/11 | `memo.py` (`_screen_jaxpr`), tests | #5216 |
| T2 | §3.1/3.2/3.4/3.6 signature table + trace modes + AC-1..8, 12, 13 | `memo.py` (`_MemoCore`, `_wrap`), tests | #5214, #5215 |
| T3 | §3.5 spot-check replay + AC-14/15 | `memo.py`, tests | S-4 (new) |
| T4 | AC-16 docs | `docs/api/inference.md`, `test_ac17` | — |

Order: T1 → T2 → T3 → T4. Each task is a separate commit. After each commit, the orchestrator
runs `git show --stat` and checks the commit touches only the files owned by that task.

## 6. Risks

| Risk | Mitigation |
|---|---|
| STATIC mode retraces once per distinct scalar value. A function called with a fresh float every time, *which also branches on it in Python*, retraces on every call. | Only functions whose abstract trace fails pay this; that is the price of screening them at all (today they are refused outright, P2). The cost is documented (AC-16). The 1024 table cap bounds memory. |
| The `ConcretizationTypeError` import path changes across JAX versions. | Use `jax.errors.ConcretizationTypeError`, a public API. |
| Behaviour change: scalar leaves are traced abstractly (as today), but strings are now held static rather than refused. | This widens admission only for inputs that are fully screened. |
| The latch now fires on a later-signature rejection, so a previously working small-shape call path starts raising. | Intended (§3.4), and documented. |

## 7. Evidence discipline

- Every "fails on main" red control is demonstrated by running the new test against
  `b85e1e5`'s `memo.py` via a `git show b85e1e5:src/xtrax/inference/memo.py` copy.
- Mutation controls (AC-7, 12, 13) are run once by the auditor, not shipped.

## 8. Adversarial review

(pending)

## 10. Follow-ups filed

- S-4 as a backlog bug (in scope, id to record).
- N3: banned-primitive name list coverage audit against current JAX.
