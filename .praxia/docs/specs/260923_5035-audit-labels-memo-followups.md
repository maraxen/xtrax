---
title: Audit gate truth-in-labeling and memoize_jaxpr review follow-ups
description: 'Sprint spec for #5035 (coverage-DAG PASS label over failing tiers), #5241 (memo hot-path and duplicate-walker cleanup), #5240 (all-default wrap-time screening, NaN static keys)'
status: draft
task_id: 260807_autonomous-dev-loop
date: '260923'
backlog_ids: ''
adversarial_review: ''
---
# Audit gate truth-in-labeling and memoize_jaxpr review follow-ups

Sprint `260923_5035-audit-labels-memo-followups`. Base: `origin/main` @ `49f3def`. All anchors
below were re-read against that commit.

## Overview

This sprint has three parts:

- **#5035:** the non-blocking coverage-DAG step prints `PASS` over failing tests. Change what it prints so the verdict line states failures and says where enforcement happens. Exit codes stay the same, except that the usage error `--enforce ''` now exits 1 like any other unknown enforce tier (ODQ-2).
- **#5241:** make the `memoize_jaxpr` per-call path flatten `(args, kwargs)` once and use cached int bounds. Merge the two duplicate jaxpr walkers into one. Cache keys and error behaviour stay byte-identical.
- **#5240:** correct the misleading wrap-time comment. Key Python-float NaNs by their bit pattern so distinct payloads stop colliding.

### Current behaviour on `49f3def` (verified)

**#5035: `scripts/audit_coverage_dag.py`**

- `main()` calls `audit_coverage_dag()` at :457-462. The verdict block is at :464-480.
- In report-only mode (`args.enforce is None`) the script always prints `PASS: coverage DAG report-only (non-blocking)` and returns 0 (:475-477). It does this even when `results[i].tests_failed > 0`.
- `audit-deterministic` (Justfile:328-330) runs `just audit-coverage-dag` (Justfile:291-293), which runs the script with no `--enforce`.
- Enforcement lives in three recipes, and `ci.yml` runs all three:

  | Recipe | Justfile | ci.yml |
  |---|---|---|
  | `audit-coverage-tier1` | :298-299 | :89 |
  | `audit-coverage-tier2` | :301-302 | :92 |
  | `audit-coverage-tier4` | :312-313 | :123 |

- **Second PASS-over-failures path (not in the backlog text).** In enforce mode the script prints `PASS: coverage DAG enforce` (:479) whenever `passed` is true. `passed` is true even when tests failed in two cases:
  - `evaluate_enforce` (:277-316) returns `enforce_passed=None` for a tier with `measure_coverage=false` or with no `enforce_*` floors. `tier0_audit` and `tier3_port` are such tiers (`distribution/coverage_dag.toml`).
  - `--enforce X` names a tier that `--tier` did not select, so `enforce_tier == tier.id` (:388) never matches.

  No recipe or workflow reaches either path today. They are still label defects under AC (a).
- **Scope check across `scripts/audit_*.py`.** Grepping for `print(...PASS...)` finds 22 lines. The only report-only or non-blocking PASS is `audit_coverage_dag.py:476`. Every other `PASS:` is printed after a gate check that returns 1 on failure. `audit_refute_promote.py:103` reports `dropped=` as expected golden-candidate behaviour, not as a failure. No other script is relabelled.

**#5241: `src/xtrax/inference/memo.py`**

- `call()` (:748-817) calls `_ensure_screened` (:671-712), which runs `tree_flatten((args, kwargs))` (:678). It then calls `build_key` (:714-732), which runs `tree_structure` (via `_structure_token` :191-193) and `tree_leaves` (via `_pytree_leaves` :187-188). That is three pytree traversals per call.
- `_fits_default_int` (:216-221) calls `jax.dtypes.canonicalize_dtype(np.int64)` and builds `np.iinfo` for every plain-`int` leaf on every call, hits included. `_mode_token` (:286-312) already reads `jax.config.jax_enable_x64` once per call (:309).
- `_screen_jaxpr` (:382-408) and `_screen_donation` (:524-551) run the same explicit-stack walk:
  - both push onto the stack in the same order;
  - both build paths as `f"{jaxpr_path}.{_eqn_label(eqn)}"` and `f"{eqn_path}.{sub_path}"`;
  - the donation walk also tracks `top_level`.

  They run back to back at :662-663 (STATIC) and :707-708 (ABSTRACT).
- **Both hazards present.** `_screen_jaxpr` runs first, so a program with an impure primitive and a donation site raises `MemoImpurityError`, exactly that type and not the subclass `MemoDonationError` (`errors.py:50,54`). The donation walk never runs.
- **Direct test callers that the refactor must keep working:**
  - `core._ensure_screened((x,), {})` returning `str` (`test_memo.py:230-231, :1331, :1735`);
  - `core.build_key(digest, args, kwargs)` positionally (`:232-233, :1334-1335, :1736`);
  - a monkeypatched `core.build_key` that finds `A` by scanning positional tuple args (`:1553-1572`);
  - `_classify_leaf(leaf, mode=...)` / `_classify_leaf(leaf, "ABSTRACT")` (`:934-945, :1671-1675`).

  No test calls `_screen_jaxpr` or `_screen_donation` directly.
- **Measurement anchor in the backlog is wrong.** The backlog says to measure `cum_hash_seconds` on a small-call hit loop. `cum_hash_seconds` is only accumulated on a MISS (:812). It times only `_ensure_screened` (`t0` at :759, `hash_seconds` at :766), not `build_key`. A hit loop therefore leaves it at its warm-up value. See ODQ-9.

**#5240**

- `_wrap` (:983-996) has the comment "Zero-arg (or all-default) callables can be screened at wrap time". The guard `if not inspect.signature(f).parameters` fires only for zero-parameter functions.
- `docs/api/inference.md:317-318` already says "at wrap time (for zero-parameter callables)". The docs are right and the comment is wrong.
- NaN handling:
  - The STATIC float token (:273) uses `repr(leaf)`.
  - `_static_exact_token` (:224-240) uses `repr` for float subclasses.
  - The **cache-key** digest `_leaf_digest` (:155-157) uses `repr` for every Python `float`, in **every** mode.

  `repr(nan)` is `'nan'` for every payload. So two calls whose float arguments are NaNs with different payloads share one cache key even in ABSTRACT mode, and the second call is served the first call's entry. The collision is not limited to the STATIC-mode screen token.
- `repr` is exact (shortest round-trip) for every non-NaN double, and `-0.0` and `0.0` already have distinct reprs.
- Array leaves (`np.float64(nan)` included, since it has `.shape`/`.dtype`) digest via `tobytes` (:132-143), so they are already keyed by bits.

## Acceptance Criteria

"Red on `49f3def`" means the test fails when run against `49f3def`'s version of the file under test. "Pin" means the test passes on `49f3def` and must keep passing. All memo tests go in `tests/inference/test_memo.py`. All coverage-DAG tests go in `tests/distribution/test_coverage_dag.py`.

**#5035: coverage-DAG verdict labels**

- **AC-1 (backlog a, b, c).** A NEW test, `test_main_report_only_labels_failures` (the existing `test_main_report_only_exits_zero_with_mocks` stays unmodified, mock at `tests_failed=11`). Report-only run with failures, using a mocked `audit_coverage_dag` that returns `tier1_core` with `tests_failed=19` and `pytest_exit_code=1`:
  - `main([... "--tier", "tier1_core"])` returns 0.
  - The captured stdout contains no line starting with `PASS`.
  - The captured stdout contains exactly this line: `REPORT (non-blocking): coverage DAG -- 19 test failures in tier1_core (pytest exit 1); enforcement lives in just audit-coverage-tier1`.

  Red on `49f3def` (prints `PASS: coverage DAG report-only (non-blocking)`).
- **AC-2.** Report-only run with no failures (`tests_failed=0`, `pytest_exit_code=0`):
  - `main` returns 0.
  - The stdout line is exactly `REPORT (non-blocking): coverage DAG -- no test failures in tier1_core`.
  - No stdout line starts with `PASS`.

  Red on `49f3def`.
- **AC-3.** A nonzero pytest exit with `tests_failed=0` (for example a collection error, exit 2) counts as observed failure. The line reads `REPORT (non-blocking): coverage DAG -- 0 test failures in tier1_core (pytest exit 2); enforcement lives in just audit-coverage-tier1`. Red on `49f3def`.
- **AC-4.** `format_verdict` with several results joins the failure segments with ` | ` in results order and leaves clean tiers out of the segment list.
  - A failing tier with no enforcing recipe (`tier0_audit`) renders as `2 test failures in tier0_audit (pytest exit 1); not enforced by any recipe`.
  - Example: `tier0_audit` failed 2 / exit 1 and `tier1_core` failed 19 / exit 1 give `REPORT (non-blocking): coverage DAG -- 2 test failures in tier0_audit (pytest exit 1); not enforced by any recipe | 19 test failures in tier1_core (pytest exit 1); enforcement lives in just audit-coverage-tier1`.

  Red on `49f3def` (ImportError).
- **AC-5.** Enforce-mode exit codes do not change, except the single usage-error exception in sub-case d:
  - a. The existing `test_main_enforce_exits_nonzero_when_below_floor` stays unmodified and still returns 1. A NEW `capsys` test, `test_main_enforce_fail_reports_on_stderr`, uses the same mock (`evaluate_enforce` over `tier1_core` failed 11 / exit 1, returning `(False, (result,), list(result.enforce_failures))`) and asserts `main` returns 1 and captured stderr contains `FAIL: coverage DAG enforce`.
  - b. An enforce run that passes with no observed failures still prints `PASS: coverage DAG enforce` and returns 0.
  - c. An enforce run with `passed=True` but failures observed returns 0. Its stdout line is `REPORT (not enforced): coverage DAG --enforce tier0_audit -- 2 test failures in tier0_audit (pytest exit 1); not enforced by any recipe`, and no stdout line starts with `PASS`. The test mocks `--enforce tier0_audit --tier tier0_audit`, with `tier0_audit` failed 2 / exit 1.
  - d. `main(["--root", str(tmp_path), "--config", str(config_path), "--tier", "tier1_core", "--enforce", ""])` with `audit_coverage_dag` NOT mocked (its unknown-tier branch at :380-381 returns `(False, (), ["unknown enforce tier: ''"])` before any pytest run; the test monkeypatches `scripts.audit_coverage_dag.run_tier_pytest` to raise `AssertionError` as a guard) returns 1. Captured stderr contains `unknown enforce tier`. No stdout line equals `None` and no stdout line starts with `PASS`.

  Sub-cases c and d are red on `49f3def`: c prints `PASS: coverage DAG enforce` (:479); d skips the truthy guard at :469 (`''` is falsy), fails `args.enforce is None` at :475, and prints `PASS: coverage DAG enforce` with exit 0.
- **AC-6.** Module constant `ENFORCEMENT_RECIPES` equals `{"tier1_core": "audit-coverage-tier1", "tier2_eda": "audit-coverage-tier2", "tier4_controller": "audit-coverage-tier4"}`. Two consistency checks run against the real files:
  - A test parses `Justfile` and finds every recipe whose body invokes `audit_coverage_dag.py` with `--enforce <tier>`. The set of `(tier, recipe)` pairs equals `ENFORCEMENT_RECIPES.items()`.
  - For every value in `ENFORCEMENT_RECIPES`, `.github/workflows/ci.yml` contains `just <recipe>`.

  Red on `49f3def` (ImportError).
- **AC-7.** Blocking layering is unchanged. The single exit-code exception: `--enforce ''` is a usage error and now exits 1, exactly like any other unknown enforce tier (`--enforce foo` already exits 1 through the same :380-381 path). No Justfile recipe or ci.yml step passes an empty `--enforce`, and every report-only exit code is unchanged.
  - `git diff 49f3def -- Justfile .github/workflows/ci.yml` is empty. This is a gate command, not a unit test, because CI has no `49f3def`-relative ref guarantee.
  - Every pre-existing test in `tests/distribution/test_coverage_dag.py` passes unmodified. New assertions go in new tests only.

**#5241: memo pins, which land before any refactor**

- **AC-8 (pin).** A program with both hazards raises exactly `MemoImpurityError`. The fixture is `f(x) = jax.jit(lambda y: y * 2, donate_argnums=0)(x) + jax.random.uniform(jax.random.key(0), x.shape)`, called with a `(4,)` float32 array.
  - `type(exc) is MemoImpurityError`, so it is not `MemoDonationError`.
  - `str(exc)` contains `"purity screen"` and does not contain `"donation screen"`.
  - The wrapper is latched: `screen_latched_error is not None`, and a second call raises the same type.
- **AC-9 (pin).** Screen-output equivalence. For each fixture F_imp, F_don and F_both (defined in T2), the exception raised by `memoize_jaxpr(f)(*args)` matches the reference screen. The reference is a test-local verbatim copy of `49f3def`'s `_screen_jaxpr` followed by `_screen_donation`, together with test-local verbatim copies of the five helpers they call: `_iter_subjaxprs` (:418-434), `_eqn_label` (:437-441), `_wrapped_leaf_indices` (:444-470), `_eqn_donation_sites` (:473-505) and `_donation_message` (:508-521), plus the `_DonationSite` alias (:415). Only the three primitive sets (`_STATEFUL_PRIMITIVES`, `_CALLBACK_PRIMITIVES`, `_RANDOM_PRIMITIVES`) and the error classes (`MemoImpurityError`, `MemoDonationError`) are imported live, so a T4 edit to any helper body cannot move the oracle with it. It is applied to the closed jaxpr from `memo._trace_closed(f, leaves, treedef, traced_positions)`, where `traced_positions` is all leaf indices.
  - The exception type is the same (checked with `is`).
  - `str(exc)` is identical.
  - For `MemoDonationError`, `exc.sites` is identical.
  - For F_don, additionally `str(exc).startswith("Function rejected by donation screen (spec §4.2 item 6): ")` (literal from memo.py:515), so a shared drift in message wording is caught even if reference and live code agreed.
- **AC-10 (pin).** Key equivalence. For the T2 key fixture set, two keys equal `_reference_build_key(core, digest, args, kwargs)`, a test-local verbatim copy of `49f3def`'s `build_key` algorithm that reads `core.policy.salt` and `core.stamp`. Its per-leaf digest is `_reference_leaf_digest`, a test-local verbatim copy of `49f3def`'s `_leaf_digest` (memo.py:125-161), NOT the live `memo._leaf_digest` (which T7 edits); only `MemoKeyUnsupportedLeafError` is imported live:
  - `core.build_key(digest, args, kwargs)`;
  - the key that `call()` actually stores, checked as `ref_key in core.cache` after one `wrapped(*args, **kwargs)`.

  The fixture set contains no NaN.
- **AC-11 (pin).** `_fits_default_int` boundaries under both x64 settings:
  - x64 off: True for `2**31 - 1` and `-2**31`; False for `2**31` and `-2**31 - 1`.
  - Inside `with jax.enable_x64():`: True for `2**31` and `2**63 - 1`; False for `2**63` and `-2**63 - 1`.

**#5241: refactors**

- **AC-12.** One walker:
  - `memo.py` defines neither `def _screen_jaxpr` nor `def _screen_donation`. It defines `def _screen_program`.
  - `grep -cE 'for \w+ in \w+\.eqns' src/xtrax/inference/memo.py` prints `1`.

  Red on `49f3def` (prints `2`: the two loops at :392 and :540). AC-8, AC-9 and every existing `TestDonation`/`TestPurityWalk` test still pass.
- **AC-13.** Flatten once per call. Setup: after one warm-up miss on `f(x) = x * 2.0` with `x = np.ones((4,), np.float32)`, the test monkeypatches `jax.tree_util.tree_flatten`, `jax.tree_util.tree_leaves` and `jax.tree_util.tree_structure` with counting pass-through wrappers. Then one cache hit (`copy_on_return=False`, `spot_check_every=0`) records `tree_flatten == 1`, `tree_leaves == 0` and `tree_structure == 0`. Red on `49f3def` (`tree_structure == 1`, `tree_leaves == 1`).
- **AC-14.** Cached int bounds. Setup: after one warm-up miss on `f(x, n) = x + n` with `x = np.ones((4,), np.float32)` and `n = 3`, the test monkeypatches `jax.dtypes.canonicalize_dtype` with a counting pass-through wrapper. Then one cache hit on `(x, 3)` records zero `canonicalize_dtype` calls. Red on `49f3def`. AC-11 still passes.
- **AC-15.** `scripts/prof_memo_hit_loop.py` exists and runs to exit 0. It prints one line matching `^per_call_us=\d+(\.\d+)? key_us=\d+(\.\d+)? screen_us=\d+(\.\d+)?$`. The T3 commit message records the baseline line and the T5 commit message records the after line. There is no numeric threshold: this is a maintainability change with no correctness impact, and wall-clock noise on a shared box makes a threshold flaky.

**#5240**

- **AC-16 (pin + comment).**
  - `grep -n "all-default" src/xtrax/inference/memo.py` prints nothing.
  - The `_wrap` comment states that wrap-time screening runs only for zero-parameter callables, because their only possible call signature is `((), {})`.
  - Pin tests: for `def f(*, a=1.0)` and for `def g(x=1.0)`, each body returning `jax.random.uniform(jax.random.key(0), (4,)) * a` (or `* x`), `memoize_jaxpr(f)` returns without raising and `screen_latched_error is None`. The first call `f()` raises `MemoImpurityError` and latches.
- **AC-17.** NaN payloads are distinct keys. Let `nan_a = struct.unpack("<d", bytes.fromhex("000000000000f87f"))[0]` and `nan_b = struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0]`. The test first asserts `struct.pack("<d", nan_a) != struct.pack("<d", nan_b)` as a control.
  1. `_classify_leaf(nan_a, "STATIC") != _classify_leaf(nan_b, "STATIC")`.
  2. With `f(x, s) = x * s` and `x = jnp.ones((4,), jnp.float32)`, calling `f(x, nan_a)`, then `f(x, nan_b)`, then `f(x, nan_a)` gives stats `misses == 2` and `hits == 1`.
  3. Float subclass, which fails `type(leaf) is float` (:270) and reaches `_static_exact_token` via :278-279 in both modes: with test-local `class F(float): pass`, `_classify_leaf(F(nan_a), "ABSTRACT") != _classify_leaf(F(nan_b), "ABSTRACT")` and `_classify_leaf(F(nan_a), "STATIC") != _classify_leaf(F(nan_b), "STATIC")`.

  All three are red on `49f3def` (items 1 and 3 compare equal because `repr` of every NaN, including `float.__repr__` on a subclass, is `'nan'`; item 2 gives `misses == 1`).
- **AC-18 (pin).** Non-NaN float keys are unchanged:
  - `_classify_leaf(1.5, "STATIC") == ("dyn", ("static", "float", "1.5"))`.
  - `_classify_leaf(-0.0, "STATIC") != _classify_leaf(0.0, "STATIC")`.
  - Golden leaf digests, landed in T2 (so they pass on `49f3def`) and still passing after T7. For each `(leaf, preimage)` below, `h = hashlib.sha256(); memo._leaf_digest(leaf, h)` gives `h.hexdigest() == hashlib.sha256(preimage).hexdigest()`, and the frozen `_reference_leaf_digest` gives the same. The preimage is `f"{type(leaf).__name__}({leaf!r})".encode()` per memo.py:155-156 (no prefix, no separator; `bool` reaches that branch with `__name__ == "bool"`). T2 also writes each expected 64-char hexdigest into the test as a string literal, computed once on `49f3def` and asserted equal to `hashlib.sha256(preimage).hexdigest()`:

    | leaf | preimage (literal) | sha256 hexdigest (measured on `49f3def`) |
    |---|---|---|
    | `2.5` | `b"float(2.5)"` | `9fc15d7f6df8db99bc0dcde0447c9bf5ed6bc2aaccf3f4cfd46a28d4b9f72d98` |
    | `-0.0` | `b"float(-0.0)"` | `9494dcf6094b912eff00023aaee43d28149697b0be0765cd0e091cad8323e21e` |
    | `1e300` | `b"float(1e+300)"` | `361300e047c47d05ece511ef57c019d3c57f58cdafe81273922b1867ebeb431e` |
    | `5e-324` | `b"float(5e-324)"` | `cd705304fa1f0663015d3bb87b4d645c4d8ea0f4d162f46f385e274d6b8d9ce9` |
    | `3` | `b"int(3)"` | `3038d0e4056117cc63ca144b5436861036059825628dddffee5a4c3c0250d829` |
    | `True` | `b"bool(True)"` | `8fe0a14cc6b15c2a958819427d423b6ac1ea2b67fbb074167c1de6582629156c` |

    The hexdigests were measured by running live `memo._leaf_digest` on `49f3def`; all six matched `hashlib.sha256(preimage)`. T2 copies these literals rather than recomputing them.

  - AC-10 (with the frozen `_reference_leaf_digest` copy) still passes after T7.

**Docs and whole-file gates**

- **AC-19.** `docs/api/inference.md` states two things, and `test_ac17_docs_mention_donation_and_screen_ids` still passes:
  - (a) in the **Keys** paragraph, that a Python `float` NaN keys on its bit pattern: `grep -n "bit pattern" docs/api/inference.md` matches a line in the Keys paragraph;
  - (b) that a function with both an impure primitive and a donation marker raises `MemoImpurityError`.
- **AC-20.** Whole-file checks are green on the final branch:
  - `uv run --extra dev pytest tests/inference/test_memo.py -q`;
  - `uv run --extra dev pytest tests/distribution/test_coverage_dag.py -q`;
  - `uv run --extra dev ty check src/`;
  - `ruff check` and `ruff format --check` on every touched file.

## Open Design Questions

| ID | Question | Alternatives | Status | Decision |
|---|---|---|---|---|
| ODQ-1 | #5035: what does a report-only run print when no tier failed? | Always `REPORT (non-blocking): ...`; keep `PASS: ...` when clean | DECIDED | Always print `REPORT (non-blocking): coverage DAG -- no test failures in <tiers>`. A non-blocking step never says PASS, as the backlog suggests. Reserving `PASS` for steps that can fail keeps a grep for `^PASS` meaningful. |
| ODQ-2 | #5035: in enforce mode, what happens when `passed` is true but tests failed (the enforced tier has no floors, or `--enforce` names a tier `--tier` did not select)? | Relabel only (exit 0, `REPORT (not enforced): ...`); make it exit 1; reject the invocation as a usage error | DECIDED | Relabel only. The backlog forbids changing blocking semantics. No recipe or workflow reaches these paths (Justfile:298-313 always pairs `--tier T --enforce T` on floored tiers), so a label-only fix removes the lie with zero CI-behaviour risk. Changing exit codes is a separate gate-policy decision. Single exception, a usage error rather than a gate-policy change: `--enforce ''` today skips the truthy guard at :469 and prints `PASS: coverage DAG enforce` with exit 0, although `audit_coverage_dag` already rejected it as an unknown enforce tier (:380-381). T1's guard `args.enforce is not None and not passed` makes it exit 1, exactly like `--enforce foo` already does (AC-5 d, AC-7). |
| ODQ-3 | #5035: where does "where enforcement lives" come from? | Hardcoded `ENFORCEMENT_RECIPES` plus a Justfile/ci.yml consistency test; generic text `--enforce <tier>`; parse the Justfile at runtime | DECIDED | Hardcoded mapping plus the AC-6 consistency test. It names the actual recipe a reader can run, as the backlog asks. The test makes drift fail on the commit that causes it. Parsing the Justfile at runtime couples a report script to Justfile syntax on every run for no gain over the test. |
| ODQ-4 | #5035: add an end-of-chain advisory summary to `audit-deterministic`? | Add it now; defer | DECIDED | Defer. The verdict line itself now names the count and the enforcement location, which satisfies ACs (a)-(c). A chain-end summary needs cross-recipe state plumbing through `.praxia/coverage_last_measured.json` and edits to the `audit-deterministic` recipe, which AC-7 keeps frozen this sprint. |
| ODQ-5 | #5035: are other `scripts/audit_*.py` report-only over failures? | Relabel others too; none qualify | DECIDED | None qualify. All 22 `print(...PASS...)` sites were grepped. Only `audit_coverage_dag.py:476` is non-blocking. Every other PASS follows a return-1-on-failure check. Scope stays at one script. |
| ODQ-6 | #5241: when a program has both an impure primitive and a donation site, which error does the merged walker raise? | Impurity first (current behaviour); donation first; a combined error listing both | DECIDED | Impurity first, with a byte-identical message, pinned by AC-8/AC-9 before the merge. This is current behaviour on `49f3def`. `MemoDonationError` subclasses `MemoImpurityError`, so impurity is the more general rejection. A combined message would change a user-visible string for no sprint item. |
| ODQ-7 | #5241: may the cache-key format change in the refactor? | Byte-identical; allow a change (the cache is in-process) | DECIDED | Byte-identical. The cache is in-process, so a format change would be tolerable. But threading `leaves`/`treedef` through changes nothing hashed: `repr((treedef,))` is exactly `repr(_structure_token(...))`, and `tree_flatten` leaves equal `tree_leaves` leaves. Identity is free, and AC-10's reference-equivalence test becomes the refactor's proof. |
| ODQ-8 | #5241: how are the flattened leaves threaded without breaking direct callers? | Change `_ensure_screened` to return `(digest, leaves, treedef)`; add private `_ensure_screened_flat(leaves, treedef)` and keyword-only `leaves=`/`treedef=` on `build_key` | DECIDED | Use the second option. Tests call `_ensure_screened((x,), {})` expecting a `str` (`test_memo.py:230,1331,1735`), call `build_key(digest, args, kwargs)` positionally (:232,1334,1736), and monkeypatch `build_key` while scanning positional tuples for `A` (:1553-1572). The keyword-only additions keep all of these working unchanged. |
| ODQ-9 | #5241: how is the speed-up measured? | `stats.cum_hash_seconds` (as the backlog says); a dedicated tracked timing script | DECIDED | A tracked script, `scripts/prof_memo_hit_loop.py`. `cum_hash_seconds` accumulates only on misses (memo.py:812) and excludes `build_key` (:759-766), so a hit loop never moves it. That accounting gap also affects `slow_ratio_warn` and gets a follow-up backlog item; this sprint does not fix it (see Risks). |
| ODQ-10 | #5241: should per-leaf classification and digest work also be fused? | Fuse `_classify_leaf` and `_leaf_digest`; leave them separate | DECIDED | Leave them separate. They produce different things: a structural descriptor that must not read bytes, and a content digest that must. The dominant per-leaf cost is `np.asarray(...).tobytes()`, which a content key cannot avoid. |
| ODQ-11 | #5240: screen all-default callables at wrap time, or correct the comment? | Screen at wrap time with defaults; correct the comment only | DECIDED | Correct the comment. Wrap-time screening is exactly equivalent to first-call screening only when `((), {})` is the sole possible signature, which means zero parameters. For `def f(x=None)` whose `x is None` branch is impure but never called, wrap-time screening would reject a wrapper that works today. It would also surface arbitrary trace errors (not only `MemoImpurityError`) from `memoize_jaxpr` itself, and add a trace at decoration/import time. The docs (`inference.md:317-318`) already say zero-parameter. |
| ODQ-12 | #5240: how are Python-float NaNs keyed? | Document the collision only; bit pattern for NaN only (`"nan:" + struct.pack("<d", v).hex()`), `repr` otherwise; bit pattern for every float | DECIDED | Bit pattern for NaN only, plus a docs sentence. A memo key must be conservative: inputs a function can tell apart must not share an entry. The payload is observable under x64 (`lax.bitcast_convert_type` on the traced f64), and in Python when the leaf is held static (STATIC mode, float subclasses). Under x64-off ABSTRACT tracing the f64-to-f32 conversion may drop low payload bits, so there `nan_a` and `nan_b` can arrive as the same f32 NaN; the key must stay conservative regardless of backend and mode rather than depend on which case applies. Array leaves are already keyed by bits (`tobytes`), so this makes Python floats consistent with them. Keeping `repr` for non-NaN leaves every other key byte-identical (AC-18). The hit-rate cost applies only to callers that vary NaN payloads. |
| ODQ-13 | #5241: fix the `cum_hash_seconds` accounting (it excludes `build_key` and hits) in this sprint? | Fix now; file a follow-up | DECIDED | File a follow-up. No sprint item asks for it. Changing it shifts `slow_ratio_warn` behaviour, which `TestCostAdvisory` pins with a threshold. The orchestrator files it at sprint close. |

## Fixer Tasks

Order: T1 is independent. After it, T2 (pins) → T3 (baseline measurement) → T4 → T5 → T6 → T7 → T8. Use one commit per task. Before every commit, run `git status --short` and confirm that only the task's `Files:` changed. Do not stage `.praxia/` files. Cap threads for every pytest call: `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`.

### T1 — Coverage-DAG verdict labels (#5035)

1. In `scripts/audit_coverage_dag.py`, add a module constant after `DEFAULT_TIER`: `ENFORCEMENT_RECIPES: dict[str, str] = {"tier1_core": "audit-coverage-tier1", "tier2_eda": "audit-coverage-tier2", "tier4_controller": "audit-coverage-tier4"}`.
2. Add `def _failure_segment(result: TierResult) -> str`. It returns `f"{result.tests_failed} test failures in {result.tier_id} (pytest exit {result.pytest_exit_code}); "` followed by either `f"enforcement lives in just {ENFORCEMENT_RECIPES[result.tier_id]}"` when the tier is in the mapping, or `"not enforced by any recipe"` when it is not.
3. Add `def format_verdict(results: tuple[TierResult, ...], *, enforce_tier: str | None, passed: bool) -> str`. It returns the stdout verdict line and never returns `None`. The enforce-FAIL case is handled in `main` before it is called; if called with `enforce_tier is not None and not passed` it raises `ValueError("format_verdict: enforce-fail is reported on stderr by main")`. Set `failing = [r for r in results if r.tests_failed > 0 or r.pytest_exit_code != 0]`. Then:
   - if `enforce_tier is None` and `failing` is non-empty: `"REPORT (non-blocking): coverage DAG -- " + " | ".join(_failure_segment(r) for r in failing)`;
   - if `enforce_tier is None` and `failing` is empty: `"REPORT (non-blocking): coverage DAG -- no test failures in " + ", ".join(r.tier_id for r in results)`;
   - if `enforce_tier is not None` and `not passed`: raise `ValueError` (above);
   - if `enforce_tier is not None`, `passed`, and `failing` is non-empty: `f"REPORT (not enforced): coverage DAG --enforce {enforce_tier} -- " + " | ".join(...)`;
   - otherwise: `"PASS: coverage DAG enforce"`.
4. Replace `main()`'s verdict block (:469-480). Change the enforce-fail guard from the truthy `if args.enforce and not passed` to `if args.enforce is not None and not passed`; its body stays exactly as it is (stderr `FAIL: coverage DAG enforce` plus the failure list, return 1). Otherwise `print(format_verdict(results, enforce_tier=args.enforce, passed=passed))` and return 0. Exit codes are identical to today in every branch with one recorded exception: `--enforce ''` is a usage error and now exits 1 (stderr `FAIL: coverage DAG enforce` and `  - unknown enforce tier: ''`), exactly like any other unknown enforce tier; on `49f3def` it printed `PASS: coverage DAG enforce` and exited 0. See ODQ-2 and AC-7.
5. In `tests/distribution/test_coverage_dag.py`, import `ENFORCEMENT_RECIPES` and `format_verdict`. Leave every existing test unmodified, including `test_main_report_only_exits_zero_with_mocks` (mock stays at `tests_failed=11`) and `test_main_enforce_exits_nonzero_when_below_floor`. Add NEW tests only:
   - `test_main_report_only_labels_failures` for AC-1 (mock `tests_failed=19`, `pytest_exit_code=1`, with `capsys`);
   - AC-2 and AC-3 via `main` with a monkeypatched `audit_coverage_dag` and `capsys`;
   - AC-4 as a direct `format_verdict` call, plus `pytest.raises(ValueError)` for `format_verdict(..., enforce_tier="tier1_core", passed=False)`;
   - AC-5 a (`test_main_enforce_fail_reports_on_stderr`, same mock as the existing enforce test), b and c via `main` with a monkeypatched `audit_coverage_dag`; AC-5 d via `main` with `audit_coverage_dag` unmocked and `run_tier_pytest` monkeypatched to raise `AssertionError`;
   - AC-6.
   - For AC-6, read `ROOT / "Justfile"`. A recipe header is a line matching `^([A-Za-z0-9_-]+)\s*:(?!=)` with no leading whitespace. For each indented body line containing `audit_coverage_dag.py` and matching `--enforce\s+(\S+)`, record `(tier, current_recipe)`. Assert that the set equals `set(ENFORCEMENT_RECIPES.items())`. Then assert that `f"just {recipe}"` is in `(ROOT / ".github/workflows/ci.yml").read_text()` for every recipe.
6. Confirm red-on-main for the new AC-1/AC-2/AC-3/AC-5c/AC-5d assertions by reasoning, not by checkout: `49f3def` prints `PASS: coverage DAG report-only (non-blocking)` and `PASS: coverage DAG enforce` on those paths, and AC-5d exits 0 there. State this in the commit message, including the `--enforce ''` exit-code exception.

Files: `scripts/audit_coverage_dag.py` (modify), `tests/distribution/test_coverage_dag.py` (modify)

Gate:
```
uv run --extra dev pytest tests/distribution/test_coverage_dag.py -q
uv run --extra dev ruff check scripts/audit_coverage_dag.py tests/distribution/test_coverage_dag.py
uv run --extra dev ruff format --check scripts/audit_coverage_dag.py tests/distribution/test_coverage_dag.py
git diff 49f3def --stat -- Justfile .github/workflows/ci.yml   # must print nothing
```

Scope estimate: ~40 LOC script, ~140 LOC tests.

### T2 — Pin memo behaviour before refactoring (#5241 preconditions)

1. Append `class TestSprint260923Pins:` at the end of `tests/inference/test_memo.py`. Add AC-8 as a test. The wrapped function is `f(x): return jax.jit(lambda y: y * 2, donate_argnums=0)(x) + jax.random.uniform(jax.random.key(0), x.shape)`, called with `x = jnp.ones((4,), jnp.float32)`. Assert `type(exc_info.value) is MemoImpurityError`, `"purity screen" in str(...)`, `"donation screen" not in str(...)`, and the latch.
2. Add AC-9. Define module-level test helpers `_reference_screen_jaxpr(closed)` and `_reference_screen_donation(closed, invar_to_leaf)` as verbatim copies of `49f3def`'s `memo.py` lines 382-408 and 524-551. Also copy verbatim, test-locally, the helpers they call: `_iter_subjaxprs` (:418-434), `_eqn_label` (:437-441), `_wrapped_leaf_indices` (:444-470), `_eqn_donation_sites` (:473-505), `_donation_message` (:508-521) and the `_DonationSite` alias (:415). Prefix every copied name with `_ref` (for example `_ref_eqn_label`) and rewrite only the internal call sites to the prefixed names; leave every other character unchanged. Import from `xtrax.inference.memo` / `xtrax.inference.errors` only the three primitive sets (`_STATEFUL_PRIMITIVES`, `_CALLBACK_PRIMITIVES`, `_RANDOM_PRIMITIVES`) and the error classes `MemoImpurityError`, `MemoDonationError`. Add `_reference_screen(closed, traced)`, which runs purity first and then donation, and returns the raised exception, or `None`. For F_don also assert `str(exc).startswith("Function rejected by donation screen (spec §4.2 item 6): ")`. Build `closed` with `leaves, treedef = jax.tree_util.tree_flatten((args, {}))` and `closed = memo._trace_closed(f, leaves, treedef, tuple(range(len(leaves))))`. Compare against the exception from `memoize_jaxpr(f)(*args)`: same `type` (`is`), same `str`, and for donation the same `.sites`. The fixtures, all arrays-only so every leaf is traced:
   - F_imp: `f(p, x)` returns `jax.jit(lambda z: z + jax.random.uniform(jax.random.key(1), z.shape))(lax.cond(p, lambda y: y + jax.random.uniform(jax.random.key(0), y.shape), lambda y: y, x))`, called with `(jnp.array(True), jnp.ones((4,), jnp.float32))`. This gives several offenders at nested paths.
   - F_don: `f(x)`: `a = jax.jit(lambda y: y * 2, donate_argnums=0)(x)`, then `b = jax.device_put(a, donate=True)`, then return `jax.jit(lambda z: jax.jit(lambda w: w + 1, donate_argnums=0)(z))(b)`. This gives both carriers, top-level and nested.
   - F_both: the AC-8 function.
3. Add AC-10. Define `_reference_leaf_digest(leaf, sink)` as a verbatim copy of `49f3def`'s `_leaf_digest` (memo.py:125-161), importing only `MemoKeyUnsupportedLeafError` live. Do NOT import `memo._leaf_digest` into the reference: T7 edits it, and a live import would let AC-10 move with the change it is meant to guard. Define `_reference_build_key(core, digest, args, kwargs)` as a verbatim copy of `49f3def`'s `build_key` body (:724-732). Inline `_structure_token`/`_pytree_leaves` as `repr((jax.tree_util.tree_structure((args, kwargs)),))` and `jax.tree_util.tree_leaves((args, kwargs))`, and call `_reference_leaf_digest` per leaf.
   - Fixtures, using `def g(*args, **kwargs): return args[0] * 1.0` wrapped with `MemoPolicy(salt="s1")`:
     - `(np.ones((3,), np.float32),)`;
     - `(jnp.ones((2, 2), jnp.float32), 3, 2.5, -0.0, True, "é", b"\x00")`;
     - `(jnp.ones((2,), jnp.float32),)` with kwargs `{"a": [jnp.zeros((2,)), 1], "b": {"c": "s"}}`.
   - For each fixture, compute `digest = core._ensure_screened(args, kwargs)` and assert `core.build_key(digest, args, kwargs) == _reference_build_key(...)`. Then call `wrapped(*args, **kwargs)` on a fresh wrapper and assert that the reference key (computed from that wrapper's core) is in `core.cache`.
4. Add AC-11: `_fits_default_int` boundaries with x64 off, and inside `with jax.enable_x64():` (skip that half with `pytest.skip` if `not hasattr(jax, "enable_x64")`, matching `test_cr6_x64_toggle_forces_new_key`).
5. Add AC-18's golden leaf digests as `test_golden_leaf_digests`, parametrized over the six `(leaf, preimage)` rows in AC-18 plus a 64-char hex literal per row. Copy each hex literal verbatim from AC-18's table (measured on `49f3def`); do not recompute it. For each row assert: `hashlib.sha256(preimage).hexdigest() == hex_literal`; live `memo._leaf_digest(leaf, h)` gives `h.hexdigest() == hex_literal`; `_reference_leaf_digest` gives the same. Use `(1e300, b"float(1e+300)")` exactly (Python's `repr(1e300)` is `'1e+300'`).
6. Run the class. Every test must PASS on the current tree, because these are pins. If any fails, stop and report: it means the reference copy is not verbatim or the fixture assumption is wrong. Do not edit `memo.py`.

Files: `tests/inference/test_memo.py` (modify)

Gate:
```
uv run --extra dev pytest tests/inference/test_memo.py -q -k "TestSprint260923Pins"
uv run --extra dev ruff check tests/inference/test_memo.py
uv run --extra dev ruff format --check tests/inference/test_memo.py
git diff --stat -- src/   # must print nothing
```

Scope estimate: ~330 LOC tests (including ~110 LOC of verbatim reference copies).

### T3 — Hit-loop timing script and baseline (#5241 measurement)

1. Create `scripts/prof_memo_hit_loop.py`, following the `scripts/prof_stage1_*.py` header style: docstring with purpose and usage, `argparse`, `logging`.
   - Arguments: `--calls` (default 2000), `--repeats` (default 5), `--n-scalars` (default 4).
   - It wraps `f(x, *s) = x * 2.0 + sum(s)` with `x = np.ones((8,), np.float32)` and `s = tuple(range(n_scalars))` as plain ints, then makes one warm-up miss.
2. For each repeat, the script times three loops with `time.perf_counter()`:
   - (a) `--calls` hits via `wrapped(x, *s)`, giving `per_call_us`;
   - (b) `--calls` direct `core.build_key(digest, (x, *s), {})` calls, giving `key_us`;
   - (c) `--calls` direct `core._ensure_screened((x, *s), {})` calls, giving `screen_us`.

   It then prints the median per-call microseconds as exactly one stdout line: `per_call_us=<.2f> key_us=<.2f> screen_us=<.2f>`. It asserts `wrapped.memo_get_stats()["hits"] == repeats * calls` before printing, as a control that loop (a) really hit.
3. Run it on the current (pre-refactor) tree and put the printed line in the commit message as `baseline (49f3def memo.py): <line>`.

Files: `scripts/prof_memo_hit_loop.py` (create)

Gate:
```
uv run --extra dev python scripts/prof_memo_hit_loop.py --calls 200 --repeats 3
uv run --extra dev ruff check scripts/prof_memo_hit_loop.py
uv run --extra dev ruff format --check scripts/prof_memo_hit_loop.py
```

Scope estimate: ~70 LOC.

### T4 — Merge the purity and donation walkers (#5241 item 3)

1. In `src/xtrax/inference/memo.py`, add `def _screen_program(closed, invar_to_leaf: tuple[int, ...] | None = None) -> None` in place of `_screen_jaxpr` (:382-408) and `_screen_donation` (:524-551). It does one explicit-stack walk:
   - Seed the stack with `[(closed.jaxpr, "jaxpr", True)]`.
   - For each popped `(jaxpr_obj, jaxpr_path, top_level)`, loop over `for eqn in jaxpr_obj.eqns:` and compute `eqn_path = f"{jaxpr_path}.{_eqn_label(eqn)}"`.
   - If `eqn.primitive.name` is in the banned set, append `(name, eqn_path)` to `offenders`.
   - Extend `sites` with `_eqn_donation_sites(eqn, eqn_path, top_level, closed_invars, invar_to_leaf if top_level else None)`.
   - Push sub-jaxprs via `_iter_subjaxprs(param_val, param_name)` as `(sub_jaxpr, f"{eqn_path}.{sub_path}", False)`, in the same param order as today.
   - After the walk, if `offenders` is non-empty, raise `MemoImpurityError` with the message text copied character-for-character from :401-408. Otherwise, if `sites` is non-empty, raise `MemoDonationError(_donation_message(tuple(sites)), sites=tuple(sites))`.
2. Replace the two call pairs, :662-663 and :707-708, with `_screen_program(closed, static_traced)` and `_screen_program(closed, abstract_traced)`, and keep the `# CR-3` comments. Delete `_screen_jaxpr` and `_screen_donation`. Update `_iter_subjaxprs`'s docstring ("Shared traversal used by both ...") to name `_screen_program`. Move the section header comment so that the impurity sets, the helpers and `_screen_program` read in dependency order. Do not rename any other helper.
3. Run `grep -cE 'for \w+ in \w+\.eqns' src/xtrax/inference/memo.py` and confirm it prints `1` (it prints `2` on `49f3def`). Then run the gate. AC-8 and AC-9 (T2) must pass unchanged. They are the proof that the raise order, the messages and `.sites` are identical.

Files: `src/xtrax/inference/memo.py` (modify)

Gate:
```
uv run --extra dev pytest tests/inference/test_memo.py -q -k "TestSprint260923Pins or TestDonation or TestPurityWalk or TestPerSignatureScreen or TestCodeReviewFixes or TestAdmission or TestDeferredScreen"
uv run --extra dev ty check src/
uv run --extra dev ruff check src/xtrax/inference/memo.py
uv run --extra dev ruff format --check src/xtrax/inference/memo.py
```

Scope estimate: net −25 LOC.

### T5 — Flatten once per call and cache int bounds (#5241 items 1-2)

1. Add module constants `_INT32_INFO = np.iinfo(np.int32)` and `_INT64_INFO = np.iinfo(np.int64)`. Change the signature to `_fits_default_int(value: int, x64: bool | None = None) -> bool`: if `x64 is None`, read `bool(jax.config.jax_enable_x64)`; then use `_INT64_INFO` if `x64` else `_INT32_INFO`. Update the docstring: the default int dtype is int64 under x64 and int32 otherwise, which is what `canonicalize_dtype(np.int64)` returns.
2. Give `_classify_leaf` a keyword parameter `x64: bool | None = None` and pass it to `_fits_default_int` (keep the existing positional and keyword `mode` callers working). In `_mode_token`, read `x64 = bool(jax.config.jax_enable_x64)` once at the top, pass `x64=x64` to every `_classify_leaf` call, and use that same `x64` in the token.
3. Split `_ensure_screened`:
   - Move its body from :679 onward into `def _ensure_screened_flat(self, leaves: list, treedef: Any) -> str`.
   - Reduce `_ensure_screened(self, args, kwargs) -> str` to `leaves, treedef = jax.tree_util.tree_flatten((args, kwargs)); return self._ensure_screened_flat(leaves, treedef)`.
   - Change the signature to `build_key(self, digest, args, kwargs, *, leaves: list | None = None, treedef: Any = None) -> str`. If either is `None`, compute `leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))`. Hash `repr((treedef,))` where `repr(_structure_token(args, kwargs))` was hashed, and iterate `leaves`. Every other line stays byte-for-byte the same.
4. In `call()`, flatten once, immediately after `t0 = time.perf_counter()` (:759) and before the existing `try`: `leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))`. This keeps the flatten inside the `t0`..`hash_seconds` region it occupies today (via `_ensure_screened` :678), so `cum_hash_seconds` and `_maybe_warn_slow` accounting do not shift (ODQ-13). Call `digest = self._ensure_screened_flat(leaves, treedef)` inside the existing try, and `key = self.build_key(digest, args, kwargs, leaves=leaves, treedef=treedef)`. Keep `args` and `kwargs` positional, because the race test scans positional tuples. Delete `_pytree_leaves` and `_structure_token` if `grep -n "_pytree_leaves\|_structure_token" src tests` shows no other user.
5. Add AC-13 and AC-14 to `TestSprint260923Pins`, or to a new `class TestSprint260923HotPath:` after it. Use `monkeypatch.setattr(jax.tree_util, "tree_flatten", counting_wrapper(orig))` and the same for the other two, and `monkeypatch.setattr(jax.dtypes, "canonicalize_dtype", ...)`. Install the patches only after the warm-up call.
6. Run `scripts/prof_memo_hit_loop.py` with defaults. Put `after: <line>` and the T3 baseline line in the commit message.

Files: `src/xtrax/inference/memo.py` (modify), `tests/inference/test_memo.py` (modify)

Gate:
```
uv run --extra dev pytest tests/inference/test_memo.py -q -k "TestSprint260923Pins or TestSprint260923HotPath or TestCostAdvisory"
uv run --extra dev pytest tests/inference/test_memo.py -q
uv run --extra dev python scripts/prof_memo_hit_loop.py
uv run --extra dev ty check src/
uv run --extra dev ruff check src/xtrax/inference/memo.py tests/inference/test_memo.py
uv run --extra dev ruff format --check src/xtrax/inference/memo.py tests/inference/test_memo.py
```

Scope estimate: ~40 LOC src, ~60 LOC tests.

### T6 — Correct the wrap-time screening comment and pin all-default deferral (#5240 item 1)

1. In `_wrap` (memo.py, `# Zero-arg (or all-default) ...` at :988), replace the comment with: `# Wrap-time screening runs only for zero-parameter callables: their only possible call signature is ((), {}), so this is exactly the first call's screen. Callables with parameters (defaulted or not) are screened per signature on first call.` Do not change the `if` condition or the try/except.
2. Add AC-16's two pin tests (`def f(*, a=1.0)` and `def g(x=1.0)`) to `TestSprint260923Pins`. Each asserts that wrapping does not raise, that `screen_latched_error is None`, that the first no-argument call raises `MemoImpurityError`, and that the error latches.
3. Run `grep -n "all-default" src/xtrax/inference/memo.py` and confirm that it prints nothing.

Files: `src/xtrax/inference/memo.py` (modify), `tests/inference/test_memo.py` (modify)

Gate:
```
uv run --extra dev pytest tests/inference/test_memo.py -q -k "TestSprint260923Pins or TestAdmission or TestDeferredScreen"
uv run --extra dev ruff check src/xtrax/inference/memo.py tests/inference/test_memo.py
uv run --extra dev ruff format --check src/xtrax/inference/memo.py tests/inference/test_memo.py
```

Scope estimate: ~5 LOC src, ~35 LOC tests.

### T7 — Key Python-float NaNs by bit pattern (#5240 item 2)

1. In `memo.py`, `import math` and `import struct` at the top. Add `def _float_token(value: float) -> str`, which returns `"nan:" + struct.pack("<d", value).hex()` if `math.isnan(value)`, else `repr(value)`. Its docstring should say that `repr` is exact for every non-NaN double but maps all NaN payloads to `'nan'`.
2. Use `_float_token` in exactly three places:
   - In `_classify_leaf`'s STATIC float branch (:273), `("static", "float", _float_token(leaf))`.
   - In `_static_exact_token` (:224-240), return `_float_token(leaf)` when `isinstance(leaf, float)`, else `repr(leaf)`. Add one docstring line.
   - In `_leaf_digest`'s scalar branch (:155-157), use `f"{type(leaf).__name__}({_float_token(leaf)})"` when `isinstance(leaf, float)`, else the existing `f"{type(leaf).__name__}({leaf!r})"`. The non-NaN bytes stay identical.
3. Add AC-17 items 1-3 (including the `struct.pack` control assertion and the `class F(float)` subclass item in both `"ABSTRACT"` and `"STATIC"`) and AC-18's two `_classify_leaf` bullets in a new `class TestSprint260923NanKeys:`. Do not touch T2's `test_golden_leaf_digests` or `_reference_leaf_digest`; confirm they and AC-10 still pass.

Files: `src/xtrax/inference/memo.py` (modify), `tests/inference/test_memo.py` (modify)

Gate:
```
uv run --extra dev pytest tests/inference/test_memo.py -q
uv run --extra dev ty check src/
uv run --extra dev ruff check src/xtrax/inference/memo.py tests/inference/test_memo.py
uv run --extra dev ruff format --check src/xtrax/inference/memo.py tests/inference/test_memo.py
```

Scope estimate: ~15 LOC src, ~45 LOC tests.

### T8 — Docs: NaN keys and the both-hazards error

1. In `docs/api/inference.md`, **Keys** paragraph (:365-367), insert after the first sentence: `Python \`float\` arguments key on their exact value; a NaN keys on its bit pattern, so NaNs with different payloads are distinct keys.`
2. In the donation section, after the sentence ending "lists every offending site." (:335-336), add: `A function that also contains an impure primitive raises \`MemoImpurityError\` for the impurity.` Change "The purity screen uses the same walk." (:332-333) to "Both screens run in one walk."
3. Keep the public-docs voice: terse, state mechanisms directly, no projected misconceptions, no new status section. Do not touch the "Fixed in this release" line.

Files: `docs/api/inference.md` (modify)

Gate:
```
uv run --extra dev pytest tests/inference/test_memo.py -q -k "docs"
uv run --extra dev pytest tests/distribution/test_narrative_docs.py -q
uv run --extra dev python scripts/audit_narrative_docs.py
grep -n "bit pattern" docs/api/inference.md
```

Scope estimate: ~4 lines.

## Risks

| Risk | Mitigation |
|---|---|
| The T2 reference copies are not verbatim, so AC-9/AC-10 pin the wrong behaviour and T4/T5 "pass" against a drifted oracle. | T2 step 5 requires every pin to pass on the untouched `49f3def` `memo.py` before any refactor lands. A failing pin stops the sprint. The reviewer diffs the reference bodies (both screens, the five helpers, `_leaf_digest`, `build_key`) against `git show 49f3def:src/xtrax/inference/memo.py`; only the `_ref` name prefixes may differ. The references import no live helper or digest function, so a T4/T5/T7 edit cannot move the oracle; AC-18's hex literals and AC-9's F_don `startswith` literal anchor them independently of both copies. |
| The merged walker changes offender or site order, and so the message text. | The push order and path construction are copied from both originals, which were already identical. AC-9 compares full `str(exc)` and `.sites` against the reference. |
| AC-13's monkeypatch of `jax.tree_util.*` also counts JAX-internal calls. | JAX internals import from `jax._src.tree_util`, not the public module attributes. The patches go in after warm-up, and the hit path runs no tracing and no `jnp` ops. The `np.ndarray` leaf avoids a device transfer path. If counts are inflated, the fallback is a caller-frame filter: the counting wrapper increments only when `sys._getframe(1).f_globals["__name__"] == "xtrax.inference.memo"`. (Patching `xtrax.inference.memo.jax.tree_util` is not a fallback: it is the same module object as `jax.tree_util`, memo.py:42.) Keep the red control: temporarily call `tree_leaves` in `call()` and confirm AC-13 fails. |
| The `jax.enable_x64` context is unavailable. | AC-11's x64 half skips exactly as `test_cr6_x64_toggle_forces_new_key` does (`test_memo.py:1858-1860`). |
| NaN keying lowers the hit rate for callers that vary NaN payloads. | This is only reachable by constructing payloads (`struct`, bit casts). A conservative key is the documented contract. Non-NaN keys are unchanged (AC-18). |
| Relabelling hides the failure count from someone grepping for `FAIL`. | The REPORT line carries the count, the exit code and the enforcing recipe. `FAIL` remains reserved for steps that block (enforce mode, exit 1). |
| The `ENFORCEMENT_RECIPES` mapping drifts from the Justfile. | The AC-6 test parses the real Justfile and ci.yml and fails on the drifting commit. |
| The `praxia` hook auto-commits fixer output, or `.praxia/` telemetry gets swept into a commit. | Run `git status --short` and a per-commit `--stat` ownership check against the task's `Files:` (fixer preamble). |
| Rollback | Each task is one commit touching only its `Files:`; `git revert <sha>` undoes a task. T4, T5 and T7 each depend only on T2's pins, not on each other's internals. |
| Known unfixed: `cum_hash_seconds` excludes `build_key` and hits (ODQ-9, ODQ-13), so `slow_ratio_warn` mostly measures trace cost. | The orchestrator files a follow-up backlog item at sprint close, citing memo.py:759-768 and :812. |

## References

- Backlog:
  - #5035: coverage-DAG PASS label;
  - #5241: memo hot-path and walker cleanup, deferred /code-review findings on PR #159;
  - #5240: all-default comment and NaN static keys, filed from spec 260922 §11.
- Prior work:
  - PR #159 (`49f3def`): memoize_jaxpr per-signature screening, #5214/#5215/#5216/#5231;
  - PR #158 (`b85e1e5`): memo donation conformance;
  - prior spec `.praxia/docs/specs/260922_memo-screen-hardening.md`: §3.1-§3.7, including §3.7 "Zero-arg wrap-time screening ... unchanged", and §11 follow-ups listing #5240.
- Anchors (`49f3def`), coverage DAG:
  - `scripts/audit_coverage_dag.py`: :277-316 `evaluate_enforce`, :370-397 `audit_coverage_dag`, :457-462 call, :464-480 verdict;
  - `Justfile`: :291-293 `audit-coverage-dag`, :298-313 tier recipes, :328-330 `audit-deterministic`;
  - `.github/workflows/ci.yml`: :62, :89, :92, :123;
  - `tests/distribution/test_coverage_dag.py`: :289-326 report-only test, :329-369 enforce test;
  - `distribution/coverage_dag.toml`: `tier0_audit`/`tier3_port` have `measure_coverage=false`.
- Anchors (`49f3def`), memo:
  - `src/xtrax/inference/memo.py`:

    | Lines | Symbol |
    |---|---|
    | :125-161 | `_leaf_digest` (float branch :155-157) |
    | :187-193 | `_pytree_leaves`/`_structure_token` |
    | :216-221 | `_fits_default_int` |
    | :224-240 | `_static_exact_token` |
    | :243-283 | `_classify_leaf` (STATIC float :273) |
    | :286-312 | `_mode_token` |
    | :382-408 | `_screen_jaxpr` |
    | :418-434 | `_iter_subjaxprs` |
    | :524-551 | `_screen_donation` |
    | :618-669 | `_resolve_static` (screens :662-663) |
    | :671-712 | `_ensure_screened` (screens :707-708) |
    | :714-732 | `build_key` |
    | :748-817 | `call` (timing :759-766, `cum_hash_seconds` :812) |
    | :983-996 | `_wrap` comment |

  - `src/xtrax/inference/errors.py`: :50, :54 (`MemoDonationError(MemoImpurityError)`);
  - `tests/inference/test_memo.py`:
    - :230-233, :1331-1335, :1735-1736: direct `_ensure_screened`/`build_key` callers;
    - :1553-1572: `build_key` monkeypatch race test;
    - :934-945, :1671-1675: `_classify_leaf` callers;
    - :1196-1215: docs test;
    - :1455-1481: `TestCostAdvisory`;
    - :1858-1860: x64 skip idiom;
  - `docs/api/inference.md`: :316-323 admission, :325-341 donation, :365-367 Keys.
- Adversarial round 1: `/tmp/claude/loop/r1_challenger.json`, `/tmp/claude/loop/r1_defender.json`, `/tmp/claude/loop/r1_oracle.json` (verdict REVISE; C10, C11 resolved for the spec).

## Revision log

r1: addressed C1, C2, C3, C4, C5, C6, C7, C8, C9
