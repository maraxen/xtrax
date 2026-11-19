---
title: 'Intent-resolution MVP final review: fixes and float32 rulings'
description: 'Final whole-branch review outcome for the intent MVP: loop-in-map JVP bug fixed, two findings ruled float32 conditioning against float64 truth, re-review 4/4 resolved.'
status: final
task_id: 261023_equinox-decouple-pallas
date: '261023'
verdict: 'PASS (re-review: 4 resolved, 0 not resolved, 1 new Minor parked as backlog #6151)'
base_sha: '964d940'
---
# Intent-resolution MVP final review: fixes and float32 rulings

Branch `spec/equinox-decouple-261023`, fix commit `8470120`. Below are the fix report and the
scoped re-review, kept verbatim from the plan's scratch workspace. The re-review's relative
links pointed into that workspace, which has been deleted. The probe scripts cited in both
(`/tmp/claude-1000/probe_*.py`, `rr_*.py`) were throwaway and are not preserved.

**Rulings**
- Re-review new Minor (`chunked_map` crashes on zero-size outputs, `transforms/map.py:124`): it
  predates this branch, since the file is unchanged on it. Parked as backlog #6151, because the
  final review allows only one fix dispatch.
- The re-review dispatch (`9d27c4e6`) was rejected at admission. Every flagged path was ambient
  churn the reviewer did not write: sandbox stub dotfiles plus praxia telemetry and patches.
  Treated as a reviewer report, not as a failure.
- Lesson: a float32 oracle checked at 1e-5 against a float32 reference is unsafe when a carry
  feeds back into each step. Score such cases against float64 truth.

---


# Final-review fix report

task_id: 261023_equinox-decouple-pallas. Range `964d940..8470120` (one commit).

Fixer dispatch `c2f85a16` (Cursor power) hit its 3600 s wall timeout before writing a report;
praxia then reset the worktree to `964d940` (debt #2962). The controller restored its diff from
`/tmp/claude-1000/final-fix-wip-2005.patch`, verified it, kept part, replaced part, and finished.

## Issue 1 (Critical): budgeted ChunkedMap diff path

Two halves, different verdicts.

**Loop inside a budgeted map — real, fixed.** Root cause: a `StaticWhile` primal collects `ys`
in a carried buffer (batched on axis 0 under vmap); the JVP rule returned stacked `lax.scan`
outputs (batched on axis 1). Under `lax.map(batch_size=2)`, custom_jvp batching returns the rule's
`ys` with time and batch interleaved: `(20, 2, 3)` for `(8, 5, 3)`. Reproduced in plain JAX with
no xtrax (`/tmp/claude-1000/probe_cjvp_batch.py`: stacked rule -> `(20,2,3)`; buffer rule ->
`(8,5,3)`; vmap fine for both). Fix: `loop._scan_collect` (scan writing `ys` into a carried buffer,
same layout as `_while_collect`); the rule uses it when the primal is `StaticWhile`, `_scan` when the
primal is `Scan`. The derivative stays a single scan.
The fixer's version (replay the primal while under `stop_gradient` for the primal outputs) also
fixed the shape but put a `while` in the grad jaxpr (`test_loop_transforms_match_scan_reference`
failed) and computed the primal twice; replaced.

**Map inside a loop closing over the carry — not a bug (float32 conditioning).** The carry feeds
back through `sin(z*w + carry)` and amplifies float32 rounding. Against float64 truth
(`probe_cjvp_f32truth.py`): vmap reference 1.6e-3, plain lax.map 1.6e-3, custom_jvp + lax.map
5.6e-4. In float64, custom_jvp vs reference vs finite differences agree to 1e-12
(`probe_cjvp_fd.py`). The hoisted consts are identical in count/shape/order across all six
conversions (`probe_hoist.py`), refuting the controller's const-mismatch hypothesis.
The fixer's `map.py` JVP change (primal outputs from the primal strategy) was motivated by this
false positive, doubled primal compute, and is not needed for the real fix: reverted.

## Issue 2 (Important): chunked_map remainder-1 peel in two scans — not a bug

Old peel vs vmap, same shapes (`probe_peel_f64.py`, run against `964d940` source):
float32 scale 1: 7.2e-3; float32 scale 0.1: 3.8e-6; float64: 1.5e-11 / 2.8e-14.
The fixer's pad-and-mask rewrite evaluated `fn` on zero padding under vmapped `cond` (NaN risk,
docstring claimed otherwise) and added a trace: reverted; `transforms/map.py` is unchanged.

Reviewer's nested D=3/D=4 grad errors (1.08e2, 1.74e7) are the same effect
(`probe_nested_f64.py`): float64 rel 3e-13 / 3e-9; at D=4 the float32 forward itself is 29.8 vs
32.8 in float64.

## Issue 3 (Minor): loop overrides — fixed (fixer's code, kept)

`StaticWhile` override length != site length raises `IntentError` (site, path `primal`, both
lengths). A loop override with diff `None` records `Scan()` when the diff path is available.

## Red before (on 964d940)

- `test_budgeted_loop_in_map_grad_and_jvp`: TypeError, rule primal `(8,3),(20,2,3)` vs `(8,3),(8,5,3)`.
- `test_static_while_override_length_must_match_record`, `test_loop_override_diff_none_records_scan_when_available`: fail.
- `test_budgeted_map_in_loop_closes_over_carry_grad`, `test_chunked_map_remainder_inside_two_scans`:
  pass on 964d940 once well-conditioned (carry update scaled by 0.1); kept as guards.

## Gates (each its own command)

test_compile 14, test_resolve 30, test_loop 23, test_map (intent) 13, test_sites 23, test_plan 24,
test_context 9, tests/transforms/test_map 13, tests/tiling/test_iterator 27 — all pass.
ruff check --no-fix, ruff format --check, ty: clean. jaxlint: intent clean; 4 docstring findings
in `transforms/map.py` exist at 964d940 (file unchanged). Pre-push CI parity passed.

## Concerns

- The float32-vs-reference oracle at 1e-5 is unsafe for carry-feedback problems; reviewers'
  probes should score against float64 truth.
- The custom_jvp batching mismatch is likely a JAX issue worth an upstream report
  (plain repro in `probe_cjvp_batch.py`); not filed.

---

# Scoped re-review — final-review fixes

Range `964d940..8470120` (HEAD `8470120`). Branch `spec/equinox-decouple-261023`.
Judged only the four items in [final-rereview-brief.md](final-rereview-brief.md), against [final-review.md](final-review.md) Issues 1–3 and [final-fix-report.md](final-fix-report.md).
Backend: CPU (`jax.default_backend()`; CUDA jaxlib is not installed). Float64 probes set `jax.config.update("jax_enable_x64", True)` and cast inputs, including the zero carry, so the truth run is float64.

## Tests

Each file, one command: `OMP_NUM_THREADS=4 uv run --no-sync pytest <file> -q --no-cov -p no:cacheprovider`.

| File | Result |
|---|---|
| `tests/intent/test_compile.py` | 14 passed in 11.77s |
| `tests/intent/test_resolve.py` | 30 passed in 0.98s |
| `tests/intent/test_loop.py` | 23 passed in 15.82s |
| `tests/intent/test_map.py` | 13 passed in 11.02s |
| `tests/transforms/test_map.py` | 13 passed in 4.59s |

## 1. Loop JVP fix — Resolved

`_run_jvp` collects with `_scan` when the primal is `Scan` and with `_scan_collect` otherwise ([loop.py](../../../src/xtrax/intent/loop.py):313-322). `_scan_collect` writes `ys` into a carried buffer ([loop.py](../../../src/xtrax/intent/loop.py):165-184).

Command: `OMP_NUM_THREADS=4 uv run --no-sync python /tmp/claude-1000/rr_item1.py`

Budget `MemoryBudget(bytes=50, estimate=Vmap→100 else 10)`, `chunk_size=2`, `xs (8, 5, 3)` for the `xs`-given cases. Every listed nonzero case matched `vmap(scan)` with no shape error. `plans` stayed 1. The diff visit is `inner_loop#0` (no `inner_loop#1`, no `SiteMismatchError`).

| Case | fwd | grad | jvp primal | jvp tangent |
|---|---|---|---|---|
| StaticWhile, `xs` given, both paths `ChunkedMap(2)` | 5.960e-08 | 0 | 5.960e-08 | 0 |
| Scan primal, `xs` given, both paths `ChunkedMap(2)` | 5.960e-08 | 0 | 5.960e-08 | 0 |
| StaticWhile, outer diff only `ChunkedMap(2)` | 5.960e-08 | 0 | 5.960e-08 | 0 |
| StaticWhile, `xs=None`, length 5 | 0 | 0 | 0 | 0 |
| Scan, `xs=None`, length 5 | 0 | 0 | 0 | 0 |
| `vmap(grad)` of the budgeted StaticWhile case | | 0 | | |

`_scan_collect` versus `_scan` at length 32 and 256, linear and `tanh`, feature 16: values and grads match at max abs 0. Public `loop` grad at length 256 matches `lax.scan` at max abs 0.

**While in the derivative.** Command: `OMP_NUM_THREADS=4 uv run --no-sync python /tmp/claude-1000/rr_item1b.py`

```
== test-shaped grad text_while=False text_scan=True ==
TOP scan
TOP scan
== public xs grad text_while=False text_scan=True ==
TOP scan
TOP scan
```

That is the same check as [test_loop.py](../../../tests/intent/test_loop.py):78-79 (`scan[` present, `while[` absent), which passed. The budgeted grad jaxpr (`rr_item1c.py`) still contains a `while`, and only as the primal jaxpr stored on the inner `custom_jvp_call`. The chunked map on that jaxpr is a `scan` of length 4. The derivative rule is the scan collector, not a `while_loop`.

**Residuals.** Reverse mode does not save the `ys` buffer per step. The buffer is one scan-body invar, shape `(length, 16)`. Compiled temp bytes, linear body:

| | length 32 | length 256 | 256/32 |
|---|---|---|---|
| `_scan` | 240 | 240 | |
| `_scan_collect` | 4544 | 34112 | 7.5 |

Length ratio is 8. A per-step copy of the buffer would scale by about 64. The extra temp at length 256 is about two copies of the `(256, 16)` buffer (16384 bytes each), on top of `_scan`. `tanh` shows the same gap (scan 33216, scan_collect 66896). Public `loop` versus `lax.scan` at length 256: 66936 versus 33256.

**Length 0 is a separate failure.** See New issue.

## 2. False-positive rulings — Resolved

The float32 gaps are conditioning. Float64 agrees, and the float32 chunked path is not far worse than the float32 vmap reference.

Command: `OMP_NUM_THREADS=4 uv run --no-sync python /tmp/claude-1000/rr_item2.py`

Shapes are the final reviewer's: carry case from `probe_focus5.py` (`(4, 4)`, `sin(z * w + carry)`, scale 1, loss `sum(carry)`); nested case from `probe_focus3.py` (`(5, 8, 3)` and `(5, 8, 4)`, two scans, scale 1). `ratio_to_vmap_err` is `(float32 path vs float64) / (float32 vmap vs float64)`.

Carry gradient, the map-in-loop half of Issue 1. Intent override `(Vmap, ChunkedMap(2))` and `(ChunkedMap(2), ChunkedMap(2))` both give:

- grad `xs`: float64 vs float64 `1.776e-15`; float32 vmap vs float64 `1.941e-03`; float32 intent vs float64 `1.038e-04` (ratio 0.053); float32 intent vs float32 vmap `1.837e-03`
- grad `w`: float64 vs float64 `0`; float32 vmap vs float64 `1.353e-03`; float32 intent vs float64 `7.230e-05` (ratio 0.053); float32 intent vs float32 vmap `1.281e-03`

The `1.8e-3` / `1.3e-3` gaps against vmap are the same size as vmap's own distance to the float64 gradient. Intent is closer to the float64 truth than vmap is. Forward matches vmap at 0 in float32. Raw `chunked_map` and `lax.map` agree with vmap in float64 to `1e-15`.

Issue 2, remainder-1 inside two scans (`D=3`) and the divisible control (`D=4`):

- `D=3` forward: float64 vs float64 `4.438e-11`; float32 intent vs float32 vmap `7.215e-03` (the reviewer's number); ratio 0.750. Raw `chunked_map` matches intent.
- `D=3` grad: float64 relative `3.577e-12` on a truth of max `5.493e+05`; float32 intent vs float32 vmap `1.080e+02` (the reviewer's number); ratio 1.001.
- `D=4` forward: float64 `0`; float32 intent vs float32 vmap `0`; both float32 paths sit `1.506e+00` from float64.
- `D=4` grad: float64 relative `1.600e-10` on a truth of max `3.353e+07`. Float32 vmap vs float64 is already `2.095e+07` (relative 0.62). Raw `chunked_map` vs float32 vmap is `1.600e+01` (ratio 1.000). Intent vs float32 vmap is `1.740e+07` (ratio 1.831). That is the same float32 blow-up, not a float64 disagreement. One scan over axis 3 (the peel, not nested) is float64 relative `2.671e-13` and ratio 0.476.

## 3. Loop overrides — Resolved

[resolve.py](../../../src/xtrax/intent/resolve.py):516-537 rejects `StaticWhile` when `primal.length != record.length`, and records `Scan()` when the override's diff is `None` and differentiation was traced. [test_resolve.py](../../../tests/intent/test_resolve.py):520 and :538 passed inside the 30.

Command (same `rr_item1.py` run), execution rather than only the plan:

```
OK length mismatch site=body path=primal constraint=override length 99, site length 4
DIFF_NONE primal=StaticWhile(length=4) diff=Scan(...) diff_reason=override:body
OK diff_none grad max_abs 0.000e+00
```

`StaticWhile(length=99)` on a length-4 loop raises `IntentError` with site `body`, path `primal`, and both lengths. `(StaticWhile(length=4), None)` records `Scan` and its grad matches `lax.scan`.

## 4. Rescaled tests still hit the two paths — Resolved

`test_budgeted_map_in_loop_closes_over_carry_grad` ([test_compile.py](../../../tests/intent/test_compile.py):476) still forces the diff path. `rr_item2.py` printed:

```
rescaled map plan primal/outer_loop#0/primal/inner_map#0 primal=Vmap() diff=ChunkedMap(batch_size=2)
rescaled map plan primal/outer_loop#0/diff/inner_map#0 primal=Vmap() diff=ChunkedMap(batch_size=2)
```

The scale `0.1` does not change the strategy. The JVP actually runs the chunked map. Command: `OMP_NUM_THREADS=4 uv run --no-sync python /tmp/claude-1000/rr_item4.py`

```
== jvp of rescaled chunked diff ==
  scan length=4
    scan length=2
```

Outer scan length 4 is the loop. Inner scan length 2 is `lax.map` over axis 4 with `batch_size=2` (two chunks). A vmap diff path would not be that scan.

`test_chunked_map_remainder_inside_two_scans` ([test_map.py](../../../tests/transforms/test_map.py):175) still takes the peel. Inputs are `(5, 8, 3)`, so the mapped axis modulo 2 is 1. `rr_item2.py`:

```
rescaled peel axis3 primitives {'scan': 3, 'concatenate': 1}
divisible axis2 control primitives {'scan': 2}
```

The extra `concatenate` is the peel in [map.py](../../../src/xtrax/transforms/map.py):112-121. The `0.1` scale does not change that branch.

## New issue

### Minor — length 0 inside a budgeted map crashes in `lax.map`

[src/xtrax/transforms/map.py](../../../src/xtrax/transforms/map.py):124

`_scan_collect` returns the empty buffer when `length == 0` ([loop.py](../../../src/xtrax/intent/loop.py):175-176), and a plain `loop(..., length=0)` succeeds. The same loop under the budgeted `ChunkedMap(batch_size=2)` raises `ZeroDivisionError: integer modulo by zero` in `jax.lax.map`'s reshape, before grad. Reference `vmap(scan)` returns shapes `(8, 3), (8, 0, 3)`. StaticWhile and Scan primals both crash, as does an empty `xs` `(8, 0, 3)`.

`rr_item1b.py` traceback (compiled length 0):

```
map.py:166  chunked_map(...)
transforms/map.py:124  return jax.lax.map(fn, xs, batch_size=batch_size)
jax _compute_newshape: arr.size % math.prod(other_sizes) != 0
ZeroDivisionError: integer modulo by zero
```

Not a regression of the JVP collector. `rr_item1c.py` reproduces it with no intent loop:

```
chunked_map(lambda x: jnp.zeros((0,), x.dtype), ones((8, 3)), batch_size=2)
```

same `ZeroDivisionError`. `vmap` of that function returns `(8, 0)`. Nonzero lengths in item 1 are unaffected.

## Verdict counts

Resolved 4. Not resolved 0. New issue 1 (Minor).
