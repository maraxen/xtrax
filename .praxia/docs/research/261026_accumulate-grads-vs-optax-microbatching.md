---
title: 'accumulate_grads vs optax.microbatching: pre-registered parity probe'
description: 'Pre-registered CPU parity probe: optax micro_grad MEAN matches xtrax accumulate_grads (f32 grad rel L2 7.4e-8, f64 1.7e-16) on mean-of-examples losses; 11 semantic differences listed, batch-coupled losses differ by 24%'
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261026'
confidence: 'measured, deterministic CPU run; one tolerance (f32 aux abs, 9.5e-7 vs 1e-6) is tight'
sources: 'bathos run 72c46a5b (validation stage, outcome pass); script scripts/benchmarks/accumulate_grads_parity.py; sidecar scripts/benchmarks/accumulate_grads_parity.bth.toml'
---
# accumulate_grads vs optax.microbatching: pre-registered parity probe

## Question

The user ruled: deprecate `xtrax.training.grad.accumulate_grads` only if parity with
`optax.microbatching` holds. Hypothesis (sidecar): optax.microbatching computes the same
mean gradient, loss and aux as xtrax.accumulate_grads on equivalent inputs, within float
tolerance. Tolerances fixed before the run: grad rel L2 and loss/aux abs diff <= 1e-6 (f32),
<= 1e-12 (f64). Outcomes `pass` / `differs` / `invalid` are in the sidecar.

## Result

Run `72c46a5b-5693-4da1-aa9b-7217bdeb7173`, stage validation, outcome **pass**, CPU, jax 0.11.1,
optax 0.2.8, equinox 0.13.8. 18 cases ran (n_micro 1/4/7 x mlp_aux, mlp_noaux, eqxleaf_aux x f32/f64),
0 errors. Max over cases, arm (a) xtrax vs arm (b) optax `micro_grad(accumulator=MEAN)`:

| dtype | grad rel L2 | loss abs | aux abs | tolerance |
|---|---|---|---|---|
| f32 | 7.4e-8 | 4.8e-7 | 9.5e-7 | 1e-6 |
| f64 | 1.7e-16 | 8.9e-16 | 8.9e-16 | 1e-12 |

Both arms also match the plain full-batch `jax.value_and_grad` reference (c) at the same level
(f32 grad rel L2 <= 1.4e-7, f64 <= 2.0e-16). Negative controls (n_micro=4): optax SUM instead of
MEAN gives grad rel L2 31.0 (f32 and f64); one microbatch shifted by +0.5 gives grad rel L2 0.19,
loss diff 0.030, aux diff 0.29. Both exceed the tolerance in both dtypes, so `controls_ok` is true.

The f32 aux diff (9.5e-7) is 95% of its tolerance: the bound is absolute while the aux values are O(1),
so it sits within a few f32 ulps of the limit. The outcome is `pass` as registered; a re-run on a
different jax build could land on the other side of it.

## Provenance of the runs (read before citing)

- `f711f8f9`: first invocation, launched with the wrong working directory (the session worktree, not
  the clone). It bound no sidecar and recorded the wrong git hash. Not citable.
- `b34cdfe7`: sidecar stage was `exploration`, so bathos recorded outcome `observed` with
  tentative_outcome `pass` (exploratory runs never record a verdict). The sidecar was then revised
  to `stage_name = "validation"`, `novel = true` (commit after the pre-registration commit; outcome
  criteria and tolerances untouched) and re-run as `72c46a5b`.
- Before the pre-registration commit the script was executed directly once to debug it. Tolerances
  come from the task, not from that execution. The numbers were identical in all three tracked runs
  (deterministic seeds).
- bth needs the process cwd to be the repo; `bth run` has no cwd option, so it was launched from the
  clone directory with `cd`, against the instruction not to.

## Scope of the parity claim

Parity holds for a loss that is the mean of per-example terms, with every microbatch the same size.
It says nothing about memory, speed, or a loss with batch coupling (see difference 2).

## Semantic differences (11; file:line read from source on 261026)

1. **Input layout.** xtrax takes pre-stacked `(n_micro, m, ...)` (`grad.py:14,59-93`). optax takes one
   un-stacked batch and reshapes with `order='F'`, so microbatch k holds strided rows
   (`_microbatching.py:99,414`). Mean-equivalent, but microbatch membership differs.
2. **Gradient granularity.** optax `micro_grad` takes per-example gradients of batch-1 slices via vmap
   (`_microbatching.py:509,644-667`). xtrax differentiates the loss over each whole microbatch
   (`grad.py:108-112`). Equal only for a mean-of-examples loss. Probe (n_micro=4, loss plus
   `mean(err)**2`): xtrax vs optax grad rel L2 0.238, xtrax vs full batch 0.021, optax vs full batch
   0.238 (f32 and f64 agree). A BatchNorm-style or ratio loss would diverge the same way.
3. **Default reduction.** xtrax is always an unweighted mean (`grad.py:119-121`). optax defaults to SUM
   (`_microbatching.py:564`); MEAN must be passed. SUM instead of MEAN is the 31x control above.
4. **Return shape.** xtrax returns `(mean_grads, mean_loss[, mean_aux])` (`grad.py:122,127`). optax returns
   `(grads, IndividualOutputs(values, metrics, aux))` with loss and aux per example
   (`_microbatching.py:34,649-654,662`); the caller averages them (this probe does).
5. **Non-array leaves.** xtrax uses `eqx.filter_value_and_grad` (`grad.py:108,112`). optax uses
   `jax.value_and_grad` (`_microbatching.py:644`). Probe: a model with a callable leaf raised
   `TypeError: Argument '<PjitFunction of <function tanh ...>>' ... is not a valid JAX type`; xtrax ran
   it. A migration needs an `eqx.partition`/`eqx.combine` wrapper (used for arms b and c here).
6. **Uneven batches.** Both refuse and neither weights by size. optax raises `batch_size=10 must be
   divisible by microbatch_size=8` (`_microbatching.py:312`); xtrax raises on ragged leading or second
   axis (`grad.py:68,90`). Probed.
7. **Memory profile (read, not measured).** xtrax's scan stacks all n_micro gradient trees, then means
   them (`grad.py:118-121`): n_micro x |params|. optax carries one accumulator through `fori_loop` but
   vmaps per-example grads, so a microbatch holds m x |params| (`_microbatching.py:433-441,509`). Which
   is smaller depends on n_micro vs m; toy shapes here do not discriminate. Unverified empirically.
8. **Reduction order and dtype.** xtrax is a pairwise `jnp.mean` over the stacked axis. optax is a
   sequential running sum divided by n in the output dtype (`_microbatching.py:190-193`) and raises on
   non-floating outputs (`:69-77`). Observed effect at f32/f64 is at the ulp level (above); bf16 not tested.
9. **Loop and extras.** xtrax: `lax.scan`, no early stop (`grad.py:118`). optax: `lax.fori_loop` with
   `num_real_microbatches` early stop, SUM/MEAN/RUNNING_MEAN/CONCAT, `transform_fn`, `metrics_fn`
   (`_microbatching.py:439,566-568`). Neither uses `jax.checkpoint` (no hits in either file).
10. **argnums.** optax can differentiate several arguments (`_microbatching.py:557,644`); xtrax only the
    first (`grad.py:108-112`).
11. **filter_spec is dead in xtrax.** `accumulate_grads` accepts `filter_spec` but never uses it past the
    default assignment (`grad.py:15,50-51`).

## Not tested

GPU; bf16/f16; memory and wall time; sharded inputs (optax has explicit-sharding handling at
`_microbatching.py:100-122`); n_micro > 7; a microbatch size other than 8; jit-wrapped calls.
Callers of `accumulate_grads` in `src/`: only `grad.py` itself and `agent_plugin.json`; tests in
`tests/training/test_grad.py` and `tests/profiling/test_bench_records.py`.
