---
title: 'Scope rubric: when an xtrax symbol stays, moves out, or is deprecated'
description: 'An ordered binary-cue rubric (fast-and-frugal tree shape) for judging whether an xtrax class or function earns its place: evidence completeness first, then load-bearing use, owned decisions, redundancy with JAX/optax/orbax, and scope fit; applied to the 261026 audit candidates.'
status: draft
task_id: 261023_equinox-decouple-pallas
date: '261026'
backlog_ids: '6107'
---
# Scope rubric: when an xtrax symbol stays, moves out, or is deprecated

Asked for by the user (261026) instead of a blanket deprecation of the five zero-consumer `eqx.Module`
classes the thin-wrapper audit found (`.praxia/docs/audits/261026_xtrax-thin-wrapper-audit.md`, §3). The
rubric is written as an ordered set of binary cues so it can become a fast-and-frugal tree later (see
"Toward a workflow" below). It is a manual rubric today: every cue is answered by hand from the audit's
evidence, and nothing here has been scored against a reference set.

## The question each cue answers

xtrax's value is the decision layer (context → strategy, recorded in a Plan) plus infrastructure JAX does
not have (bucketing, export composition, sinks, resumable training, CLI). A symbol earns its place when
deleting it would make a consumer re-implement a decision. It does not when deleting it only changes an
import.

## Cues, in order

Each cue exits as soon as it fires. Cheap, deterministic cues come first; the two judgement cues (C4, C5)
run only on what is left.

| # | Cue (binary) | How it is answered | Exit if YES | Exit if NO |
|---|---|---|---|---|
| C0 | **Is the usage evidence complete enough to conclude from absence?** | Every known consumer scanned at its current `origin/main` (Python, notebooks, configs, string-built imports); usage telemetry, if any, covers this symbol's import and call path and has been live long enough to have seen a typical release cycle | continue | **HOLD**: gather evidence, or treat a deprecation release as the probe (below) |
| C1 | **Is it load-bearing inside xtrax?** | Another kept module, a CLI verb, an on-disk format, a schema, or name-based dispatch (`type(x).__name__`) depends on it | **KEEP** | continue |
| C2 | **Does a consumer use it?** | C0's evidence: import or call sites at consumer `origin/main` | continue to C3 (KEEP is likely; C3 decides how) | continue to C3 (deprecation is likely) |
| C3 | **Does it encode a decision or guard that JAX/optax/orbax lack?** | Deletion test: would a caller re-implement a policy, default, guard or contract? Verified by a probe that shows a behaviour difference (e.g. `safe_scan` raises on length 0 where `lax.scan` does not; `chunked_map` peels a size-1 remainder that `lax.map` vmaps) | **KEEP** (and document the difference it encodes) | continue |
| C4 | **Is there a verified equivalent in JAX/optax/orbax, or in a kept xtrax symbol?** | The equivalent read in the installed library's source (not docs) and, when outputs are numeric, a pre-registered parity probe | **DEPRECATE** (alias + warning + codemod, one release) | continue |
| C5 | **Is it in xtrax's scope?** | Is the problem batching, loop/strategy resolution, kernels, export, runs or training infrastructure, or is it a model-family or domain concern (compression, a specific loss family)? | **KEEP** (owes a consumer: file who should use it) | **MOVE OUT** to the consumer or a sibling package, or deprecate if nobody wants it |

Modifiers, applied after the exit:

- **Defects found while answering** (dead fields, a wrong docstring claim) strengthen DEPRECATE and become
  fix items under KEEP. They never decide on their own.
- **Churn** (commits touching the symbol over the last N months) is a weak prior, not a cue: low churn
  plus no use suggests abandonment, but stable infrastructure is also low churn. It may order the review
  queue; it never exits.
- **Consumer-visible identity**: a symbol consumers store in pytrees or compare by class name cannot be
  replaced by a JAX function even if C4 fires; its deprecation shim must return the same class, or the
  verdict becomes KEEP.

## Why C0 comes first

Every DEPRECATE verdict rests on "nobody uses it", and absence of evidence is the cheapest thing to get
wrong. The audit's scan covers `*.py` in six repos at fixed SHAs; it cannot see other repos, notebooks,
string-built imports, or a user's private scripts, and xtrax has no import or call telemetry today. So for
every zero-consumer symbol C0 is currently "no".

That does not stall everything. A one-release deprecation is itself a probe: the `DeprecationWarning` names
the caller (`_renamed.py` uses `skip_file_prefixes`), so an unknown user surfaces during the warning
release instead of breaking at removal. The rubric therefore allows **DEPRECATE (probe)** where C0 is "no"
but C1–C4 all point to deprecation: ship the warning, and remove only after a release with no reports.
Removal requires C0 "yes" or that clean release.

## Applied to the audit's candidates (261026)

Evidence for every row is in the audit; section numbers point there.

| Symbol | C0 | C1 | C2 | C3 | C4 | C5 | Verdict |
|---|---|---|---|---|---|---|---|
| `SafetyManager` (§4 T3) | no (static scan only) | no | no | no: exactly `jit(checkify(fn, errors=float_checks))` + `err.throw()` | yes: that expression | — | **DEPRECATE (probe)**; defects: `check_nans`/`check_infs` never read, docstring overstates `float_checks` |
| `SafetyTrainStep`, `create_train_step(safety=True)` (§4 T3) | no | no (only `create_train_step` reaches it) | no | no | yes: `checkify(trainer.step, ...)` | — | **DEPRECATE (probe)** |
| `WeightedLoss` (§4 T4) | no | no | no | no: `weight * loss_fn(...)` | yes: a kept xtrax symbol, `ComposedLoss`, covers it with traced weights | — | **DEPRECATE (probe)** |
| `MultiTaskLoss` (§4 T4) | no | no | no | no: a `jnp.sum` of losses times a scalar | yes: `ComposedLoss` | — | **DEPRECATE (probe)** |
| `ShardingPolicy` (§3) | no | no | no | weak: regex rules to `PartitionSpec` | no exact JAX equivalent | lives in `xtrax.distributed`, which the user ruled deprecated (261026) | **DEPRECATE (probe)** with the package; the regex-rule idea, if wanted later, is a C5 question for a sharding planner |
| `xtrax.distributed` (§4 T2) | no | yes, weakly: `DataModule(distributed=True)` reads its flag | no | no: JAX's `initialize()` auto-detects Slurm and parses node lists better | yes: `jax.distributed.initialize`, `jax.make_mesh` | — | **DEPRECATE (probe)**, user-ruled; the C1 dependency becomes a `jax.process_count() > 1` guard |
| `sparse_filter_jit`, `make_sparse_forward_fn` (§4 T1) | no | no | no | no (its "BCOO held static" claim is false, probe A1) | yes: `jax.jit`, `functools.partial` | — | **DEPRECATE (probe)** |
| rest of `xtrax.sparse` (`SparsePolicy`, `sparsify_model`, `SparseMaskManager`) | no | no | no | yes: magnitude top-k masking policy | — | **open**: model compression is not batching/strategy/export/runs | **HOLD at C5**: user call between keep (name a consumer) and move out |
| `xtrax.io` (§4 T6) | no | no | no | no | yes: `xtrax.engine.io` (kept) | — | **DEPRECATE (probe)** |
| `partition_labels` (§4 T5) | no | no | no | no | yes: one `jax.tree.map` for `optax.partition` | — | **DEPRECATE (probe)** |
| `accumulate_grads` (§4 T5) | no | no | no | unknown until probed | **unknown**: `optax.microbatching` parity not yet measured | — | **HOLD at C4**: pre-registered parity probe (user-ruled 261026) |
| `make_optimizer` (§4 T5) | — | no | yes (aminx) | yes: a default clip of 1.0 | — | — | **KEEP**; document the default |
| `safe_scan`, `safe_norm`, `chunked_map` (§4 T7–T8) | — | yes (stages, demistify fingerprint) | yes | yes (probed differences) | — | — | **KEEP** |
| `JaxScanIterator`, `WhileLoopIterator` (§3) | — | no | yes (aminx stores them as pytree fields) | no: `lax.scan` / `lax.while_loop` | yes, but consumer-visible identity | — | **KEEP** (identity modifier) |
| `Trainer` (§5.6) | — | yes: `xtrax run` / `resume` | yes | yes | — | — | **KEEP** |

Net against the audit: the audit's six "deprecate or plain dataclass" calls hold, as DEPRECATE (probe)
rather than plain deprecation, because C0 is unanswered for every one of them. Two calls stay open:
`accumulate_grads` (C4, needs the parity probe) and the rest of `xtrax.sparse` (C5, a scope call).

## What this process taught (input for the workflow idea)

- The expensive mistake is concluding from absence. Absence needs its own evidence channel: a complete
  consumer scan plus usage telemetry, and the deprecation warning release as the last-resort probe.
- Several cues are deterministic and cheap (C1 dependency graph, C2 consumer grep, churn); two are
  judgement (C3 deletion test, C5 scope). The deterministic ones should be precomputed fields, and only the
  residual should reach a model.
- C3 and C4 need probes, not reading alone: three of the audit's "thin wrapper" calls flipped once probed
  (`safe_scan`, `safe_norm`, `chunked_map`), and one docstring claim was false (`sparse_filter_jit`).
- Consumer-visible identity (stored pytree fields, class-name dispatch) overrides redundancy. A pure
  rename can still be load-bearing.
- The audit's prescribed deprecation mechanism was wrong in the backlog text (`tombstone.py`); the real
  one is `_renamed.py` + PEP 562 `__getattr__` + an ast-grep codemod. A workflow should read the
  mechanism from the last deprecation's commits, not from prose.

## Toward a workflow

Filed as a praxia idea (261026): a deprecation fast-and-frugal tree with an agentic gather (consumer scan,
telemetry query, churn, dependency graph, probes) feeding these cues as a fixed ordered verdict. Before it
can run unattended it needs usage telemetry with a coverage check (C0), a certified reference set of past
keep/deprecate decisions to measure each cue's firing precision against, and a rule for when the
deprecation warning release counts as a clean probe.
