---
name: xtrax-activation-parity
description: This skill should be used when two implementations of the same model disagree numerically and the question is WHERE -- a vendored/published reference versus a JAX port, a wheel versus a source tree, two checkpoints loaded by different loaders. Triggers on "find the first divergence", "which tensor diverges first", "trace activations between two implementations", "my port's outputs don't match the reference", "cross-implementation parity failed but the graded parity test doesn't say why", "capture intermediates without changing the forward pass", "diff parameter inventories against a reference checkpoint", or mentions first-divergence tracing, activation capture via Tap/Sink, ZarrStagingSink for intermediates, zarr_content_digest for capture integrity, or parameter-inventory diffing. Covers the escalation ladder (constants, then parameters, then inputs, then activations), non-perturbing capture through xtrax.stages boundaries, the first-divergence table, and the harness negative control that makes "no divergence found" mean something.
xtrax_version: 0.4.0a11
triggers:
  - first divergence / first-divergence trace / which tensor diverges first
  - cross-implementation parity / reference vs port disagree numerically
  - activation capture / capture intermediates without perturbing the forward pass
  - Tap / Sink / AxisBoundary for observation (not transformation)
  - ZarrStagingSink for encoder intermediates / keyed staging of activations
  - zarr_content_digest / update_array_digest / fsync_tree for capture integrity
  - parameter inventory diff / orphan reference tensor / phantom untrained bias
  - dropped constant / atom_context_num-class constant mismatch
  - non-degeneracy assertion before parity comparison
  - planted-perturbation negative control for a trace harness
---

# xtrax-activation-parity

## Purpose

Localize a cross-implementation numerical disagreement to the **first tensor**
that diverges, instead of guessing from source reading. The output is a
first-divergence table: per-tensor max-abs-diff at a stated tolerance,
reporting the earliest tensor in execution order to exceed it.

Verify-paths (house convention): every rule below cites the module that owns
it. When this skill and the code disagree, the code wins -- then update this
skill.

Scope boundary: this skill is the **diagnostic** you reach for when a parity
gate has already failed. The orchestration of a whole port -- oracle sealing,
jaxtyping contracts, the graded T1..T5 parity gate, port-repair cycles -- is
`agent_assets/workflows/port_validation.yaml` (`p3_parity`). That workflow
tells you parity failed; this skill tells you where.

## Non-Negotiables

1. **Exhaust the ladder below before instrumenting anything.** An activation
   trace is the most expensive rung and the last one. Constants and parameter
   inventories are compared with zero instrumentation.
2. **Capture must not perturb what it observes.** Prefer `sink=` over `tap=`:
   the executor discards a sink's return value structurally, while a tap's
   return value **replaces the step output** (`src/xtrax/stages/executor.py`,
   `_wrap_step`). Never edit a forward pass to return extras -- that changes
   the thing you are measuring and cannot be left in place.
3. **Assert non-degeneracy before comparing.** Two implementations both
   emitting zeros agree perfectly. Every captured tensor gets a
   shape/dtype/finite/non-constant check first, or the comparison is not
   evidence.
4. **Plant a known perturbation and require the harness to localize it.**
   Without that negative control, "no divergence found" is indistinguishable
   from a capture that recorded nothing.
5. **Report the FIRST tensor over tolerance in execution order, not the
   largest.** A large late divergence is usually downstream of a small early
   one; ranking by magnitude inverts the causal order.
6. **`zarr_content_digest` is not a cross-side equality oracle.** It proves a
   store has not changed since you wrote it. For content comparison across
   sides use `update_array_digest` -- see `references/capture-mechanics.md`.

## The Escalation Ladder -- do not skip rungs

| rung | question | instrumentation | why it comes first |
|---|---|---|---|
| **L0 constants** | Do the *loaded* models agree on every scalar hyperparameter, read off the objects rather than the config that built them? | none | a dropped or conflated constant reproduces as a pure numeric gap with no structural symptom |
| **L1 parameters** | Which reference tensors have no home in the port? Which port tensors have no source in the checkpoint? Both directions. | none | an orphan trained bias and a phantom untrained bias are both invisible to output-level parity |
| **L2 inputs** | Are both sides receiving identical inputs, in identical order, in identical frame and alphabet? | assertions only | alphabet/frame/index-convention mismatches masquerade as numerical divergence |
| **L3 activations** | Which intermediate diverges first? | capture harness | only worth building once L0-L2 are clean |

L0 and L1 are decisive far more often than their cost suggests. In the
motivating case (external to this repo; aminx LigandMPNN vs the Dauparas
reference) **three** defects were localized at the constant/parameter level
with no instrumentation at all: a dropped neighbour-count constant, a trained
bias with no slot to load into, and a phantom untrained bias. Only after those
were filed did an activation trace become the right next move -- and the
headline constant defect was then measured to close **0.19%** of the
cross-implementation gap, which is precisely why L3 exists.

> Those numbers were measured **outside this repository** and are not
> reproducible from it -- cite the owning project's run record, never this
> skill. And a defect being *real* is not the same as it being *the driver*:
> confirm the magnitude it explains before closing an investigation.

**L1 is two-directional or it is worthless.** A one-directional inventory
check passes while the defect ships. Compare on `(name, shape, dtype)` and
report both orphan sets separately: reference tensors with no consumer in the
port (a trained weight silently ignored) and port tensors with no source in
the checkpoint (a parameter running untrained at its initialization, every
shape check passing). A count match is not an inventory match -- equal totals
with disjoint membership is exactly the shape of a rename-plus-omission.

## L3 Quick Start: non-perturbing capture

Capture is a `Sink` (`T -> None`, `src/xtrax/stages/boundaries.py`) firing
`io_callback` from the pinned shim, staging into a keyed Zarr store:

```python
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from xtrax.stages._callback import io_callback        # NOT jax.experimental directly
from xtrax.stages.boundaries import AxisBoundary
from xtrax.run import SinkSpec, ZarrStagingSink

class CaptureSink:
    """Sink[T]: observes and returns None. The executor discards its return value."""
    ordered = True                                     # host-observed step order matters here

    def __init__(self, sink: ZarrStagingSink, side: str, name: str) -> None:
        self.sink, self.side, self.name = sink, side, name
        self._step = 0

    def __call__(self, x: jax.Array) -> None:
        def _write(v):
            self.sink.stage((self.side, self.name, self._step), value=np.asarray(v))
            self._step += 1
            return np.zeros((), dtype=np.int32)
        io_callback(_write, jax.ShapeDtypeStruct((), jnp.int32), x, ordered=self.ordered)
        return None

spec = SinkSpec(run_id="parity-ref", output_dir=Path("outputs/parity/ref.zarr"),
                format="zarr", flush_every=1)          # format MUST be set; default is "jsonl"
sink = ZarrStagingSink(spec)
boundary = AxisBoundary(sink=CaptureSink(sink, "ref", "encoder_h1"))
```

Give each side its **own** `output_dir` and `run_id`: one directory cannot
host two run ids (`ZarrStagingSink.__init__` refuses). Then `drain()`,
`finalize()`, `fsync_tree()`, and only then digest.

An `ordered=True` capture on a `Vmap` axis **halts** -- and an ordered capture
on a `ChunkedMap` axis silently discards its `batch_size`. Both, plus every other
trap in the capture path, are enumerated with verify-paths in
`references/capture-mechanics.md`; read it before wiring a harness.

For capture *inside* a forward pass rather than at an axis boundary, the
`AxisBoundary` mechanism does not reach -- call the same shim `io_callback` at
the point of interest and stage into the same sink. `AxisBoundary` gives you
per-axis hooks only (`src/xtrax/stages/executor.py` executes one axis).

## Citing the result

A first-divergence table is a measurement, so it is stamped and claim-scoped
under the sibling `xtrax-probing` skill's `ProbeRecord` contract -- never cited
bare. In short: it executes, so `stage >= 1`; `metrics` is float-only so the
tensor *name* goes in `config`; and the claim class is **STRUCTURAL**. It is
never TERM_RANKING or END_TO_END -- those are timing claims and this is not a
timing measurement. Full mapping in
`references/first-divergence-protocol.md`.

## Additional Resources

Load as needed -- do not read both up front:

- **`references/capture-mechanics.md`** -- the capture API surface with
  verify-paths (Tap/Sink/AxisBoundary, the `_callback` shim, `ZarrStagingSink`,
  the digest and fsync primitives) and the full trap catalogue: what halts,
  what silently degrades, and what the digest does not prove.
- **`references/first-divergence-protocol.md`** -- the method: matched-input
  preconditions, the non-degeneracy gate, building and reading the table, the
  planted-perturbation negative control, and the `ProbeRecord` field mapping.
