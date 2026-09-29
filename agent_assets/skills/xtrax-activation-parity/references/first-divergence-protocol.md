# First-Divergence Protocol

Owner modules for the citation half: `src/xtrax/profiling/record.py`
(`ProbeRecord` schema + guards), `src/xtrax/profiling/claims.py`
(`permitted_claims`, `ClaimClass`), `src/xtrax/profiling/emitters.py`
(`emit_probe_record`). The capture half is
`references/capture-mechanics.md`. When this document and the code
disagree, the code wins.

The mechanics of capture are the easy part. This document is the part that
makes a trace evidence rather than a pile of tensors.

## Step 1 -- Matched inputs are a precondition, not a detail

Both sides must receive **identical inputs, in identical order, in identical
frame**. Assert it; do not assume it. The whole method reduces to noise
otherwise, because a frame or alphabet mismatch is numerically
indistinguishable from an implementation defect:

- **Alphabet**: two token orderings over the same symbol set are both valid
  `int32[L]` arrays, so nothing errors. Decode a known landmark on each side
  and assert it reads the expected symbol *in the alphabet you are handing
  over*.
- **Frame / index convention**: raw-versus-canonical indexing, padding, and
  1-based-versus-0-based offsets all produce plausible-looking wrong answers.
  Assert a landmark index resolves to the same element on both sides.
- **Order**: if either side iterates a dict, a set, or a glob, pin the
  ordering explicitly before capture.

**Make these checks discriminating.** A landmark assertion that passes under
both the right and the wrong convention has told you nothing. Prefer the
paired form: the correct mapping must succeed **and** the plausible wrong
mapping must fail. A check that cannot fail is not a check.

## Step 2 -- The non-degeneracy gate

Run before any comparison, on every captured tensor, on both sides:

| assertion | catches |
|---|---|
| expected `shape` and `dtype` | a capture wired to the wrong intermediate |
| all finite (`np.isfinite(...).all()`) | NaN/inf that would make every comparison vacuously "equal" or vacuously "different" |
| not constant (`np.ptp(...) > 0`, or a variance floor) | an all-zeros or all-ones capture -- **two implementations both emitting zeros agree perfectly** |
| non-empty (`size > 0`) | a capture that recorded nothing and reports no error |

Record the gate's outcome per tensor. A tensor that fails it is **excluded and
reported as excluded** -- never silently compared, and never silently dropped.

## Step 3 -- Build the table

For each tensor name present on **both** sides, in **execution order**:

```
max_abs_diff = np.max(np.abs(ref - port))
max_rel_diff = np.max(np.abs(ref - port) / np.maximum(np.abs(ref), tiny))
exceeded     = not np.allclose(ref, port, atol=atol, rtol=rtol)
```

`atol = rtol = 1e-6` is a reasonable default for f32. State the tolerance in
the output -- a first-divergence claim without its tolerance is unreadable.

Report three things:

1. **The first tensor in execution order whose `exceeded` is true.** This is
   the result. A large divergence at the output is almost always downstream of
   a small one earlier; sorting by magnitude inverts the causal order and
   points at a symptom.
2. **The full ordered table**, so the reader can see the growth profile -- a
   divergence that appears at tolerance and then stays flat reads very
   differently from one that compounds layer over layer.
3. **The two name-set asymmetries**: tensors captured on only one side. A
   missing capture is a different finding from a matching-but-divergent one,
   and conflating them is how a wiring bug gets filed as a numerics bug.

Execution order must come from the capture, not from a sorted name list --
which is why the Quick Start's sink stamps a host-side `step` into the key.
Sorting `"layer_10"` before `"layer_2"` is the default lexicographic trap.

## Step 4 -- The harness negative control

**"No divergence found" is worthless until the harness has demonstrated it can
find one.** Plant a known perturbation on exactly one side, at a known tensor,
and require the table to localize it:

- perturb one tensor by a value comfortably above tolerance (e.g. `+1e-3` at
  `atol=1e-6`);
- assert the reported first-divergence tensor **is** the planted one;
- assert the tensors **upstream** of it are still reported clean -- this is what
  distinguishes a working localizer from one that flags everything;
- then remove the perturbation and re-run.

Keep the control in the harness, not in a comment. A positive control that can
only pass is not a check; pair it with the negative case -- a run with no
perturbation that must report no divergence -- so both directions are exercised.

Two harness-level failure modes this catches that nothing else does: a capture
that silently recorded zero tensors (both name sets empty, so the intersection
loop never runs and the table is trivially clean), and a comparison that reads
the *same* side twice (a copy-paste path bug -- every diff is exactly 0.0,
which looks like a triumphant parity pass).

## Step 5 -- Stamp it, then cite the stamp

A first-divergence table is a measurement, so it becomes a `ProbeRecord` under
the sibling `xtrax-probing` skill's contract. Never hand-write the JSON; emit
via `emit_probe_record` (`src/xtrax/profiling/emitters.py`), which constructs,
validates, writes and returns in one call -- the record lands on disk only
after construction succeeded.

Field mapping for this measurement class:

| field | value | why |
|---|---|---|
| `stage` | **`1`** on CPU; `2` only if genuinely GPU-measured | the trace executes, so it is not stage 0 (cost-analysis, no execution). `stage >= 2` *requires* `platform == "gpu"` **and** a `device_kind`, enforced in `ProbeRecord.__post_init__` |
| `platform` | `"cpu"` or `"gpu"` | must agree with `stage` per the guard above |
| `n_atoms` | the problem scale the trace ran at (**must be `> 0`**) | guarded; a first-divergence result at one scale is not automatically a result at another |
| `metrics` | `max_abs_diff`, `max_rel_diff`, `atol`, `rtol`, `n_tensors_compared`, `n_tensors_excluded` | `metrics` is **float-only** after construction: ints and numeric strings coerce, but booleans are **rejected** outright, so a pass/fail flag cannot be laundered in here |
| `config` | `first_divergent_tensor`, the two implementation identities, the frame/alphabet convention, the negative-control verdict | `config` is `dict[str, str]` and is where non-numeric identity belongs |
| `scopes` / `attribution_method` | leave `None` | those are wall-clock attribution fields; this is not a timing measurement. If you set `scopes`, every measured label must appear in `attribution_method` or construction raises |

**Claim class: `STRUCTURAL`.** `permitted_claims` grants `STRUCTURAL`
unconditionally and `DISPATCH_COUNT` at `stage >= 1`
(`src/xtrax/profiling/claims.py`). `TERM_RANKING` and `END_TO_END` are
**never** granted for this measurement -- they are wall-clock ranking and
scale-extrapolation claims, and a divergence table is neither. Reaching for
them here means the claim has drifted from what was measured; narrow the claim
rather than widening a guard.

Provenance is auto-captured (`git_sha`, `timestamp`, `x64_enabled`,
`jax_version`, `jaxlib_version`, `xla_flags`, `device_kind`) -- leave the
default factories alone. Export `XTRAX_GIT_SHA=$(git rev-parse HEAD)` for the
run: a `-dirty` or `unverified` sha is rejected as unverifiable for any
set-backed claim.

## Step 6 -- What a localized divergence does and does not license

A first tensor is a **location**, not a cause. It bounds the search to the
computation between the last agreeing tensor and the first divergent one; it
does not identify which operation inside that span is wrong. Ligand/atom
selection, frame handling and floating-point accumulation order can all
produce the same localization.

State the residual explicitly. In the motivating case a constant-level defect
was real, filed, and fixed -- and then measured to explain **0.19%** of the
cross-implementation gap, leaving the cause live. "Found a defect" and
"explained the disagreement" are different claims, and only the second closes
an investigation.

Finally: re-run the ladder from L0 after any fix. A repair that changes loaded
constants or parameter wiring invalidates the earlier rungs, and the cheapest
rungs are exactly the ones worth re-running.
