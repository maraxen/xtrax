---
title: Ship a runnable compiled artifact, and price wasm honestly
description: 'Sprint spec: give xtrax a portable EXECUTED target, iteratively unblock IREE compilation of aminx''s scoring function (two blockers known, count unknown), and ship a verified artifact in aminx''s wheel; wasm execution is measured to need an emsdk-built IREE runtime with no published prior art, so it is a spike, not a deliverable'
status: draft
task_id: 260909_aminx-wasm-export-sprint
date: '260909'
sprint: '260909'
backlog_ids: ''
---
# Ship a runnable compiled artifact, and price wasm honestly

## The finding

The request is: *"ship compiled binaries (ideally wasm at a minimum) with that
package and forward to my PI."* The operative word is **forward** — the PI has to
run the thing. An artifact that compiles but cannot execute does not satisfy the
request in any degree.

Two facts, both measured this sprint, set the shape of the work.

**First, aminx's scoring function does not compile at all, and the number of
reasons is unknown.** This is the sprint's central risk and it was established by
removing blockers one at a time and re-running.

Blocker 1 is `src/aminx/model/features.py:48`, `return jax.lax.top_k(x, k)` — the
k-nearest-neighbour selection at the heart of ProteinMPNN's graph construction. It
lowers to a `stablehlo.composite` wrapping `chlo.top_k`, which IREE 3.11
**explicitly marks illegal**:

```
features.py:48: error: failed to legalize operation 'stablehlo.composite'
  that was explicitly marked illegal:
  "stablehlo.composite"(...) <{composite_attributes = {k = 40 : i64},
   decomposition = @chlo.top_k.impl, name = "chlo.top_k", version = 1}>
```

The MLIR *carries* the decomposition; IREE refuses to inline it, and
`--iree-input-type=stablehlo` does not change that. It is not a shape problem — it
fails identically with every dimension concrete.

**Blocker 2 was invisible until blocker 1 was removed.** Substituting an
`argsort`-based `top_k` (verified: `chlo.top_k` occurrences drop from 2 to 0) and
recompiling produces a completely different failure, in a different file:

```
aminx/utils/coordinates.py:44: error: expected rank to be smaller or equal to the other rank.
    noise = jax.random.normal(coord_key, coords.shape, dtype=coords.dtype)
```

**There are exactly two, and with both removed the whole path works.** A
prototype removing blocker 1 (argsort substitute) and blocker 2 (noise bypass)
compiles the real scoring function to a **7,707,585-byte artifact** and executes
it against JAX:

```
out[0] shape=()       match=True   max|diff|=4.768e-07
out[1] shape=(40, 21) match=False  max|diff|=1.624e-05   <-- see tolerance, below
out[2] shape=(40,)    dtype=int32  exact=True
```

So Phase B is no longer an open-ended search. It is: port two known fixes
properly, and settle a tolerance. That is a materially different sprint from the
one the previous revision described, and the difference was worth the two hours
it took to establish.

**A lead recorded in the previous revision is now dead, and would have wasted
time.** It suggested blocker 2 might be a configuration flag, since coordinate
noise is meaningless for deterministic scoring. It is not: the noise sits inside
`jax.lax.cond(backbone_noise > 0, add_noise, no_noise, coordinates)`
(`utils/coordinates.py:51-56`), and `lax.cond` traces **both** branches, so
`jax.random.normal` is in the graph whatever `backbone_noise` is set to. Removing
it requires bypassing the call, not configuring it.

**Second, blocker 1's fix is small and it exists.** Three alternative formulations
of the same selection compile cleanly in isolation; only `top_k` fails:

| formulation | result |
|---|---|
| `jax.lax.top_k` | **FAIL** — illegal op `stablehlo.composite` |
| `argsort` + `take_along_axis` | **OK**, 15738 B |
| `jax.lax.sort_key_val` | **OK**, 14135 B |
| `jnp.sort` | **OK**, 12492 B |

Note the qualifier **in isolation**. These were compiled as standalone functions,
not inside aminx's 13.8 MB module, and blocker 2 is proof that in-module behaviour
is what counts. Treat this table as "a substitute plausibly exists", not as "the
substitute is proven".

So the sprint is: unblock the compile iteratively until it succeeds or the cost is
shown to be unreasonable, prove any substitution numerically identical, and ship
what that unlocks. The first sub-task of Phase B is to find out how many blockers
there are, because nothing downstream can be sized until that is known.

**Third, wasm is the unverified path and native is the verified one.** These sit on
opposite sides of the line the request cares about:

| Path | Executes today? | Cost | Distribution |
|---|---|---|---|
| **Portable native** | **Yes — measured** | one compiler flag, plus packaging | one artifact per architecture |
| **wasm32** | **No — cannot be executed at all** | rebuild the IREE runtime under emsdk; **no published prior art** | one artifact, every platform |

wasm is what was asked for and it is the one nothing can run. This document makes
that trade explicit rather than quietly picking a side.

### Measured, not inferred

Run first-hand. Nothing here is quoted from a docstring or inferred from reading
source.

**Toolchain provenance, corrected.** An earlier revision said all of this ran
"against the pinned toolchain (jax 0.11.1, IREE 3.11) in this worktree". That is
true of the xtrax-side rows but **false of every aminx-side row (2, 3, 3b–3f)**,
which ran in aminx's own venv at **jax 0.10.2** — aminx's lock is capped there by
the old xtrax pin. IREE is 3.11 throughout. The discrepancy is real but narrow:
blocker 1's emission was re-checked in both environments and is unchanged (the
`chlo.top_k` composite appears at 0.10.2 and 0.11.1 alike; 0.11.1 only adds
`is_stable = true` to its `composite_attributes`). It matters anyway, because B0's
repin **frees aminx's jax ceiling** — see G3 in B0.

| # | Question | Result |
|---|---|---|
| 1 | How big are aminx's weights? | Flagship `proteinmpnn_v_48_020.eqx.zst` = **6,186,365 B**; largest `ligandmpnn_sc_v_32_002_16.eqx.zst` = **13,268,250 B**; all fifteen total 110 MB |
| 2 | Does the real scoring fn export at concrete shapes? | **Yes** — 6,711,213 serialized bytes, outputs `()` f32, `(40,21)` f32, `(40,)` i32 |
| 3 | Does that artifact compile in IREE? | **No** — `top_k` composite illegal, at concrete *and* symbolic shapes |
| 3b | With `top_k` substituted, does it compile? | **No** — a *different* failure appears at `coordinates.py:44` (`jax.random.normal` rank error). Blocker count unknown |
| 3c | With BOTH blockers removed, does it compile? | **Yes** — 7,707,585-byte artifact, no errors. The count is two |
| 3d | Does that artifact execute and match JAX? | **Yes**, with a caveat — indices exact, scalar `max diff 4.8e-07`, per-element logits `max diff 1.6e-05` which *fails* the default `1e-5` |
| 3e | Does "exactly two" survive a **shape change**? | **Yes** — at `L=17`, `L=40` and `L=128` the export is identical in kind: zero `chlo.top_k`, zero RNG ops, zero composites, and all three compile clean. No third blocker is hiding behind `L=40` |
| 3f | Does the parity margin survive a shape change? | **No — and this is the important one.** `L=17` and `L=128` *pass* the default tolerance; `L=40` *fails* it. The margin is thin and data-dependent, not a systematic gap |
| 4 | Is there a legalizable substitute for `top_k`? | **Yes** — `argsort`+`take_along_axis`, `sort_key_val`, and `sort` all compile in isolation; the argsort form is the one used in the working prototype above |
| 5 | Is a *portable* native artifact executable? | **Yes, at real size** — the 7,027,777 B `L=17` aminx artifact carries `cpu = "x86-64-v2"`, `cpu_features = "+cmov,+mmx,+popcnt,+sse,+sse2,+sse4.2,+cx16,+sahf,+cx8,+crc32,+x87,+fxsr"`, `target_triple = "x86_64-unknown-unknown-eabi-elf"` and **executed correctly** (`max|diff| 7.391e-06`, indices exact). The toy fixtures (12296 B / 12248 B) agree but are no longer the evidence |
| 6 | Can any wasm triple avoid the emsdk runtime? | **No.** All five compile, but IREE **silently overrides** `--iree-llvmcpu-link-embedded=true`, always emitting `system-wasm-wasm_32` |
| 7 | Do symbolic shapes work in general? | **Only for reshape-free programs.** See the boundary below |

**The three-shape sweep (rows 3e and 3f), measured.** The prototype that
established the blocker count ran at exactly one shape, which is the weakest point
a reviewer could press on. Re-running export, compile and execution at two further
lengths — one below the model's 48-neighbour count, one three times longer than the
original — settles both halves of that worry, and the two halves come out
differently:

| L | artifact | out[0] scalar | out[1] per-element `max|diff|` | passes default `1e-5` |
|---|---|---|---|---|
| 17 | 7,027,777 B | 2.384e-07 | 7.391e-06 | **yes** |
| 40 | 7,707,585 B | 4.768e-07 | **1.624e-05** | **no** |
| 128 | 6,992,225 B | 4.768e-07 | 7.808e-06 | **yes** |

Decoded indices (`out[2]`) are bit-exact at all three lengths.

Two consequences, and the second is the one that changes B3.

**The blocker count is safe to state.** Three lengths, spanning the interesting
boundary (`L=17` is *below* the 48-neighbour count the model gathers over), all
produce a module with no illegal construct and all compile. "Exactly two" is no
longer a claim resting on a single lucky shape.

**A single-shape parity test would have been a coin flip.** `L=40` fails the
default tolerance while `L=17` and `L=128` pass it comfortably, at roughly half the
error. So the `1.6e-05` figure is not "IREE is systematically less accurate than
XLA" — it is one draw from a distribution whose tail crosses the default threshold.
Had B3 been written against `L=17` alone it would have gone green and shipped a
gate that fails intermittently on real inputs, which is exactly the class of false
green this sprint exists to remove. **B3 must sweep lengths, and must not infer a
tolerance from one of them.**

Note also that artifact size is **not** monotonic in `L` — `L=40` produces the
largest of the three. Size is dominated by the weight constants, with kernel
specialisation varying non-monotonically on top. Do not use artifact size as a
proxy for input size in any budget assertion.

### The finding that changes Phase B: padding is not score-preserving

**This falsifies B2 as it was written, and it is the most important thing in this
document.** The previous revision told the implementer to "specify the bucket set
and the masking semantics explicitly: a padded residue must not change the score of
a real one, and a test must show that." **No masking convention satisfies that**,
because the problem is upstream of masking.

`model/features.py:183`:

```python
k = min(self.k_neighbors, structure_coordinates.shape[0])
```

`k` is derived from the **padded array length**, not from the masked residue count.
Pad an `L=17` structure into an `L=40` bucket and the k-NN graph goes from 17
neighbours to 40. `distances_masked` sets masked entries to `jnp.inf`, so `top_k`
fills the 23 surplus slots with tied `-inf` picks whose features are then aggregated
into the graph. The real residues' logits move as a result.

Measured, `L=17` padded into a 40-bucket, comparing real residues against the
unpadded call:

| padding convention | logits `max|diff|` | NLL diff (nats) |
|---|---|---|
| control (identical call) | 0.0 | 0.0 |
| `mask=0`, coords 0, resi 0, chain 0 | 5.117 | 0.738 |
| `mask=0`, coords 1e3, resi continues, chain 0 | 0.608 | 0.0229 |
| `mask=0`, coords 1e3, resi continues, chain 1 | 1.142 | 0.0104 |
| `mask=0`, coords **1e6**, resi continues, chain 1 | 1.142 | 0.0104 |
| `mask=0`, coords replicate last real | 5.041 | 0.477 |
| `mask=1` on padding, coords 1e3 | 4.627 | 1.312 |

`1e3` and `1e6` giving **identical** results proves this is not a "move the padding
far enough away" problem — it saturates, and the error does not go to zero.

**Put the scale next to the sprint's other numbers.** This document spends a whole
section deciding a `1.6e-05` parity tolerance. The smallest NLL shift any convention
achieves is `1.04e-02` nats — three orders of magnitude larger, and the same order as
the 0.036-nat self-leak that `scoring/score.py:155-166` documents as a measured,
serious defect (t = 41.3).

Note there is no single "best" convention: the smallest *logit* shift (0.608) and the
smallest *NLL* shift (`1.04e-02`) come from two **different** rows of the table above,
so improving one worsens the other. An earlier revision quoted `2.3e-02` as "the best
available", pairing the winning logit row's NLL figure with the word "best".

**Every measurement in this document is exact-length.** L=17, L=40 and L=128 were
each their own artifact at their own `L`. Bucketing was never tested, and it does
not work. A bucketed artifact would return different numbers than aminx for every
sequence shorter than its bucket — shipped to a PI, that is a silent wrong answer,
which is worse than the crash this sprint set out to prevent.

**Consequence: the sprint ships exact-length artifacts.** See B2, rewritten. Fixing
mask-aware `k` in aminx is the alternative and it is a genuine piece of work with
its own numerical-equivalence burden — it is **not** in this sprint.

**The symbolic-shape boundary (row 7), measured case by case.** This matters
because an earlier draft of this spec claimed shape polymorphism was generally
available and used that to retire a bucketing requirement. It is not general:

| program | result |
|---|---|
| elementwise + reduce, two symbolic dims | compiles, executes correctly |
| rank-3 batched matmul, **one** symbolic dim | `failed to legalize 'stablehlo.dynamic_reshape'` |
| `x.reshape(-1)`, one symbolic dim | `failed to legalize 'stablehlo.dynamic_reshape'` |
| gather → reshape, one symbolic dim | `'stablehlo.reshape' must be statically shaped` |
| matmul, two symbolic dims | `failed to legalize 'stablehlo.dynamic_reshape'` |

The witness that made symbolic shapes look general — `tanh(x @ w).sum()` — is a
rank-2 matmul with no reshape and a scalar output, which is close to the only
shape that survives. **Symbolic shapes are not available to aminx** and this sprint
does not depend on them.

### What the adversarial pass falsified

An earlier revision of this document led with the claim that shape polymorphism
retires the need to bucket aminx's scoring function. An adversarial review ran the
real function and falsified it: `jax.random.permutation` raises
`NotImplementedError` for polymorphic shapes (`utils/decoding_order.py:67`), and
`min(self.k_neighbors, L)` raises `InconclusiveDimensionOperation`
(`model/features.py:183`). Following that up produced the deeper finding above —
the concrete path fails too, for an unrelated reason.

Recording this because the corrected shape of the work is the opposite of the
original: the sprint is **not** about shapes, and the op that actually blocks it
was invisible until someone compiled the real function.

## What "ship a compiled binary" actually requires

Four things must be true at once.

1. **A traceable callable.** aminx has one: `src/aminx/scoring/score.py:114`,
   `score_sequence`, `jax.jit`-wrapped with
   `static_argnames=("multi_state_strategy", "use_rolling_state")`.
2. **A program IREE can legalize.** aminx does **not** have this out of the box,
   but the gap is now fully mapped: exactly two constructs block it, and a
   prototype with both bypassed compiles and runs. Phase B1a.
3. **Weights the artifact can use.** Already satisfied, and this corrects an earlier
   draft: at concrete shapes the Equinox weights are **baked into the module as
   constants** — that is most of the 6.7 MB of MLIR. There is no weight-conversion
   task, and there must not be one: `xtrax.export.hf_weights` is an *input-side
   loader* (`hf_weights.py:115`) with no writer and no artifact-side consumer, so
   emitting safetensors would produce a file nothing in this pipeline reads.
4. **A wheel that carries the artifact, tagged correctly.** aminx's does not:
   `pyproject.toml:220` sets `include-package-data = false` and `:291-292` excludes
   the checkpoints. It also builds `py3-none-any` while `[tool.uv].environments`
   declares `sys_platform == 'darwin'`. Embedding an `embedded-elf-x86_64` artifact
   in a pure wheel would silently ship something unloadable to macOS and arm64.

There is also a prerequisite nobody has done. **aminx pins `xtrax[io]` at git sha
`56a9f551` (`pyproject.toml:26`), and that tree has no `src/xtrax/export/` and no
`src/xtrax/telemetry/`.** Moving to a sha with Phase A also crosses every other
change since, across the ten xtrax modules aminx imports (`xtrax.run`,
`xtrax.tiling`, `xtrax.stages`, `.stages.boundaries`, `.stages.bundle`, `xtrax.eda`,
`xtrax.profiling.*`, `xtrax.checkpoint`, `xtrax.engine`, `xtrax.data.module`). The
repin is one line and an unsized amount of breakage; B0 exists to size it before
anything depends on it.

## Rubric

`.praxia/sprint_rubric.toml` sets `ITEM_CAP = 3` and `DIFFICULTY_BUDGET = 6`
(`quick = 1`, `standard = 3`, `extended = 5`). Its own first line scopes it to the
autonomous loop, and no loop controller exists — the reading Marielle accepted for
sprint 260909.

- **Phase A** (xtrax: A1 portable target, A2 symbolic-shape boundary, A3
  per-element parity, A4 budgets + target list + docs, A5 extra split, A6 skip
  enforcement) — **standard, 3**
- **Phase B** (aminx: replace `top_k`, export, verify, ship) — **extended, 5**
- **Phase C** (wasm execution spike) — **extended, 5**

A + B is 8 points over 2 items. B remains **extended** — it carries a
numerical-equivalence proof on a model's feature extraction, a tolerance
decision, an unsized dependency repin, and a packaging/tagging decision. But it is
no longer *unbounded*: the compile blockers are enumerated and a working prototype
exists end to end, so the largest unknown in the previous revision is closed.

Phase A picked up two tasks it did not originally have — A5 (the extra split) and
A6 (the skip enforcement) — both moved *into* A because they are edits to xtrax files
that Phase B was wrongly assumed to cover. A stays **standard**: they are small,
mechanical, and local to files Phase A already opens.

Phase C is **not scored against this sprint**; it is listed above only for
continuity with the earlier revision. The sprint's total is A + B = 8.

**The residual risk in B is no longer the repin.** An earlier revision said it was,
"which nothing has yet sized" — it has now been sized and it is roughly a half-hour
job (zero removed symbols, 53/54 imported names resolving, five additive
signature changes, no dependency conflict; details in B0).

The real risk was found by the same pass that sized it: **padding is not
score-preserving**, which falsified B2's central acceptance criterion and forced the
artifact to be exact-length rather than bucketed. B2 now absorbs that, and the
open question it leaves is a product one rather than a technical one — *which*
lengths to emit — which only Marielle and the PI's actual use can answer.

B stays **extended**: exact-length emission, a NaN-safe `top_k` replacement, a
tolerance that must be justified rather than chosen, a new CI job that can actually
run B3, and a packaging decision with four options.

**Recommended cut: A + B.** It ends with an artifact the PI can run. Phase C is
excluded and re-filed as research for one reason: its cost cannot be bounded from
evidence. Its central step — build IREE's runtime under emsdk, embed a real
`.vmfb`, execute it, get a correct number — has, as far as this sprint's research
could establish, **never been published by anyone**, for IREE, under Node or
wasmtime or a browser. That is first-time integration work, not a checklist item.

### Decision point 1 — what must the PI be able to run it on? **ANSWERED**

**Correction to an earlier revision of this document.** A previous draft claimed
"there is evidence the answer is *not* Linux: aminx's own `[tool.uv].environments`
declares `sys_platform == 'darwin'`." That was an overread of a file I had only
partially quoted. aminx declares **both**, and says so deliberately
(`aminx/pyproject.toml:294-299`):

```toml
[tool.uv]
# Only resolve for platforms we actually target (including WSL which reports as linux)
environments = [
    "sys_platform == 'darwin' and python_version >= '3.13'",
    "sys_platform == 'linux' and python_version >= '3.13'",
]
```

Linux is not a fallback aminx tolerates — it is a first-class declared target, with
a comment explaining it covers WSL. The evidence I cited for "Linux is probably
wrong" does not say that.

**Decided: Linux x86-64.** It is a declared aminx target *and* the only one this
repo can verify by executing, which is what "verified native artifacts first"
requires. macOS arm64 is a **named follow-up, not a default** — see below.

- **Linux x86-64** → satisfied by Phase A as written. Measured working today.
  **This is the sprint's target.**
- **macOS (arm64)** → IREE can cross-compile to `aarch64-apple-darwin`, but nothing
  in this repo or CI can execute it, so it would ship at `CODEGEN_ONLY` — the exact
  "green over a false fact" the last sprint removed. Needs a Mac runner, or an
  honest statement that the artifact is unverified on the target machine.
- **"Anywhere / in a browser"** → only wasm, and only after Phase C.

**macOS arm64 is deferred as a named decision, not silently dropped.** If the PI
works on a Mac, Phase A's Linux artifact does not serve them and the sprint ends
one step short of the actual request. Cross-compiling to `aarch64-apple-darwin` is
possible but would ship at `CODEGEN_ONLY` — compiled, never executed — which is
precisely the "green over a false fact" the previous sprint removed. Closing it
honestly needs a Mac runner.

**Marielle: if your PI is on macOS, say so and this becomes a Phase A2b with a
hardware dependency.** Silence is read as Linux, which is now a positive choice
backed by aminx's own declaration rather than a default taken for lack of an
answer.

### Decision point 2 — native-first, or take the wasm risk now? **ANSWERED**

**Decided 2026-09-10 (Marielle): ship the verified native artifacts first, before
the wasm spike.**

So the sprint is **Phase A + Phase B**, ending in a runnable artifact that has
actually been executed. **Phase C is not in this sprint.** It stays written down
below — the research is real and the write-up is worth keeping — but it is not
scheduled, not estimated against this sprint's budget, and nothing in A or B may
be shaped around it. wasm is filed as research beside #4856.

This also resolves the rubric: the recommended cut (A + B, 8 points) *is* the
sprint, rather than a recommendation competing with a 13-point alternative.

## Phase A — a target that is both verified and distributable

Branch `feat/export-portable-native`, xtrax. No aminx dependency.

**A1 — add a portable executed target.** In `src/xtrax/export/targets.py`, add a
target beside `NATIVE` keeping `verification_level=EXECUTED` and
`supported_dtypes=_EXECUTABLE_DTYPES`, replacing `--iree-llvmcpu-target-cpu=host`
with `--iree-llvmcpu-target-cpu=x86-64-v2`.

**Name it `NATIVE_PORTABLE`.** The constant needs a name written down here, because
Phase B has to reference it by name and an unnamed target is how the shipped
artifact ends up built with the wrong one (see B4).

**Pass the CPU flag only. Do not pass a target triple.** Measured: adding
`--iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu` produces a **byte-identical
artifact** (same md5), and IREE rewrites the embedded triple to
`x86_64-unknown-unknown-eabi-elf` for its embedded loader regardless. A triple
naming an operating system would be inert *and* would misdescribe an artifact that
commits to no OS.

State the portability claim precisely, because it is narrower than "distributable":
the artifact is an `embedded-elf-x86_64` module with
`cpu_features = "+cmov,+mmx,+popcnt,+sse,+sse2,+sse4.2,+cx16,+sahf,+cx8,+crc32,+x87,+fxsr"`,
using IREE's own ELF loader with no libc or dylib dependency. It still declares
`Module Dependencies: hal, version >= 6, required` — **it is not a standalone
binary and requires an IREE runtime on the recipient's machine.**

On the baseline: x86-64-v2 is not quite "any CPU since 2009". SSE4.2 arrived with
Intel Nehalem in late 2008 but AMD only at Bulldozer in 2011, and Intel Atom lacked
it through Silvermont in 2013. Say "any x86-64 CPU from roughly 2013 onward" or
name the ISA level and stop.

**Do not repurpose `NATIVE`.** Host tuning is correct for its job as a parity
oracle, and `tests/export/test_size_budget.py` records its measured size. Add a
sibling.

**A2 — pin the symbolic-shape boundary; do not claim general support.** Symbolic
shapes work through `export_pipeline` — verified: it returns `verified=True` with
`parity` reporting `max|diff| = 1.192e-07`, and the same artifact re-executes at
n=3/8/40. But that holds **only for reshape-free, `top_k`-free programs**, per the
boundary table above.

So A2 is two tests, not one: a **positive** test that a reshape-free symbolic
export round-trips, and a **negative** test that a program containing a dynamic
reshape fails with a clear diagnostic rather than silently producing something
unverified. Document the boundary next to the target. A single passing positive
test would certify a capability that does not hold for the consumer this exists
for — that is the failure mode this whole sprint is about.

**A3 — parity at more than one input size.** *(Rescoped 2026-09-10 during
implementation; the previous rationale was measured and found false. See below.)*

Add a test that verifies per-element parity across **at least three input sizes** and
across **both** `NATIVE` and `NATIVE_PORTABLE`. Covering the portable target at several
shapes is the substance: Phase B ships a `native-portable` artifact to a real person.
Have it also read the artifact's own `cpu_features` back with `iree-dump-module` and
assert the x86-64-v2 set with no host-only feature — that is the check a build which
silently used the wrong target cannot pass.

**What the previous revision claimed, and why it was wrong.** It said existing tests
compare a *reduced scalar*, where `np.allclose`'s `atol + rtol·|b|` makes the default
`1e-5` far weaker than it reads, citing an observed 3.0e-5 diff against a value of 179.6
passing on a 1.8e-3 effective tolerance. Checked against the real files: the shared
oracle at `tests/export/conftest.py:70` already returns an **array**
(`jnp.stack([model(arr[i]) for i in ...])`), and `test_size_budget.py:77` uses
`jax.vmap`. Array oracles are already the established pattern here, so "add an array
oracle" was work already done.

Measured directly at three cardinalities on a 4→64→8 MLP, the scalar-vs-array gap is
also much smaller than implied — the scalar's effective tolerance came out **2.1–2.3×**
looser, not ~180×, and both pass:

| cardinality | per-element `max|diff|` | scalar-sum `max|diff|` | scalar tol / array tol |
|---|---|---|---|
| 256 | 4.77e-06 | 7.63e-06 | 2.1× |
| 1024 | 5.72e-06 | 1.14e-05 | 2.3× |
| 4096 | 7.63e-06 | 1.14e-05 | 2.3× |

The tolerance asymmetry is real and worth a comment in the test, but it is not the gap.

**The real gap is shape coverage.** Every existing export test runs one fixed 32×8
input. Phase B's sweep found that a single-shape parity test would have been a coin
flip — and that the *green* draw is the more dangerous one, because it hides an
intermittent failure behind a passing check. A3 closes that here, in xtrax, on the
target Phase B ships.

**A4 — budgets, target list, docs.** Adding a fifth target touches more than one
file: `tests/export/test_targets.py:24` asserts `ALL_TARGETS` equals an exact
4-tuple, and `tests/export/test_size_budget.py` parametrises over
`tuple(ALL_TARGETS)`, so a new `EXECUTED` target adds a full compile-and-execute
round to that module. Update both. **The size instruction in the previous revision was inert**: it said
to record "the measured size (12296 B for the spike fixture)", but
`tests/export/test_size_budget.py` holds a flat `32 * 1024` ceiling per target and a
shared `1024` B floor — there is nowhere for a specific byte count to go, and the
spike fixture is not that module's `_fixture_model`. The real edit is a
`"native-portable": 32 * 1024` entry in `SIZE_BUDGET_BYTES`, plus the measured
fixture size added to the docstring's table beside the other four.

**Also fix `CHANGELOG.md` while in the docs pass.** Its 0.4.0a8 entry says `NATIVE`
and `WASM32` are both `EXECUTED`; `src/xtrax/export/targets.py:131-134` registers
`WASM32` at `CODEGEN_ONLY`. The code is right and the changelog is wrong. Left
alone, a reader of the changelog concludes wasm already executes — which is the
premise this whole sprint exists to correct. Update `docs/api/export.md` and
`agent_assets/skills/using-xtrax/references/` — that tree ships in the wheel and its
only gate checks a version marker, not prose, so stale text there reaches consumers
silently.

**A5 — split the `export` extra so a consumer can install the runtime alone.**
This task exists because B0 currently asks for something that does not exist. B0
says to "add the `export` extra" and then, two sentences later, to "depend on
`iree-base-runtime`, not `iree-base-compiler`". Those instructions contradict each
other against the real file — `pyproject.toml:55-65` defines a single `export`
extra carrying **both**:

```toml
export = [
  "iree-base-compiler>=3.11,<4",
  "iree-base-runtime>=3.11,<4",
  "huggingface_hub>=1,<2",
  "safetensors>=0.4,<1",
]
```

There is no runtime-only extra anywhere in the repo, so B0 taken literally pulls in
the ~349 MB compiler it is explicitly trying to avoid. Neither phase owned the fix:
Phase A is scoped "no aminx dependency" and Phase B's branch is aminx-only.

Split it here, in Phase A, where the file lives:

- `export-runtime` — `iree-base-runtime`, `safetensors`, `huggingface_hub`. What a
  consumer needs to *load and run* an artifact.
- `export` — `export-runtime` plus `iree-base-compiler`. What xtrax needs to *build*
  one, and what CI installs.

Use the self-referential alias form (`xtrax[export-runtime]` inside `export`), not
a restated dependency list — a restated list is how sprint 260909 silently dropped
`tyro` and four version floors.

**That form is not yet used anywhere in xtrax's `pyproject.toml`.** An earlier
revision claimed it was "already established in this repo"; it is not, and an
implementer looking for the precedent would not find one. What *was* established is
that it works: it was verified against a hatchling replica of xtrax's build config,
where `uv build` flattens the alias into the concrete requirement list, so the built
wheel's metadata carries the real dependencies rather than a dangling self-reference.

**Name both extras explicitly in B0** so the aminx side cannot guess wrong.

If A5 slips, B0's fallback is to depend on **plain `xtrax` (no extra) plus a direct
`iree-base-runtime`** — `xtrax.export.compile.run_native_vmfb` needs only
`iree.runtime` and xtrax's toolchain imports are lazy, so that combination works
today without any xtrax change. A5 is the better shape because it makes the runtime
set discoverable and versioned in one place; the fallback exists so B0 is never
blocked on it.

**A6 — put a gate behind the "0 skips" invariant.** The Verification section below
requires `export-toolchain-tests` to report **0 skips**, and nothing enforces it:
`.github/workflows/ci.yml:151` runs a bare `uv run pytest tests/export/ -q`, which is
exactly as green with every export test skipped as with all of them run. The
invariant is prose. Run that step with `-rs` and fail it on any `SKIPPED` line, so a
silently-failed IREE install cannot present as a passing real-toolchain claim.

This was named in the Rubric as one of Phase A's tasks but had no numbered task of
its own, which is how an item gets scored and then not built.

**Gate for Phase A:**

```bash
uv run --extra dev --extra io --extra export pytest tests/export/ -q
uv run --extra dev ruff check src/ tests/ && uv run --extra dev ty check src/
just audit-public-api
```

Green, with measured sizes pasted into the PR body.

**The previous revision also required A2's negative test and A3's test to be
"demonstrated **red** against `origin/main` first". That requirement is wrong for both,
and was dropped rather than satisfied by contrivance.** Red-first is the right discipline
for a test that pins a *fix*; neither of these does. Measured before implementation:

- **A2's negative case already fails correctly.** A dynamic reshape under symbolic shapes
  raises `CompileError` naming `stablehlo.dynamic_broadcast_in_dim` as explicitly illegal,
  with the offending source line. The diagnostic is already clear, so A2 *pins* good
  behaviour; the regression it guards against is that diagnostic degrading.
- **A3's case already passes**, at every size measured. It is shape-regression coverage,
  not a bug fix.

Requiring red here would have meant either writing a test that fails for a manufactured
reason, or reporting a red that was not real — both of which are the false-evidence
failure mode this sprint exists to remove, just pointed the other way. What each test
must instead demonstrate is stated in its own task: for A2, that the negative half fails
loudly *and names the op*; for A3, that parity holds at three sizes on both executed
targets, with `cpu_features` read back from the artifact.

## Phase B — make aminx's scoring function compilable, then ship it

Branch `feat/export-scoring-artifact`, aminx. Serial after Phase A.

**B0 — repin xtrax.** `pyproject.toml:26` pins a sha with no `export` and no
`telemetry`. Move to a sha containing Phase A and depend on
**`xtrax[io,export-runtime]`** — *both* extras. This task **adds** an extra; it does
not replace one. The current pin is `xtrax[io]`, and dropping `io` breaks aminx at
import time, not at runtime: `src/aminx/sampling/multistate_poe.py:48` does
`from xtrax.run import SinkSpec, ZarrStagingSink` at module scope, which needs zarr,
which `io` carries. An earlier revision wrote `xtrax[export-runtime]` alone and would
have done exactly that.

Do **not** depend on `xtrax[export]` — that pulls the 349 MB compiler onto a consumer
that only needs to load and run an artifact.

**The previous revision's instruction here was unimplementable.** It said "add the
`export` extra" and then, four lines later, "depend on `iree-base-runtime`, not
`iree-base-compiler` … Split it." You cannot take half an extra, and no runtime-only
extra existed. Hence A5. The size argument behind it stands and is why A5 is worth
doing: the compiler is **349 MB installed** (83 MB wheel) against ~7 MB for the
runtime, and executing an artifact needs only the runtime.

**The repin is NOT the sprint's largest risk, contrary to the previous revision.**
It has now been sized rather than guessed at. The sha range is 34 commits / 204
files / +29,605 −4,386, but restricted to what aminx actually imports:

- **zero** removed symbols, **zero** missing modules, **53 of 54** imported names
  resolve;
- **5** signature changes, **all additive with defaults** — kw-only `ledger` /
  `run_id` / `context` on `Engine.fit` / `fit_sync` / `eval`, a defaulted
  `materialize: bool` field on `AxisBoundary`, and a kw-only `export_safe` on
  `validate_plan_topology`;
- no dependency conflict — aminx is already on `huggingface-hub 1.18.0`, inside
  xtrax's `>=1,<2`.

**Treat B0 as roughly a half-hour job.** The contingency the previous revision
attached to it ("if the repin's breakage is larger than a day, stop and report") was
pointed at the wrong hazard. The sprint's real risk is the padding finding above,
which B2 now absorbs.

**Pin jax explicitly in aminx as part of this repin.** aminx sits at jax 0.10.2 only
because the *old* xtrax capped it at `<0.11`; current xtrax allows `<0.12`. The
moment that cap lifts, any `uv lock --upgrade` floats aminx to 0.11.x and changes
the StableHLO emitter that produced every artifact and every parity number in this
document. Either pin jax in aminx, or re-baseline B3 after the repin — do not let
it float silently.

**One expected `ty` failure is pre-existing and must not be read as repin
fallout.** `src/aminx/host/plan.py:28` imports `DedupSpec` from `xtrax.tiling` inside
a `TYPE_CHECKING` block; `xtrax/tiling/__init__.py` has never exported it (not at
the pinned sha, not at HEAD), and aminx's own `pyproject.toml:162` bans the real
path `xtrax.tiling.dedup`. It is type-check-only and already broken today.

**B1a — port the two known fixes properly.** A throwaway prototype has already
proved the shape of this: with `top_k` substituted and the noise call bypassed,
the real scoring function compiles to a 7.7 MB artifact that executes and matches
JAX. B1a is turning that into real code, not rediscovering it.

**What the prototype does NOT establish**, and B1a must:

- It used **monkeypatches**, not edits. `features.top_k` and
  `features.apply_noise_to_coordinates` were rebound at runtime. The real change
  must live in aminx and must not degrade the JAX path.
- It ran at **one shape only** (`L=40`, `S=1`). Nothing is known about other
  lengths or multi-structure inputs.
- It **removed noise entirely**, which is a semantic change. The previous revision
  left "is `backbone_noise > 0` reachable from scoring?" as an open question for
  B1a. **It is answered: yes, by four routes**, and the answer changes the remedy.

  - `aminx.score(..., backbone_noise=...)` — public, re-exported at
    `aminx/__init__.py:24`.
  - `score_sequence(..., backbone_noise=...)` — a **traced** argument, not in
    `static_argnames` (`scoring/score.py:114-122`).
  - CLI `--backbone-noise` (`cli.py:470`, `cli.py:1042`) — defaults to `"0.0"` but
    accepts a **comma-separated list**.
  - `host/runner.py:273 _make_averaged_score_fn`, selected when
    `spec.average_node_features` (`host/runner.py:499`) — builds one bundle per noise
    level and averages.

  **A documented precondition on `score_sequence` is not sufficient**, for two
  independent reasons. First, the noise-averaging path is a *different callable*
  (`score_sequence_averaged`) that `del`s `backbone_noise` and sources noise from the
  spec, so a precondition on `score_sequence` says nothing about the mode a user
  reaches with `--average-node-features`. Second, `inference/bundle_builder.py:300`
  does
  `backbone_noise=jnp.array(backbone_noise)`, making the predicate always a tracer —
  so an export that accepted noise as an input would compile a noise-stripped graph
  that **silently accepts and ignores** a nonzero argument. That is the exact
  false-green class this sprint exists to remove.

  **Therefore: `average_node_features` scoring is out of scope for the artifact**,
  and the artifact's entry point must not accept a `backbone_noise` parameter at all
  — omitting it is honest, whereas accepting and ignoring it is not. Say both in the
  artifact's provenance.

Two traps, both hit while producing this spec — the second cost a wrong
conclusion that survived into a draft:

- `score_sequence` is wrapped in `@partial(jax.jit, ...)` inside `make_score_fn`,
  so calling it once caches a trace. Compute a reference with the original code
  first and the patched call silently reuses that trace — the patch never reaches
  the graph. Export in a **fresh process**, patched before anything is traced.
- **Verify the patch landed by counting the op in the emitted MLIR.** On the first
  attempt `chlo.top_k` stayed at 2 while the run looked entirely successful; only
  counting caught it. It went to 0 on the second.

**B1b — replace `top_k`, and prove the replacement identical. The failure mode is
NaN, not ties.** The previous revision asked for an index-equality test "including
deliberate ties". That test has been run, and it passes everywhere: `argsort` +
`take_along_axis` matches `jax.lax.top_k` on all-equal rows, half-tied-at-max,
integer duplicates, signed zeros, and an aminx-like masked-sentinel row. Ties are
**not** where this breaks.

NaN is:

```
with-NaN  idx top_k  : [0 9 8 7 6 5 4 3]     <- NaN sorts FIRST
with-NaN  idx argsort: [9 8 7 6 5 4 3 2]     <- NaN sorts LAST
```

A PDB with missing backbone atoms yields NaN coordinates, hence NaN distances, hence
a **silently different neighbour set** — with no error raised. A tie-only
equivalence test goes green and misses it entirely. So: put NaN rows in the
equivalence test, **and** either assert NaN-free coordinates at the export boundary
or define the NaN convention explicitly. Keep the tie cases as regression coverage.

The call site is `model/features.py:48`, `return jax.lax.top_k(x, k)`. Replace it
with an IREE-legalizable formulation; `argsort` + `take_along_axis` and
`sort_key_val` both compile (measured above).

**The equivalence proof is the substance of this sub-task, not a formality.**
`top_k` and a sort-based selection differ in tie-breaking order, and k-neighbour
indices feed graph construction, so a different tie-break changes which edges exist
and can move outputs without any numerical error being visible. The test must
assert index-level equality against `top_k` over randomised inputs — **NaN rows
first**, since that is the one case measured to actually diverge, with tie rows kept
as regression coverage — not merely that scores are close.

Keep the change behind the smallest possible surface. If `top_k` is faster on the
JAX path, keep it there and use the substitute only for export — but then the
exported artifact is no longer the same program as the library, and that must be
stated in the artifact's provenance rather than assumed away.

**B2 — a concrete entry point, at exact lengths. Do not bucket.** `score_sequence`
takes `multi_state_strategy` and `use_rolling_state` as static arguments. Pick one
combination, name it, and export that; do not attempt the cross-product. Fix `S = 1`
and say so.

Shapes are **concrete**, per the boundary in A2 — symbolic shapes are unavailable to
this function.

**The previous revision said to bucket-and-pad, and required that "a padded residue
must not change the score of a real one". That requirement cannot be met** — see
"padding is not score-preserving" above. `k` comes from the padded array length
(`features.py:183`), so padding enlarges the k-NN graph and moves real residues'
logits by up to 5.1. No convention escapes it: the smallest logit shift measured is
0.61 and the smallest NLL shift is `1.04e-02` nats, and those come from *different*
conventions. It saturates, so no choice of pad coordinates fixes it.

So the artifact is **exact-length: one `.vmfb` per `L`, and it refuses any other
`L`.**

- **Emit for a declared list of lengths**, chosen from what the PI will actually
  score, not a power-of-two ladder. Each is an independent compile; measured cost is
  **8.9 s** per artifact and ~7 MB on disk, so a handful is cheap in time and the
  real budget question is disk.
  **Absent an answer, emit `{17, 40, 128}`** — the three lengths B3 already sweeps,
  so the shipped set and the verified set are the same set by construction. This is a
  default that lets B2 start, not an answer: it is a product question, and the real
  list should come from Marielle and the PI's actual sequences. Record in the PR
  which list was used and whether it was the default.
- **The entry point must reject a mismatched `L` loudly**, with an error naming the
  artifact's `L` and the one it was handed. A silent wrong answer is the failure
  this whole finding is about; a refusal is correct behaviour, not a limitation to
  apologise for.
- **State the exact-length restriction in the artifact's provenance and in the PR.**
  Someone will otherwise assume it generalises, which is precisely the assumption
  the measurement above kills.

Mask semantics still need stating (a `mask=0` residue inside a *real* structure is a
different thing from padding), but the mask no longer has to carry a burden it
cannot bear.

**If exact-length proves too restrictive in practice, the fix is mask-aware `k` in
aminx — deriving `k` from the masked residue count rather than the array length.
That is its own piece of work, with its own numerical-equivalence burden against
every existing score, and it is explicitly not in this sprint.**

**B3 — parity against JAX on a real checkpoint.** Not a mock. The existing
`tests/export/test_jax_export_smoke.py` uses a `MockModel` returning zero-filled
tensors, establishing only that a signature serialises. B3 loads a real checkpoint,
runs the JAX path and the compiled artifact on the same real input, and compares.

Compare **out[1], the `(40, 21)` per-element logits array**, not out[0]'s reduced
scalar. This is not a stylistic preference — it is measured. On the working
prototype the scalar **passes** at `max diff 4.8e-07` while the per-element array
**fails** the same default tolerance at `max diff 1.6e-05`. A B3 that compared the
scalar would have reported success over a real discrepancy, which is precisely the
failure mode this sprint exists to remove.

**Settle the tolerance explicitly, and justify it.** Across the three measured
lengths the per-element error runs `7.4e-06 / 1.6e-05 / 7.8e-06` — the same order
of magnitude throughout, consistent with ordinary float32 accumulation differences
between XLA and IREE rather than with a defect. `np.allclose`
computes `atol + rtol·|b|`, so the failures are concentrated on small-magnitude
logits where the effective tolerance collapses toward `atol`. Do **not** simply
widen `rtol` until it passes.

**Establish an absolute tolerance appropriate to the logits' range and say why.
That is the only route — the alternative the previous revision offered is
impossible.** It suggested you could instead "show the divergence is smaller than
the model's own run-to-run variation". That variation is **exactly zero**, measured
both ways at `L=40` on the real checkpoint: two calls with the same key are
bit-identical, and six *different* keys are also bit-identical on `out[0]` and
`out[1]` (only `out[2]`, the decoding order, changes). This is by design —
`scoring/score.py:155-171` replaced the order-dependent AR mask precisely to make
scoring order-free and key-invariant at `backbone_noise=0`. Any nonzero divergence
exceeds zero, so that criterion could never be satisfied by anything. It is deleted
rather than left as a tempting escape hatch. Record the chosen number and its reasoning in
the artifact's provenance — a tolerance picked to make a test green is the same
false green in a different costume.

**B3 needs a CI job that actually runs it, and today none exists.** This is not a
detail — the document names B3 as the sprint's end-to-end evidence and says no other
green check substitutes for it, then never asks for anywhere to run it. aminx's
`.github/workflows/ci.yml:46` installs `--extra cpu --extra dev --extra tests`, with
**no IREE at all**, and runs `pytest -n auto -m "$MARKER"` where PRs use
`MARKER="not slow and not parity_heavy and not parity_audit"`. So a B3 test guarded
by `importorskip("iree.runtime")` **skips green**, and one marked `slow` is
**deselected green**, on every PR. Either way the sprint's central claim would be
gated by a check that never executes.

Add a dedicated aminx CI job that installs **`xtrax[export]` — the compiler extra,
not `export-runtime`** — and runs B3 unmarked and unskipped, failing on a skip. The
compiler is required because this job *builds* the artifacts it then executes:
`xtrax.export.compile` imports `iree.compiler.tools`
(`src/xtrax/export/compile.py:74`), which A5 places in `export` only. `export-runtime`
is what the shipped consumer needs; that asymmetry is the entire point of the split,
and mis-stating this job is how it would collapse back into one extra. Cost is not the
obstacle: the real compile is **8.9 s** for the 13.8 MB MLIR, so the whole
three-length sweep is a sub-two-minute job.

If you would rather build the artifacts once and cache them, say so explicitly and
say where they come from — and have the job re-check their `cpu_features`. A job that
silently executes a stale cached `.vmfb` is the same false green relocated.

**Sweep at least three sequence lengths, and treat that as load-bearing rather
than as thoroughness.** The measured sweep above shows `L=40` failing the default
tolerance while `L=17` and `L=128` pass at roughly half the error. A test written
against any single length would therefore be a coin flip on whether the gate is
green — and a green one would be the more dangerous outcome, because it would hide
an intermittent failure behind a passing check. Include `L=40` explicitly as a
regression case, since it is the known-worst draw.

**B4 — ship it, tagged honestly.** Reverse `include-package-data = false` for the
artifact path only. Report wheel size before and after.

**Compile the shipped artifact with `NATIVE_PORTABLE` (from A1), never `NATIVE`.**
This is the single most consequential wiring instruction in the sprint and the
previous revision omitted it entirely: Phase A built a portable target and Phase B
never said to use it. `NATIVE` is `target-cpu=host`, correct as a parity oracle and
wrong as a deliverable — an artifact tuned to this machine's CPU may fault with an
illegal instruction on the PI's. Every gate in the previous revision would have gone
green over exactly that, which is the false-green class this sprint exists to close,
landing at the worst possible point: the artifact handed to the PI.

**Assert it mechanically, not by eye.** The check is cheap — `iree-dump-module` on
the shipped `.vmfb` and grep the `cpu_features` string. It must be the
`x86-64-v2` feature set recorded in A1 and must **not** contain host-only features
(on this machine, `+avx512*`). A test that reads the artifact's own metadata cannot
be satisfied by a build that silently used the wrong target.

**The wheel tag is not optional.** An `embedded-elf-x86_64` artifact inside a
`py3-none-any` wheel is a wheel that installs cleanly on macOS and then fails at
runtime. The previous revision framed this as a binary; there are **four** options,
and the two it named are not the strongest.

1. **Platform-tagged wheel.** Achievable, but not by default and not obviously:
   aminx has no `setup.py`/`setup.cfg`, only declarative setuptools, so bare
   `uv build` always emits `py3-none-any`. Measured on a replica of aminx's build
   config, the incantation is
   `uv build --wheel -C--build-option=--plat-name=manylinux_2_28_x86_64`. Note also
   that a bare `linux_x86_64` tag is **rejected by PyPI** — it must be a
   `manylinux_*` tag, which then asserts a glibc floor the `embedded-elf-x86_64`
   artifact does not actually have.
2. **Optional download**, wheel stays pure.
3. **A separate platform-tagged companion distribution** (`aminx-artifact-linux-x86_64`)
   that the pure `aminx` wheel depends on under an environment marker — the
   `jaxlib` / `nvidia-*` pattern. `pip install aminx` then stays correct on macOS by
   simply not resolving the artifact there, rather than installing something
   unloadable.
4. **Ship the portable `.mlir` and compile to `.vmfb` on first use, cached.** MLIR is
   architecture-neutral, which sidesteps the tag question entirely. The price is an
   `iree-base-compiler` dependency (349 MB, already priced above) and a one-time
   **8.9 s** compile, both measured.

**Recommended: (3).** It is the only one that stays correct if the macOS question in
Decision 1 is later answered yes, and it does not put a third of a gigabyte into
every install the way (4) does. Decide explicitly and say which in the PR — and note
that whichever is chosen, the gate below asserts it rather than trusting the prose.

Add a CLI path that runs inference through the artifact instead of JAX, plus a test
that the two agree. Do **not** ship weights in the wheel: the artifact already
carries them as constants, and bundling the `.eqx.zst` files as well would duplicate
6 MB for nothing.

**Gate for Phase B:**

```bash
uv run pytest tests/export/ -q
uv build
uv run python scripts/check_shipped_artifact.py dist/*.whl
```

The previous revision's gate was `pytest` + `uv build` while its prose claimed to
gate "the artifact present in the built wheel, and the wheel's tag" — neither
command inspects a wheel. That is a criterion checkable only by the author's own
prose in the PR body, in a repo that has already shipped a coverage step printing
`PASS` over 19 failing tests (#5035). Close it with a real assertion.

`scripts/check_shipped_artifact.py` is new in B4 and asserts against the built
wheels directly. **What it asserts depends on which packaging option was chosen, and
an earlier revision got this wrong:** it recommended option (3) and then wrote a gate
that option (3) fails three of four criteria on, because under (3) the `aminx` wheel
is deliberately pure and holds no artifact at all. So the script takes the chosen
option as an explicit argument and asserts the matching set.

*Under option (1) — a platform-tagged `aminx` wheel:*

1. the artifact path **is** present in `dist/aminx-*.whl`;
2. that wheel's tag is **not** `py3-none-any` — an `embedded-elf-x86_64` artifact in
   a pure wheel installs cleanly on macOS and then fails at runtime;
3. no `.eqx.zst` weight file is present (the artifact carries weights as constants);
4. the artifact's `cpu_features` is the `NATIVE_PORTABLE` set, not host's.

*Under option (3) — the recommended companion distribution:*

1. the artifact **is** present in the companion wheel
   (`dist/aminx_artifact_linux_x86_64-*.whl`) and **absent** from `dist/aminx-*.whl`;
2. the companion wheel's tag is **not** `py3-none-any`, and the `aminx` wheel's tag
   **is** — a pure `aminx` is the whole point of this option, so asserting its purity
   belongs in the gate rather than contradicting it;
3. `aminx`'s metadata declares the companion as a dependency under a platform
   environment marker, so `pip install aminx` resolves it on Linux x86-64 and skips
   it elsewhere. This is the criterion that actually protects the macOS user under
   (3), and it has no analogue under (1);
4. criteria 3 and 4 from option (1) — no weights, `cpu_features` is
   `NATIVE_PORTABLE` — apply unchanged, to the companion wheel.

*Under option (2) or (4)* no artifact ships in any wheel, so the wheel assertions
would pass vacuously. Replace them: gate instead on a test that exercises the
download-or-compile path end to end and checks the resulting artifact's
`cpu_features`. Do not leave the wheel checks in place to go green over nothing.

**Mirror xtrax's pattern — and note the job it mirrors is in the other repo.**
xtrax's `.github/workflows/ci.yml` has a `wheel-smoke` job doing zipfile-based wheel
assertions for `port/` and `controller/`; copy its shape. But **`wheel-smoke` is an
xtrax job, and B4 is aminx work**: aminx's `ci.yml` has only `epic-attribution-lint`
(`:12`) and `unit-tests` (`:26`). An earlier revision told the aminx implementer to
"wire the new script into that job", naming a job in a repository their branch does
not touch. Add a **new** wheel-assertion job to aminx's `ci.yml`, modelled on xtrax's.

Then: green, with B1b's equivalence test — NaN rows and tie rows both — passing, and
B3 passing against a real checkpoint on the per-element output at three lengths.

## Phase C — the wasm spike (DEFERRED, not in this sprint)

> **Not scheduled.** Decision point 2 was answered native-first on 2026-09-10.
> This section is retained as the research record for whenever wasm is picked up;
> it is not part of this sprint's scope, budget, or gates.

**Not a deliverable.** Its output is a written answer and a decision, and it must be
allowed to conclude "no". Time-box it; if C1 and C2 are not both working inside the
box, stop and write up where it stopped. A bounded negative result is the point.

**C1 — build an IREE runtime that can host wasm.** Install emsdk (it downloads a
prebuilt toolchain; it does not build LLVM from source) and build IREE's runtime
through `emcmake` in `local-sync` rather than `local-task` configuration.
`local-sync` avoids the pthread/Web-Worker dependency, and IREE's CPU deployment
guide recommends it for exactly this reason. Note that xtrax's current wasm flags
request `+atomics,+bulk-memory` (`targets.py:139`), which point the other way —
reconcile or record the conflict.

**C2 — execute a real artifact and check a number.** IREE's `simple_embedding`
sample embeds a *static* vmfb, so adapting it demonstrates execution for a shape
the eventual aminx artifact may not share — note that limitation rather than
overclaiming from it. Run under **headless Node**, not a browser: scriptable and
CI-native.

**C3 — record the answer, whatever it is,** into the research row this sprint files
beside #4856, with measured costs: emsdk install time, runtime build time, and
whether the number came out right. A success unlocks raising `WASM32` off
`CODEGEN_ONLY` later. A failure closes the row as answered.

**The risk, plainly:** no published instance of anyone executing an IREE-compiled
wasm artifact under Node, wasmtime, or wasmer was found. IREE's own issue #8327,
"Port the IREE runtime to WebAssembly+JavaScript without Emscripten," has been open
since 2022.

## Verification

Per phase, before any push:

```bash
uv run --extra dev --extra io --extra export pytest tests/export/ -q
uv run --extra dev ruff check . && uv run --extra dev ruff format --check .
uv run --extra dev ty check src/
just audit-project-hygiene
```

Then CI on the PR: all eight checks, with `export-toolchain-tests` reporting
**0 skips** — and note this invariant is currently **prose with no gate behind it**.
The job runs a bare `uv run pytest tests/export/ -q` (`ci.yml:151`), which passes
just as green with every export test skipped as with all of them run. **A6 adds the
enforcement**, not merely a restatement of the expectation. Until A6 lands, treat a
green `export-toolchain-tests` as unproven.

**Do not run `just audit-deterministic` locally.** It chains `tier1_core`, whose
`pytest_args` are `tests/`, so it runs the whole suite, which this machine cannot
survive. Its exit code is also not evidence the suite passed — the coverage-DAG step
prints `PASS` over a non-zero failure count (#5035).

The sprint's end-to-end evidence is **B3**: a real aminx checkpoint, compiled to an
artifact, executed outside JAX, agreeing with JAX on a per-element output. No other
green check substitutes for it.

## Out of scope

- **WebGPU, in any form.** Confirmed dead upstream, not merely blocked: IREE 3.11
  lists no `webgpu` backend, and the dedicated Tint-based backend that once existed
  was dropped rather than fixed. #4856 stays open as the standing question.
- **Symbolic-shape support for aminx.** Blocked by `jax.random.permutation`
  (`decoding_order.py:67`) and `min(k_neighbors, L)` (`features.py:183`) before IREE
  is even reached, and by dynamic-reshape legalization after. Concrete shapes only.
- **IRPA / IREE Parameter Archive.** IREE recommends externalising weights this way
  and warns safetensors is poorly aligned. Irrelevant while weights are constants;
  it becomes relevant only if artifacts-per-checkpoint becomes a problem.
- **Shipping weights in the wheel.** The artifact carries them already.
- **Mask-aware `k` in aminx.** Deriving `k` from the masked residue count instead of
  the padded array length is what bucketing would require. Real work, own
  equivalence burden, not this sprint. It is the reason B2 ships exact-length.
- **Bucketed / padded artifacts.** Falsified above; a padded residue changes a real
  residue's score by up to 5.1 logits.
- **`average_node_features` scoring.** Reaches a different callable that sources
  backbone noise from the spec; the artifact strips noise, so this mode is not
  representable in it.
- **`wasm32-unknown-unknown` without Emscripten.** IREE issue #8327, open since 2022.
- **macOS or Windows artifacts.** Decision 1 is answered Linux x86-64, so these are
  out. Neither is executable by anything this repo runs, so neither could be
  verified here even if built. macOS arm64 is the named follow-up if the PI is on a
  Mac — flagged under Decision 1, not silently dropped.
- **Cross-IREE-version artifact loading.** Only one IREE version is installed, so
  whether a vmfb built against 3.11 loads on a different runtime version is
  **unresolved in either direction**. If the PI's machine will have its own IREE,
  pin the runtime version in aminx's dependency and say so.
- **#5002** (phase-2 orphan gating), **#5012** (`build_run_spec` never sets
  `RunSpec.run_id`), **#5008** (asr's broken aminx submodule pin) — all open, all
  untouched.
