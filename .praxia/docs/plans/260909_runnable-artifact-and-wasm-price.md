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
(`coordinates.py:50-55`), and `lax.cond` traces **both** branches, so
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

Run first-hand against the pinned toolchain (jax 0.11.1, IREE 3.11) in this
worktree. Nothing here is quoted from a docstring or inferred from reading source.

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
| 5 | Is a *portable* native artifact executable? | **Yes** — `target-cpu=x86-64-v2` (12296 B) and `generic` (12248 B) both execute correctly |
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
   prototype with both bypassed compiles and runs. Phase B1.
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

- **Phase A** (xtrax: portable target, extra split, boundary documented, skip
  enforcement) — **standard, 3**
- **Phase B** (aminx: replace `top_k`, export, verify, ship) — **extended, 5**
- **Phase C** (wasm execution spike) — **extended, 5**

A + B is 8 points over 2 items. B remains **extended** — it carries a
numerical-equivalence proof on a model's feature extraction, a tolerance
decision, an unsized dependency repin, and a packaging/tagging decision. But it is
no longer *unbounded*: the compile blockers are enumerated and a working prototype
exists end to end, so the largest unknown in the previous revision is closed.

Phase A picked up two tasks it did not originally have — A4 (the extra split) and
the skip enforcement — both moved *into* A because they are edits to xtrax files
that Phase B was wrongly assumed to cover. A stays **standard**: they are small,
mechanical, and local to files Phase A already opens.

Phase C is **not scored against this sprint**; it is listed above only for
continuity with the earlier revision. The sprint's total is A + B = 8.

The residual risk in B is concentrated in the repin (B0), which crosses every
xtrax change since a sha that predates two whole subpackages, and which nothing
has yet sized.

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

**A4 — split the `export` extra so a consumer can install the runtime alone.**
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

Use the self-referential alias form already established in this repo
(`xtrax[export-runtime]` inside `export`), not a restated dependency list — a
restated list is how sprint 260909 silently dropped `tyro` and four version floors.
**Name both extras explicitly in B0** so the aminx side cannot guess wrong.

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

**A3 — parity on a per-element output, at more than one input.** `verify_native_parity`
(`src/xtrax/export/parity.py:107`) defaults to `atol=rtol=1e-5`, and `np.allclose`
computes `atol + rtol·|b|`. On a **reduced scalar** that is far weaker than "1e-5"
reads: at L=1024 the observed diff was 3.0e-5 against a value of 179.6, passing
because rtol allowed 1.8e-3. Add a test whose oracle output is an **array, not a
scalar sum**, verified at three input sizes.

**A4 — budgets, target list, docs.** Adding a fifth target touches more than one
file: `tests/export/test_targets.py:24` asserts `ALL_TARGETS` equals an exact
4-tuple, and `tests/export/test_size_budget.py` parametrises over
`tuple(ALL_TARGETS)`, so a new `EXECUTED` target adds a full compile-and-execute
round to that module. Update both, with the measured size (12296 B for the spike
fixture). Update `docs/api/export.md` and
`agent_assets/skills/using-xtrax/references/` — that tree ships in the wheel and its
only gate checks a version marker, not prose, so stale text there reaches consumers
silently.

**Gate for Phase A:**

```bash
uv run --extra dev --extra io --extra export pytest tests/export/ -q
uv run --extra dev ruff check src/ tests/ && uv run --extra dev ty check src/
just audit-public-api
```

Green, with A2's negative test and A3's per-element test both demonstrated **red**
against `origin/main` first, and measured sizes pasted into the PR body.

## Phase B — make aminx's scoring function compilable, then ship it

Branch `feat/export-scoring-artifact`, aminx. Serial after Phase A.

**B0 — repin xtrax, and size the breakage before committing to the rest.**
`pyproject.toml:26` pins a sha with no `export` and no `telemetry`. Move to a sha
containing Phase A and add the `export` extra.

**Depend on `iree-base-runtime`, not `iree-base-compiler`.** Measured: the compiler
is **349 MB installed** (83 MB wheel) against ~7 MB for the runtime. Executing an
artifact needs only the runtime; the compiler is a build-time tool. Putting the
whole `export` extra into aminx's runtime dependencies would add a third of a
gigabyte to every `pip install aminx`. Split it.

If the repin's breakage is larger than a day, **stop and report** rather than
absorbing it silently — that is a separate piece of work and it should be visible.

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
- It **removed noise entirely**, which is a semantic change if any scoring caller
  ever passes `backbone_noise > 0`. Establish whether that is reachable in
  scoring; if it is, the export path needs a documented precondition
  (`backbone_noise == 0`) enforced at the boundary rather than silently assumed.

Two traps, both hit while producing this spec — the second cost a wrong
conclusion that survived into a draft:

- `score_sequence` is wrapped in `@partial(jax.jit, ...)` inside `make_score_fn`,
  so calling it once caches a trace. Compute a reference with the original code
  first and the patched call silently reuses that trace — the patch never reaches
  the graph. Export in a **fresh process**, patched before anything is traced.
- **Verify the patch landed by counting the op in the emitted MLIR.** On the first
  attempt `chlo.top_k` stayed at 2 while the run looked entirely successful; only
  counting caught it. It went to 0 on the second.

**B1b — replace `top_k`, and prove the replacement identical.** `model/features.py:48`
is `return jax.lax.top_k(x, k)`. Replace it with an IREE-legalizable formulation;
`argsort` + `take_along_axis` and `sort_key_val` both compile (measured above).

**The equivalence proof is the substance of this sub-task, not a formality.**
`top_k` and a sort-based selection differ in tie-breaking order, and k-neighbour
indices feed graph construction, so a different tie-break changes which edges exist
and can move outputs without any numerical error being visible. The test must
assert index-level equality against `top_k` over randomised inputs **including
deliberate ties**, not merely that scores are close.

Keep the change behind the smallest possible surface. If `top_k` is faster on the
JAX path, keep it there and use the substitute only for export — but then the
exported artifact is no longer the same program as the library, and that must be
stated in the artifact's provenance rather than assumed away.

**B2 — a concrete entry point, and which one.** `score_sequence` takes
`multi_state_strategy` and `use_rolling_state` as static arguments. Pick one
combination, name it, and export that; do not attempt the cross-product.

Shapes are **concrete**, per the boundary in A2 — symbolic shapes are unavailable
to this function. That means choosing a fixed `L` (and deciding whether `S` is 1 or
fixed), and it means padding and masking at the call site. Specify the bucket set
and the masking semantics explicitly: a padded residue must not change the score of
a real one, and a test must show that.

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
widen `rtol` until it passes. Either establish an absolute tolerance appropriate
to the logits' range and say why, or show the divergence is smaller than the
model's own run-to-run variation. Record the chosen number and its reasoning in
the artifact's provenance — a tolerance picked to make a test green is the same
false green in a different costume.

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
runtime. Either build a platform-tagged wheel, or ship the artifact as an optional
download and keep the wheel pure — decide, and say which in the PR.

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

`scripts/check_shipped_artifact.py` is new in B4 and must assert, by reading
`dist/*.whl` directly:

1. the artifact path **is** present in the wheel;
2. the wheel's tag is **not** `py3-none-any` — an `embedded-elf-x86_64` artifact in
   a pure wheel installs cleanly on macOS and then fails at runtime;
3. no `.eqx.zst` weight file is present (the artifact carries weights as constants);
4. the artifact's `cpu_features` is the `NATIVE_PORTABLE` set, not host's.

**Mirror the existing pattern rather than inventing one.** `.github/workflows/ci.yml`'s
`wheel-smoke` job already does zipfile-based wheel assertions for `port/` and
`controller/`; follow it, and wire the new script into that job.

Then: green, with B1's tie-breaking equivalence test passing and B3 passing against
a real checkpoint on the per-element output at three lengths.

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
just as green with every export test skipped as with all of them run. **Phase A must
add the enforcement**, not merely restate the expectation: run with `-rs` and fail
the step on any `SKIPPED` line, so a silently-failed toolchain install cannot
present as a passing real-toolchain claim. Until that lands, treat a green
`export-toolchain-tests` as unproven

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
