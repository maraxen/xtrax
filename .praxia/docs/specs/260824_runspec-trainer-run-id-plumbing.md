---
title: RunSpec/Trainer -> SinkSpec.run_id plumbing mechanism
task_id: 260824_runspec-trainer-run-id-plumbing
date: 260824
status: design-proposed
resolves_tbd: 260824_default-sink-provenance-tracking ("Concrete mechanism for RunSpec/Trainer construction to plumb run_id")
---

# RunSpec/Trainer run_id Plumbing Design

## Context

PR #96 (#96, task 260824_default-sink-provenance-tracking) landed `SinkSpec.run_id`
as a required field and auto-captures it into Zarr store provenance. The parent
spec left one TBD open *before implementation of upstream wiring begins*: the
concrete mechanism for RunSpec/Trainer construction to populate that field so
consumers do not thread it manually.

Code facts this design is grounded in (verified on main @ aebfc2e):

- `RunSpec` (`src/xtrax/run/spec.py`) is an `eqx.Module` base config (`seed`,
  `axes`, `carry_specs`, `boundaries`); `aminox.run.RunSpec` extends it and is
  built from a `RunSpecification`. It is the one object already threaded
  through every execution path.
- `Trainer` (`src/xtrax/training/trainer.py`) has **zero sink surface**: grep
  finds no `make_sink` reference outside `xtrax/run/*`; `SinkSpec` likewise
  appears only inside the package (`sink.py`, `zarr_sink.py`, `__init__.py`).
  No training- or driver-layer code touches either.
- xtrax convention: `run_id` is caller-supplied text (`repro_floor.py` threads
  an arbitrary caller string through to TOML attestations with escaping;
  devtools gates accept it as a parameter). One in-tree generator already
  exists: `generate_run_id(cfg)` in `src/xtrax/cli/run.py` (config-hash id +
  `uuid4().hex[:6]` collision suffix, consumed by the sweep verb and echoed
  into `manifest.json` by the CLI run path). Ad-hoc `uuid4()` fallbacks also
  live in devtools (`bootstrap.py`, `judgment.py`, `emit.py`). This design
  adds a second, deliberately distinct stdlib-only generator and names it to
  avoid that collision (Decision Log).
- `make_sink(spec)` forwards `spec` unmodified -- any `SinkSpec`-shaped factory
  composes with it for free.

## Chosen Mechanism (three pieces)

1. **Optional `run_id` on base `RunSpec`**
   `run_id: str | None = None` added to `RunSpec`. Optional, not required: the
   provenance spec deliberately rejected making construction-time calls
   mandatory for automaticity; the sink still auto-captures whatever it
   receives, so `None` costs nothing until a sink is built. Subclasses inherit
   the field; `from_spec` builders may populate it from their own inputs.

2. **`new_run_id()` in `xtrax/run/ident.py` (new, stdlib-only)**
   Returns `"run-" + uuid4().hex[:12]` (charset `[0-9a-f]` -- `uuid4().hex`
   contains no dashes: path-safe, TOML-safe without escaping, shell-safe).
   Named `new_run_id`, *not* `generate_run_id`, because the latter is taken by
   `cli/run.py`'s config-hash generator with incompatible semantics. Callers
   who need meaningful or reproducible ids pass explicit ones instead -- the
   generator exists so the common case never blocks on naming.

3. **`derive_sink_spec()` in `xtrax/run/sink.py` -- the single canonical seam**

   ```python
   def derive_sink_spec(
       run_spec: RunSpec, *,
       run_id: str | None = None,
       output_dir: Path | None,
       format: Literal["jsonl", "h5", "zarr", "none"] = "zarr",
       flush_every: int = 1,
       extension_schema: dict[str, Any] | None = None,
   ) -> SinkSpec:
       return SinkSpec(
           run_id=run_id or run_spec.run_id or new_run_id(),
           output_dir=output_dir, format=format,
           flush_every=flush_every, extension_schema=extension_schema,
       )
   ```

   Precedence: explicit `run_id=` override > `run_spec.run_id` > generated.
   Drivers (and the future `xtrax run` CLI) call this instead of hand-building
   `SinkSpec`; direct `ZarrStagingSink` consumers (the provenance spec's
   primary audience) are untouched and may keep constructing `SinkSpec`
   manually.

   Note on defaults: the helper pins `format="zarr"` (the provenance seam it
   serves) while bare `SinkSpec` defaults `"jsonl"`. That divergence is
   deliberate -- drivers reaching for a provenance sink get zarr without
   restating it -- and covered by the forwarding AC below.

**None-propagation is closed at two layers.** `SinkSpec.run_id: str` is
runtime-enforced at construction (jaxtyping/beartype rejects `None`
outright), and `ZarrStagingSink.__init__` additionally rejects any falsy
`run_id` (e.g. `""`) with `ValueError` naming the field -- landed in this
batch. Neither the manual nor the derived construction path can stamp an
empty provenance key into a store.

**Static-field caveat (recorded).** `run_id` on an eqx.Module is static aux
data: a jitted function receiving a RunSpec-bearing pytree would re-trace per
distinct run_id value, and two RunSpecs differing only in run_id compare
unequal under structural equality. Latent today -- Trainer never receives a
RunSpec -- but any future plumbing of RunSpec into jit must account for it.

**Trainer: deliberately unchanged.** There is nothing to plumb through it --
it never sees a sink today, and adding one would couple jit'd step machinery
to host-side IO lifecycle. This is recorded as a decision, not an omission:
any future trainer-level sink integration belongs to the CLI/driver-layer
follow-on ("both consumers via one shared layer"), which the provenance spec
already deferred.

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| Optional `RunSpec.run_id` + `derive_sink_spec()` seam + generator | Selected | Zero breaking change; single obvious construction site; keeps sink auto-injection semantics intact. |
| Required `RunSpec.run_id` | Rejected | Breaks every subclass constructor and serialized spec for a value most non-sink runs never use; contradicts the parent spec's anti-mandatory-call posture. |
| Plumb through `Trainer.__init__` | Rejected | Wrong layer: Trainer has no sink surface (verified); sinks are driver-scoped, constructed outside jit. |
| ContextVar / module-global "current run id" | Rejected | Hidden state; silently cross-wires concurrent sinks; violates the fail-loud posture the provenance spec established (collision raises, multi-run guard). |
| bathos-style env/sidecar injection | Deferred | Parent spec already deferred multi-channel injection until a compute-node-without-.git consumer actually appears. |
| Generate inside `SinkSpec.__post_init__` when None | Rejected | Would require making `run_id` Optional (weakening the required-field contract #96 just established) and hides generation inside a dataclass side effect. |
| `make_sink(spec, run_id=...)` overload | Rejected | Conjoins the routing factory with run-identity concerns and duplicates the precedence logic `derive_sink_spec` owns; two seams to keep in sync for no added reach. |
| Reuse `cli/run.py`'s `generate_run_id(cfg)` instead of a new generator | Rejected for now | Config-hash semantics require a `TrainConfig`; `xtrax/run` must stay decoupled from CLI config types. Coexist under distinct names; extract a shared helper only if the formats ever need to unify. |

## Acceptance Criteria (for the small implementation sprint)

- Given any existing `RunSpec`/subclass construction site, when it is
  re-instantiated unchanged, then behavior is unchanged and existing tests
  pass unmodified; the eqx treedef gains exactly one static field
  (`run_id=None`), observable only to pytree introspection.
- Given `run_spec.run_id` unset and no override, when `derive_sink_spec`
  runs, then the returned `SinkSpec.run_id` matches `^run-[0-9a-f]{12}$` and
  two consecutive calls produce distinct values.
- Given an explicit `run_id=` override arg, when `derive_sink_spec` runs,
  then it wins over both `run_spec.run_id` and generation; given
  `run_spec.run_id = "explicit-id"` with no override, then `run_spec.run_id`
  wins; given neither, the generated id wins.
- Given other kwargs (`format`, `output_dir`, `flush_every`, `extension_schema`),
  when `derive_sink_spec` runs, then they are forwarded verbatim onto the
  `SinkSpec`.
- Given a falsy-but-type-valid `run_id` (e.g. `""`) on any path into
  `ZarrStagingSink`, when the sink is constructed, then it raises `ValueError`
  naming `run_id`; given `run_id=None`, `SinkSpec` construction itself is
  rejected by runtime type enforcement (both landed in this batch, pinned by
  tests).
- Given the wiring lands, then `agent_assets/skills/using-xtrax/references/run.md`
  shows the driver-side snippet (`spec = derive_sink_spec(run_spec, output_dir=...)`).

## Assumptions

| Assumption | Owner | Verification |
|------------|-------|--------------|
| `aminox.run.RunSpec` can surface `run_id` from its own inputs when its maintainers adopt this | aminx maintainer | Sign-off at adoption time (parent spec assumption #4 rolls forward). |
| Generated-id collisions across a single process are negligible (uuid4) | xtrax maintainer | None needed at this scale; explicit ids remain the escape hatch. |

## TBDs

| Item | Owner | When |
|------|-------|------|
| Unify the CLI run path's manifest `run_id` echo with `derive_sink_spec` so CLI-driven runs and sink provenance share one id | xtrax maintainer | When the CLI run path first constructs a sink. |
| Promote `derive_sink_spec` to the only documented path once first real consumer adopts | xtrax maintainer | First adoption review. |

## INVEST Gate

```
✓ Independent -- new field + new helper; touches only xtrax/run/*.
✓ Negotiable -- generator format and helper signature can flex.
✓ Valuable -- closes the last TBD gating real consumer adoption of #96.
✓ Estimable -- ~80 lines incl. tests; single sprint slice.
✓ Small -- 6 tight ACs, three source files plus docs (incl. the landed falsy-run_id guard).
✓ Testable -- pure functions plus the construction guards; every criterion is a direct assertion.
```

No overrides required.
