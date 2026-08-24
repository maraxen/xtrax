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
  confirms no `make_sink`/`SinkSpec` reference outside `xtrax/run/sink.py` and
  `xtrax/run/__init__.py`. Sinks are constructed by drivers/consumers before
  training, not inside `Trainer.step` (host-side io_callback boundaries sit
  outside jit anyway).
- xtrax convention: `run_id` is caller-supplied text (`repro_floor.py` threads
  an arbitrary caller string through to TOML attestations with escaping;
  devtools gates accept it as a parameter). There is no in-tree generator yet.
- `make_sink(spec)` forwards `spec` unmodified -- any `SinkSpec`-shaped factory
  composes with it for free.

## Chosen Mechanism (three pieces)

1. **Optional `run_id` on base `RunSpec`**
   `run_id: str | None = None` added to `RunSpec`. Optional, not required: the
   provenance spec deliberately rejected making construction-time calls
   mandatory for automaticity; the sink still auto-captures whatever it
   receives, so `None` costs nothing until a sink is built. Subclasses inherit
   the field; `from_spec` builders may populate it from their own inputs.

2. **`generate_run_id()` in `xtrax/run/ident.py` (new, stdlib-only)**
   Returns `"run-" + uuid4().hex[:12]` (charset `[0-9a-f-]`: path-safe,
   TOML-safe without escaping, shell-safe). Callers who need meaningful or
   reproducible ids pass explicit ones instead -- the generator exists so the
   common case never blocks on naming.

3. **`derive_sink_spec()` in `xtrax/run/sink.py` -- the single canonical seam**

   ```python
   def derive_sink_spec(
       run_spec: RunSpec, *,
       output_dir: Path | None,
       format: Literal["jsonl", "h5", "zarr", "none"] = "zarr",
       flush_every: int = 1,
       extension_schema: dict[str, Any] | None = None,
   ) -> SinkSpec:
       return SinkSpec(
           run_id=run_spec.run_id or generate_run_id(),
           output_dir=output_dir, format=format,
           flush_every=flush_every, extension_schema=extension_schema,
       )
   ```

   Precedence: explicit constructor arg > `run_spec.run_id` > generated.
   Drivers (and the future `xtrax run` CLI) call this instead of hand-building
   `SinkSpec`; direct `ZarrStagingSink` consumers (the provenance spec's
   primary audience) are untouched and may keep constructing `SinkSpec`
   manually.

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

## Acceptance Criteria (for the small implementation sprint)

- Given any existing `RunSpec`/subclass construction site, when it is
  re-instantiated unchanged, then behavior and eqx structure are identical
  except `run_id` defaults to `None` (existing tests pass unmodified).
- Given `run_spec.run_id` unset, when `derive_sink_spec(run_spec, ...)` runs,
  then the returned `SinkSpec.run_id` matches `^run-[0-9a-f]{12}$` and two
  consecutive calls produce distinct values.
- Given `run_spec.run_id = "explicit-id"`, when `derive_sink_spec` runs (with
  or without an explicit override arg), then precedence is override >
  `run_spec.run_id` > generated.
- Given other kwargs (`format`, `output_dir`, `flush_every`, `extension_schema`),
  when `derive_sink_spec` runs, then they are forwarded verbatim onto the
  `SinkSpec`.
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
| Does the future `xtrax run` CLI auto-generate and echo the chosen run_id into manifest.json? | xtrax maintainer | When that CLI is scoped. |
| Promote `derive_sink_spec` to the only documented path once first real consumer adopts | xtrax maintainer | First adoption review. |

## INVEST Gate

```
✓ Independent -- new field + new helper; touches only xtrax/run/*.
✓ Negotiable -- generator format and helper signature can flex.
✓ Valuable -- closes the last TBD gating real consumer adoption of #96.
✓ Estimable -- ~60 lines incl. tests; single sprint slice.
✓ Small -- 5 tight ACs, two files plus docs.
✓ Testable -- pure functions; every criterion is a direct assertion.
```

No overrides required.
