# Dependency Boundaries

This page states xtrax's policy for how it depends on other libraries: the
interface types domain code uses to interoperate, which dependencies are
foundational versus swappable, and how xtrax tracks JAX's own internal
namespace stability tiers.

## The Pytree Protocol as the Interlingua

xtrax's boundary types are pytrees, not classes. A concrete boundary is a
pytree of arrays; an abstract boundary (shape inference, tracing, export) is
a pytree of `jax.ShapeDtypeStruct`. Any function that accepts or returns
these shapes interoperates with xtrax, whether or not it imports xtrax at
all.

This is the same structural-subtyping principle as `LossFunction` and the
other protocols described in {doc}`architecture`'s "Composition Over
Inheritance" section, applied at the data-shape level instead of the
callable-signature level. A domain model library can build, tile, and
checkpoint its own pytrees through xtrax without inheriting from any xtrax
type.

## Substrate vs. Adapters

xtrax depends on JAX and Equinox everywhere — they're the substrate, used
throughout the codebase without confinement. Everything else is an adapter,
each confined to a single subpackage:

- **optax** — training. Imported only from `xtrax.training` (`optim.py`,
  `state.py`, `step.py`, `trainer.py`).
- **orbax** — checkpointing. Imported only from `xtrax.checkpoint`
  (`orbax.py`).
- **grain** — data. Declared as a dependency for `xtrax.data`; the
  distributed-pipeline integration (`create_distributed_pipeline`) is
  currently a stub and does not yet import grain directly.

An adapter's confinement is a promise about surface area: swapping optax for
another optimizer library, or orbax for another checkpoint format, is a
`xtrax.training` / `xtrax.checkpoint` change, not a codebase-wide one.

## The `jax.*` Namespace Stability Policy

JAX ships several namespaces with different stability guarantees. xtrax's
policy per tier:

- **`jax.*`** (public) — free use, no restrictions.
- **`jax.experimental.*`** — used directly today for `checkify` (training
  safety checks, in `training/step.py`, `safety/manager.py`, and
  `loop/checkified_execution.py`) and `sparse.BCOO` (in `sparse/policy.py`,
  `sparse/inference.py`, and `export/safety.py`). These are long-stable
  experimental APIs, imported directly.

  `io_callback` is handled differently: it has a history of namespace and
  signature changes, so it's wrapped in a single internal shim
  (`xtrax.stages._callback`) that pins a validated jax version range and
  raises a loud import-time error if the callback's parameter signature
  drifts. Callers import `io_callback` from the shim, never from
  `jax.experimental` directly. The shim pattern exists for APIs with a
  track record of moving; it isn't applied uniformly to every
  `jax.experimental` import.
- **`jax.extend.*`** — currently unused; zero imports in `src/`. Adopting it
  would require the same treatment as a move-prone experimental API: a
  shim, an exact-minor version pin, and an import-linter fence. The
  motivating use case would be true jaxpr introspection for inference or
  composition tooling.
- **`jax._src`** — never. No imports exist; the only textual reference is a
  comment in `xtrax.loop.checkified_execution` documenting that
  `checkify.JaxRuntimeError` is publicly importable from
  `jax.experimental.checkify` and does not require reaching into
  `jax._src`.

## Graduation-Watching

xtrax adopts a capability from `jax.experimental` or `jax.extend` once it
graduates to a stable namespace, rather than building against the unstable
path pre-emptively. `jax.ffi` is a precedent: it lived at `jax.extend.ffi`,
which was deprecated once the API moved to the stable top-level `jax.ffi`
and later removed. Code that had waited for graduation needed no shim and
no migration; code that imported the pre-graduation path did.
