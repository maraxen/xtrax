# Codemod: thin wrappers deprecated in the probe release (xtrax #6107)

The probe release deprecates xtrax names that are a rename of, or a few lines over, a JAX or
optax call. Each one warns (naming the **caller** and the replacement) and is otherwise
unchanged; nothing is removed in this release. The warning is the probe: if you see one,
migrate, or tell us the name is in use.

| Deprecated | Replacement | Rewritten by the rules |
|---|---|---|
| `init_dist(...)` | `jax.distributed.initialize(...)` | yes (see the one-process note) |
| `get_device_mesh(shape, names)` | `jax.make_mesh(shape, names)` | yes |
| `partition_labels(model, f)` | `jax.tree.map(lambda leaf: "frozen" if f(leaf) else "train", model)` | yes |
| `xtrax.io` (`BoundedCallbackHandler`, `async_indexed_stream`) | `xtrax.engine.io` | yes |
| `SafetyManager`, `with_safety` | `jax.jit(checkify.checkify(fn, errors=checkify.float_checks))`, then `err.throw()` | report only |
| `SafetyTrainStep`, `create_train_step(..., safety=True)` | the same expression around `trainer.step` | report only |
| `WeightedLoss`, `MultiTaskLoss` | `xtrax.ComposedLoss` | report only |
| `ShardingPolicy`, `get_hardware_mesh_profile`, `is_distributed` | your own `PartitionSpec` rules / `jax.make_mesh` / `jax.distributed.is_initialized()` | report only |

`create_train_step(..., safety=False)` is **not** deprecated.

## Run it

Needs [ast-grep](https://ast-grep.github.io/) >= 0.42. Report first, then rewrite:

```bash
ast-grep scan --rule codemods/thin-wrappers-to-jax/rules.yml src tests
ast-grep scan --rule codemods/thin-wrappers-to-jax/rules.yml --update-all src tests
```

Then add `import jax` where a rewrite introduced `jax.`, and let your linter drop the
imports that are now unused (`ruff check --select F401 --fix`). The rules are idempotent.

`fixture_before.py` -> `fixture_after.py` is the exact expected transformation. xtrax's
`tests/test_deprecated_probe_release.py` checks it, and checks that a second run changes
nothing.

## Differences the rewrite cannot see

- **`init_dist` with one process.** `init_dist(num_processes=1)` skipped the JAX call and only
  recorded the arguments. `jax.distributed.initialize()` with one process starts a coordinator
  (and with no arguments and no cluster environment it raises). Keep a `num_processes > 1`
  guard in your own code if you ran single-process.
- **`DataModule(distributed=True)`.** It used to need `init_dist()`; it now needs
  `jax.process_count() > 1` and raises `RuntimeError` otherwise. A one-process `init_dist()`
  no longer satisfies it.
- **`get_device_mesh`** raised unless the shape's product equalled the device count;
  `jax.make_mesh` has its own validation and orders devices for the physical topology.
- **`checkify.float_checks` is NaN plus division-by-zero.** It does not catch an `Inf` that comes
  from overflow, whatever `SafetyManager`'s old docstring said. Its `check_nans` / `check_infs`
  fields were never read. xtrax debt #3042 tracks a Context-driven rebuild.
- **`ComposedLoss`** returns `(loss, aux)` (pair it with `Trainer(has_aux=True)`), takes traced
  weights, passes the same predictions and targets to every term, and has no `weight_schedule`.
- **A lambda passed to `partition_labels`** is rewritten to a lambda calling a lambda. It is
  correct; inline it by hand.
- **`xtrax.init_dist(...)` (attribute form)** is not matched; only the bare name is. Search
  for it by hand. xtrax debt #3043 tracks a planner-based rebuild of `ShardingPolicy`.

Not covered here: `xtrax.sparse` was removed outright in the same release (no consumer
used it); see the CHANGELOG `### Removed` entry for how to recover it.
