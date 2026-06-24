---
title: xtrax sweep local execution MVP
task_id: 260624_sweep-verb
date: 260624
status: draft
brainstorm_session: true
invest_overrides: []
---

# Sweep Verb MVP

## Overview
`xtrax sweep sweep_config.toml` executes a grid search of training runs locally. It avoids complex distributed execution by generating a grid of `TrainConfig` objects and running them sequentially in the same Python process. It relies on JAX's persistent compilation cache to automatically reuse JAX graphs across runs that share static properties, providing optimal execution speed without complex manual grouping.

## Acceptance Criteria

**AC1 — Sweep Configuration Parsing & Traversal Overrides**
*Given* a `sweep_config.toml` that contains a base configuration and a dedicated `[sweep.axes]` section (e.g., `[sweep.axes]\n[sweep.axes.optimizer.kwargs]\nlearning_rate = [0.01, 0.001]`), *when* `sweep` parses it, *then* it validates that every leaf value in `[sweep.axes]` is a list (raising a ConfigError if not). It then computes the cartesian product of the explicit axes. If `[sweep.axes]` is missing or empty, it treats the base config as a 1-item grid. To apply overrides safely, `sweep` MUST flatten the `[sweep.axes]` dictionary into tuple-paths (e.g., `("optimizer", "kwargs", "learning_rate")`) and traverse a deep copy of the base configuration dictionary. During traversal, `sweep` MUST safely initialize missing intermediate dictionaries (`current.setdefault(key, {})`) and set the leaf value directly. During this traversal, list values in overrides MUST wholly replace/overwrite base lists, and never concatenate. Regular lists in the base config (e.g., `hidden_sizes = [256]`) are left intact and NOT expanded. (Note: list values in the config dict are NOT converted to tuples — `TrainConfig` fields are plain Python dicts passed to user factories via `**kwargs`, never JAX static arguments. The config hash uses JSON serialization, which handles lists natively.)

**AC2 — In-Process Sequential Execution & Fault Tolerance**
*Given* a list of resolved `TrainConfig` objects, *when* the `sweep` verb executes, *then* it iterates over the list and executes each run sequentially within the exact same Python process. The execution loop MUST wrap the execution in a `try/except Exception` block, log any failures, and proceed to the next config. To prevent the exception traceback from retaining JAX frame locals and causing an OOM leak across the grid, the caught exception's traceback MUST be explicitly cleared (`traceback.clear_frames(exc.__traceback__)`). Because JAX dispatches asynchronously, the loop MUST flush the device queue before proceeding. On success, it must call `.block_until_ready()` on all arrays in the output state. On failure (where state is lost), it MUST enqueue and block on a dummy computation (e.g., `jax.device_put(jnp.zeros(1), d).block_until_ready()` for all devices) to force device synchronization.

**AC3 — JAX Compilation Caching**
*Given* an in-process execution loop, *when* `sweep` starts, *then* it explicitly configures JAX's persistent compilation cache (`jax.config.update("jax_compilation_cache_dir", ".xtrax/jax_cache")`) *before* evaluating any runtime or model imports, so that subsequent runs sharing the same static graph signatures skip recompilation and load directly from the cache. (Note: The cache grows unbounded and requires manual user clearing to prevent disk exhaustion over very wide static sweeps).

**AC4 — Memory Isolation / Cleanup**
*Given* sequential execution in the same process, *when* one run completes and before the next begins, *then* `sweep` ensures that device execution queues are flushed via `block_until_ready()` strategies (see AC2), any `final_state` references are explicitly deleted (`del`), `gc.collect()` is invoked, and `jax.clear_caches()` is called. (Note: `clear_caches()` clears the fast in-memory JIT cache, imposing a minor disk-deserialization penalty for the next run, accepted as a necessary evil for VRAM safety).

**AC5 — Run Isolation via E3 Manifests and Traceability**
*Given* the sequential execution, *when* each individual `TrainConfig` is executed, *then* it ensures each run gets its own unique `run_id`, namespaced checkpoint directory, and `manifest.json`. To support traceability even on failed runs, `run_from_config` MUST be decoupled so that `run_id` is generated and returned *before* the vulnerable `fit_sync` training loop starts (e.g. split into `prepare_run` and `execute_run`). `sweep` MUST track these `run_id`s incrementally and write a `sweep_manifest.json` mapped to their hyperparameters. This sweep manifest MUST be written to an isolated directory (e.g. `.xtrax/sweeps/<sweep_timestamp>/sweep_manifest.json`) AND must be updated progressively/atomically during the loop so that mid-sweep terminations (e.g. `Ctrl-C`) do not lose the traceability record.

## Decision Log

| Option | Verdict | Rationale |
|--------|---------|-----------|
| Delegate to bathos | Rejected | Hard dependency on a cluster orchestrator limits local development; creates unnecessary tech debt for users without bathos. |
| Subprocess execution | Rejected | JAX graph compilation takes minutes. Spawning isolated subprocesses forces full recompilation for every run, destroying iteration speed. |
| Grouping by static vs dynamic args | Rejected (Long-term) | Too much implementation cost for MVP to inspect TOML args and guess JAX static shapes. Relocated to praxia ideas backlog. |
| **In-process with JAX cache** | **Winner** | Relies on JAX's native persistent compilation cache to skip recompilation for matching static shapes automatically, while maintaining a simple sequential execution loop. |

## Assumptions
- JAX's persistent compilation cache is stable enough to handle repeated re-initializations of similar models.
- The user has enough system RAM/VRAM to hold one run at a time; we assume explicit GC between runs is sufficient to prevent OOM.
- The `sweep_config.toml` format will closely mirror `TrainConfig` but allow lists for values.

## TBDs
- **Gather ops for run collections:** A more advanced scheduler that parses the grid and groups runs by compilation footprint (static vs dynamic) to execute optimally. Logged as a low-priority idea in Praxia.
- **Reporting:** How does the sweep summarize all run metrics at the end? MVP addresses this by producing no metrics at all (inheriting the E3 AC13 limitation of `callbacks=()`). Runs must be evaluated offline by examining the saved checkpoints and `sweep_manifest.json` mapping.

## Pre-mortem Record
- **Risk:** Memory leaks. If `run_from_config` leaves dangling JAX arrays, the 10th run in the sweep will OOM. **Mitigation:** AC4 mandates explicit reference deletion and JAX cache clearing between runs.
- **Risk:** Cache misses. If every parameter in the sweep triggers a static shape change (e.g., hidden size), the cache provides no benefit. **Mitigation:** Documented limitation; acceptable for MVP since it's no worse than subprocesses.
