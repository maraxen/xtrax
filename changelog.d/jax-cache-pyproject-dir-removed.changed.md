- **Breaking:** a `jax_compilation_cache_dir` under `[tool.xtrax]` in `pyproject.toml` is no
  longer read. A committed `pyproject.toml` is pushed to other machines, so it must not hold a
  machine path. `configure()` and `resolve()` refuse it with a `JaxCacheConfigError` unless the
  caller chose explicitly (a `path` argument, `off`, `policy="off"`, `$XTRAX_JAX_CACHE=off`,
  `$XTRAX_JAX_CACHE_DIR` or `$JAX_COMPILATION_CACHE_DIR`), in which case the key is ignored and a
  warning names it, so `xtrax run --compilation-cache off` works on a project that still has it.
  Migration: move the path to the user config `~/.config/xtrax/config.toml`
  (`jax_compilation_cache_dir = "..."`) or export `$XTRAX_JAX_CACHE_DIR`; keep only
  `jax_compilation_cache = "on" | "off"` in `pyproject.toml`.
- An empty `$XTRAX_JAX_CACHE` now counts as unset, not as policy off; `none` and `off` still
  disable. A relative `compilation_cache` path passed to the Slurm executor is refused at
  generation (it would resolve against each task's working directory).
