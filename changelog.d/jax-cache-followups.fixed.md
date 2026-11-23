- `xtrax.jax_cache`: reconfiguring resets JAX's cache object when it was initialised even if the
  directory flag was cleared; `prune_corrupt` skips entries deleted concurrently; the
  world-writable warning also checks the configured base directory (mode, and owner other than
  the current user or root).
