- `xtrax.model_jit` keeps at most 256 output static parts per function (least recently
  used first) instead of one per trace signature forever. An evicted entry is rebuilt by an
  abstract retrace (`jax.eval_shape`, no compile) when it is needed again.
