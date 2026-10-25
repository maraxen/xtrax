import logging

import jax.numpy as jnp
from common import blockify, bsmm_raw, dense_ref, make_problem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q1")

x, w, mask = make_problem()
blocks = blockify(w)
y_ref = dense_ref(x, w, mask)
y = bsmm_raw(x, blocks, mask)
ok = bool(jnp.allclose(y, y_ref, atol=1e-5, rtol=1e-5))
log.info("density=%.3f max_abs_err=%.3e", float(mask.mean()), float(jnp.max(jnp.abs(y - y_ref))))

# CONTROL: corrupt one unmasked block in the kernel input; the check must go red.
ii, jj = jnp.argwhere(mask, size=1)[0]
bad_blocks = blocks.at[ii, jj, 0, 0].add(1.0)
y_bad = bsmm_raw(x, bad_blocks, mask)
control_red = not bool(jnp.allclose(y_bad, y_ref, atol=1e-5, rtol=1e-5))
log.info("control: corrupted unmasked block (%d,%d) -> check red=%s", int(ii), int(jj), control_red)

# Masked-out block corruption must be ignored (proves pl.when skip is live).
zi, zj = jnp.argwhere(~mask, size=1)[0]
skip_blocks = blocks.at[zi, zj, 0, 0].add(1.0)
y_skip = bsmm_raw(x, skip_blocks, mask)
skip_ok = bool(jnp.allclose(y_skip, y_ref, atol=1e-5, rtol=1e-5))
log.info("masked-out block corruption ignored=%s", skip_ok)

ratio = float(jnp.max(jnp.abs(y - y_ref) / (1e-5 + 1e-5 * jnp.abs(y_ref))))
log.info("worst |err| / (atol + rtol*|ref|) = %.3f (1.0 is the spec edge)", ratio)

passed = ok and control_red and skip_ok
print(
    f"RESULT q1: {'PASS' if passed else 'FAIL'} parity_ok={ok} "
    f"control_red={control_red} masked_block_skipped={skip_ok}"
)
