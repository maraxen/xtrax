import logging

import jax
import jax.numpy as jnp
from common import blockify, bsmm_raw, dense_ref, make_problem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q3")

x, w, mask = make_problem()
blocks = blockify(w)

traces = {"jit": 0, "vmap": 0}


def f_jit(x, blocks, mask):
    traces["jit"] += 1  # Python side effect: runs only at trace time
    return bsmm_raw(x, blocks, mask)


jit_f = jax.jit(f_jit)
y0 = jit_f(x, blocks, mask)
ok_jit = bool(jnp.allclose(y0, dense_ref(x, w, mask), atol=1e-5, rtol=1e-5))
for _ in range(2):  # same shapes, different values
    xi = x + 0.1
    jit_f(xi, blocks * 2.0, mask)
ok_jit_values = bool(
    jnp.allclose(jit_f(x, blocks, mask), dense_ref(x, w, mask), atol=1e-5, rtol=1e-5)
)
n_jit_compiles = traces["jit"]
log.info("jit parity=%s traces after 3 same-shape calls=%d", ok_jit, n_jit_compiles)

B = 3
xb = jnp.stack([x, x * 0.5, x + 1.0])


def f_vmap(x, blocks, mask):
    traces["vmap"] += 1
    return bsmm_raw(x, blocks, mask)


vmapped = jax.jit(jax.vmap(f_vmap, in_axes=(0, None, None)))
yb = vmapped(xb, blocks, mask)
ref_b = jnp.stack([dense_ref(xb[i], w, mask) for i in range(B)])
ok_vmap = bool(jnp.allclose(yb, ref_b, atol=1e-5, rtol=1e-5))
for _ in range(2):
    vmapped(xb * 1.5, blocks, mask)
n_vmap_compiles = traces["vmap"]
log.info("vmap parity=%s traces after 3 same-shape calls=%d", ok_vmap, n_vmap_compiles)

ok = ok_jit and ok_jit_values and ok_vmap and n_jit_compiles == 1 and n_vmap_compiles == 1
print(
    f"RESULT q3: {'PASS' if ok else 'FAIL'} jit_parity={ok_jit and ok_jit_values} "
    f"vmap_parity={ok_vmap} jit_traces={n_jit_compiles} vmap_traces={n_vmap_compiles}"
)
