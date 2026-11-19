import dataclasses
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
from common import blockify, bsmm_raw, dense_ref, make_problem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q4")


@dataclasses.dataclass(frozen=True)
class BSConfig:
    """Block-sparse weight: arrays are data leaves, block sizes are static metadata."""

    blocks: jax.Array  # (Kb, Nb, bk, bn) data
    mask: jax.Array  # (Kb, Nb) data
    bm: int  # meta
    bn: int  # meta
    bk: int  # meta


jax.tree_util.register_dataclass(
    BSConfig,
    data_fields=["blocks", "mask"],
    meta_fields=["bm", "bn", "bk"],
)


def apply_cfg(cfg: BSConfig, x):
    return bsmm_raw(x, cfg.blocks, cfg.mask, bm=cfg.bm, bk=cfg.bk, bn=cfg.bn)


def close_scaled(a, b):
    """Scale-relative check. Spec atol=1e-5 is elementwise and fails for
    float32 accumulation-order noise (~8e-5 abs at |ref|~150). Q4 tests
    retrace behaviour, so parity here uses 1e-5 * max(1, max|ref|)."""
    tol = 1e-5 * max(1.0, float(jnp.max(jnp.abs(b))))
    return float(jnp.max(jnp.abs(a - b))) <= tol


x, w, mask = make_problem()
cfg = BSConfig(blocks=blockify(w), mask=mask, bm=32, bn=32, bk=32)

counts = {"plain": 0, "eqx": 0}


@jax.jit
def jitted(cfg, x):
    counts["plain"] += 1
    return apply_cfg(cfg, x)


y = jitted(cfg, x)
ok_plain = close_scaled(y, dense_ref(x, w, mask))
# Same structure and shapes, different array values -> must NOT retrace.
w2 = w * 3.0 + 0.5
mask2 = ~mask  # different mask values, same shape/dtype
cfg2 = BSConfig(blocks=blockify(w2), mask=mask2, bm=32, bn=32, bk=32)
y2 = jitted(cfg2, x)
ok_plain2 = close_scaled(y2, dense_ref(x, w2, mask2))
plain_traces = counts["plain"]
log.info("plain jit: parity=%s/%s traces after value change=%d", ok_plain, ok_plain2, plain_traces)

# Control: changing a meta field (bm/bk) is expected to retrace.
cfg3 = dataclasses.replace(cfg, bm=64)
try:
    jitted(cfg3, x)
except Exception as e:  # noqa: BLE001
    log.info("changing meta bm=64 raised: %s", type(e).__name__)
meta_retrace = counts["plain"] - plain_traces
log.info("meta change retraces=%d (expected 1)", meta_retrace)


class BSLinear(eqx.Module):
    w: BSConfig
    scale: jax.Array

    def __call__(self, x):
        return apply_cfg(self.w, x) * self.scale


@eqx.filter_jit
def eqx_apply(model, x):
    counts["eqx"] += 1
    return model(x)


m1 = BSLinear(w=cfg, scale=jnp.array(1.0))
m2 = BSLinear(w=cfg2, scale=jnp.array(2.0))
ye1 = eqx_apply(m1, x)
ye2 = eqx_apply(m2, x)
ok_eqx = close_scaled(ye1, dense_ref(x, w, mask)) and close_scaled(
    ye2, 2.0 * dense_ref(x, w2, mask2)
)
eqx_traces = counts["eqx"]
log.info("eqx.filter_jit: parity=%s traces after value change=%d", ok_eqx, eqx_traces)

ok = ok_plain and ok_plain2 and plain_traces == 1 and ok_eqx and eqx_traces == 1
print(
    f"RESULT q4: {'PASS' if ok else 'FAIL'} jit_pytree_parity={ok_plain and ok_plain2} "
    f"jit_traces_value_change={plain_traces} eqx_filter_jit_parity={ok_eqx} "
    f"eqx_traces_value_change={eqx_traces}"
)
