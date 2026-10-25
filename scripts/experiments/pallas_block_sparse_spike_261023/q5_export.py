import logging

import jax
import jax.numpy as jnp
from common import blockify, bsmm_raw, make_problem
from jax import export

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("q5")

x, w, mask = make_problem()
blocks = blockify(w)
x_spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
b_spec = jax.ShapeDtypeStruct(blocks.shape, blocks.dtype)
m_spec = jax.ShapeDtypeStruct(mask.shape, mask.dtype)

# Part A: interpret=True, default platform (cpu).
f_interp = jax.jit(lambda x, b, m: bsmm_raw(x, b, m, interpret=True))
try:
    exp_cpu = export.export(f_interp)(x_spec, b_spec, m_spec)
    txt = exp_cpu.mlir_module()
    has_custom = "custom_call" in txt
    pallas_marker = "tpu_custom_call" in txt or "mosaic" in txt.lower()
    log.info(
        "interpret=True cpu export OK; platforms=%s custom_call_in_text=%s "
        "mosaic/tpu marker=%s bytes=%d",
        exp_cpu.platforms,
        has_custom,
        pallas_marker,
        len(txt),
    )
    # Round-trip: deserialize and run on the same platform.
    rt = export.deserialize(exp_cpu.serialize())
    y_rt = rt.call(x, blocks, mask)
    from common import dense_ref

    ok_rt = bool(jnp.allclose(y_rt, dense_ref(x, w, mask), atol=1e-5, rtol=1e-5))
    log.info("round-trip deserialized call parity=%s", ok_rt)
    a_ok = ok_rt
except Exception as e:  # noqa: BLE001
    log.info("Part A FAILED: %s: %s", type(e).__name__, str(e)[:600])
    a_ok = False
    has_custom = pallas_marker = None

# Part B: interpret=False, platforms=["cuda"] (expected to fail on CPU-only box).
f_gpu = jax.jit(lambda x, b, m: bsmm_raw(x, b, m, interpret=False))
try:
    exp_gpu = export.export(f_gpu, platforms=["cuda"])(x_spec, b_spec, m_spec)
    log.info("Part B cuda export OK; bytes=%d", len(exp_gpu.mlir_module()))
    b_ok = True
except Exception as e:  # noqa: BLE001
    msg = f"{type(e).__name__}: {str(e)[:800]}"
    log.info("Part B cuda export FAILED: %s", msg)
    b_ok = False

# Part C: interpret=True exported for cuda (is the interpret StableHLO platform-portable?).
try:
    exp_c = export.export(f_interp, platforms=["cuda"])(x_spec, b_spec, m_spec)
    c_ok = "custom_call" not in exp_c.mlir_module()
    log.info("interpret=True cuda-platform export OK; custom_call_free=%s", c_ok)
except Exception as e:  # noqa: BLE001
    c_ok = False
    log.info("interpret=True cuda-platform export FAILED: %s: %s", type(e).__name__, str(e)[:400])
log.info("Part C interpret->cuda export ok=%s", c_ok)

print(
    f"RESULT q5: {'PASS' if a_ok else 'FAIL'} interpret_cuda_platform_export={c_ok} "
    f"interpret_cpu_export_roundtrip={a_ok} "
    f"interpret_has_custom_call={has_custom} cuda_noninterp_export={'OK' if b_ok else 'FAILED'}"
)
