"""THROWAWAY measurement script — probe branch only, never merged.

Measures, on a GitHub `ubuntu-latest` runner, the two facts the
`xtrax.export` spec (`.praxia/docs/specs/260901_xtrax-export-webgpu.md`)
leaves unmeasured for AC-7/AC-8:

1. Can wgpu obtain an adapter at all, with and without a Vulkan ICD installed?
2. Does real IREE-emitted SPIR-V from a composed xtrax pipeline pass naga on a
   feature-free (browser-WebGPU-shaped) device, and on one with `immediates`?

Exits 0 always: this reports, it does not gate.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

SPIRV_MAGIC = (0x07230203).to_bytes(4, "little")
MODE = os.environ.get("PROBE_MODE", "adapter")


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}", flush=True)


section("environment")
print("python:", sys.version.split()[0])
icd_dir = Path("/usr/share/vulkan/icd.d")
print("ICD dir exists:", icd_dir.exists())
if icd_dir.exists():
    print("ICDs:", sorted(p.name for p in icd_dir.iterdir()))

section("adapter enumeration")
try:
    import wgpu
except ImportError as e:
    print("wgpu not importable:", e)
    raise SystemExit(0) from None

print("wgpu:", wgpu.__version__)
try:
    adapters = wgpu.gpu.enumerate_adapters_sync()
    print(f"{len(adapters)} adapter(s)")
    for a in adapters:
        info = a.info
        print(
            "  -",
            info.get("adapter_type"),
            "/",
            info.get("backend_type"),
            "/",
            info.get("device"),
        )
except Exception as e:  # noqa: BLE001
    print("enumerate FAILED:", type(e).__name__, e)
    adapters = []

section("request_adapter_sync + request_device_sync")
adapter = None
try:
    adapter = wgpu.gpu.request_adapter_sync(power_preference="low-power")
    print("adapter:", adapter.info.get("device"), "/", adapter.info.get("adapter_type"))
    dev = adapter.request_device_sync()
    print("device: OK")
    print("'immediates' on adapter:", "immediates" in adapter.features)
    print("'shader-f16' on adapter:", "shader-f16" in adapter.features)
except Exception as e:  # noqa: BLE001
    print("ADAPTER/DEVICE FAILED:", type(e).__name__, str(e)[:300])
    print("VERDICT: no wgpu adapter on this runner configuration")
    raise SystemExit(0) from None

print("VERDICT: wgpu adapter available on this runner configuration")

if MODE != "full":
    raise SystemExit(0)

section("real xtrax pipeline -> IREE vulkan-spirv -> SPIR-V -> naga")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from iree.compiler import tools as iree_tools  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.iree_export_spike.composer import compose_exportable  # noqa: E402
from scripts.iree_export_spike.hf_weights import random_mlp  # noqa: E402
from xtrax.tiling.plan import AxisSpec, BatchPlanner  # noqa: E402

model = random_mlp(8, 16, 4)
plan = BatchPlanner().plan([AxisSpec(name="batch", cardinality=32, default_batch_size=8)])
mlir = jax.export.export(jax.jit(compose_exportable(model, plan)))(
    jnp.ones((32, 8), jnp.float32)
).mlir_module()
print("StableHLO chars:", len(mlir))

dump = Path(tempfile.mkdtemp()) / "dump"
if dump.exists():
    shutil.rmtree(dump)
dump.mkdir(parents=True)
iree_tools.compile_str(
    mlir,
    input_type="stablehlo",
    extra_args=[
        "--iree-hal-target-backends=vulkan-spirv",
        f"--iree-hal-dump-executable-binaries-to={dump}",
    ],
)
spvs = [p for p in sorted(dump.rglob("*")) if p.is_file() and p.read_bytes()[:4] == SPIRV_MAGIC]
print(f"{len(spvs)} SPIR-V module(s) extracted")
for p in spvs:
    has_pc = b"__push_constant_var__" in p.read_bytes()
    print(f"  - {p.name} ({p.stat().st_size} B), push-constants: {has_pc}")

for req in ([], ["immediates"]):
    label = "no features (browser-WebGPU-shaped)" if not req else "required_features=['immediates']"
    try:
        d = adapter.request_device_sync(required_features=req)
    except Exception as e:  # noqa: BLE001
        print(f"[{label}] device FAILED: {type(e).__name__}: {str(e)[:160]}")
        continue
    ok, errs = 0, []
    for p in spvs:
        try:
            d.create_shader_module(code=p.read_bytes())
            ok += 1
        except Exception as e:  # noqa: BLE001
            errs.append(
                next(
                    (ln.strip() for ln in str(e).splitlines() if "not supported" in ln),
                    str(e).splitlines()[0] if str(e).splitlines() else "?",
                )[:110]
            )
    print(f"[{label}] {ok}/{len(spvs)} VALID")
    for e in dict.fromkeys(errs):
        print("      " + e)
