"""xtrax ONNX route spike: jax2onnx -> ONNX Runtime CPU EP.

Answers whether an ``onnx`` target for ``xtrax.export`` could vouch for the
integer outputs of the ops that have already produced silent divergences on the
IREE route (tied sort/argsort/top_k, threefry draws), plus a float MLP composed
through ``xtrax.export.composer.build_traceable_callable``.

Every case is converted under both ``export_mode="standard"`` and ``"web"`` and
executed under ORT's ``ORT_DISABLE_ALL`` and ``ORT_ENABLE_ALL`` graph
optimisation levels. Integer outputs are compared EXACTLY against the JAX
oracle; float outputs against a max-abs bar. A dtype census of each converted
graph records where int64 appears and whether graph I/O dtypes match JAX's.

ORT's Python CPU EP is not ORT Web; nothing here is browser evidence.

JAX oracles are evaluated BEFORE jax2onnx is imported: jax2onnx patches jnp
globally at import (measured in aminx, p07_split_export.py:250-261).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger("onnx_route_spike")

N = 256
TOP_K = 16
FLOAT_BAR = 1e-5
EXPORT_MODES = ("standard", "web")
OPT_LEVELS = ("disable_all", "enable_all")


class TinyMLP(eqx.Module):
    """Same shape as tests/export/conftest.py's TinyMLP."""

    w1: jax.Array
    w2: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.tanh(x @ self.w1) @ self.w2


@dataclass
class Case:
    name: str
    fn: Callable[..., Any]
    inputs: list[np.ndarray]
    kind: str  # "int_exact" | "bits_exact" | "float_bar"
    oracle: list[np.ndarray] | None = None


def _iota() -> jax.Array:
    return jnp.arange(N, dtype=jnp.int32)


def build_cases(seed: int) -> tuple[list[Case], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    # Tie-rich fixtures: 4 distinct int keys and 8 distinct float values over 256 slots.
    keys = rng.integers(0, 4, size=N).astype(np.int32)
    vals = (rng.integers(0, 8, size=N) / 8.0).astype(np.float32)
    raw_key = np.asarray(jax.random.key_data(jax.random.PRNGKey(seed)), dtype=np.uint32)

    def sort_2key_tiebreak(k):
        return jax.lax.sort((k, _iota()), num_keys=2, is_stable=False)[1]

    def argsort_stable(k):
        return jnp.argsort(k, stable=True).astype(jnp.int32)

    def argsort_desc_stable_float(v):
        return jnp.argsort(-v, stable=True).astype(jnp.int32)

    def lax_top_k(v):
        vv, ii = jax.lax.top_k(v, TOP_K)
        return vv, ii.astype(jnp.int32)

    def top_k_via_2key_sort(v):
        s = jax.lax.sort((-v, _iota()), num_keys=2, is_stable=False)
        return -s[0][:TOP_K], s[1][:TOP_K]

    def argmax_ties(v):
        return jnp.argmax(v.reshape(16, 16), axis=1).astype(jnp.int32)

    def int_scatter_add_gather(k):
        hist = jnp.zeros(4, dtype=jnp.int32).at[k].add(1)
        return hist, hist[k]

    def threefry_uniform(key):
        return jax.random.uniform(key, (64,), dtype=jnp.float32)

    def threefry_nested_split(key):
        k1, k2 = jax.random.split(key)
        k3 = jax.random.split(k1)[0]
        return jax.random.uniform(k3, (32,)), jax.random.uniform(k2, (32,))

    from xtrax.export.composer import build_traceable_callable
    from xtrax.tiling.plan import AxisSpec, BatchPlanner

    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    model = TinyMLP(
        w1=jax.random.normal(k1, (8, 16), dtype=jnp.float32) * 0.1,
        w2=jax.random.normal(k2, (16, 4), dtype=jnp.float32) * 0.1,
    )
    plan = BatchPlanner().plan([AxisSpec(name="batch", cardinality=32, default_batch_size=8)])
    xs = (np.arange(32 * 8, dtype=np.float32).reshape(32, 8) / 256.0).astype(np.float32)
    mlp_fn = build_traceable_callable(model, plan)
    perturbed = eqx.tree_at(lambda m: m.w1, model, model.w1 + 1e-3)
    mlp_perturbed_fn = build_traceable_callable(perturbed, plan)

    cases = [
        Case("sort_2key_tiebreak", sort_2key_tiebreak, [keys], "int_exact"),
        Case("argsort_stable", argsort_stable, [keys], "int_exact"),
        Case("argsort_desc_stable_float", argsort_desc_stable_float, [vals], "int_exact"),
        Case("lax_top_k", lax_top_k, [vals], "int_exact"),
        Case("top_k_via_2key_sort", top_k_via_2key_sort, [vals], "int_exact"),
        Case("argmax_ties", argmax_ties, [vals], "int_exact"),
        Case("int_scatter_add_gather", int_scatter_add_gather, [keys], "int_exact"),
        Case("threefry_uniform", threefry_uniform, [raw_key], "bits_exact"),
        Case("threefry_nested_split", threefry_nested_split, [raw_key], "bits_exact"),
        Case("mlp_xtrax_plan", mlp_fn, [xs], "float_bar"),
    ]
    extra = {
        "keys": keys,
        "vals": vals,
        "xs": xs,
        "mlp_perturbed_fn": mlp_perturbed_fn,
        "n_distinct_keys": int(len(np.unique(keys))),
        "n_distinct_vals": int(len(np.unique(vals))),
    }
    return cases, extra


def _as_list(out: Any) -> list[np.ndarray]:
    if isinstance(out, (tuple, list)):
        return [np.asarray(o) for o in out]
    return [np.asarray(out)]


def compare(kind: str, oracle: list[np.ndarray], got: list[np.ndarray]) -> dict[str, Any]:
    """Compare ORT outputs to the oracle. Values only; dtype is reported separately."""
    if len(oracle) != len(got):
        return {"values_exact": False, "n_mismatch": -1, "max_abs": None, "why": "arity"}
    n_mismatch = 0
    max_abs = 0.0
    for o, g in zip(oracle, got, strict=True):
        if o.shape != g.shape:
            return {"values_exact": False, "n_mismatch": -1, "max_abs": None, "why": "shape"}
        if kind == "float_bar":
            d = float(np.max(np.abs(o.astype(np.float64) - g.astype(np.float64))))
            max_abs = max(max_abs, d)
            n_mismatch += int(np.sum(np.abs(o.astype(np.float64) - g.astype(np.float64)) > FLOAT_BAR))
        elif kind == "bits_exact":
            ob = o.astype(np.float32).view(np.uint32)
            gb = g.astype(np.float32).view(np.uint32)
            n_mismatch += int(np.sum(ob != gb))
        else:
            n_mismatch += int(np.sum(o.astype(np.int64) != g.astype(np.int64)))
    exact = n_mismatch == 0
    return {"values_exact": exact, "n_mismatch": n_mismatch, "max_abs": max_abs if kind == "float_bar" else None}


def run_controls(cases: list[Case], extra: dict[str, Any]) -> dict[str, Any]:
    """Controls that must be computed from JAX alone (before jax2onnx import)."""
    by = {c.name: c for c in cases}
    keys, vals = extra["keys"], extra["vals"]

    # (1) Tie order is observable: breaking ties by DESCENDING index must change the answer.
    alt_argsort = np.lexsort((-np.arange(N), keys)).astype(np.int32)
    ties_visible_argsort = not np.array_equal(alt_argsort, by["argsort_stable"].oracle[0])
    alt_topk = np.lexsort((-np.arange(N), -vals))[:TOP_K].astype(np.int32)
    ties_visible_topk = not np.array_equal(alt_topk, by["lax_top_k"].oracle[1])

    # (2) Comparator must fire on a two-slot swap and stay silent on identity.
    o = by["argsort_stable"].oracle[0]
    swapped = o.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    cmp_detects_swap = not compare("int_exact", [o], [swapped])["values_exact"]
    cmp_identity_ok = compare("int_exact", [o], [o.copy()])["values_exact"]
    b = by["threefry_uniform"].oracle[0]
    b2 = b.copy()
    b2[0] = np.nextafter(b2[0], np.float32(2.0))
    bits_detects_ulp = not compare("bits_exact", [b], [b2])["values_exact"]

    return {
        "ties_visible_argsort": bool(ties_visible_argsort),
        "ties_visible_topk": bool(ties_visible_topk),
        "cmp_detects_swap": bool(cmp_detects_swap),
        "cmp_identity_ok": bool(cmp_identity_ok),
        "bits_detects_ulp": bool(bits_detects_ulp),
        "n_distinct_keys": extra["n_distinct_keys"],
        "n_distinct_vals": extra["n_distinct_vals"],
    }


def _walk_graphs(graph: Any) -> list[Any]:
    out = [graph]
    for node in graph.node:
        for attr in node.attribute:
            if attr.g is not None and attr.HasField("g"):
                out.extend(_walk_graphs(attr.g))
            for g in attr.graphs:
                out.extend(_walk_graphs(g))
    return out


def dtype_census(model: Any) -> dict[str, Any]:
    import onnx

    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001 -- census degrades to un-inferred model
        inferred = model
    name_to_type: dict[str, int] = {}
    graphs = _walk_graphs(inferred.graph)
    for g in graphs:
        for vi in list(g.value_info) + list(g.input) + list(g.output):
            if vi.type.HasField("tensor_type"):
                name_to_type[vi.name] = vi.type.tensor_type.elem_type
        for init in g.initializer:
            name_to_type[init.name] = init.data_type
    dt_name = onnx.helper.tensor_dtype_to_np_dtype
    counts: Counter[str] = Counter()
    int64_producers: Counter[str] = Counter()
    op_hist: Counter[str] = Counter()
    for g in graphs:
        for node in g.node:
            op_hist[node.op_type] += 1
            for o in node.output:
                t = name_to_type.get(o)
                if t is None:
                    counts["unknown"] += 1
                    continue
                n = str(dt_name(t))
                counts[n] += 1
                if n == "int64":
                    int64_producers[node.op_type] += 1

    def _io(vis: Any) -> list[str]:
        return [str(dt_name(v.type.tensor_type.elem_type)) for v in vis]

    return {
        "opset": [(o.domain or "ai.onnx", o.version) for o in model.opset_import],
        "node_output_dtypes": dict(counts),
        "int64_producers": dict(int64_producers),
        "op_hist": dict(op_hist),
        "graph_inputs": _io(model.graph.input),
        "graph_outputs": _io(model.graph.output),
        "n_subgraphs": len(graphs) - 1,
    }


def _extract_primitive(msg: str) -> str | None:
    for token in ("primitive", "Primitive"):
        if token in msg:
            tail = msg.split(token, 1)[1]
            return tail.strip(" :'\"`").split()[0].strip("'\"`,.") if tail.strip() else None
    return None


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def run_case(case: Case, export_mode: str, out_dir: Path) -> dict[str, Any]:
    import jax2onnx
    import onnxruntime as ort

    rec: dict[str, Any] = {"case": case.name, "kind": case.kind, "export_mode": export_mode}
    specs = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in case.inputs]
    try:
        model = jax2onnx.to_onnx(case.fn, specs, model_name=case.name, export_mode=export_mode)
    except Exception as exc:  # noqa: BLE001 -- conversion failure is a recorded result
        msg = f"{type(exc).__name__}: {exc}"
        rec.update(converted=False, error=msg[:2000], primitive=_extract_primitive(msg))
        return rec
    data = model.SerializeToString()
    path = out_dir / f"{case.name}.{export_mode}.onnx"
    path.write_bytes(data)
    rec.update(converted=True, artifact=str(path), sha256=_sha256(data), size_bytes=len(data))
    rec["census"] = dtype_census(model)
    oracle_dtypes = [str(o.dtype) for o in case.oracle]
    rec["oracle_output_dtypes"] = oracle_dtypes
    rec["io_dtype_changed"] = rec["census"]["graph_outputs"] != oracle_dtypes
    runs = {}
    for level in OPT_LEVELS:
        so = ort.SessionOptions()
        so.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            if level == "disable_all"
            else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        try:
            sess = ort.InferenceSession(data, sess_options=so, providers=["CPUExecutionProvider"])
            feeds = {i.name: a for i, a in zip(sess.get_inputs(), case.inputs, strict=True)}
            got = [np.asarray(g) for g in sess.run(None, feeds)]
            r = compare(case.kind, case.oracle, got)
            r["ort_output_dtypes"] = [str(g.dtype) for g in got]
        except Exception as exc:  # noqa: BLE001 -- ORT failure is a recorded result
            r = {"values_exact": False, "n_mismatch": -1, "ort_error": f"{type(exc).__name__}: {exc}"[:2000]}
        runs[level] = r
    rec["ort"] = runs
    rec["values_exact"] = all(r.get("values_exact") for r in runs.values())
    rec["ort_failed"] = any("ort_error" in r for r in runs.values())
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("outputs/onnx_spike/result.json"))
    parser.add_argument("--smoke", action="store_true", help="one case, standard mode only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if jax.config.jax_enable_x64:
        raise SystemExit("refusing: jax_enable_x64 is set; the spike measures the 32-bit graph")
    if "jax2onnx" in sys.modules:
        raise SystemExit("refusing: jax2onnx imported before JAX oracles were evaluated")

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cases, extra = build_cases(args.seed)
    for c in cases:
        c.oracle = _as_list(jax.jit(c.fn)(*[jnp.asarray(a) for a in c.inputs]))
    perturbed_oracle = _as_list(jax.jit(extra["mlp_perturbed_fn"])(jnp.asarray(extra["xs"])))
    controls = run_controls(cases, extra)
    logger.info("controls (pre-import): %s", controls)

    import jax2onnx  # noqa: F401 -- deliberately after oracle evaluation

    modes = ("standard",) if args.smoke else EXPORT_MODES
    todo = cases[:1] if args.smoke else cases
    per_case_path = out_dir / "cases.jsonl"
    records = []
    with per_case_path.open("w") as fh:
        for mode in modes:
            for c in todo:
                try:
                    rec = run_case(c, mode, out_dir)
                except Exception:  # noqa: BLE001 -- never lose the rest of the run
                    rec = {"case": c.name, "export_mode": mode, "converted": False,
                           "error": traceback.format_exc()[-2000:]}
                logger.info("%s/%s converted=%s exact=%s", c.name, mode,
                            rec.get("converted"), rec.get("values_exact"))
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()
                records.append(rec)

    # (3) Float control: perturbed-weight MLP, converted and run by the SAME path, must miss the bar
    # against the UNPERTURBED oracle.
    mlp = next(c for c in cases if c.name == "mlp_xtrax_plan")
    ctrl_case = Case("mlp_perturbed_ctrl", extra["mlp_perturbed_fn"], mlp.inputs, "float_bar",
                     oracle=mlp.oracle)
    ctrl = run_case(ctrl_case, "standard", out_dir)
    ctrl_max_abs = (ctrl.get("ort") or {}).get("enable_all", {}).get("max_abs")
    controls["perturbed_mlp_converted"] = bool(ctrl.get("converted"))
    controls["perturbed_mlp_max_abs"] = ctrl_max_abs
    controls["perturbed_mlp_detected"] = bool(ctrl_max_abs is not None and ctrl_max_abs > FLOAT_BAR)
    controls["perturbed_oracle_differs"] = bool(
        np.max(np.abs(perturbed_oracle[0] - mlp.oracle[0])) > FLOAT_BAR
    )

    controls_ok = all(
        controls[k]
        for k in ("ties_visible_argsort", "ties_visible_topk", "cmp_detects_swap", "cmp_identity_ok",
                  "bits_detects_ulp", "perturbed_mlp_detected", "perturbed_oracle_differs")
    )
    converted = [r for r in records if r.get("converted")]
    # A graph ORT refused to execute is not a value divergence; it is counted in n_ort_failed.
    divergent = [r for r in converted if not r.get("ort_failed") and not r.get("values_exact")]
    result = {
        "versions": {p: version(p) for p in ("jax", "jaxlib", "jax2onnx", "onnx", "onnxruntime", "equinox", "xtrax")},
        "seed": args.seed,
        "smoke": args.smoke,
        "controls": controls,
        "controls_ok": controls_ok,
        "n_variants": len(records),
        "n_converted": len(converted),
        "n_unconverted": len(records) - len(converted),
        "n_value_divergent": len(divergent),
        "n_ort_failed": sum(1 for r in converted if r.get("ort_failed")),
        "n_io_dtype_changed": sum(1 for r in converted if r.get("io_dtype_changed")),
        "n_graphs_with_int64": sum(1 for r in converted if r["census"]["node_output_dtypes"].get("int64")),
        "divergent": sorted({f"{r['case']}/{r['export_mode']}" for r in divergent}),
        "unconverted": sorted({f"{r['case']}/{r['export_mode']}" for r in records if not r.get("converted")}),
        "io_dtype_changed": sorted({f"{r['case']}/{r['export_mode']}" for r in converted if r.get("io_dtype_changed")}),
        "per_case_path": str(per_case_path),
        "note": "ORT Python CPU EP is not ORT Web; this is not browser evidence.",
    }
    args.out.write_text(json.dumps(result, indent=2, default=str))
    rp = os.environ.get("BTH_RESULTS_PATH")
    if rp:
        Path(rp).write_text(json.dumps(result, default=str))
    logger.info("summary: %s", {k: result[k] for k in ("controls_ok", "n_variants", "n_converted",
                                                       "n_value_divergent", "n_io_dtype_changed")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
