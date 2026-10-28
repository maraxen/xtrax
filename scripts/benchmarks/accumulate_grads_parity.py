#!/usr/bin/env python3
"""Parity probe: ``xtrax.accumulate_grads`` vs ``optax.microbatching.micro_grad``.

Pre-registered in ``accumulate_grads_parity.bth.toml`` (committed before the
first run; every tolerance below is fixed there). Question: on equivalent
inputs, does optax's microbatching return the same mean gradient, loss and aux
as xtrax's ``accumulate_grads``, so that ``accumulate_grads`` can be deprecated?

Three arms per case:

- (a) ``accumulate_grads(loss_fn, params, stacked, has_aux=...)`` on microbatches
  pre-stacked as ``(n_micro, MICRO, ...)`` (contiguous row blocks).
- (b) ``optax.microbatching.micro_grad(..., accumulator=MEAN)`` on the same rows
  as one un-stacked ``(n_micro * MICRO, ...)`` batch. Its loss and aux come back
  per example, so they are averaged over axis 0 here.
- (c) a plain full-batch ``jax.value_and_grad`` of the mean loss (reference).

Cases: n_micro in {1, 4, 7} x variant in {mlp_aux, mlp_noaux, eqxleaf_aux}
(eqxleaf carries a callable and an int pytree leaf; arm (b) and (c) differentiate
the ``eqx.partition`` array half because jax.value_and_grad rejects non-array
leaves) x dtype in {float32, float64}. Each dtype runs in its own worker
process (``jax_enable_x64`` is process-global).

Negative controls (must exceed the tolerance, else the instrument is untested):
(i) a mis-weighted accumulation (optax SUM instead of MEAN) and (ii) one
perturbed microbatch fed to arm (a) only.

Non-gating probes record the semantic differences that the numbers above cannot
see: a batch-coupled loss, the uneven-batch refusals, and what optax says about a
non-array leaf.

Usage (through bathos, from the repo root):
    bth run --project-slug xtrax -- uv run --no-sync python \\
        scripts/benchmarks/accumulate_grads_parity.py \\
        --out outputs/accumulate_grads_parity/result.json
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from xtrax.training.grad import accumulate_grads

log = logging.getLogger("accumulate_grads_parity")

SEED = 261026
MICRO = 8
D_IN, HIDDEN, D_OUT = 6, 16, 3
N_MICRO = (1, 4, 7)
VARIANTS = ("mlp_aux", "mlp_noaux", "eqxleaf_aux")
DTYPES = ("f32", "f64")
CONTROL_N_MICRO = 4
CONTROL_SHIFT = 0.5
# Fixed in the sidecar; the controls must exceed these.
TOL = {"f32": 1e-6, "f64": 1e-12}
MISSING = 1.0e30  # sentinel for a metric that could not be computed (case error)
METRICS = ("grad_rel", "loss_abs", "aux_abs")
PAIRS = ("ab", "ac", "bc")

# Read from source (file:line) on 261026; see the research note for the table.
SEMANTIC_DIFFERENCES = (
    "input_layout: xtrax needs pre-stacked (n_micro, m, ...) (grad.py:14,59-93); optax "
    "reshapes an un-stacked batch with order='F', so microbatch k holds strided rows "
    "(_microbatching.py:99,414)",
    "grad_granularity: optax micro_grad takes per-EXAMPLE grads of batch-1 slices via vmap "
    "(_microbatching.py:509,644-667); xtrax differentiates the loss over each whole microbatch "
    "(grad.py:108-112); equal only for a loss that is a mean of per-example terms",
    "reduction_default: xtrax is always an unweighted mean (grad.py:119-121); optax defaults "
    "to SUM (_microbatching.py:564) and needs AccumulationType.MEAN",
    "aux_and_loss_shape: xtrax returns (mean_grads, mean_loss[, mean_aux]) "
    "(grad.py:122,127); optax returns (grads, IndividualOutputs(values, metrics, aux)) "
    "per example, caller must average "
    "(_microbatching.py:34,649-654,662)",
    "non_array_leaves: xtrax uses eqx.filter_value_and_grad (grad.py:108,112); optax uses "
    "jax.value_and_grad (_microbatching.py:644) so a callable or int leaf in params raises",
    "uneven_batches: both refuse; optax raises unless batch % microbatch_size == 0 "
    "(_microbatching.py:312), xtrax raises on ragged leading or second axis (grad.py:68,90); "
    "neither weights by microbatch size",
    "memory_profile: xtrax scan stacks all n_micro gradient trees then means them "
    "(grad.py:118-121); optax carries one accumulator through fori_loop but vmaps "
    "per-example grads, so a microbatch holds m copies of the gradient tree "
    "(_microbatching.py:433-441,509)",
    "reduction_order_and_dtype: xtrax is a pairwise jnp.mean over the stacked axis; optax is a "
    "sequential running sum then divide by n in the output dtype (_microbatching.py:190-193), and "
    "raises on non-floating outputs (_microbatching.py:69-77)",
    "loop_and_extras: xtrax is lax.scan with no early stop (grad.py:118); optax is "
    "lax.fori_loop with num_real_microbatches early stop, SUM/MEAN/RUNNING_MEAN/CONCAT "
    "accumulators, transform_fn and metrics_fn (_microbatching.py:439,566-568); neither "
    "uses jax.checkpoint (grep: no hits)",
    "argnums: optax can differentiate several arguments (_microbatching.py:557,644); xtrax "
    "differentiates only params, the first argument (grad.py:108-112)",
    "filter_spec_unused: xtrax accepts filter_spec but never uses it past the default assignment "
    "(grad.py:15,50-51); optax has no equivalent argument",
)


def jdtype(tag: str) -> Any:
    return jnp.float64 if tag == "f64" else jnp.float32


class Net(eqx.Module):
    """Two-layer net with two non-array pytree leaves (a callable and an int)."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    act: Callable
    n_pad: int

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.act(x @ self.w1 + self.b1) @ self.w2 + self.b2


def predict_dict(params: Any, x: jax.Array) -> jax.Array:
    h = jnp.tanh(x @ params["l1"]["w"] + params["l1"]["b"])
    return h @ params["l2"]["w"] + params["l2"]["b"]


def predict_net(net: Net, x: jax.Array) -> jax.Array:
    return net(x)


def make_loss(predict: Callable, has_aux: bool, coupled: bool = False) -> Callable:
    """Mean-of-rows loss with a scalar and a vector aux; ``coupled`` adds a batch-mean term."""

    def loss_fn(params: Any, batch: Any) -> Any:
        pred = predict(params, batch["x"])
        err = pred - batch["y"]
        loss = jnp.mean(0.5 * jnp.sum(err**2, axis=-1))
        if coupled:
            loss = loss + jnp.mean(err) ** 2
        if not has_aux:
            return loss
        aux = {
            "mse": jnp.mean(err**2),
            "mean_abs_pred": jnp.mean(jnp.abs(pred)),
            "per_dim": jnp.mean(err**2, axis=0),
        }
        return loss, aux

    return loss_fn


def build(variant: str, tag: str, n_micro: int, coupled: bool = False) -> tuple:
    """Return ``(params, batch, loss_fn, has_aux)`` for one case; data is drawn in float64."""
    rng = np.random.default_rng(SEED + n_micro)
    dt = jdtype(tag)
    n_rows = MICRO * n_micro

    def arr(*shape: int, scale: float = 1.0) -> jax.Array:
        return jnp.asarray(rng.normal(size=shape) * scale, dtype=dt)

    w1, b1 = arr(D_IN, HIDDEN, scale=0.5), arr(HIDDEN, scale=0.1)
    w2, b2 = arr(HIDDEN, D_OUT, scale=0.5), arr(D_OUT, scale=0.1)
    batch = {"x": arr(n_rows, D_IN), "y": arr(n_rows, D_OUT)}
    has_aux = variant != "mlp_noaux"
    if variant == "eqxleaf_aux":
        params: Any = Net(w1, b1, w2, b2, act=jnp.tanh, n_pad=3)
        predict = predict_net
    else:
        params = {"l1": {"w": w1, "b": b1}, "l2": {"w": w2, "b": b2}}
        predict = predict_dict
    leaf = jax.tree.leaves(eqx.filter(params, eqx.is_inexact_array))[0]
    if leaf.dtype != dt:
        raise RuntimeError(f"expected {dt}, built {leaf.dtype}; is jax_enable_x64 set for {tag}?")
    return params, batch, make_loss(predict, has_aux, coupled), has_aux


def stack(batch: Any, n_micro: int) -> Any:
    return jax.tree.map(lambda a: a.reshape(n_micro, MICRO, *a.shape[1:]), batch)


def arm_a(loss_fn: Callable, params: Any, stacked: Any, has_aux: bool) -> tuple:
    out = accumulate_grads(loss_fn, params, stacked, has_aux=has_aux)
    return out if has_aux else (*out, None)


def arm_b(
    loss_fn: Callable,
    params: Any,
    batch: Any,
    has_aux: bool,
    accumulator: Any = optax.microbatching.AccumulationType.MEAN,
) -> tuple:
    """optax micro_grad on the array half of ``params``; loss and aux averaged over examples."""
    diff, static = eqx.partition(params, eqx.is_inexact_array)

    def f(d: Any, b: Any) -> Any:
        return loss_fn(eqx.combine(d, static), b)

    mg = optax.microbatching.micro_grad(
        f,
        has_aux=has_aux,
        argnums=0,
        batch_argnums=1,
        keep_batch_dim=True,
        microbatch_size=MICRO,
        accumulator=accumulator,
    )
    grads, ind = mg(diff, batch)
    loss = jnp.mean(ind.values)
    aux = jax.tree.map(lambda v: jnp.mean(v, axis=0), ind.aux) if has_aux else None
    return grads, loss, aux


def arm_c(loss_fn: Callable, params: Any, batch: Any, has_aux: bool) -> tuple:
    """Plain full-batch jax.value_and_grad over the array half of ``params``."""
    diff, static = eqx.partition(params, eqx.is_inexact_array)

    def f(d: Any) -> Any:
        return loss_fn(eqx.combine(d, static), batch)

    if has_aux:
        (loss, aux), grads = jax.value_and_grad(f, has_aux=True)(diff)
        return grads, loss, aux
    loss, grads = jax.value_and_grad(f)(diff)
    return grads, loss, None


def _flat64(tree: Any) -> np.ndarray:
    leaves = [np.asarray(x, dtype=np.float64).ravel() for x in jax.tree.leaves(tree)]
    return np.concatenate(leaves) if leaves else np.zeros(0)


def rel_l2(x: Any, ref: Any) -> float:
    """||x - ref||_2 / ||ref||_2 over all leaves, in float64 on host."""
    fx, fr = _flat64(x), _flat64(ref)
    if fx.shape != fr.shape:
        raise ValueError(f"leaf layout differs: {fx.shape} vs {fr.shape}")
    return float(np.linalg.norm(fx - fr) / max(np.linalg.norm(fr), 1e-300))


def abs_diff(x: Any, y: Any) -> float:
    """Max absolute difference over all leaves (0.0 when both are None)."""
    if x is None and y is None:
        return 0.0
    fx, fy = _flat64(x), _flat64(y)
    if fx.shape != fy.shape:
        raise ValueError(f"leaf layout differs: {fx.shape} vs {fy.shape}")
    return float(np.max(np.abs(fx - fy)))


def compare(r1: tuple, r2: tuple, ref_grad: Any) -> dict[str, float]:
    """Grad rel-L2 against ``ref_grad`` plus loss and aux abs diffs between two arms."""
    out = {
        "grad_rel": rel_l2(r1[0], ref_grad),
        "loss_abs": abs_diff(r1[1], r2[1]),
        "aux_abs": abs_diff(r1[2], r2[2]),
    }
    bad = [k for k, v in out.items() if not math.isfinite(v)]
    if bad:
        raise FloatingPointError(f"non-finite metric(s): {bad}")
    return out


def run_case(tag: str, n_micro: int, variant: str) -> dict[str, Any]:
    name = f"{tag}_n{n_micro}_{variant}"
    row: dict[str, Any] = {"name": name, "error": ""}
    try:
        params, batch, loss_fn, has_aux = build(variant, tag, n_micro)
        ra = arm_a(loss_fn, params, stack(batch, n_micro), has_aux)
        rb = arm_b(loss_fn, params, batch, has_aux)
        rc = arm_c(loss_fn, params, batch, has_aux)
        # Grad rel-L2 reference: arm (a) for a-vs-b, arm (c) for a-vs-c and b-vs-c.
        pairs = {
            "ab": compare(rb, ra, ra[0]),
            "ac": compare(ra, rc, rc[0]),
            "bc": compare(rb, rc, rc[0]),
        }
        for pair, m in pairs.items():
            for k, v in m.items():
                row[f"{k}_{pair}"] = v
    except Exception as exc:  # noqa: BLE001 - recorded, counted, never swallowed
        log.exception("case %s failed", name)
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        for k in METRICS:
            for pair in PAIRS:
                row[f"{k}_{pair}"] = MISSING
    return row


def run_controls(tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {"error": ""}
    try:
        params, batch, loss_fn, has_aux = build("mlp_aux", tag, CONTROL_N_MICRO)
        stacked = stack(batch, CONTROL_N_MICRO)
        ra = arm_a(loss_fn, params, stacked, has_aux)
        rb = arm_b(loss_fn, params, batch, has_aux)
        base = compare(rb, ra, ra[0])
        rb_sum = arm_b(
            loss_fn, params, batch, has_aux, accumulator=optax.microbatching.AccumulationType.SUM
        )
        sum_m = compare(rb_sum, ra, ra[0])
        shifted = jax.tree.map(lambda a: a, stacked)
        shifted["x"] = shifted["x"].at[1].add(CONTROL_SHIFT)
        pert_m = compare(arm_a(loss_fn, params, shifted, has_aux), rb, rb[0])
        tol = TOL[tag]
        out.update(
            {
                "base_grad_rel": base["grad_rel"],
                "sum_grad_rel": sum_m["grad_rel"],
                "pert_grad_rel": pert_m["grad_rel"],
                "pert_loss_abs": pert_m["loss_abs"],
                "pert_aux_abs": pert_m["aux_abs"],
                "sum_exceeds": bool(sum_m["grad_rel"] > tol),
                "pert_exceeds": bool(
                    pert_m["grad_rel"] > tol
                    and pert_m["loss_abs"] > tol
                    and pert_m["aux_abs"] > tol
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("controls failed")
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out.update(
            {
                "base_grad_rel": MISSING,
                "sum_grad_rel": MISSING,
                "pert_grad_rel": MISSING,
                "pert_loss_abs": MISSING,
                "pert_aux_abs": MISSING,
                "sum_exceeds": False,
                "pert_exceeds": False,
            }
        )
    return out


def run_probes(tag: str) -> dict[str, Any]:
    """Non-gating observations backing the semantic-difference list."""
    probes: dict[str, Any] = {}
    # (1) batch-coupled loss: per-microbatch (xtrax) vs per-example (optax) vs full batch.
    params, batch, loss_fn, has_aux = build("mlp_aux", tag, CONTROL_N_MICRO, coupled=True)
    ra = arm_a(loss_fn, params, stack(batch, CONTROL_N_MICRO), has_aux)
    rb = arm_b(loss_fn, params, batch, has_aux)
    rc = arm_c(loss_fn, params, batch, has_aux)
    probes["coupled_grad_rel_ab"] = rel_l2(rb[0], ra[0])
    probes["coupled_grad_rel_ac"] = rel_l2(ra[0], rc[0])
    probes["coupled_grad_rel_bc"] = rel_l2(rb[0], rc[0])
    # (2) optax on a pytree with non-array leaves, without the eqx.partition workaround.
    net, nbatch, nloss, naux = build("eqxleaf_aux", tag, CONTROL_N_MICRO)
    try:
        mg = optax.microbatching.micro_grad(
            nloss, has_aux=naux, microbatch_size=MICRO,
            accumulator=optax.microbatching.AccumulationType.MEAN,
        )  # fmt: skip
        mg(net, nbatch)
        probes["optax_direct_nonarray_error"] = "none"
    except Exception as exc:  # noqa: BLE001
        probes["optax_direct_nonarray_error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        accumulate_grads(nloss, net, stack(nbatch, CONTROL_N_MICRO), has_aux=naux)
        probes["xtrax_nonarray_ok"] = True
    except Exception as exc:  # noqa: BLE001
        probes["xtrax_nonarray_ok"] = False
        probes["xtrax_nonarray_error"] = f"{type(exc).__name__}: {exc}"[:200]
    # (3) uneven inputs: optax batch not divisible; xtrax ragged second axis.
    dparams, dbatch, dloss, daux = build("mlp_aux", tag, 2)
    ragged = jax.tree.map(lambda a: a[: MICRO * 2 - 6], dbatch)  # 10 rows, microbatch 8
    try:
        arm_b(dloss, dparams, ragged, daux)
        probes["optax_uneven_raises"] = False
    except ValueError as exc:
        probes["optax_uneven_raises"] = True
        probes["optax_uneven_error"] = str(exc)[:120]
    bad = {"x": jnp.zeros((3, 4, D_IN), jdtype(tag)), "y": jnp.zeros((3, 5, D_OUT), jdtype(tag))}
    try:
        accumulate_grads(dloss, dparams, bad, has_aux=daux)
        probes["xtrax_ragged_raises"] = False
    except ValueError as exc:
        probes["xtrax_ragged_raises"] = True
        probes["xtrax_ragged_error"] = str(exc)[:120]
    return probes


def worker(tag: str, out_path: Path) -> int:
    want_x64 = tag == "f64"
    if bool(jax.config.jax_enable_x64) != want_x64:
        log.error("jax_enable_x64=%s does not match %s", jax.config.jax_enable_x64, tag)
        return 2
    cases = [run_case(tag, n, v) for n in N_MICRO for v in VARIANTS]
    result = {
        "dtype": tag,
        "cases": cases,
        "controls": run_controls(tag),
        "probes": run_probes(tag),
        "platform": jax.default_backend(),
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "equinox_version": eqx.__version__,
    }
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False))
    log.info("worker %s wrote %s", tag, out_path)
    return 0


def spawn_worker(tag: str, out_path: Path, timeout_s: int) -> dict[str, Any] | None:
    env = dict(os.environ)
    env["JAX_ENABLE_X64"] = "1" if tag == "f64" else "0"
    env.setdefault("OMP_NUM_THREADS", "4")
    env["JAX_PLATFORMS"] = "cpu"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", tag, "--out", str(out_path)]
    log.info("spawning worker %s (timeout %ds)", tag, timeout_s)
    out_path.unlink(missing_ok=True)
    try:
        proc = subprocess.run(cmd, env=env, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        log.error("worker %s timed out", tag)
        return None
    if proc.returncode != 0 or not out_path.exists():
        log.error("worker %s failed rc=%s", tag, proc.returncode)
        return None
    return json.loads(out_path.read_text())


def flatten(results: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    """Merge per-dtype worker results into the flat, schema-typed result record."""
    flat: dict[str, Any] = {}
    per_dtype = len(N_MICRO) * len(VARIANTS)
    n_expected = per_dtype * len(DTYPES)
    n_ran = n_errors = 0
    controls_ok = True
    platform = jax_v = optax_v = eqx_v = "unknown"
    for tag in DTYPES:
        res = results.get(tag)
        maxes = {f"{k}_{p}": 0.0 for k in METRICS for p in PAIRS}
        if res is None:
            n_errors += per_dtype
            maxes = {k: MISSING for k in maxes}
            controls_ok = False
            flat[f"{tag}_worker_ok"] = False
        else:
            flat[f"{tag}_worker_ok"] = True
            platform, jax_v = res["platform"], res["jax_version"]
            optax_v, eqx_v = res["optax_version"], res["equinox_version"]
            for case in res["cases"]:
                if case["error"]:
                    n_errors += 1
                else:
                    n_ran += 1
                for key in maxes:
                    value = float(case[key])
                    maxes[key] = max(maxes[key], value)
                    flat[f"case_{case['name']}_{key}"] = value
                if case["error"]:
                    flat[f"case_{case['name']}_error"] = case["error"]
            ctl = res["controls"]
            ok = bool(ctl["sum_exceeds"] and ctl["pert_exceeds"] and not ctl["error"])
            controls_ok = controls_ok and ok
            for key in (
                "base_grad_rel",
                "sum_grad_rel",
                "pert_grad_rel",
                "pert_loss_abs",
                "pert_aux_abs",
            ):
                flat[f"ctrl_{tag}_{key}"] = float(ctl[key])
            flat[f"ctrl_{tag}_sum_exceeds"] = bool(ctl["sum_exceeds"])
            flat[f"ctrl_{tag}_pert_exceeds"] = bool(ctl["pert_exceeds"])
            for key, value in res["probes"].items():
                flat[f"probe_{tag}_{key}"] = value
        for key, value in maxes.items():
            flat[f"{tag}_max_{key}"] = value
        flat[f"{tag}_tol"] = TOL[tag]
    flat.update(
        {
            "n_cases_expected": n_expected,
            "n_cases_ran": n_ran,
            "n_case_errors": n_errors,
            "controls_ok": controls_ok,
            "semantic_differences": len(SEMANTIC_DIFFERENCES),
            "semantic_difference_list": " || ".join(SEMANTIC_DIFFERENCES),
            "platform": platform,
            "jax_version": jax_v,
            "optax_version": optax_v,
            "equinox_version": eqx_v,
        }
    )
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="result JSON (durable path)")
    parser.add_argument("--worker", choices=DTYPES, default=None, help="internal: run one dtype")
    parser.add_argument("--worker-timeout", type=int, default=900, help="seconds per dtype worker")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.worker:
        return worker(args.worker, args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {
        tag: spawn_worker(
            tag, args.out.with_name(f"{args.out.stem}.worker_{tag}.json"), args.worker_timeout
        )
        for tag in DTYPES
    }
    flat = flatten(results)
    text = json.dumps(flat, indent=2, allow_nan=False)
    args.out.write_text(text)
    results_path = os.environ.get("BTH_RESULTS_PATH")
    if results_path:
        Path(results_path).write_text(text)
    summary_keys = [k for k in flat if k.startswith(("f32_max_", "f64_max_"))]
    log.info(
        "controls_ok=%s n_ran=%s n_err=%s",
        flat["controls_ok"],
        flat["n_cases_ran"],
        flat["n_case_errors"],
    )
    for key in summary_keys:
        log.info("%s = %.3e", key, flat[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
