"""Measure the JAX/numpy behaviours that sprint 260922_conformance-residuals rests on.

This script is the evidence behind
`.praxia/docs/specs/260922_conformance-residuals.md` §1.1 (probes P1-P12). It is
tracked rather than throwaway because the spec cites its results, and because
several of them are JAX-version facts that can change between releases: where
donation surfaces in a jaxpr, how sub-byte dtypes bitcast, and whether a numpy
conversion goes through `ArrayImpl._value`. Re-run it after a JAX bump.

Measurements (all on the default backend; the spec's numbers are JAX_PLATFORMS=cpu):

P0   Where `donate_argnums` / `donate_argnames` / a nested donating jit appear in
     `jax.make_jaxpr` output, and whether a jitted callable exposes donation.
P1   Whether the backend honours donation (`x.is_deleted()` after the call).
P1b  Whether an eager fn calling a donating inner jit deletes the CALLER's buffer.
P2   Whether `np.unique(rows, axis=0)` merges `-0.0` with `+0.0`, and NaN rows.
P3   Whether mixed-dtype `jnp.concatenate` collapses distinct int32 rows.
P4   Whether `jnp.moveaxis` truncates a numpy int64 leaf when x64 is off.
P5   How `jax.device_put(x, donate=True)` is recorded, and whether eager use deletes.
P6   Which dtypes `lax.bitcast_convert_type(., uint8)` accepts, and their shapes.
P7   Whether `np.asarray` accepts a typed PRNG key array.
P8   Which host-conversion routes go through `jax._src.array.ArrayImpl._value`.
P9   Whether signed zero survives a byte bitcast (incl. complex imag via lax.imag).
P10  Whether `jax.transfer_guard_device_to_host('disallow')` fires.
P11  int4 bitcast with odd/even trailing dims, and exactness of astype(int8).
P12  Zero-width rows through np.unique / reshape / bitcast / concatenate.

Usage:

    JAX_PLATFORMS=cpu uv run --extra dev python \
        scripts/measure_conformance_residuals_260922.py [--json out.json] [--only P1 P8]
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.array import ArrayImpl

logger = logging.getLogger("measure_conformance_residuals_260922")


def _donated_params(closed: Any) -> list[tuple[str, Any]]:
    return [(e.primitive.name, e.params.get("donated_invars")) for e in closed.jaxpr.eqns]


def _add(x: jax.Array, y: jax.Array) -> jax.Array:
    return x * 2.0 + y


def _outer_donating(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.jit(_add, donate_argnums=1)(x, y) + 1.0


def measure_p0() -> dict[str, Any]:
    x, y = jnp.ones((4,)), jnp.ones((4,))
    jf = jax.jit(_add, donate_argnums=0)
    return {
        "plain": _donated_params(jax.make_jaxpr(_add)(x, y)),
        "jit_no_donation": _donated_params(jax.make_jaxpr(jax.jit(_add))(x, y)),
        "jit_donate_argnums_0": _donated_params(jax.make_jaxpr(jf)(x, y)),
        "jit_donate_argnames_y": _donated_params(
            jax.make_jaxpr(jax.jit(_add, donate_argnames="y"))(x, y)
        ),
        "nested_inner_donation": _donated_params(jax.make_jaxpr(_outer_donating)(x, y)),
        "jitted_has_donate_attr": any(
            hasattr(jf, n) for n in ("donate_argnums", "_donate_argnums")
        ),
    }


def measure_p1() -> dict[str, Any]:
    x = jnp.ones((1024,), jnp.float32)
    jax.jit(lambda a: a * 2.0, donate_argnums=0)(x)
    inner = jax.jit(lambda a: a + 1.0, donate_argnums=0)
    y = jnp.ones((1024,), jnp.float32)
    _ = inner(y) * 2.0  # eager outer body: the inner jit dispatches with donation
    return {"p1_top_level_deleted": x.is_deleted(), "p1b_caller_buffer_deleted": y.is_deleted()}


def measure_p2() -> dict[str, Any]:
    zeros = np.array([[0.0, 1.0], [-0.0, 1.0]], dtype=np.float32)
    nans = np.array([[np.nan, 1.0], [np.nan, 1.0]], dtype=np.float32)
    return {
        "signed_zero_unique_rows": int(len(np.unique(zeros, axis=0))),
        "identical_nan_unique_rows": int(len(np.unique(nans, axis=0))),
    }


def measure_p3() -> dict[str, Any]:
    ints = jnp.array([[2**24], [2**24 + 1]], dtype=jnp.int32)
    flts = jnp.array([[1.0], [1.0]], dtype=jnp.float32)
    cat = jnp.concatenate([ints, flts], axis=1)
    return {
        "concat_dtype": str(cat.dtype),
        "distinct_rows_compare_equal": bool(jnp.all(cat[0] == cat[1])),
    }


def measure_p4() -> dict[str, Any]:
    a = np.array([[2**31 + 5], [2**31 + 6]], dtype=np.int64)
    m = jnp.moveaxis(a, 0, 0)
    return {
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "moveaxis_dtype": str(m.dtype),
        "values": np.asarray(m).ravel().tolist(),
    }


def _device_put_donating(x: jax.Array) -> jax.Array:
    return jax.device_put(x, donate=True) * 2.0


def measure_p5() -> dict[str, Any]:
    closed = jax.make_jaxpr(_device_put_donating)(jnp.ones((4,)))
    eqns = [
        {"prim": e.primitive.name, "copy_semantics": str(e.params.get("copy_semantics"))}
        for e in closed.jaxpr.eqns
    ]
    x = jnp.ones((256,))
    _device_put_donating(x)
    return {"eqns": eqns, "eager_deletes_caller_buffer": x.is_deleted()}


def _bitcast_shape(dt: Any, shape: tuple[int, ...]) -> str:
    v = jnp.zeros(shape, dtype=dt)
    try:
        out = v.astype(jnp.uint8) if dt == jnp.bool_ else lax.bitcast_convert_type(v, jnp.uint8)
    except (TypeError, ValueError) as exc:
        return f"{type(exc).__name__}"
    return f"{shape}->{tuple(out.shape)}"


def measure_p6() -> dict[str, Any]:
    names = ["complex64", "bfloat16", "bool_", "float16", "float8_e4m3fn", "float8_e5m2", "int8"]
    return {n: _bitcast_shape(getattr(jnp, n), (3, 2)) for n in names if hasattr(jnp, n)}


def measure_p7() -> dict[str, Any]:
    keys = jax.random.split(jax.random.key(0), 4)
    try:
        np.asarray(keys)
    except TypeError:
        return {"np_asarray_typed_keys": "TypeError"}
    return {"np_asarray_typed_keys": "ok"}


def measure_p8() -> dict[str, Any]:
    hits: list[int] = []
    original = ArrayImpl._value

    def spy(self: Any) -> Any:
        hits.append(1)
        return original.fget(self)

    routes: dict[str, Callable[[jax.Array], Any]] = {
        "np.asarray": np.asarray,
        "np.array": np.array,
        "np.ascontiguousarray": np.ascontiguousarray,
        "jax.device_get": jax.device_get,
        ".tolist()": lambda a: a.tolist(),
        "float(a[0])": lambda a: float(a[0]),
        ".__array__()": lambda a: a.__array__(),
        "a[0].item()": lambda a: a[0].item(),
    }
    results: dict[str, int] = {}
    ArrayImpl._value = property(spy)
    try:
        for label, route in routes.items():
            hits.clear()
            route(jnp.arange(8.0))  # fresh array per route: _value caches _npy_value
            results[label] = len(hits)
    finally:
        ArrayImpl._value = original
    return results


def measure_p9() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("float16", "bfloat16", "float32"):
        z = jnp.array([[0.0], [-0.0]], dtype=getattr(jnp, name))
        b = lax.bitcast_convert_type(z, jnp.uint8).reshape(2, -1)
        out[f"{name}_signed_zero_bytes_distinct"] = not bool(jnp.all(b[0] == b[1]))
    c = jnp.array([complex(1.0, 0.0), complex(1.0, -0.0)], dtype=jnp.complex64)
    im = lax.bitcast_convert_type(lax.imag(c), jnp.uint8)
    out["complex64_imag_signed_zero_distinct"] = not bool(jnp.all(im[0] == im[1]))
    return out


def measure_p10() -> dict[str, Any]:
    routes: dict[str, Callable[[jax.Array], Any]] = {
        "np.asarray": np.asarray,
        "np.array": np.array,
        "jax.device_get": jax.device_get,
    }
    out: dict[str, str] = {}
    for label, route in routes.items():
        try:
            with jax.transfer_guard_device_to_host("disallow"):
                route(jnp.arange(8.0))
            out[label] = "no error (guard did not fire)"
        except Exception as exc:  # noqa: BLE001 - recording which error, if any
            out[label] = type(exc).__name__
    return out


def measure_p11() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for shape in ((4, 3), (4, 2)):
        out[f"int4_{shape[0]}x{shape[1]}_bitcast"] = _bitcast_shape(jnp.int4, shape)
    widened = jnp.array([[-8], [7]], dtype=jnp.int4).astype(jnp.int8)
    out["int4_astype_int8_values"] = np.asarray(widened).ravel().tolist()
    out["itemsize_vs_itemsize_bits"] = {
        n: [jnp.dtype(getattr(jnp, n)).itemsize, jax.dtypes.itemsize_bits(getattr(jnp, n))]
        for n in ("int4", "uint4", "int8", "float4_e2m1fn")
        if hasattr(jnp, n)
    }
    return out


def measure_p12() -> dict[str, Any]:
    z = np.zeros((5, 0), dtype=np.uint8)
    uniq, inv = np.unique(z, axis=0, return_inverse=True)
    zj = jnp.zeros((5, 0), jnp.float32)
    b = lax.bitcast_convert_type(zj, jnp.uint8)
    cat = jnp.concatenate([jnp.zeros((5, 0), jnp.uint8), jnp.ones((5, 2), jnp.uint8)], axis=1)
    return {
        "np_unique_zero_width_shape": list(uniq.shape),
        "np_unique_zero_width_inverse": np.asarray(inv).ravel().tolist(),
        "jnp_reshape_zero_width": list(zj.reshape(5, -1).shape),
        "bitcast_zero_width": f"{tuple(b.shape)}->{tuple(b.reshape(5, -1).shape)}",
        "concat_zero_plus_normal": list(cat.shape),
    }


MEASUREMENTS: dict[str, Callable[[], dict[str, Any]]] = {
    "P0": measure_p0,
    "P1": measure_p1,
    "P2": measure_p2,
    "P3": measure_p3,
    "P4": measure_p4,
    "P5": measure_p5,
    "P6": measure_p6,
    "P7": measure_p7,
    "P8": measure_p8,
    "P9": measure_p9,
    "P10": measure_p10,
    "P11": measure_p11,
    "P12": measure_p12,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", metavar="PATH", help="also write all results as JSON to PATH")
    parser.add_argument(
        "--only", nargs="*", choices=sorted(MEASUREMENTS), help="run only these probes"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    logger.info(
        "jax %s, numpy %s, backend %s", jax.__version__, np.__version__, jax.default_backend()
    )
    results: dict[str, Any] = {
        "jax": jax.__version__,
        "numpy": np.__version__,
        "backend": jax.default_backend(),
    }
    for name in args.only or list(MEASUREMENTS):
        results[name] = MEASUREMENTS[name]()
        logger.info("%s %s", name, json.dumps(results[name], default=str))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("wrote %s", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
