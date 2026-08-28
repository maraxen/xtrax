#!/usr/bin/env python3
"""Measure how well captured JAX IR compresses, to size the ledger blob store.

Answers a design question that should not be settled by intuition: is a
content-addressed file store with per-blob compression enough for StableHLO and
jaxpr text, or does the ledger need a columnar/embedded-DB backend to keep
artifact size manageable?

Run:  uv run python scripts/measure_ir_compression.py
"""

import argparse
import gzip
import hashlib
import lzma
import sys


def _build_fn(size: int, layers: int = 4):
    """An MLP-ish function whose lowered IR is representative, not trivial.

    ``layers`` is unrolled in Python, so it drives the *op count* -- which is
    what IR size actually tracks. Batch size does not change IR size at all;
    that asymmetry is the whole point of measuring both.
    """
    import jax
    import jax.numpy as jnp

    def fn(x, w1, w2, w3):
        h = jnp.tanh(x @ w1)
        h = jnp.tanh(h @ w2)
        for _ in range(layers):
            h = h + jnp.tanh(h @ w2)
        return jax.nn.log_softmax(h @ w3)

    rng = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    x = jax.random.normal(k1, (size, 128))
    w1 = jax.random.normal(k2, (128, 256))
    w2 = jax.random.normal(k3, (256, 256))
    w3 = jax.random.normal(k4, (256, 32))
    return fn, (x, w1, w2, w3)


def _capture(fn, args):
    import jax

    jaxpr = str(jax.make_jaxpr(fn)(*args))
    exported = jax.export.export(jax.jit(fn))(*args)
    stablehlo = exported.mlir_module()
    optimized = jax.jit(fn).lower(*args).compile().as_text()
    return {"jaxpr": jaxpr, "stablehlo": stablehlo, "optimized_hlo": optimized}


def _report(name: str, text: str) -> None:
    raw = text.encode("utf-8")
    gz = gzip.compress(raw, compresslevel=6)
    xz = lzma.compress(raw, preset=6)
    digest = hashlib.sha256(raw).hexdigest()[:12]
    print(
        f"{name:>16}  raw={len(raw):>10,}B  "
        f"gzip={len(gz):>9,}B ({len(raw) / len(gz):5.1f}x)  "
        f"lzma={len(xz):>9,}B ({len(raw) / len(xz):5.1f}x)  sha={digest}"
    )


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64, help="batch dimension")
    parser.add_argument("--layers", type=int, default=4, help="unrolled layer count")
    args = parser.parse_args(argv)

    fn, inputs = _build_fn(args.size, args.layers)
    captured = _capture(fn, inputs)
    print(
        f"\nIR compression, batch={args.size}, layers={args.layers} (gzip/lzma are stdlib-only)\n"
    )
    total_raw = 0
    total_gz = 0
    for name, text in captured.items():
        _report(name, text)
        total_raw += len(text.encode("utf-8"))
        total_gz += len(gzip.compress(text.encode("utf-8"), compresslevel=6))
    print(f"\n  jaxpr+stablehlo+hlo raw={total_raw:,}B  gzip={total_gz:,}B")
    print(f"  ratio={total_raw / total_gz:.1f}x\n")

    # Determinism check (BATHOS.md: verify the measurement pipeline itself).
    # A content-addressed store that always-hits or always-misses would
    # invalidate every audit built on it, so prove the digest is stable across
    # a re-trace and sensitive to a real change.
    again = _capture(fn, inputs)
    for name in captured:
        a = hashlib.sha256(captured[name].encode()).hexdigest()
        b = hashlib.sha256(again[name].encode()).hexdigest()
        print(f"  stable[{name}]: {a == b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
