"""Measure whether IREE preserves XLA's sort tie-breaking, and whether a
multi-output export survives the round trip.

This script is the evidence behind
`.praxia/docs/specs/260911_export-divergence-mapping.md` §2. It is tracked
rather than throwaway because the spec cites its numbers, and because the
headline claim -- that IREE does not honour `jnp.argsort(stable=True)` -- is
the identified mechanism for the aminx #5093 divergence and will be re-run
against future IREE releases to see when it closes.

Four measurements:

M1  A multi-output (pytree) exported function survives jax.export -> IREE ->
    runtime, and what Python type comes back. The probe design in the spec
    rests on ONE compile yielding MANY named intermediates.

M2  Whether `ireert.Config("local-task")` -- the multi-threaded executor
    `xtrax.export.compile.run_native_vmfb` uses -- is run-to-run
    deterministic. If it is not, every comparison downstream is noise-limited.

M3  Whether IREE reproduces XLA's INTEGER `lax.sort_key_val` tie-break.
    `jax.random.permutation` is implemented as a sort over uint32 random bits
    (`jax._src.random.core._shuffle`), so no floating-point arithmetic
    participates -- FMA/reassociation cannot perturb it, and a stability
    difference is the only remaining mechanism.

M4  Whether IREE honours `jnp.argsort(..., stable=True)` on a tie-heavy float
    row. This is verbatim the construct aminx PR #155 shipped to replace the
    IREE-illegal `jax.lax.top_k`, whose correctness argument depends on stable
    tie-breaking.

Requires the `export` and `export-runtime` extras:

    uv run --extra export --extra export-runtime python \
        scripts/measure_iree_sort_stability.py
"""

import argparse
import logging
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import export as jexport

logger = logging.getLogger("measure_iree_sort_stability")

DEFAULT_FLAGS = ("--iree-llvmcpu-target-cpu=host",)


def _build(fn, *avals, flags=DEFAULT_FLAGS):
    """Export `fn`, compile it for llvm-cpu, and return a callable entry point."""
    import iree.compiler as ireec  # ty: ignore[unresolved-import]
    import iree.runtime as ireert  # ty: ignore[unresolved-import]

    exported = jexport.export(jax.jit(fn))(*avals)
    vmfb = ireec.compile_str(
        exported.mlir_module(),
        target_backends=["llvm-cpu"],
        input_type="stablehlo",
        extra_args=list(flags),
    )
    ctx = ireert.SystemContext(config=ireert.Config("local-task"))
    module = ireert.VmModule.copy_buffer(ctx.instance, vmfb)
    ctx.add_vm_module(module)
    return ctx.modules[module.name]["main"]


def _split_by_dtype(pair):
    """Return (float_array, int_array) from a 2-tuple in unknown pytree order.

    IREE returns a flat tuple carrying no key information, so declaration order
    cannot be assumed -- JAX flattens dict keys in sorted order. Disambiguating
    by dtype is safe only for this script's 2-tuples; real code must recover
    names from the exported ``out_tree``.
    """
    first, second = (np.asarray(v) for v in pair)
    if first.dtype == np.int32:
        return second, first
    return first, second


def measure_multi_output() -> bool:
    """M1: a pytree-output export survives the round trip. Returns True if so."""

    def multi(x):
        return {
            "scaled": x * 2.0,
            "total": jnp.sum(x),
            "order": jnp.argsort(x).astype(jnp.int32),
        }

    x = jnp.arange(8, dtype=jnp.float32) * 0.37
    entry = _build(multi, jax.ShapeDtypeStruct(x.shape, x.dtype))
    out = entry(np.asarray(x))

    logger.info("M1 multi-output: python type=%s len=%d", type(out).__name__, len(out))
    for i, item in enumerate(out):
        arr = np.asarray(item)
        logger.info("M1   [%d] shape=%s dtype=%s", i, arr.shape, arr.dtype)
    logger.info("M1 eager dict keys (sorted): %s", sorted(multi(x).keys()))
    logger.info(
        "M1 VERDICT: multi-output works; keys are NOT carried -- names must be "
        "recovered from the exported out_tree, never from declaration order."
    )
    return len(out) == 3


def measure_determinism(runs: int) -> bool:
    """M2: the multi-threaded local-task executor is run-to-run stable."""

    def multi(x):
        return {"scaled": x * 2.0, "total": jnp.sum(x)}

    x = jnp.arange(64, dtype=jnp.float32) * 0.37
    entry = _build(multi, jax.ShapeDtypeStruct(x.shape, x.dtype))

    first = [np.asarray(v).copy() for v in entry(np.asarray(x))]
    stable = True
    for _ in range(runs - 1):
        again = [np.asarray(v) for v in entry(np.asarray(x))]
        stable = stable and all(np.array_equal(a, b) for a, b in zip(first, again))

    logger.info("M2 determinism: bit-identical across %d runs = %s", runs, stable)
    return stable


def measure_integer_sort_tiebreak(n: int, distinct: int) -> bool:
    """M3: IREE vs XLA on an integer sort with forced exact ties.

    Returns True if they agree (i.e. no divergence found).
    """

    def perm_like(keys):
        """Exactly what jax.random.permutation does: sort values BY uint32 keys."""
        vals = jnp.arange(keys.shape[0], dtype=jnp.int32)
        _, out = jax.lax.sort_key_val(keys, vals)
        return out

    tied = jnp.arange(n, dtype=jnp.uint32) % distinct
    entry = _build(perm_like, jax.ShapeDtypeStruct(tied.shape, tied.dtype))
    iree_perm = np.asarray(entry(np.asarray(tied)))
    xla_perm = np.asarray(perm_like(tied))

    agree = bool(np.array_equal(xla_perm, iree_perm))
    logger.info("M3 integer sort (%d slots, %d distinct keys)", n, distinct)
    logger.info("M3   XLA : %s ...", xla_perm[:16])
    logger.info("M3   IREE: %s ...", iree_perm[:16])
    if not agree:
        differing = np.nonzero(xla_perm != iree_perm)[0]
        logger.info(
            "M3 VERDICT: DIVERGES at %d of %d positions, first at index %d",
            len(differing),
            n,
            differing[0],
        )
    else:
        logger.info("M3 VERDICT: identical")
    return agree


def measure_stable_argsort(length: int, k: int, n_real: int) -> bool:
    """M4: IREE vs XLA on `jnp.argsort(stable=True)` over a tie-heavy float row.

    Mirrors aminx PR #155's top_k replacement on a masked distance matrix.
    Returns True if they agree (i.e. no divergence found).
    """

    def topk_shipped(x):
        order = jnp.argsort(-x, axis=-1, stable=True)[..., :k]
        return jnp.take_along_axis(x, order, axis=-1), order.astype(jnp.int32)

    # A masked distance row: real values in the first `n_real` slots, the rest
    # clamped to one identical sentinel -- exactly what padding produces.
    row = np.concatenate([
        np.linspace(0.1, 1.2, n_real).astype(np.float32),
        np.full(length - n_real, 1.0e4, dtype=np.float32),
    ])
    tied = jnp.asarray(np.stack([row, row[::-1].copy()]))

    entry = _build(topk_shipped, jax.ShapeDtypeStruct(tied.shape, tied.dtype))
    iree_vals, iree_idx = _split_by_dtype(entry(np.asarray(tied)))
    xla_vals, xla_idx = _split_by_dtype(topk_shipped(tied))

    idx_same = bool(np.array_equal(xla_idx, iree_idx))
    val_same = bool(np.array_equal(xla_vals, iree_vals))
    logger.info("M4 stable argsort on a tie-heavy (masked) float row")
    for r in range(xla_idx.shape[0]):
        logger.info("M4   row %d XLA  idx: %s", r, xla_idx[r])
        logger.info("M4   row %d IREE idx: %s", r, iree_idx[r])
    logger.info("M4   indices identical=%s values identical=%s", idx_same, val_same)
    logger.info("M4   max|value diff| = %.3e", float(np.max(np.abs(xla_vals - iree_vals))))

    # Control: the same function on strictly distinct values must agree, or the
    # measurement says nothing about ties specifically.
    distinct = jnp.asarray(np.stack([
        np.linspace(0.1, 9.9, length).astype(np.float32),
        np.linspace(9.9, 0.1, length).astype(np.float32),
    ]))
    entry2 = _build(topk_shipped, jax.ShapeDtypeStruct(distinct.shape, distinct.dtype))
    c_iree_vals, c_iree_idx = _split_by_dtype(entry2(np.asarray(distinct)))
    c_xla_vals, c_xla_idx = _split_by_dtype(topk_shipped(distinct))
    control_same = bool(np.array_equal(c_xla_idx, c_iree_idx)) and bool(
        np.array_equal(c_xla_vals, c_iree_vals)
    )
    logger.info("M4   CONTROL (all-distinct input) identical=%s", control_same)

    if not idx_same and control_same:
        logger.info(
            "M4 VERDICT: indices diverge ONLY under ties while values stay "
            "bit-identical -- IREE does not honour jnp.argsort(stable=True)."
        )
    return idx_same


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=20, help="M2 replay count")
    parser.add_argument("--n", type=int, default=64, help="M3 slot count")
    parser.add_argument("--distinct", type=int, default=4, help="M3 distinct key count")
    parser.add_argument("--length", type=int, default=32, help="M4 row length")
    parser.add_argument("--k", type=int, default=8, help="M4 top-k")
    parser.add_argument("--n-real", type=int, default=12, help="M4 unmasked slots")
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help="Exit non-zero if M3 or M4 diverge (for use as a regression gate).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    try:
        import iree.compiler  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError:
        logger.error(
            "iree-base-compiler is not installed. Re-run with: "
            "uv run --extra export --extra export-runtime python %s",
            sys.argv[0],
        )
        return 2

    measure_multi_output()
    measure_determinism(args.runs)
    m3_agree = measure_integer_sort_tiebreak(args.n, args.distinct)
    m4_agree = measure_stable_argsort(args.length, args.k, args.n_real)

    if args.fail_on_divergence and not (m3_agree and m4_agree):
        logger.error("sort tie-breaking diverges between XLA and IREE")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
