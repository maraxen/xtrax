"""Measure the IREE split-derived-key miscompile behind aminx #5093.

This is the reduced, self-contained reproducer for the divergence that
`xtrax.export.safety`'s ``"random-permutation"`` rule refuses at plan time. It
imports **only jax and iree** -- no xtrax, no aminx, no checkpoint -- so it can
be handed to IREE upstream as-is.

How it was reduced (each step is a measurement, not a guess):

1. The aminx configuration that reproduces was cut to a 0-encoder-layer model
   with random weights, 39x smaller than the original, at which point the
   decoding order was wrong in every position while the logits were EXACT --
   localising the whole defect to the integer path.
2. ``score.py`` draws that order with ``jax.random.permutation`` and then
   **returns it unconsumed**, so the surrounding model was scenery. Removing it
   entirely still reproduces.
3. ``jax.random.permutation`` is one round of ``lax.sort_key_val`` over drawn
   uint32 keys. Returning those drawn keys alongside the order showed **8 of 8
   key bits differ**: the sort is innocent, it is faithfully sorting different
   numbers. Removing the sort keeps the divergence.
4. What remains is two necessary ingredients, each measured by removal below.

**This is NOT the stable-sort tie-order defect** that `measure_iree_sort_stability.py`
covers and that aminx #156/#157 fixed, despite both surfacing as a scrambled
decoding order. No sort participates here, and the two bugs need different
fixes.

Six measurements:

M1  The minimal reproducer: a NESTED split (a split of a split half) feeding a
    draw, with a second draw live elsewhere. Both outputs come back with
    completely different bits. Valid-looking values, no crash, and a
    float-tolerance parity check on an integer-carried divergence reports 0.0.

M2  NEGATIVE CONTROL -- remove the nesting. One split, both halves drawn from,
    is bit-exact. Without this, "randomness is broken under IREE" would be
    indistinguishable from the far narrower thing actually measured.

M3  NEGATIVE CONTROL -- remove the second draw. The nested split alone, with
    its sibling half dead, is bit-exact.

M4  The repair that made this findable: one extra live read of the RAW,
    undivided key makes the whole program exact again. The scale factor is
    1e-30 deliberately -- ``* 0`` or a tautological branch constant-folds away
    and silently turns this back into M1.

M5  The sort is not involved. The same program with an explicit
    ``lax.sort_key_val`` added diverges identically, and sorting CONSTANT keys
    with the second draw live is exact.

M6  A SECOND, SEPARATE DEFECT: the same program without ``jax.vmap`` does not
    diverge, it makes the IREE runtime abort with a nonsense buffer length
    (``OUT_OF_RANGE ... length=549755813960`` against a 192-byte binding).
    Recorded here because it reproduces on the same tiny input.

Measured 260912 on iree-base-compiler / iree-base-runtime 3.11. Reproduces
under both the partitionable and non-partitionable threefry lowerings, so it is
not specific to ``jax_threefry_partitionable``.

Requires the `export` and `export-runtime` extras:

    uv run --extra export --extra export-runtime python \
        scripts/measure_iree_split_derived_key.py
"""

import argparse
import logging
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import export as jexport

logger = logging.getLogger("measure_iree_split_derived_key")

DEFAULT_FLAGS = ("--iree-llvmcpu-target-cpu=host",)

#: Two lanes holding the SAME key. Identical lanes keep eager and compiled
#: trivially comparable, and make a lane-dependent result obvious.
LANES = 2


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


def _keys(n: int = LANES) -> jax.Array:
    """The input: a raw uint32[2] PRNG key per lane, as aminx's scorer takes."""
    return jnp.tile(jnp.array([0, 0], dtype=jnp.uint32), (n, 1))


def _agree(tag: str, fn, size: int) -> bool:
    """Run `fn` eagerly and through IREE under vmap; log and compare every output."""
    keys = _keys()
    entry = _build(jax.vmap(fn), jax.ShapeDtypeStruct(keys.shape, keys.dtype))
    got = entry(np.asarray(keys))
    if not isinstance(got, (list, tuple)):
        got = [got]

    eager = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        *[fn(keys[i]) for i in range(LANES)],
    )
    expected = [np.asarray(v) for v in jax.tree.leaves(eager)]

    agree = True
    for i, (exp, act) in enumerate(zip(expected, (np.asarray(v) for v in got), strict=True)):
        bad = int((exp != act).sum())
        agree = agree and bad == 0
        logger.info("%s out[%d]: %d/%d elements differ", tag, i, bad, exp.size)
        logger.info("%s   eager = %s", tag, np.asarray(exp[0]).ravel().tolist()[:size])
        logger.info("%s   iree  = %s", tag, np.asarray(act[0]).ravel().tolist()[:size])
    return agree


def _second_consumer(key: jax.Array) -> jax.Array:
    """One extra LIVE read of the RAW key, scaled so it cannot fold away."""
    bits = jax.random.bits(key, (8,), dtype=jnp.uint32)
    return bits.sum().astype(jnp.float32) * jnp.float32(1e-30)


def measure_minimal(n: int) -> bool:
    """M1: nested split + a second live draw. Returns True if IREE agrees."""

    def minimal(key):
        k1, k2 = jax.random.split(key)
        nested, _ = jax.random.split(k1)
        return (
            jax.random.bits(nested, (n,), dtype=jnp.uint32),
            jax.random.normal(k2, (n,), dtype=jnp.float32),
        )

    agree = _agree("M1", minimal, n)
    logger.info(
        "M1 VERDICT: %s -- a nested split feeding a draw, with a second draw "
        "live, returns different bits under IREE.",
        "AGREES (defect not reproduced)" if agree else "DIVERGES",
    )
    return agree


def measure_without_nesting(n: int) -> bool:
    """M2 (negative control): one split, both halves drawn. Expect agreement."""

    def flat(key):
        k1, k2 = jax.random.split(key)
        return (
            jax.random.bits(k1, (n,), dtype=jnp.uint32),
            jax.random.normal(k2, (n,), dtype=jnp.float32),
        )

    agree = _agree("M2", flat, n)
    logger.info(
        "M2 VERDICT: %s -- removing ONLY the nesting is expected to be exact; "
        "a divergence here would mean the defect is broader than M1 claims.",
        "agrees, as expected" if agree else "DIVERGES -- M1's scope is wrong",
    )
    return agree


def measure_without_second_draw(n: int) -> bool:
    """M3 (negative control): nested split, sibling half dead. Expect agreement."""

    def lone(key):
        k1, _k2 = jax.random.split(key)
        nested, _ = jax.random.split(k1)
        return (jax.random.bits(nested, (n,), dtype=jnp.uint32),)

    agree = _agree("M3", lone, n)
    logger.info(
        "M3 VERDICT: %s -- removing ONLY the second draw is expected to be exact.",
        "agrees, as expected" if agree else "DIVERGES -- M1's scope is wrong",
    )
    return agree


def measure_extra_raw_read(n: int) -> bool:
    """M4: one more live read of the raw key repairs M1. Expect agreement."""

    def repaired(key):
        k1, k2 = jax.random.split(key)
        nested, _ = jax.random.split(k1)
        drawn = jax.random.bits(nested, (n,), dtype=jnp.uint32)
        floats = jax.random.normal(k2, (n,), dtype=jnp.float32) + _second_consumer(key)
        return drawn, floats

    agree = _agree("M4", repaired, n)
    logger.info(
        "M4 VERDICT: %s -- one extra live read of the RAW key restores exactness. "
        "This is diagnostic, not a supported workaround.",
        "agrees, as expected" if agree else "DIVERGES -- the repair no longer holds",
    )
    return agree


def measure_sort_not_involved(n: int) -> tuple[bool, bool]:
    """M5: adding a sort changes nothing; sorting constants is exact."""

    def with_sort(key):
        k1, k2 = jax.random.split(key)
        nested, _ = jax.random.split(k1)
        drawn = jax.random.bits(nested, (n,), dtype=jnp.uint32)
        _, order = jax.lax.sort_key_val(drawn, jnp.arange(n, dtype=jnp.int32))
        floats = jax.random.normal(k2, (n,), dtype=jnp.float32)
        return order, drawn, floats

    def const_sort(key):
        _k1, k2 = jax.random.split(key)
        fixed = jnp.array([7, 3, 5, 1, 6, 0, 4, 2], dtype=jnp.uint32)[:n]
        _, order = jax.lax.sort_key_val(fixed, jnp.arange(fixed.size, dtype=jnp.int32))
        return order, jax.random.normal(k2, (n,), dtype=jnp.float32)

    sorted_agree = _agree("M5a", with_sort, n)
    const_agree = _agree("M5b", const_sort, n)
    logger.info(
        "M5 VERDICT: with a sort -> %s; sorting CONSTANTS with the second draw "
        "live -> %s. The sort follows the drawn values; it does not cause this.",
        "diverges" if not sorted_agree else "agrees",
        "agrees" if const_agree else "diverges",
    )
    return sorted_agree, const_agree


def measure_unvmapped_crash(n: int) -> bool:
    """M6: the same program with no vmap aborts the runtime. True if it ran."""

    def minimal(key):
        k1, k2 = jax.random.split(key)
        nested, _ = jax.random.split(k1)
        return (
            jax.random.bits(nested, (n,), dtype=jnp.uint32),
            jax.random.normal(k2, (n,), dtype=jnp.float32),
        )

    key = jnp.array([0, 0], dtype=jnp.uint32)
    try:
        entry = _build(minimal, jax.ShapeDtypeStruct(key.shape, key.dtype))
        entry(np.asarray(key))
    except Exception as exc:  # noqa: BLE001 - the abort is the measurement
        logger.info("M6 runtime aborted: %s", str(exc).splitlines()[0])
        logger.info(
            "M6 VERDICT: the un-vmapped program does not merely diverge -- the "
            "IREE runtime rejects its own command buffer. Separate defect."
        )
        return False
    logger.info("M6 VERDICT: the un-vmapped program ran; the runtime abort has closed.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="draw width")
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help="exit 1 while the defect still reproduces",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        import iree.compiler  # noqa: F401  # ty: ignore[unresolved-import]
        import iree.runtime  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError:
        logger.error(
            "IREE is not installed. Run: uv run --extra export --extra export-runtime python %s",
            sys.argv[0],
        )
        return 2

    minimal_agree = measure_minimal(args.n)
    flat_agree = measure_without_nesting(args.n)
    lone_agree = measure_without_second_draw(args.n)
    repaired_agree = measure_extra_raw_read(args.n)
    sorted_agree, const_agree = measure_sort_not_involved(args.n)
    unvmapped_ran = measure_unvmapped_crash(args.n)

    controls_hold = flat_agree and lone_agree and repaired_agree and const_agree
    reproduces = not minimal_agree and not sorted_agree

    logger.info(
        "SUMMARY: reproduces=%s controls_hold=%s unvmapped_ran=%s",
        reproduces,
        controls_hold,
        unvmapped_ran,
    )
    if not controls_hold:
        logger.error(
            "A negative control diverged. The finding's SCOPE is wrong, not just "
            "its status -- do not report M1 without re-reducing first."
        )
        return 1
    if args.fail_on_divergence and reproduces:
        logger.error("the split-derived-key miscompile still reproduces")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
