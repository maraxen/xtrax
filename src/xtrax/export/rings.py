"""Execution-layer comparison rings (R0-R3) for the divergence-mapping ladder.

See ``.praxia/docs/specs/260911_export-divergence-mapping.md`` (revision 5) for
the full design. This module owns the *execution* half of the ladder -- tracing,
compiling, running, and stratifying by input class -- and imports every pure
classification/metric primitive (``LeafDivergence``, ``RingResult``,
``budget_leaf``, ``budget_key``, ``compare_pytree``, ``classify_probes``,
``probe_resolution``, ``validate_probe_deps``) from ``xtrax.export.divergence``
rather than redefining any of them (spec section 6.0).

**Probe model (section 5.1).** A probe is an ordinary named entry in ``fn``'s
own return pytree -- there is no separate "instrumented callable" type. Every
rung receives the *same* ``fn``; what varies is whether that rung reads every
name in its output or only some:

- R0/R1/R2a/R2b run ``fn`` exactly as given and compare (or replay) its whole
  output; none of them edit what ``fn`` computes, and none derive a
  "primary-only" view of it.
- R0 and R2a accept no probe argument at all (AC-11). R1 and R2b optionally
  accept ``probe_deps`` (revision 6, spec AC-11, backlog #5210) **solely as
  classification metadata**: when given, it selects what ``classify_probes``
  reports in ``probes`` from leaves each rung already computed for its own
  comparison -- it never changes what ``fn`` computes or which leaves are
  compared. Omitted (``None``, the default), both rungs behave exactly as
  before revision 6: ``probes=()``. Either way, "nothing type-level
  distinguishes an instrumented callable from a plain one" (spec AC-11), so a
  caller can still pass a probe-carrying ``fn`` to any rung.
- R3 is the only rung that *derives* anything from ``probe_deps`` beyond
  classification: it uses the **DAG sinks** of ``probe_deps`` -- the names
  that are never anyone else's declared predecessor -- to build a second,
  filtered callable returning only the model's true ("primary") output(s), to
  run the section-5.4 fidelity precondition before trusting anything else it
  measures. R1/R2b never strip or filter ``fn``'s output this way.

**Budget keying (divergence.py's own module docstring).** ``classify_probes``
looks up a float leaf's calibrated budget under
``divergence.budget_key(probe_name, leaf.path)``. R2a therefore measures
``m_leaf`` **per top-level name** (comparing each name's own value in
isolation, so ``leaf.path`` is relative to that name alone, per
``budget_key``'s documented convention), not once over the whole output.

Nothing here is measured for coverage (``coverage_omit``, spec section 5.5a) --
every non-trivial branch below requires an IREE toolchain to execute for real.
The pure JAX-only pieces (``recover_probe_names``, the section-6.2 input-class
generators, the pytree-splitting helpers) are exercised directly by
``tests/export/test_rings.py`` without a toolchain; the execution rungs are
exercised there against fakes standing in for ``compile_for_target`` and
``run_native_vmfb``.
"""

import math
import re
import subprocess
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.export.compile import compile_for_target, run_native_vmfb
from xtrax.export.composer import build_traceable_callable
from xtrax.export.divergence import (
    LeafDivergence,
    ProbeReport,
    RingResult,
    _unbudgeted_probes,
    budget_key,
    budget_leaf,
    classify_probes,
    compare_pytree,
    probe_resolution,
    validate_probe_deps,
)
from xtrax.export.targets import NATIVE, NATIVE_PORTABLE, Target

__all__ = [
    "BUCKET_LADDER",
    "InputClassResult",
    "magnitude_extremes",
    "nominal",
    "r0_replay_gate",
    "r1_target_isa",
    "r2a_fusion",
    "r2b_lowering",
    "r3_probe",
    "recover_probe_names",
    "run_ladder",
    "sub_k_neighbours",
    "symmetric_geometry",
]

# The bucket ladder length is not a free axis (spec section 6.2): every
# input-class generator below draws its length from here, never a sweep.
BUCKET_LADDER: tuple[int, ...] = (64, 128, 256, 512, 1024, 1536, 2048)


# ---------------------------------------------------------------------------
# Shared pytree helpers
# ---------------------------------------------------------------------------


def _bare_name(key_path: tuple[Any, ...]) -> str:
    """Strip ``jax.tree_util.keystr``'s container-specific decoration.

    ``keystr`` renders a top-level dict entry as ``"['name']"`` and a
    top-level NamedTuple field as ``".name"``. A declared ``probe_deps`` key
    is a bare name in either case, so both forms are normalized to it.
    """
    return jax.tree_util.keystr(key_path).strip(".[]'\"")


def _top_level_items(structured: Any) -> dict[str, Any]:
    """Split a pytree into its top-level {name: value} entries.

    ``value`` is returned as-is -- possibly itself a nested pytree -- which is
    what lets a per-name ``compare_pytree`` call produce a ``leaf.path``
    relative to that name alone, matching ``divergence.budget_key``'s
    documented convention.

    A structure with no named top level (a bare array, or a plain tuple) is
    treated as a single implicit entry keyed ``""`` -- the same bare-name
    convention ``budget_key`` uses for a probe whose own value is a single
    array.
    """
    if isinstance(structured, Mapping):
        return dict(structured)
    if hasattr(structured, "_asdict"):  # NamedTuple
        return dict(structured._asdict())
    return {"": structured}


def recover_probe_names(out_tree: Any, flat_values: Sequence[Any]) -> dict[str, Any]:
    """Bind a flat runtime output tuple back to top-level names, via ``out_tree``.

    The IREE runtime returns a flat tuple carrying no key information (M1).
    Binding positionally in declaration order is silently wrong for a dict
    (JAX flattens dicts in **sorted key** order) and silently right for a
    NamedTuple (flattened in **field** order) -- so names must be recovered
    from ``out_tree`` rather than assumed from whatever order a caller
    originally wrote fields in.

    This works because ``flat_values`` is guaranteed to already be in
    ``out_tree``'s own flatten order (that is what the runtime actually
    returns); unflattening with ``out_tree`` reconstructs the original
    structure exactly, dict-vs-NamedTuple trap and all, and the top-level
    items of that reconstruction are correctly named regardless of which
    container type was used.

    Args:
        out_tree: A ``jax.tree_util.PyTreeDef``, e.g. ``Exported.out_tree``.
        flat_values: The runtime's flat output tuple, in ``out_tree`` order.

    Returns:
        A dict from bare top-level name to value (AC-2).
    """
    structured = jax.tree_util.tree_unflatten(out_tree, list(flat_values))
    return _top_level_items(structured)


def _probe_result_paths(out_tree: Any) -> dict[str, str]:
    """Map each top-level output name to its exact ``jax.result_info`` string.

    ``divergence.probe_resolution``/``validate_probe_deps`` need this to
    locate a probe's return operand in the exported MLIR module (see their
    docstrings). JAX embeds ``jax.result_info = "result" + keystr(path)`` on
    every return operand (verified empirically against a real
    ``jax.export.export`` output); this only resolves the common case where a
    probe's value is a single array (``len(path) <= 1`` -- 0 for a bare-array
    whole output, 1 for a top-level named entry -- no further nesting) -- a
    probe holding a nested structure has no single result_info string and is
    out of scope for this helper (see ``divergence.py``'s ``budget_key``
    docstring for the general per-leaf case).
    """
    placeholder = jax.tree_util.tree_unflatten(out_tree, [0] * out_tree.num_leaves)
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(placeholder)
    result: dict[str, str] = {}
    for path, _leaf in paths_and_leaves:
        if len(path) > 1:
            continue
        result[_bare_name(path)] = f"result{jax.tree_util.keystr(path)}"
    return result


def _primary_only(fn: Callable[..., Any], strip_names: frozenset[str]) -> Callable[..., Any]:
    """Wrap ``fn`` so its return pytree drops every top-level name in ``strip_names``.

    Used only by R3 to derive the model's "uninstrumented" view (section 5.4)
    -- the true primary output(s), with every probe removed. R0/R1/R2a/R2b
    never call this; they run ``fn`` exactly as given (AC-11).
    """

    def _wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        items = _top_level_items(fn(*args, **kwargs))
        return {name: value for name, value in items.items() if name not in strip_names}

    return _wrapped


def _as_flat_sequence(x: Any) -> Sequence[Any]:
    """Normalize a runtime return value to a sequence of per-output arrays.

    A single-output entry point's runtime call returns a bare array rather
    than a length-1 tuple; every other call site here wants a sequence either
    way.
    """
    if isinstance(x, tuple | list):
        return x
    return (x,)


def _compare_by_name(expected: Any, actual: Any) -> dict[str, tuple[LeafDivergence, ...]]:
    """``compare_pytree``, called once per top-level name, kept separate.

    Doing this per-name (rather than one ``compare_pytree`` call over the
    whole structure) is what makes ``leaf.path`` relative to *that name's own*
    value, matching ``divergence.budget_key``'s documented convention -- a
    whole-structure comparison would make every ``leaf.path`` relative to the
    outer container instead, double-counting the name.

    Raises:
        ValueError: The two sides have different top-level names.
        ProbeStructureError: (from ``compare_pytree``) A shared name's value
            has mismatched treedefs between ``expected`` and ``actual``.
    """
    expected_by_name = _top_level_items(expected)
    actual_by_name = _top_level_items(actual)
    only_expected = expected_by_name.keys() - actual_by_name.keys()
    only_actual = actual_by_name.keys() - expected_by_name.keys()
    if only_expected or only_actual:
        msg = (
            f"top-level name mismatch: only in expected={sorted(only_expected)}, "
            f"only in actual={sorted(only_actual)}"
        )
        raise ValueError(msg)
    return {
        name: compare_pytree(expected_by_name[name], actual_by_name[name])
        for name in sorted(expected_by_name)
    }


def _format_named_leaf_notes(
    leaves_by_name: Mapping[str, tuple[LeafDivergence, ...]],
    budgets: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Render every leaf across every name as a human-readable note."""
    notes: list[str] = []
    for name, leaves in leaves_by_name.items():
        for leaf in leaves:
            key = budget_key(name, leaf.path)
            if leaf.failed:
                notes.append(f"{key}: FAILED ({leaf.message})")
                continue
            metrics = ", ".join(
                f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in leaf.metrics.items()
            )
            budget_note = ""
            if budgets is not None and key in budgets:
                budget_note = f", budget={budgets[key]:.6g}"
            notes.append(f"{key} [{leaf.dtype_class}]: {metrics}{budget_note}")
    return tuple(notes)


# ---------------------------------------------------------------------------
# T8 -- section 6.2 input-class generators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputClassResult:
    """One input-class generator's output, carrying its own contract label.

    Attributes:
        label: The class name, e.g. ``"symmetric_geometry"`` (AC-17).
        in_contract: Whether the artifact is expected to serve this input at
            all. ``False`` only for ``sub_k_neighbours`` -- it is retained as
            a diagnostic but must never carry a verdict alone (AC-17).
        abstract_inputs: ``jax.ShapeDtypeStruct``s for tracing.
        concrete_inputs: Concrete arrays to execute with.
    """

    label: str
    in_contract: bool
    abstract_inputs: tuple[Any, ...]
    concrete_inputs: tuple[Any, ...]


def nominal(abstract_inputs: Sequence[Any], concrete_inputs: Sequence[Any]) -> InputClassResult:
    """Label the caller's own reference input as the ``nominal`` class.

    xtrax cannot manufacture a downstream model's own reference input (that is
    what M6/section 2.3 leaves an open question for a specific model), so this
    generator is a thin, labelled pass-through of whatever the caller already
    uses as its reference fixture.
    """
    return InputClassResult(
        label="nominal",
        in_contract=True,
        abstract_inputs=tuple(abstract_inputs),
        concrete_inputs=tuple(concrete_inputs),
    )


def _shape_dtype(arr: jax.Array) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(arr.shape, arr.dtype)


def _require_bucket_length(length: int) -> None:
    if length not in BUCKET_LADDER:
        msg = (
            f"length {length} is not on the bucket ladder {BUCKET_LADDER}; "
            f"length is not a free axis (spec section 6.2)."
        )
        raise ValueError(msg)


def symmetric_geometry(
    length: int,
    *,
    ndim: int = 3,
    dtype: Any = jnp.float32,
    rise: float = 1.5,
    turn_degrees: float = 100.0,
) -> InputClassResult:
    """An ideal alpha-helix: constant rise and turn per residue (M6, section 2.2).

    This is the **primary** class (not ``sub_k_neighbours``): constant rise and
    turn make ``d(i, j)`` depend only on ``|i - j|``, so pairs at equal
    separation are exactly tied in exact arithmetic -- the load-bearing,
    in-contract tie route the ladder exists to localize. The mask is all-ones:
    this class carries no padding.

    Args:
        length: Residue count. Must be a bucket-ladder length.
        ndim: Coordinate dimensionality.
        dtype: Coordinate dtype.
        rise: Per-residue rise along the helix axis.
        turn_degrees: Per-residue rotation, in degrees.

    Returns:
        An ``InputClassResult`` with ``(coords, mask)`` as its inputs, labelled
        ``"symmetric_geometry"`` and in-contract.
    """
    _require_bucket_length(length)
    if ndim < 2:
        msg = f"symmetric_geometry needs ndim >= 2 to place points on a helix, got {ndim}"
        raise ValueError(msg)
    radius = 2.3
    i = np.arange(length, dtype=np.float64)
    theta = np.deg2rad(turn_degrees) * i
    coords = np.zeros((length, ndim), dtype=np.float64)
    coords[:, 0] = radius * np.cos(theta)
    coords[:, 1] = radius * np.sin(theta)
    if ndim > 2:
        coords[:, 2] = rise * i
    # Extra dims beyond the helix axes stay at 0, preserving exact ties.
    coords_arr = jnp.asarray(coords, dtype=dtype)
    mask = jnp.ones((length,), dtype=jnp.bool_)
    return InputClassResult(
        label="symmetric_geometry",
        in_contract=True,
        abstract_inputs=(_shape_dtype(coords_arr), _shape_dtype(mask)),
        concrete_inputs=(coords_arr, mask),
    )


def magnitude_extremes(
    length: int,
    *,
    ndim: int = 3,
    dtype: Any = jnp.float32,
    seed: int = 0,
) -> InputClassResult:
    """Coordinates spanning ``dtype``'s own denormal and near-overflow magnitudes.

    Half the points sit near ``dtype``'s own overflow boundary, half span
    ``dtype``'s own denormal range -- log-uniform between its smallest
    subnormal and ``1e4x`` its smallest normal, so the class actually
    contains subnormal values rather than only clustering near the
    normal/denormal boundary -- stressing reassociation sensitivity (section
    6.2). The mask is all-ones -- this class is in-contract.

    The generating ``finfo`` is derived from ``dtype`` itself (via
    ``jnp.finfo``, which -- unlike ``np.finfo`` -- accepts ``bfloat16``
    directly, and which needs no dependency this package does not already
    declare) rather than hardcoded to ``float32``: a hardcoded
    ``np.float32`` finfo, cast down into a narrower requested dtype such as
    ``float16``, silently overflowed to ``inf`` and underflowed to exact
    ``0.0`` (finding 3, 260914 code review round 6) -- manufacturing
    non-finite/zero coordinates in a class still labelled ``in_contract=True``.
    """
    _require_bucket_length(length)
    rng = np.random.default_rng(seed)
    finfo = jnp.finfo(dtype)
    half = length // 2
    large = rng.uniform(finfo.max * 0.1, finfo.max * 0.9, size=(half, ndim))
    # `finfo.tiny` is the smallest NORMAL value for `dtype` (~1.18e-38 for
    # float32), not the smallest subnormal (~1.4e-45 for float32) --
    # `rng.uniform(finfo.tiny, ...)` can therefore never draw a genuine
    # denormal value, even though the class is advertised as spanning
    # "denormal" magnitudes (finding 6, 260914 code review round 5; the same
    # `tiny`-vs-`smallest_subnormal` confusion that broke the ULP metric in
    # divergence.py). Drawing in LOG space between `smallest_subnormal` and
    # `tiny * 1e4` -- rather than a straight `rng.uniform` over that huge
    # dynamic range, which would sample almost entirely from the (far wider,
    # in absolute terms) normal side -- guarantees genuine subnormal
    # coverage. `finfo` itself is derived from `dtype`, not hardcoded to
    # float32 (finding 3, 260914 code review round 6), so this is genuinely
    # `dtype`'s own subnormal range, not float32's cast down into it.
    log_lo = np.log(finfo.smallest_subnormal)
    log_hi = np.log(finfo.tiny * 1e4)
    small = np.exp(rng.uniform(log_lo, log_hi, size=(length - half, ndim)))
    signs = rng.choice([-1.0, 1.0], size=(length, ndim))
    coords = np.concatenate([large, small], axis=0) * signs
    coords_arr = jnp.asarray(coords, dtype=dtype)
    mask = jnp.ones((length,), dtype=jnp.bool_)
    return InputClassResult(
        label="magnitude_extremes",
        in_contract=True,
        abstract_inputs=(_shape_dtype(coords_arr), _shape_dtype(mask)),
        concrete_inputs=(coords_arr, mask),
    )


def sub_k_neighbours(
    length: int,
    k_neighbors: int,
    *,
    ndim: int = 3,
    dtype: Any = jnp.float32,
    seed: int = 0,
) -> InputClassResult:
    """A mask leaving fewer than ``k_neighbors`` valid entries -- OUT of contract.

    M5 (section 2.2): this is exactly the regime the artifact *refuses* -- the
    bucket ladder makes padding score-preserving only when
    ``L >= k_neighbors``, and this generator deliberately violates that.
    Retained because a diagnostic tool should be able to characterise inputs
    the artifact rejects, but every result from it must carry
    ``in_contract=False`` (AC-17) so no verdict rests on it alone.

    Args:
        length: Residue count. Must be a bucket-ladder length.
        k_neighbors: The clamp. The generated mask leaves strictly fewer than
            this many valid (unmasked) rows.
        ndim: Coordinate dimensionality.
        dtype: Coordinate dtype.
        seed: RNG seed.

    Returns:
        An out-of-contract ``InputClassResult``.
    """
    _require_bucket_length(length)
    if not 0 < k_neighbors <= length:
        msg = f"k_neighbors={k_neighbors} must be in (0, {length}]"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    coords = rng.normal(size=(length, ndim))
    coords_arr = jnp.asarray(coords, dtype=dtype)
    valid = max(k_neighbors - 1, 0)
    mask_np = np.zeros((length,), dtype=bool)
    mask_np[:valid] = True
    rng.shuffle(mask_np)
    mask = jnp.asarray(mask_np)
    return InputClassResult(
        label="sub_k_neighbours",
        in_contract=False,
        abstract_inputs=(_shape_dtype(coords_arr), _shape_dtype(mask)),
        concrete_inputs=(coords_arr, mask),
    )


# ---------------------------------------------------------------------------
# T4 -- R0 gate + R1
# ---------------------------------------------------------------------------


def _replays_agree(replays: Sequence[Sequence[Any]]) -> tuple[bool, tuple[str, ...]]:
    """Check bit-identity of ``replays`` -- each a flat output tuple."""
    if len(replays) < 2:
        return True, ()
    baseline = replays[0]
    notes: list[str] = []
    ok = True
    for idx, other in enumerate(replays[1:], start=1):
        if len(other) != len(baseline):
            ok = False
            notes.append(f"replay {idx}: output arity differs ({len(other)} vs {len(baseline)})")
            continue
        for slot, (a, b) in enumerate(zip(baseline, other, strict=True)):
            a_np, b_np = np.asarray(a), np.asarray(b)
            if a_np.shape != b_np.shape or not np.array_equal(a_np, b_np, equal_nan=True):
                ok = False
                notes.append(f"replay {idx}, output slot {slot}: not bit-identical to replay 0")
    return ok, tuple(notes)


def r0_replay_gate(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    target: Target = NATIVE,
    replays: int = 3,
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    input_class: str = "nominal",
    in_contract: bool = True,
) -> RingResult:
    """R0: replay one artifact on one input several times. Not a ring (section 3).

    Varies nothing -- it is the ladder's validity gate. If the runtime is
    nondeterministic, ``passed`` is False and **no** other rung may run
    (AC-16); ``run_ladder`` enforces that refusal.

    Args:
        fn: Per-element function, run exactly as given (AC-11: this runner
            accepts no probe argument).
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to replay.
        target: Compilation target for the replayed artifact.
        replays: Number of executions to compare. Must be >= 2 to detect
            anything; a value of 1 trivially "passes" and is a caller error.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label this replay was run under.
        in_contract: Whether ``input_class`` is in-contract (AC-17).

    Returns:
        A ``RingResult`` for ``"R0"`` with ``probes=()`` -- R0 has no probe
        concept, only a pass/fail gate.

    Raises:
        ValueError: ``replays < 2``. Determinism cannot be established from
            fewer than 2 replays -- ``_replays_agree`` trivially returns
            True for a single replay, and (before this check existed)
            ``max(replays, 1)`` let a caller pass ``replays=0`` and still
            get a "passing" gate that had tested nothing (finding 5, 260914
            code review round 5). Raised before any compile.
    """
    if replays < 2:
        msg = (
            f"replays must be >= 2 to detect nondeterminism; got {replays}. "
            f"Fewer than 2 replays trivially 'passes' this gate having "
            f"tested nothing."
        )
        raise ValueError(msg)

    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(callable_))(*abstract_inputs)
    compiled = compile_for_target(exported.mlir_module(), target)

    outputs = [
        list(_as_flat_sequence(run_native_vmfb(compiled.path, *concrete_inputs)))
        for _ in range(replays)
    ]
    passed, notes = _replays_agree(outputs)
    if not notes:
        notes = (f"{len(outputs)} replay(s) bit-identical.",)
    return RingResult(
        ring="R0",
        passed=passed,
        input_class=input_class,
        in_contract=in_contract,
        probes=(),
        notes=notes,
    )


def _read_cpu_features(vmfb_path: Path) -> str:
    """Read back a compiled artifact's ``cpu_features`` attribute string.

    Reuses the approach in ``tests/export/test_parity_multi_size.py``:
    ``iree-dump-module --output=all`` disassembles the embedded HAL-device
    target attribute, which is where ``cpu_features`` actually lives.
    """
    result = subprocess.run(  # noqa: S603, S607
        ["iree-dump-module", "--output=all", str(vmfb_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r'cpu_features = "([^"]*)"', result.stdout)
    if not match:
        msg = f"no cpu_features attribute found in {vmfb_path}"
        raise ValueError(msg)
    return match.group(1)


# The complete x86-64-v2 feature set, as measured off a native-portable
# artifact and quoted verbatim in targets.py / docs/api/export.md / the
# shipped skill reference (same source as
# tests/export/test_parity_multi_size.py's own `_V2_FEATURE_STRING`). Used
# to POSITIVELY confirm a native-portable readback actually carries the
# x86-64-v2 set, rather than merely differing from `native`'s -- see
# `_declares_x86_64_v2` (finding 1, 260914 code review round 5).
_X86_64_V2_FEATURE_STRING = (
    "+cmov,+mmx,+popcnt,+sse,+sse2,+sse4.2,+cx16,+sahf,+cx8,+crc32,+x87,+fxsr"
)
_X86_64_V2_FEATURES: frozenset[str] = frozenset(
    f.lstrip("+") for f in _X86_64_V2_FEATURE_STRING.split(",")
)


def _declares_x86_64_v2(features: str) -> bool:
    """Whether a ``cpu_features`` readback positively declares the x86-64-v2 set.

    Two DIFFERENT ``cpu_features`` strings is not by itself proof of ISA
    divergence: on a non-x86-64 build host, LLVM warns-and-ignores an
    unknown ``-mcpu=x86-64-v2`` target-feature request, so
    ``native-portable`` can carry a completely unrelated (e.g. arm64)
    feature string that still differs from ``native``'s -- differing for the
    WRONG reason (finding 1, 260914 code review round 5). Only a portable
    artifact that actually carries the x86-64-v2 set is trustworthy
    evidence that the two legs' divergence reflects real ISA-dependent
    codegen.
    """
    declared = frozenset(f.lstrip("+") for f in features.split(",") if f)
    return _X86_64_V2_FEATURES <= declared


def _classify_rung_probes(
    ring: str,
    leaves_by_name: Mapping[str, tuple[LeafDivergence, ...]],
    probe_deps: Mapping[str, tuple[str, ...]] | None,
    budgets: Mapping[str, float] | None,
) -> tuple[tuple[ProbeReport, ...], tuple[str, ...]]:
    """Classify leaves R1/R2b already computed, against a caller-declared probe DAG.

    Shared by ``r1_target_isa`` and ``r2b_lowering`` (revision 6, spec AC-11,
    backlog #5210): both rungs already compute a full ``leaves_by_name`` for
    their own structural comparison (``passed``) -- this only decides what
    ``classify_probes`` reports in ``probes`` from those same leaves. It never
    re-runs ``fn`` and never changes which leaves were compared.

    Classification is **per probe** (260915 code review): one probe with a
    diverged float leaf and no calibrated budget for it must not erase every
    other probe's class. Only that probe, and any probe that transitively
    depends on it, is omitted -- everything else is classified normally.

    Args:
        ring: The calling rung's name (``"R1"`` or ``"R2b"``), used only to
            label a "not classified" note.
        leaves_by_name: This rung's own per-name comparison, from
            ``_compare_by_name`` -- classification never re-derives leaves.
        probe_deps: The caller-declared probe DAG, or ``None`` to opt out of
            classification entirely (the pre-revision-6 behaviour).
        budgets: Per-leaf budgets, normally from R2a. Treated as ``{}`` when
            ``None`` (an integer/bool-only probe DAG needs no budget at all;
            a missing float budget is caught below, not here).

    Returns:
        ``((), ())`` when ``probe_deps`` is ``None`` -- ``probes=()``,
        unchanged from before revision 6, so a direct call to either rung
        without ``probe_deps`` keeps its existing behaviour exactly.

        ``((), (note,))`` when a declared probe name (``_probe_iteration_order
        (probe_deps)``, the same declaration-then-sorted order R3 uses) is
        absent from ``leaves_by_name`` -- this rung's own comparison never
        produced a leaf set for it, so there is nothing to classify at all.
        Not raised: ``run_ladder``'s Layer 1 ``_default_validate`` already
        rejects a ``probe_deps`` name absent from ``fn``'s own output before
        any rung runs, so a name reaching here missing from
        ``leaves_by_name`` is not that failure mode, and raising here would
        be caught by Layer 2 and lose this rung's other notes.

        Otherwise, ``(probes, notes)``: ``probes`` holds one ``ProbeReport``
        per **classifiable** declared probe, in declaration order. A probe is
        excluded from ``probes`` iff it has a diverged float leaf with no
        entry in ``budgets`` for it (per :func:`_unbudgeted_probes`), or it
        transitively depends -- via a declared ``probe_deps`` edge -- on a
        probe that is excluded for either reason; every other declared probe
        is classified. When any probe is excluded, ``notes`` names the
        unbudgeted probes and (when non-empty) their dependents; a
        cyclic or unknown probe name in ``probe_deps`` is a declaration error
        that ``run_ladder``'s Layer 1 preflight (``_default_validate``)
        already rejects before any rung runs, so ``classify_probes`` raising
        ``ValueError`` for either is allowed to propagate here rather than
        being silently downgraded to a note.
    """
    if probe_deps is None:
        return (), ()

    names = _probe_iteration_order(probe_deps)
    missing = [name for name in names if name not in leaves_by_name]
    if missing:
        note = (
            f"{ring} probes not classified: probe_deps name(s) {missing} are "
            f"absent from this rung's own leaves; nothing to classify them "
            f"against."
        )
        return (), (note,)

    resolved_budgets: Mapping[str, float] = budgets if budgets is not None else {}
    named = {name: leaves_by_name[name] for name in names}
    unbudgeted = _unbudgeted_probes(named, resolved_budgets)

    # A descendant of an unbudgeted probe cannot itself be honestly
    # classified: its class depends on `max_pred_severity`, which is
    # unresolved for an unbudgeted predecessor (its float leaf could still
    # land WITHIN_BUDGET or BEYOND_BUDGET once a real budget exists) -- so
    # e.g. AMPLIFIED vs ATTENUATED is undecidable. Grown to a fixpoint over
    # `probe_deps`'s declared edges, since a blocked probe can itself have
    # further descendants.
    blocked = set(unbudgeted)
    changed = True
    while changed:
        changed = False
        for name in names:
            if name in blocked:
                continue
            if any(pred in blocked for pred in probe_deps.get(name, ())):
                blocked.add(name)
                changed = True

    classifiable = [name for name in names if name not in blocked]
    classifiable_set = set(classifiable)
    # Every predecessor of a classifiable probe is itself classifiable, by
    # construction of `blocked` above -- `sub_deps` never references a name
    # missing from `classifiable`.
    sub_deps = {k: v for k, v in probe_deps.items() if k in classifiable_set}
    probes = (
        classify_probes({name: named[name] for name in classifiable}, sub_deps, resolved_budgets)
        if classifiable
        else ()
    )

    notes: tuple[str, ...] = ()
    if blocked:
        dependents = sorted(blocked - unbudgeted)
        note = (
            f"{ring} probes not classified: {sorted(unbudgeted)} have a diverged "
            f"float leaf with no R2a budget (see R2a's notes)"
        )
        if dependents:
            note += (
                f"; {dependents} depend on them, so their class against an "
                f"unresolved predecessor severity is undecidable"
            )
        note += ". Every other probe is classified."
        notes = (note,)

    return probes, notes


def r1_target_isa(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    budgets: Mapping[str, float] | None = None,
    probe_deps: Mapping[str, tuple[str, ...]] | None = None,
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    input_class: str = "nominal",
    in_contract: bool = True,
) -> RingResult:
    """R1: target CPU, ``native`` vs ``native-portable``. Isolates ISA-dependent codegen.

    Has exactly two executable legs (``wasm32`` is compiled-only, section 3).
    Asserts its own precondition before interpreting anything: on a
    non-x86-64 host, LLVM warns-and-ignores the ``x86-64-v2`` flag and
    ``native-portable`` silently falls back to host-tuned codegen
    (``targets.py:155-165``), which would read as "no ISA divergence" for the
    wrong reason. Reading back real ``cpu_features`` (as
    ``test_parity_multi_size.py`` already does) is what catches that -- and a
    bare "the two strings differ" check is not enough on its own (finding 1,
    260914 code review round 5): the host-tuned fallback ALSO produces a
    differing string, so this ring only trusts a difference once
    ``native-portable`` positively declares the x86-64-v2 set
    (``_declares_x86_64_v2``).

    ``passed`` reports whether this rung completed a structural comparison
    over comparable leaves -- it never reflects whether the two legs' *values*
    agree, and it never consults ``probes``. Concretely: ``passed=False`` when
    the ``cpu_features`` precondition fails (the ring refuses to interpret its
    legs at all, per AC-16) or when an interpreted leaf comparison itself
    FAILED (a shape/dtype mismatch between ``native`` and ``native-portable``,
    consistent with R2a/R2b -- round 6 finding 4, 260914 code review); it is
    still ``True`` when the legs were successfully compared and every leaf
    diverges arbitrarily in *value* -- including a flipped integer index. That
    value-level divergence is reported per probe in ``probes`` (revision 6,
    spec AC-11, backlog #5210), not folded into ``passed``.

    Args:
        fn: Per-element function, run exactly as given (AC-11).
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to execute both legs with.
        budgets: Per-leaf budgets from R2a, if available -- advisory only
            (section 3.2: a fusion-sensitivity budget is not a bound on
            ISA-dependent codegen). Included in ``notes`` where a matching
            key exists, and consulted when classifying a diverged float leaf
            in ``probes``: a probe with such a leaf and no matching budget,
            and any probe that transitively depends on it, is omitted from
            ``probes`` and named in a note -- every other declared probe is
            still classified (260915 code review; see
            ``_classify_rung_probes``).
        probe_deps: The declared probe DAG (section 5.3), or ``None`` (the
            default) to skip classification entirely -- ``probes`` is then
            ``()``, unchanged from before revision 6. When given, every
            declared name is classified from the SAME native-vs-portable
            leaves this rung already computed for ``passed`` -- classifying
            never changes what is run or compared. See
            ``_classify_rung_probes``.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label.
        in_contract: AC-17 label.

    Returns:
        A ``RingResult`` for ``"R1"``. ``probes`` holds one ``ProbeReport``
        per classifiable declared name when classification ran, in
        declaration order -- see ``passed``'s note above and
        ``_classify_rung_probes`` for when classification is skipped
        entirely (no ``probe_deps``, or a name absent from this rung's own
        leaves) versus per-probe (a diverged float leaf with no matching
        budget omits that probe and its dependents, not every probe).
    """
    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(callable_))(*abstract_inputs)
    mlir = exported.mlir_module()

    native = compile_for_target(mlir, NATIVE)
    portable = compile_for_target(mlir, NATIVE_PORTABLE)
    native_features = _read_cpu_features(native.path)
    portable_features = _read_cpu_features(portable.path)

    if native_features == portable_features:
        return RingResult(
            ring="R1",
            passed=False,
            input_class=input_class,
            in_contract=in_contract,
            probes=(),
            notes=(
                "R1 refuses to interpret: native and native-portable report "
                "identical cpu_features. On a non-x86-64 build host LLVM "
                "warns-and-ignores the x86-64-v2 flag and silently falls back "
                "to host-tuned codegen (targets.py:155-165); reporting 'no ISA "
                "divergence' here would be for the wrong reason.",
                f"cpu_features={native_features!r}",
            ),
        )

    if not _declares_x86_64_v2(portable_features):
        return RingResult(
            ring="R1",
            passed=False,
            input_class=input_class,
            in_contract=in_contract,
            probes=(),
            notes=(
                "R1 refuses to interpret: native and native-portable "
                "cpu_features differ, but native-portable does not "
                "positively declare the x86-64-v2 feature set. On a "
                "non-x86-64 build host LLVM silently ignores the x86-64-v2 "
                "target-feature request and native-portable falls back to "
                "host-tuned codegen (targets.py:155-165) -- which reads as "
                "'ISA divergence' for the wrong reason (finding 1, 260914 "
                "code review round 5).",
                f"native cpu_features={native_features!r}",
                f"native-portable cpu_features={portable_features!r}",
            ),
        )

    native_out = _as_flat_sequence(run_native_vmfb(native.path, *concrete_inputs))
    portable_out = _as_flat_sequence(run_native_vmfb(portable.path, *concrete_inputs))
    native_structured = jax.tree_util.tree_unflatten(exported.out_tree, list(native_out))
    portable_structured = jax.tree_util.tree_unflatten(exported.out_tree, list(portable_out))

    leaves_by_name = _compare_by_name(native_structured, portable_structured)
    probes, probe_notes = _classify_rung_probes("R1", leaves_by_name, probe_deps, budgets)
    notes = (
        f"native cpu_features={native_features!r}",
        f"native-portable cpu_features={portable_features!r}",
        *_format_named_leaf_notes(leaves_by_name, budgets),
        *probe_notes,
    )
    return RingResult(
        ring="R1",
        # A structurally FAILED leaf (shape/dtype mismatch between native
        # and native-portable) must fail R1 the same way R2a already fails
        # on the identical condition -- not report passed=True with the
        # only sign being a "FAILED" string buried in `notes` (round 6
        # finding 4, 260914 code review). `passed` never consults `probes`
        # (see this function's own docstring) -- a flipped index is visible
        # only in `probes`, never in `passed`.
        passed=not _any_leaf_failed(leaves_by_name),
        input_class=input_class,
        in_contract=in_contract,
        probes=probes,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# T5 -- R2a / R2b + section 3.2 budget derivation
# ---------------------------------------------------------------------------


def _any_leaf_failed(leaves_by_name: Mapping[str, tuple[LeafDivergence, ...]]) -> bool:
    """Whether any leaf across any name FAILED (a shape/dtype mismatch).

    A structural failure is knowable right here, not a magnitude any ring
    judges -- shared by R1 and R2b so both report ``passed=False`` over a
    FAILED leaf the same way R2a already does, rather than a caller having
    to notice a "FAILED" string buried in ``notes`` (round 6 finding 4,
    260914 code review).
    """
    return any(leaf.failed for leaves in leaves_by_name.values() for leaf in leaves)


def _coerce_like_parity(value: Any) -> Any:  # noqa: ANN401
    """Coerce a pytree exactly as ``parity.compare`` coerces its reference
    argument, applied leaf by leaf.

    ``parity.compare`` normalizes via ``np.asarray(jnp.asarray(expected))``
    -- under this build's disabled x64, ``jnp.asarray`` silently narrows a
    float64 leaf to float32 *before* any comparison happens.
    ``compare_pytree``'s own leaf coercion (``np.asarray(expected_leaf)``,
    in ``divergence._compare_leaf``) does not go through ``jnp`` at all, so
    it never observes that narrowing -- a caller's float64 NumPy oracle
    reads as a dtype mismatch on every leaf, FAILING R2a entirely (round 6
    finding 2, 260914 code review). Applying ``parity.compare``'s own
    formula here, per leaf, is what makes ``r2a_fusion``'s ``eager_fn``
    docstring claim -- "same contract as ``export_pipeline``'s
    ``reference_fn``" -- literally true rather than aspirational.
    """
    return jax.tree_util.tree_map(lambda leaf: np.asarray(jnp.asarray(leaf)), value)


def _resolve_oracle_by_name(
    eager_result: Any,
    jit_by_name: Mapping[str, Any],
    primary_names: frozenset[str],
) -> dict[str, Any]:
    """Normalize ``eager_fn``'s return value onto ``fn``'s own top-level
    output names -- the one boundary every ``r2a_fusion`` oracle case routes
    through (round 6, 260914 code review).

    Three prior rounds each patched a single oracle-shape case (a bare value
    against a named ``fn``, round 5; a named-but-mismatched value, round 6
    finding 1; a float64 NumPy value, round 6 finding 2) and each patch left
    the next case open -- the same shape as the ULP metric bugs, which only
    converged once point fixes were replaced by one invariant. This is that
    invariant for the oracle: every case funnels through here exactly once,
    and every returned name is guaranteed to be one ``fn`` actually
    produced, or this function raises -- never a silent drop.

    Rules (mirrors ``r2a_fusion``'s ``eager_fn``/``primary_names``
    docstring):

    - A bare (unnamed) oracle against a NAMED ``fn`` output is mapped onto
      the single entry in ``primary_names`` -- ambiguous (raises) unless
      exactly one name is given.
    - A bare oracle against a bare ``fn`` output is used as-is (both use the
      single ``""`` entry).
    - Every other oracle name is expected to be a SUBSET of ``fn``'s own
      top-level names: the oracle legitimately covers only the model's true
      output(s), never pipeline-internal probes -- ``r2a_fusion``'s own
      ``disable_jit`` fallback fills every name the oracle omits.
    - Any oracle name that is NOT one of ``fn``'s outputs is a caller error,
      not a name to silently discard: raises, naming the unmatched name(s)
      and ``fn``'s actual output names, so the mismatch is obvious instead
      of the oracle quietly measuring nothing against itself (round 6
      finding 1).

    Every value in the returned mapping is coerced via
    ``_coerce_like_parity`` before ``compare_pytree`` ever sees it (round 6
    finding 2).

    Raises:
        ValueError: A bare oracle can't be matched to exactly one primary
            name, or an oracle name is not among ``fn``'s own output names.
    """
    oracle_by_name = _top_level_items(eager_result)
    oracle_is_bare = set(oracle_by_name) == {""}
    jit_is_named = set(jit_by_name) != {""}

    if oracle_is_bare and jit_is_named:
        if len(primary_names) != 1:
            msg = (
                f"eager_fn returned a bare (unnamed) value, but fn's output "
                f"is named ({sorted(jit_by_name)}) and primary_names does "
                f"not identify exactly one output to map it to (got "
                f"{sorted(primary_names)}); the independent oracle cannot "
                f"be matched to fn's outputs without this."
            )
            raise ValueError(msg)
        (primary_name,) = primary_names
        oracle_by_name = {primary_name: oracle_by_name[""]}

    unmatched = oracle_by_name.keys() - jit_by_name.keys()
    if unmatched:
        msg = (
            f"eager_fn returned name(s) {sorted(unmatched)} that are not "
            f"among fn's own top-level output names {sorted(jit_by_name)}; "
            f"the independent oracle does not correspond to fn's outputs, "
            f"so it cannot be silently dropped -- that would measure the "
            f"mismatched name only against fn's own jit'd output (never an "
            f"independent oracle) and report a spurious pass."
        )
        raise ValueError(msg)

    return {name: _coerce_like_parity(value) for name, value in oracle_by_name.items()}


def r2a_fusion(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    eager_fn: Callable[[Sequence[Any]], Any],
    primary_names: frozenset[str] = frozenset(),
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    input_class: str = "nominal",
    in_contract: bool = True,
) -> tuple[RingResult, dict[str, float]]:
    """R2a: eager vs ``jit`` -- same compiler, different graph. Isolates fusion sensitivity.

    Runs **before any budget exists** and is therefore reported, never judged
    (section 6.0): gating it on a budget it is itself measuring would be
    circular. Its job is to *produce* the per-leaf budget every later rung
    consumes -- ``m_leaf`` (this leaf's measured ``max_ulp_diff``) feeds
    ``divergence.budget_leaf`` directly.

    Measured **per top-level name** (not once over the whole output), so each
    leaf's ``max_ulp_diff`` is looked up and re-keyed via
    ``divergence.budget_key(name, leaf.path)`` -- the convention
    ``classify_probes`` requires of its ``budgets`` argument.

    Args:
        fn: Per-element function, run exactly as given (AC-11). If ``fn``
            carries probes, they are measured here too -- R2a produces budgets
            for every top-level name, not only the model's true output(s).
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to run both legs with.
        eager_fn: An independently-computed oracle over ``concrete_inputs`` --
            the model applied directly, not through the composed/jitted
            callable under test. Same contract as ``export_pipeline``'s
            ``reference_fn`` (``xtrax.export.parity``) -- including
            ``parity.compare``'s own ``np.asarray(jnp.asarray(...))``
            coercion, applied here via ``_coerce_like_parity`` (round 6
            finding 2, 260914 code review), so a float64 NumPy oracle is not
            read as a dtype mismatch on every leaf. Normally covers only the
            model's true output(s), not pipeline-internal probes -- any
            top-level name ``jit_result`` has but ``eager_fn`` omits is
            measured here too, using ``fn`` itself run eagerly (un-jitted),
            since no separate independent oracle is possible for a probe.
            Every other shape (bare vs. named, named-but-mismatched, ...)
            is resolved by ``_resolve_oracle_by_name`` -- see its docstring
            for the full rule set. In particular, an oracle name that is
            not one of ``fn``'s own top-level output names now RAISES
            rather than being silently discarded as "present on only one
            side" (round 6 finding 1, 260914 code review) -- that used to
            leave the mismatched name completely unmeasured while every
            real name fell back to the jit-vs-itself leg and reported a
            spurious pass.
        primary_names: The top-level name(s) of ``fn``'s output that a bare
            (unnamed) ``eager_fn`` return value corresponds to -- normally
            ``probe_deps``'s DAG sinks (see ``_sink_names``), which is what
            ``run_ladder`` passes. AC-11 forbids this rung from taking
            ``probe_deps`` directly, so the caller resolves the mapping.
            Unused when ``eager_fn``'s return value is already named, or
            when ``fn``'s own output has no named top level either. If
            ``eager_fn`` returns bare AND ``fn``'s output is named, exactly
            one name must be given here -- more than one is ambiguous (which
            name the bare value belongs to cannot be inferred) and raises.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label. ``m_leaf`` is per ``(model,
            input_class)`` (section 3.2) -- re-derive under every
            stratification, never reuse across classes.
        in_contract: AC-17 label.

    Returns:
        A ``(RingResult, budgets)`` pair. ``budgets`` maps
        ``budget_key(name, leaf.path)`` to ``budget_leaf(m_leaf)`` for every
        float leaf found; integer/bool leaves are absent (section 3.2:
        calibration applies to float leaves only). ``passed`` is ``False``
        when any leaf comparison itself FAILED (a shape/dtype mismatch
        between ``eager_fn`` and ``fn``'s jit'd output; finding 3, 260914
        code review round 5) or when a leaf's measured divergence is
        non-finite (round 6 finding 3, 260914 code review -- an unbounded
        eager-vs-jit divergence is a real divergence, not fusion
        sensitivity to calibrate a budget against, and ``budget_leaf`` would
        otherwise turn it into an infinite tolerance). Both are structural
        problems knowable right here, not a magnitude this rung judges; it
        is still ``True`` whenever every leaf compared successfully with a
        finite measurement, even for a leaf whose divergence a later rung's
        calibrated budget would find excessive -- R2a itself never judges a
        float leaf's magnitude (section 6.0).
    """
    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    eager_result = eager_fn(concrete_inputs)
    jit_result = jax.jit(callable_)(*concrete_inputs)

    jit_by_name = _top_level_items(jit_result)
    # Every oracle shape (bare, subset-of-names, exact-names, a mismatched
    # name, float64 NumPy, ...) routes through this one boundary -- see its
    # own docstring (round 6, 260914 code review: three prior rounds each
    # patched a single case here and each patch left the next one open).
    # Raises rather than silently dropping an unmatched oracle name.
    eager_by_name = _resolve_oracle_by_name(eager_result, jit_by_name, primary_names)

    # `eager_fn` is documented as "the model applied directly" (section 5.1)
    # -- a real caller's oracle normally covers only the model's own true
    # output(s), not pipeline-internal probes. A probe absent from it must
    # still get a budget: no separate "independent oracle" is possible for a
    # pipeline intermediate, so measure it the same way every other name
    # here is measured -- eager `fn` (via `callable_`, un-jitted) against
    # `jax.jit(callable_)`, exactly the eager-vs-jit fusion sensitivity R2a
    # exists to produce a budget for. This closes the coverage gap up front,
    # rather than silently skipping the name here and letting it surface,
    # many rungs later, as a `MissingBudgetError` deep inside R3 -- after
    # every compile has already run (finding 5, 260914 code review).
    #
    # A plain un-jitted call is NOT actually eager for a Scan (or
    # Vmap-over-Scan) composition: `lax.scan` always compiles its own body
    # into one XLA computation regardless of an enclosing `jax.jit`, and
    # worse, when called with the same abstract shapes `jax.jit(callable_)`
    # was just traced with (immediately above), JAX's own trace cache serves
    # the identical jaxpr without re-invoking `fn` at all -- verified
    # empirically (this jax version) with a Python-level call counter: 0
    # additional Python calls, not even 1. Either way this "eager" leg would
    # silently become jit-vs-jit (or literally the same artifact vs itself),
    # so every probe measured through it reads m_leaf == 0 and gets the bare
    # ULP floor budget instead of one calibrated against real fusion
    # sensitivity -- read later as false BEYOND_BUDGET noise (finding 3,
    # 260914 code review round 4). `jax.disable_jit()` bypasses both the
    # compilation and the trace cache and forces `lax.scan` to execute its
    # body once per step in the Python interpreter -- verified empirically,
    # do not rely on documentation memory, jax versions have changed this
    # before.
    missing_from_eager_fn = jit_by_name.keys() - eager_by_name.keys()
    if missing_from_eager_fn:
        with jax.disable_jit():
            raw_eager_by_name = _top_level_items(callable_(*concrete_inputs))
        for name in missing_from_eager_fn:
            eager_by_name[name] = raw_eager_by_name[name]

    names = sorted(eager_by_name.keys() | jit_by_name.keys())

    budgets: dict[str, float] = {}
    notes: list[str] = []
    any_leaf_failed = False
    for name in names:
        if name not in eager_by_name or name not in jit_by_name:
            notes.append(f"{name!r}: present on only one side (eager vs jit) -- skipped")
            continue
        leaves = compare_pytree(eager_by_name[name], jit_by_name[name])
        for leaf in leaves:
            key = budget_key(name, leaf.path)
            if leaf.failed:
                # A shape/dtype mismatch between the eager oracle and fn's
                # own jit'd output is knowable right here -- do not let it
                # travel forward unbudgeted to R3, where it would surface
                # many rungs (and several IREE compiles) later as a bare
                # KeyError/MissingBudgetError, discarding everything in
                # between (finding 3, 260914 code review round 5).
                notes.append(f"{key}: FAILED ({leaf.message})")
                any_leaf_failed = True
                continue
            if leaf.dtype_class == "float":
                m_leaf = float(leaf.metrics.get("max_ulp_diff", 0.0))
                if not math.isfinite(m_leaf):
                    # A non-finite `m_leaf` (e.g. `+finfo.max` eager vs
                    # `-finfo.max` jit) is a genuine unbounded divergence,
                    # not fusion sensitivity to calibrate a budget against
                    # (round 6 finding 3, 260914 code review): `budget_leaf`
                    # would otherwise return `inf`, granting this leaf
                    # unconditional tolerance forever after. Recorded as a
                    # failed measurement -- same treatment as a shape/dtype
                    # FAILED leaf just above, and for the same reason: R2a
                    # cannot produce a complete, trustworthy budget map for
                    # this input class, so it must not report passed=True
                    # with an incomplete one.
                    notes.append(
                        f"{key}: m_leaf={m_leaf!r} ULP is non-finite -- no "
                        f"bounded budget can be derived; recorded as a "
                        f"failed measurement, not an infinite tolerance"
                    )
                    any_leaf_failed = True
                    continue
                budget = budget_leaf(m_leaf)
                budgets[key] = budget
                notes.append(f"{key}: m_leaf={m_leaf:.6g} ULP -> budget={budget:.6g} ULP")
            else:
                notes.append(
                    f"{key} [{leaf.dtype_class}]: no calibrated budget "
                    f"(exact-match criterion, section 3.2)"
                )

    result = RingResult(
        ring="R2a",
        passed=not any_leaf_failed,
        input_class=input_class,
        in_contract=in_contract,
        probes=(),
        notes=tuple(notes),
    )
    return result, budgets


def r2b_lowering(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    budgets: Mapping[str, float],
    probe_deps: Mapping[str, tuple[str, ...]] | None = None,
    target: Target = NATIVE,
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    input_class: str = "nominal",
    in_contract: bool = True,
) -> RingResult:
    """R2b: ``jit`` XLA vs IREE -- same graph, different backend. Isolates lowering fidelity.

    ``budgets`` (from R2a) is **advisory here, not authoritative** (section
    3.2): a bound on fusion sensitivity is not a bound on lowering fidelity.
    R2a and R2b are always reported separately for exactly this reason. That
    same advisory relationship is what "advisory on R2b" (spec section 3.2)
    means operationally when ``probe_deps`` is given: a probe's class is
    computed against R2a's budgets and reported in ``probes``, not folded
    into this rung's own authoritative ``passed`` (see below).

    ``passed`` reports whether this rung completed a structural comparison
    over comparable leaves -- it never reflects whether the two legs' *values*
    agree, and it never consults ``probes``. Concretely: ``passed`` is
    ``False`` only when a leaf comparison itself FAILED (a shape/dtype
    mismatch between the ``jit`` and IREE legs, consistent with R2a/R1 --
    round 6 finding 4, 260914 code review); it is still ``True`` when every
    leaf compared successfully and diverges arbitrarily in *value* --
    including a flipped integer index. That value-level divergence is
    reported per probe in ``probes`` (revision 6, spec AC-11, backlog #5210),
    not folded into ``passed``.

    Args:
        fn: Per-element function, run exactly as given (AC-11).
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to run both legs with.
        budgets: Per-leaf budgets from R2a for this ``(model, input_class)``.
            A probe with a diverged float leaf and no matching budget, and
            any probe that transitively depends on it, is omitted from
            ``probes`` and named in a note -- every other declared probe is
            still classified (260915 code review; see
            ``_classify_rung_probes``).
        probe_deps: The declared probe DAG (section 5.3), or ``None`` (the
            default) to skip classification entirely -- ``probes`` is then
            ``()``, unchanged from before revision 6. When given, every
            declared name is classified from the SAME jit-vs-IREE leaves
            this rung already computed for ``passed`` -- classifying never
            changes what is run or compared. See ``_classify_rung_probes``.
        target: Compilation target for the IREE leg.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label.
        in_contract: AC-17 label.

    Returns:
        A ``RingResult`` for ``"R2b"``. ``probes`` holds one ``ProbeReport``
        per classifiable declared name when classification ran, in
        declaration order -- see ``passed``'s note above and
        ``_classify_rung_probes`` for when classification is skipped
        entirely (no ``probe_deps``, or a name absent from this rung's own
        leaves) versus per-probe (a diverged float leaf with no matching
        budget omits that probe and its dependents, not every probe).
    """
    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(callable_))(*abstract_inputs)
    jit_result = jax.jit(callable_)(*concrete_inputs)

    compiled = compile_for_target(exported.mlir_module(), target)
    iree_out = _as_flat_sequence(run_native_vmfb(compiled.path, *concrete_inputs))
    iree_structured = jax.tree_util.tree_unflatten(exported.out_tree, list(iree_out))

    leaves_by_name = _compare_by_name(jit_result, iree_structured)
    probes, probe_notes = _classify_rung_probes("R2b", leaves_by_name, probe_deps, budgets)
    return RingResult(
        ring="R2b",
        # See R1's identical comment (round 6 finding 4): a structurally
        # FAILED leaf must fail this ring, not just appear as a "FAILED"
        # string buried in `notes`. `passed` never consults `probes` (see
        # this function's own docstring) -- a flipped index is visible only
        # in `probes`, never in `passed`.
        passed=not _any_leaf_failed(leaves_by_name),
        input_class=input_class,
        in_contract=in_contract,
        probes=probes,
        notes=(*_format_named_leaf_notes(leaves_by_name, budgets), *probe_notes),
    )


# ---------------------------------------------------------------------------
# T6 -- R3, with the section 5.4 two-sided fidelity precondition
# ---------------------------------------------------------------------------


def _all_probe_names(probe_deps: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """Every declared probe name -- a ``probe_deps`` key OR a predecessor value.

    A **source** probe with no predecessors of its own -- declared purely as
    someone else's predecessor, e.g. spec SS6.1's own example,
    ``{"rbf": ("neighbor_indices",), ...}`` -- is never a dict key.
    Restricting "every probe name" to ``probe_deps.keys()`` therefore silently
    drops it from the probe set entirely. Every place in this module that
    needs "every declared probe name" -- R3's strip set, its per-probe
    classification input, and ``_sink_names``'s DAG-sink computation -- derives
    it from here once, rather than re-deriving ``.keys()`` ad hoc (finding 2,
    260914 code review).
    """
    preds = {p for preds in probe_deps.values() for p in preds}
    return frozenset(probe_deps) | preds


def _probe_iteration_order(probe_deps: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """A deterministic order over every declared probe name (see ``_all_probe_names``).

    ``_all_probe_names`` returns a ``frozenset``, whose iteration order
    depends on Python's per-process string hash randomization
    (``PYTHONHASHSEED``) -- building a dict (or anything order-sensitive) by
    iterating it directly makes the result differ run to run, for a tool
    whose first rung (R0) is itself a determinism gate (finding 5, 260914
    code review round 3). Declaration order (``probe_deps``'s own key order,
    which the caller wrote) is used first; any probe that appears only as
    somebody else's predecessor -- never its own ``probe_deps`` key, e.g. a
    DAG source per spec SS6.1 -- is appended afterward in sorted order.
    """
    declared = tuple(probe_deps)
    remaining = sorted(_all_probe_names(probe_deps) - frozenset(declared))
    return declared + tuple(remaining)


def _sink_names(probe_deps: Mapping[str, tuple[str, ...]]) -> frozenset[str]:
    """DAG sinks: declared probe names that are never anyone else's predecessor.

    This module's derivation of "primary" leaves for the section-5.4 fidelity
    check -- the model's own true output(s) (e.g. aminx's ``final``, per the
    section-5.3 DAG diagram), as opposed to intermediates declared purely to
    be inspected. Derived over the full probe-name union (``_all_probe_names``),
    not just ``probe_deps`` keys, so a source probe declared only as a
    predecessor is correctly excluded here whenever it has a successor (it is
    only a sink if nothing depends on it).
    """
    all_preds = {p for preds in probe_deps.values() for p in preds}
    return _all_probe_names(probe_deps) - all_preds


def _probe_deps_cycle(probe_deps: Mapping[str, tuple[str, ...]]) -> str | None:
    """DFS cycle detection over ``probe_deps``'s declared edges.

    ``divergence.py``'s own cycle check (inside its private
    ``_topological_order``, used by ``classify_probes``) only runs at the
    very end of R3, after R0/R2a/R1/R2b have already compiled and run for
    every input class -- so a genuinely cyclic declaration used to be caught
    only there, discarding every result already produced (finding 4, 260914
    code review round 5). This mirrors that same check so it can run in
    preflight instead, before any rung.

    Note that a plain "no sink node" check (``_sink_names`` returning empty)
    does NOT subsume this: a cycle with an extra, unrelated sink node (e.g.
    ``a -> b -> c -> a`` plus ``sink -> a``) has a perfectly valid-looking
    non-empty sink set, yet is still an invalid DAG.

    Returns:
        The name of a probe involved in a cycle, or ``None`` if the
        declared edges are acyclic.
    """
    visited: set[str] = set()
    in_progress: set[str] = set()
    found: list[str] = []

    def _visit(name: str) -> bool:
        if name in visited:
            return False
        if name in in_progress:
            found.append(name)
            return True
        in_progress.add(name)
        for pred in probe_deps.get(name, ()):
            if _visit(pred):
                return True
        in_progress.discard(name)
        visited.add(name)
        return False

    for name in _all_probe_names(probe_deps):
        if _visit(name):
            return found[0]
    return None


def _fidelity_holds(
    leaves_by_name: Mapping[str, tuple[LeafDivergence, ...]], budgets: Mapping[str, float]
) -> tuple[bool, tuple[str, ...]]:
    """Section 5.4 items 1-2: full metric set over every primary leaf, both directions.

    One symmetric comparison of primary leaves catches both suppression and
    creation of a divergence by instrumentation (revision-5 clarifying note on
    "two-sided") -- there is exactly one comparison here, not two
    direction-discriminating code paths.
    """
    problems: list[str] = []
    for name, leaves in leaves_by_name.items():
        for leaf in leaves:
            key = budget_key(name, leaf.path)
            if leaf.failed:
                problems.append(f"{key}: FAILED ({leaf.message})")
                continue
            if leaf.dtype_class == "float":
                ulp = float(leaf.metrics.get("max_ulp_diff", 0.0))
                nonfinite = leaf.metrics.get("n_nonfinite_mismatch", 0.0)
                if key not in budgets:
                    if ulp == 0.0 and not nonfinite:
                        continue  # bit-identical; no budget needed to know that.
                    msg = (
                        f"primary leaf {key!r} has no calibrated budget from R2a; "
                        f"cannot evaluate the R3 fidelity precondition for it."
                    )
                    raise KeyError(msg)
                budget = budgets[key]
                if ulp > budget or nonfinite:
                    problems.append(f"{key}: max_ulp_diff={ulp:.6g} exceeds budget={budget:.6g}")
            elif leaf.dtype_class == "integer":
                if float(leaf.metrics.get("exact_match_fraction", 0.0)) < 1.0:
                    problems.append(f"{key}: integer leaf not an exact-set match")
            elif leaf.dtype_class == "bool":
                if int(leaf.metrics.get("hamming_distance", 1)) != 0:
                    problems.append(f"{key}: bool leaf not an exact-set match")
    return (not problems), tuple(problems)


def r3_probe(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    probe_deps: Mapping[str, tuple[str, ...]],
    budgets: Mapping[str, float],
    *,
    r0_result: RingResult,
    target: Target = NATIVE,
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    input_class: str = "nominal",
    in_contract: bool = True,
) -> RingResult:
    """R3: cut depth -- named intermediates as extra outputs. Edits program outputs.

    Runs **last** among the rungs (section 3.1): it is uninterpretable until
    the section-5.4 fidelity precondition passes, and that precondition is
    itself uninterpretable if R0 found nondeterminism -- exact equality would
    then be unachievable by construction, so R3 refuses outright when
    ``r0_result.passed`` is False, without attempting anything (section 5.4
    item 2).

    On a fidelity failure this reports ``INSTRUMENTATION_CHANGED_RESULT`` and
    emits **no** probe map (``probes=()``) -- never "no divergence found",
    since a vanishing divergence under instrumentation is itself the finding.

    Args:
        fn: The per-element function, returning primary output(s) and probes
            together as ordinary named pytree entries (section 5.1). This
            rung derives its own "uninstrumented" (primary-only) view
            internally, using ``probe_deps``'s DAG sinks -- see
            ``_sink_names``.
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to run every leg with.
        probe_deps: The declared probe DAG (section 5.3); every key is
            classified. The keys that are DAG sinks (nobody's predecessor)
            are treated as the model's primary output(s) for the section-5.4
            fidelity check.
        budgets: Per-leaf budgets from R2a, keyed by ``budget_key``.
        r0_result: R0's result for this same input class; gates R3 outright.
        target: Compilation target.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label.
        in_contract: AC-17 label.

    Returns:
        A ``RingResult`` for ``"R3"``. On success, ``probes`` holds one
        ``ProbeReport`` per entry in ``probe_deps``, from ``classify_probes``.

    Raises:
        ValueError: ``probe_deps`` has no sink node (every key is also
            declared as someone else's predecessor), so no primary output can
            be derived.
    """
    if not r0_result.passed:
        return RingResult(
            ring="R3",
            passed=False,
            input_class=input_class,
            in_contract=in_contract,
            probes=(),
            notes=(
                "R3 refuses to run: R0 found nondeterminism for this input "
                "class, so exact equality is unachievable by construction "
                "(section 5.4).",
            ),
        )

    sinks = _sink_names(probe_deps)
    if not sinks:
        msg = (
            "probe_deps has no sink node (every key also appears as someone "
            "else's predecessor); cannot derive a primary output for the "
            "section-5.4 fidelity check."
        )
        raise ValueError(msg)
    all_probes = _all_probe_names(probe_deps)
    strip_names = all_probes - sinks

    # Instrumented artifact: fn as given -- primary output(s) plus every
    # declared probe. Composed ONCE; the primary (uninstrumented) view below
    # is derived from this same composed callable, not from a second
    # composition of a pre-stripped `fn`.
    full_callable = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)

    # Uninstrumented artifact: only the sink (true primary) output(s). The
    # strip is applied to the COMPOSED callable's own output, never to `fn`
    # before composition -- for a Scan axis (including the certified
    # Vmap-over-Scan shape), `fn` is the per-step TRANSITION returning
    # `(carry, y)`, not a plain per-element function. Stripping `fn`'s return
    # value pre-composition turns that 2-tuple into a one-key dict, which the
    # composer's own internal transition wrapper then fails to unpack as
    # `carry, y = fn(carry, x)` -- at trace time, before IREE ever runs
    # (finding 1, 260914 code review round 3). The probe names live inside
    # the composed callable's output (`y`, stacked across scan steps), which
    # only exists after `build_traceable_callable` has already folded the
    # scan -- so stripping must happen after composition, on `full_callable`
    # itself.
    primary_callable = _primary_only(full_callable, strip_names)
    primary_exported = jax.export.export(jax.jit(primary_callable))(*abstract_inputs)
    primary_compiled = compile_for_target(primary_exported.mlir_module(), target)
    primary_flat = _as_flat_sequence(run_native_vmfb(primary_compiled.path, *concrete_inputs))
    primary_actual = jax.tree_util.tree_unflatten(primary_exported.out_tree, list(primary_flat))

    full_exported = jax.export.export(jax.jit(full_callable))(*abstract_inputs)
    full_compiled = compile_for_target(full_exported.mlir_module(), target)
    full_flat = _as_flat_sequence(run_native_vmfb(full_compiled.path, *concrete_inputs))
    full_by_name = recover_probe_names(full_exported.out_tree, full_flat)
    # Same filter as `primary_callable`/`primary_actual` above (`_primary_only`
    # keeps every name NOT in `strip_names`) -- filtering by `name in sinks`
    # here instead silently dropped any undeclared top-level output (one that
    # is neither a declared probe nor a sink) from this side only, while it
    # stayed on `primary_actual`'s side, so `_compare_by_name` raised a
    # top-level name mismatch after both artifacts had already compiled and
    # run (finding 2, 260914 code review round 3). Deriving the filter once
    # and applying it to both sides is what keeps them from drifting apart
    # again.
    instrumented_primary = {
        name: value for name, value in full_by_name.items() if name not in strip_names
    }

    fidelity_leaves_by_name = _compare_by_name(primary_actual, instrumented_primary)
    holds, problems = _fidelity_holds(fidelity_leaves_by_name, budgets)
    if not holds:
        return RingResult(
            ring="R3",
            passed=False,
            input_class=input_class,
            in_contract=in_contract,
            probes=(),
            notes=(
                "INSTRUMENTATION_CHANGED_RESULT: primary-leaf outputs diverge "
                "between the instrumented and uninstrumented artifacts beyond "
                "their calibrated budget; refusing to emit a probe map "
                "(section 5.4).",
                *problems,
            ),
        )

    # Probe classification: each declared name's IREE-executed value against
    # an independently-computed (in-process jit, not exported/compiled) oracle
    # for the same name -- the same "jit vs IREE" comparison R2b makes for the
    # whole output, applied per declared name.
    jit_full_result = jax.jit(full_callable)(*concrete_inputs)
    expected_by_name = _top_level_items(jit_full_result)

    probe_leaves: dict[str, tuple[LeafDivergence, ...]] = {}
    for name in _probe_iteration_order(probe_deps):
        if name not in full_by_name or name not in expected_by_name:
            msg = (
                f"probe {name!r} is declared in probe_deps but was not found in fn's output pytree."
            )
            raise KeyError(msg)
        probe_leaves[name] = compare_pytree(expected_by_name[name], full_by_name[name])

    reports = classify_probes(probe_leaves, probe_deps, budgets)

    resolution_notes: tuple[str, ...] = ()
    try:
        probe_result_paths = _probe_result_paths(full_exported.out_tree)
        resolution = probe_resolution(full_exported, probe_deps, probe_result_paths)
    except Exception as exc:  # noqa: BLE001 - diagnostic best-effort, never fatal to R3
        resolution_notes = (f"probe_resolution unavailable: {exc}",)
    else:
        resolution_notes = (
            *(f"slice_delta[{edge}]={n}" for edge, n in resolution.slice_delta.items()),
            f"unattributed_ops={resolution.unattributed_ops}",
            f"total_ops={resolution.total_ops}",
        )

    return RingResult(
        ring="R3",
        passed=True,
        input_class=input_class,
        in_contract=in_contract,
        probes=reports,
        notes=resolution_notes,
    )


# ---------------------------------------------------------------------------
# T8b -- run_ladder, the top-level orchestrator
# ---------------------------------------------------------------------------


def _unvalidatable_probe_deps(
    probe_deps: Mapping[str, tuple[str, ...]], probe_result_paths: Mapping[str, str]
) -> frozenset[str]:
    """Declared probe names with no single ``result_info`` string to validate against.

    A probe whose own value is a nested pytree (more than one leaf) has no
    single-array result path -- ``_probe_result_paths`` only resolves that
    common case (see its own docstring). This is a *structural* gap, not a
    violated dependency, so it is kept separate from the slice-subset check
    itself.

    Callers must first rule out a name that is not a top-level output name at
    ALL (see ``_missing_probe_names``) -- that is a different failure mode
    (a typo'd/stale ``probe_deps`` entry, never confirmed-nested) and must
    raise immediately rather than fall into this warn-and-skip bucket
    (finding 2, 260914 code review round 4).
    """
    all_names = frozenset(probe_deps) | {p for preds in probe_deps.values() for p in preds}
    return all_names - frozenset(probe_result_paths)


def _top_level_output_names(out_tree: Any) -> frozenset[str]:
    """The top-level names of ``fn``'s output pytree, from a traced ``out_tree``.

    Unlike ``_probe_result_paths`` (which only resolves a name whose OWN
    value is a single array), this reflects every top-level name regardless
    of what its value holds -- including a name whose value is itself a
    nested pytree, e.g. ``"pair": (a, b)`` -- so it can tell "declared name
    absent entirely" (a typo/stale entry) apart from "declared name present
    but nested" (see ``_unvalidatable_probe_deps``). A structure with no
    named top level (a bare array or plain tuple) is reported as the single
    implicit name ``""``, matching ``_top_level_items``'s own convention.
    """
    placeholder = jax.tree_util.tree_unflatten(out_tree, [0] * out_tree.num_leaves)
    return frozenset(_top_level_items(placeholder))


def _missing_probe_names(
    probe_deps: Mapping[str, tuple[str, ...]], top_level_names: frozenset[str]
) -> frozenset[str]:
    """Declared probe names that are not top-level names of ``fn``'s output at all.

    This is the typo/stale-entry case: a name that never appears in ``fn``'s
    output pytree under any form, as opposed to a name that appears but
    holds a nested pytree (``_unvalidatable_probe_deps``'s job). Conflating
    the two used to let a bad name reach ``run_ladder`` unrejected, since the
    generic "nested, skip and warn" path only warns; every rung then ran to
    completion before a plain ``KeyError`` in R3 lost every earlier result
    (finding 2, 260914 code review round 4).
    """
    return _all_probe_names(probe_deps) - top_level_names


def _default_validate(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    boundaries: Mapping[str, Any] | None,
    scan_init: Any,
    probe_deps: Mapping[str, tuple[str, ...]],
) -> None:
    """Trace ``fn`` and validate its declared probe DAG. Raises on a bad edge.

    Deliberately touches only ``jax.export`` (no IREE) -- ``validate_probe_deps``
    must refuse before any toolchain-backed rung runs at all (AC-20).

    A declared name that is not a top-level output name of ``fn`` AT ALL --
    a typo, a stale ``probe_deps`` entry, or ``probe_deps`` naming probes
    against an output with no named top level -- raises immediately
    (``ValueError``), before any nested-pytree handling below even runs
    (finding 2, 260914 code review round 4). This is distinct from, and
    checked before, the nested-pytree case:

    An edge touching a nested-pytree-valued probe (no single ``result_info``
    string -- see ``_unvalidatable_probe_deps``) is skipped here rather than
    turned into a hard refusal for the whole ladder: R3 itself already treats
    the identical gap as best-effort (it catches ``probe_resolution``'s
    equivalent ``ValueError`` and only downgrades to a note, never refusing).
    A silent skip would hide exactly the probes this validation exists to
    protect, so every skipped edge is surfaced via a visible ``UserWarning``
    naming both endpoints -- it is UNVALIDATED, not confirmed correct (finding
    3, 260914 code review round 3).

    Before any of the above -- before ``fn`` is even traced -- ``probe_deps``
    must have a well-formed DAG shape at all: at least one sink node (an
    empty declaration, or one where every name is also someone else's
    predecessor, has none) and no cycle. Both are structural properties of
    ``probe_deps`` alone, independent of ``fn``, and both used to be caught
    only deep inside R3 (a bare "no sink node" ``raise``, or
    ``classify_probes``' own topological sort) -- after every earlier rung
    had already compiled and run for every input class (finding 4, 260914
    code review round 5).
    """
    sinks = _sink_names(probe_deps)
    if not sinks:
        msg = (
            "probe_deps has no sink node (it is empty, or every declared "
            "name also appears as someone else's predecessor); cannot "
            "derive a primary output for the section-5.4 fidelity check."
        )
        raise ValueError(msg)

    cycle_name = _probe_deps_cycle(probe_deps)
    if cycle_name is not None:
        msg = (
            f"probe_deps contains a cycle involving {cycle_name!r}; cannot "
            f"derive a DAG order for probe classification."
        )
        raise ValueError(msg)

    full_callable = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(full_callable))(*abstract_inputs)
    probe_result_paths = _probe_result_paths(exported.out_tree)
    top_level_names = _top_level_output_names(exported.out_tree)

    missing = _missing_probe_names(probe_deps, top_level_names)
    if missing:
        if top_level_names == frozenset({""}):
            msg = (
                f"probe_deps declares probe(s) {sorted(missing)}, but fn's output has "
                f"no named top level (a bare array or plain tuple) -- probes require "
                f"named top-level outputs (a dict or NamedTuple)."
            )
        else:
            msg = (
                f"probe_deps declares probe(s) {sorted(missing)} that are not "
                f"top-level names in fn's output; available top-level name(s): "
                f"{sorted(top_level_names)}."
            )
        raise ValueError(msg)

    unresolved = _unvalidatable_probe_deps(probe_deps, probe_result_paths)
    if unresolved:
        skipped_edges = sorted(
            f"{pred}->{probe}"
            for probe, preds in probe_deps.items()
            for pred in preds
            if probe in unresolved or pred in unresolved
        )
        warnings.warn(
            f"probe_deps edge(s) {skipped_edges} could not be validated (AC-15 "
            f"slice-subset check): probe(s) {sorted(unresolved)} hold a nested "
            f"pytree with no single jax.result_info string. These edges are "
            f"UNVALIDATED, not confirmed correct.",
            stacklevel=2,
        )
        validatable_deps = {
            probe: tuple(pred for pred in preds if pred not in unresolved)
            for probe, preds in probe_deps.items()
            if probe not in unresolved
        }
    else:
        validatable_deps = dict(probe_deps)

    validate_probe_deps(exported, validatable_deps, probe_result_paths)


def run_ladder(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    probe_deps: Mapping[str, tuple[str, ...]],
    input_classes: Sequence[InputClassResult] = (),
    eager_fn: Callable[[Sequence[Any]], Any],
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    replays: int = 3,
    target: Target = NATIVE,
    validate_fn: Callable[..., None] = _default_validate,
    r0: Callable[..., RingResult] = r0_replay_gate,
    r1: Callable[..., RingResult] = r1_target_isa,
    r2a: Callable[..., tuple[RingResult, dict[str, float]]] = r2a_fusion,
    r2b: Callable[..., RingResult] = r2b_lowering,
    r3: Callable[..., RingResult] = r3_probe,
) -> tuple[RingResult, ...]:
    """Run the full comparison ladder: validate, then R0, then the non-editing rings, then R3.

    Calls ``validate_probe_deps`` (via ``validate_fn``) **first** and refuses
    -- by letting its exception propagate -- before executing anything if a
    declared ``probe_deps`` edge is rejected (AC-20). On a well-formed input,
    runs the rungs in section 3.1's order, per input class: R0 first (as the
    validity gate), then the non-editing rings (R2a, since it produces the
    budget every later rung consumes; then R1 and R2b, which consume it), then
    R3 last, since R3 is uninterpretable until R0 and the section-5.4
    precondition (checked inside R3 itself) both hold.

    If R0 fails for an input class, no other rung runs for that class (AC-16)
    -- R0's own ``RingResult`` is still returned, so the gate failure is
    visible in the output.

    **Two-layer failure handling (finding 1-6, 260914 code review round 5).**
    Three rounds of prior review each fixed a single instance of the same
    class of bug: a validation gap that let R0/R2a/R1/R2b compile and run to
    completion, only for something LATE (typically deep inside R3) to raise
    and discard every ``RingResult`` already produced. The root cause is
    structural, not any one gap: this function used to have no ``try``/
    ``except`` and no partial return, so ANY exception from ANY rung
    propagated and discarded everything. Two layers close the class, not
    just the latest instance of it:

    - **Layer 1 (preflight, before R0 for the class it applies to).** A
      *declaration* error -- something wrong with the arguments to this call
      itself, knowable without running ``fn`` on real data -- RAISES,
      unconditionally, before any rung or compile for the class it concerns.
      This function's own top raises for ``replays < 2`` (finding 5), once,
      before the per-class loop even starts (it does not depend on any
      class). ``validate_fn`` (``_default_validate`` by default) raises for
      an empty/sinkless/cyclic ``probe_deps`` (finding 4) and for a
      ``probe_deps`` name absent from ``fn``'s output (round 4 finding 2);
      it is called once **per input class**, against that class's own
      ``abstract_inputs`` (finding 5, round 6) -- not once, upfront, against
      this function's positional ``abstract_inputs`` -- because every rung
      traces and compiles at the class's own shapes, and a class whose
      length differs from the positional default is the bucket ladder's
      NORMAL case, not an edge case. These exceptions are never caught here
      -- they propagate unmodified, exactly as AC-20 requires, so a typo or
      a malformed declaration is never silently downgraded into a partial
      run.
    - **Layer 2 (per rung, inside the per-class loop).** A genuine RUNTIME
      failure -- one that could not have been known before actually
      executing a rung (a toolchain crash, an IREE compile failure, an
      unexpected shape at execution time) -- is caught individually around
      each of R0/R2a/R1/R2b/R3 and converted into a ``RingResult`` for that
      ring with ``passed=False`` and the exception's type and message in
      ``notes``, instead of propagating and discarding every result already
      produced for that (or an earlier) input class. Only ``Exception`` is
      caught -- never ``BaseException`` -- so ``KeyboardInterrupt``/
      ``SystemExit`` are never swallowed, and the notes always name the
      exception's type, so a bug in the ladder's own code (e.g. a bad
      keyword passed to a rung here) is still visible as a loud, identifiable
      failure, never a silent one.

    The two layers are deliberately NOT symmetric: Layer 1 must never be
    weakened into a Layer-2-style caught-and-recorded failure, or AC-20's
    "refuse before any toolchain work" guarantee is lost and a malformed
    declaration becomes a quiet partial run instead of a hard error. This
    function's own ``replays`` check runs once, entirely before the
    per-class loop. ``validate_fn`` runs once per class, INSIDE the
    per-class loop body -- but still strictly before that class's own
    ``r0``/``r2a``/``r1``/``r2b``/``r3`` calls, and outside every one of
    their ``try``/``except`` blocks, so it remains Layer 1 in effect: an
    exception from it still propagates unmodified out of this function,
    never downgraded into a ``RingResult``. The one behavioural
    consequence of moving it into the loop: if input classes run in
    sequence and an EARLIER class's rungs already executed before a LATER
    class's ``validate_fn`` call raises, this function still has no
    partial-return path for a Layer 1 failure, so the earlier class's
    results are discarded along with everything else -- a declaration
    error still aborts the whole call, just detected per class instead of
    once upfront for every class combined.

    Args:
        fn: The per-element function, run identically by every rung (section
            5.1: a probe is just an ordinary named output, there is no
            separate instrumented callable). R3 derives its own
            primary-only view internally via ``probe_deps``.
        plan: A BatchPlan.
        abstract_inputs: Default abstract inputs, used when ``input_classes``
            is empty (wrapped as a single ``nominal`` class) and for probe-DAG
            validation.
        concrete_inputs: Default concrete inputs, paired with
            ``abstract_inputs``.
        probe_deps: The declared probe DAG (section 5.3). Forwarded to R1 and
            R2b as classification metadata too (revision 6, spec AC-11,
            backlog #5210) -- populating their own ``probes`` from leaves
            they already compute, never changing what either runs -- as well
            as to R3, which additionally derives its primary-only view from
            it (see ``fn`` above).
        input_classes: Section 6.2 stratification. Defaults to a single
            ``nominal`` class built from ``abstract_inputs``/``concrete_inputs``.
        eager_fn: An independently-computed oracle over concrete inputs, for
            R2a. See ``r2a_fusion``.
        boundaries: Passed through to every rung.
        scan_init: Passed through to every rung.
        replays: R0's replay count.
        target: Compilation target used by R0-R3 (R1 additionally compiles
            ``NATIVE_PORTABLE``).
        validate_fn: Injection point for probe-DAG validation. Defaults to
            tracing ``fn`` and calling ``divergence.validate_probe_deps``.
        r0: Injection point for the R0 runner.
        r1: Injection point for the R1 runner.
        r2a: Injection point for the R2a runner.
        r2b: Injection point for the R2b runner.
        r3: Injection point for the R3 runner.

    Returns:
        A flat tuple of every ``RingResult`` produced, in execution order --
        including a Layer-2-captured failure's synthetic ``RingResult`` for
        the ring it happened in, and every ring's result produced before it.

    Raises:
        ValueError: ``replays < 2`` (Layer 1: determinism cannot be
            established from fewer than 2 replays; finding 5). Raised before
            ``validate_fn`` is even called.
        Whatever ``validate_fn`` raises when a declared probe-DAG edge fails
        the section-5.3 slice-subset check, or when ``probe_deps`` is
        structurally invalid (Layer 1) -- propagated unmodified, before any
        rung executes.
    """
    if replays < 2:
        msg = (
            f"replays must be >= 2 to detect nondeterminism; got {replays}. "
            f"Fewer than 2 replays trivially 'passes' the R0 gate having "
            f"tested nothing."
        )
        raise ValueError(msg)

    # R2a cannot take `probe_deps` directly (AC-11), so this is the only
    # place that can resolve a bare `eager_fn` return value onto `fn`'s
    # declared primary (sink) output -- see `r2a_fusion`'s own
    # `primary_names` docstring (finding 2, 260914 code review round 5).
    # Class-independent (the DAG shape does not vary by input class), so
    # computed once, like `validate_fn`'s check above.
    primary_names = _sink_names(probe_deps)

    classes = tuple(input_classes) or (
        InputClassResult(
            label="nominal",
            in_contract=True,
            abstract_inputs=tuple(abstract_inputs),
            concrete_inputs=tuple(concrete_inputs),
        ),
    )

    results: list[RingResult] = []
    for input_class in classes:
        ai, ci = input_class.abstract_inputs, input_class.concrete_inputs
        label = input_class.label
        in_contract = input_class.in_contract
        common = {
            "boundaries": boundaries,
            "scan_init": scan_init,
            "input_class": label,
            "in_contract": in_contract,
        }

        # Layer 1, per class (finding 5, 260914 code review round 6):
        # validate against THIS class's own `ai`, not the outer positional
        # `abstract_inputs` -- every rung below traces and compiles at `ai`,
        # so `ai` is the graph that must be certified. This call sits
        # directly in the loop body, before the `try` blocks below start --
        # it is NOT inside any Layer-2 try/except, so an exception here
        # still propagates unmodified and aborts the whole call (AC-20),
        # exactly as it did when it ran once before the loop. The one
        # observable difference from the old single upfront call: if an
        # EARLIER class's rungs already ran before a LATER class's
        # validation raises, that earlier class's results are still
        # discarded, because Layer 1 has no partial-return path -- a
        # declaration error still aborts the entire ladder, it is just
        # detected per class now instead of once for all classes combined.
        validate_fn(fn, plan, ai, boundaries, scan_init, probe_deps)

        # target applies to R0 exactly as it does to R2b/R3 (finding 4,
        # 260914 code review): R0 is the ladder's hard validity gate, and it
        # must gate the SAME artifact the other toolchain-backed rungs judge
        # -- passing it here, not just to R2b/R3, keeps that true.
        #
        # Layer 2 (see this function's own docstring): a genuine RUNTIME
        # failure from a rung is recorded as a failed `RingResult` for that
        # ring instead of propagating and discarding every result already
        # produced. `KeyboardInterrupt`/`SystemExit` are never caught, and
        # the notes always carry the exception's type and message, so a bug
        # in the ladder's own code is still loud, never silent.
        try:
            r0_result = r0(fn, plan, ai, ci, replays=replays, target=target, **common)
        except Exception as exc:  # noqa: BLE001 - Layer 2, see docstring
            r0_result = RingResult(
                ring="R0",
                passed=False,
                input_class=label,
                in_contract=in_contract,
                probes=(),
                notes=(f"R0 raised {type(exc).__name__}: {exc}",),
            )
        results.append(r0_result)
        if not r0_result.passed:
            continue

        # R2a has no `target`: it compares eager JAX against `jax.jit`, never
        # compiling through IREE at all, so no compilation target applies.
        try:
            r2a_result, budgets = r2a(
                fn, plan, ai, ci, eager_fn=eager_fn, primary_names=primary_names, **common
            )
        except Exception as exc:  # noqa: BLE001 - Layer 2, see docstring
            r2a_result = RingResult(
                ring="R2a",
                passed=False,
                input_class=label,
                in_contract=in_contract,
                probes=(),
                notes=(f"R2a raised {type(exc).__name__}: {exc}",),
            )
            budgets = {}
        results.append(r2a_result)

        # R1 deliberately has no `target` either: its own two legs are the
        # FIXED pair `NATIVE` vs `NATIVE_PORTABLE` (isolating ISA-dependent
        # codegen is the whole point of that comparison), so the ladder's
        # `target` parameter does not apply to it -- this is a deliberate
        # omission, not the same oversight as R0's (finding 4).
        #
        # `probe_deps` is forwarded to R1/R2b too (revision 6, spec AC-11,
        # backlog #5210): unlike R3, this never changes what either rung
        # runs or compares, only what `probes` reports from leaves each rung
        # already computes for its own `passed`. `budgets` is still advisory
        # for both (a fusion/lowering budget is not authoritative here) --
        # they still run and classify below, per probe (260915 code review):
        # a probe whose own diverged float leaf has no matching budget, and
        # any probe that transitively depends on it, is omitted and named in
        # a note (never a raise); every OTHER declared probe is still
        # classified normally.
        try:
            r1_result = r1(fn, plan, ai, ci, budgets=budgets, probe_deps=probe_deps, **common)
        except Exception as exc:  # noqa: BLE001 - Layer 2, see docstring
            r1_result = RingResult(
                ring="R1",
                passed=False,
                input_class=label,
                in_contract=in_contract,
                probes=(),
                notes=(f"R1 raised {type(exc).__name__}: {exc}",),
            )
        results.append(r1_result)

        try:
            r2b_result = r2b(
                fn, plan, ai, ci, budgets=budgets, probe_deps=probe_deps, target=target, **common
            )
        except Exception as exc:  # noqa: BLE001 - Layer 2, see docstring
            r2b_result = RingResult(
                ring="R2b",
                passed=False,
                input_class=label,
                in_contract=in_contract,
                probes=(),
                notes=(f"R2b raised {type(exc).__name__}: {exc}",),
            )
        results.append(r2b_result)

        # R3 classifies every probe against R2a's budgets, so it needs R2a to
        # have PASSED -- not merely to have run. R2a fails two ways: it raises
        # (budgets is then {}), or it returns passed=False with incomplete
        # budgets (e.g. a leaf whose shape/dtype differs between the oracle and
        # jit). Either way R3 would only surface a downstream MissingBudgetError
        # on the unbudgeted leaf, recorded against R3 and hiding R2a as the real
        # cause. R1 and R2b treat budgets as advisory, so they still run and
        # classify above per probe (260915 code review): a probe with an
        # unbudgeted diverged float leaf, and its dependents, are omitted and
        # named in a note (never a raise); every other declared probe is
        # still classified.
        if not r2a_result.passed:
            r3_result = RingResult(
                ring="R3",
                passed=False,
                input_class=label,
                in_contract=in_contract,
                probes=(),
                notes=(
                    "R3 skipped: R2a did not pass, so no complete per-leaf budgets "
                    "exist to classify probes against. See R2a's notes for the cause.",
                ),
            )
        else:
            try:
                r3_result = r3(
                    fn,
                    plan,
                    ai,
                    ci,
                    probe_deps,
                    budgets,
                    r0_result=r0_result,
                    target=target,
                    **common,
                )
            except Exception as exc:  # noqa: BLE001 - Layer 2, see docstring
                r3_result = RingResult(
                    ring="R3",
                    passed=False,
                    input_class=label,
                    in_contract=in_contract,
                    probes=(),
                    notes=(f"R3 raised {type(exc).__name__}: {exc}",),
                )
        results.append(r3_result)

    return tuple(results)
