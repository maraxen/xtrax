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
  output. Per AC-11 they take no ``probe_deps``/``probes`` argument at all --
  "nothing type-level distinguishes an instrumented callable from a plain
  one" (spec AC-11), so if the caller passes a probe-carrying ``fn``, these
  rungs simply treat every entry as an ordinary leaf.
- R3 is the only rung that receives ``probe_deps`` and therefore knows which
  top-level names are probes. It derives the model's true ("primary")
  output(s) as the **DAG sinks** of ``probe_deps`` -- the names that are never
  anyone else's declared predecessor -- and builds a second, filtered
  callable returning only those, to run the section-5.4 fidelity precondition
  before trusting anything else it measures.

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

from __future__ import annotations

import re
import subprocess
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
    RingResult,
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
    """Coordinates spanning float32 denormal and near-overflow magnitudes.

    Half the points sit near ``float32`` overflow (``~3e38``), half near the
    smallest normal/denormal boundary (``~1e-38``), stressing reassociation
    sensitivity (section 6.2). The mask is all-ones -- this class is
    in-contract.
    """
    _require_bucket_length(length)
    rng = np.random.default_rng(seed)
    finfo = np.finfo(np.float32)
    half = length // 2
    large = rng.uniform(finfo.max * 0.1, finfo.max * 0.9, size=(half, ndim))
    small = rng.uniform(finfo.tiny, finfo.tiny * 1e4, size=(length - half, ndim))
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
    """
    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(callable_))(*abstract_inputs)
    compiled = compile_for_target(exported.mlir_module(), target)

    outputs = [
        list(_as_flat_sequence(run_native_vmfb(compiled.path, *concrete_inputs)))
        for _ in range(max(replays, 1))
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


def r1_target_isa(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    budgets: Mapping[str, float] | None = None,
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
    ``test_parity_multi_size.py`` already does) is what catches that.

    Args:
        fn: Per-element function, run exactly as given (AC-11).
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to execute both legs with.
        budgets: Per-leaf budgets from R2a, if available -- advisory only,
            included in ``notes`` where a matching key exists.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label.
        in_contract: AC-17 label.

    Returns:
        A ``RingResult`` for ``"R1"``. ``passed=False`` when the
        ``cpu_features`` precondition fails -- the ring refuses to interpret
        its legs, per AC-16 -- rather than reporting a false "no divergence".
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

    native_out = _as_flat_sequence(run_native_vmfb(native.path, *concrete_inputs))
    portable_out = _as_flat_sequence(run_native_vmfb(portable.path, *concrete_inputs))
    native_structured = jax.tree_util.tree_unflatten(exported.out_tree, list(native_out))
    portable_structured = jax.tree_util.tree_unflatten(exported.out_tree, list(portable_out))

    leaves_by_name = _compare_by_name(native_structured, portable_structured)
    notes = (
        f"native cpu_features={native_features!r}",
        f"native-portable cpu_features={portable_features!r}",
        *_format_named_leaf_notes(leaves_by_name, budgets),
    )
    return RingResult(
        ring="R1",
        passed=True,
        input_class=input_class,
        in_contract=in_contract,
        probes=(),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# T5 -- R2a / R2b + section 3.2 budget derivation
# ---------------------------------------------------------------------------


def r2a_fusion(
    fn: Callable[..., Any],
    plan: Any,
    abstract_inputs: Sequence[Any],
    concrete_inputs: Sequence[Any],
    *,
    eager_fn: Callable[[Sequence[Any]], Any],
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
            ``reference_fn`` (``xtrax.export.parity``). Normally covers only
            the model's true output(s), not pipeline-internal probes -- any
            top-level name ``jit_result`` has but ``eager_fn`` omits is
            measured here too, using ``fn`` itself run eagerly (un-jitted),
            since no separate independent oracle is possible for a probe.
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
        calibration applies to float leaves only).
    """
    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    eager_result = eager_fn(concrete_inputs)
    jit_result = jax.jit(callable_)(*concrete_inputs)

    eager_by_name = dict(_top_level_items(eager_result))
    jit_by_name = _top_level_items(jit_result)

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
    missing_from_eager_fn = jit_by_name.keys() - eager_by_name.keys()
    if missing_from_eager_fn:
        raw_eager_by_name = _top_level_items(callable_(*concrete_inputs))
        for name in missing_from_eager_fn:
            eager_by_name[name] = raw_eager_by_name[name]

    names = sorted(eager_by_name.keys() | jit_by_name.keys())

    budgets: dict[str, float] = {}
    notes: list[str] = []
    for name in names:
        if name not in eager_by_name or name not in jit_by_name:
            notes.append(f"{name!r}: present on only one side (eager vs jit) -- skipped")
            continue
        leaves = compare_pytree(eager_by_name[name], jit_by_name[name])
        for leaf in leaves:
            key = budget_key(name, leaf.path)
            if leaf.failed:
                notes.append(f"{key}: FAILED ({leaf.message})")
                continue
            if leaf.dtype_class == "float":
                m_leaf = float(leaf.metrics.get("max_ulp_diff", 0.0))
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
        passed=True,
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
    target: Target = NATIVE,
    boundaries: Mapping[str, Any] | None = None,
    scan_init: Any = None,
    input_class: str = "nominal",
    in_contract: bool = True,
) -> RingResult:
    """R2b: ``jit`` XLA vs IREE -- same graph, different backend. Isolates lowering fidelity.

    ``budgets`` (from R2a) is **advisory here, not authoritative** (section
    3.2): a bound on fusion sensitivity is not a bound on lowering fidelity.
    R2a and R2b are always reported separately for exactly this reason.

    Args:
        fn: Per-element function, run exactly as given (AC-11).
        plan: A BatchPlan.
        abstract_inputs: Abstract inputs to trace with.
        concrete_inputs: Concrete inputs to run both legs with.
        budgets: Per-leaf budgets from R2a for this ``(model, input_class)``.
        target: Compilation target for the IREE leg.
        boundaries: Passed through to ``build_traceable_callable``.
        scan_init: Passed through to ``build_traceable_callable``.
        input_class: Section 6.2 label.
        in_contract: AC-17 label.

    Returns:
        A ``RingResult`` for ``"R2b"``.
    """
    callable_ = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(callable_))(*abstract_inputs)
    jit_result = jax.jit(callable_)(*concrete_inputs)

    compiled = compile_for_target(exported.mlir_module(), target)
    iree_out = _as_flat_sequence(run_native_vmfb(compiled.path, *concrete_inputs))
    iree_structured = jax.tree_util.tree_unflatten(exported.out_tree, list(iree_out))

    leaves_by_name = _compare_by_name(jit_result, iree_structured)
    return RingResult(
        ring="R2b",
        passed=True,
        input_class=input_class,
        in_contract=in_contract,
        probes=(),
        notes=_format_named_leaf_notes(leaves_by_name, budgets),
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
    primary_fn = _primary_only(fn, strip_names)

    # Uninstrumented artifact: only the sink (true primary) output(s).
    primary_callable = build_traceable_callable(primary_fn, plan, boundaries, scan_init=scan_init)
    primary_exported = jax.export.export(jax.jit(primary_callable))(*abstract_inputs)
    primary_compiled = compile_for_target(primary_exported.mlir_module(), target)
    primary_flat = _as_flat_sequence(run_native_vmfb(primary_compiled.path, *concrete_inputs))
    primary_actual = jax.tree_util.tree_unflatten(primary_exported.out_tree, list(primary_flat))

    # Instrumented artifact: fn as given -- primary output(s) plus every
    # declared probe.
    full_callable = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    full_exported = jax.export.export(jax.jit(full_callable))(*abstract_inputs)
    full_compiled = compile_for_target(full_exported.mlir_module(), target)
    full_flat = _as_flat_sequence(run_native_vmfb(full_compiled.path, *concrete_inputs))
    full_by_name = recover_probe_names(full_exported.out_tree, full_flat)
    instrumented_primary = {name: value for name, value in full_by_name.items() if name in sinks}

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
    for name in all_probes:
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
    """
    full_callable = build_traceable_callable(fn, plan, boundaries, scan_init=scan_init)
    exported = jax.export.export(jax.jit(full_callable))(*abstract_inputs)
    probe_result_paths = _probe_result_paths(exported.out_tree)
    validate_probe_deps(exported, probe_deps, probe_result_paths)


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
        probe_deps: The declared probe DAG (section 5.3).
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
        A flat tuple of every ``RingResult`` produced, in execution order.

    Raises:
        Whatever ``validate_fn`` raises when a declared probe-DAG edge fails
        the section-5.3 slice-subset check -- propagated unmodified, before
        any rung executes.
    """
    validate_fn(fn, plan, abstract_inputs, boundaries, scan_init, probe_deps)

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
        common = {
            "boundaries": boundaries,
            "scan_init": scan_init,
            "input_class": input_class.label,
            "in_contract": input_class.in_contract,
        }

        # target applies to R0 exactly as it does to R2b/R3 (finding 4,
        # 260914 code review): R0 is the ladder's hard validity gate, and it
        # must gate the SAME artifact the other toolchain-backed rungs judge
        # -- passing it here, not just to R2b/R3, keeps that true.
        r0_result = r0(fn, plan, ai, ci, replays=replays, target=target, **common)
        results.append(r0_result)
        if not r0_result.passed:
            continue

        # R2a has no `target`: it compares eager JAX against `jax.jit`, never
        # compiling through IREE at all, so no compilation target applies.
        r2a_result, budgets = r2a(fn, plan, ai, ci, eager_fn=eager_fn, **common)
        results.append(r2a_result)

        # R1 deliberately has no `target` either: its own two legs are the
        # FIXED pair `NATIVE` vs `NATIVE_PORTABLE` (isolating ISA-dependent
        # codegen is the whole point of that comparison), so the ladder's
        # `target` parameter does not apply to it -- this is a deliberate
        # omission, not the same oversight as R0's (finding 4).
        r1_result = r1(fn, plan, ai, ci, budgets=budgets, **common)
        results.append(r1_result)

        r2b_result = r2b(fn, plan, ai, ci, budgets=budgets, target=target, **common)
        results.append(r2b_result)

        r3_result = r3(
            fn, plan, ai, ci, probe_deps, budgets, r0_result=r0_result, target=target, **common
        )
        results.append(r3_result)

    return tuple(results)
