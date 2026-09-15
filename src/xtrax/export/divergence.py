"""Divergence mapping: pure metrics, classification, and MLIR slice analysis.

This module answers *where* a compiled artifact stopped matching production
JAX, instead of ``xtrax.export.parity.compare``'s single scalar. See
``.praxia/docs/specs/260911_export-divergence-mapping.md`` (the spec this
module implements) for the full design rationale.

**This module is pure.** It imports nothing from ``iree`` -- only
``jax``/``jaxlib``/``numpy`` -- so it is importable and testable under
``tier1_core``, which syncs only the ``dev``/``io`` extras and never the
``export`` extra (spec S5.5a, AC-12). Execution against a real toolchain
(R0-R3 ring runners) lives in ``xtrax.export.rings``, which is
``coverage_omit``'d because it cannot be exercised without IREE.

Two non-pinned design decisions this module makes, called out here because a
parallel implementation of ``rings.py`` must agree with them:

1. **Budget key convention.** :func:`classify_probes` looks up a float leaf's
   calibrated budget in its ``budgets: Mapping[str, float]`` argument under
   the key :func:`budget_key` produces -- a probe whose value is a single
   array (``leaf.path == ""``) keys as the bare probe name; a probe holding a
   nested structure keys as ``"<probe_name><_BUDGET_KEY_SEP><keystr path>"``
   (Finding 6: a separator, not bare concatenation, so the mapping stays
   injective -- see :func:`budget_key`'s docstring). Whatever builds
   ``budgets`` (rings.py's R2a measurement, per SS3.2/6.0) MUST use
   :func:`budget_key` to construct it.
2. **Division of labour between `compare_pytree` and `classify_probes`.**
   ``compare_pytree`` has no notion of a calibrated budget (SS6.0 pins that
   budgets are only known to ``classify_probes``, derived downstream of R2a).
   So for a float leaf, ``compare_pytree`` can only ever assert
   ``Severity.IDENTICAL`` (bit-identical) or ``Severity.BEYOND_BUDGET``
   (provisional -- "not proven identical yet"); ``classify_probes`` is what
   turns a provisional ``BEYOND_BUDGET`` into ``WITHIN_BUDGET`` once it has a
   budget to compare against. For integer/bool leaves ``compare_pytree``'s
   severity is authoritative and final -- SS3.2 is explicit that the discrete
   criterion is exact-match with no budget, ever.
"""

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any, Literal

import jax
import numpy as np

if TYPE_CHECKING:
    from jaxlib.mlir import ir as mlir_ir

__all__ = [
    "ULP_FLOOR",
    "SLACK",
    "Severity",
    "DivergenceClass",
    "LeafDivergence",
    "ProbeReport",
    "RingResult",
    "ProbeResolution",
    "ProbeStructureError",
    "MissingBudgetError",
    "ProbeDependencyError",
    "budget_leaf",
    "budget_key",
    "compare_pytree",
    "classify_probes",
    "probe_resolution",
    "validate_probe_deps",
]


# --------------------------------------------------------------------------
# SS6.0 pinned types
# --------------------------------------------------------------------------


class Severity(IntEnum):
    """Comparable ordinal used to order probes/leaves across dtype classes (SS5.3).

    Magnitude metrics (``max_ulp_diff``, ``n_mismatched``, ...) order two
    leaves only *within* a dtype class; ``Severity`` is what lets a
    discrete leaf and a float leaf be compared at all.
    """

    IDENTICAL = 0
    WITHIN_BUDGET = 1
    BEYOND_BUDGET = 2


class DivergenceClass(StrEnum):
    """The five per-probe classes of SS5.3, in the precedence order they are
    evaluated (:func:`classify_probes` takes the first match).
    """

    DISCRETE_FLIP = "DISCRETE_FLIP"
    INJECTED = "INJECTED"
    AMPLIFIED = "AMPLIFIED"
    ATTENUATED = "ATTENUATED"
    CLEAN = "CLEAN"


@dataclass(frozen=True)
class LeafDivergence:
    """One leaf's comparison result (SS5.2, SS6.0).

    Attributes:
        path: ``jax.tree_util.keystr`` of the leaf within its probe's pytree.
        dtype_class: Which SS5.2 metric table applies.
        severity: See the module docstring's division-of-labour note --
            authoritative for integer/bool leaves, provisional
            (``IDENTICAL``/``BEYOND_BUDGET`` only) for float leaves until
            :func:`classify_probes` applies a calibrated budget.
        metrics: The SS5.2 metric set for ``dtype_class``. Empty when
            ``failed`` is True (a shape/dtype mismatch has no leaf metrics).
        failed: True for a **leaf** mismatch (matching treedefs, differing
            leaf shape or dtype) -- SS5.2 distinguishes this from a
            **treedef** mismatch, which raises :class:`ProbeStructureError`
            instead of producing a ``LeafDivergence`` at all.
        message: Human-readable detail; set only when ``failed`` is True.
    """

    path: str
    dtype_class: Literal["float", "integer", "bool"]
    severity: Severity
    metrics: Mapping[str, float]
    failed: bool
    message: str | None


@dataclass(frozen=True)
class ProbeReport:
    """One probe's classification result (SS5.3, SS6.0)."""

    name: str
    predecessors: tuple[str, ...]
    leaves: tuple[LeafDivergence, ...]
    severity: Severity
    divergence_class: DivergenceClass


@dataclass(frozen=True)
class RingResult:
    """One ring's outcome (SS3, SS6.0). Constructed by ``rings.py``; the type
    lives here so both modules share one definition (SS6.0).
    """

    ring: Literal["R0", "R1", "R2a", "R2b", "R3"]
    passed: bool
    input_class: str
    in_contract: bool
    probes: tuple[ProbeReport, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ProbeResolution:
    """Backward-slice-based resolution report (SS5.5b, T3, AC-9).

    Attributes:
        slice_delta: ``f"{predecessor}->{probe}"`` -> ``|slice(probe) \\
            slice(predecessor)|``, one entry per declared ``probe_deps`` edge.
        unattributed_ops: Count of ops in the module that appear in no
            ``slice_delta`` value (SS5.5b) -- ops never distinguishing any
            declared edge, whether because they are shared by every probe or
            because no declared probe needs them at all.
        total_ops: Total operation count in the exported function (including
            nested region bodies, counted once each -- see the "counts are
            static" caveat on :func:`probe_resolution`).
    """

    slice_delta: Mapping[str, int]
    unattributed_ops: int
    total_ops: int


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ProbeStructureError(ValueError):
    """Raised by :func:`compare_pytree` when two pytrees' treedefs differ (SS5.2, AC-13).

    Unlike a leaf mismatch (matching treedefs, differing leaf shape/dtype,
    which produces a per-leaf failure record and continues), a treedef
    mismatch means there is no leaf correspondence at all -- pairing by
    position would silently compare unrelated arrays, so this raises instead.
    """

    def __init__(self, expected_treedef: Any, actual_treedef: Any) -> None:  # noqa: ANN401
        message = (
            "pytree structure mismatch: expected treedef "
            f"{expected_treedef!r} vs actual treedef {actual_treedef!r}"
        )
        super().__init__(message)
        self.expected_treedef = expected_treedef
        self.actual_treedef = actual_treedef


class MissingBudgetError(ValueError):
    """Raised by :func:`classify_probes` when a float leaf that diverged has
    no entry in ``budgets`` (SS6.0: "a leaf absent from budgets is an error,
    not a default").
    """


class ProbeDependencyError(ValueError):
    """Raised by :func:`validate_probe_deps` when a declared edge fails the
    slice-subset check (SS5.3, AC-15).
    """

    def __init__(
        self,
        predecessor: str,
        probe: str,
        predecessor_slice: frozenset[int],
        probe_slice: frozenset[int],
    ) -> None:
        missing = predecessor_slice - probe_slice
        shown = sorted(missing)[:10]
        ellipsis = "..." if len(missing) > 10 else ""
        message = (
            f"declared edge {predecessor!r} -> {probe!r} failed the slice-subset check: "
            f"slice({predecessor!r}) has {len(predecessor_slice)} op(s), "
            f"slice({probe!r}) has {len(probe_slice)} op(s), and "
            f"{len(missing)} op(s) in slice({predecessor!r}) are absent from slice({probe!r}) "
            f"(indices {shown}{ellipsis}); {probe!r} does not appear to depend on {predecessor!r}"
        )
        super().__init__(message)
        self.predecessor = predecessor
        self.probe = probe
        self.predecessor_slice = predecessor_slice
        self.probe_slice = probe_slice


# --------------------------------------------------------------------------
# T1 -- compare_pytree (SS5.2, AC-1, AC-13)
# --------------------------------------------------------------------------


def _dtype_class(dtype: np.dtype) -> Literal["float", "integer", "bool"]:
    if dtype == np.bool_:
        return "bool"
    # `np.issubdtype(bfloat16, np.floating)` is False -- ml_dtypes registers
    # bfloat16 as a `void`-kind extension dtype, not a numpy floating
    # subtype -- even though xtrax explicitly supports bf16 output
    # (`_CODEGEN_DTYPES` in targets.py). Checked by name rather than
    # importing ml_dtypes, since every numpy-facing operation this module
    # needs (isfinite, isnan, ==, abs, spacing, astype) is already correctly
    # registered on it via its ufunc overloads.
    if np.issubdtype(dtype, np.floating) or dtype.name == "bfloat16":
        return "float"
    if np.issubdtype(dtype, np.integer):
        return "integer"
    raise ValueError(f"unsupported leaf dtype for divergence comparison: {dtype!r}")


def _ulp_distance(exp: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Overflow-free per-element ULP distance: ``|exp - act| /
    spacing(max(|exp|, |act|))``, evaluated in ``exp``/``act``'s own dtype
    -- falling back to ``spacing(min(|exp|, |act|))`` wherever the first
    scale is infinite (see Round-4 finding below).

    Within one binade this counts representable steps exactly, matching the
    classic sign-bit-fold bit-cast trick for same-width floats (Finding 7:
    that trick is preserved as a private ``_ulp_ordered`` helper in
    ``tests/export/test_divergence.py``, not shipped here -- see its
    docstring for the float64 overflow this function replaces); across a
    binade boundary or a sign change ``_ulp_distance`` reads correspondingly
    large -- which is correct (a sign flip IS a huge divergence) -- and it
    can never be negative, unlike that bit-cast diff for float64.

    Evaluating ``np.spacing`` in the leaf's own dtype (rather than upcasting
    to float64 first) is also what makes this correct for bfloat16:
    bfloat16 shares float32's exponent range with far fewer mantissa bits,
    so its representable step at a given magnitude is much coarser than
    float64's -- upcasting before computing spacing would measure the
    float64 grid's spacing instead of bfloat16's, misreading a last-bit
    bfloat16 nudge as an enormous distance. ``np.spacing`` is dtype-aware
    for every width this module supports, including registered extension
    types like ``ml_dtypes.bfloat16`` -- only the final subtraction and
    division are done in float64, for a leaf whose own dtype cannot
    represent the ratio.
    """
    # No floor here (Finding 4): `np.spacing` never returns exactly 0 for any
    # finite input, including 0.0 itself, in every dtype this module supports
    # (checked directly -- see the test suite's `_ulp_ordered` helper
    # docstring for the analogous per-width check). A
    # `np.finfo(np.float64).tiny` floor was here
    # previously to guard a division by zero that cannot occur; it instead
    # raised a genuine float64 subnormal step (`np.spacing` ~4.9e-324) up to
    # `tiny` (~2.2e-308, the smallest NORMAL float64) -- ~15 orders of
    # magnitude too coarse, which read a divergence billions of
    # representable steps wide as ~0.0045 ULP, well within any real budget.
    abs_exp = np.abs(exp)
    abs_act = np.abs(act)
    hi = np.maximum(abs_exp, abs_act)
    lo = np.minimum(abs_exp, abs_act)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # `diff`'s own subtraction can overflow for a float64 pair at
        # opposite extremes (e.g. `-finfo.max` vs `+finfo.max`) -- silenced
        # here for the same "overflow is a legitimate answer" reason as
        # everything else in this block.
        diff = np.abs(exp.astype(np.float64) - act.astype(np.float64))
        # Round-4 finding (HIGH): at a leaf's OWN dtype `max` (or a sign
        # flip spanning +-max), `np.spacing(hi)` is `inf` -- the next
        # representable value would overflow the dtype's range, and IEEE
        # 754 defines the spacing at the top of the range as infinite. The
        # old bare `diff / spacing(hi)` then read `finite / inf == 0.0`,
        # silently classifying a divergence as large as `2 * finfo.max`
        # (~3.4e38 for float32) as bit-identical. `finfo.max` is a realistic
        # value here, not a contrived one -- it is a common masked-distance
        # fill, and float16 activations routinely saturate at 65504.
        #
        # Falling back to `spacing(lo)` when `spacing(hi)` is infinite
        # recovers a genuine measurement for the specific case that matters
        # most: `hi` sitting exactly at the dtype's `max` with `lo` an
        # ordinary in-range value (e.g. `nextafter(max, 0)` one step below
        # `max`, or a small `lo` far below it) -- `spacing(lo)` is always
        # finite there, so a one-step nudge at the top of the range still
        # measures ~1 ULP instead of collapsing to 0 (Finding 1) or flipping
        # to an opaque `inf` (which would erase the magnitude information
        # that motivated `_ulp_distance` over ``_ulp_ordered`` in the first
        # place). The fallback only leaves a leaf unmeasurable when BOTH
        # `hi` and `lo` sit at that same infinite-spacing edge (`lo == hi ==
        # max`, e.g. `-max` vs `+max`) -- there is no finite scale left to
        # fall back to, and that pair genuinely is the largest possible
        # divergence at that magnitude.
        scale_hi = np.spacing(hi).astype(np.float64)
        scale_lo = np.spacing(lo).astype(np.float64)
        scale = np.where(np.isfinite(scale_hi), scale_hi, scale_lo)
        # `diff` can independently overflow to `inf` too, but only when
        # `exp`/`act` are themselves float64 (the only width this module
        # upcasts *from* float64, so nothing else can produce an
        # unrepresentable subtraction here) -- e.g. `-finfo.max - finfo.max`
        # overflows float64's own range. `inf / inf` is `nan`, which I2
        # forbids outright, so both the "scale still infinite after the
        # fallback" and the "diff overflowed" cases are folded into the same
        # explicit "unmeasurable, but genuinely divergent" branch: `inf`,
        # never `nan`. `errstate` suppresses the resulting overflow/invalid
        # warnings -- they fire on every leaf at this domain edge, which is
        # signal-free noise once the branch below already accounts for them
        # (the round-3 errstate precedent for the `max_rel_diff`
        # zero-reference case, below).
        ratio = diff / scale
        measurable = np.isfinite(diff) & np.isfinite(scale)
        return np.where(diff == 0.0, 0.0, np.where(measurable, ratio, np.inf))


def _float_metrics(exp: np.ndarray, act: np.ndarray) -> dict[str, float]:
    exp_finite = np.isfinite(exp)
    act_finite = np.isfinite(act)
    both_finite = exp_finite & act_finite
    both_nan = np.isnan(exp) & np.isnan(act)
    same_value = exp == act
    nonfinite_mismatch = ~(both_finite | both_nan | same_value)
    n_nonfinite_mismatch = float(np.count_nonzero(nonfinite_mismatch))

    if np.any(both_finite):
        ef = exp[both_finite]
        af = act[both_finite]
        # A float64 pair at opposite extremes (e.g. `-finfo.max` vs
        # `+finfo.max`) overflows this subtraction to `inf` -- both `ef` and
        # `af` are individually finite (that's what put them in this
        # branch), the SUBTRACTION is what cannot be represented. `inf` is
        # the legitimate reading (the true distance genuinely exceeds
        # float64's range), the same "overflow is a correct answer, not an
        # accident" precedent as `max_rel_diff`'s zero-reference division
        # below -- silenced here for the identical reason: a warning per
        # such leaf is noise, not signal (round-4 finding).
        with np.errstate(over="ignore"):
            abs_diff = np.abs(ef.astype(np.float64) - af.astype(np.float64))
        max_abs_diff = float(np.max(abs_diff))
        # Unlike `_ulp_distance`'s scale (see Finding 4's fix above), `ef`
        # genuinely can be exactly 0.0 here, so a floor is still needed to
        # avoid a literal division by zero -- but the same audit finding
        # applies: `np.finfo(np.float64).tiny` (smallest NORMAL float64,
        # ~2.2e-308) floors a real, nonzero, *subnormal* `ef` far above its
        # true magnitude, masking a genuine large relative divergence the
        # same way it masked `_ulp_distance`'s scale. `smallest_subnormal`
        # (~4.9e-324) is the true minimum positive float64 magnitude, so it
        # only guards the literal-zero case without distorting any subnormal
        # `ef` that is merely small.
        denom = np.maximum(np.abs(ef.astype(np.float64)), np.finfo(np.float64).smallest_subnormal)
        # An exact-zero reference with a nonzero actual has unbounded relative
        # error, so this division legitimately overflows to inf. That is the
        # correct reading, not an accident, and must not surface as a
        # RuntimeWarning on every leaf whose reference is zero.
        with np.errstate(over="ignore", divide="ignore"):
            max_rel_diff = float(np.max(abs_diff / denom))
        max_ulp_diff = float(np.max(_ulp_distance(ef, af)))
    else:
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        max_ulp_diff = 0.0

    return {
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "max_ulp_diff": max_ulp_diff,
        "n_nonfinite_mismatch": n_nonfinite_mismatch,
    }


def _first_mismatch_index(mismatch: np.ndarray) -> float:
    idx = np.flatnonzero(mismatch.reshape(-1))
    return float(idx[0]) if idx.size else -1.0


def _integer_metrics(exp: np.ndarray, act: np.ndarray) -> dict[str, float]:
    mismatch = exp != act
    n = exp.size
    n_mismatched = float(np.count_nonzero(mismatch))
    exact_match_fraction = 1.0 if n == 0 else float(1.0 - n_mismatched / n)
    return {
        "exact_match_fraction": exact_match_fraction,
        "first_mismatch_index": _first_mismatch_index(mismatch),
        "n_mismatched": n_mismatched,
    }


def _bool_metrics(exp: np.ndarray, act: np.ndarray) -> dict[str, float]:
    mismatch = exp != act
    return {
        "hamming_distance": float(np.count_nonzero(mismatch)),
        "first_mismatch_index": _first_mismatch_index(mismatch),
    }


def _compare_leaf(path: str, expected_leaf: Any, actual_leaf: Any) -> LeafDivergence:  # noqa: ANN401
    exp = np.asarray(expected_leaf)
    dtype_class = _dtype_class(exp.dtype)
    act = np.asarray(actual_leaf)

    if exp.shape != act.shape or exp.dtype != act.dtype:
        return LeafDivergence(
            path=path,
            dtype_class=dtype_class,
            severity=Severity.BEYOND_BUDGET,
            metrics={},
            failed=True,
            message=(
                f"leaf shape/dtype mismatch at {path!r}: "
                f"expected shape={exp.shape} dtype={exp.dtype}, "
                f"actual shape={act.shape} dtype={act.dtype}"
            ),
        )

    if dtype_class == "float":
        metrics = _float_metrics(exp, act)
        identical = metrics["n_nonfinite_mismatch"] == 0.0 and metrics["max_ulp_diff"] == 0.0
    elif dtype_class == "integer":
        metrics = _integer_metrics(exp, act)
        identical = metrics["exact_match_fraction"] == 1.0
    else:
        metrics = _bool_metrics(exp, act)
        identical = metrics["hamming_distance"] == 0.0

    severity = Severity.IDENTICAL if identical else Severity.BEYOND_BUDGET
    return LeafDivergence(
        path=path,
        dtype_class=dtype_class,
        severity=severity,
        metrics=metrics,
        failed=False,
        message=None,
    )


def compare_pytree(expected: Any, actual: Any) -> tuple[LeafDivergence, ...]:  # noqa: ANN401
    """Compare two pytrees leaf-by-leaf with dtype-aware metrics (SS5.2).

    A **leaf** mismatch (treedefs agree, a leaf's shape or dtype does not)
    yields a per-leaf failure record and the walk **continues** -- leaves
    still correspond pairwise, so a pytree walk is a report, not an assertion
    (AC-1). A **treedef** mismatch means there is no leaf correspondence at
    all, and **raises** :class:`ProbeStructureError` naming both treedefs
    (AC-13) rather than producing a nonsensical positional pairing.

    Args:
        expected: The reference pytree (e.g. eager/XLA output).
        actual: The pytree under test (e.g. IREE output).

    Returns:
        One :class:`LeafDivergence` per leaf, in flatten order.

    Raises:
        ProbeStructureError: If the two pytrees' treedefs differ.
    """
    exp_leaves, exp_treedef = jax.tree_util.tree_flatten_with_path(expected)
    act_leaves, act_treedef = jax.tree_util.tree_flatten_with_path(actual)
    if exp_treedef != act_treedef:
        raise ProbeStructureError(exp_treedef, act_treedef)

    return tuple(
        _compare_leaf(jax.tree_util.keystr(path), exp_leaf, act_leaf)
        for (path, exp_leaf), (_, act_leaf) in zip(exp_leaves, act_leaves, strict=True)
    )


# --------------------------------------------------------------------------
# SS3.2 -- budget_leaf
# --------------------------------------------------------------------------

ULP_FLOOR: float = 4.0
SLACK: float = 4.0


def budget_leaf(m_leaf_ulp: float) -> float:
    """The SS3.2 calibrated tolerance budget for one float leaf, in ULP.

    ``max(ULP_FLOOR, SLACK * m_leaf_ulp)``. Flat (== ``ULP_FLOOR``) at and
    below ``m_leaf_ulp = ULP_FLOOR / SLACK``; strictly increasing above it.
    The floor is mandatory, not a defeat: ``eager == jit`` bit-identical is
    the normal case for fusion-insensitive graphs, and without a floor such a
    model would get budget 0 and fail on ordinary last-bit fusion noise.

    Args:
        m_leaf_ulp: R2a's measured eager-vs-jit divergence for this leaf,
            denominated in ``max_ulp_diff`` (SS5.2), so it and the floor share
            units.

    Returns:
        The budget, in ULP, to compare a leaf's ``max_ulp_diff`` against.
    """
    return max(ULP_FLOOR, SLACK * m_leaf_ulp)


_BUDGET_KEY_SEP = "\x1f"
"""Separator inserted between ``probe_name`` and a non-empty ``leaf_path`` in
:func:`budget_key` (Finding 6). ASCII Unit Separator (0x1F) -- a control
character reserved exactly for delimiting structured fields, so it is never
emitted by ``jax.tree_util.keystr`` (whose alphabet for a non-empty path is
limited to brackets, quotes, dots, digits, and the literal characters of
dict keys/attrs/NamedTuple fields, all printable) and is not a character any
ordinary probe name would contain. A bare concatenation collided a probe
``"a"`` with leaf path ``"['b']"`` against a sibling top-level probe
literally named ``"a['b']"`` whose own value is a single array (path
``""``); inserting this separator before a non-empty path makes the two
keys ``"a\\x1f['b']"`` and ``"a['b']"`` distinct.
"""


def budget_key(probe_name: str, leaf_path: str) -> str:
    """The key convention :func:`classify_probes` uses to look up a float
    leaf's calibrated budget in its ``budgets`` argument (see the module
    docstring's "Budget key convention" note). Exposed so a budget-producing
    caller (rings.py's R2a measurement) builds matching keys.

    A probe whose value is a single leaf (``leaf_path == ""``) keys as the
    bare probe name, unchanged from before Finding 6 -- there is nothing to
    delimit, and every existing caller keying on ``probe_name`` alone for
    such a leaf keeps working. A nested leaf's non-empty ``leaf_path``
    (always starting with ``[`` or ``.``, per ``jax.tree_util.keystr``) is
    joined to ``probe_name`` with :data:`_BUDGET_KEY_SEP` instead of bare
    concatenation, so the mapping is injective (Finding 6): two distinct
    ``(probe_name, leaf_path)`` pairs can no longer produce the same key.
    """
    if not leaf_path:
        return probe_name
    return f"{probe_name}{_BUDGET_KEY_SEP}{leaf_path}"


# --------------------------------------------------------------------------
# T2 -- classify_probes (SS5.3, AC-3, AC-3b, AC-4, AC-5, AC-14, AC-19)
# --------------------------------------------------------------------------


def _topological_order(
    names: Iterable[str], probe_deps: Mapping[str, tuple[str, ...]]
) -> list[str]:
    names = list(names)
    name_set = set(names)
    for probe, preds in probe_deps.items():
        if probe not in name_set:
            raise ValueError(f"probe_deps declares dependencies for unknown probe {probe!r}")
        for pred in preds:
            if pred not in name_set:
                raise ValueError(
                    f"probe {probe!r} declares an undeclared predecessor {pred!r} (not in probes)"
                )

    order: list[str] = []
    visited: set[str] = set()
    in_progress: set[str] = set()

    def _visit(name: str) -> None:
        if name in visited:
            return
        if name in in_progress:
            raise ValueError(f"probe_deps contains a cycle involving {name!r}")
        in_progress.add(name)
        for pred in probe_deps.get(name, ()):
            _visit(pred)
        in_progress.discard(name)
        visited.add(name)
        order.append(name)

    for name in names:
        _visit(name)
    return order


def _resolve_leaf_severity(
    probe_name: str, leaf: LeafDivergence, budgets: Mapping[str, float]
) -> LeafDivergence:
    """Upgrade a float leaf's provisional severity using a calibrated budget.

    Integer/bool leaves and already-failed leaves pass through unchanged --
    SS3.2 gives them no budget to apply. A float leaf already IDENTICAL needs
    no budget either (that is a measured fact, not a default); only a float
    leaf that measured non-identical needs a budget to decide
    WITHIN_BUDGET vs BEYOND_BUDGET, and its absence from ``budgets`` is an
    error (SS6.0).
    """
    if leaf.failed or leaf.dtype_class != "float" or leaf.severity == Severity.IDENTICAL:
        return leaf

    key = budget_key(probe_name, leaf.path)
    if key not in budgets:
        raise MissingBudgetError(
            f"no calibrated budget for float leaf {key!r} (probe {probe_name!r}, "
            f"leaf path {leaf.path!r}); classify_probes never measures or derives "
            "a budget, it must be supplied in `budgets`"
        )
    budget = budgets[key]

    if leaf.metrics.get("n_nonfinite_mismatch", 0.0) > 0.0:
        new_severity = Severity.BEYOND_BUDGET
    elif leaf.metrics.get("max_ulp_diff", float("inf")) <= budget:
        new_severity = Severity.WITHIN_BUDGET
    else:
        new_severity = Severity.BEYOND_BUDGET
    return dataclasses.replace(leaf, severity=new_severity)


def classify_probes(
    probes: Mapping[str, tuple[LeafDivergence, ...]],
    probe_deps: Mapping[str, tuple[str, ...]],
    budgets: Mapping[str, float],
) -> tuple[ProbeReport, ...]:
    """Classify each probe against its declared predecessors (SS5.3, T2).

    Classification is against the **worst predecessor**
    (``max``-over-predecessors on :class:`Severity`, never an arbitrary one),
    over a DAG the caller declares via ``probe_deps`` -- dataflow edges cannot
    be recovered from a flat output tuple, so there is no inference fallback.

    Precedence (first match wins): ``DISCRETE_FLIP``, ``INJECTED``,
    ``AMPLIFIED``, ``ATTENUATED``, ``CLEAN``.

    - ``DISCRETE_FLIP``: an integer/bool leaf mismatched (and did **not**
      fail structurally -- see the Finding 4 note below), and every
      predecessor was severity 0 on *all* leaves, discrete included (an empty
      predecessor set counts as satisfying this by rule).
    - ``INJECTED``: every predecessor is severity 0, this probe is severity 2
      (**not** "severity >= 1" -- that phrasing made every within-budget
      float leaf below a clean predecessor read as an injected semantic
      change, exactly the outcome the SS3.2 floor exists to prevent), and no
      leaf on this probe failed structurally (Finding 4, below).
    - ``AMPLIFIED``: this probe has a non-``IDENTICAL`` severity, and either
      some predecessor is above severity 0 with this probe's severity ``>=``
      every predecessor's, **or** a leaf on this probe failed structurally
      while every predecessor was clean (Finding 4: treated as severity
      rising off a clean-predecessor baseline of ``IDENTICAL``).
    - ``ATTENUATED``: some predecessor is above severity 0 (and no leaf on
      this probe failed structurally), and this probe's severity is strictly
      below the predecessor max.
    - ``CLEAN``: severity <= 1 and none of the above.

    **Finding 4 -- a structurally failed leaf is not a value flip.** A leaf
    with ``failed=True`` (shape/dtype mismatch, ``metrics={}``) never had an
    element-wise comparison run, so it cannot support ``DISCRETE_FLIP`` or
    ``INJECTED``'s claim that specific values were compared and diverged --
    those two classes exclude any probe with a failed leaf outright,
    regardless of predecessor cleanliness. Such a probe is not reported
    CLEAN either (the comparison genuinely could not be performed): it falls
    through to the same severity-ordinal AMPLIFIED/ATTENUATED machinery every
    other probe uses, comparing against ``IDENTICAL`` when predecessors were
    otherwise clean. The leaf's own ``failed=True`` and empty ``metrics``
    remain visible on :attr:`ProbeReport.leaves` regardless of the class this
    produces.

    Args:
        probes: Probe name -> that probe's leaves, as produced by
            :func:`compare_pytree` (float leaf severities are provisional and
            get resolved here against ``budgets``).
        probe_deps: Probe name -> tuple of immediate predecessor probe names.
            A probe absent from this mapping is treated as having no declared
            predecessors.
        budgets: Calibrated per-leaf budgets, keyed by :func:`budget_key`.
            Never measured or derived here -- an absent entry for a
            non-identical float leaf is a :class:`MissingBudgetError`, not a
            default.

    Returns:
        One :class:`ProbeReport` per entry in ``probes``, in ``probes``'
        iteration order (not topological order).

    Raises:
        MissingBudgetError: A diverged float leaf has no entry in ``budgets``.
        ValueError: ``probe_deps`` names an unknown probe, or contains a
            cycle.
    """
    order = _topological_order(probes.keys(), probe_deps)
    reports: dict[str, ProbeReport] = {}

    for name in order:
        preds = probe_deps.get(name, ())
        leaves = tuple(_resolve_leaf_severity(name, leaf, budgets) for leaf in probes[name])
        severity = max((leaf.severity for leaf in leaves), default=Severity.IDENTICAL)
        # Finding 4: a leaf with `failed=True` (shape/dtype mismatch, empty
        # `metrics`) never had an element-wise comparison run, so it cannot
        # support `discrete_mismatch`'s claim that specific index/bool values
        # were compared and differed -- excluded here regardless of
        # dtype_class.
        discrete_mismatch = any(
            leaf.dtype_class in ("integer", "bool")
            and leaf.severity != Severity.IDENTICAL
            and not leaf.failed
            for leaf in leaves
        )
        any_leaf_failed = any(leaf.failed for leaf in leaves)

        pred_severities = [reports[pred].severity for pred in preds]
        all_predecessors_clean = not pred_severities or all(
            s == Severity.IDENTICAL for s in pred_severities
        )
        max_pred_severity = max(pred_severities) if pred_severities else Severity.IDENTICAL
        # Finding 4: DISCRETE_FLIP and INJECTED are both "value-flip" claims
        # -- they assert a specific value-level narrative (an exact-match
        # index/bool flip, or a measured semantic change) that a leaf whose
        # comparison never ran cannot support. A probe with any such failed
        # leaf is excluded from both regardless of `all_predecessors_clean`,
        # and instead falls through to the plain severity-ordinal
        # AMPLIFIED/ATTENUATED comparison below (against a clean-predecessor
        # baseline of IDENTICAL when nothing upstream actually diverged) --
        # not CLEAN, since the comparison genuinely could not be performed,
        # and not a sixth DivergenceClass member (see this sprint's fixer
        # report for the escalation this decision avoided).
        value_flip_eligible = all_predecessors_clean and not any_leaf_failed

        if discrete_mismatch and value_flip_eligible:
            divergence_class = DivergenceClass.DISCRETE_FLIP
        elif value_flip_eligible and severity == Severity.BEYOND_BUDGET:
            divergence_class = DivergenceClass.INJECTED
        elif (
            not value_flip_eligible
            and severity > Severity.IDENTICAL
            and severity >= max_pred_severity
        ):
            divergence_class = DivergenceClass.AMPLIFIED
        elif not value_flip_eligible and severity < max_pred_severity:
            divergence_class = DivergenceClass.ATTENUATED
        else:
            divergence_class = DivergenceClass.CLEAN

        reports[name] = ProbeReport(
            name=name,
            predecessors=tuple(preds),
            leaves=leaves,
            severity=severity,
            divergence_class=divergence_class,
        )

    return tuple(reports[name] for name in probes)


# --------------------------------------------------------------------------
# T3 / T3b -- MLIR backward-slice machinery (SS5.5b, SS5.3)
# --------------------------------------------------------------------------


def _module_entry_func(module: "mlir_ir.Module") -> Any:  # noqa: ANN401
    from jaxlib.mlir import ir as mlir_ir_rt

    funcs = [op for op in module.body.operations if op.operation.name == "func.func"]
    if not funcs:
        raise ValueError("exported StableHLO module has no func.func entry point")
    if len(funcs) == 1:
        return funcs[0]
    for op in funcs:
        name_attr = mlir_ir_rt.StringAttr(op.operation.attributes["sym_name"])
        if name_attr.value == "main":
            return op
    return funcs[0]


def _entry_block_ops(func_op: Any) -> list[Any]:  # noqa: ANN401
    return list(func_op.operation.regions[0].blocks[0].operations)


def _all_ops_in_region_tree(op: Any) -> list[Any]:  # noqa: ANN401
    """All operations nested (transitively) inside ``op``'s regions.

    This is what keeps SS5.5b's "counts are static" caveat true: a loop body
    (e.g. a ``stablehlo.while`` from a ``jax.lax.scan``) is walked once here,
    not once per iteration.
    """
    ops: list[Any] = []
    for region in op.operation.regions:
        for block in region.blocks:
            for nested in block.operations:
                ops.append(nested)
                ops.extend(_all_ops_in_region_tree(nested))
    return ops


def _all_function_ops(func_op: Any) -> list[Any]:  # noqa: ANN401
    top_ops = _entry_block_ops(func_op)
    all_ops = list(top_ops)
    for op in top_ops:
        all_ops.extend(_all_ops_in_region_tree(op))
    return all_ops


def _all_module_ops(module: Any) -> list[Any]:  # noqa: ANN401
    """Every operation across every ``func.func`` in ``module`` (including
    nested region bodies), each counted once.

    Covers not just the entry function's own body but every function
    reachable via a ``call`` -- a probe's backward slice can cross into a
    callee (Finding 3 / T3), so ``op_index`` must have an entry for any op
    that walk might visit, not just ``main``'s own ops.
    """
    ops: list[Any] = []
    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        ops.extend(_all_function_ops(op))
    return ops


def _call_callee_sym_name(op: Any) -> str | None:  # noqa: ANN401
    """The callee symbol name if ``op`` is a ``func.call``, else ``None``."""
    if op.operation.name != "func.call":
        return None
    from jaxlib.mlir import ir as mlir_ir_rt

    return mlir_ir_rt.FlatSymbolRefAttr(op.operation.attributes["callee"]).value


def _resolve_func_by_sym_name(module: Any, sym_name: str) -> Any:  # noqa: ANN401
    """Look up a ``func.func`` in ``module`` by its ``sym_name`` attribute --
    the same resolution :func:`_module_entry_func` does for ``main``, reused
    here to follow a ``call @callee`` edge.
    """
    from jaxlib.mlir import ir as mlir_ir_rt

    for op in module.body.operations:
        if op.operation.name != "func.func":
            continue
        name_attr = mlir_ir_rt.StringAttr(op.operation.attributes["sym_name"])
        if name_attr.value == sym_name:
            return op
    raise ValueError(
        f"call references callee {sym_name!r}, but no func.func with that sym_name "
        "exists in the module"
    )


def _backward_slice_ops(value: Any, module: Any) -> set[Any]:  # noqa: ANN401
    """All operations transitively needed to compute ``value``.

    A def-use walk over operands, stopping at block arguments (the owning
    function's own inputs). When a visited op has regions (e.g. a loop),
    every op nested inside those regions is included too, since the op's
    result depends on its whole body.

    When the walk reaches a value produced by a ``func.call`` (e.g. a
    nested ``jax.jit`` or an equinox ``filter_jit`` inner call, which both
    lower to a ``call`` into a separate ``func.func``), it additionally
    resolves the callee by ``sym_name`` and continues the walk from the
    callee's own return operand matching that value's result index -- so
    two outputs of the *same* call (e.g. probes ``a`` and ``b`` where
    ``b`` further transforms ``a`` inside the callee) still get distinct
    slices, instead of both collapsing to the bare call op with zero delta
    between them. Reached via the bulk "nested op in a visited op's region"
    path (e.g. a ``call`` inside a ``stablehlo.while`` body), where no
    single result value drives inclusion, every one of that call's results
    is followed conservatively -- consistent with that path's existing
    "the whole body is needed" over-approximation.
    """
    from jaxlib.mlir import ir as mlir_ir_rt

    visited: set[Any] = set()
    followed_call_results: set[tuple[Any, int]] = set()
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, mlir_ir_rt.BlockArgument):
            continue
        op = v.owner
        if op not in visited:
            visited.add(op)
            for nested in _all_ops_in_region_tree(op):
                if nested not in visited:
                    visited.add(nested)
                    stack.extend(nested.operands)
                    if _call_callee_sym_name(nested) is not None:
                        stack.extend(nested.results)
            stack.extend(op.operands)

        callee_name = _call_callee_sym_name(op)
        if callee_name is not None:
            result_index = v.result_number
            follow_key = (op, result_index)
            if follow_key not in followed_call_results:
                followed_call_results.add(follow_key)
                callee_func = _resolve_func_by_sym_name(module, callee_name)
                callee_returns = _entry_block_ops(callee_func)[-1].operands
                if result_index >= len(callee_returns):
                    raise ValueError(
                        f"call @{callee_name} result {result_index} has no matching "
                        f"return operand ({len(callee_returns)} returned)"
                    )
                stack.append(callee_returns[result_index])

    return visited


def _result_info_strings(func_op: Any) -> list[str]:  # noqa: ANN401
    from jaxlib.mlir import ir as mlir_ir_rt

    res_attrs = func_op.operation.attributes["res_attrs"]
    infos = []
    for i in range(len(res_attrs)):
        entry = mlir_ir_rt.DictAttr(res_attrs[i])
        infos.append(mlir_ir_rt.StringAttr(entry["jax.result_info"]).value)
    return infos


def _compute_probe_slices(
    exported: Any,  # noqa: ANN401
    probe_result_paths: Mapping[str, str],
) -> tuple[dict[str, frozenset[int]], int]:
    """Backward slice (as a set of stable op indices) for every declared probe.

    Opens its own ``jaxlib.mlir.ir.Context`` around
    ``exported.mlir_module(serialized=False)`` and does all IR walking before
    returning -- the live IR objects do not survive the context exiting, so
    only plain ints/strings leave this function.
    """
    from jaxlib.mlir import ir as mlir_ir_rt
    from jaxlib.mlir.dialects import stablehlo as mlir_stablehlo_rt

    with mlir_ir_rt.Context() as ctx:
        mlir_stablehlo_rt.register_dialect(ctx)
        module = exported.mlir_module(serialized=False)
        func_op = _module_entry_func(module)
        result_infos = _result_info_strings(func_op)
        operands = list(_entry_block_ops(func_op)[-1].operands)
        all_ops = _all_module_ops(module)
        op_index = {op: i for i, op in enumerate(all_ops)}

        slices: dict[str, frozenset[int]] = {}
        for name, target in probe_result_paths.items():
            try:
                idx = result_infos.index(target)
            except ValueError:
                raise ValueError(
                    f"no return operand with jax.result_info == {target!r} for probe "
                    f"{name!r}; available result_info strings: {result_infos}"
                ) from None
            ops = _backward_slice_ops(operands[idx], module)
            slices[name] = frozenset(op_index[op] for op in ops)

        return slices, len(all_ops)


def probe_resolution(
    exported: Any,  # noqa: ANN401
    probe_deps: Mapping[str, tuple[str, ...]],
    probe_result_paths: Mapping[str, str],
) -> ProbeResolution:
    """Backward-slice resolution over a declared probe DAG (SS5.5b, T3, AC-9).

    For each declared edge ``predecessor -> probe``, reports
    ``slice_delta = |slice(probe) \\ slice(predecessor)|`` -- the ops needed
    for ``probe`` that were not already needed for ``predecessor`` -- via a
    def-use walk over ``exported.mlir_module(serialized=False)``.

    Three caveats (SS5.5b), stated here since this is a heuristic proxy for
    localization resolution, not a bound:

    1. Sibling slices **overlap** (shared subexpressions belong to both), so
       slice deltas do not partition the module and do not sum to the total.
    2. Counts are **static**. A loop body (e.g. a ``jax.lax.scan``) executing
       L times counts its ops once, not L times -- so the most heavily
       executed region can look like the best covered.
    3. It measures the **instrumented** module's StableHLO, *before* IREE's
       own fusion and DCE -- which is where the divergence actually lives.

    Args:
        exported: A ``jax.export.Exported`` for the instrumented step
            function.
        probe_deps: Probe name -> tuple of immediate predecessor probe names.
        probe_result_paths: Probe name -> the exact ``jax.result_info``
            string (as embedded by ``jax.export`` on the function's return
            attributes, e.g. ``"result['rbf']"`` or ``"result.rbf"``)
            identifying that probe's return operand. Resolving *which*
            result_info string belongs to which probe name is T7's job
            (``rings.py``'s ``out_tree`` name recovery); this function only
            consumes the resolved mapping.

    Returns:
        A :class:`ProbeResolution`.
    """
    all_names = (
        set(probe_result_paths)
        | {p for preds in probe_deps.values() for p in preds}
        | set(probe_deps)
    )
    missing = all_names - set(probe_result_paths)
    if missing:
        raise ValueError(f"probe_result_paths is missing entries for {sorted(missing)!r}")

    slice_ops, total_ops = _compute_probe_slices(exported, probe_result_paths)

    slice_delta: dict[str, int] = {}
    covered: set[int] = set()
    for probe, preds in probe_deps.items():
        for pred in preds:
            diff = slice_ops[probe] - slice_ops[pred]
            slice_delta[f"{pred}->{probe}"] = len(diff)
            covered |= diff

    return ProbeResolution(
        slice_delta=slice_delta,
        unattributed_ops=total_ops - len(covered),
        total_ops=total_ops,
    )


def validate_probe_deps(
    exported: Any,  # noqa: ANN401
    probe_deps: Mapping[str, tuple[str, ...]],
    probe_result_paths: Mapping[str, str],
) -> None:
    """Validate every declared ``probe_deps`` edge via the slice-subset check (SS5.3, T3b, AC-15).

    For a declared edge ``p -> q``, ``slice(p)`` must be a **subset** of
    ``slice(q)``: if ``q`` truly depends on ``p``, every op needed for ``p``
    is needed for ``q``. A failing edge is rejected with a
    :class:`ProbeDependencyError` naming both slices.

    The check is **one-directional** -- no false rejections (backward slices
    are transitively closed, and pre-optimization StableHLO preserves
    sharing) -- but has three false-accept modes it must not be sold as
    covering:

    - a **transitive** edge ``p -> r`` passes when the truth is
      ``p -> q -> r``;
    - a **passthrough** probe with an empty slice is a subset of everything;
    - a **missing** edge is invisible -- this validates declared edges, never
      completeness.

    Args:
        exported: A ``jax.export.Exported`` for the instrumented step
            function.
        probe_deps: Probe name -> tuple of immediate predecessor probe names.
        probe_result_paths: Probe name -> ``jax.result_info`` string (see
            :func:`probe_resolution`).

    Raises:
        ProbeDependencyError: A declared edge fails the slice-subset check.
    """
    slice_ops, _total_ops = _compute_probe_slices(exported, probe_result_paths)
    for probe, preds in probe_deps.items():
        for pred in preds:
            if pred not in slice_ops or probe not in slice_ops:
                raise ValueError(
                    f"probe_result_paths is missing an entry for {pred!r} or {probe!r}"
                )
            if not slice_ops[pred] <= slice_ops[probe]:
                raise ProbeDependencyError(pred, probe, slice_ops[pred], slice_ops[probe])
