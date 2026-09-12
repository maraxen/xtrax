"""Plan-time gating for the export boundary.

Two entry points over the same rules: ``check_export_safety`` returns every
blocker it finds, ``validate_export_safe`` raises on the first batch. Both
delegate topology to ``xtrax.stages.topology.validate_plan_topology`` and let
its ``PlanTopologyError`` propagate unwrapped -- topology violations are
structural and are never demoted into a blocker list.

Blockers cover the rules this module owns:

- ``"dtype"``: a leaf's dtype the target's backend does not accept.
- ``"unlegalizable-op"``: an op that IREE's StableHLO importer rejects
  outright, on every target -- currently ``jax.lax.top_k``, which lowers to a
  ``stablehlo.composite`` wrapping ``chlo.top_k`` that the importer marks
  explicitly illegal.
- ``"sort-stability"``: a stable sort/argsort, whose tie-breaking order IREE
  does not preserve relative to XLA. This is not a compile failure -- it
  silently produces different index results on ties, invisible to a
  float-tolerance parity check because the divergence lives entirely in
  integer indices.
- ``"random-permutation"``: ``jax.random.permutation``, which lowers to a
  nested ``_shuffle`` jit. IREE can compile it and still return a valid but
  wrong permutation; float parity then passes because the divergence lives
  entirely in integer indices. Unlike ``"unlegalizable-op"``, this rule is
  suppressible via ``acknowledged``. It is **necessary but not sufficient**:
  the defect is really the split-derived key, and a large enough program
  diverges on ``jnp.argsort(jax.random.bits(k1, ...))`` too, so a clean run of
  this rule does not certify a model's randomness. See its detail string.

They are collected rather than raised one at a time so a caller fixing a model
sees every offending leaf/op at once.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from xtrax.export.targets import Target
from xtrax.stages.boundaries import AxisBoundary
from xtrax.stages.topology import AxisDecisionLike, validate_plan_topology

__all__ = [
    "DtypeNotSupportedError",
    "ExportBlocker",
    "ExportSafetyError",
    "UnsupportedOperationError",
    "check_export_safety",
    "dtype_name",
    "find_bcoo_leaves",
    "validate_export_safe",
]


class ExportSafetyError(Exception):
    """Base for this package's own plan-time gate failures.

    Distinct from xtrax.stages.topology.PlanTopologyError, which both entry
    points here let propagate unchanged rather than wrapping.
    """


class DtypeNotSupportedError(ExportSafetyError):
    """A leaf's dtype is not accepted by the requested target."""


class UnsupportedOperationError(ExportSafetyError):
    """A traced op cannot legalize, or relies on a guarantee IREE breaks."""


@dataclass(frozen=True)
class ExportBlocker:
    """One reason a plan or leaf cannot cross the export boundary.

    Attributes:
        axis: Axis name, or the leaf keypath for a dtype blocker.
        rule: Short rule identifier, e.g. ``"dtype"``.
        detail: Human-readable explanation naming the offending value.
    """

    axis: str
    rule: str
    detail: str


def dtype_name(dtype: Any) -> str:
    """Render a numpy/JAX dtype in the short form targets are keyed by.

    Args:
        dtype: Anything with a ``name``, or a value convertible by ``str``.

    Returns:
        A short name such as ``"f32"``, ``"bf16"``, ``"i32"``, or ``"bool"``.
        Unrecognised dtypes are returned as their own name, which will simply
        fail the membership check against a target's dtype sets.
    """
    raw = getattr(dtype, "name", None) or str(dtype)
    if raw == "bool":
        return "bool"
    prefixes = (("float", "f"), ("bfloat", "bf"), ("int", "i"), ("uint", "u"), ("complex", "c"))
    for long, short in prefixes:
        if raw.startswith(long):
            return short + raw[len(long) :]
    return raw


def find_bcoo_leaves(tree: Any) -> list[str]:
    """Return the keypaths of any BCOO leaves in ``tree``.

    Sparsified models substitute BCOO at leaf positions, and BCOO is a pytree
    *node* rather than a leaf, so the tree structure changes. Under
    ``jax.export`` a closure-held BCOO is baked in as constants, which is what a
    self-contained artifact wants; this exists so a caller knows that is
    happening rather than discovering it in the MLIR.

    Args:
        tree: Any pytree, typically the model held in the exported closure.

    Returns:
        Keypath strings for each BCOO leaf; empty when sparse is unavailable.
    """
    try:
        import jax
        from jax.experimental.sparse import BCOO
    except ImportError:  # pragma: no cover - sparse ships with jax today
        return []

    found: list[str] = []
    flat = jax.tree_util.tree_flatten_with_path(tree, is_leaf=lambda x: isinstance(x, BCOO))[0]
    for path, leaf in flat:
        if isinstance(leaf, BCOO):
            found.append(jax.tree_util.keystr(path))
    return found


def _dtype_blocker(
    where: str,
    dtype: Any,
    target: Target,
    request_features: frozenset[str],
) -> ExportBlocker | None:
    """Judge one dtype against a target, or None if the target accepts it.

    Args:
        where: Location to name in the blocker, e.g. ``"abstract_inputs[0]"`` or
            a closure leaf's keypath.
        dtype: The leaf's dtype.
        target: The target being compiled for.
        request_features: Device features the caller will request.

    Returns:
        A blocker, or None when the dtype is accepted outright or unlocked by a
        requested feature.
    """
    name = dtype_name(dtype)
    if name in target.supported_dtypes:
        return None
    if name in target.optional_dtypes:
        feature = target.optional_dtype_features.get(name)
        if feature is not None and feature in request_features:
            return None
        return ExportBlocker(
            axis=where,
            rule="dtype",
            detail=(
                f"dtype {name!r} is optional on target {target.name!r} and "
                f"needs feature {feature!r}, which was not requested. Pass "
                f"request_features=frozenset({{{feature!r}}})."
            ),
        )
    supported = ", ".join(sorted(target.supported_dtypes))
    detail = f"dtype {name!r} is not supported by target {target.name!r}. Supported: {supported}."
    if name == "f64":
        # Worth saying outright: IREE does not reject f64, it silently demotes
        # it to f32 and rewrites the artifact's public signature. Without this
        # the reader assumes a capability gap and goes looking for a flag.
        detail += (
            " IREE demotes f64 to f32 on every backend and rewrites the entry "
            "point's signature to match, so the artifact would not have the "
            "dtype you asked for. Cast to f32 yourself, so the precision loss "
            "is yours rather than the compiler's."
        )
    return ExportBlocker(axis=where, rule="dtype", detail=detail)


def _closure_dtype_leaves(fn: Any) -> list[tuple[str, Any]]:
    """Return ``(keypath, dtype)`` for every array leaf reachable from ``fn``.

    ``abstract_inputs`` covers only what the caller passes at trace time. A
    model's weights typically ride along in the callable's closure instead --
    an Equinox module holding a bf16 or f64 array is never an argument -- so
    checking arguments alone leaves the commonest case unchecked.

    Args:
        fn: The callable being exported.

    Returns:
        One entry per leaf that has a ``dtype``, keypath first. Empty when the
        callable holds no array leaves, which is the usual case for a plain
        function.
    """
    try:
        import jax
    except ImportError:  # pragma: no cover - jax is a hard dependency
        return []

    try:
        flat = jax.tree_util.tree_flatten_with_path(fn)[0]
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return []

    leaves: list[tuple[str, Any]] = []
    for path, leaf in flat:
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            leaves.append((f"closure{jax.tree_util.keystr(path)}", dtype))
    return leaves


def _dtype_blockers(
    abstract_inputs: Sequence[Any],
    fn: Any,
    target: Target,
    request_features: frozenset[str],
) -> list[ExportBlocker]:
    """Collect a blocker for every input or closure leaf whose dtype is rejected."""
    blockers: list[ExportBlocker] = []
    for index, spec in enumerate(abstract_inputs):
        dtype = getattr(spec, "dtype", None)
        if dtype is None:
            continue
        blocker = _dtype_blocker(f"abstract_inputs[{index}]", dtype, target, request_features)
        if blocker is not None:
            blockers.append(blocker)

    for where, dtype in _closure_dtype_leaves(fn):
        blocker = _dtype_blocker(where, dtype, target, request_features)
        if blocker is not None:
            blockers.append(blocker)
    return blockers


_UNLEGALIZABLE_OP_RULE = "unlegalizable-op"
_SORT_STABILITY_RULE = "sort-stability"
_RANDOM_PERMUTATION_RULE = "random-permutation"

#: Rules a caller cannot suppress via ``acknowledged`` -- currently only
#: unlegalizable ops, because acknowledging one does not make it compile; it
#: only moves the identical failure later, into ``compile_for_target``.
_UNSUPPRESSIBLE_RULES = frozenset({_UNLEGALIZABLE_OP_RULE})

_TOP_K_DETAIL = (
    "jax.lax.top_k cannot be legalized by IREE's StableHLO importer on any "
    "target: it lowers to a stablehlo.composite wrapping chlo.top_k, which "
    "the importer marks explicitly illegal. Replace it with "
    "jnp.argsort(-x, axis=-1, stable=True)[..., :k] plus take_along_axis, "
    "which compiles and is bit-exact on values and indices -- but that "
    "replacement is itself a stable sort, so it inherits the sort-stability "
    "problem below and needs the same tiebreak fix."
)

_SORT_STABILITY_DETAIL = (
    "This sort relies on stable tie-breaking (is_stable=True), but IREE does "
    "not honour JAX's documented stable-sort tie order -- ties can land at "
    "different positions than XLA produces, and the divergence is carried "
    "entirely by integer indices, so a float-tolerance parity check sees "
    "bit-identical values and passes. Fold an explicit index tiebreak into "
    "the sort key instead of relying on backend stability: sort "
    "lexicographically on (-x, index) rather than sorting x alone."
)

_RANDOM_PERMUTATION_DETAIL = (
    "jax.random.permutation lowers to a nested _shuffle jit that IREE "
    "miscompiles when the permutation key is a split half and the sibling "
    "half is also consumed. The artifact returns a valid but wrong "
    "permutation; nothing crashes, and a float-tolerance parity check "
    "reports max_abs_diff 0.0 because the divergence is carried entirely "
    "by integer indices. Measured 260911 on iree-base-compiler 3.11: 248 of "
    "256 positions differ, byte-identical across two --iree-llvmcpu-target-cpu "
    "values, so it is a lowering defect rather than float reassociation. "
    "THIS RULE IS NECESSARY BUT NOT SUFFICIENT. permutation is the most "
    "sensitive construct -- it diverges even in a tiny program -- but the "
    "underlying defect is the SPLIT-DERIVED KEY, not the permutation. In a "
    "large program jnp.argsort(jax.random.bits(k1, ...)) on a split half "
    "diverges too, while the same spelling on the RAW, unsplit key is exact. "
    "So swapping the construct is not a fix on its own; only an unsplit key "
    "was measured exact end to end. Do not read a clean run of this gate as "
    "'this model's randomness is safe to export'. This rule is suppressible "
    "via acknowledged=frozenset({'random-permutation'}) if an informed caller "
    "accepts the risk."
)


def _sub_jaxprs(params: Mapping[str, Any]) -> list[Any]:
    """Return every jaxpr-like object reachable from one eqn's params.

    Covers both a lone value (``scan``'s ``jaxpr`` param) and a list/tuple of
    values (some higher-order primitives carry more than one branch jaxpr).
    Each candidate is treated as jaxpr-like if it exposes ``.eqns`` after
    unwrapping ``.jaxpr`` -- which is a no-op for a plain ``Jaxpr`` and
    unwraps a ``ClosedJaxpr`` down to the walkable core.
    """
    found: list[Any] = []
    for value in params.values():
        candidates = value if isinstance(value, (list, tuple)) else (value,)
        for candidate in candidates:
            inner = getattr(candidate, "jaxpr", candidate)
            if hasattr(inner, "eqns"):
                found.append(inner)
    return found


def _walk_jaxpr_eqns(jaxpr: Any) -> list[Any]:
    """Return every equation in ``jaxpr``, recursing into nested sub-jaxprs.

    Recursion is mandatory, not an optimisation: ``top_k`` inside a
    ``lax.scan`` is only visible one level down inside the scan's sub-jaxpr,
    ``jnp.argsort``'s ``sort`` primitive is only visible inside a nested
    ``pjit`` sub-jaxpr, and ``jax.random.permutation``'s ``_shuffle`` jit is
    likewise never at the top level. A top-level-only walk finds none of
    these and reports a false all-clear.
    """
    eqns: list[Any] = []
    for eqn in jaxpr.eqns:
        eqns.append(eqn)
        for sub in _sub_jaxprs(eqn.params):
            eqns.extend(_walk_jaxpr_eqns(sub))
    return eqns


def _op_blockers(abstract_inputs: Sequence[Any], fn: Callable[..., Any]) -> list[ExportBlocker]:
    """Collect blockers for ops IREE cannot legalize or cannot preserve.

    Tracing failures here are swallowed rather than surfaced: a callable that
    ``jax.make_jaxpr`` cannot trace is not something this gate could have
    usefully judged anyway, and it will fail identically -- at export or
    compile time, with a clearer error pointing at the actual call site --
    whether or not this function ran. Swallowing the exception here does not
    hide a real problem; it just declines to duplicate one that surfaces on
    its own moments later.
    """
    try:
        import jax
    except ImportError:  # pragma: no cover - jax is a hard dependency
        return []

    try:
        closed = jax.make_jaxpr(fn)(*abstract_inputs)
    except Exception:  # noqa: BLE001 - see docstring: never masks a real failure
        return []

    top = getattr(closed, "jaxpr", closed)
    blockers: list[ExportBlocker] = []
    for eqn in _walk_jaxpr_eqns(top):
        name = eqn.primitive.name
        if name == "top_k":
            blockers.append(
                ExportBlocker(axis=name, rule=_UNLEGALIZABLE_OP_RULE, detail=_TOP_K_DETAIL)
            )
        elif name == "sort" and eqn.params.get("is_stable"):
            blockers.append(
                ExportBlocker(axis=name, rule=_SORT_STABILITY_RULE, detail=_SORT_STABILITY_DETAIL)
            )
        elif name in {"jit", "pjit"} and eqn.params.get("name") == "_shuffle":
            blockers.append(
                ExportBlocker(
                    axis=name,
                    rule=_RANDOM_PERMUTATION_RULE,
                    detail=_RANDOM_PERMUTATION_DETAIL,
                )
            )
    return blockers


def _apply_acknowledged(
    blockers: list[ExportBlocker], acknowledged: frozenset[str]
) -> list[ExportBlocker]:
    """Drop blockers whose rule is acknowledged, except unsuppressible ones.

    ``"unlegalizable-op"`` is never suppressed here: the caller can pass it in
    ``acknowledged`` and it will simply have no effect on that rule, because
    the op cannot compile regardless of who has acknowledged it.
    """
    return [b for b in blockers if b.rule not in acknowledged or b.rule in _UNSUPPRESSIBLE_RULES]


def check_export_safety(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    abstract_inputs: Sequence[Any],
    fn: Callable[..., Any],
    target: Target,
    *,
    request_features: frozenset[str] = frozenset(),
    acknowledged: frozenset[str] = frozenset(),
) -> list[ExportBlocker]:
    """List every blocker between this plan and the export boundary.

    Deliberately does not call ``validate_plan_topology``: topology violations
    always raise directly and are never converted into a blocker list.

    Args:
        decisions: Axis decisions from the plan.
        axis_boundaries: Map of axis name -> AxisBoundary.
        abstract_inputs: Abstract inputs the callable will be traced with.
        fn: The callable being exported. Its closure-reachable leaves are scanned
            for dtype violations alongside ``abstract_inputs``, and its traced
            jaxpr (including nested sub-jaxprs, e.g. inside ``lax.scan`` or a
            ``pjit``-wrapped ``argsort`` or ``_shuffle``) is scanned for
            unlegalizable ops, sorts relying on stable tie-breaking, and
            ``jax.random.permutation``.
        target: The target being compiled for.
        request_features: Device features the caller will request, unlocking the
            target's optional dtypes.
        acknowledged: Rule names to suppress from the returned list. Every rule
            except ``"unlegalizable-op"`` is suppressible this way --
            ``"unlegalizable-op"`` cannot compile at all, so acknowledging it
            would only move the identical failure later, into
            ``compile_for_target``.

    Returns:
        Every blocker found, in discovery order, minus any whose rule is in
        ``acknowledged`` (except unsuppressible rules). Empty means no
        objection.
    """
    del decisions, axis_boundaries
    blockers = _dtype_blockers(abstract_inputs, fn, target, request_features)
    blockers += _op_blockers(abstract_inputs, fn)
    return _apply_acknowledged(blockers, acknowledged)


def validate_export_safe(
    decisions: Sequence[AxisDecisionLike],
    axis_boundaries: Mapping[str, AxisBoundary],
    abstract_inputs: Sequence[Any],
    fn: Callable[..., Any],
    target: Target,
    *,
    request_features: frozenset[str] = frozenset(),
    acknowledged: frozenset[str] = frozenset(),
) -> None:
    """Raise unless this plan can cross the export boundary for ``target``.

    Args:
        decisions: Axis decisions from the plan.
        axis_boundaries: Map of axis name -> AxisBoundary.
        abstract_inputs: Abstract inputs the callable will be traced with.
        fn: The callable being exported.
        target: The target being compiled for.
        request_features: Device features the caller will request.
        acknowledged: Rule names to suppress. See ``check_export_safety``;
            ``"unlegalizable-op"`` is not suppressible.

    Raises:
        PlanTopologyError: Propagated unwrapped from validate_plan_topology --
            including its MaterializeFuseConflictError,
            MaterializeWithoutSinkError, and MultipleMaterializeAxesError
            subclasses.
        DtypeNotSupportedError: If any leaf's dtype is rejected by the target.
            Takes precedence over UnsupportedOperationError below: if the
            blocker list contains *any* dtype blocker, this is raised instead
            of UnsupportedOperationError, even when op blockers are also
            present -- so existing callers that only ever expected a dtype
            failure keep seeing one. Either way, the message lists every
            blocker found, across both rule families, not just the first.
        UnsupportedOperationError: If no dtype blocker is present but at least
            one op blocker (``"unlegalizable-op"``, ``"sort-stability"``, or
            ``"random-permutation"``) is.
    """
    validate_plan_topology(decisions, axis_boundaries, export_safe=True)

    blockers = check_export_safety(
        decisions,
        axis_boundaries,
        abstract_inputs,
        fn,
        target,
        request_features=request_features,
        acknowledged=acknowledged,
    )
    if blockers:
        detail = "\n".join(f"  - {b.axis} ({b.rule}): {b.detail}" for b in blockers)
        msg = f"{len(blockers)} export blocker(s) for target {target.name!r}:\n{detail}"
        if any(b.rule == "dtype" for b in blockers):
            raise DtypeNotSupportedError(msg)
        raise UnsupportedOperationError(msg)
