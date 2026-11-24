"""Parity helpers for kernels behind :func:`xtrax.op`.

A kernel is compared to its plain-JAX reference. Forward outputs use exact
equality for integer and bool leaves and the scale-relative tolerance
``rel * max(1, max|ref|)`` for floats (an elementwise ``1e-5`` fails at
``|ref| ~ 150`` from float32 accumulation order). A backward kernel is compared
to ``jax.vjp(reference)`` under a random output cotangent, per-argument
relative L2 error. An all-ones cotangent (``grad(sum(out))``) hides sign and
permutation errors, so it is not offered.

A green parity check proves nothing without a control that turns it red. Use
:func:`require_corruption_red` and :func:`require_vjp_controls_red`.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.kernel_seam import VjpKernel
from xtrax.testing.lanes import NegativeControlError

__all__ = [
    "KernelParityError",
    "ParityResult",
    "ProbeParityPayload",
    "VjpParityResult",
    "assert_kernel_parity",
    "assert_kernel_ran",
    "assert_vjp_parity",
    "build_probe_parity_payload",
    "compare_outputs",
    "dropped_gradient_control",
    "evaluate_kernel_parity_for_probe",
    "permuted_weight_control",
    "require_corruption_red",
    "require_vjp_controls_red",
    "scale_error_control",
    "scale_relative_atol",
    "vjp_parity",
]


class KernelParityError(AssertionError):
    """A kernel disagrees with its reference."""


@dataclass(frozen=True)
class ParityResult:
    """Outcome of comparing kernel outputs to reference outputs.

    Attributes
    ----------
    ok : bool
        True when structures match, every integer or bool leaf is equal, and
        every float leaf is within its scale-relative tolerance.
    max_abs_err : float
        Largest absolute difference over float leaves (0.0 when there are none).
    int_mismatches : int
        Number of integer or bool elements that differ.
    detail : str
        Human-readable reason when ``ok`` is False, otherwise ``""``.
    """

    ok: bool
    max_abs_err: float
    int_mismatches: int
    detail: str


@dataclass(frozen=True)
class VjpParityResult:
    """Outcome of comparing a backward kernel to ``jax.vjp(reference)``.

    Attributes
    ----------
    ok : bool
        True when the forward output matches and every float argument's
        relative L2 error is at most ``rtol``.
    rel_l2 : tuple of float or None
        Per-argument ``||kernel - reference|| / ||reference||``. ``None`` for
        integer and bool arguments, which have no cotangent.
    forward : ParityResult
        Comparison of the forward kernel output.
    rtol : float
        Threshold applied to each ``rel_l2``.
    """

    ok: bool
    rel_l2: tuple[float | None, ...]
    forward: ParityResult
    rtol: float


def scale_relative_atol(ref: Any, rel: float = 1e-5) -> float:
    """Absolute tolerance for a float reference: ``rel * max(1, max|ref|)``.

    Parameters
    ----------
    ref : array-like
        Reference values.
    rel : float
        Relative factor.

    Returns
    -------
    float
        The tolerance.
    """
    arr = np.asarray(ref, dtype=np.float64)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    return rel * max(1.0, peak)


def _is_inexact(x: Any) -> bool:
    """True when ``x`` has an inexact dtype."""
    return bool(jnp.issubdtype(jnp.result_type(x), jnp.inexact))


def compare_outputs(got: Any, want: Any, *, rel: float = 1e-5) -> ParityResult:
    """Compare two output pytrees.

    Parameters
    ----------
    got : pytree
        Kernel output.
    want : pytree
        Reference output.
    rel : float
        Relative factor of the float tolerance (see :func:`scale_relative_atol`).

    Returns
    -------
    ParityResult
        Exact equality for integer and bool leaves, scale-relative tolerance for
        floats, and a failure on any structure, shape or dtype difference.
    """
    got_leaves, got_def = jax.tree.flatten(got)
    want_leaves, want_def = jax.tree.flatten(want)
    if got_def != want_def:
        return ParityResult(False, float("inf"), 0, f"structure {got_def} != {want_def}")
    max_err = 0.0
    int_bad = 0
    detail: list[str] = []
    for i, (g, w) in enumerate(zip(got_leaves, want_leaves, strict=True)):
        g, w = np.asarray(g), np.asarray(w)
        if g.shape != w.shape or g.dtype != w.dtype:
            detail.append(f"leaf {i}: {g.dtype}{g.shape} != {w.dtype}{w.shape}")
            continue
        if _is_inexact(w):
            err = float(np.max(np.abs(g.astype(np.float64) - w.astype(np.float64)), initial=0.0))
            max_err = max(max_err, err)
            if not err <= scale_relative_atol(w, rel):
                detail.append(f"leaf {i}: max|diff| {err:.3e} > {scale_relative_atol(w, rel):.3e}")
        else:
            bad = int(np.sum(g != w))
            int_bad += bad
            if bad:
                detail.append(f"leaf {i}: {bad} integer elements differ")
    return ParityResult(not detail, max_err, int_bad, "; ".join(detail))


def assert_kernel_parity(
    kernel: Callable[..., Any], reference: Callable[..., Any], *args: Any, rel: float = 1e-5
) -> ParityResult:
    """Assert that ``kernel(*args)`` matches ``reference(*args)``.

    Parameters
    ----------
    kernel : Callable
        Kernel under test.
    reference : Callable
        Plain-JAX reference.
    args : Any
        Positional arguments for both.
    rel : float
        Relative factor of the float tolerance.

    Returns
    -------
    ParityResult
        The passing result.

    Raises
    ------
    KernelParityError
        The outputs differ.
    """
    result = compare_outputs(kernel(*args), reference(*args), rel=rel)
    if not result.ok:
        raise KernelParityError(result.detail)
    return result


def assert_kernel_ran(fn: Callable[..., Any], *args: Any) -> None:
    """Assert that tracing ``fn(*args)`` contains a ``pallas_call``.

    A kernel whose output equals the reference is often a silent fallback, so
    every parity test should also show the kernel path was taken.

    Parameters
    ----------
    fn : Callable
        Function expected to contain the kernel (an op under a Context that
        selects it).
    args : Any
        Positional arguments to trace with.

    Returns
    -------
    None

    Raises
    ------
    KernelParityError
        No ``pallas_call`` appears in the jaxpr.
    """
    # A fresh lambda: make_jaxpr caches by callable, so a function traced earlier would
    # otherwise return its stale jaxpr.
    if "pallas_call" not in str(jax.make_jaxpr(lambda *a: fn(*a))(*args)):
        raise KernelParityError("no pallas_call in the traced jaxpr: the kernel did not run")


def require_corruption_red(
    kernel: Callable[..., Any],
    reference: Callable[..., Any],
    args: tuple[Any, ...],
    corrupt: Callable[[Callable[..., Any]], Callable[..., Any]],
    *,
    rel: float = 1e-5,
) -> ParityResult:
    """Require that a corrupted copy of ``kernel`` fails parity.

    Parameters
    ----------
    kernel : Callable
        Kernel under test.
    reference : Callable
        Plain-JAX reference.
    args : tuple
        Positional arguments for both.
    corrupt : Callable
        Maps a kernel to a deliberately broken kernel (for example, one that
        perturbs one block of the output).
    rel : float
        Relative factor of the float tolerance.

    Returns
    -------
    ParityResult
        The failing result of the corrupted kernel.

    Raises
    ------
    NegativeControlError
        The corrupted kernel still passes, so the parity check cannot fail.
    """
    result = compare_outputs(corrupt(kernel)(*args), reference(*args), rel=rel)
    if result.ok:
        raise NegativeControlError("corrupted kernel still matches the reference")
    return result


def _norm(x: Any) -> float:
    """Float64 L2 norm."""
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def _random_cotangent(out: Any, key: jax.Array) -> Any:
    """Normal cotangent for float leaves, ``float0`` zeros for the rest."""
    leaves, treedef = jax.tree.flatten(out)
    keys = jax.random.split(key, max(len(leaves), 1))
    cts = [
        jax.random.normal(k, leaf.shape, leaf.dtype)
        if _is_inexact(leaf)
        else np.zeros(leaf.shape, jax.dtypes.float0)
        for k, leaf in zip(keys, leaves, strict=False)
    ]
    return jax.tree.unflatten(treedef, cts)


def vjp_parity(
    vk: VjpKernel,
    reference: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    key: jax.Array,
    rtol: float = 1e-5,
    rel: float = 1e-5,
) -> VjpParityResult:
    """Compare a backward kernel to ``jax.vjp(reference)`` under a random cotangent.

    Parameters
    ----------
    vk : VjpKernel
        Forward and backward kernel pair.
    reference : Callable
        Plain-JAX reference.
    args : tuple
        Positional arguments. Use randomised weights: reference modules that
        zero-initialise output weights make parity on a fresh module vacuous.
    key : jax.Array
        PRNG key for the output cotangent.
    rtol : float
        Threshold on each argument's relative L2 error.
    rel : float
        Relative factor of the forward float tolerance.

    Returns
    -------
    VjpParityResult
        Forward comparison and per-argument relative L2 errors.
    """
    out, residuals = vk.fwd(*args)
    forward = compare_outputs(out, reference(*args), rel=rel)
    ct = _random_cotangent(out, key)
    got = tuple(vk.bwd(residuals, ct))
    want = jax.vjp(reference, *args)[1](ct)
    if len(got) != len(args):
        raise KernelParityError(f"bwd returned {len(got)} cotangents for {len(args)} arguments")
    errs: list[float | None] = []
    for arg, g, w in zip(args, got, want, strict=True):
        if not _is_inexact(arg):
            errs.append(None)
            continue
        scale = _norm(w)
        diff = _norm(np.asarray(g, np.float64) - np.asarray(w, np.float64))
        errs.append(diff / scale if scale > 0.0 else (0.0 if diff == 0.0 else float("inf")))
    ok = forward.ok and all(e is None or e <= rtol for e in errs)
    return VjpParityResult(ok, tuple(errs), forward, rtol)


def assert_vjp_parity(
    vk: VjpKernel,
    reference: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    key: jax.Array,
    rtol: float = 1e-5,
) -> VjpParityResult:
    """Assert :func:`vjp_parity` passes.

    Parameters
    ----------
    vk : VjpKernel
        Forward and backward kernel pair.
    reference : Callable
        Plain-JAX reference.
    args : tuple
        Positional arguments.
    key : jax.Array
        PRNG key for the output cotangent.
    rtol : float
        Threshold on each argument's relative L2 error.

    Returns
    -------
    VjpParityResult
        The passing result.

    Raises
    ------
    KernelParityError
        The forward output or any cotangent differs.
    """
    result = vjp_parity(vk, reference, args, key=key, rtol=rtol)
    if not result.ok:
        raise KernelParityError(
            f"vjp parity failed: rel_l2={result.rel_l2} (rtol={rtol}); forward: "
            f"{result.forward.detail or 'ok'}"
        )
    return result


def scale_error_control(vk: VjpKernel, factor: float = 1.001) -> VjpKernel:
    """Return ``vk`` with every input cotangent scaled by ``factor`` (default 0.1% error).

    Parameters
    ----------
    vk : VjpKernel
        Kernel pair to corrupt.
    factor : float
        Multiplier applied to each float cotangent.

    Returns
    -------
    VjpKernel
        Corrupted pair.
    """

    def bwd(residuals: Any, ct: Any) -> tuple[Any, ...]:
        """Scale each float cotangent."""
        return tuple(
            c * factor if c is not None and _is_inexact(c) else c for c in vk.bwd(residuals, ct)
        )

    return VjpKernel(vk.fwd, bwd)


def permuted_weight_control(vk: VjpKernel, argnum: int, args: tuple[Any, ...]) -> VjpKernel:
    """Return ``vk`` whose backward sees the weight reversed along axis 0.

    Every residual leaf with the weight's shape and dtype is reversed, so a
    backward that reads the weight from its residuals computes with permuted
    rows. The forward output is untouched, which isolates the backward.

    Parameters
    ----------
    vk : VjpKernel
        Kernel pair to corrupt.
    argnum : int
        Index of the weight argument.
    args : tuple
        The arguments the pair will run on, used to match residual leaves to
        the weight by shape and dtype.

    Returns
    -------
    VjpKernel
        Corrupted pair.
    """
    weight = args[argnum]

    def permute(leaf: Any) -> Any:
        """Reverse a residual leaf that holds the weight."""
        same = getattr(leaf, "shape", None) == weight.shape and leaf.dtype == weight.dtype
        return leaf[::-1] if same else leaf

    def bwd(residuals: Any, ct: Any) -> tuple[Any, ...]:
        """Run the backward on permuted residuals."""
        return vk.bwd(jax.tree.map(permute, residuals), ct)

    return VjpKernel(vk.fwd, bwd)


def dropped_gradient_control(vk: VjpKernel, argnum: int) -> VjpKernel:
    """Return ``vk`` whose backward returns zeros for argument ``argnum``.

    Parameters
    ----------
    vk : VjpKernel
        Kernel pair to corrupt.
    argnum : int
        Index of the argument whose cotangent is dropped (a weight gradient).

    Returns
    -------
    VjpKernel
        Corrupted pair.
    """

    def bwd(residuals: Any, ct: Any) -> tuple[Any, ...]:
        """Zero the dropped cotangent."""
        cts = list(vk.bwd(residuals, ct))
        cts[argnum] = jnp.zeros_like(cts[argnum])
        return tuple(cts)

    return VjpKernel(vk.fwd, bwd)


def require_vjp_controls_red(
    vk: VjpKernel,
    reference: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    weight_argnum: int,
    key: jax.Array,
    rtol: float = 1e-5,
) -> Mapping[str, VjpParityResult]:
    """Require the three standard negative controls to fail vjp parity.

    The controls are a 0.1% scale error, a permuted weight, and a dropped
    weight gradient.

    Parameters
    ----------
    vk : VjpKernel
        Kernel pair under test.
    reference : Callable
        Plain-JAX reference.
    args : tuple
        Positional arguments.
    weight_argnum : int
        Index of the weight argument the permute and drop controls target.
    key : jax.Array
        PRNG key for the output cotangent.
    rtol : float
        Threshold on each argument's relative L2 error.

    Returns
    -------
    Mapping[str, VjpParityResult]
        The failing result of each control, keyed ``scale``, ``permute`` and ``drop``.

    Raises
    ------
    NegativeControlError
        Any control still passes, so the gate cannot detect that error.
    """
    controls = {
        "scale": scale_error_control(vk),
        "permute": permuted_weight_control(vk, weight_argnum, args),
        "drop": dropped_gradient_control(vk, weight_argnum),
    }
    results = {
        name: vjp_parity(c, reference, args, key=key, rtol=rtol) for name, c in controls.items()
    }
    survivors = [name for name, r in results.items() if r.ok]
    if survivors:
        raise NegativeControlError(f"vjp controls still pass: {survivors}")
    return results


@dataclass(frozen=True, slots=True)
class ProbeParityPayload:
    """Numeric parity metrics and status tags formatted for ``ProbeRecord``."""

    metrics: dict[str, float]
    config_statuses: dict[str, str]


def _tree_max_abs(tree: Any) -> float:
    """Largest absolute numeric value across all array leaves in ``tree``."""
    peak = 0.0
    for leaf in jax.tree.flatten(tree)[0]:
        arr = np.asarray(leaf, dtype=np.float64)
        if arr.size:
            peak = max(peak, float(np.max(np.abs(arr))))
    return peak


def _tree_l2_norm(tree: Any) -> float:
    """Float64 L2 norm across all leaves of ``tree``."""
    total_sq = 0.0
    for leaf in jax.tree.flatten(tree)[0]:
        arr = np.asarray(leaf, dtype=np.float64)
        if arr.size:
            total_sq += float(np.sum(arr * arr))
    return math.sqrt(total_sq)


def _tree_shapes_match(got_leaves: list[Any], ref_leaves: list[Any]) -> bool:
    """Return True when two PyTree leaf lists have matching lengths and shapes."""
    if len(got_leaves) != len(ref_leaves):
        return False
    return all(
        np.asarray(g).shape == np.asarray(w).shape
        for g, w in zip(got_leaves, ref_leaves, strict=True)
    )


def _tree_rel_err(got_val: Any, ref_val: Any, *, int_mismatches: int) -> float:
    """Relative error aligned with ``scale_relative_atol`` across PyTree leaves."""
    if int_mismatches > 0:
        return float("inf")
    got_leaves, got_def = jax.tree.flatten(got_val)
    ref_leaves, ref_def = jax.tree.flatten(ref_val)
    if got_def != ref_def or not _tree_shapes_match(got_leaves, ref_leaves):
        return float("inf")
    worst = 0.0
    for g, w in zip(got_leaves, ref_leaves, strict=True):
        if not _is_inexact(w):
            continue
        g_arr = np.asarray(g, dtype=np.float64)
        w_arr = np.asarray(w, dtype=np.float64)
        if not np.all(np.isfinite(g_arr)) or not np.all(np.isfinite(w_arr)):
            return float("inf")
        if w_arr.size == 0:
            continue
        max_diff = float(np.max(np.abs(g_arr - w_arr)))
        denom = max(1.0, float(np.max(np.abs(w_arr))))
        worst = max(worst, max_diff / denom)
    return worst


def _corr_rel_err_for_corruption(
    *,
    corrupted_output: Any | None,
    corruption_result: ParityResult | None,
    reference_output: Any,
    corr_res: ParityResult,
) -> float:
    """Compute float relative divergence for a negative-control run."""
    if corruption_result is not None:
        return float(corr_res.max_abs_err) / max(1.0, _tree_max_abs(reference_output))
    got_leaves, got_def = jax.tree.flatten(corrupted_output)
    ref_leaves, ref_def = jax.tree.flatten(reference_output)
    if got_def != ref_def or not _tree_shapes_match(got_leaves, ref_leaves):
        return float("inf")
    return _tree_rel_err(corrupted_output, reference_output, int_mismatches=0)


def _filter_finite_metrics(
    raw_metrics: list[tuple[str, float, str]],
    config_statuses: dict[str, str],
) -> dict[str, float]:
    """Keep finite metrics and force corresponding status tags to ``'fail'`` on NaN/Inf."""
    metrics: dict[str, float] = {}
    bad_keys: list[str] = []
    for key, val, status_key in raw_metrics:
        if math.isfinite(val):
            metrics[key] = float(val)
        else:
            bad_keys.append(key)
            config_statuses[status_key] = "fail"
    if bad_keys:
        config_statuses["parity_nonfinite_metrics"] = ",".join(bad_keys)
    return metrics


def _evaluate_forward_non_degen(
    forward: ParityResult,
    got_output: Any,
    reference_output: Any,
) -> tuple[bool, float, float]:
    """Compute forward non-degeneracy flag, output L2 norm, and relative error."""
    got_leaves, got_def = jax.tree.flatten(got_output)
    ref_leaves, ref_def = jax.tree.flatten(reference_output)
    structure_ok = got_def == ref_def and _tree_shapes_match(got_leaves, ref_leaves)
    output_l2_norm = _tree_l2_norm(got_output)
    parity_rel_err = _tree_rel_err(
        got_output, reference_output, int_mismatches=forward.int_mismatches
    )
    non_degen_ok = (
        forward.ok
        and forward.int_mismatches == 0
        and structure_ok
        and math.isfinite(forward.max_abs_err)
        and math.isfinite(parity_rel_err)
        and math.isfinite(output_l2_norm)
        and output_l2_norm > 0.0
    )
    return non_degen_ok, output_l2_norm, parity_rel_err


def _evaluate_vjp_parity(
    vjp_result: VjpParityResult,
) -> tuple[str, tuple[str, float, str]]:
    """Summarize VJP parity into status tag and metric entry."""
    vjp_err = max((float(e) for e in vjp_result.rel_l2 if e is not None), default=0.0)
    vjp_ok = bool(vjp_result.ok) and math.isfinite(vjp_err)
    return (
        "pass" if vjp_ok else "fail",
        ("parity_vjp_rel_err", float(vjp_err), "vjp_parity_status"),
    )


def _evaluate_corruption_control(
    reference_output: Any,
    *,
    rel: float = 1e-4,
    corrupted_output: Any | None = None,
    corruption_result: ParityResult | None = None,
) -> tuple[str, tuple[str, float, str], float | None]:
    """Evaluate negative corruption control into status tag, metric tuple, and int mismatches."""
    corr_res = (
        corruption_result
        if corruption_result is not None
        else compare_outputs(corrupted_output, reference_output, rel=rel)
    )
    corr_rel_err = _corr_rel_err_for_corruption(
        corrupted_output=corrupted_output,
        corruption_result=corruption_result,
        reference_output=reference_output,
        corr_res=corr_res,
    )
    corr_ok = (not corr_res.ok) and (
        (math.isfinite(corr_rel_err) and corr_rel_err > rel) or corr_res.int_mismatches > 0
    )
    extra_int = float(corr_res.int_mismatches) if corr_res.int_mismatches > 0 else None
    return (
        "pass" if corr_ok else "fail",
        ("corruption_rel_err", float(corr_rel_err), "corruption_control_status"),
        extra_int,
    )


def build_probe_parity_payload(
    forward: ParityResult,
    got_output: Any,
    reference_output: Any,
    *,
    rel: float = 1e-4,
    vjp_result: VjpParityResult | None = None,
    corrupted_output: Any | None = None,
    corruption_result: ParityResult | None = None,
) -> ProbeParityPayload:
    """Convert ``kernel_parity`` checks into finite ``ProbeRecord`` metrics and status tags."""
    if corrupted_output is not None and corruption_result is not None:
        raise ValueError("pass at most one of corrupted_output or corruption_result")

    non_degen_ok, output_l2_norm, parity_rel_err = _evaluate_forward_non_degen(
        forward, got_output, reference_output
    )
    config_statuses: dict[str, str] = {
        "non_degeneracy_status": "pass" if non_degen_ok else "fail",
    }
    raw_metrics: list[tuple[str, float, str]] = [
        ("parity_max_abs_err", float(forward.max_abs_err), "non_degeneracy_status"),
        ("parity_rel_err", float(parity_rel_err), "non_degeneracy_status"),
        ("output_l2_norm", float(output_l2_norm), "non_degeneracy_status"),
    ]

    if vjp_result is not None:
        vjp_status, vjp_metric = _evaluate_vjp_parity(vjp_result)
        config_statuses["vjp_parity_status"] = vjp_status
        raw_metrics.append(vjp_metric)

    extra_int_mismatches: float | None = None
    if corrupted_output is not None or corruption_result is not None:
        corr_status, corr_metric, extra_int_mismatches = _evaluate_corruption_control(
            reference_output,
            rel=rel,
            corrupted_output=corrupted_output,
            corruption_result=corruption_result,
        )
        config_statuses["corruption_control_status"] = corr_status
        raw_metrics.append(corr_metric)
    else:
        config_statuses["corruption_control_status"] = "not_measured"

    metrics = _filter_finite_metrics(raw_metrics, config_statuses)
    if extra_int_mismatches is not None:
        metrics["corruption_int_mismatches"] = extra_int_mismatches
    return ProbeParityPayload(metrics=metrics, config_statuses=config_statuses)


def evaluate_kernel_parity_for_probe(
    kernel: Callable[..., Any],
    reference: Callable[..., Any],
    args: tuple[Any, ...],
    corrupt: Callable[[Callable[..., Any]], Callable[..., Any]],
    *,
    rel: float = 1e-4,
    vjp_kernel: VjpKernel | None = None,
    vjp_key: Any | None = None,
) -> ProbeParityPayload:
    """Execute forward, negative-control corruption, and optional VJP checks for a probe."""
    if vjp_kernel is not None:
        if vjp_key is None:
            raise ValueError("vjp_key is required when vjp_kernel is provided")
        vjp_res: VjpParityResult | None = vjp_parity(
            vjp_kernel, reference, args, key=vjp_key, rtol=rel, rel=rel
        )
    else:
        vjp_res = None
    got = kernel(*args)
    ref = reference(*args)
    fwd = compare_outputs(got, ref, rel=rel)
    corr_out = corrupt(kernel)(*args)
    return build_probe_parity_payload(
        fwd,
        got,
        ref,
        rel=rel,
        vjp_result=vjp_res,
        corrupted_output=corr_out,
    )
