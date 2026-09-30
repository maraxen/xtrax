"""The ``onnx`` target: jax2onnx conversion and ONNX Runtime execution.

jax2onnx traces the composed Python callable directly rather than consuming
StableHLO, so this module is a sibling of ``xtrax.export.compile``, not a
wrapper around it. Both toolchain imports are lazy: ``xtrax.export`` stays
importable on a base install, and a missing ``onnx`` extra surfaces as a
``CompileError`` naming it.

Three behaviours of the toolchain shape this module, all measured 260930
(jax2onnx 0.17.0, onnxruntime 1.30.0; see
.praxia/docs/research/260930_onnx-route-spike.md):

- **jax2onnx leaks a patch.** A process's first ``to_onnx`` call leaves
  ``jax.numpy.cumsum`` replaced after it returns (later calls do not re-patch).
  ``convert_to_onnx`` snapshots the public ``jax.numpy``/``jax.lax``/``jax.nn``/
  ``jax.random`` namespaces and restores any attribute the conversion replaced,
  so exporting cannot change what the caller's later JAX code runs. Restoring
  is safe for jax2onnx itself: a ``cumsum`` program converted afterwards is
  still exact.
- **In-graph RNG is not preserved.** ``jax.random.uniform`` lowers to ONNX
  ``RandomUniform``/``RandomUniformLike``, ONNX's own attribute-seeded RNG,
  instead of threefry arithmetic, so the artifact cannot reproduce JAX's
  key-determined bits. The export gate refuses JAX RNG primitives for this
  target, and ``find_onnx_rng_ops`` is a backstop on the converted graph.
- **int64 lives inside index graphs.** The ONNX spec forces int64 on
  ``TopK``/``ArgMax`` index outputs; jax2onnx casts back, so graph I/O keeps
  JAX's dtypes. ``onnx_dtype_census`` records where int64 appears, because ORT
  Web's WebGPU execution provider has none.

Integer and bool leaves are compared exactly in ``verify_onnx_parity``. A
relative tolerance on an index is not a tolerance, it is a wrong answer that
happens to be close: ``np.allclose(1_000_000, 1_000_009, rtol=1e-5)`` is True.
"""

import contextlib
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.export.compile import CompileError, CompileResult
from xtrax.export.parity import ParityResult, compare
from xtrax.export.targets import Backend, Target

__all__ = [
    "ONNX_OPSET",
    "ONNX_RNG_OP_TYPES",
    "LeafParityResult",
    "OnnxDtypeCensus",
    "convert_to_onnx",
    "find_onnx_rng_ops",
    "onnx_dtype_census",
    "run_onnx",
    "verify_onnx_parity",
]

_MISSING_EXTRA = "install the ONNX toolchain with: pip install xtrax[onnx]"

#: The opset every graph is emitted at. jax2onnx 0.17's default; pinned here so
#: a converter upgrade cannot move it silently.
ONNX_OPSET = 23

#: ONNX operators that draw random numbers. None of them is seeded by a JAX key
#: tensor, so none can reproduce a JAX draw.
ONNX_RNG_OP_TYPES = frozenset(
    {
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)


def _require_jax2onnx() -> Any:
    """Import jax2onnx, or raise CompileError naming the extra."""
    try:
        import jax2onnx  # ty: ignore[unresolved-import]
    except ImportError as exc:
        msg = f"jax2onnx is not installed: {_MISSING_EXTRA}"
        raise CompileError(msg) from exc
    return jax2onnx


def _require_ort() -> Any:
    """Import onnxruntime, or raise CompileError naming the extra."""
    try:
        import onnxruntime  # ty: ignore[unresolved-import]
    except ImportError as exc:
        msg = f"onnxruntime is not installed: {_MISSING_EXTRA}"
        raise CompileError(msg) from exc
    return onnxruntime


def _require_onnx() -> Any:
    """Import onnx, or raise CompileError naming the extra."""
    try:
        import onnx  # ty: ignore[unresolved-import]
    except ImportError as exc:
        msg = f"onnx is not installed: {_MISSING_EXTRA}"
        raise CompileError(msg) from exc
    return onnx


def _guarded_modules() -> tuple[ModuleType, ...]:
    """The public JAX namespaces a conversion must leave as it found them."""
    import jax.nn
    import jax.numpy
    import jax.random

    return (jax.numpy, jax.lax, jax.nn, jax.random)


@contextlib.contextmanager
def _restoring_jax_namespaces() -> Iterator[None]:
    """Undo any public-attribute replacement made inside the block.

    Only attributes that existed before and were *replaced* are restored.
    Identity, not equality, is the test: a patch is a different object. Names
    the block *adds* are left in place (jax2onnx adds e.g. ``jnp.cumsum_p`` and
    ``lax.remat2_p``, which no pre-existing code references and which jax2onnx
    itself may rely on). A replacement made concurrently by other code during
    the block cannot be told apart from jax2onnx's and is reverted too.
    """
    modules = _guarded_modules()
    snapshot = [(m, dict(vars(m))) for m in modules]
    try:
        yield
    finally:
        for module, before in snapshot:
            for name, value in before.items():
                if name.startswith("_"):
                    continue
                if vars(module).get(name, value) is not value:
                    setattr(module, name, value)


def _flat_callable(
    callable_: Callable[..., Any], abstract_inputs: Sequence[Any]
) -> tuple[Callable[..., Any], list[Any]]:
    """Wrap ``callable_`` to take and return flat array leaves.

    jax2onnx takes one spec per positional input and emits one graph output per
    returned array, so pytree structure is flattened on the way in and out. The
    graph's inputs are therefore ``jax.tree_util.tree_leaves(abstract_inputs)``
    in order, and its outputs are the leaves of the callable's result.
    """
    leaves, treedef = jax.tree_util.tree_flatten(list(abstract_inputs))

    def flat(*flat_args: Any) -> tuple[Any, ...]:
        args = jax.tree_util.tree_unflatten(treedef, list(flat_args))
        return tuple(jax.tree_util.tree_leaves(callable_(*args)))

    specs = [jax.ShapeDtypeStruct(np.shape(x), x.dtype) for x in leaves]
    return flat, specs


def _walk_graphs(graph: Any) -> Iterator[Any]:
    """Yield ``graph`` and every subgraph reachable through node attributes."""
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from _walk_graphs(attr.g)
            for sub in attr.graphs:
                yield from _walk_graphs(sub)


def _all_nodes(model: Any) -> Iterator[Any]:
    """Every node in the model: the main graph, its subgraphs, and functions."""
    for graph in _walk_graphs(model.graph):
        yield from graph.node
    for function in model.functions:
        yield from function.node


def find_onnx_rng_ops(model: Any) -> list[str]:
    """Name every random-number op in an ONNX model, subgraphs and functions included.

    Args:
        model: An ``onnx.ModelProto``.

    Returns:
        ``"<op_type>:<node name>"`` for each RNG node, in discovery order.
    """
    return [f"{n.op_type}:{n.name}" for n in _all_nodes(model) if n.op_type in ONNX_RNG_OP_TYPES]


@dataclass(frozen=True)
class OnnxDtypeCensus:
    """Where each tensor dtype occurs in a converted graph.

    Attributes:
        node_output_dtypes: Count of node outputs per numpy dtype name, across
            the main graph and every subgraph. ``"unknown"`` counts outputs whose
            type shape inference could not resolve.
        int64_producers: Count of int64-producing nodes per op type. Non-empty
            means ORT Web's WebGPU EP (which has no int64) would have to place
            those nodes on another EP.
        graph_inputs: Dtype of each graph input, in order.
        graph_outputs: Dtype of each graph output, in order.
    """

    node_output_dtypes: dict[str, int] = field(default_factory=dict)
    int64_producers: dict[str, int] = field(default_factory=dict)
    graph_inputs: tuple[str, ...] = ()
    graph_outputs: tuple[str, ...] = ()

    @property
    def n_int64(self) -> int:
        """Total int64 node outputs in the graph."""
        return self.node_output_dtypes.get("int64", 0)


def onnx_dtype_census(model: Any) -> OnnxDtypeCensus:
    """Count the dtypes flowing through an ONNX model.

    Args:
        model: An ``onnx.ModelProto``.

    Returns:
        The census. Shape inference is attempted first; if it fails, the census
        falls back to whatever value_info the converter emitted, and unresolved
        outputs are counted as ``"unknown"`` rather than guessed.
    """
    onnx = _require_onnx()
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # noqa: BLE001 - the census degrades; it never blocks export
        inferred = model

    graphs = list(_walk_graphs(inferred.graph))
    types: dict[str, int] = {}
    for graph in graphs:
        for info in (*graph.value_info, *graph.input, *graph.output):
            if info.type.HasField("tensor_type"):
                types[info.name] = info.type.tensor_type.elem_type
        for init in graph.initializer:
            types[init.name] = init.data_type

    def name_of(elem_type: int | None) -> str:
        # elem_type 0 is UNDEFINED (a partial inference result); the helper
        # raises KeyError on it. The census must never block an export.
        if not elem_type:
            return "unknown"
        try:
            return str(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)))
        except (KeyError, TypeError, ValueError):
            return "unknown"

    def io_type(info: Any) -> int | None:
        return info.type.tensor_type.elem_type if info.type.HasField("tensor_type") else None

    counts: Counter[str] = Counter()
    int64_producers: Counter[str] = Counter()
    for graph in graphs:
        for node in graph.node:
            for out in node.output:
                dtype = name_of(types.get(out))
                counts[dtype] += 1
                if dtype == "int64":
                    int64_producers[node.op_type] += 1

    return OnnxDtypeCensus(
        node_output_dtypes=dict(counts),
        int64_producers=dict(int64_producers),
        graph_inputs=tuple(name_of(io_type(v)) for v in model.graph.input),
        graph_outputs=tuple(name_of(io_type(v)) for v in model.graph.output),
    )


def convert_to_onnx(
    callable_: Callable[..., Any],
    abstract_inputs: Sequence[Any],
    target: Target,
    *,
    out_path: Path | None = None,
) -> tuple[CompileResult, OnnxDtypeCensus]:
    """Convert a traceable callable to an ONNX graph for ``target``.

    Args:
        callable_: The composed callable, as passed to ``jax.jit``.
        abstract_inputs: Abstract inputs, one per positional argument. Pytrees
            are flattened; see ``run_onnx`` for the resulting input order.
        target: An ``onnx``-backend target.
        out_path: Destination for the ``.onnx`` file. A fresh temp file is used
            if omitted.

    Returns:
        The written artifact as a ``CompileResult`` (``spirv_bytes`` None,
        ``downgraded_stablehlo`` False) and the graph's dtype census.

    Raises:
        ValueError: If ``target`` is not an ONNX-backend target.
        CompileError: If the toolchain is missing, ``jax_enable_x64`` is set,
            jax2onnx rejects the callable, or the graph contains RNG ops.
    """
    if target.backend is not Backend.ONNX:
        msg = f"convert_to_onnx needs an onnx-backend target, got {target.name!r}"
        raise ValueError(msg)
    if jax.config.jax_enable_x64:
        # Under x64 every default int and float widens, so the graph's I/O
        # would be int64/f64 -- a different contract from the 32-bit program
        # the caller tested, and one the target's dtype envelope excludes.
        msg = (
            "jax_enable_x64 is set in this process, so the traced program would "
            "carry int64/f64 where a 32-bit run carries int32/f32. Export from a "
            "process with x64 disabled."
        )
        raise CompileError(msg)

    jax2onnx = _require_jax2onnx()
    flat, specs = _flat_callable(callable_, abstract_inputs)
    try:
        with _restoring_jax_namespaces():
            model = jax2onnx.to_onnx(flat, specs, model_name="xtrax_export", opset=ONNX_OPSET)
    except Exception as exc:
        msg = f"jax2onnx could not convert the pipeline for target {target.name!r}: {exc}"
        raise CompileError(msg) from exc

    rng_ops = find_onnx_rng_ops(model)
    if rng_ops:
        msg = (
            f"the converted graph for target {target.name!r} contains random-number "
            f"ops {rng_ops}. ONNX RNG is seeded by attribute, not by a JAX key, so "
            f"the artifact cannot reproduce the pipeline's draws. Pass the random "
            f"values (or keys consumed outside the graph) in as inputs instead."
        )
        raise CompileError(msg)

    if out_path is None:
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed immediately below
            suffix=f".{target.name}.onnx", delete=False
        )
        handle.close()
        out_path = Path(handle.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = model.SerializeToString()
    out_path.write_bytes(data)

    compiled = CompileResult(
        target=target,
        path=out_path,
        size_bytes=len(data),
        spirv_bytes=None,
        downgraded_stablehlo=False,
        stderr="",
    )
    return compiled, onnx_dtype_census(model)


def run_onnx(onnx_path: Path, *args: Any) -> list[np.ndarray]:
    """Execute an ONNX artifact on ORT's CPU execution provider.

    Args:
        onnx_path: Path to a ``.onnx`` file from ``convert_to_onnx``.
        *args: Concrete arguments, structured like the ``abstract_inputs`` the
            graph was converted with. Pytrees are flattened in the same order.

    Returns:
        One numpy array per graph output: the leaves of the pipeline's result.
    """
    ort = _require_ort()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(list(args))]
    feeds = {spec.name: leaf for spec, leaf in zip(session.get_inputs(), leaves, strict=True)}
    return [np.asarray(out) for out in session.run(None, feeds)]


@dataclass(frozen=True)
class LeafParityResult(ParityResult):
    """A ``ParityResult`` over every output leaf, integers compared exactly.

    The inherited fields describe the first failing leaf, or the first leaf when
    all pass, except ``passed`` (every leaf) and ``max_abs_diff`` (the maximum
    across float leaves; inf on a shape or dtype mismatch).

    Attributes:
        leaf_results: One ``ParityResult`` per leaf, in output order.
        dtype_mismatches: ``"leaf <i>: expected <dtype>, got <dtype>"`` for each
            leaf whose dtype changed across the export.
    """

    leaf_results: tuple[ParityResult, ...] = ()
    dtype_mismatches: tuple[str, ...] = ()

    def summary(self) -> str:
        """Render a verdict naming the failing leaf and how it failed."""
        verdict = "PASS" if self.passed else "FAIL"
        if self.dtype_mismatches:
            return f"{verdict}: " + "; ".join(self.dtype_mismatches)
        n = len(self.leaf_results)
        for i, leaf in enumerate(self.leaf_results):
            if leaf.passed:
                continue
            if leaf.shape_expected != leaf.shape_actual:
                detail = f"shape mismatch expected {leaf.shape_expected}, got {leaf.shape_actual}"
            elif leaf.atol == 0.0 and leaf.rtol == 0.0:
                detail = f"max|diff| = {leaf.max_abs_diff:.3e} (exact integer comparison)"
            else:
                detail = (
                    f"max|diff| = {leaf.max_abs_diff:.3e} (atol={leaf.atol:g}, rtol={leaf.rtol:g})"
                )
            return f"{verdict}: leaf {i} of {n}: {detail}"
        return f"{verdict}: {n} leaf/leaves, max|diff| = {self.max_abs_diff:.3e}"


def _exact(expected: np.ndarray, actual: np.ndarray) -> ParityResult:
    """Exact comparison for integer and bool leaves."""
    shapes_match = expected.shape == actual.shape
    passed = shapes_match and bool(np.array_equal(expected, actual))
    if not shapes_match:
        diff = float("inf")
    elif expected.size:
        wide_e = expected.astype(np.int64)
        diff = float(np.max(np.abs(wide_e - actual.astype(np.int64))))
    else:
        diff = 0.0
    return ParityResult(
        passed=passed,
        max_abs_diff=diff,
        atol=0.0,
        rtol=0.0,
        shape_expected=tuple(expected.shape),
        shape_actual=tuple(actual.shape),
    )


def verify_onnx_parity(
    expected: Any,
    onnx_path: Path,
    concrete_inputs: Sequence[Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> LeafParityResult:
    """Execute an ONNX artifact and compare every output leaf against a reference.

    Args:
        expected: An independently-computed reference value (any pytree). The
            same rule as ``verify_native_parity`` applies: it must not be
            derived from the callable under test.
        onnx_path: Path to the ``.onnx`` artifact.
        concrete_inputs: Concrete arguments to execute with.
        atol: Absolute tolerance for float leaves.
        rtol: Relative tolerance for float leaves.

    Returns:
        A ``LeafParityResult``. It fails on a leaf-count mismatch, on any dtype
        change, on any integer/bool difference, or on a float leaf outside
        tolerance.

    Each reference leaf is first passed through ``jnp.asarray``, exactly as
    ``compare`` does for the IREE targets. In a 32-bit process that narrows a
    NumPy oracle's float64/int64 to float32/int32, so one ``reference_fn``
    written in NumPy verifies the same way on every target; the dtype rule then
    catches a genuine change across the export, e.g. int32 -> int64.
    """
    exp_leaves = [np.asarray(jnp.asarray(x)) for x in jax.tree_util.tree_leaves(expected)]
    act_leaves = run_onnx(onnx_path, *concrete_inputs)

    if len(exp_leaves) != len(act_leaves):
        return LeafParityResult(
            passed=False,
            max_abs_diff=float("inf"),
            atol=atol,
            rtol=rtol,
            shape_expected=(len(exp_leaves),),
            shape_actual=(len(act_leaves),),
            dtype_mismatches=(f"expected {len(exp_leaves)} output leaves, got {len(act_leaves)}",),
        )

    results: list[ParityResult] = []
    mismatches: list[str] = []
    for i, (exp, act) in enumerate(zip(exp_leaves, act_leaves, strict=True)):
        if exp.dtype != act.dtype:
            mismatches.append(f"leaf {i}: expected {exp.dtype}, got {act.dtype}")
        if np.issubdtype(exp.dtype, np.integer) or exp.dtype == np.bool_:
            results.append(_exact(exp, act))
        else:
            results.append(compare(exp, act, atol=atol, rtol=rtol))

    passed = not mismatches and all(r.passed for r in results)
    lead = next((r for r in results if not r.passed), results[0] if results else None)
    max_diff = float("inf") if mismatches else max((r.max_abs_diff for r in results), default=0.0)
    return LeafParityResult(
        passed=passed,
        max_abs_diff=max_diff,
        atol=atol,
        rtol=rtol,
        shape_expected=lead.shape_expected if lead else (),
        shape_actual=lead.shape_actual if lead else (),
        leaf_results=tuple(results),
        dtype_mismatches=tuple(mismatches),
    )
