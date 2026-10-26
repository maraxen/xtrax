"""Kernel seam: one callable that runs a backend kernel where one applies.

``op(reference, kernels=...)`` pairs a plain-JAX *reference* with optional
per-backend *kernels* that share its signature. The returned callable runs the
kernel for the resolved backend when the :class:`~xtrax.intent.context.Context`
it is bound to allows it, and the reference otherwise. The Context is explicit:
``Op.with_context(ctx)`` returns a new op, and an unbound op uses the default
``Context()`` (``kernels=False``), so it runs the reference. No ambient state
decides, and differently bound ops are different callables for ``jax.jit``.

Derivatives come from the reference. The kernel form is a ``jax.custom_jvp``
whose primal is the kernel and whose JVP is ``jax.jvp(reference)``; reverse
mode, ``vmap`` and Hessians follow because JAX transposes that linear JVP. A
backward kernel is an opt-in (``vjp_kernels``) that turns the selected call into
a ``jax.custom_vjp``: reverse mode runs the kernel, and forward mode raises
:class:`~xtrax.intent.errors.IntentError`.

Ops nest. While a derivative rule differentiates a reference, any op that
reference calls runs its own reference, so derivatives come from references all
the way down and a nested ``vjp_kernels`` op never meets the seam's own
``jax.jvp``. Forward passes still run kernels. ``jax.jit`` caches do not see this
switch, so do not wrap an inner op in its own ``jax.jit`` inside a reference.

This is the standalone form. It records nothing in a ``Plan``.
"""

import contextlib
import contextvars
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import update_wrapper
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from xtrax.intent.context import Context
from xtrax.intent.errors import IntentError

__all__ = ["Op", "OpSelection", "VjpKernel", "op"]

# TODO(#6098, spec section 6 step 2): once ``xtrax.compile`` has an intent site kind "op",
# pass the session's Context to the op explicitly (``Op.with_context``), never through ambient
# state, and record ``OpSelection.primal`` and ``OpSelection.reason`` in ``SiteDecision``. Not
# done here: the standalone seam records nothing in a Plan.


class VjpKernel(NamedTuple):
    """A forward and backward kernel pair for one backend.

    Attributes
    ----------
    fwd : Callable
        ``fwd(*args) -> (out, residuals)``. ``out`` has the reference's output
        structure, shapes and dtypes. ``residuals`` is any pytree of arrays.
    bwd : Callable
        ``bwd(residuals, cotangent) -> input cotangents``, one per positional
        argument (a tuple). ``cotangent`` has the structure of ``out``, with
        ``float0`` leaves for integer and bool outputs. Entries for integer and
        bool arguments are ignored.
    """

    fwd: Callable[..., tuple[Any, Any]]
    bwd: Callable[[Any, Any], tuple[Any, ...]]


@dataclass(frozen=True)
class OpSelection:
    """Which implementation an op call uses, and why.

    Attributes
    ----------
    form : {"reference", "kernel", "vjp_kernel"}
        ``"kernel"`` is the ``custom_jvp`` form. ``"vjp_kernel"`` is the
        ``custom_vjp`` form.
    backend : str or None
        Backend whose kernel is used. ``None`` for the reference.
    reason : str
        Why this implementation was chosen.
    """

    form: Literal["reference", "kernel", "vjp_kernel"]
    backend: str | None
    reason: str

    @property
    def primal(self) -> str:
        """Primal label: ``kernel:<backend>`` or ``reference``.

        Parameters
        ----------
        self : OpSelection
            Selection whose backend is read.

        Returns
        -------
        str
            The label a plan records for this call.
        """
        return "reference" if self.backend is None else f"kernel:{self.backend}"


def _is_float(x: Any) -> bool:
    """True when ``x`` has an inexact dtype (a differentiable input or output)."""
    return bool(jnp.issubdtype(jnp.result_type(x), jnp.inexact))


def _zero_tangent(x: Any) -> Any:
    """Zero tangent of ``x``: zeros for floats, a ``float0`` array otherwise."""
    if _is_float(x):
        return jnp.zeros_like(x)
    return np.zeros(np.shape(x), jax.dtypes.float0)


def _coerce_tangents(primals: tuple[Any, ...], tangents: tuple[Any, ...]) -> tuple[Any, ...]:
    """Replace the tangent of every integer or bool input with a ``float0`` array."""
    return tuple(
        t if _is_float(p) else np.zeros(np.shape(p), jax.dtypes.float0)
        for p, t in zip(primals, tangents, strict=True)
    )


_IN_REFERENCE_DERIVATIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "xtrax_op_in_reference_derivative", default=False
)


@contextlib.contextmanager
def _reference_derivative() -> Iterator[None]:
    """Run nested ops as their references while a derivative rule traces a reference."""
    token = _IN_REFERENCE_DERIVATIVE.set(True)
    try:
        yield
    finally:
        _IN_REFERENCE_DERIVATIVE.reset(token)


def _forward_mode_active(args: tuple[Any, ...]) -> bool:
    """True when an argument is traced by forward-mode ``jvp`` (not by reverse mode).

    With direct linearize on (the default), reverse mode traces with
    ``LinearizeTracer`` and only forward mode uses ``JVPTracer``. With it off,
    both use ``JVPTracer``, so this returns False and the caller relies on
    catching JAX's own error, which JAX raises when an eager forward-mode
    call reaches the ``custom_vjp``. Under ``jit`` that error comes at lowering,
    outside any frame this module can catch.
    """
    if not getattr(jax.config, "jax_use_direct_linearize", False):
        return False
    tracer = getattr(jax.interpreters.ad, "JVPTracer", None)
    return tracer is not None and any(isinstance(leaf, tracer) for leaf in jax.tree.leaves(args))


def _normalize_cotangents(args: tuple[Any, ...], cts: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return ``cts`` with ``float0`` zeros at integer and bool inputs."""
    cts = tuple(cts)
    if len(cts) != len(args):
        raise ValueError(f"bwd returned {len(cts)} cotangents for {len(args)} arguments")
    return tuple(c if _is_float(a) else _zero_tangent(a) for a, c in zip(args, cts, strict=True))


def _structs(tree: Any) -> tuple[Any, list[tuple[tuple[int, ...], Any]]]:
    """Tree structure and per-leaf ``(shape, dtype)`` of ``tree`` (abstract values)."""
    leaves, treedef = jax.tree.flatten(tree)
    return treedef, [(tuple(leaf.shape), jnp.dtype(leaf.dtype)) for leaf in leaves]


class Op:
    """A reference plus per-backend kernels behind one callable.

    Build one with :func:`op`. Call it with positional arguments.

    Attributes
    ----------
    reference : Callable
        The plain-JAX implementation. It defines the output structure, the
        derivatives, and what runs when no kernel is selected.
    name : str
        Name used in errors and selections.
    kernels : Mapping[str, Callable]
        Backend name to kernel with the reference's signature.
    vjp_kernels : Mapping[str, VjpKernel]
        Backend name to forward and backward kernel pair.
    context : Context
        Context selection uses. An unbound op has the default ``Context()``
        (``kernels=False``), so it runs the reference. Bind one with
        :meth:`with_context`.

    Notes
    -----
    An op is immutable and hashed by identity. :meth:`with_context` returns a new
    op, so two differently bound ops are different callables and a ``jax.jit``
    cache entry made for one is never reused for the other.
    """

    def __init__(
        self,
        reference: Callable[..., Any],
        kernels: Mapping[str, Callable[..., Any]],
        vjp_kernels: Mapping[str, VjpKernel],
        name: str,
        context: Context | None = None,
    ) -> None:
        """Store the pair and the per-backend implementations.

        Parameters
        ----------
        reference : Callable
            Plain-JAX implementation.
        kernels : Mapping[str, Callable]
            Backend name to kernel.
        vjp_kernels : Mapping[str, VjpKernel]
            Backend name to forward and backward kernel pair.
        name : str
            Name used in errors.
        context : Context or None
            Context to select against. ``None`` is the default ``Context()``.

        Returns
        -------
        None
        """
        self.reference = reference
        self.kernels = dict(kernels)
        self.vjp_kernels = dict(vjp_kernels)
        self.name = name
        self.context = Context() if context is None else context
        self._built: dict[tuple[str, str], Callable[..., Any]] = {}
        # updated=(): copying reference.__dict__ would overwrite these fields when the
        # reference is itself an Op.
        update_wrapper(self, reference, updated=())
        self._frozen = True

    def __setattr__(self, key: str, value: Any) -> None:
        """Reject assignment once the op is built.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Value to store.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            The op is already built.
        """
        if self.__dict__.get("_frozen"):
            raise AttributeError(f"Op is immutable; cannot set {key!r}")
        object.__setattr__(self, key, value)

    def with_context(self, ctx: Context) -> "Op":
        """Return a new op that selects against ``ctx``.

        Parameters
        ----------
        ctx : Context
            Constraints for selection. The op itself is not modified.

        Returns
        -------
        Op
            An op with the same reference, kernels, vjp kernels and name,
            bound to ``ctx``.
        """
        return Op(self.reference, self.kernels, self.vjp_kernels, self.name, ctx)

    def select(self, ctx: Context | None = None) -> OpSelection:
        """Apply the selection rules to ``ctx``.

        Parameters
        ----------
        ctx : Context or None
            Context to select against. ``None`` uses the op's bound context.

        Returns
        -------
        OpSelection
            In order: export target selects the reference; ``kernels=False``
            selects the reference; a ``vjp_kernels`` entry for the resolved
            backend selects the ``custom_vjp`` form; a ``kernels`` entry selects
            the ``custom_jvp`` form; otherwise the reference.
        """
        ctx = self.context if ctx is None else ctx
        if ctx.target == "export":
            return OpSelection("reference", None, "export target")
        if not ctx.kernels:
            return OpSelection("reference", None, "kernels=False")
        backend = ctx.resolved_backend()
        if backend in self.vjp_kernels:
            return OpSelection("vjp_kernel", backend, f"vjp kernel for backend {backend!r}")
        if backend in self.kernels:
            return OpSelection("kernel", backend, f"kernel for backend {backend!r}")
        return OpSelection("reference", None, f"no kernel for backend {backend!r}")

    def __call__(self, *args: Any) -> Any:
        """Run the selected implementation.

        Parameters
        ----------
        args : Any
            Positional arguments of the reference.

        Returns
        -------
        Any
            The reference's output pytree.

        Raises
        ------
        IntentError
            A kernel's output structure, shape or dtype differs from the
            reference's, or forward mode was applied through a selected vjp
            kernel.
        """
        sel = self.select()
        if sel.form == "reference" or _IN_REFERENCE_DERIVATIVE.get():
            return self.reference(*args)
        self._check_signature(sel, args)
        fn = self._function(sel)
        if sel.form == "kernel":
            return fn(*args)
        if _forward_mode_active(args):
            raise self._forward_mode_error(sel)
        try:
            return fn(*args)
        except TypeError as err:
            if "forward-mode" in str(err) and "custom_vjp" in str(err):
                raise self._forward_mode_error(sel) from err
            raise

    def _forward_mode_error(self, sel: OpSelection) -> IntentError:
        """Error for forward-mode autodiff through a selected vjp kernel."""
        return IntentError(
            f"op {self.name!r}: vjp kernel selected for backend {sel.backend!r}; "
            "forward mode needs the JVP form",
            site=self.name,
            path="diff",
            constraint="forward mode (jvp, jacfwd) needs the custom_jvp form; "
            "remove the vjp kernel for this backend or use reverse mode",
        )

    def _function(self, sel: OpSelection) -> Callable[..., Any]:
        """Build (once) the differentiable function for a selection."""
        key = (sel.form, sel.backend)
        if key not in self._built:
            if sel.form == "kernel":
                self._built[key] = self._build_jvp_form(self.kernels[sel.backend])
            else:
                self._built[key] = self._build_vjp_form(self.vjp_kernels[sel.backend])
        return self._built[key]

    def _check_signature(self, sel: OpSelection, args: tuple[Any, ...]) -> None:
        """Raise when the kernel's abstract output differs from the reference's."""
        if sel.form == "vjp_kernel":
            kernel_out = jax.eval_shape(lambda *a: self.vjp_kernels[sel.backend].fwd(*a)[0], *args)
        else:
            kernel_out = jax.eval_shape(self.kernels[sel.backend], *args)
        ref_out = jax.eval_shape(self.reference, *args)
        got, want = _structs(kernel_out), _structs(ref_out)
        if got != want:
            raise IntentError(
                f"op {self.name!r}: kernel for backend {sel.backend!r} returns {got[1]} "
                f"but the reference returns {want[1]}",
                site=self.name,
                path="primal",
                constraint="kernel output structure, shapes and dtypes must match the reference",
            )

    def _build_jvp_form(self, kernel: Callable[..., Any]) -> Callable[..., Any]:
        """``custom_jvp``: primal is ``kernel``, JVP is ``jax.jvp(reference)``."""
        reference = self.reference

        @jax.custom_jvp
        def fn(*args):
            """Run the kernel."""
            return kernel(*args)

        @fn.defjvp
        def fn_jvp(primals, tangents):
            """Reference JVP with ``float0`` tangents for integer and bool inputs."""
            tangents = _coerce_tangents(primals, tangents)
            out = fn(*primals)
            with _reference_derivative():
                _, tangent_out = jax.jvp(reference, primals, tangents)
            return out, tangent_out

        return fn

    def _build_vjp_form(self, vk: VjpKernel) -> Callable[..., Any]:
        """``custom_vjp``: reverse mode runs ``vk.bwd``; second order uses the reference.

        The forward pair and the backward call are each a ``custom_jvp`` whose JVP
        is taken from the reference, so differentiating the gradient (reverse over
        reverse) never reaches a kernel.
        """
        reference = self.reference

        @jax.custom_jvp
        def fwd_pair(*args):
            """Run the forward kernel: ``(out, residuals)``."""
            return vk.fwd(*args)

        @fwd_pair.defjvp
        def fwd_pair_jvp(primals, tangents):
            """Reference JVP for ``out``; residuals are constants."""
            tangents = _coerce_tangents(primals, tangents)
            out, residuals = fwd_pair(*primals)
            with _reference_derivative():
                _, tangent_out = jax.jvp(reference, primals, tangents)
            return (out, residuals), (tangent_out, jax.tree.map(_zero_tangent, residuals))

        def reference_bwd(args, ct):
            """Input cotangents by reference autodiff."""
            with _reference_derivative():
                return _normalize_cotangents(args, jax.vjp(reference, *args)[1](ct))

        @jax.custom_jvp
        def bwd_apply(args, residuals, ct):
            """Run the backward kernel."""
            return _normalize_cotangents(args, vk.bwd(residuals, ct))

        @bwd_apply.defjvp
        def bwd_apply_jvp(primals, tangents):
            """Second-order JVP of the backward call, from the reference."""
            args, _, ct = primals
            t_args, _, t_ct = tangents
            t_args = _coerce_tangents(args, t_args)
            _, tangent_out = jax.jvp(reference_bwd, (args, ct), (t_args, t_ct))
            return bwd_apply(*primals), tangent_out

        @jax.custom_vjp
        def fn(*args):
            """Run the forward kernel, dropping residuals."""
            return vk.fwd(*args)[0]

        def fn_fwd(*args):
            """Forward pass saving arguments and kernel residuals."""
            out, residuals = fwd_pair(*args)
            return out, (args, residuals)

        def fn_bwd(saved, ct):
            """Backward pass through the kernel."""
            args, residuals = saved
            return bwd_apply(args, residuals, ct)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn


def op(
    reference: Callable[..., Any],
    *,
    kernels: Mapping[str, Callable[..., Any]] | None = None,
    vjp_kernels: Mapping[str, VjpKernel] | None = None,
    name: str | None = None,
    context: Context | None = None,
) -> Op:
    """Pair a plain-JAX reference with per-backend kernels.

    The result runs the kernel for ``Context.resolved_backend()`` when the
    Context is not an export target and has ``kernels=True``, and the reference
    otherwise. Derivatives of a kernel call come from the reference (see the
    module docstring). ``kernels`` and ``vjp_kernels`` entries are checked
    against the reference's output structure, shapes and dtypes at call time.

    Parameters
    ----------
    reference : Callable
        Plain-JAX implementation. Takes positional arguments.
    kernels : Mapping[str, Callable] or None
        Backend name (``"gpu"``, ``"tpu"``, ``"cpu"``) to a kernel with the
        reference's signature and output. The call is a ``custom_jvp``: primal
        is the kernel, JVP is ``jax.jvp(reference)`` with ``float0`` tangents for
        integer and bool inputs. An op whose outputs are all integer or bool has
        ``float0`` output tangents.
    vjp_kernels : Mapping[str, VjpKernel] or None
        Backend name to a forward and backward kernel pair. When one is
        selected the call is a ``custom_vjp``: reverse mode runs the backward
        kernel, second-order reverse mode falls back to the reference, and
        forward mode raises :class:`~xtrax.intent.errors.IntentError`. A
        backend present in both mappings uses the vjp kernel.
    name : str or None
        Name used in errors. Defaults to the reference's ``__name__``.
    context : Context or None
        Convenience for ``op(...).with_context(context)``. ``None`` leaves the op
        unbound, so it selects against the default ``Context()`` and runs the
        reference.

    Returns
    -------
    Op
        Callable with the reference's signature.
    """
    label = name if name is not None else getattr(reference, "__name__", "op")
    return Op(reference, kernels or {}, vjp_kernels or {}, label, context)
