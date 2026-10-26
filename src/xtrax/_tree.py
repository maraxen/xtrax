"""Plain-JAX pytree base and helpers (no Equinox import, ever).

``Struct`` is a frozen dataclass registered as a pytree: fields marked static are
treedef metadata, every other field is a child. ``model_jit`` is the one place a
user model (any pytree, including an ``equinox.Module``) enters a trace: array
leaves are traced, everything else rides along as a hashable static argument.

Spec: .praxia/docs/specs/261023_equinox-decoupling.md section 3.
"""

import dataclasses
import functools
import sys
from typing import dataclass_transform

import jax
import numpy as np

__all__ = [
    "Struct",
    "combine",
    "field",
    "inference_mode",
    "is_array",
    "is_inexact_array",
    "model_jit",
    "partition",
    "value_and_grad",
]


def field(*, static=False, **kwargs):
    """``dataclasses.field`` whose metadata marks the field static or traced.

    The metadata key matches ``equinox.field(static=True)``, so either spelling works
    on a ``Struct`` subclass. Extra ``metadata`` entries are kept.
    """
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata["static"] = static
    return dataclasses.field(metadata=metadata, **kwargs)


def _register(cls, data, meta):
    """Register ``cls`` as a pytree whose unflatten bypasses ``__init__``.

    ``jax.tree_util.register_dataclass`` rebuilds nodes by calling ``cls(**fields)``
    (jax 0.11.1, ``tree_util.py:unflatten_func``), so every ``tree_map``, ``vmap`` or
    ``jit`` boundary would rerun ``__init__`` and ``__check_init__`` on tracers or
    ``None`` placeholders. Equinox sets fields on an uninitialised instance instead,
    and so do we.
    """

    def flatten_with_keys(x):
        children = tuple((jax.tree_util.GetAttrKey(n), getattr(x, n)) for n in data)
        return children, tuple(getattr(x, n) for n in meta)

    def flatten(x):
        return tuple(getattr(x, n) for n in data), tuple(getattr(x, n) for n in meta)

    def unflatten(aux, children):
        obj = object.__new__(cls)
        for name, value in zip(meta, aux, strict=True):
            object.__setattr__(obj, name, value)
        for name, value in zip(data, children, strict=True):
            object.__setattr__(obj, name, value)
        return obj

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)


# Tells type checkers that subclasses are frozen dataclasses (PEP 681), which
# ``__init_subclass__`` makes them at runtime.
@dataclass_transform(frozen_default=True, field_specifiers=(field, dataclasses.field))
class Struct:
    """Frozen-dataclass pytree base; subclasses need no decorator.

    ``__init_subclass__`` applies ``dataclasses.dataclass(frozen=True)`` to each
    subclass and registers it as a pytree, splitting fields by
    ``field.metadata.get("static")``. ``__post_init__`` runs every ``__check_init__``
    defined along the MRO (base classes first), so validation is inherited. A subclass
    that defines its own ``__post_init__`` must call ``super().__post_init__()``, and a
    class-defined ``__init__`` must end with ``self.__post_init__()``: dataclass keeps
    such an ``__init__`` and does not call ``__post_init__`` itself.

    Static field values must be hashable and comparable (they live in the treedef).
    Unflattening does not call ``__init__`` or ``__post_init__`` (see ``_register``), so
    checks and custom ``__init__`` signatures never see tracers or placeholders.

    ``==`` compares trees as ``equinox.Module`` does: same structure and static values,
    and equal leaves, giving a Python bool for concrete arrays (a traced bool under a
    trace). ``hash`` hashes the field values, so it raises for a struct holding arrays.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # eq=False keeps the dataclass from generating a field-wise ``==`` that returns
        # an array for array fields; ``__eq__``/``__hash__`` below replace it.
        dataclasses.dataclass(frozen=True, eq=False)(cls)
        data, meta = [], []
        # ty cannot see the dataclass() call above. An annotation on Struct would leak
        # into subclass annotation walks (StageBundle's validator).
        for f in dataclasses.fields(cls):  # ty: ignore[invalid-argument-type]
            if "converter" in f.metadata:
                raise TypeError(
                    f"{cls.__qualname__}.{f.name}: converter= is not supported by "
                    "xtrax.Struct; convert in __init__ or the caller"
                )
            (meta if f.metadata.get("static", False) else data).append(f.name)
        _register(cls, tuple(data), tuple(meta))

    def __post_init__(self):
        for klass in reversed(type(self).__mro__):
            check = klass.__dict__.get("__check_init__")
            if check is not None:
                check(self)

    def __eq__(self, other):
        return tree_equal(self, other)

    def __hash__(self):
        fields = dataclasses.fields(self)  # ty: ignore[invalid-argument-type]
        return hash(tuple((f.name, getattr(self, f.name)) for f in fields))


def tree_equal(a, b):
    """True if ``a`` and ``b`` have the same structure, static values and leaves.

    Array leaves must match in shape, dtype and values. Concrete arrays give a Python
    bool; if a leaf is traced, the result is a traced scalar bool (as
    ``equinox.tree_equal`` returns).
    """
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    if treedef_a != treedef_b:
        return False
    result = True
    for x, y in zip(leaves_a, leaves_b, strict=True):
        x_arr, y_arr = is_array(x), is_array(y)
        if x_arr != y_arr:
            return False
        if not x_arr:
            if type(x) is not type(y) or x != y:
                return False
            continue
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        try:
            if not np.array_equal(np.asarray(x), np.asarray(y)):
                return False
        except jax.errors.TracerArrayConversionError:
            result = jax.numpy.logical_and(result, jax.numpy.all(x == y))
    return result


def is_array(x):
    """True for ``jax.Array``, ``np.ndarray`` and objects with ``__jax_array__``."""
    return isinstance(x, (jax.Array, np.ndarray)) or hasattr(x, "__jax_array__")


def _is_none(x):
    return x is None


def is_inexact_array(x):
    """True for floating and complex arrays: the leaves ``jax.grad`` can differentiate."""
    dtype = getattr(x, "dtype", None)
    return is_array(x) and dtype is not None and jax.numpy.issubdtype(dtype, jax.numpy.inexact)


def partition(tree, pred=is_array):
    """Split ``tree`` into ``(selected, rest)``; each holds ``None`` where the other has a leaf.

    ``pred`` picks the leaves that go to ``selected`` (default: arrays).
    """
    selected = jax.tree_util.tree_map(lambda x: x if pred(x) else None, tree)
    rest = jax.tree_util.tree_map(lambda x: None if pred(x) else x, tree)
    return selected, rest


def combine(arrays, rest):
    """Inverse of ``partition``: take the array where present, else the ``rest`` leaf."""
    return jax.tree_util.tree_map(
        lambda a, r: r if a is None else a, arrays, rest, is_leaf=_is_none
    )


def value_and_grad(fn, *, has_aux=False, pred=is_inexact_array):
    """``jax.value_and_grad`` over the ``pred`` leaves of the first argument.

    The other leaves of that argument (non-array values, integer arrays, anything
    ``pred`` rejects) are held constant by closure. Gradients come back in the
    first argument's structure with ``None`` where a leaf was not differentiated.
    """

    def wrapped(tree, *args, **kwargs):
        diff, rest = partition(tree, pred)

        def inner(diff):
            return fn(combine(diff, rest), *args, **kwargs)

        return jax.value_and_grad(inner, has_aux=has_aux)(diff)

    return wrapped


class _Static:
    """Hashable wrapper for the non-array part of a pytree (hash/eq by treedef + leaves)."""

    __slots__ = ("value", "_key", "_hash")

    def __init__(self, tree):
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        self.value = tree
        self._key = (treedef, tuple(leaves))
        try:
            self._hash = hash(self._key)
        except TypeError:
            paths, _ = jax.tree_util.tree_flatten_with_path(tree)
            for path, leaf in paths:
                try:
                    hash(leaf)
                except TypeError:
                    raise TypeError(
                        "model_jit: non-array leaf at "
                        f"{jax.tree_util.keystr(path)} is unhashable "
                        f"({type(leaf).__name__}); hold it as a tuple or make it an array"
                    ) from None
            raise

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _Static) and self._key == other._key


def _aval_key(arrays):
    return tuple(
        (getattr(a, "shape", None), getattr(a, "dtype", None))
        for a in jax.tree_util.tree_leaves(arrays)
    )


def model_jit(fn):
    """``jax.jit`` for functions whose arguments are pytrees mixing arrays and Python values.

    Every positional and keyword argument is partitioned: array leaves are traced,
    the rest becomes a static argument (``_Static``), so changing an array value never
    retraces and changing a non-array leaf or the tree structure does. The return value
    may also mix arrays and non-array leaves (for example an updated model); its static
    part is recorded at trace time and recombined on every call.

    Apply it to a module-level function and pass the model as an argument; it is not a
    method descriptor.
    """
    out_static = {}

    def run(arrays, static):
        args, kwargs = combine(arrays, static.value)
        out_arrays, out_rest = partition(fn(*args, **kwargs))
        out_static[(static, _aval_key(arrays))] = out_rest
        return out_arrays

    jitted = jax.jit(run, static_argnums=1)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        arrays, rest = partition((args, kwargs))
        static = _Static(rest)
        out_arrays = jitted(arrays, static)
        return combine(out_arrays, out_static[(static, _aval_key(arrays))])

    # Tells tools that wrap callables for trace counting (the performance gate) that this
    # is already a jit boundary, as they would for a ``jax.jit`` result.
    setattr(wrapper, "_xtrax_model_jit", True)  # noqa: B010 (read with getattr)
    return wrapper


def inference_mode(model):
    """Switch ``model`` to inference behaviour without importing Equinox.

    Uses ``model.__xtrax_inference__()`` when defined; otherwise, only if ``equinox`` is
    already imported and ``model`` is one of its modules, ``equinox.nn.inference_mode``;
    otherwise returns ``model`` unchanged.
    """
    hook = getattr(model, "__xtrax_inference__", None)
    if hook is not None:
        return hook()
    eqx = sys.modules.get("equinox")
    if eqx is not None and isinstance(model, eqx.Module):
        return eqx.nn.inference_mode(model)
    return model
