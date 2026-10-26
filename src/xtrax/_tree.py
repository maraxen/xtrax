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

import jax
import numpy as np

__all__ = [
    "Struct",
    "combine",
    "field",
    "inference_mode",
    "is_array",
    "model_jit",
    "partition",
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
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclasses.dataclass(frozen=True)(cls)
        data, meta = [], []
        for f in dataclasses.fields(cls):
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


def is_array(x):
    """True for ``jax.Array``, ``np.ndarray`` and objects with ``__jax_array__``."""
    return isinstance(x, (jax.Array, np.ndarray)) or hasattr(x, "__jax_array__")


def _is_none(x):
    return x is None


def partition(tree):
    """Split ``tree`` into ``(arrays, rest)``; each holds ``None`` where the other has a leaf."""
    arrays = jax.tree_util.tree_map(lambda x: x if is_array(x) else None, tree)
    rest = jax.tree_util.tree_map(lambda x: None if is_array(x) else x, tree)
    return arrays, rest


def combine(arrays, rest):
    """Inverse of ``partition``: take the array where present, else the ``rest`` leaf."""
    return jax.tree_util.tree_map(
        lambda a, r: r if a is None else a, arrays, rest, is_leaf=_is_none
    )


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
