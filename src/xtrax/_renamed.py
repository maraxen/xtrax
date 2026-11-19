"""Deprecation helpers, and name matching for the #3644 rename (SafeMap -> ChunkedMap).

Two things live here.

1. The deprecated `SafeMap` / `SafeMapIterator` / `safe_map` aliases shipped for one release
   (0.4.0a11) and are removed. Migrate with the ast-grep rules in
   `codemods/safemap-to-chunkedmap/` (see its README). What remains is the set of strategy
   class NAMES that name-based matching accepts for a chunked map (stages/topology,
   export/composer). It still includes the legacy "SafeMap" because a consumer can define its
   OWN duck-typed class of that name: aminx ships `aminx.tiling.strategy.SafeMap` until its
   deprecation (aminx debt #2371). Drop it then (xtrax #5680, step 4).

2. `deprecated_getattr`, the shared PEP 562 hook for the #6107 probe release. A name whose
   replacement is a JAX function (`jax.jit`, `checkify`, `jax.distributed.initialize`) cannot
   be a `deprecated_alias` that returns a new xtrax object, so the hook returns the ORIGINAL
   object unchanged and warns on access. Nothing is removed: behaviour and class identity
   (`type(x).__name__`) are exactly what they were. The warning names the caller, because a
   warning that points into xtrax would be hidden by Python's default filter and could not
   surface an unknown user during the probe release.
"""

import os
import sys
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

CHUNKED_MAP_NAMES = ("ChunkedMap", "SafeMap")


class Deprecated(NamedTuple):
    """One deprecated public name: the live object, what replaces it, and an optional note."""

    obj: Any
    replacement: str
    note: str = ""


def warn_deprecated(qualname: str, replacement: str, note: str = "") -> None:
    """Emit the #6107 `DeprecationWarning`, attributed to the first non-xtrax caller."""
    detail = f" {note}" if note else ""
    warnings.warn(
        f"{qualname} is deprecated (xtrax #6107, probe release) and will be removed in a "
        f"later release if no one reports a use. Replacement: {replacement}.{detail}",
        DeprecationWarning,
        skip_file_prefixes=_internal_prefixes(),
    )


def deprecated_getattr(module_name: str, table: dict[str, Deprecated]) -> Callable[[str], Any]:
    """Build a module `__getattr__` that warns, then returns the original object.

    Args:
        module_name: the module the hook is installed in (used in the message).
        table: public name -> `Deprecated`. The real objects are held here, NOT bound as
            module globals, or normal attribute lookup would find them and skip the hook.
    """

    def __getattr__(name: str) -> Any:
        entry = table.get(name)
        if entry is None:
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        warn_deprecated(f"{module_name}.{name}", entry.replacement, entry.note)
        return entry.obj

    return __getattr__


def deprecated_reexports(package_name: str, homes: dict[str, str]) -> Callable[[str], Any]:
    """Build a package `__getattr__` that forwards deprecated names to their home module.

    The home module's own `deprecated_getattr` hook does the warning, so a package re-export
    (`from xtrax.training import WeightedLoss`) and the defining module's path warn alike.
    This keeps the deprecated names out of the package's eager imports, which is what lets
    the hook fire at all.

    Args:
        package_name: the package the hook is installed in (used in the error message).
        homes: deprecated public name -> dotted path of the module that defines it.
    """

    def __getattr__(name: str) -> Any:
        home = homes.get(name)
        if home is None:
            msg = f"module {package_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        import importlib

        return getattr(importlib.import_module(home), name)

    return __getattr__


def _internal_prefixes() -> tuple[str, ...]:
    """Directories whose frames are never "the caller": xtrax itself (this module and the
    __getattr__ hooks), plus jaxtyping when loaded -- its import hook wraps xtrax functions
    and adds a frame, so a fixed stacklevel would blame the wrong file."""
    prefixes = [os.path.dirname(os.path.abspath(__file__)) + os.sep]
    jaxtyping_file = getattr(sys.modules.get("jaxtyping"), "__file__", None)
    if isinstance(jaxtyping_file, str):
        prefixes.append(os.path.dirname(os.path.abspath(jaxtyping_file)) + os.sep)
    return tuple(prefixes)
