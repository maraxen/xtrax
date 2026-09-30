"""Deprecated aliases for names renamed in #3644 (SafeMap -> ChunkedMap).

`SafeMap` collided in meaning with JAX's own `jax._src.util.safe_map` (a length-checked
map) and falsely suggested the checkify-based `xtrax.safety` subsystem; the strategy
is memory-bounded chunking, so it is now `ChunkedMap`. The old names resolve for one
release through module `__getattr__` hooks that call `deprecated_alias`, each access
raising a DeprecationWarning. Migrate mechanically with the ast-grep rules in
`codemods/safemap-to-chunkedmap/` (see its README).

The alias returns the NEW object, so `type(SafeMap(...)).__name__` is "ChunkedMap".
Code that dispatches on the strategy's class NAME therefore sees the new name;
xtrax's own name-based matching accepts both for this release (topology, export).
"""

import warnings
from typing import Any

RENAMED: dict[str, str] = {
    "SafeMap": "ChunkedMap",
    "SafeMapIterator": "ChunkedMapIterator",
    "safe_map": "chunked_map",
}

# Strategy class names that name-based matching accepts for a chunked map: the new
# name, plus the legacy one so a consumer's own duck-typed `SafeMap` class still matches.
CHUNKED_MAP_NAMES = ("ChunkedMap", "SafeMap")


def deprecated_alias(module_name: str, name: str, namespace: dict[str, Any]) -> Any:  # noqa: ANN401
    """Resolve a renamed attribute of `module_name`, warning; AttributeError otherwise."""
    new = RENAMED.get(name)
    if new is None or new not in namespace:
        msg = f"module {module_name!r} has no attribute {name!r}"
        raise AttributeError(msg)
    warnings.warn(
        f"{module_name}.{name} was renamed to {new} (xtrax #3644); the old name is a "
        "deprecated alias and will be removed in the next release. Migrate with the "
        "ast-grep rules in codemods/safemap-to-chunkedmap/.",
        DeprecationWarning,
        stacklevel=3,
    )
    return namespace[new]
