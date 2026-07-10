"""Vendored io_callback shim (O1, T1-03) -- pins jax's still-experimental io_callback.

Wraps `jax.experimental.io_callback` behind this internal module so an
upstream move in the still-experimental namespace is a one-file change. All
Tap/Sink implementations (see xtrax.stages.boundaries) must import
`io_callback` from here, never directly from `jax.experimental`.

Two independent checks run at MODULE IMPORT time (not first call), so
signature or version drift surfaces as a loud import-time exception, never
an opaque runtime traceback inside a future dispatch loop (PM5,
.praxia/docs/specs/260702_design-2174-next-slices-minimal-composit.md):

1. `_check_jax_version` -- the resolved jax is within the range this shim
   has been validated against, even if its signature happens to still
   coincidentally match.
2. `_check_signature` -- io_callback's actual parameter shape still matches
   what xtrax's Tap/Sink implementations rely on, even inside the pinned
   version range.

Both checks are extracted as standalone functions (rather than inlined at
module scope) so they stay independently unit-testable with synthetic
inputs -- no monkeypatching jax or reloading modules required.

Keep PINNED_JAX_RANGE in sync with pyproject.toml's `jax`/`jaxlib` dependency
bounds by hand; there is no single source of truth linking the two yet.
"""

import inspect

import jax.experimental

#: (min inclusive, max exclusive) -- keep in sync with pyproject.toml's jax bound.
PINNED_JAX_RANGE = ((0, 10, 2), (0, 11, 0))

_EXPECTED_PARAMS = (
    "callback",
    "result_shape_dtypes",
    "args",
    "sharding",
    "ordered",
    "kwargs",
)


class IoCallbackSignatureError(ImportError):
    """Raised when jax.experimental.io_callback no longer matches the pinned shape."""


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse a dotted version string's leading numeric components.

    Ignores any non-numeric suffix (e.g. "0.10.2.dev0" -> (0, 10, 2)) so a
    dev/rc build within the pinned range is not rejected on a technicality.
    """
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_jax_version(
    version: str,
    pinned_range: tuple[tuple[int, ...], tuple[int, ...]] = PINNED_JAX_RANGE,
) -> None:
    """Raise if `version` falls outside `pinned_range` (min inclusive, max exclusive)."""
    parsed = _parse_version(version)
    low, high = pinned_range
    if not (low <= parsed < high):
        msg = (
            f"jax {version} (parsed {parsed}) is outside the pinned range "
            f"[{low}, {high}) xtrax.stages._callback has been validated against. "
            f"Update PINNED_JAX_RANGE (and pyproject.toml's jax/jaxlib bounds) "
            f"after re-verifying io_callback's signature on the new jax."
        )
        raise IoCallbackSignatureError(msg)


def _check_signature(
    sig: inspect.Signature,
    expected_params: tuple[str, ...] = _EXPECTED_PARAMS,
) -> None:
    """Raise if `sig`'s parameter names, or `ordered`'s default, have drifted."""
    names = tuple(sig.parameters)
    if names != expected_params:
        msg = (
            f"jax.experimental.io_callback signature changed: expected params "
            f"{expected_params}, got {names}. Update xtrax.stages._callback "
            f"(and its tests) for the new signature."
        )
        raise IoCallbackSignatureError(msg)

    ordered_default = sig.parameters["ordered"].default
    if ordered_default is not False:
        msg = (
            f"jax.experimental.io_callback's 'ordered' parameter default changed "
            f"from False to {ordered_default!r}; xtrax.stages.boundaries.Tap/Sink "
            f"rely on the False default for unordered callers."
        )
        raise IoCallbackSignatureError(msg)


_check_jax_version(jax.__version__)
_check_signature(inspect.signature(jax.experimental.io_callback))

io_callback = jax.experimental.io_callback

__all__ = [
    "PINNED_JAX_RANGE",
    "IoCallbackSignatureError",
    "io_callback",
]
