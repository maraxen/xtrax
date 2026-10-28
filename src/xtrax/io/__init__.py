"""Deprecated (xtrax #6107, probe release): import from `xtrax.engine.io`.

`xtrax.io` is a second import path for `xtrax.engine.io`. Both names warn on access and
return the very same objects.
"""

from xtrax import _renamed
from xtrax.engine import io as _engine_io
from xtrax.io import callbacks  # noqa: F401  (kept importable as `xtrax.io.callbacks`)

__all__ = [
    "BoundedCallbackHandler",
    "async_indexed_stream",
]

_DEPRECATED = {
    name: _renamed.Deprecated(getattr(_engine_io, name), f"xtrax.engine.io.{name}")
    for name in __all__
}
__getattr__ = _renamed.deprecated_getattr(__name__, _DEPRECATED)
