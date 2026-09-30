from xtrax.transforms.map import chunked_map
from xtrax.transforms.scan import safe_scan

__all__ = ["chunked_map", "safe_scan"]


def __getattr__(name: str):  # noqa: ANN202 -- PEP 562 module hook
    """Deprecated pre-#3644 names (SafeMap, SafeMapIterator, safe_map) for one release."""
    from xtrax._renamed import deprecated_alias

    return deprecated_alias(__name__, name, globals())
