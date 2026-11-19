from xtrax._renamed import deprecated_reexports
from xtrax.transforms.map import chunked_map

__all__ = ["chunked_map", "safe_scan"]

# Not imported eagerly: the defining module's `__getattr__` hook must be the one that fires.
__getattr__ = deprecated_reexports(
    __name__,
    {
        "safe_scan": "xtrax.transforms.scan",
    },
)
