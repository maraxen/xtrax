"""Deprecated (xtrax #6107, probe release): use JAX's own distributed and mesh APIs.

Every name here warns on access and is otherwise unchanged. See each warning for its
replacement: `jax.distributed.initialize()`, `jax.make_mesh(shape, axis_names)`.
"""

from xtrax._renamed import deprecated_reexports

__all__ = [
    "init_dist",
    "is_distributed",
    "ShardingPolicy",
    "get_device_mesh",
    "get_hardware_mesh_profile",
]

# Not imported eagerly: the defining module's `__getattr__` hook must be the one that fires.
__getattr__ = deprecated_reexports(
    __name__,
    {
        "init_dist": "xtrax.distributed.init",
        "is_distributed": "xtrax.distributed.init",
        "ShardingPolicy": "xtrax.distributed.sharding",
        "get_device_mesh": "xtrax.distributed.sharding",
        "get_hardware_mesh_profile": "xtrax.distributed.sharding",
    },
)
