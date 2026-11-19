"""Distributed sharding and multi-device placement utilities."""

from xtrax._renamed import deprecated_reexports
from xtrax.distributed.sharding import (
    ShardingPlanner,
    ShardingPlannerError,
    ShardingPolicy,
)

__all__ = [
    "ShardingPlanner",
    "ShardingPlannerError",
    "ShardingPolicy",
    "init_dist",
    "is_distributed",
    "get_device_mesh",
    "get_hardware_mesh_profile",
]

# Not imported eagerly: the defining module's `__getattr__` hook must be the one that fires.
__getattr__ = deprecated_reexports(
    __name__,
    {
        "init_dist": "xtrax.distributed.init",
        "is_distributed": "xtrax.distributed.init",
        "get_device_mesh": "xtrax.distributed.sharding",
        "get_hardware_mesh_profile": "xtrax.distributed.sharding",
    },
)
