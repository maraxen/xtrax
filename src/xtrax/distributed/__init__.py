"""Distributed training utilities for JAX."""

from xtrax.distributed.init import init_dist, is_distributed
from xtrax.distributed.sharding import (
    ShardingPolicy,
    get_device_mesh,
    get_hardware_mesh_profile,
)

__all__ = [
    "init_dist",
    "is_distributed",
    "ShardingPolicy",
    "get_device_mesh",
    "get_hardware_mesh_profile",
]
