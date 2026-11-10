"""Distributed sharding utilities for JAX arrays and pytrees."""

import math
import re
from typing import Any

import jax
import jax.sharding
import jax.tree_util

from xtrax._renamed import Deprecated, deprecated_getattr
from xtrax._tree import Struct, field


class ShardingPolicy(Struct):
    """Policy for assigning PartitionSpec to pytree leaves.

    Uses regex pattern matching where first match wins. Matching is done via re.search,
    so patterns can match anywhere in the path.

    Attributes:
        rules: Ordered tuple of (regex_pattern, PartitionSpec) pairs.
    """

    rules: tuple[tuple[str, jax.sharding.PartitionSpec], ...] = field(static=True)

    def get_partition_spec(self, path: str) -> jax.sharding.PartitionSpec:
        """Get the PartitionSpec for a given path by matching against rules in order.

        Args:
            path: String path to a pytree leaf (e.g., "layer_0/weight" or "bias").

        Returns:
            The PartitionSpec for the first matching rule, or PartitionSpec() if no
            rule matches (defaults to fully replicated).
        """
        for pattern, partition_spec in self.rules:
            if re.search(pattern, path):
                return partition_spec
        return jax.sharding.PartitionSpec()

    def apply_to_pytree(self, pytree: Any) -> Any:
        """Apply sharding policy to all leaves of a pytree.

        Args:
            pytree: A pytree (dict, list, etc.) with any leaf values.

        Returns:
            A pytree with the same structure but with leaves replaced by PartitionSpec
            objects derived from their paths.
        """

        def apply_to_leaf(path, leaf):
            # Convert jax.tree_util path to a string representation
            path_str = self._path_to_string(path)
            return self.get_partition_spec(path_str)

        # Use jax.tree_util.tree_map_with_path to traverse with path tracking
        # This preserves the pytree structure and applies the function to each leaf
        return jax.tree_util.tree_map_with_path(apply_to_leaf, pytree)

    @staticmethod
    def _path_to_string(path: tuple) -> str:
        """Convert a jax.tree_util path to a string representation.

        Args:
            path: A tuple of GetAttrKey, DictKey, SequenceKey, etc.

        Returns:
            A string representation of the path (e.g., "layer/weight").
        """
        parts = []
        for key in path:
            if isinstance(key, jax.tree_util.DictKey):
                parts.append(str(key.key))
            elif isinstance(key, jax.tree_util.GetAttrKey):
                parts.append(key.name)
            elif isinstance(key, jax.tree_util.SequenceKey):
                parts.append(f"[{key.idx}]")
            else:
                parts.append(str(key))
        return "/".join(parts)


def get_device_mesh(
    shape: tuple[int, ...],
    axis_names: tuple[str, ...],
) -> jax.sharding.Mesh:
    """Create a Mesh object for JAX distributed arrays.

    Args:
        shape: The mesh shape, e.g., (2, 4) for 2 data-parallel x 4 model-parallel.
        axis_names: The axis names, e.g., ("data", "model").

    Returns:
        A jax.sharding.Mesh with the specified shape and axis names.

    Raises:
        ValueError: If the product of shape does not equal the number of JAX devices.
    """
    import numpy as np

    num_devices = len(jax.devices())
    shape_product = math.prod(shape) if shape else 1

    if shape_product != num_devices:
        raise ValueError(
            f"Shape product ({shape_product}) must equal number of devices "
            f"({num_devices}). Got shape={shape}."
        )

    # Reshape devices array to match the requested shape
    devices_array = np.array(jax.devices()).reshape(shape)
    return jax.sharding.Mesh(devices_array, axis_names)


def get_hardware_mesh_profile() -> dict[str, Any]:
    """Get recommended mesh configuration for the current hardware.

    Returns a dict with the following keys:
        - device_type (str): "cpu", "gpu", or "tpu"
        - num_devices (int): Number of available devices
        - recommended_shape (tuple[int, ...]): Suggested mesh shape
        - recommended_axis_names (tuple[str, ...]): Suggested axis names

    This function never raises, even on unknown hardware. For a single CPU device,
    it returns shape=(1,) and axis_names=("batch",).

    Returns:
        A dictionary with device info and recommendations.
    """
    try:
        devices = jax.devices()
        num_devices = len(devices)

        # Determine device type from first device
        device_type = devices[0].platform if hasattr(devices[0], "platform") else "cpu"

        # Normalize device_type
        if device_type.lower() in ("gpu", "cuda"):
            device_type = "gpu"
        elif device_type.lower() in ("tpu",):
            device_type = "tpu"
        else:
            device_type = "cpu"

        # Recommend shape and axis names based on device count
        if num_devices == 1:
            recommended_shape = (1,)
            recommended_axis_names = ("batch",)
        elif num_devices == 2:
            recommended_shape = (2,)
            recommended_axis_names = ("data",)
        elif num_devices % 8 == 0:
            # For TPU pods or large GPU clusters, suggest data-parallel axis
            recommended_shape = (num_devices,)
            recommended_axis_names = ("data",)
        else:
            # Default: single axis, data-parallel
            recommended_shape = (num_devices,)
            recommended_axis_names = ("data",)

        return {
            "device_type": device_type,
            "num_devices": num_devices,
            "recommended_shape": recommended_shape,
            "recommended_axis_names": recommended_axis_names,
        }
    except Exception:
        # Fallback for any errors
        return {
            "device_type": "cpu",
            "num_devices": 1,
            "recommended_shape": (1,),
            "recommended_axis_names": ("batch",),
        }


class ShardingPlannerError(ValueError):
    """Raised when tensor shapes are incompatible with device mesh placement."""


class ShardingPlanner:
    """Active placement planner resolving PartitionSpecs into NamedShardings across a device Mesh.

    Attributes
    ----------
    mesh : jax.sharding.Mesh
        Active device Mesh instance.
    policy : ShardingPolicy
        ShardingPolicy containing path matching rules.
    """

    mesh: jax.sharding.Mesh
    policy: ShardingPolicy

    def __init__(self, mesh: jax.sharding.Mesh, policy: ShardingPolicy) -> None:
        self.mesh = mesh
        self.policy = policy

    def plan_sharding(self, pytree: Any) -> Any:
        """Derive concrete NamedSharding for each PyTree leaf, verifying divisibility.

        Parameters
        ----------
        pytree : Any
            Target PyTree containing arrays and static values.

        Returns
        -------
        Any
            PyTree of NamedSharding instances matching the structure of pytree.

        Raises
        ------
        ShardingPlannerError
            If a tensor dimension is not divisible by mesh axis, or mesh axis is unknown,
            or PartitionSpec rank exceeds leaf rank.
        """

        def _plan_leaf(path: tuple, leaf: Any) -> jax.sharding.NamedSharding:
            if not hasattr(leaf, "shape") or leaf.shape is None or len(leaf.shape) == 0:
                return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())

            path_str = self.policy._path_to_string(path)
            pspec = self.policy.get_partition_spec(path_str)

            # Validate rank compatibility (Debt #3043 Finding M2)
            if len(pspec) > len(leaf.shape):
                raise ShardingPlannerError(
                    f"ShardingPlanner: PartitionSpec for '{path_str}' has rank {len(pspec)} "
                    f"exceeding leaf rank {len(leaf.shape)} (shape {leaf.shape})."
                )

            # Validate divisibility across single and tuple mesh axes (Debt #3043 Finding M2)
            mesh_shape_dict = dict(self.mesh.shape)
            for dim_idx, axis_name in enumerate(pspec):
                if axis_name is None:
                    continue
                dim_size = leaf.shape[dim_idx]
                if isinstance(axis_name, (tuple, list)):
                    total_axis_size = 1
                    for sub_axis in axis_name:
                        if sub_axis not in mesh_shape_dict:
                            raise ShardingPlannerError(
                                f"ShardingPlanner: PartitionSpec for '{path_str}' "
                                f"references unknown mesh axis '{sub_axis}'. "
                                f"Available: {list(mesh_shape_dict.keys())}"
                            )
                        total_axis_size *= mesh_shape_dict[sub_axis]
                    if dim_size % total_axis_size != 0:
                        raise ShardingPlannerError(
                            f"ShardingPlanner: Incompatible placement for '{path_str}'. "
                            f"Dimension {dim_idx} (size {dim_size}) is not divisible by "
                            f"combined mesh axes {axis_name} (product {total_axis_size})."
                        )
                else:
                    if axis_name not in mesh_shape_dict:
                        raise ShardingPlannerError(
                            f"ShardingPlanner: PartitionSpec for '{path_str}' references unknown "
                            f"mesh axis '{axis_name}'. Available: {list(mesh_shape_dict.keys())}"
                        )
                    axis_size = mesh_shape_dict[axis_name]
                    if dim_size % axis_size != 0:
                        raise ShardingPlannerError(
                            f"ShardingPlanner: Incompatible placement for '{path_str}'. "
                            f"Dimension {dim_idx} (size {dim_size}) is not divisible by "
                            f"mesh axis '{axis_name}' (size {axis_size})."
                        )

            return jax.sharding.NamedSharding(self.mesh, pspec)

        return jax.tree_util.tree_map_with_path(_plan_leaf, pytree)


# Deprecated in the #6107 probe release: all three names warn on access and are otherwise
# unchanged.
_DEPRECATED = {
    "get_device_mesh": Deprecated(
        get_device_mesh,
        "jax.make_mesh(shape, axis_names)",
    ),
    "get_hardware_mesh_profile": Deprecated(
        get_hardware_mesh_profile,
        "jax.make_mesh((jax.device_count(),), ('data',)); the profile only ever suggested "
        "one data-parallel axis over all devices",
    ),
}
del get_device_mesh, get_hardware_mesh_profile
__getattr__ = deprecated_getattr(__name__, _DEPRECATED)
