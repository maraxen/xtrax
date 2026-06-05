"""Distributed sharding utilities for JAX arrays and pytrees."""

import math
import re
from typing import Any

import equinox as eqx
import jax
import jax.sharding
import jax.tree_util


class ShardingPolicy(eqx.Module):
    """Policy for assigning PartitionSpec to pytree leaves.

    Uses regex pattern matching where first match wins. Matching is done via re.search,
    so patterns can match anywhere in the path.

    Attributes:
        rules: Ordered tuple of (regex_pattern, PartitionSpec) pairs.
    """

    rules: tuple[tuple[str, jax.sharding.PartitionSpec], ...] = eqx.field(static=True)

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
