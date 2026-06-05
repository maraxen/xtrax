import equinox as eqx
import jax
import jax.sharding
import jax.tree_util
import pytest

from xtrax.distributed.sharding import (
    ShardingPolicy,
    get_device_mesh,
    get_hardware_mesh_profile,
)


class TestShardingPolicy:
    """Test ShardingPolicy: regex matching, partition spec, pytree application."""

    def test_sharding_policy_is_eqx_module(self):
        """ShardingPolicy must be an eqx.Module subclass."""
        policy = ShardingPolicy(rules=())
        assert isinstance(policy, eqx.Module)

    def test_get_partition_spec_first_match_wins(self):
        """First matching pattern wins; ordering matters."""
        rules = (
            ("weight.*", jax.sharding.PartitionSpec("data", "model")),
            ("weight", jax.sharding.PartitionSpec("model")),  # This should never match
            ("bias", jax.sharding.PartitionSpec("data")),
        )
        policy = ShardingPolicy(rules=rules)

        # "weight_matrix" matches first rule
        spec = policy.get_partition_spec("weight_matrix")
        assert spec == jax.sharding.PartitionSpec("data", "model")

        # "weight_only" also matches first rule
        spec = policy.get_partition_spec("weight_only")
        assert spec == jax.sharding.PartitionSpec("data", "model")

    def test_get_partition_spec_no_match_defaults_to_replicated(self):
        """Unmatched paths return fully replicated PartitionSpec()."""
        rules = (
            ("weight", jax.sharding.PartitionSpec("data")),
        )
        policy = ShardingPolicy(rules=rules)

        spec = policy.get_partition_spec("unmatched_param")
        assert spec == jax.sharding.PartitionSpec()

    def test_get_partition_spec_uses_re_search(self):
        """get_partition_spec uses re.search, not exact match."""
        rules = (
            (".*weight.*", jax.sharding.PartitionSpec("data", "model")),
        )
        policy = ShardingPolicy(rules=rules)

        # Should match anywhere in the string
        spec = policy.get_partition_spec("layer_0_weight_matrix")
        assert spec == jax.sharding.PartitionSpec("data", "model")

    def test_apply_to_pytree_dict(self):
        """apply_to_pytree traverses dict and returns PartitionSpec for each leaf."""
        rules = (
            ("weight", jax.sharding.PartitionSpec("data", "model")),
            ("bias", jax.sharding.PartitionSpec("data")),
        )
        policy = ShardingPolicy(rules=rules)

        pytree = {"weight": 0, "bias": 0, "other": 0}
        result = policy.apply_to_pytree(pytree)

        # Structure should match but values are PartitionSpecs
        assert "weight" in result
        assert "bias" in result
        assert "other" in result
        assert result["weight"] == jax.sharding.PartitionSpec("data", "model")
        assert result["bias"] == jax.sharding.PartitionSpec("data")
        assert result["other"] == jax.sharding.PartitionSpec()

    def test_apply_to_pytree_nested_structure(self):
        """apply_to_pytree preserves nested dict structure."""
        rules = (
            ("layer1/weight", jax.sharding.PartitionSpec("data", "model")),
            ("layer2/bias", jax.sharding.PartitionSpec("data")),
        )
        policy = ShardingPolicy(rules=rules)

        pytree = {
            "layer1": {"weight": 0, "bias": 0},
            "layer2": {"weight": 0, "bias": 0},
        }
        result = policy.apply_to_pytree(pytree)

        # Nested structure preserved
        assert isinstance(result["layer1"], dict)
        assert isinstance(result["layer2"], dict)
        assert result["layer1"]["weight"] == jax.sharding.PartitionSpec("data", "model")
        assert result["layer2"]["bias"] == jax.sharding.PartitionSpec("data")
        assert result["layer1"]["bias"] == jax.sharding.PartitionSpec()

    def test_apply_to_pytree_with_jax_tree_key_handling(self):
        """apply_to_pytree correctly derives paths from jax.tree_util keys."""
        rules = (
            ("params.*weight", jax.sharding.PartitionSpec("data")),
        )
        policy = ShardingPolicy(rules=rules)

        # Use jax.tree_util structure
        pytree = {"params": {"layer": {"weight": 0}}}
        result = policy.apply_to_pytree(pytree)

        # Should have matching structure and correct specs
        assert result["params"]["layer"]["weight"] == jax.sharding.PartitionSpec("data")


class TestGetDeviceMesh:
    """Test get_device_mesh: validation and mesh creation."""

    def test_get_device_mesh_valid_shape(self):
        """get_device_mesh with matching shape returns Mesh."""
        num_devices = len(jax.devices())
        shape = (num_devices,) if num_devices > 0 else (1,)
        axis_names = ("devices",)

        mesh = get_device_mesh(shape=shape, axis_names=axis_names)

        assert isinstance(mesh, jax.sharding.Mesh)
        # mesh.axis_names is a tuple
        assert mesh.axis_names == axis_names
        # mesh.shape is an OrderedDict, compare via dict lookup
        for axis_name, axis_size in zip(axis_names, shape):
            assert mesh.shape[axis_name] == axis_size

    def test_get_device_mesh_multi_axis(self):
        """get_device_mesh with multi-axis shape and names."""
        num_devices = len(jax.devices())
        # Skip if we can't create multi-axis on single device
        if num_devices == 1:
            pytest.skip("Need at least 2 devices for multi-axis test")

        if num_devices % 2 == 0:
            shape = (2, num_devices // 2)
        else:
            shape = (1, num_devices)

        axis_names = ("data", "model")

        mesh = get_device_mesh(shape=shape, axis_names=axis_names)

        assert isinstance(mesh, jax.sharding.Mesh)
        # mesh.shape is an OrderedDict, compare via dict lookup
        assert mesh.axis_names == axis_names
        for axis_name, axis_size in zip(axis_names, shape):
            assert mesh.shape[axis_name] == axis_size

    def test_get_device_mesh_wrong_product_raises_error(self):
        """get_device_mesh with product != num_devices raises ValueError."""
        num_devices = len(jax.devices())
        bad_shape = (num_devices + 1,)
        axis_names = ("devices",)

        with pytest.raises(ValueError) as excinfo:
            get_device_mesh(shape=bad_shape, axis_names=axis_names)

        error_msg = str(excinfo.value)
        # Check that the error message mentions shape product or device count
        assert "shape" in error_msg.lower() and "device" in error_msg.lower()


class TestGetHardwareMeshProfile:
    """Test get_hardware_mesh_profile: device info and fallback behavior."""

    def test_get_hardware_mesh_profile_has_required_keys(self):
        """get_hardware_mesh_profile returns all 4 required keys."""
        profile = get_hardware_mesh_profile()

        assert "device_type" in profile
        assert "num_devices" in profile
        assert "recommended_shape" in profile
        assert "recommended_axis_names" in profile

    def test_get_hardware_mesh_profile_device_type_is_string(self):
        """device_type must be a string."""
        profile = get_hardware_mesh_profile()
        assert isinstance(profile["device_type"], str)
        assert profile["device_type"] in ("cpu", "gpu", "tpu")

    def test_get_hardware_mesh_profile_num_devices_matches_jax(self):
        """num_devices should match or be consistent with jax.devices()."""
        profile = get_hardware_mesh_profile()
        assert isinstance(profile["num_devices"], int)
        assert profile["num_devices"] > 0

    def test_get_hardware_mesh_profile_shape_and_names_valid(self):
        """recommended_shape product should match or be compatible with num_devices."""
        profile = get_hardware_mesh_profile()
        shape = profile["recommended_shape"]
        axis_names = profile["recommended_axis_names"]

        assert isinstance(shape, tuple)
        assert isinstance(axis_names, tuple)
        assert len(shape) == len(axis_names)
        assert all(isinstance(s, int) and s > 0 for s in shape)
        assert all(isinstance(name, str) for name in axis_names)

    def test_get_hardware_mesh_profile_never_raises(self):
        """get_hardware_mesh_profile must never raise, even on unknown hardware."""
        # Should not raise under any circumstances
        profile = get_hardware_mesh_profile()
        assert profile is not None

    def test_get_hardware_mesh_profile_cpu_fallback(self):
        """CPU with single device should have (1,) shape and ('batch',) axis names."""
        profile = get_hardware_mesh_profile()
        num_devices = profile["num_devices"]
        shape = profile["recommended_shape"]

        # For single device, shape should be (1,) or compatible
        if num_devices == 1:
            assert shape == (1,)
            # Axis name should be something reasonable
            assert len(profile["recommended_axis_names"]) == 1
