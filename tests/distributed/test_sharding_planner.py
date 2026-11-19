"""Tests for active ShardingPlanner multi-device placement (#3043)."""

import jax
import jax.numpy as jnp
import jax.sharding
import numpy as np
import pytest

from xtrax.distributed.sharding import (
    ShardingPlanner,
    ShardingPlannerError,
    ShardingPolicy,
)


class TestShardingPlanner:
    """Verification controls for active ShardingPlanner."""

    @pytest.fixture
    def mock_mesh_2x2(self):
        """Create a 2x2 mock device mesh."""
        devices = np.array(jax.devices()[:1] * 4).reshape((2, 2))
        return jax.sharding.Mesh(devices, ("data", "model"))

    def test_plan_valid_placement(self, mock_mesh_2x2):
        """Valid placement produces NamedSharding across PyTree leaves."""
        policy = ShardingPolicy(
            rules=(
                ("weight", jax.sharding.PartitionSpec("data", "model")),
                ("bias", jax.sharding.PartitionSpec("data")),
            )
        )
        planner = ShardingPlanner(mesh=mock_mesh_2x2, policy=policy)

        # weight shape (8, 4): dim 0 is 8 (divisible by data=2), dim 1 is 4 (divisible by model=2)
        # bias shape (8,): dim 0 is 8 (divisible by data=2)
        pytree = {
            "weight": jnp.zeros((8, 4)),
            "bias": jnp.zeros((8,)),
            "scalar": 1.0,
        }

        sharded = planner.plan_sharding(pytree)

        assert isinstance(sharded["weight"], jax.sharding.NamedSharding)
        assert sharded["weight"].spec == jax.sharding.PartitionSpec("data", "model")
        assert sharded["weight"].mesh == mock_mesh_2x2

        assert isinstance(sharded["bias"], jax.sharding.NamedSharding)
        assert sharded["bias"].spec == jax.sharding.PartitionSpec("data")

        # Scalar has no shape -> fully replicated
        assert isinstance(sharded["scalar"], jax.sharding.NamedSharding)
        assert sharded["scalar"].spec == jax.sharding.PartitionSpec()

    def test_plan_with_tuple_mesh_axes(self, mock_mesh_2x2):
        """PartitionSpec with tuple of mesh axes validates product divisibility (Finding M2)."""
        # Dim 0 sharded over combined ('data', 'model'): product is 2 * 2 = 4
        policy = ShardingPolicy(
            rules=(("matrix", jax.sharding.PartitionSpec(("data", "model"), None)),)
        )
        planner = ShardingPlanner(mesh=mock_mesh_2x2, policy=policy)

        # Shape (8, 5): 8 is divisible by 4
        valid_tree = {"matrix": jnp.zeros((8, 5))}
        sharded = planner.plan_sharding(valid_tree)
        assert sharded["matrix"].spec == jax.sharding.PartitionSpec(("data", "model"), None)

        # Shape (6, 5): 6 is NOT divisible by 4 -> raises ShardingPlannerError
        invalid_tree = {"matrix": jnp.zeros((6, 5))}
        with pytest.raises(ShardingPlannerError, match="not divisible by combined mesh axes"):
            planner.plan_sharding(invalid_tree)

    def test_partition_spec_rank_exceeding_leaf_rank_raises_error(self, mock_mesh_2x2):
        """PartitionSpec rank exceeding tensor rank raises ShardingPlannerError (Finding M2)."""
        policy = ShardingPolicy(
            rules=(("vec", jax.sharding.PartitionSpec("data", "model")),)  # 2D spec
        )
        planner = ShardingPlanner(mesh=mock_mesh_2x2, policy=policy)

        # 1D tensor with 2D spec
        tree = {"vec": jnp.zeros((8,))}
        with pytest.raises(ShardingPlannerError, match="exceeding leaf rank"):
            planner.plan_sharding(tree)

    def test_indivisible_dimension_raises_sharding_planner_error(self, mock_mesh_2x2):
        """Dimension size not divisible by mesh axis size raises ShardingPlannerError."""
        policy = ShardingPolicy(rules=(("weight", jax.sharding.PartitionSpec("data", "model")),))
        planner = ShardingPlanner(mesh=mock_mesh_2x2, policy=policy)

        # dim 1 size is 3, which is not divisible by model (size 2)
        bad_tree = {"weight": jnp.zeros((8, 3))}
        with pytest.raises(ShardingPlannerError, match="not divisible by mesh axis"):
            planner.plan_sharding(bad_tree)

    def test_unknown_mesh_axis_raises_sharding_planner_error(self, mock_mesh_2x2):
        """PartitionSpec referencing non-existent mesh axis raises ShardingPlannerError."""
        policy = ShardingPolicy(rules=(("param", jax.sharding.PartitionSpec("non_existent_axis")),))
        planner = ShardingPlanner(mesh=mock_mesh_2x2, policy=policy)

        tree = {"param": jnp.zeros((8,))}
        with pytest.raises(ShardingPlannerError, match="references unknown mesh axis"):
            planner.plan_sharding(tree)
