"""Tests for xtrax.tiling.carry_shape — CarryShape."""

import jax.numpy as jnp
import pytest

from xtrax.tiling.carry_shape import CarryShape


class TestCarryShapeInstantiation:
    """CarryShape must instantiate with shape and dtype metadata."""

    def test_carryshape_instantiates(self):
        """CarryShape instantiates with name, shape, and dtype."""
        carry_shape = CarryShape(
            name="sequence",
            shape=(10,),
            dtype=jnp.float32,
        )
        assert carry_shape.name == "sequence"
        assert carry_shape.shape == (10,)
        assert carry_shape.dtype == jnp.float32

    def test_carryshape_materialize_scalar(self):
        """CarryShape.materialize() returns a zero-filled array (scalar)."""
        carry_shape = CarryShape(
            name="counter",
            shape=(),
            dtype=jnp.int32,
        )
        array = carry_shape.materialize()
        assert array.shape == ()
        assert array.dtype == jnp.int32
        assert array == 0

    def test_carryshape_materialize_1d(self):
        """CarryShape.materialize() returns a zero-filled array (1D)."""
        carry_shape = CarryShape(
            name="sequence",
            shape=(5,),
            dtype=jnp.float32,
        )
        array = carry_shape.materialize()
        assert array.shape == (5,)
        assert array.dtype == jnp.float32
        assert jnp.all(array == 0)

    def test_carryshape_materialize_2d(self):
        """CarryShape.materialize() returns a zero-filled array (2D)."""
        carry_shape = CarryShape(
            name="state",
            shape=(3, 4),
            dtype=jnp.float32,
        )
        array = carry_shape.materialize()
        assert array.shape == (3, 4)
        assert array.dtype == jnp.float32
        assert jnp.all(array == 0)

    def test_carryshape_materialize_different_dtypes(self):
        """CarryShape.materialize() respects different dtypes (float32, int32)."""
        for dtype in [jnp.float32, jnp.int32]:
            carry_shape = CarryShape(
                name="test",
                shape=(2,),
                dtype=dtype,
            )
            array = carry_shape.materialize()
            assert array.dtype == dtype

    def test_carryshape_is_frozen(self):
        """CarryShape is immutable (frozen dataclass)."""
        carry_shape = CarryShape(
            name="sequence",
            shape=(5,),
            dtype=jnp.float32,
        )
        with pytest.raises(AttributeError):
            carry_shape.name = "new_name"
