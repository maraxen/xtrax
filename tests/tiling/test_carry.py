"""Tests for xtrax.tiling.carry — CarrySpec."""

import pytest

from xtrax.tiling.carry import CarrySpec
from xtrax.tiling.strategy import ScanTransition


class TestCarrySpecInstantiation:
    """CarrySpec must instantiate with valid axis names."""

    def test_carryspec_instantiates_with_valid_axis(self):
        """CarrySpec instantiates with a non-heterogeneous axis name."""
        def transition(carry, x):
            return carry, x

        spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
        )
        assert spec.axis_name == "n_samples"
        assert spec.init == 0
        assert spec.transition is transition
        assert spec.ordered_sinks is True

    def test_carryspec_rejects_heterogeneous_n_states(self):
        """CarrySpec raises ValueError for 'n_states' (known heterogeneous axis)."""
        def transition(carry, x):
            return carry, x

        with pytest.raises(ValueError, match="heterogeneous.*n_states"):
            CarrySpec(
                axis_name="n_states",
                init=0,
                transition=transition,
            )

    def test_carryspec_rejects_heterogeneous_n_structures(self):
        """CarrySpec raises ValueError for 'n_structures' (known heterogeneous axis)."""
        def transition(carry, x):
            return carry, x

        with pytest.raises(ValueError, match="heterogeneous.*n_structures"):
            CarrySpec(
                axis_name="n_structures",
                init=0,
                transition=transition,
            )

    def test_carryspec_is_frozen(self):
        """CarrySpec is immutable (frozen dataclass)."""
        def transition(carry, x):
            return carry, x

        spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
        )
        with pytest.raises(AttributeError):
            spec.axis_name = "n_new_axis"

    def test_carryspec_ordered_sinks_default_true(self):
        """CarrySpec.ordered_sinks defaults to True."""
        def transition(carry, x):
            return carry, x

        spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
        )
        assert spec.ordered_sinks is True

    def test_carryspec_ordered_sinks_can_be_false(self):
        """CarrySpec.ordered_sinks can be set to False."""
        def transition(carry, x):
            return carry, x

        spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
            ordered_sinks=False,
        )
        assert spec.ordered_sinks is False
