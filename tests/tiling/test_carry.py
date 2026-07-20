"""Tests for xtrax.tiling.carry — CarrySpec."""

import pytest

from xtrax.tiling.carry import CarrySpec


class TestCarrySpecInstantiation:
    """CarrySpec instantiates with any axis name; validation is delegated to BatchPlanner."""

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

    def test_carryspec_instantiates_with_any_axis_name(self):
        """CarrySpec instantiates with any axis name; validation is delegated to BatchPlanner."""

        def transition(carry, x):
            return carry, x

        # CarrySpec itself does not validate heterogeneous axes.
        # Validation is delegated to BatchPlanner, which checks against
        # the heterogeneous_axes parameter.
        spec_states = CarrySpec(
            axis_name="n_states",
            init=0,
            transition=transition,
        )
        assert spec_states.axis_name == "n_states"

        spec_structures = CarrySpec(
            axis_name="n_structures",
            init=0,
            transition=transition,
        )
        assert spec_structures.axis_name == "n_structures"

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

    def test_carryspec_collect_outputs_defaults_true(self):
        """CarrySpec.collect_outputs defaults to True (preserves existing Scan behavior)."""

        def transition(carry, x):
            return carry, x

        spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
        )
        assert spec.collect_outputs is True

    def test_carryspec_cond_defaults_none(self):
        """CarrySpec.cond defaults to None (unused when collect_outputs=True)."""

        def transition(carry, x):
            return carry, x

        spec = CarrySpec(
            axis_name="n_samples",
            init=0,
            transition=transition,
        )
        assert spec.cond is None

    def test_carryspec_collect_outputs_false_with_while_body(self):
        """CarrySpec accepts collect_outputs=False with a carry->carry body and a cond."""

        def body(carry):
            return carry + 1

        def cond(carry):
            return carry < 10

        spec = CarrySpec(
            axis_name="n_steps",
            init=0,
            transition=body,
            collect_outputs=False,
            cond=cond,
        )
        assert spec.collect_outputs is False
        assert spec.transition is body
        assert spec.cond is cond
