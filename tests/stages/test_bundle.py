from collections.abc import Callable

import equinox as eqx
import jax.tree_util as jtu
import pytest

from xtrax.stages.bundle import StageBundle


class ConcreteStageBundle(StageBundle):
    """Concrete subclass for testing with typed stage slots."""

    stage_a: Callable | None = None
    stage_b: Callable | None = None
    stage_c: Callable | None = None


class TestActiveStages:
    """Tests for active_stages() method."""

    def test_active_stages_returns_list(self):
        """active_stages should return a list."""
        bundle = ConcreteStageBundle()
        result = bundle.active_stages()
        assert isinstance(result, list)

    def test_active_stages_returns_empty_when_all_none(self):
        """active_stages should return empty list when all fields are None."""
        bundle = ConcreteStageBundle()
        result = bundle.active_stages()
        assert result == []

    def test_active_stages_returns_names_only(self):
        """active_stages should return only field names, not callables."""

        def dummy_fn():
            pass

        bundle = ConcreteStageBundle(stage_a=dummy_fn)
        result = bundle.active_stages()

        # Should be list of strings
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        # Should contain the field name, not the callable
        assert "stage_a" in result
        assert dummy_fn not in result

    def test_active_stages_single_active(self):
        """active_stages should list one field when one is set."""

        def fn():
            pass

        bundle = ConcreteStageBundle(stage_a=fn)
        result = bundle.active_stages()
        assert result == ["stage_a"]

    def test_active_stages_multiple_active(self):
        """active_stages should list all non-None fields."""

        def fn1():
            pass

        def fn2():
            pass

        bundle = ConcreteStageBundle(stage_a=fn1, stage_c=fn2)
        result = bundle.active_stages()

        assert len(result) == 2
        assert "stage_a" in result
        assert "stage_c" in result
        assert "stage_b" not in result

    def test_active_stages_order_consistent(self):
        """active_stages should have consistent ordering."""

        def fn1():
            pass

        def fn2():
            pass

        def fn3():
            pass

        bundle = ConcreteStageBundle(stage_a=fn1, stage_b=fn2, stage_c=fn3)
        result1 = bundle.active_stages()
        result2 = bundle.active_stages()

        assert result1 == result2


class TestHasStage:
    """Tests for has_stage() method."""

    def test_has_stage_returns_bool(self):
        """has_stage should return a bool."""
        bundle = ConcreteStageBundle()
        result = bundle.has_stage("stage_a")
        assert isinstance(result, bool)

    def test_has_stage_true_when_set(self):
        """has_stage should return True when field is set to a callable."""

        def dummy_fn():
            pass

        bundle = ConcreteStageBundle(stage_a=dummy_fn)
        assert bundle.has_stage("stage_a") is True

    def test_has_stage_false_when_none(self):
        """has_stage should return False when field is None."""
        bundle = ConcreteStageBundle(stage_a=None, stage_b=None)
        assert bundle.has_stage("stage_a") is False
        assert bundle.has_stage("stage_b") is False

    def test_has_stage_false_for_missing_field(self):
        """has_stage should return False for non-existent field."""
        bundle = ConcreteStageBundle()
        assert bundle.has_stage("nonexistent") is False

    def test_has_stage_multiple_fields(self):
        """has_stage should distinguish between multiple fields."""

        def fn1():
            pass

        def fn2():
            pass

        bundle = ConcreteStageBundle(stage_a=fn1, stage_b=None, stage_c=fn2)

        assert bundle.has_stage("stage_a") is True
        assert bundle.has_stage("stage_b") is False
        assert bundle.has_stage("stage_c") is True


class TestStageBundlePytree:
    """Tests for pytree compliance."""

    def test_stage_bundle_is_eqx_module(self):
        """StageBundle subclass should be an eqx.Module."""
        bundle = ConcreteStageBundle()
        assert isinstance(bundle, eqx.Module)

    def test_stage_bundle_jax_tree_leaves(self):
        """jax.tree_util.tree_leaves should work on StageBundle."""

        def fn1():
            return 1

        def fn2():
            return 2

        bundle = ConcreteStageBundle(stage_a=fn1, stage_c=fn2)
        leaves = jtu.tree_leaves(bundle)

        # Should be valid pytree leaves
        assert isinstance(leaves, list)

    def test_stage_bundle_with_mixed_callables(self):
        """StageBundle should handle mix of None and callable fields."""

        def fn():
            pass

        bundle = ConcreteStageBundle(stage_a=fn, stage_b=None, stage_c=fn)
        leaves = jtu.tree_leaves(bundle)

        # Both callables should be in leaves
        assert fn in leaves

    def test_stage_bundle_pytree_roundtrip(self):
        """StageBundle should support tree_flatten/tree_unflatten."""

        def fn1():
            pass

        def fn2():
            pass

        bundle = ConcreteStageBundle(stage_a=fn1, stage_b=None, stage_c=fn2)
        leaves, treedef = jtu.tree_flatten(bundle)
        reconstructed = jtu.tree_unflatten(treedef, leaves)

        assert isinstance(reconstructed, ConcreteStageBundle)
        assert reconstructed.has_stage("stage_a")
        assert not reconstructed.has_stage("stage_b")
        assert reconstructed.has_stage("stage_c")


class TestStageBundleNoCall:
    """Tests verifying no __call__ on StageBundle itself."""

    def test_stage_bundle_has_no_call(self):
        """StageBundle base class should not have __call__ method."""
        # The __call__ method should not be on StageBundle itself
        # (subclasses are free to define it, but the base bundle doesn't)
        bundle = ConcreteStageBundle()

        # Verify we can't call the bundle
        with pytest.raises(TypeError):
            bundle()


class TestStageBundleValidation:
    """Tests for __init_subclass__ validation."""

    def test_subclass_with_optional_callable_fields(self):
        """Subclass with Callable | None fields should be valid."""

        # This should not raise
        class ValidBundle(StageBundle):
            stage1: Callable | None = None
            stage2: Callable | None = None

        bundle = ValidBundle()
        assert isinstance(bundle, StageBundle)

    def test_subclass_with_non_callable_field_raises(self):
        """Subclass with non-Callable field raises TypeError at class definition."""
        with pytest.raises(TypeError):

            class InvalidBundle(StageBundle):
                stage1: Callable | None = None
                invalid_field: str = "not a callable"

    def test_subclass_with_non_optional_callable_field_raises(self):
        """Subclass with non-optional Callable field raises TypeError."""
        with pytest.raises(TypeError):

            class InvalidBundle(StageBundle):
                stage1: Callable  # Not Optional
                stage2: Callable | None = None

    def test_subclass_with_untyped_field_raises(self):
        """Subclass with untyped field raises TypeError at class definition."""
        with pytest.raises(TypeError):

            class InvalidBundle(StageBundle):
                stage1: Callable | None = None
                untyped_field = None  # No type annotation

    def test_invalid_non_optional_callable_raises(self):
        """Field annotated as int (non-callable) raises TypeError at class def."""
        with pytest.raises(TypeError):

            class InvalidBundle(StageBundle):
                field1: int  # Not Callable

    def test_optional_int_raises(self):
        """Field annotated as Optional[int] raises TypeError at class def."""

        with pytest.raises(TypeError):

            class InvalidBundle(StageBundle):
                field1: int | None

    def test_list_callable_raises(self):
        """Field annotated as list[Callable] (not Optional) raises TypeError."""
        with pytest.raises(TypeError):

            class InvalidBundle(StageBundle):
                field1: list[Callable]

    def test_optional_callable_ok(self):
        """Field annotated as Optional[Callable] does NOT raise (regression guard)."""

        # Should not raise
        class ValidBundle(StageBundle):
            field1: Callable | None = None

        assert isinstance(ValidBundle(), StageBundle)
