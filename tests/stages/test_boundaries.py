"""Test boundaries module for axis-level stacking operations."""

import warnings

import jax

from xtrax.stages import (
    AxisBoundary,
    Fuse,
    FuseFn,  # noqa: F401 AC-4: deprecated path still resolves
    Sink,
    Tap,
)


class TestBoundariesImports:
    """AC-1: Test importing Fuse, Tap, Sink, AxisBoundary from xtrax.stages."""

    def test_imports_from_xtrax_stages(self):
        """All four types should import without ImportError."""
        from xtrax.stages import AxisBoundary, Fuse, Sink, Tap

        assert AxisBoundary is not None
        assert Fuse is not None
        assert Sink is not None
        assert Tap is not None

    def test_fusefn_deprecated_path_still_works(self):
        """AC-4: from xtrax.stages import FuseFn should still work."""
        # FuseFn is deprecated, so importing should raise DeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from xtrax.stages import FuseFn as FuseFnDeprecated

            assert FuseFnDeprecated is not None
            # Filter to only DeprecationWarnings about FuseFn
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "FuseFn is deprecated" in str(warning.message)
            ]
            assert len(deprecation_warnings) >= 1
            assert "FuseFn is deprecated" in str(deprecation_warnings[0].message)


class TestAxisBoundaryRoundTrip:
    """AC-2: AxisBoundary eqx round-trip test."""

    def test_axis_boundary_all_none_tree_flatten(self):
        """eqx.tree_flatten on AxisBoundary() with all-None fields should round-trip."""
        ab = AxisBoundary()

        # tree_flatten should succeed
        leaves, treedef = jax.tree_util.tree_flatten(ab)

        # Since all fields are static, leaves should be empty
        assert leaves == []

        # tree_unflatten should reconstruct
        ab_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert ab_reconstructed.fuse is None
        assert ab_reconstructed.tap is None
        assert ab_reconstructed.sink is None

    def test_axis_boundary_with_callable_static_fields(self):
        """AxisBoundary fields are static, so callables don't enter leaves."""
        # Create simple callables
        fuse_fn = lambda x: x  # noqa: E731
        tap_fn = lambda x: x  # noqa: E731
        sink_fn = lambda x: None  # noqa: E731

        ab = AxisBoundary(fuse=fuse_fn, tap=tap_fn, sink=sink_fn)  # type: ignore

        # tree_flatten should succeed (all fields are static)
        leaves, treedef = jax.tree_util.tree_flatten(ab)

        # All fields are static, so leaves should be empty
        assert leaves == []

        # tree_unflatten should reconstruct
        ab_reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert ab_reconstructed.fuse is fuse_fn
        assert ab_reconstructed.tap is tap_fn
        assert ab_reconstructed.sink is sink_fn


class TestFuseIsinstanceCheck:
    """AC-3: isinstance check for Fuse protocol."""

    def test_callable_with_call_method_is_fuse(self):
        """A class with __call__(self, stacked) should pass isinstance(obj, Fuse)."""

        class SimpleReducer:
            def __call__(self, stacked):
                return stacked[0]

        obj = SimpleReducer()
        assert isinstance(obj, Fuse)

    def test_lambda_is_fuse(self):
        """A lambda with single argument should pass isinstance check."""
        fn = lambda x: x * 2  # noqa: E731
        assert isinstance(fn, Fuse)

    def test_builtin_callable_is_fuse(self):
        """A builtin like sum should pass isinstance check."""
        assert isinstance(sum, Fuse)


class TestTapIsinstanceCheck:
    """isinstance checks for Tap protocol."""

    def test_callable_with_ordered_attribute_is_tap(self):
        """Callable with __call__ and ordered attribute passes isinstance for Tap."""

        class LoggingTap:
            ordered = True

            def __call__(self, x):
                return x

        obj = LoggingTap()
        assert isinstance(obj, Tap)

    def test_callable_without_ordered_attribute_fails_tap(self):
        """A callable without ordered attribute should not pass isinstance check."""

        class NoOrderedAttr:
            def __call__(self, x):
                return x

        obj = NoOrderedAttr()
        # Should fail isinstance because ordered attribute is missing
        assert not isinstance(obj, Tap)


class TestSinkIsinstanceCheck:
    """isinstance checks for Sink protocol."""

    def test_callable_with_ordered_returns_none_is_sink(self):
        """Callable with __call__ and ordered attribute passes isinstance for Sink."""

        class WriterSink:
            ordered = False

            def __call__(self, x):
                # In reality would write to file/H5/etc
                pass

        obj = WriterSink()
        assert isinstance(obj, Sink)

    def test_callable_without_ordered_fails_sink(self):
        """A callable without ordered attribute should not pass isinstance check."""

        class NoOrderedAttr:
            def __call__(self, x):
                pass

        obj = NoOrderedAttr()
        # Should fail isinstance because ordered attribute is missing
        assert not isinstance(obj, Sink)
