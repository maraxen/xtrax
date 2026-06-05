"""Test protocols for tier-1 generic stages."""

from xtrax.stages.protocols import FuseFn, RollingFn, TransformFn


class TestTransformFn:
    """TransformFn protocol tests."""

    def test_lambda_passes_isinstance(self):
        """Simple lambda should pass isinstance check."""
        fn = lambda x: x + 1  # noqa: E731
        assert isinstance(fn, TransformFn)

    def test_lambda_with_return_annotation(self):
        """Lambda with explicit typing should pass isinstance."""
        fn: TransformFn = lambda x: str(x)  # noqa: E731
        assert isinstance(fn, TransformFn)

    def test_callable_object_passes_isinstance(self):
        """Callable class instance should pass isinstance check."""

        class AddOne:
            def __call__(self, x):
                return x + 1

        fn = AddOne()
        assert isinstance(fn, TransformFn)


class TestRollingFn:
    """RollingFn protocol tests."""

    def test_lambda_passes_isinstance(self):
        """Simple lambda should pass isinstance check."""
        fn = lambda carry, x: (carry + x, x * 2)  # noqa: E731
        assert isinstance(fn, RollingFn)

    def test_lambda_with_return_annotation(self):
        """Lambda with explicit typing should pass isinstance."""
        fn: RollingFn = lambda carry, x: (carry + 1, x + 1)  # noqa: E731
        assert isinstance(fn, RollingFn)

    def test_callable_object_passes_isinstance(self):
        """Callable class instance should pass isinstance check."""

        class Accumulate:
            def __call__(self, carry, x):
                return (carry + x, carry)

        fn = Accumulate()
        assert isinstance(fn, RollingFn)


class TestFuseFn:
    """FuseFn protocol tests."""

    def test_lambda_passes_isinstance(self):
        """Simple lambda should pass isinstance check."""
        fn = lambda items: sum(items)  # noqa: E731
        assert isinstance(fn, FuseFn)

    def test_lambda_with_return_annotation(self):
        """Lambda with explicit typing should pass isinstance."""
        fn: FuseFn = lambda items: max(items)  # noqa: E731
        assert isinstance(fn, FuseFn)

    def test_callable_object_passes_isinstance(self):
        """Callable class instance should pass isinstance check."""

        class Concatenate:
            def __call__(self, items):
                return "".join(items)

        fn = Concatenate()
        assert isinstance(fn, FuseFn)
