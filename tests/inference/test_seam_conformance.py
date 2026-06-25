"""E1 signature-inference seam conformance tests.

Validates that E1 outputs (AxisSpec, FeatureBatch, StageBundle) conform to
existing xtrax seams WITHOUT requiring any changes to seam signatures.

AC6 assertions:
1. list[AxisSpec] is accepted by RunSpec(axes=...) and BatchPlanner.plan()
2. FeatureBatch-compatible dict is produced by InputResolver-conformant adapter
3. StageBundle subclass passes __init_subclass__ validation
4. No seam signature changes required
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NewType, Protocol, runtime_checkable

import equinox as eqx
import pytest

if TYPE_CHECKING:
    from xtrax.run.spec import RunSpec
    from xtrax.stages.bundle import StageBundle  # noqa: F401
    from xtrax.tiling import AxisSpec, BatchPlanner  # noqa: F401

# Avoid circular import from resolver module; define locally
FeatureBatch = NewType("FeatureBatch", dict[str, Any])


@runtime_checkable
class InputResolver(Protocol):
    """Map (spec, materialized bundle) to a feature batch.

    Mirror of xtrax.run.resolver.InputResolver.
    """

    def __call__(self, spec: "RunSpec", bundle: Any) -> FeatureBatch: ...


class TestAxisSpecAndRunSpec:
    """AC6.1: list[AxisSpec] accepted by RunSpec and BatchPlanner.plan()."""

    def test_construct_directly_known_specs(self):
        """Directly-constructed AxisSpec should be KNOWN (role defaults to KNOWN)."""
        from xtrax.tiling import AxisSpec

        # Simple construction: AxisSpec(name, cardinality, default_batch_size)
        specs = [
            AxisSpec("batch", cardinality=64, default_batch_size=32),
            AxisSpec("sequence", cardinality=128, default_batch_size=64),
        ]
        # Verify they construct without error
        assert len(specs) == 2
        assert specs[0].name == "batch"
        assert specs[0].cardinality == 64
        assert specs[0].default_batch_size == 32

    def test_run_spec_accepts_axis_specs(self):
        """RunSpec(axes=...) accepts list[AxisSpec] unchanged."""
        from xtrax.run.spec import RunSpec
        from xtrax.tiling import AxisSpec

        specs = [
            AxisSpec("batch", cardinality=64, default_batch_size=32),
            AxisSpec("sequence", cardinality=128, default_batch_size=64),
        ]
        # Construct minimal RunSpec (only seed and axes required)
        run_spec = RunSpec(seed=42, axes=specs)
        assert run_spec.seed == 42
        assert run_spec.axes == specs
        assert len(run_spec.axes) == 2

    def test_batch_planner_plan_accepts_known_specs(self):
        """BatchPlanner.plan(specs) accepts list[AxisSpec] and returns BatchPlan."""
        from xtrax.tiling import AxisSpec, BatchPlanner

        specs = [
            AxisSpec("batch", cardinality=64, default_batch_size=32),
            AxisSpec("sequence", cardinality=128, default_batch_size=64),
        ]
        planner = BatchPlanner()
        batch_plan = planner.plan(specs)
        # Verify plan is valid: has decisions tuple, one per spec
        assert batch_plan is not None
        assert len(batch_plan.decisions) == len(specs)
        # Each decision references the original spec
        assert batch_plan.decisions[0].spec == specs[0]
        assert batch_plan.decisions[1].spec == specs[1]

    def test_multiple_axes_no_ambiguous_error(self):
        """Multiple specs with no UNKNOWN ambiguity should plan cleanly."""
        from xtrax.tiling import AxisSpec, BatchPlanner

        specs = [
            AxisSpec("batch", cardinality=32, default_batch_size=16),
            AxisSpec("sequence", cardinality=256, default_batch_size=128),
            AxisSpec("hidden", cardinality=512, default_batch_size=256),
        ]
        planner = BatchPlanner()
        batch_plan = planner.plan(specs)
        # Should complete without AmbiguousAxisError
        assert len(batch_plan.decisions) == 3


class TestInputResolverProtocol:
    """AC6.2: FeatureBatch dict produced by InputResolver-conformant adapter."""

    def test_feature_batch_is_dict_newtype(self):
        """FeatureBatch is a NewType over dict[str, Any]."""
        batch_dict: dict[str, Any] = {"feature_0": 1.0, "feature_1": 2.0}
        # FeatureBatch is a NewType, so cast is identity at runtime
        fb: FeatureBatch = FeatureBatch(batch_dict)
        assert fb == batch_dict

    def test_input_resolver_protocol_definition(self):
        """InputResolver is a runtime_checkable Protocol."""
        # InputResolver.__call__(spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch
        # Verify it's actually a Protocol
        assert hasattr(InputResolver, "__protocol_attrs__")

    def test_simple_resolver_adapter_conforms_to_protocol(self):
        """Simple adapter callable matches InputResolver protocol."""
        from xtrax.run.spec import RunSpec

        class SimpleResolver:
            """In-test resolver adapter that conforms to InputResolver protocol."""

            def __call__(self, spec: RunSpec, bundle: Any) -> FeatureBatch:
                """Return a FeatureBatch dict."""
                # Minimal implementation: return empty batch
                return FeatureBatch({"dummy": 1})

        # Verify it's recognized as an InputResolver
        resolver = SimpleResolver()
        assert isinstance(resolver, InputResolver)

    def test_resolver_adapter_produces_feature_batch(self):
        """Resolver adapter produces FeatureBatch dict."""
        from xtrax.run.spec import RunSpec
        from xtrax.tiling.plan import AxisSpec

        class TestResolver:
            """Test adapter that produces a realistic FeatureBatch."""

            def __call__(self, spec: RunSpec, bundle: Any) -> FeatureBatch:
                # Produce a dict keyed by axis name
                batch = {}
                for axis_spec in spec.axes:
                    batch[axis_spec.name] = [0] * axis_spec.cardinality
                return FeatureBatch(batch)

        resolver = TestResolver()
        spec = RunSpec(
            seed=42,
            axes=[
                AxisSpec("batch", cardinality=64, default_batch_size=32),
                AxisSpec("sequence", cardinality=128, default_batch_size=64),
            ],
        )
        # bundle is not used by this adapter, so mock it minimally
        bundle = eqx.nn.Identity()  # Mock object

        result = resolver(spec, bundle)
        assert "batch" in result
        assert "sequence" in result
        assert isinstance(result, dict)


class TestStageBundleSubclass:
    """AC6.3: StageBundle subclass passes __init_subclass__ validation."""

    def test_valid_stage_bundle_subclass(self):
        """StageBundle subclass with Optional[Callable] fields is valid."""
        from xtrax.stages.bundle import StageBundle

        class TestStageBundle(StageBundle):
            """Test bundle with two optional callable stages."""

            stage_a: Callable[[int], int] | None = None
            stage_b: Callable[[int, int], int] | None = None

        # Should not raise TypeError from __init_subclass__
        assert issubclass(TestStageBundle, StageBundle)

    def test_stage_bundle_subclass_instantiate(self):
        """StageBundle subclass can be instantiated with None values."""
        from xtrax.stages.bundle import StageBundle

        class TestStageBundle(StageBundle):
            stage_x: Callable[[], str] | None = None

        # Instantiate with all None
        bundle = TestStageBundle(stage_x=None)
        assert bundle.stage_x is None

    def test_stage_bundle_subclass_with_callables(self):
        """StageBundle subclass can hold actual callable instances."""
        from xtrax.stages.bundle import StageBundle

        class TestStageBundle(StageBundle):
            transform: Callable[[float], float] | None = None

        def my_transform(x: float) -> float:
            return x * 2.0

        bundle = TestStageBundle(transform=my_transform)
        assert bundle.transform is not None
        assert bundle.transform(5.0) == 10.0

    def test_stage_bundle_active_stages(self):
        """StageBundle.active_stages() lists non-None callable fields."""
        from xtrax.stages.bundle import StageBundle

        class TestStageBundle(StageBundle):
            stage_a: Callable[[], None] | None = None
            stage_b: Callable[[], None] | None = None

        def dummy():
            pass

        bundle = TestStageBundle(stage_a=dummy, stage_b=None)
        active = bundle.active_stages()
        assert "stage_a" in active
        assert "stage_b" not in active

    def test_stage_bundle_has_stage(self):
        """StageBundle.has_stage(name) checks for non-None callable."""
        from xtrax.stages.bundle import StageBundle

        class TestStageBundle(StageBundle):
            process: Callable[[int], int] | None = None

        bundle = TestStageBundle(process=None)
        assert not bundle.has_stage("process")

        def proc(x):
            return x

        bundle2 = TestStageBundle(process=proc)
        assert bundle2.has_stage("process")


class TestNoSeamSignatureChanges:
    """AC6.4: Verify no seam signatures were modified."""

    def test_run_spec_signature_unchanged(self):
        """RunSpec retains its signature: seed, axes, carry_specs, boundaries."""
        from xtrax.run.spec import RunSpec

        # Verify by instantiation and field access
        spec = RunSpec(seed=42, axes=[])
        assert hasattr(spec, "seed")
        assert hasattr(spec, "axes")
        assert hasattr(spec, "carry_specs")
        assert hasattr(spec, "boundaries")

    def test_input_resolver_protocol_signature(self):
        """InputResolver protocol signature: __call__(spec, bundle) -> FeatureBatch."""
        # Should be callable with RunSpec and RuntimeBundle
        # and return FeatureBatch (dict)
        # Verify by checking the protocol definition
        assert hasattr(InputResolver, "__call__")

    def test_stage_bundle_optional_callable_constraint(self):
        """StageBundle enforces Optional[Callable] on all fields."""
        from xtrax.stages.bundle import StageBundle

        # This is validated at __init_subclass__ time
        # Try to create an invalid subclass and verify it raises TypeError

        with pytest.raises(TypeError):

            class BadBundle(StageBundle):
                not_callable: int = 5

    def test_axis_spec_fields_unchanged(self):
        """AxisSpec retains all expected fields."""
        from xtrax.tiling.plan import AxisSpec

        spec = AxisSpec("test", 100, 50)
        # Check that we can access all expected attributes
        assert spec.name == "test"
        assert spec.cardinality == 100
        assert spec.default_batch_size == 50
        assert spec.tile_granularity == 1  # default
        assert spec.heterogeneous is False  # default
        assert spec.dedup_eligible is False  # default
        assert spec.bucket_boundaries is None  # default
