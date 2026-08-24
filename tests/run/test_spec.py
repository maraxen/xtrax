"""Tests for xtrax.run module."""

from xtrax.run import FeatureBatch, InputResolver, RunSpec, RuntimeBundle, SinkSpec


def test_run_spec_constructs():
    """RunSpec constructs with required values."""
    spec = RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None)
    assert spec.seed == 0
    assert spec.axes == []
    assert spec.carry_specs == []
    assert spec.boundaries is None


def test_sink_spec_defaults():
    """SinkSpec uses correct default values."""
    s = SinkSpec(run_id="r")
    assert s.format == "jsonl"
    assert s.flush_every == 1
    assert s.output_dir is None


def test_input_resolver_protocol():
    """InputResolver protocol accepts runtime-checkable implementations."""

    class MyResolver:
        def __call__(self, spec, bundle):
            return FeatureBatch({})

    assert isinstance(MyResolver(), InputResolver)


def test_run_spec_pytree_roundtrip():
    """RunSpec survives JAX tree_flatten/unflatten (eqx.Module pytree contract)."""
    import jax

    spec = RunSpec(seed=42, axes=[], carry_specs=[], boundaries=None)
    leaves, treedef = jax.tree_util.tree_flatten(spec)
    spec2 = treedef.unflatten(leaves)
    assert spec2.seed == spec.seed
    assert spec2.axes == spec.axes


def test_runtime_bundle_constructs():
    """RuntimeBundle constructs and exposes its fields."""
    import equinox as eqx

    bundle = RuntimeBundle(iterator=None, model=eqx.nn.Identity())
    assert bundle.iterator is None
    assert isinstance(bundle.model, eqx.Module)
