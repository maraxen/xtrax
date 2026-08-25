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


def test_run_spec_run_id_defaults_none():
    """run_id is optional static metadata defaulting to None (#4397)."""
    spec = RunSpec(seed=0, axes=[], carry_specs=[], boundaries=None)
    assert spec.run_id is None


def test_run_spec_run_id_is_static_not_leaf():
    """run_id rides as static aux data, not a pytree leaf (#4397)."""
    import jax

    spec = RunSpec(seed=42, axes=[], carry_specs=[], boundaries=None, run_id="run-abc123def456")
    leaves, _ = jax.tree_util.tree_flatten(spec)
    assert all(leaf != "run-abc123def456" for leaf in leaves)


def test_run_spec_treedef_tracks_run_id():
    """Differing run_id => differing treedef (the documented re-trace caveat).

    Leaves stay identical; only static aux data moves, so jit re-traces per
    distinct run_id and structural equality separates the two specs.
    """
    import jax

    a = RunSpec(seed=1, axes=[], carry_specs=[], boundaries=None, run_id="run-a")
    b = RunSpec(seed=1, axes=[], carry_specs=[], boundaries=None, run_id="run-b")
    leaves_a, treedef_a = jax.tree_util.tree_flatten(a)
    leaves_b, treedef_b = jax.tree_util.tree_flatten(b)
    assert leaves_a == leaves_b
    assert treedef_a != treedef_b
    assert a != b  # eqx structural equality sees the static field
