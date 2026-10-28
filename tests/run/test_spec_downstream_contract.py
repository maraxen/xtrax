"""Downstream RunSpec subclass contract (equinox decoupling step 4).

aminx and plegadx subclass ``xtrax.run.RunSpec`` with undecorated ``class X(RunSpec):``
bodies. This file copies those field declarations faithfully, so a change of the base
(``eqx.Module`` to ``xtrax.Struct``) is checked against the shapes that consume it. It was
written and run green against the ``eqx.Module`` base first, then kept green on ``Struct``.

Sources (read-only copies, field declarations verbatim):

- aminx ``src/aminx/run/spec.py`` at origin/main a6e1573ba62c2fd3b48caed143406eb4e3bb1743
  (``IOConfig`` :30 ... ``SamplingConfig`` :101, ``RunSpec`` :144). The TYPE_CHECKING-only
  annotations (``PottsMPNNOptions``, ``ProtonPottsOptions``, ``LaserOptions``,
  ``EncodingFusionFn``, ``DecodingFusionFn``) are written ``Any`` here: annotations do not
  enter the dataclass field machinery, only the names and defaults do.
- plegadx ``src/plegadx/infra/runspec.py`` at origin/main
  cb166601359aa1fd0e392c9694ffd3dfac838ed8 (``RF3RunSpec`` :43, ``ChaiRunSpec`` :353,
  ``AF3RunSpec`` :839).
"""

import copy
import dataclasses
import hashlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import numpy as np
import pytest

from xtrax.run import RunSpec, canonical_digest, derive_sink_spec
from xtrax.tiling import AxisSpec

# ---------------------------------------------------------------------------
# aminx shape: eqx sub-modules as data fields, dataclasses.field(default_factory=...),
# eqx.field(static=True) mixed on one subclass (aminx run/spec.py:30-160).
# ---------------------------------------------------------------------------


class IOConfig(eqx.Module):
    """aminx run/spec.py:30."""

    output_dir: Path | None = eqx.field(static=True)
    sink_kind: str = eqx.field(static=True)
    manifest_path: Path | None = eqx.field(static=True)
    output_h5_path: Path | None = eqx.field(static=True)
    cache_path: Path | None = eqx.field(static=True)


class ResourceConfig(eqx.Module):
    """aminx run/spec.py:40."""

    n_devices: int = eqx.field(static=True)
    sample_batch_size: int = eqx.field(static=True)
    structure_batch_size: int = eqx.field(static=True)
    max_buffer_size: int | None = eqx.field(static=True)


class PrecisionConfig(eqx.Module):
    """aminx run/spec.py:78."""

    compute: Literal["fp32", "fp16", "bf16"] = eqx.field(static=True)


class PlannerTopology(eqx.Module):
    """aminx run/spec.py:84."""

    use_unified_driver: bool = eqx.field(static=True)


class SamplingConfig(eqx.Module):
    """aminx run/spec.py:101 (``bias`` and the other ``Any = None`` fields are data)."""

    num_samples: int = eqx.field(static=True)
    random_seed: int = eqx.field(static=True)
    return_logits: bool = eqx.field(static=True)
    compute_pseudo_perplexity: bool = eqx.field(static=True)
    return_decoding_orders: bool = eqx.field(static=True)
    return_logit_fingerprint: bool = eqx.field(static=True)
    backbone_noise: tuple[float, ...] = eqx.field(static=True)
    temperature: tuple[float | None, ...] = eqx.field(static=True)
    omit_aa: tuple[str, ...] = eqx.field(static=True, default=())
    omit_aa_per_position: Any = eqx.field(static=True, default=None)
    bias: Any = None
    fixed_mask: Any = None
    fixed_positions: Any = None
    fixed_tokens: Any = None
    sampling_strategy: str = eqx.field(static=True, default="temperature")
    multi_state_strategy: str = eqx.field(static=True, default="arithmetic_mean")
    multi_state_temperature: float = eqx.field(static=True, default=1.0)
    multi_state_sharpness: float | None = eqx.field(static=True, default=1.0)
    use_rolling_state: bool = eqx.field(static=True, default=False)
    decoding_order_fn: Any = eqx.field(static=True, default=None)
    tie_group_map: Any = None
    state_position_map: Any = None
    state_weights: Any = None
    structure_mapping: Any = None


class AminxRunSpec(RunSpec):
    """aminx run/spec.py:144 (class name differs: the xtrax base is imported as RunSpec)."""

    io: IOConfig = dataclasses.field(default_factory=lambda: None)  # type: ignore
    resource: ResourceConfig = dataclasses.field(default_factory=lambda: None)  # type: ignore
    # aminx names this ``precision_config``: the xtrax base owns ``precision: str | None``.
    precision_config: PrecisionConfig = dataclasses.field(default_factory=lambda: None)  # type: ignore
    plan: PlannerTopology = dataclasses.field(default_factory=lambda: None)  # type: ignore
    sampling: SamplingConfig = dataclasses.field(default_factory=lambda: None)  # type: ignore
    potts_mpnn: Any = eqx.field(static=True, default=None)
    protonpotts: Any = eqx.field(static=True, default=None)
    laser: Any = eqx.field(static=True, default=None)
    encoding_fusion: Any = eqx.field(static=True, default=None)
    decoding_fusion: Any = eqx.field(static=True, default=None)


# ---------------------------------------------------------------------------
# plegadx shape: undecorated subclasses with eqx.field(static=True) scalars
# (plegadx infra/runspec.py:43, 353, 839).
# ---------------------------------------------------------------------------

RF3_BACKEND_ID = "rf3"
RF3_ORACLE_ID = "vendor_rf3"
CHAI_BACKEND_ID = "chai"
CHAI_ORACLE_ID = "vendor_chai_jit"
DEFAULT_CHAI_CROP_SIZE = 256
AF3_BACKEND_ID = "af3"
AF3_ORACLE_ID = "haiku_af3"


class RF3RunSpec(RunSpec):
    """plegadx infra/runspec.py:43."""

    backend_id: str = eqx.field(static=True, default=RF3_BACKEND_ID)
    n_recycles: int = eqx.field(static=True, default=1)
    diffusion_steps: int = eqx.field(static=True, default=0)
    n_tokens: int = eqx.field(static=True, default=32)
    checkpoint_variant: str = eqx.field(static=True, default="latest")
    oracle_id: str = eqx.field(static=True, default=RF3_ORACLE_ID)


class ChaiRunSpec(RunSpec):
    """plegadx infra/runspec.py:353."""

    backend_id: str = eqx.field(static=True, default=CHAI_BACKEND_ID)
    n_recycles: int = eqx.field(static=True, default=1)
    num_trunk_recycles: int = eqx.field(static=True, default=1)
    diffusion_steps: int = eqx.field(static=True, default=0)
    n_tokens: int = eqx.field(static=True, default=32)
    crop_size: int = eqx.field(static=True, default=DEFAULT_CHAI_CROP_SIZE)
    use_msa: bool = eqx.field(static=True, default=False)
    use_esm: bool = eqx.field(static=True, default=True)
    use_templates: bool = eqx.field(static=True, default=False)
    # Empty string = restraints off; set to a vendor contact/pocket CSV path for E2E-022.
    constraint_path: str = eqx.field(static=True, default="")
    oracle_id: str = eqx.field(static=True, default=CHAI_ORACLE_ID)


class AF3RunSpec(RunSpec):
    """plegadx infra/runspec.py:839."""

    backend_id: str = eqx.field(static=True, default=AF3_BACKEND_ID)
    n_recycles: int = eqx.field(static=True, default=1)
    diffusion_steps: int = eqx.field(static=True, default=0)
    n_tokens: int = eqx.field(static=True, default=32)
    oracle_id: str = eqx.field(static=True, default=AF3_ORACLE_ID)
    checkpoint_variant: str = eqx.field(static=True, default="af3")


def _token_axis(n_tokens):
    """plegadx infra/runspec.py:34-40 (``rf3_token_axis_spec``) with a literal axis name."""
    return AxisSpec(
        name="token",
        cardinality=n_tokens,
        default_batch_size=8,
        bucket_boundaries=(32, 64, 128, 256, 384, 512),
    )


# Every RunSpec static the base declares, set to a non-default value.
_BASE_STATICS = {
    "run_id": "run-abc123",
    "output_root": Path("/runs/out"),
    "device_count": 4,
    "precision": "bf16",
    "shard_lineage": ("root", "shard-1"),
}


def _make_rf3(**kw):
    return RF3RunSpec(seed=42, axes=[_token_axis(64)], n_tokens=64, n_recycles=3, **kw)


def _make_chai(**kw):
    return ChaiRunSpec(
        seed=7,
        axes=[_token_axis(96)],
        n_tokens=96,
        use_msa=True,
        constraint_path="restraints.csv",
        **kw,
    )


def _make_af3(**kw):
    return AF3RunSpec(seed=5, axes=[_token_axis(48)], n_recycles=2, **kw)


def _make_aminx(**kw):
    return AminxRunSpec(
        seed=3,
        axes=[],
        io=IOConfig(
            output_dir=Path("/o"),
            sink_kind="zarr",
            manifest_path=None,
            output_h5_path=None,
            cache_path=None,
        ),
        resource=ResourceConfig(
            n_devices=1, sample_batch_size=8, structure_batch_size=4, max_buffer_size=None
        ),
        precision_config=PrecisionConfig(compute="fp32"),
        plan=PlannerTopology(use_unified_driver=True),
        sampling=SamplingConfig(
            num_samples=2,
            random_seed=42,
            return_logits=False,
            compute_pseudo_perplexity=False,
            return_decoding_orders=False,
            return_logit_fingerprint=False,
            backbone_noise=(0.0,),
            temperature=(0.1,),
            bias=np.arange(3.0),
        ),
        **kw,
    )


_MAKERS = {"rf3": _make_rf3, "chai": _make_chai, "af3": _make_af3, "aminx": _make_aminx}
_CLASSES = {"rf3": RF3RunSpec, "chai": ChaiRunSpec, "af3": AF3RunSpec, "aminx": AminxRunSpec}
_SHAPES = tuple(_MAKERS)

# Static (treedef) subclass fields a test may vary: (field, alternative value).
_STATIC_VARIANT = {
    "rf3": ("n_recycles", 9),
    "chai": ("crop_size", 128),
    "af3": ("diffusion_steps", 5),
    "aminx": ("encoding_fusion", "fuse-a"),
}


@pytest.fixture(params=_SHAPES)
def shape(request):
    return request.param


def _static_names(cls):
    return [f.name for f in dataclasses.fields(cls) if f.metadata.get("static", False)]


def _data_names(cls):
    return [f.name for f in dataclasses.fields(cls) if not f.metadata.get("static", False)]


# ---------------------------------------------------------------------------
# Field order, static flags and __init__ signatures
# ---------------------------------------------------------------------------

_BASE_FIELDS = (
    ("seed", False),
    ("axes", False),
    ("carry_specs", False),
    ("boundaries", False),
    ("run_id", True),
    ("output_root", True),
    ("device_count", True),
    ("precision", True),
    ("shard_lineage", True),
)

_SUBCLASS_FIELDS = {
    "rf3": (
        ("backend_id", True),
        ("n_recycles", True),
        ("diffusion_steps", True),
        ("n_tokens", True),
        ("checkpoint_variant", True),
        ("oracle_id", True),
    ),
    "chai": (
        ("backend_id", True),
        ("n_recycles", True),
        ("num_trunk_recycles", True),
        ("diffusion_steps", True),
        ("n_tokens", True),
        ("crop_size", True),
        ("use_msa", True),
        ("use_esm", True),
        ("use_templates", True),
        ("constraint_path", True),
        ("oracle_id", True),
    ),
    "af3": (
        ("backend_id", True),
        ("n_recycles", True),
        ("diffusion_steps", True),
        ("n_tokens", True),
        ("oracle_id", True),
        ("checkpoint_variant", True),
    ),
    "aminx": (
        ("io", False),
        ("resource", False),
        ("precision_config", False),
        ("plan", False),
        ("sampling", False),
        ("potts_mpnn", True),
        ("protonpotts", True),
        ("laser", True),
        ("encoding_fusion", True),
        ("decoding_fusion", True),
    ),
}


def test_base_field_order_and_static_flags():
    got = tuple(
        (f.name, bool(f.metadata.get("static", False))) for f in dataclasses.fields(RunSpec)
    )
    assert got == _BASE_FIELDS


def test_subclass_field_order_and_static_flags(shape):
    cls = _CLASSES[shape]
    got = tuple((f.name, bool(f.metadata.get("static", False))) for f in dataclasses.fields(cls))
    assert got == _BASE_FIELDS + _SUBCLASS_FIELDS[shape]


def test_init_signature_is_base_fields_then_subclass_fields(shape):
    cls = _CLASSES[shape]
    params = list(inspect.signature(cls).parameters.values())
    assert [p.name for p in params] == [n for n, _ in _BASE_FIELDS + _SUBCLASS_FIELDS[shape]]
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params)
    # seed and axes are the only parameters without a default.
    assert [p.name for p in params if p.default is inspect.Parameter.empty] == ["seed", "axes"]


def test_positional_construction_matches_field_order():
    spec = RunSpec(3, [])
    assert (spec.seed, spec.axes) == (3, [])


def test_default_factory_gives_each_instance_its_own_list():
    a = RunSpec(seed=0, axes=[])
    b = RunSpec(seed=0, axes=[])
    assert a.carry_specs == [] and a.carry_specs is not b.carry_specs


def test_non_default_field_after_defaults_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="non-default argument 'extra' follows default"):

        class Bad(RunSpec):
            """Subclass adding a required field after the base's defaults."""

            extra: int


def test_kw_only_field_after_defaults_is_accepted():
    class WithKwOnly(RunSpec):
        """Subclass adding a required keyword-only field."""

        extra: int = dataclasses.field(kw_only=True)

    params = inspect.signature(WithKwOnly).parameters
    assert params["extra"].kind is inspect.Parameter.KEYWORD_ONLY
    assert WithKwOnly(seed=1, axes=[], extra=2).extra == 2
    with pytest.raises(TypeError):
        WithKwOnly(seed=1, axes=[])


def test_mutable_class_default_is_rejected():
    with pytest.raises(ValueError, match="mutable default"):

        class Bad(RunSpec):
            """Subclass with a list default instead of default_factory."""

            extra: list = []


def test_static_default_factory_subclass_field():
    class WithFactory(RunSpec):
        """Subclass using eqx.field(static=True, default_factory=...)."""

        tags: tuple = eqx.field(static=True, default_factory=tuple)

    spec = WithFactory(seed=0, axes=[])
    assert spec.tags == ()
    assert "tags" in _static_names(WithFactory)


# ---------------------------------------------------------------------------
# Construction, defaults, inherited validation
# ---------------------------------------------------------------------------


def test_base_defaults():
    spec = RunSpec(seed=0, axes=[])
    assert spec.carry_specs == []
    assert spec.boundaries is None
    assert spec.run_id is None
    assert spec.output_root is None
    assert spec.device_count is None
    assert spec.precision is None
    assert spec.shard_lineage is None


def test_subclass_construction_with_defaults():
    rf3 = RF3RunSpec(seed=1, axes=[])
    assert (rf3.backend_id, rf3.n_recycles, rf3.n_tokens, rf3.oracle_id) == (
        "rf3",
        1,
        32,
        "vendor_rf3",
    )
    chai = ChaiRunSpec(seed=1, axes=[])
    assert (chai.crop_size, chai.use_esm, chai.use_msa, chai.constraint_path) == (
        256,
        True,
        False,
        "",
    )
    af3 = AF3RunSpec(seed=1, axes=[])
    assert (af3.checkpoint_variant, af3.oracle_id) == ("af3", "haiku_af3")
    aminx = AminxRunSpec(seed=1, axes=[])
    assert aminx.io is None and aminx.sampling is None and aminx.precision_config is None
    assert aminx.encoding_fusion is None and aminx.potts_mpnn is None


def test_full_construction_keeps_values(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    assert type(spec) is _CLASSES[shape]
    for name, value in _BASE_STATICS.items():
        assert getattr(spec, name) == value


_BAD_BASE_VALUES = (
    ({"device_count": 0}, ValueError),
    ({"device_count": -2}, ValueError),
    ({"device_count": True}, ValueError),
    ({"device_count": 1.5}, ValueError),
    ({"precision": 16}, TypeError),
    ({"output_root": "relative/str"}, TypeError),
    ({"shard_lineage": ["a", "b"]}, TypeError),
    ({"shard_lineage": ("a", 2)}, TypeError),
)


@pytest.mark.parametrize(("bad", "exc"), _BAD_BASE_VALUES)
def test_inherited_check_init_rejects_bad_base_values(shape, bad, exc):
    with pytest.raises(exc):
        _MAKERS[shape](**bad)


def test_aminx_precision_config_would_have_collided_with_base_precision():
    """The reason aminx renamed its field: a PrecisionConfig in ``precision`` is rejected."""
    with pytest.raises(TypeError, match="precision must be a str or None"):
        AminxRunSpec(seed=0, axes=[], precision=PrecisionConfig(compute="fp32"))


def test_replace_reruns_inherited_check_init(shape):
    spec = _MAKERS[shape]()
    with pytest.raises(ValueError, match="device_count"):
        dataclasses.replace(spec, device_count=0)


def test_subclass_check_init_chains_with_base_check_init():
    class Checked(RF3RunSpec):
        """Subclass adding its own __check_init__."""

        def __check_init__(self):
            if self.n_tokens <= 0:
                raise ValueError("n_tokens must be positive")

    Checked(seed=0, axes=[], n_tokens=4)
    with pytest.raises(ValueError, match="n_tokens"):
        Checked(seed=0, axes=[], n_tokens=0)
    with pytest.raises(ValueError, match="device_count"):
        Checked(seed=0, axes=[], n_tokens=4, device_count=0)


def test_two_level_subclass_keeps_static_fields_and_checks():
    class Deeper(AF3RunSpec):
        """Subclass of a subclass, still undecorated."""

        extra: int = eqx.field(static=True, default=11)

    spec = Deeper(seed=0, axes=[], extra=12)
    assert spec.extra == 12 and spec.checkpoint_variant == "af3"
    assert "extra" in _static_names(Deeper)
    with pytest.raises(TypeError):
        Deeper(seed=0, axes=[], precision=3)


def test_from_spec_is_identity_on_subclasses(shape):
    spec = _MAKERS[shape]()
    assert _CLASSES[shape].from_spec(spec) is spec


def test_instances_are_frozen(shape):
    spec = _MAKERS[shape]()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.seed = 99


def test_specs_are_not_hashable_because_axes_is_a_list(shape):
    with pytest.raises(TypeError, match="unhashable"):
        hash(_MAKERS[shape]())


def test_equality_separates_statics_and_ignores_identity(shape):
    name, alt = _STATIC_VARIANT[shape]
    a = _MAKERS[shape]()
    b = _MAKERS[shape]()
    assert a is not b and a == b
    assert a != dataclasses.replace(b, **{name: alt})
    assert a != dataclasses.replace(b, seed=a.seed + 1)


# ---------------------------------------------------------------------------
# Pytree behaviour
# ---------------------------------------------------------------------------


def test_flatten_unflatten_round_trip(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    leaves, treedef = jax.tree_util.tree_flatten(spec)
    rebuilt = treedef.unflatten(leaves)
    assert type(rebuilt) is type(spec)
    assert rebuilt == spec
    for name in _static_names(type(spec)):
        assert getattr(rebuilt, name) == getattr(spec, name)
    assert jax.tree_util.tree_structure(rebuilt) == treedef


def test_statics_are_never_leaves(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    leaves = jax.tree_util.tree_leaves(spec)
    for value in (*_BASE_STATICS.values(), "run-abc123"):
        assert not any(leaf is value for leaf in leaves)
    assert not any(isinstance(leaf, str) for leaf in leaves)


_EXPECTED_LEAF_PATHS = {
    "rf3": [".seed", ".axes[0]"],
    "chai": [".seed", ".axes[0]"],
    "af3": [".seed", ".axes[0]"],
    "aminx": [".seed", ".sampling.bias"],
}


def test_leaf_key_paths(shape):
    spec = _MAKERS[shape]()
    paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(spec)[0]]
    assert paths == _EXPECTED_LEAF_PATHS[shape]


def test_leaves_are_seed_axes_and_array_not_statics():
    leaves = jax.tree_util.tree_leaves(_make_rf3(**_BASE_STATICS))
    assert leaves[0] == 42 and leaves[1] == _token_axis(64) and len(leaves) == 2
    amin = jax.tree_util.tree_leaves(_make_aminx())
    assert amin[0] == 3 and np.array_equal(amin[1], np.arange(3.0)) and len(amin) == 2


def test_treedef_tracks_each_static_field(shape):
    base = _MAKERS[shape](**_BASE_STATICS)
    base_def = jax.tree_util.tree_structure(base)
    assert jax.tree_util.tree_structure(_MAKERS[shape](**_BASE_STATICS)) == base_def
    variants = {
        "run_id": "run-other",
        "output_root": Path("/runs/other"),
        "device_count": 2,
        "precision": "fp32",
        "shard_lineage": ("root",),
        _STATIC_VARIANT[shape][0]: _STATIC_VARIANT[shape][1],
    }
    for name, value in variants.items():
        changed = dataclasses.replace(base, **{name: value})
        assert jax.tree_util.tree_structure(changed) != base_def, name
        assert len(jax.tree_util.tree_leaves(changed)) == len(jax.tree_util.tree_leaves(base))


def test_seed_change_does_not_change_treedef(shape):
    a = _MAKERS[shape]()
    b = dataclasses.replace(a, seed=a.seed + 100)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def test_tree_map_preserves_type_and_statics(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    mapped = jax.tree_util.tree_map(lambda x: x, spec)
    assert type(mapped) is type(spec) and mapped == spec


def test_replace_changes_one_field_and_keeps_the_rest(shape):
    name, alt = _STATIC_VARIANT[shape]
    spec = _MAKERS[shape](**_BASE_STATICS)
    out = dataclasses.replace(spec, **{name: alt}, seed=1000)
    assert type(out) is type(spec)
    assert getattr(out, name) == alt and out.seed == 1000
    for f in dataclasses.fields(spec):
        if f.name not in (name, "seed"):
            before, after = getattr(spec, f.name), getattr(out, f.name)
            assert before is after or before == after or f.name == "sampling"


def test_aminx_submodules_stay_pytree_children():
    spec = _make_aminx()
    assert isinstance(spec.io, eqx.Module) and isinstance(spec.sampling, eqx.Module)
    mapped = jax.tree_util.tree_map(lambda x: x * 2, spec)
    assert type(mapped) is AminxRunSpec
    assert type(mapped.sampling) is SamplingConfig and type(mapped.io) is IOConfig
    assert mapped.seed == 6
    assert np.array_equal(mapped.sampling.bias, np.arange(3.0) * 2)
    # Static fields of a sub-module live in the treedef: no leaf, but a retrace trigger.
    other = dataclasses.replace(
        spec,
        resource=ResourceConfig(
            n_devices=2, sample_batch_size=8, structure_batch_size=4, max_buffer_size=None
        ),
    )
    assert jax.tree_util.tree_leaves(other)[0] == 3
    assert jax.tree_util.tree_structure(other) != jax.tree_util.tree_structure(spec)


def test_aminx_unset_submodules_contribute_no_leaves():
    assert jax.tree_util.tree_leaves(AminxRunSpec(seed=1, axes=[])) == [1]


def test_aminx_equality_with_array_bearing_submodules():
    assert _make_aminx() == _make_aminx()
    other = _make_aminx()
    other = dataclasses.replace(
        other,
        sampling=dataclasses.replace(other.sampling, bias=np.arange(3.0) + 1),
    )
    assert other != _make_aminx()


def test_top_level_array_field_equality_is_a_bool():
    """Struct's tree-aware == (eq=False + tree_equal) gives a bool, as eqx.Module did."""

    class WithArray(RunSpec):
        """Subclass holding an array directly (neither aminx nor plegadx does)."""

        weights: Any = None

    a = WithArray(seed=0, axes=[], weights=np.arange(3.0))
    b = WithArray(seed=0, axes=[], weights=np.arange(3.0))
    assert (a == b) is True
    assert (a == WithArray(seed=0, axes=[], weights=np.arange(1.0, 4.0))) is False


def test_eqx_tree_at_replaces_leaves(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    out = eqx.tree_at(lambda s: s.seed, spec, 99)
    assert type(out) is type(spec) and out.seed == 99
    for name in _static_names(type(spec)):
        assert getattr(out, name) == getattr(spec, name)
    assert jax.tree_util.tree_structure(out) == jax.tree_util.tree_structure(spec)


def test_eqx_tree_at_reaches_into_aminx_submodule():
    spec = _make_aminx()
    out = eqx.tree_at(lambda s: s.sampling.bias, spec, np.zeros(3))
    assert type(out.sampling) is SamplingConfig
    assert np.array_equal(out.sampling.bias, np.zeros(3))
    assert np.array_equal(spec.sampling.bias, np.arange(3.0))


def test_eqx_partition_and_combine_round_trip():
    spec = _make_aminx()
    arrays, rest = eqx.partition(spec, eqx.is_array)
    assert np.array_equal(arrays.sampling.bias, np.arange(3.0))
    again = eqx.combine(arrays, rest)
    assert type(again) is AminxRunSpec and again == spec


def test_pickle_and_deepcopy_round_trip(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert copy.deepcopy(spec) == spec


# ---------------------------------------------------------------------------
# jit
# ---------------------------------------------------------------------------


def _traced_total(traces):
    def total(spec):
        traces.append(1)
        out = spec.seed * 2 + spec.n_tokens if hasattr(spec, "n_tokens") else spec.seed * 2
        if isinstance(spec, AminxRunSpec):
            out = out + spec.sampling.bias.sum()
        return out

    return total


@pytest.mark.parametrize("shape", ["rf3", "chai", "af3", "aminx"])
def test_jax_jit_over_spec_traces_seed_and_retraces_on_static_change(shape):
    # jax.jit needs every leaf to be an array/number: axes must hold no AxisSpec here.
    make = {
        "rf3": lambda **kw: RF3RunSpec(seed=2, axes=[], **kw),
        "chai": lambda **kw: ChaiRunSpec(seed=2, axes=[], **kw),
        "af3": lambda **kw: AF3RunSpec(seed=2, axes=[], **kw),
        "aminx": lambda **kw: _make_aminx(**kw),
    }[shape]
    traces = []
    jitted = jax.jit(_traced_total(traces))
    first = jitted(make())
    jitted(dataclasses.replace(make(), seed=50))
    assert len(traces) == 1  # a new seed value is a new array value, not a new trace
    name, alt = _STATIC_VARIANT[shape]
    jitted(dataclasses.replace(make(), **{name: alt}))
    assert len(traces) == 2  # a static field is treedef metadata: it retraces
    base = 2 * 2 + (32 if hasattr(make(), "n_tokens") else 0)
    expected = _make_aminx().seed * 2 + 3.0 if shape == "aminx" else base
    assert float(first) == expected


def test_jax_jit_can_return_a_spec_with_traced_seed():
    spec = RF3RunSpec(seed=2, axes=[], n_recycles=4)
    out = jax.jit(lambda s: dataclasses.replace(s, seed=s.seed + 1))(spec)
    assert type(out) is RF3RunSpec
    assert int(out.seed) == 3 and out.n_recycles == 4


@pytest.mark.parametrize("shape", ["rf3", "chai", "af3"])
def test_eqx_filter_jit_with_axis_spec_leaves(shape):
    traces = []
    spec = _MAKERS[shape]()

    @eqx.filter_jit
    def total(s):
        traces.append(1)
        return s.seed * 2 + s.axes[0].cardinality + s.n_tokens

    expected = spec.seed * 2 + spec.axes[0].cardinality + spec.n_tokens
    assert int(total(spec)) == expected
    total(spec)
    assert len(traces) == 1
    name, alt = _STATIC_VARIANT[shape]
    total(dataclasses.replace(spec, **{name: alt}))
    assert len(traces) == 2


def test_eqx_filter_jit_over_aminx_spec():
    spec = _make_aminx()

    @eqx.filter_jit
    def total(s):
        return s.seed + s.sampling.bias.sum() + s.resource.n_devices

    assert float(total(spec)) == 3 + 3.0 + 1


# ---------------------------------------------------------------------------
# Serialisation and digest surface
#
# No xtrax code path digests a RunSpec directly. The surfaces that read one are:
# ``derive_sink_spec`` (xtrax.run.sink), ``dataclasses.fields`` / ``asdict`` (the shape
# aminx ``topology_hash`` uses, run/spec.py:~330, on its sub-modules), and
# ``canonical_digest`` over a field payload. The literals below were computed on the
# eqx.Module base; they must not move when the base changes.
# ---------------------------------------------------------------------------


def _static_payload(spec):
    return {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.metadata.get("static", False)
    }


def _sha(obj):
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


# (static payload json sha256, asdict json sha256, canonical_digest of str-ified statics)
_EXPECTED_DIGESTS = {
    "rf3": (
        "184be227e74dc592ad5ec987c183c061edd220eafd05c5ebe86713e1c3eaab91",
        "97c5d003f6c1533da0c7d0105306434c25af9d8811c6b5e6a817a4dacea5c9b6",
        "fb49d6be79f7fb9ee22a4c79e4a43a418ed05e61734d09483bc21fc6e792e768",
    ),
    "chai": (
        "1e4c0835663b6c9801ad4879f04e91f12ef51c2fe3ecaa3f64965c0a07b41fa7",
        "725cb3966f26d9fd757daf8965f9d3d6b3e24157c4025c88cb97063a454143b9",
        "2177c3fca683611c7e0e60f03a67350abaabc831a3ffec1da7a8f1397041a046",
    ),
    "af3": (
        "66c721f8ac30e89c717b518e3647762b1842e5fb0c2a7c0625f565b9c6ed3324",
        "2ced642ea37ca31ecccd830a093fbd997cf464df5d11402130222f2871eb627a",
        "11e03ecaf994d4802825c336f15d95b45a48016990f0430a433b388d69bdff7a",
    ),
    "aminx": (
        "053b41d22049e6bd82e8d8a300a50ea53fac7b86a99d2fc9a41722fa537e88bc",
        "5c00ebae5f4fe755be209e74d9048ed5f873539daaa9b6a50b0bae8e0ce9ac1c",
        "e0b952014edc2dbc083fbf91a5d5c14cba9487ad6e4f8ae30a85c7c325cd2099",
    ),
}


def _digests(spec):
    payload = _static_payload(spec)
    stringified = {k: json.loads(json.dumps(v, default=str)) for k, v in payload.items()}
    return (
        _sha(payload),
        _sha(dataclasses.asdict(spec)),
        canonical_digest(stringified),
    )


def test_field_payload_digests_are_stable(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    assert _digests(spec) == _EXPECTED_DIGESTS[shape]


def test_digest_check_can_fail():
    """Negative control: a changed static field moves every pinned digest."""
    spec = _make_rf3(**_BASE_STATICS)
    moved = dataclasses.replace(spec, n_tokens=65)
    assert _digests(moved)[0] != _digests(spec)[0]
    assert _digests(moved)[1] != _digests(spec)[1]
    assert _digests(moved)[2] != _digests(spec)[2]


def test_asdict_and_fields_order_match_declaration(shape):
    spec = _MAKERS[shape](**_BASE_STATICS)
    names = [n for n, _ in _BASE_FIELDS + _SUBCLASS_FIELDS[shape]]
    assert list(dataclasses.asdict(spec)) == names
    assert [f.name for f in dataclasses.fields(spec)] == names


def test_derive_sink_spec_reads_run_id_seed_and_output_root(shape, tmp_path):
    spec = _MAKERS[shape](run_id="run-xyz", output_root=tmp_path / "root")
    sink = derive_sink_spec(spec, output_dir=None)
    assert sink.run_id == "run-xyz"
    assert sink.seed == spec.seed
    assert sink.output_dir == tmp_path / "root"
    assert sink.format == "zarr"
    explicit = derive_sink_spec(spec, output_dir=tmp_path / "x", run_id="override")
    assert (explicit.run_id, explicit.output_dir) == ("override", tmp_path / "x")
