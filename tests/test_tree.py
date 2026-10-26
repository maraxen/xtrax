"""xtrax._tree: plain-JAX Struct base, partition/combine, model_jit, inference_mode.

Spec: .praxia/docs/specs/261023_equinox-decoupling.md sections 3, 6, 7. Equinox appears here
only as a fixture (a model type xtrax must accept), never as an xtrax dependency. Each check
has a negative control: a broken input the check must reject.
"""

import dataclasses
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import xtrax
from xtrax._tree import (
    Struct,
    _Static,
    combine,
    field,
    inference_mode,
    is_array,
    model_jit,
    partition,
)

# --- fixtures -------------------------------------------------------------------------


class Cfg(Struct):
    w: jax.Array
    scale: float = field(static=True, default=1.0)


class LeakyCfg(Struct):
    """Negative control: `scale` left non-static by mistake."""

    w: jax.Array
    scale: float = 1.0


class Base(Struct):
    n: int = field(static=True, default=1)

    def __check_init__(self):
        if self.n < 0:
            raise ValueError("n must be non-negative")


class Derived(Base):  # undecorated subclass, plegadx style
    w: jax.Array = None


class EqxSpelled(Base):
    tag: str = eqx.field(static=True, default="a")
    w: jax.Array = None


class XtraxSpelled(Base):
    tag: str = field(static=True, default="a")
    w: jax.Array = None


class Outer(Struct):
    inner: Cfg
    other: jax.Array
    name: str = field(static=True, default="o")


def _arr(*vals):
    return jnp.asarray(vals, dtype=jnp.float32)


def _assert_static_not_leaf(obj, name):
    value = getattr(obj, name)
    leaves = jax.tree_util.tree_leaves(obj)
    assert not any(leaf is value for leaf in leaves), f"static field {name!r} is a leaf"


def _assert_rejects(factory, exc=ValueError):
    try:
        factory()
    except exc:
        return
    raise AssertionError("constructor accepted invalid input")


# --- Struct semantics -----------------------------------------------------------------


def test_static_fields_are_not_leaves():
    c = Cfg(w=_arr(1, 2), scale=3.0)
    leaves, treedef = jax.tree_util.tree_flatten(c)
    assert len(leaves) == 1 and leaves[0] is c.w
    _assert_static_not_leaf(c, "scale")
    # treedef carries the static value; array values do not enter it
    assert treedef == jax.tree_util.tree_structure(Cfg(w=_arr(9, 9), scale=3.0))
    assert treedef != jax.tree_util.tree_structure(Cfg(w=_arr(1, 2), scale=4.0))


def test_static_not_leaf_control_fails_when_static_is_missing():
    with pytest.raises(AssertionError, match="is a leaf"):
        _assert_static_not_leaf(LeakyCfg(w=_arr(1), scale=3.0), "scale")
    assert len(jax.tree_util.tree_leaves(LeakyCfg(w=_arr(1)))) == 2


def test_inherited_check_init_raises():
    _assert_rejects(lambda: Base(n=-1))
    _assert_rejects(lambda: Derived(n=-1, w=_arr(1)))  # inherited, subclass is undecorated
    assert Derived(n=2, w=_arr(1)).n == 2


def test_check_init_rejection_control_detects_a_missing_hook():
    class Loose(Struct):  # misspelled hook: never runs
        n: int = field(static=True, default=1)

        def _check_init(self):
            raise ValueError

    with pytest.raises(AssertionError, match="accepted invalid input"):
        _assert_rejects(lambda: Loose(n=-1))


def test_check_init_runs_along_mro_base_first():
    order = []

    class A(Struct):
        def __check_init__(self):
            order.append("A")

    class B(A):
        def __check_init__(self):
            order.append("B")

    class C(B):  # no hook of its own
        pass

    C()
    assert order == ["A", "B"]


def test_check_init_is_not_run_on_unflatten():
    class Positive(Struct):
        w: jax.Array

        def __check_init__(self):
            if float(self.w[0]) < 0:  # concretises: fails on a tracer
                raise ValueError("negative")

    p = Positive(w=_arr(1.0))
    out = jax.jit(lambda s: s)(p)  # unflatten with tracers must not call the check
    assert isinstance(out, Positive)
    # control: the check does choke on a tracer, so the test above is meaningful
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(lambda w: Positive(w=w).w)(_arr(1.0))


def test_frozen():
    c = Cfg(w=_arr(1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.scale = 2.0

    @dataclasses.dataclass  # control: a mutable dataclass would not raise
    class Mutable:
        scale: float = 1.0

    m = Mutable()
    m.scale = 2.0
    assert m.scale == 2.0


def test_dataclasses_replace_keeps_static_and_reruns_checks():
    d = Derived(n=3, w=_arr(1, 2))
    r = dataclasses.replace(d, w=_arr(5, 6))
    assert r.n == 3 and r.w is not d.w and d.w[0] == 1 and r.w[0] == 5
    _assert_rejects(lambda: dataclasses.replace(d, n=-4))
    # control: replace on a non-dataclass is an error, not a silent copy
    with pytest.raises(TypeError):
        dataclasses.replace(object())


@pytest.mark.parametrize("cls", [EqxSpelled, XtraxSpelled])
def test_undecorated_subclass_with_either_static_spelling(cls):
    obj = cls(n=1, tag="a", w=_arr(1, 2))
    leaves = jax.tree_util.tree_leaves(obj)
    assert len(leaves) == 1 and leaves[0] is obj.w
    _assert_static_not_leaf(obj, "tag")
    assert jax.tree_util.tree_structure(obj) != jax.tree_util.tree_structure(
        cls(n=1, tag="b", w=_arr(1, 2))
    )
    _assert_rejects(lambda: cls(n=-1, tag="a"))  # inherited check still runs


def test_eqx_field_without_static_is_a_leaf_control():
    class Plain(Base):
        t: float = eqx.field(default=0.5)

    assert len(jax.tree_util.tree_leaves(Plain(n=1))) == 1  # t is data
    with pytest.raises(AssertionError, match="is a leaf"):
        _assert_static_not_leaf(Plain(n=1), "t")


def test_converter_is_refused_loudly():
    with pytest.raises(TypeError, match="converter"):

        class Conv(Struct):
            x: int = eqx.field(static=True, converter=int, default=0)

    # control: the same field without a converter is accepted
    class NoConv(Struct):
        x: int = eqx.field(static=True, default=0)

    assert NoConv().x == 0


def test_field_keeps_user_metadata_and_default_factory():
    class M(Struct):
        extras: dict = field(default_factory=dict, metadata={"doc": "x"})

    f = dataclasses.fields(M)[0]
    assert f.metadata["doc"] == "x" and f.metadata["static"] is False
    assert M().extras == {}


def test_nested_struct():
    o = Outer(inner=Cfg(w=_arr(1, 2), scale=2.0), other=_arr(3))
    leaves = jax.tree_util.tree_leaves(o)
    assert len(leaves) == 2
    doubled = jax.tree_util.tree_map(lambda x: x * 2, o)
    assert isinstance(doubled, Outer) and isinstance(doubled.inner, Cfg)
    assert doubled.inner.scale == 2.0 and float(doubled.other[0]) == 6.0
    assert jax.tree_util.tree_structure(o) != jax.tree_util.tree_structure(
        Outer(inner=Cfg(w=_arr(1, 2), scale=5.0), other=_arr(3))
    )


# --- jit / vmap / grad / tree_at over a Struct ----------------------------------------


def _traces(make_obj_a, make_obj_b):
    """Count traces of a jitted fn called on a, a-again, then b."""
    count = []

    @jax.jit
    def f(c):
        count.append(1)
        return c.w * 2

    f(make_obj_a())
    f(make_obj_a())
    after_same = len(count)
    f(make_obj_b())
    return after_same, len(count)


def test_jit_array_change_does_not_retrace_static_change_does():
    same, array_change = _traces(
        lambda: Cfg(w=_arr(1, 2), scale=1.0), lambda: Cfg(w=_arr(7, 8), scale=1.0)
    )
    assert same == 1 and array_change == 1
    _, static_change = _traces(
        lambda: Cfg(w=_arr(1, 2), scale=1.0), lambda: Cfg(w=_arr(1, 2), scale=2.0)
    )
    assert static_change == 2
    # control: with `scale` mistakenly traced, a value change no longer retraces
    _, leaky = _traces(
        lambda: LeakyCfg(w=_arr(1, 2), scale=1.0), lambda: LeakyCfg(w=_arr(1, 2), scale=2.0)
    )
    assert leaky == 1


def test_vmap_and_grad_over_struct():
    stacked = Cfg(w=jnp.arange(6.0).reshape(3, 2), scale=2.0)
    out = jax.vmap(lambda c: c.w.sum() * c.scale)(stacked)
    np.testing.assert_allclose(out, [2.0, 10.0, 18.0])

    c = Cfg(w=_arr(1, 2, 3), scale=2.0)
    g = jax.grad(lambda c: (c.w**2).sum() * c.scale)(c)
    assert isinstance(g, Cfg) and g.scale == 2.0
    np.testing.assert_allclose(g.w, [4.0, 8.0, 12.0])
    # control: a wrong expectation is rejected by the comparison
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(g.w, [2.0, 4.0, 6.0])


def test_eqx_tree_at_on_struct():
    c = Cfg(w=_arr(1, 2), scale=3.0)
    new = eqx.tree_at(lambda s: s.w, c, _arr(8, 9))
    assert isinstance(new, Cfg) and new.scale == 3.0 and float(new.w[0]) == 8.0
    assert float(c.w[0]) == 1.0  # original untouched
    # control: a static field is not a leaf, so tree_at refuses to replace it
    with pytest.raises(Exception):  # noqa: B017 - eqx raises its own ValueError subtype
        eqx.tree_at(lambda s: s.scale, c, 4.0)


# --- Struct construction probes (spec section 7, "unverified") ------------------------


def test_probe_class_defined_init_skips_post_init():
    ran = []

    class Custom(Struct):
        a: jax.Array
        b: float = field(static=True, default=0.0)

        def __init__(self, a, b=0.0):
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)

        def __check_init__(self):
            ran.append("check")

    Custom(_arr(1))
    assert ran == []  # dataclass kept our __init__; __post_init__ was not called

    class CustomOk(Custom):
        def __init__(self, a, b=0.0):
            object.__setattr__(self, "a", a)
            object.__setattr__(self, "b", b)
            self.__post_init__()  # the spec's ComposedLoss pattern

    CustomOk(_arr(1))
    assert ran == ["check"]
    # unflatten bypasses __init__ entirely and still round-trips the custom-init class
    c = Custom(_arr(1, 2), b=5.0)
    c2 = jax.tree_util.tree_map(lambda x: x + 1, c)
    assert isinstance(c2, Custom) and c2.b == 5.0 and float(c2.a[0]) == 2.0
    assert ran == ["check"]  # nothing ran on unflatten


def test_probe_subclass_defining_post_init_must_call_super():
    class Bad(Base):
        def __post_init__(self):  # forgets super().__post_init__()
            pass

    Bad(n=-1)  # the inherited check silently does not run

    class Good(Base):
        def __post_init__(self):
            super().__post_init__()

    _assert_rejects(lambda: Good(n=-1))


# --- partition / combine / is_array ---------------------------------------------------


def test_is_array():
    class Wrapped:
        def __jax_array__(self):
            return jnp.zeros(1)

    assert is_array(jnp.zeros(2)) and is_array(np.zeros(2)) and is_array(Wrapped())
    assert not any(is_array(x) for x in (1.0, 2, "s", None, jnp.tanh, (1, 2)))


def test_partition_combine_round_trip():
    tree = {"w": _arr(1, 2), "act": jnp.tanh, "n": 3, "nest": [np.ones(2), "tag", None]}
    arrays, rest = partition(tree)
    assert arrays["act"] is None and arrays["n"] is None and arrays["nest"][1] is None
    assert rest["w"] is None and rest["act"] is jnp.tanh and rest["n"] == 3
    back = combine(arrays, rest)
    assert back["w"] is tree["w"] and back["act"] is jnp.tanh and back["n"] == 3
    assert back["nest"][1] == "tag" and back["nest"][2] is None
    # control: a mismatched rest does not reproduce the tree
    _, other_rest = partition({**tree, "n": 4})
    assert combine(arrays, other_rest)["n"] == 4 != tree["n"]


def test_partition_on_struct_keeps_static_and_type():
    c = Cfg(w=_arr(1, 2), scale=2.0)
    arrays, rest = partition(c)
    assert isinstance(arrays, Cfg) and arrays.scale == 2.0 and rest.w is None
    assert combine(arrays, rest).w is c.w


# --- model_jit ------------------------------------------------------------------------


def _mlp(key=0, width=8, activation=jax.nn.relu):
    return eqx.nn.MLP(3, 2, width, 2, activation=activation, key=jax.random.key(key))


def test_model_jit_eqx_mlp_traces_once_per_static_signature():
    traces = []

    @model_jit
    def forward(model, x):
        traces.append(1)
        return jax.vmap(model)(x)

    x = jnp.ones((4, 3))
    ref = jax.vmap(_mlp(0))(x)
    np.testing.assert_allclose(forward(_mlp(0), x), ref, rtol=1e-6)
    forward(_mlp(0), x)
    forward(_mlp(1), x)  # same structure, different array values
    assert len(traces) == 1
    out = forward(_mlp(1, activation=jnp.tanh), x)  # non-array leaf changed
    assert len(traces) == 2
    np.testing.assert_allclose(out, jax.vmap(_mlp(1, activation=jnp.tanh))(x), rtol=1e-6)
    forward(_mlp(1, width=16), x)  # array shape changed
    assert len(traces) == 3


def test_model_jit_dict_model_static_leaf_change_retraces():
    traces = []

    @model_jit
    def apply(model, x):
        traces.append(1)
        return model["w"] @ x * model["scale"]

    x = _arr(1, 2)
    m = {"w": jnp.eye(2), "scale": 2.0}
    np.testing.assert_allclose(apply(m, x), [2.0, 4.0])
    apply({"w": 3 * jnp.eye(2), "scale": 2.0}, x)
    assert len(traces) == 1  # array value change: no retrace
    np.testing.assert_allclose(apply({"w": jnp.eye(2), "scale": 5.0}, x), [5.0, 10.0])
    assert len(traces) == 2  # python-float change is static: retraced and correct
    apply({"w": jnp.eye(2), "scale": 5.0, "extra": None}, x)
    assert len(traces) == 3  # tree structure change


def test_model_jit_returns_model_with_static_parts_intact():
    @model_jit
    def sgd(model, x):
        arrays, rest = partition(model)
        g = jax.grad(lambda a: jax.vmap(combine(a, rest))(x).sum())(arrays)
        return combine(jax.tree_util.tree_map(lambda p, d: p - 0.1 * d, arrays, g), rest)

    model = _mlp(0)
    new = sgd(model, jnp.ones((4, 3)))
    assert isinstance(new, eqx.nn.MLP) and new.activation is model.activation
    before = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    after = jax.tree_util.tree_leaves(eqx.filter(new, eqx.is_array))
    assert any(not np.allclose(a, b) for a, b in zip(before, after, strict=True))


def test_model_jit_output_static_is_keyed_by_input_static():
    @model_jit
    def tag(model, x):
        return {"y": x * 2, "kind": model["kind"]}

    assert tag({"kind": "a"}, _arr(1))["kind"] == "a"
    assert tag({"kind": "b"}, _arr(1))["kind"] == "b"
    assert tag({"kind": "a"}, _arr(1))["kind"] == "a"


def test_model_jit_unhashable_leaf_error_names_the_path():
    @model_jit
    def f(model):
        return model["w"]

    with pytest.raises(TypeError, match=r"\['bad'\].*unhashable"):
        f({"w": _arr(1), "bad": {1, 2}})
    # control: the same model without the unhashable leaf runs
    np.testing.assert_allclose(f({"w": _arr(1), "bad": (1, 2)}), [1.0])


def test_static_wrapper_hash_and_eq():
    a = _Static({"f": jnp.tanh, "n": 1, "w": None})
    b = _Static({"f": jnp.tanh, "n": 1, "w": None})
    c = _Static({"f": jnp.tanh, "n": 2, "w": None})
    assert a == b and hash(a) == hash(b) and a != c
    assert _Static({"n": 1}) != _Static([1])  # different treedef


def _plain_loss(w, b, x, y, scale):
    return jnp.mean((scale * (x @ w + b) - y) ** 2)


def _grad_fixture():
    key = jax.random.key(0)
    kw, kx, ky = jax.random.split(key, 3)
    w = jax.random.normal(kw, (3, 2))
    b = jnp.zeros(2)
    x = jax.random.normal(kx, (5, 3))
    y = jax.random.normal(ky, (5, 2))
    return {"w": w, "b": b, "scale": 1.5}, x, y


def test_partition_combine_grad_matches_plain_jax_grad():
    model, x, y = _grad_fixture()

    def loss(m, x, y):
        return _plain_loss(m["w"], m["b"], x, y, m["scale"])

    arrays, rest = partition(model)
    g = jax.grad(lambda a: loss(combine(a, rest), x, y))(arrays)
    ref_w, ref_b = jax.grad(_plain_loss, argnums=(0, 1))(model["w"], model["b"], x, y, 1.5)
    np.testing.assert_allclose(g["w"], ref_w, rtol=1e-6)
    np.testing.assert_allclose(g["b"], ref_b, rtol=1e-6)
    assert g["scale"] is None
    # control: a different static scale gives a different gradient, so equality is not vacuous
    other_w = jax.grad(_plain_loss)(model["w"], model["b"], x, y, 3.0)
    assert not np.allclose(g["w"], other_w)


def test_model_jit_value_and_grad_matches_plain_and_loss_decreases():
    model, x, y = _grad_fixture()
    opt = optax.sgd(0.1)

    @model_jit
    def step(model, opt_state, x, y):
        arrays, rest = partition(model)

        def loss(a):
            m = combine(a, rest)
            return _plain_loss(m["w"], m["b"], x, y, m["scale"])

        value, grads = jax.value_and_grad(loss)(arrays)
        updates, opt_state = opt.update(grads, opt_state, arrays)
        arrays = optax.apply_updates(arrays, updates)
        return combine(arrays, rest), opt_state, value, grads

    opt_state = opt.init(partition(model)[0])
    ref_val, (ref_w, ref_b) = jax.value_and_grad(_plain_loss, argnums=(0, 1))(
        model["w"], model["b"], x, y, 1.5
    )
    _, _, value, grads = step(model, opt_state, x, y)
    np.testing.assert_allclose(value, ref_val, rtol=1e-6)
    np.testing.assert_allclose(grads["w"], ref_w, rtol=1e-6)
    np.testing.assert_allclose(grads["b"], ref_b, rtol=1e-6)

    first = float(value)
    for _ in range(30):
        model, opt_state, value, _ = step(model, opt_state, x, y)
    assert model["scale"] == 1.5 and float(value) < first


def test_model_jit_grad_matches_eqx_filter_grad_on_mlp():
    model = _mlp(0)
    x = jnp.ones((4, 3))

    def loss(m):
        return jax.vmap(m)(x).sum()

    ref = eqx.filter_grad(loss)(model)

    @model_jit
    def grads(m):
        arrays, rest = partition(m)
        return jax.grad(lambda a: loss(combine(a, rest)))(arrays)

    got = grads(model)
    for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(ref), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    assert len(jax.tree_util.tree_leaves(got)) > 0


# --- inference_mode -------------------------------------------------------------------


def test_inference_mode_hook_wins():
    class Hooked(Struct):
        w: jax.Array
        inference: bool = field(static=True, default=False)

        def __xtrax_inference__(self):
            return dataclasses.replace(self, inference=True)

    h = Hooked(w=_arr(1))
    assert inference_mode(h).inference is True and h.inference is False


def test_inference_mode_equinox_module_when_loaded():
    model = eqx.nn.Sequential([eqx.nn.Dropout(0.5), eqx.nn.Linear(2, 2, key=jax.random.key(0))])
    assert model.layers[0].inference is False
    assert inference_mode(model).layers[0].inference is True


def test_inference_mode_identity_for_plain_pytrees():
    d = {"w": _arr(1)}
    assert inference_mode(d) is d
    c = Cfg(w=_arr(1))
    assert inference_mode(c) is c


def test_inference_mode_does_not_import_equinox(monkeypatch):
    model = eqx.nn.Dropout(0.5)
    saved = {k: v for k, v in sys.modules.items() if k == "equinox" or k.startswith("equinox.")}
    for k in saved:
        monkeypatch.delitem(sys.modules, k)
    try:
        assert inference_mode(model) is model  # equinox not loaded: identity
        assert "equinox" not in sys.modules  # and nothing imported it
    finally:
        sys.modules.update(saved)
    # control: with equinox loaded the same call does switch the flag
    assert inference_mode(model).inference is True


# --- no-equinox import gate -----------------------------------------------------------

_BLOCKER = """
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "equinox" or name.startswith("equinox."):
            raise ImportError("equinox import blocked")
        return None

sys.meta_path.insert(0, _Block())
"""

_EXERCISE = """
import jax, jax.numpy as jnp
import xtrax
from xtrax._tree import Struct, field, model_jit, partition, combine, inference_mode

class S(Struct):
    w: jax.Array
    k: int = field(static=True, default=1)

@model_jit
def f(m, x):
    return m["w"] * x

assert float(f({"w": jnp.ones(2), "c": 2.0}, jnp.ones(2)).sum()) == 2.0
assert len(jax.tree_util.tree_leaves(S(w=jnp.ones(2)))) == 1
assert inference_mode({"a": 1}) == {"a": 1}
assert xtrax.Struct is Struct and xtrax.field is field
assert "equinox" not in sys.modules
print("ok")
"""


def _run_py(code):
    return subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )


def test_tree_module_never_imports_equinox():
    res = _run_py("import sys; import xtrax._tree; assert 'equinox' not in sys.modules")
    assert res.returncode == 0, res.stderr


def test_tree_works_with_equinox_import_blocked():
    res = _run_py("import sys\n" + _BLOCKER + _EXERCISE)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


def test_no_equinox_gate_control_fails_when_equinox_is_imported():
    # prove red: an injected `import equinox` must make both gates fail
    leaked = _run_py(
        "import sys; import equinox; import xtrax._tree; assert 'equinox' not in sys.modules"
    )
    assert leaked.returncode != 0
    blocked = _run_py("import sys\n" + _BLOCKER + "import equinox\n")
    assert blocked.returncode != 0 and "equinox import blocked" in blocked.stderr


# --- public exports -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["Struct", "field", "partition", "combine", "is_array", "model_jit", "inference_mode"]
)
def test_root_exports(name):
    import xtrax._tree as tree

    assert name in xtrax.__all__ and getattr(xtrax, name) is getattr(tree, name)


# --- checkpoint compatibility probe (spec section 7: orbax restore, was unverified) ----


def test_eqx_resumable_state_checkpoint_restores_into_struct(tmp_path):
    from xtrax.checkpoint import get_checkpoint_manager, load_checkpoint, save_checkpoint
    from xtrax.training.types import ResumableState

    class StructState(Struct):  # same field layout as today's eqx ResumableState
        step: jax.Array
        key: jax.Array
        model: object
        opt_state: object
        extras: dict = field(default_factory=dict)

    def linear(i):  # no function leaf: orbax cannot serialise callables (eqx.nn.MLP.activation)
        return eqx.nn.Linear(3, 2, key=jax.random.key(i))

    model = linear(0)
    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    old = ResumableState(
        step=jnp.asarray(7, jnp.int32),
        key=jax.random.PRNGKey(3),
        model=model,
        opt_state=opt_state,
        extras={"e": jnp.arange(3.0)},
    )
    manager = get_checkpoint_manager(tmp_path / "ckpt")
    save_checkpoint(manager, old)

    template = StructState(
        step=jnp.asarray(0, jnp.int32),
        key=jax.random.PRNGKey(0),
        model=linear(5),
        opt_state=opt.init(eqx.filter(linear(5), eqx.is_array)),
        extras={"e": jnp.zeros(3)},
    )
    restored = load_checkpoint(manager, template)
    assert isinstance(restored, StructState) and isinstance(restored.model, eqx.nn.Linear)
    got = jax.tree_util.tree_leaves(restored)
    want = jax.tree_util.tree_leaves(old)
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        np.testing.assert_array_equal(a, b)
    # control: the template's own values were not what came back
    assert int(template.step) == 0 and int(restored.step) == 7
