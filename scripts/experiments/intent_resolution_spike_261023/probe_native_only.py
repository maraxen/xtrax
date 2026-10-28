"""Throwaway: can plain jax (no equinox in xtrax) give safe struct + model handling?

1. Struct base via __init_subclass__: undecorated subclasses (plegadx style, with
   eqx.field(static=True)) become frozen registered dataclasses.
2. partition/combine + jax.jit + jax.value_and_grad train an eqx.nn.MLP (non-array
   leaves such as activation functions) with one compile, and grads equal eqx's.
"""

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class Struct:
    """Native base: frozen dataclass + pytree registration for every subclass."""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        dataclasses.dataclass(frozen=True)(cls)
        fs = dataclasses.fields(cls)
        jax.tree_util.register_dataclass(
            cls,
            data_fields=[f.name for f in fs if not f.metadata.get("static")],
            meta_fields=[f.name for f in fs if f.metadata.get("static")],
        )

    def __post_init__(self):
        for klass in reversed(type(self).__mro__):
            chk = klass.__dict__.get("__check_init__")
            if chk is not None:
                chk(self)


class RunSpec(Struct):
    """Stand-in for xtrax RunSpec."""

    a: jax.Array
    device_count: int | None = dataclasses.field(default=None, metadata=dict(static=True))

    def __check_init__(self):
        if self.device_count is not None and self.device_count < 1:
            raise ValueError("device_count must be positive")


class RF3RunSpec(RunSpec):
    """plegadx-style subclass, undecorated, using eqx.field."""

    n_recycles: int = eqx.field(static=True, default=1)


s = RF3RunSpec(jnp.ones(2), device_count=2, n_recycles=4)
print("1a subclass leaves:", len(jax.tree_util.tree_leaves(s)), "static kept:", s.n_recycles)
try:
    RF3RunSpec(jnp.ones(2), device_count=0)
    print("1b __check_init__ inherited: NOT RAISED (bad)")
except ValueError:
    print("1b __check_init__ inherited: raised (good)")
print("1c replace works:", dataclasses.replace(s, n_recycles=5).n_recycles)


def is_array(x):
    return isinstance(x, (jax.Array, jnp.ndarray)) or hasattr(x, "__jax_array__")


def partition(tree):
    arrays = jax.tree_util.tree_map(lambda x: x if is_array(x) else None, tree)
    rest = jax.tree_util.tree_map(lambda x: None if is_array(x) else x, tree)
    return arrays, rest


def combine(a, b):
    return jax.tree_util.tree_map(
        lambda x, y: y if x is None else x, a, b, is_leaf=lambda x: x is None
    )


class _Static:
    """Hashable-by-structure wrapper so non-array leaves ride as a static jit arg."""

    def __init__(self, tree):
        self.leaves, self.treedef = jax.tree_util.tree_flatten(tree)

    def __hash__(self):
        return hash((self.treedef, tuple(self.leaves)))

    def __eq__(self, other):
        return self.treedef == other.treedef and self.leaves == other.leaves

    @property
    def tree(self):
        return jax.tree_util.tree_unflatten(self.treedef, self.leaves)


traces = []


@functools.partial(jax.jit, static_argnums=1)
def _step(params, static, opt_state, x, y):
    traces.append(1)

    def loss(p):
        model = combine(p, static.tree)
        return jnp.mean((jax.vmap(model)(x) - y) ** 2)

    val, g = jax.value_and_grad(loss)(params)
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, val, g


key = jax.random.PRNGKey(0)
model = eqx.nn.MLP(3, 1, 16, 2, key=key)
x = jax.random.normal(key, (64, 3))
y = jnp.sum(x, axis=1, keepdims=True)
opt = optax.adam(1e-2)
params, static = partition(model)
opt_state = opt.init(params)

_, _, _, g_native = _step(params, _Static(static), opt_state, x, y)
g_eqx = eqx.filter_grad(lambda m: jnp.mean((jax.vmap(m)(x) - y) ** 2))(model)
same = jax.tree_util.tree_all(
    jax.tree_util.tree_map(
        lambda a, b: bool(jnp.allclose(a, b)), g_native, eqx.filter(g_eqx, eqx.is_array)
    )
)
print("2a grads equal eqx.filter_grad:", same)

losses = []
for _ in range(50):
    params, opt_state, val, _ = _step(params, _Static(static), opt_state, x, y)
    losses.append(float(val))
print(f"2b loss {losses[0]:.3f} -> {losses[-1]:.3f}; traces={len(traces)}")
trained = combine(params, static)
print("2c combined result is eqx.Module again:", isinstance(trained, eqx.Module))
