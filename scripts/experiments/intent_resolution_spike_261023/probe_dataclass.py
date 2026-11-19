import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class S:
    a: jax.Array
    n: int = dataclasses.field(default=3, metadata=dict(static=True))


s = S(jnp.ones(2))
print("leaves", jax.tree_util.tree_leaves(s))
s2 = eqx.tree_at(lambda t: t.a, s, jnp.zeros(2))
print("tree_at ok", type(s2).__name__, s2.n, s2.a)
traces = []


@jax.jit
def f(s):
    traces.append(1)
    return s.a * s.n


f(s)
f(S(jnp.zeros(2)))
print("traces same static:", len(traces))
f(S(jnp.zeros(2), n=4))
print("traces new static:", len(traces))


class M(eqx.Module):
    inner: S
    w: jax.Array


print(
    "eqx.Module holding S, filter_jit:",
    eqx.filter_jit(lambda m: m.w + m.inner.a)(M(s, jnp.ones(2))),
)
print("isinstance(S, eqx.Module):", isinstance(s, eqx.Module))
