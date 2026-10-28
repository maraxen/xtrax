"""Throwaway: what does eqx.field(static=True) produce, and does an equinox
subclass of a native register_dataclass base keep static-ness?"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

f = eqx.field(static=True, default=1)
print("eqx.field type:", type(f).__name__, "metadata:", dict(f.metadata))


def native(cls):
    cls = dataclasses.dataclass(frozen=True)(cls)
    fs = dataclasses.fields(cls)
    meta = [x.name for x in fs if x.metadata.get("static")]
    data = [x.name for x in fs if not x.metadata.get("static")]
    return jax.tree_util.register_dataclass(cls, data_fields=data, meta_fields=meta)


@native
class Base:
    """Native stand-in for RunSpec."""

    a: jax.Array
    run_id: str | None = dataclasses.field(default=None, metadata=dict(static=True))


@native
class Sub(Base):
    """plegadx-style subclass declaring eqx.field(static=True)."""

    n_recycles: int = eqx.field(static=True, default=1)


s = Sub(jnp.ones(2), run_id="r", n_recycles=3)
print("Sub leaves:", jax.tree_util.tree_leaves(s))
print("Sub isinstance eqx.Module:", isinstance(s, eqx.Module))
try:

    class EqxSub(Base, eqx.Module):
        """Subclass mixing native base with eqx.Module."""

        n: int = eqx.field(static=True, default=1)

    print("mixing eqx.Module into native base: OK", EqxSub(jnp.ones(2)))
except Exception as e:  # noqa: BLE001
    print("mixing eqx.Module into native base FAILS:", type(e).__name__, str(e)[:150])
