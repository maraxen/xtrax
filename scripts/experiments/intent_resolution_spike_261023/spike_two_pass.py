"""Throwaway spike: two-pass resolution of intent sites.

pass 1 (abstract, jax.make_jaxpr): every xmap call records (site_key, input avals).
resolver: joint decision over all sites (chunk size per site from a shared budget).
pass 2 (jax.jit trace): every xmap call looks up its decision by the same key and
checks its avals equal pass 1's; any mismatch raises before execution.

Checks:
  t1 a Python loop calling one source line at 3 shapes gets 3 distinct stable keys
  t2 decisions take effect (jaxpr shows lax.map/scan for big sites, no scan for small)
  t3 result equals plain vmap reference
  t4 non-reproducible trace (shape changes between passes) is caught with a clear error
  t5 calling the compiled fn again does not re-plan or retrace
"""

import contextvars
import dataclasses
import inspect

import jax
import jax.numpy as jnp

_SESSION = contextvars.ContextVar("xtrax_session", default=None)


@dataclasses.dataclass
class _Session:
    mode: str  # "plan" | "exec"
    sites: list = dataclasses.field(default_factory=list)  # [(key, avals)]
    decisions: dict = dataclasses.field(default_factory=dict)
    cursor: dict = dataclasses.field(default_factory=dict)
    seen: list = dataclasses.field(default_factory=list)


def _site_key(name):
    s = _SESSION.get()
    fr = inspect.stack()[2]
    loc = name or f"{fr.filename.rsplit('/', 1)[-1]}:{fr.lineno}"
    n = s.cursor.get(loc, 0)
    s.cursor[loc] = n + 1
    return (loc, n)


def xmap(f, xs, *, name=None):
    """Intent: map f over the leading axis of xs; strategy chosen by the resolver."""
    s = _SESSION.get()
    if s is None:
        return jax.vmap(f)(xs)
    key = _site_key(name)
    avals = jax.tree_util.tree_map(lambda a: (a.shape, str(a.dtype)), xs)
    if s.mode == "plan":
        s.sites.append((key, avals))
        return jax.vmap(f)(xs)
    s.seen.append((key, avals))
    planned = dict(s.sites)
    if key not in planned or planned[key] != avals:
        raise RuntimeError(
            f"intent site {key} not reproducible: planned {planned.get(key)}, traced {avals}"
        )
    bs = s.decisions[key]
    return jax.vmap(f)(xs) if bs is None else jax.lax.map(f, xs, batch_size=bs)


def _resolve(sites, budget_rows):
    """Joint decision: sites whose leading axis exceeds the shared budget get chunked."""
    out = {}
    for key, avals in sites:
        shape, _dtype = jax.tree_util.tree_leaves(avals, is_leaf=lambda x: isinstance(x, tuple))[0]
        n = shape[0]
        out[key] = None if n <= budget_rows else budget_rows
    return out


def xcompile(fn, *, budget_rows):
    """Plan once from abstract inputs, then jit with decisions applied."""
    plan = {}

    def run(*args):
        tok = _SESSION.set(_Session(mode="exec", sites=plan["sites"], decisions=plan["decisions"]))
        try:
            out = fn(*args)
            s = _SESSION.get()
            if [k for k, _ in s.seen] != [k for k, _ in plan["sites"]]:
                raise RuntimeError("intent site sequence differs between plan and trace")
            return out
        finally:
            _SESSION.reset(tok)

    jitted = jax.jit(run)

    def call(*args):
        if not plan:
            tok = _SESSION.set(_Session(mode="plan"))
            try:
                jax.make_jaxpr(fn)(*args)
                sites = _SESSION.get().sites
            finally:
                _SESSION.reset(tok)
            plan.update(sites=sites, decisions=_resolve(sites, budget_rows))
            call.plans += 1
        return jitted(*args)

    call.plans = 0
    call.jitted = jitted
    call.plan = plan
    return call


def layer(x):
    return jnp.tanh(x * 2.0).sum()


def model(xs_list):
    return [xmap(layer, xs) for xs in xs_list]  # one source line, three shapes


inputs = [jnp.ones((n, 4)) for n in (8, 64, 512)]
ref = [jax.vmap(layer)(x) for x in inputs]

f = xcompile(model, budget_rows=32)
out = f(inputs)
keys = [k for k, _ in f.plan["sites"]]
print("t1 keys:", keys, "| distinct:", len(set(keys)) == 3)
print("t1 decisions:", list(f.plan["decisions"].values()))
jx = str(jax.make_jaxpr(lambda xs: f.jitted(xs))(inputs))
print("t2 jaxpr has scan (chunked sites):", "scan[" in jx)
print("t3 matches vmap reference:", all(bool(jnp.allclose(a, b)) for a, b in zip(out, ref)))

flip = {"n": 0}


def flaky(xs):
    flip["n"] += 1
    x = xs if flip["n"] == 1 else xs[:-1]  # trace-time side effect changes shapes
    return xmap(layer, x)


g = xcompile(flaky, budget_rows=32)
try:
    g(jnp.ones((16, 4)))
    print("t4 non-reproducible trace: NOT caught (bad)")
except RuntimeError as e:
    print("t4 caught:", str(e)[:110])

before = f.plans
f(inputs)
f(inputs)
print("t5 plans after 3 calls:", f.plans, "| jit cache size:", f.jitted._cache_size())
