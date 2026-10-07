"""jax.lax.map(batch_size=k) vmaps every chunk, so a chunk of length 1 is a vmap-of-1.

That is the pattern aminx #2391 found silently miscompiling. These tests pin the
guard: batch_size=1, a remainder of 1, a leading axis of length 1, and the same
cases through ChunkedMapIterator / VmapIterator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from xtrax.stages.executor import execute_map_axis
from xtrax.tiling.iterator import ChunkedMapIterator, VmapIterator
from xtrax.tiling.strategy import ChunkedMap
from xtrax.transforms.map import chunked_map

# Indexing and stitching touch a length-1 axis on purpose. The banned pattern is a
# compute op, or a lax.map scan body, whose input is that length-1 mapped axis.
_STRUCTURAL = frozenset(
    {
        "broadcast_in_dim",
        "concatenate",
        "convert_element_type",
        "dynamic_slice",
        "gather",
        "reshape",
        "slice",
        "squeeze",
    }
)


def _double(x):
    return x * 2 + 1


def _python_loop(fn, xs):
    n = jax.tree.leaves(xs)[0].shape[0]
    rows = [fn(jax.tree.map(lambda leaf, i=i: leaf[i], xs)) for i in range(n)]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows)


def _size1_batched_sites(closed, element_shape: tuple[int, ...]) -> list[str]:
    """Names of compute ops or scan-body inputs shaped (1, *element_shape)."""
    bad = (1, *element_shape)
    sites: list[str] = []

    def walk(jaxpr, where: str) -> None:
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "scan":
                sub = eqn.params["jaxpr"].jaxpr
                for invar in sub.invars:
                    if getattr(getattr(invar, "aval", None), "shape", None) == bad:
                        sites.append(f"{where}/scan-body {invar.aval.str_short()}")
                walk(sub, f"{where}/scan")
                continue
            if eqn.primitive.name not in _STRUCTURAL:
                for invar in eqn.invars:
                    aval = getattr(invar, "aval", None)
                    if getattr(aval, "shape", None) == bad:
                        sites.append(f"{where}/{eqn.primitive.name} {aval.str_short()}")
            for value in eqn.params.values():
                nested = getattr(value, "jaxpr", None)
                if nested is not None and hasattr(nested, "eqns"):
                    walk(nested, f"{where}/{eqn.primitive.name}")

    walk(closed.jaxpr, "top")
    return sites


def _jaxpr_of(thunk, example):
    return jax.make_jaxpr(thunk)(example)


@pytest.mark.parametrize(
    ("n", "batch_size"),
    [(5, 1), (4, 3), (1, 1), (1, None), (7, 3)],
)
def test_chunked_map_jaxpr_has_no_size1_batch(n, batch_size):
    """No lax.map chunk or compute op may see a mapped axis of length 1."""
    element_shape = (3,)
    xs = jnp.ones((n, *element_shape))
    closed = _jaxpr_of(lambda x: chunked_map(_double, x, batch_size=batch_size), xs)
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"n={n} batch_size={batch_size}: {sites}\n{closed}"


def test_chunked_map_iterator_jaxpr_has_no_size1_batch():
    """ChunkedMapIterator(tile=1) and a size-1 axis both go through the guard."""
    element_shape = (3,)
    iterator = ChunkedMapIterator(tile=1)
    xs = jnp.ones((5, *element_shape))
    closed = _jaxpr_of(lambda x: iterator(_double, x), xs)
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"tile=1: {sites}\n{closed}"

    single = ChunkedMapIterator(tile=4)
    xs1 = jnp.ones((1, *element_shape))
    closed1 = _jaxpr_of(lambda x: single(_double, x), xs1)
    sites1 = _size1_batched_sites(closed1, element_shape)
    assert not sites1, f"size-1 axis: {sites1}\n{closed1}"


def test_vmap_iterator_size1_axis_jaxpr_has_no_size1_batch():
    """VmapIterator on a length-1 axis must call the function, not vmap it."""
    element_shape = (3,)
    xs = jnp.ones((1, *element_shape))
    closed = _jaxpr_of(lambda x: VmapIterator()(_double, x), xs)
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"{sites}\n{closed}"


def test_vmap_iterator_nonleading_size1_axis_matches_vmap_without_batching_it():
    """A size-1 axis that is not axis 0 is still applied unbatched."""
    xs = jnp.arange(12, dtype=jnp.float32).reshape(3, 1, 4)
    got = VmapIterator()(_double, xs, in_axes=1)
    assert jnp.array_equal(got, jax.vmap(_double, in_axes=1)(xs))

    closed = _jaxpr_of(lambda x: VmapIterator()(_double, x, in_axes=1), xs)
    batched = [
        f"{eqn.primitive.name} {invar.aval.str_short()}"
        for eqn in closed.jaxpr.eqns
        if eqn.primitive.name not in _STRUCTURAL
        for invar in eqn.invars
        if getattr(getattr(invar, "aval", None), "shape", None) == (3, 1, 4)
    ]
    assert not batched, f"{batched}\n{closed}"


def test_execute_map_axis_chunked_batch_size_one_jaxpr_has_no_size1_batch():
    """execute_map_axis(ChunkedMap(batch_size=1)) must not reintroduce the vmap."""
    element_shape = (3,)
    xs = jnp.ones((5, *element_shape))
    closed = _jaxpr_of(
        lambda x: execute_map_axis(_double, x, ChunkedMap(batch_size=1)),
        xs,
    )
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"{sites}\n{closed}"


@pytest.mark.parametrize(
    ("n", "batch_size"),
    [(5, 1), (4, 3), (7, 3), (1, 1), (1, None), (1, 8), (5, 3), (6, 2)],
)
def test_chunked_map_matches_python_loop(n, batch_size):
    xs = jnp.arange(n * 3, dtype=jnp.float32).reshape(n, 3)
    got = chunked_map(_double, xs, batch_size=batch_size)
    expected = _python_loop(_double, xs)
    assert got.shape == expected.shape
    assert jnp.array_equal(got, expected)


def test_chunked_map_pytree_remainder_matches_python_loop():
    xs = {
        "a": jnp.arange(4, dtype=jnp.float32).reshape(4, 1),
        "b": jnp.arange(4, 8, dtype=jnp.float32).reshape(4, 1),
    }

    def fn(tree):
        return {"s": tree["a"] + tree["b"], "p": tree["a"] * tree["b"]}

    got = chunked_map(fn, xs, batch_size=3)
    expected = _python_loop(fn, xs)
    for left, right in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
        assert jnp.array_equal(left, right)


def test_iterators_match_python_loop_on_size1_cases():
    xs = jnp.arange(5, dtype=jnp.float32)
    assert jnp.array_equal(ChunkedMapIterator(tile=1)(_double, xs), _python_loop(_double, xs))

    ragged = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    assert jnp.array_equal(
        ChunkedMapIterator(tile=3)(_double, ragged), _python_loop(_double, ragged)
    )

    single = jnp.arange(3, dtype=jnp.float32).reshape(1, 3)
    assert jnp.array_equal(VmapIterator()(_double, single), _python_loop(_double, single))
    assert jnp.array_equal(
        ChunkedMapIterator(tile=8)(_double, single), _python_loop(_double, single)
    )
