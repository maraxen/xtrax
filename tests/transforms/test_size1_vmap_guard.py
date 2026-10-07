"""jax.lax.map(batch_size=k) vmaps every chunk, so a chunk of length 1 is a vmap-of-1.

That is the pattern aminx #2391 found silently miscompiling. These tests pin the
guard: batch_size=1, a remainder of 1, a leading axis of length 1, and every
xtrax-dispatched vmap (VmapIterator, execute_map_axis, axis_dispatch, and the
dedup-synthesis row maps).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.stages.executor import execute_map_axis
from xtrax.tiling.dedup import DedupSpec
from xtrax.tiling.dedup_synthesis import _dedup_output_maps
from xtrax.tiling.dispatch import axis_dispatch
from xtrax.tiling.iterator import ChunkedMapIterator, VmapIterator
from xtrax.tiling.strategy import ChunkedMap, Vmap
from xtrax.transforms.map import _apply_size1, _is_size1_axis, chunked_map

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


def _double_cond(x):
    """Mapped body whose branch is a lax.cond, so a vmap-of-1 can hide in branches."""
    return jax.lax.cond(x[0] > 0, lambda y: y * 2, lambda y: y + 3, x)


def _python_loop(fn, xs):
    n = jax.tree.leaves(xs)[0].shape[0]
    rows = [fn(jax.tree.map(lambda leaf, i=i: leaf[i], xs)) for i in range(n)]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *rows)


def _visit_nested(value, visit) -> None:
    """Walk jaxpr params, including tuple/list containers such as cond branches."""
    if isinstance(value, tuple | list):
        for item in value:
            _visit_nested(item, visit)
        return
    if hasattr(value, "eqns"):
        visit(value)
        return
    nested = getattr(value, "jaxpr", None)
    if nested is not None and hasattr(nested, "eqns"):
        visit(nested)


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
            where_nested = f"{where}/{eqn.primitive.name}"

            def visit(nested, where_nested=where_nested):
                walk(nested, where_nested)

            for value in eqn.params.values():
                _visit_nested(value, visit)

    walk(closed.jaxpr, "top")
    return sites


def _contains_primitive(closed, name: str) -> bool:
    found = False

    def walk(jaxpr) -> None:
        nonlocal found
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == name:
                found = True
            for value in eqn.params.values():
                _visit_nested(value, walk)

    walk(closed.jaxpr)
    return found


def _jaxpr_of(thunk, example):
    return jax.make_jaxpr(thunk)(example)


@pytest.mark.parametrize(
    ("n", "batch_size"),
    [(5, 1), (4, 3), (1, 1), (1, None), (7, 3), (5, 3), (8, 3)],
)
def test_chunked_map_jaxpr_has_no_size1_batch(n, batch_size):
    """No lax.map chunk or compute op may see a mapped axis of length 1.

    (5, 3) and (8, 3) leave a remainder of 2. Peeling one element there would
    make a later chunk length 1, so those shapes must stay unpeeled.
    """
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
    assert _contains_primitive(closed, "scan"), f"tile=1 ignored the tile\n{closed}"

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


def test_jaxpr_walker_sees_size1_batch_inside_cond_branches():
    """cond stores branch jaxprs in a tuple param; the walker must enter it.

    The length-1 batch is built inside the branch, so the cond eqn's own inputs
    are scalars. A walker that skips ``branches`` reports no size-1 site.
    """

    def thunk(s):
        def branch(_):
            row = jnp.stack([s, s + 1, s + 2])
            return jax.vmap(_double)(jnp.expand_dims(row, 0))

        return jax.lax.cond(s > 0, branch, lambda _: jnp.zeros((1, 3)), s)

    closed = _jaxpr_of(thunk, jnp.float32(1.0))
    sites = _size1_batched_sites(closed, (3,))
    assert sites, f"walker missed a size-1 vmap inside cond\n{closed}"


def test_execute_map_axis_vmap_size1_jaxpr_has_no_size1_batch():
    """execute_map_axis(Vmap) on a length-1 axis must call the function, not vmap it."""
    element_shape = (3,)
    xs = jnp.arange(3, dtype=jnp.float32).reshape(1, *element_shape)
    closed = _jaxpr_of(lambda x: execute_map_axis(_double, x, Vmap()), xs)
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"{sites}\n{closed}"
    assert jnp.array_equal(execute_map_axis(_double, xs, Vmap()), _python_loop(_double, xs))


def test_execute_map_axis_vmap_cond_body_has_no_size1_batch():
    """A lax.cond inside the mapped function must not hide a vmap-of-1."""
    element_shape = (3,)
    xs = jnp.arange(3, dtype=jnp.float32).reshape(1, *element_shape)
    closed = _jaxpr_of(lambda x: execute_map_axis(_double_cond, x, Vmap()), xs)
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"{sites}\n{closed}"
    assert jnp.array_equal(
        execute_map_axis(_double_cond, xs, Vmap()), _python_loop(_double_cond, xs)
    )


def test_axis_dispatch_vmap_size1_jaxpr_has_no_size1_batch():
    """axis_dispatch(Vmap) on a length-1 axis must call the function, not vmap it."""
    element_shape = (3,)
    xs = jnp.arange(3, dtype=jnp.float32).reshape(1, *element_shape)
    closed = _jaxpr_of(lambda x: axis_dispatch(Vmap(), _double, x), xs)
    sites = _size1_batched_sites(closed, element_shape)
    assert not sites, f"{sites}\n{closed}"
    assert jnp.array_equal(axis_dispatch(Vmap(), _double, xs), _python_loop(_double, xs))


def _size1_dedup_maps(fn, n: int):
    """K=1 spec: the deduped axis has length 1 (k_bucket(1) == 1)."""
    row = jnp.arange(3, dtype=jnp.float32)
    xs = jnp.broadcast_to(row, (n, 3))
    spec = DedupSpec(
        axis_name="b",
        unique_indices=np.array([0], dtype=np.int32),
        index_map=np.zeros(n, dtype=np.int32),
        k=1,
    )
    dg = spec.to_dedup_gather()
    per_row, dedup_path = _dedup_output_maps(
        fn,
        dedup_fn=dg.dedup_fn,
        gather_fn=dg.gather_fn,
        unique_idx=jnp.asarray(dg.unique_indices),
        index_map=jnp.asarray(dg.index_map),
    )
    return xs, per_row, dedup_path


def test_dedup_synthesis_size1_jaxpr_has_no_size1_batch():
    """verify_dedup_outputs' row maps must not vmap a length-1 axis."""
    element_shape = (3,)
    xs1, per_row, _dedup_n1 = _size1_dedup_maps(_double, n=1)
    closed_rows = _jaxpr_of(per_row, xs1)
    row_sites = _size1_batched_sites(closed_rows, element_shape)
    assert not row_sites, f"per-row N=1: {row_sites}\n{closed_rows}"
    assert jnp.array_equal(per_row(xs1), _python_loop(_double, xs1))

    xs3, _per_row_n3, dedup_path = _size1_dedup_maps(_double, n=3)
    closed_dedup = _jaxpr_of(dedup_path, xs3)
    dedup_sites = _size1_batched_sites(closed_dedup, element_shape)
    assert not dedup_sites, f"dedup K=1: {dedup_sites}\n{closed_dedup}"
    assert jnp.array_equal(dedup_path(xs3), _python_loop(_double, xs3))


def test_size1_helpers_accept_tree_in_axes_and_none_prefix():
    """Tree-structured in_axes, including None prefixes, take the direct call."""
    row = jnp.arange(3, dtype=jnp.float32).reshape(1, 3)
    bias = jnp.arange(4, dtype=jnp.float32)
    xs = (row, bias)
    in_axes = (0, None)
    assert _is_size1_axis(xs, in_axes)

    def fn(pair):
        mapped, static = pair
        return mapped * 2 + static[0]

    got = VmapIterator()(fn, xs, in_axes=in_axes)
    assert jnp.array_equal(got, jax.vmap(fn, in_axes=(in_axes,))(xs))
    closed = _jaxpr_of(lambda x: VmapIterator()(fn, x, in_axes=in_axes), xs)
    sites = _size1_batched_sites(closed, (3,))
    assert not sites, f"{sites}\n{closed}"

    tree = (
        jnp.arange(12, dtype=jnp.float32).reshape(3, 1, 4),
        jnp.arange(5, dtype=jnp.float32).reshape(5, 1),
    )
    axes = (1, 1)
    assert _is_size1_axis(tree, axes)

    def sum_pair(pair):
        a, b = pair
        return a.sum() + b.sum()

    assert jnp.array_equal(
        _apply_size1(sum_pair, tree, axes),
        jax.vmap(sum_pair, in_axes=(axes,))(tree),
    )

    nested = {
        "a": jnp.arange(3, dtype=jnp.float32).reshape(1, 3),
        "b": {
            "c": jnp.ones((2, 2), dtype=jnp.float32),
            "d": jnp.arange(2, dtype=jnp.float32),
        },
    }
    nested_axes = {"a": 0, "b": None}
    assert _is_size1_axis(nested, nested_axes)

    def nested_fn(tree):
        return tree["a"] * 2 + tree["b"]["c"].sum() + tree["b"]["d"].sum()

    assert jnp.array_equal(
        _apply_size1(nested_fn, nested, nested_axes),
        jax.vmap(nested_fn, in_axes=(nested_axes,))(nested),
    )
    assert not _is_size1_axis(
        (jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((9,), dtype=jnp.float32)),
        (0, None),
    )
    assert not _is_size1_axis(
        (jnp.ones((1, 3), dtype=jnp.float32), jnp.ones((4, 3), dtype=jnp.float32)),
        (0, 0),
    )
