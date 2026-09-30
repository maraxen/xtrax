import jax
import jax.numpy as jnp

from xtrax.transforms.map import safe_map


class TestSafeMapVmapPath:
    """Tests for batch_size=None (vmap) and n <= batch_size cases."""

    def test_batch_size_none_vmap(self):
        """batch_size=None should use vmap."""
        xs = jnp.arange(10).reshape(10, 1)

        def identity(x):
            return x

        result = safe_map(identity, xs, batch_size=None)
        expected = jax.vmap(identity)(xs)

        assert jnp.allclose(result, expected)

    def test_n_less_than_batch_size_vmap(self):
        """n < batch_size should use vmap."""
        xs = jnp.arange(5).reshape(5, 1)

        def double(x):
            return x * 2

        result = safe_map(double, xs, batch_size=10)
        expected = jax.vmap(double)(xs)

        assert jnp.allclose(result, expected)

    def test_n_equals_batch_size_vmap(self):
        """n == batch_size should use vmap."""
        xs = jnp.arange(10).reshape(10, 1)

        def add_one(x):
            return x + 1

        result = safe_map(add_one, xs, batch_size=10)
        expected = jax.vmap(add_one)(xs)

        assert jnp.allclose(result, expected)

    def test_vmap_shape_invariant(self):
        """Output shape should match input shape (vmap path)."""
        xs = jnp.ones((7, 3, 4))

        def identity(x):
            return x

        result = safe_map(identity, xs, batch_size=None)
        assert result.shape == xs.shape


class TestSafeMapLaxPath:
    """Tests for n > batch_size (lax.map) cases."""

    def test_n_greater_batch_size_divisible(self):
        """n > batch_size (divisible) should use lax.map and match vmap."""
        xs = jnp.arange(20).reshape(20, 1)

        def square(x):
            return x**2

        result = safe_map(square, xs, batch_size=5)
        expected = jax.vmap(square)(xs)

        assert jnp.allclose(result, expected)

    def test_lax_map_with_multiple_dimensions(self):
        """lax.map should preserve shape across multiple dimensions."""
        xs = jnp.ones((12, 3, 4))

        def identity(x):
            return x

        result = safe_map(identity, xs, batch_size=4)
        expected = jax.vmap(identity)(xs)

        assert jnp.allclose(result, expected)
        assert result.shape == xs.shape

    def test_lax_map_with_computation(self):
        """lax.map should correctly apply function across batches."""
        xs = jnp.arange(100).reshape(100, 1)

        def add_ten(x):
            return x + 10

        result = safe_map(add_ten, xs, batch_size=25)
        expected = jax.vmap(add_ten)(xs)

        assert jnp.allclose(result, expected)


class TestSafeMapPyTree:
    """Tests for pytree inputs (dict with multiple keys)."""

    def test_pytree_dict_input(self):
        """safe_map should handle pytree inputs (dict)."""
        xs = {
            "a": jnp.arange(8).reshape(8, 1),
            "b": jnp.arange(8, 16).reshape(8, 1),
        }

        def extract_a(tree):
            return tree["a"]

        result = safe_map(extract_a, xs, batch_size=None)
        expected = jax.vmap(extract_a)(xs)

        assert jnp.allclose(result, expected)

    def test_pytree_dict_with_multiple_outputs(self):
        """safe_map should handle pytrees with multiple outputs."""
        xs = {
            "x": jnp.arange(6).reshape(6, 1),
            "y": jnp.arange(6, 12).reshape(6, 1),
        }

        def process(tree):
            return {
                "sum": tree["x"] + tree["y"],
                "prod": tree["x"] * tree["y"],
            }

        result = safe_map(process, xs, batch_size=None)
        expected = jax.vmap(process)(xs)

        assert jnp.allclose(result["sum"], expected["sum"])
        assert jnp.allclose(result["prod"], expected["prod"])

    def test_pytree_lax_map_path(self):
        """lax.map should work with pytrees and n > batch_size."""
        xs = {
            "a": jnp.arange(10).reshape(10, 1),
            "b": jnp.arange(10, 20).reshape(10, 1),
        }

        def add_trees(tree):
            return tree["a"] + tree["b"]

        result = safe_map(add_trees, xs, batch_size=2)
        expected = jax.vmap(add_trees)(xs)

        assert jnp.allclose(result, expected)


class TestSafeMapErrors:
    """Tests for error handling."""

    def test_non_divisible_raises_value_error(self):
        """#5565: n % batch_size != 0 runs a ragged final chunk and matches vmap."""
        xs = jnp.arange(10).reshape(10, 1)

        def fn(x):
            return x * 3 + 1

        assert jnp.array_equal(safe_map(fn, xs, batch_size=3), jax.vmap(fn)(xs))

    def test_non_divisible_error_message_format(self):
        """#5565: ragged chunking handles pytrees and a prime cardinality."""
        xs = {"a": jnp.arange(7.0).reshape(7, 1), "b": jnp.ones((7, 3))}

        def fn(r):
            return {"s": r["a"].sum() + r["b"].sum(), "a": r["a"] * 2}

        out, ref = safe_map(fn, xs, batch_size=2), jax.vmap(fn)(xs)
        for got, want in zip(jax.tree.leaves(out), jax.tree.leaves(ref), strict=True):
            assert jnp.array_equal(got, want)
