import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

from xtrax.sparse.config import SparseConfig
from xtrax.sparse.policy import SparsePolicy


def _make_policy(budget, schedule=None, fallback="dense_mask"):
    return SparsePolicy(config=SparseConfig(
        nse_budget=budget,
        update_schedule=schedule or (lambda s: True),
        fallback_mode=fallback,
    ))


class TestSparsePolicy:
    def test_should_update_delegates(self):
        p = _make_policy(4, schedule=lambda s: s == 0)
        assert p.should_update(0) is True
        assert p.should_update(1) is False

    def test_make_mask_shape_and_dtype(self):
        p = _make_policy(budget=4)
        w = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
        mask = p.make_mask(w, step=0)
        assert mask.shape == w.shape
        assert mask.dtype == jnp.bool_

    def test_apply_mask_bcoo_nse(self):
        p = _make_policy(budget=4)
        w = jnp.ones((3, 3))
        mask = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )
        result = p.apply_mask(w, mask)
        assert isinstance(result, BCOO)
        assert result.nse == 4

    def test_apply_mask_todense_equivalence(self):
        p = _make_policy(budget=4)
        w = jnp.arange(1, 10, dtype=jnp.float32).reshape(3, 3)
        mask = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )
        result = p.apply_mask(w, mask)
        assert jnp.allclose(result.todense(), w * mask, atol=1e-6)

    def test_apply_mask_dense_fallback(self):
        p = _make_policy(budget=2, fallback="dense_mask")
        w = jnp.ones((3, 3))
        mask = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )
        result = p.apply_mask(w, mask)
        assert not isinstance(result, BCOO)
        assert jnp.allclose(result, w * mask, atol=1e-6)

    def test_apply_mask_error_fallback_raises(self):
        p = _make_policy(budget=2, fallback="error")
        w = jnp.ones((3, 3))
        mask = jnp.array(
            [[True, True, False], [True, True, False], [False, False, False]]
        )
        with pytest.raises(ValueError, match="nse_budget"):
            p.apply_mask(w, mask)
