import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

from xtrax.sparse.config import SparseConfig
from xtrax.sparse.policy import SparsePolicy


def _make_policy(budget, schedule=None, fallback="dense_mask"):
    return SparsePolicy(
        config=SparseConfig(
            nse_budget=budget,
            update_schedule=schedule or (lambda s: True),
            fallback_mode=fallback,
        )
    )


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
        assert int(jnp.sum(mask)) == 4

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

    def test_make_mask_all_ties_exact_budget(self):
        # All weights equal → tie-breaking is arbitrary but count must be exact.
        p = _make_policy(budget=4)
        w = jnp.ones((3, 3))  # 9 elements, all magnitude 1.0
        mask = p.make_mask(w, step=0)
        assert mask.shape == w.shape
        assert mask.dtype == jnp.bool_
        assert int(jnp.sum(mask)) == 4  # exactly nse_budget, not ≥

    def test_apply_mask_padding_no_zero_zero_alias(self):
        # nse_budget=4, but only 2 True values in mask — 2 padding slots → (0,0).
        # weight[0,0] is non-zero but mask[0,0] is False.
        # The padded (0,0) alias must be zeroed, so todense()[0,0] == 0.0.
        p = _make_policy(budget=4)
        w = jnp.array([[5.0, 2.0, 3.0], [0.0, 9.0, 0.0], [0.0, 0.0, 7.0]])
        # Only (1,1) and (2,2) are True — 2 trues, 2 padding slots pointing to (0,0).
        mask = jnp.array(
            [[False, False, False], [False, True, False], [False, False, True]]
        )
        result = p.apply_mask(w, mask)
        assert isinstance(result, BCOO)
        dense = result.todense()
        assert float(dense[0, 0]) == 0.0, f"(0,0) alias leaked: {dense[0, 0]}"
        assert float(dense[1, 1]) == pytest.approx(9.0, abs=1e-6)
        assert float(dense[2, 2]) == pytest.approx(7.0, abs=1e-6)
