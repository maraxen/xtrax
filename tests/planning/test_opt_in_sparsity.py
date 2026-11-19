"""Tests for opt-in sparsity specification in planning and execution (#3044)."""

import pytest

from xtrax.intent.context import Context
from xtrax.tiling.budget import MemoryBudget
from xtrax.tiling.plan import AxisSpec, BatchPlanner
from xtrax.tiling.sparse_spec import SparseSpec
from xtrax.tiling.strategy import ChunkedMap, Vmap


class TestOptInSparsity:
    """Verification controls for SparseSpec, Context.sparsity, and BatchPlanner sparsity logic."""

    def test_sparse_spec_validation(self):
        """SparseSpec validates density in (0.0, 1.0] and format."""
        spec = SparseSpec(density=0.05, format="bcoo")
        assert spec.density == 0.05
        assert spec.format == "bcoo"

        with pytest.raises(ValueError, match="density must be in"):
            SparseSpec(density=0.0)

        with pytest.raises(ValueError, match="density must be in"):
            SparseSpec(density=1.5)

        with pytest.raises(ValueError, match="density must be in"):
            SparseSpec(density=-0.1)

    def test_context_accepts_sparsity(self):
        """Context accepts scalar float or SparseSpec."""
        ctx_float = Context(sparsity=0.1)
        assert ctx_float.sparsity == 0.1

        spec = SparseSpec(density=0.02)
        ctx_spec = Context(sparsity=spec)
        assert ctx_spec.sparsity == spec

    def test_context_rejects_invalid_sparsity(self):
        """Context rejects out-of-range sparsity float."""
        with pytest.raises(ValueError, match="sparsity must be in"):
            Context(sparsity=0.0)

        with pytest.raises(ValueError, match="sparsity must be in"):
            Context(sparsity=2.0)

    def test_planner_scales_memory_and_marks_is_sparse(self):
        """BatchPlanner scales memory by density and sets is_sparse=True when density < 0.2."""
        # Spec with 1,000,000 elements, element_input_bytes = 1000
        # Dense bytes = 1,000,000 * 1000 = 1 GB.
        # With sparsity density = 0.05, sparse memory is 50 MB.
        # A budget of 100 MB would reject dense Vmap (1 GB > 100 MB demotes to ChunkedMap),
        # but easily accommodates sparse Vmap (50 MB <= 100 MB).

        dense_spec = AxisSpec(
            name="interactions",
            cardinality=1000,
            default_batch_size=100,
            element_input_bytes=1000,
            sparsity=None,
        )
        sparse_spec = AxisSpec(
            name="interactions",
            cardinality=1000,
            default_batch_size=100,
            element_input_bytes=1000,
            sparsity=SparseSpec(density=0.05),
        )

        # Budget of 200,000 bytes: dense requires 1,000,000 bytes -> demotes to ChunkedMap
        # sparse requires 50,000 bytes (under budget) -> keeps Vmap!
        budget = MemoryBudget(bytes=200_000)

        dense_plan = BatchPlanner(budget=budget).plan([dense_spec])
        dense_decision = dense_plan.decision_for("interactions")
        assert isinstance(dense_decision.strategy, ChunkedMap)
        assert not getattr(dense_decision, "is_sparse", False)

        sparse_plan = BatchPlanner(budget=budget).plan([sparse_spec])
        sparse_decision = sparse_plan.decision_for("interactions")
        assert isinstance(sparse_decision.strategy, Vmap)
        assert getattr(sparse_decision, "is_sparse", False) is True
