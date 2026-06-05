import pytest

from xtrax.data.pipeline import (
    async_indexed_stream,
    create_distributed_pipeline,
)


class TestAsyncIndexedStream:
    """Tests for async_indexed_stream."""

    @pytest.mark.asyncio
    async def test_async_indexed_stream_yields_correct_indices_and_items(self):
        """async_indexed_stream yields (index, item) tuples with correct indices."""
        items = ["a", "b", "c"]
        result = []
        async for idx, item in async_indexed_stream(items):
            result.append((idx, item))
        assert result == [(0, "a"), (1, "b"), (2, "c")]


class TestCreateDistributedPipeline:
    """Tests for create_distributed_pipeline."""

    def test_create_distributed_pipeline_accepts_valid_divisible_batch(self):
        """create_distributed_pipeline accepts divisible global_batch_size."""
        dataset = [1, 2, 3, 4, 5, 6]
        pipeline = create_distributed_pipeline(
            dataset=dataset,
            global_batch_size=6,
            num_devices=2,
            seed=42,
        )
        # Should not raise; pipeline should be a dataset or iterable
        assert pipeline is not None

    def test_create_distributed_pipeline_raises_on_non_divisible_batch(self):
        """Raises ValueError if global_batch_size not divisible by num_devices."""
        dataset = [1, 2, 3, 4, 5]
        with pytest.raises(
            ValueError, match="global_batch_size=5 must be divisible by num_devices=2"
        ):
            create_distributed_pipeline(
                dataset=dataset,
                global_batch_size=5,
                num_devices=2,
                seed=42,
            )

    def test_create_distributed_pipeline_accepts_seed_param(self):
        """create_distributed_pipeline accepts and uses seed parameter."""
        dataset = [1, 2, 3, 4]
        pipeline = create_distributed_pipeline(
            dataset=dataset,
            global_batch_size=4,
            num_devices=2,
            seed=123,
        )
        assert pipeline is not None
