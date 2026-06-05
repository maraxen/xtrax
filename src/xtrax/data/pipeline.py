from collections.abc import AsyncIterator, Iterable
from typing import Any


async def async_indexed_stream[T](
    iterable: Iterable[T],
) -> AsyncIterator[tuple[int, T]]:
    """Async generator that yields (index, item) tuples from iterable."""
    for i, item in enumerate(iterable):
        yield i, item


def create_distributed_pipeline(
    dataset: Any,
    global_batch_size: int,
    num_devices: int,
    seed: int,
) -> Any:
    """Create distributed pipeline with per-device batch size validation.

    Raises ValueError if global_batch_size is not divisible by num_devices.
    Stub implementation — real grain sharding deferred to Phase 5/6.
    """
    if global_batch_size % num_devices != 0:
        raise ValueError(
            f"create_distributed_pipeline: global_batch_size={global_batch_size} "
            f"must be divisible by num_devices={num_devices}."
        )
    return dataset
