"""Asynchronous callback utilities for training (spec §3.27).

BoundedCallbackHandler: Manages concurrent callback execution with semaphore
bounded concurrency. async_indexed_stream: Prefetching async iterator wrapper
for efficient data loading.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine, Iterable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def async_indexed_stream(  # noqa: UP047
    iterable: Iterable[T],
    buffer_size: int = 2,
) -> AsyncIterator[tuple[int, T]]:
    """Prefetch items from an iterable in a background thread.

    Semantics:
      - Wraps blocking iteration in asyncio.to_thread
      - Prefetches up to buffer_size items ahead
      - Yields tuples of (index, item) with monotonically increasing indices
        starting at 0
      - Exceptions from the underlying iterable are caught and re-raised on
        the next yield

    Args:
        iterable: Blocking iterable (e.g., from Grain dataset)
        buffer_size: Number of items to prefetch ahead (default: 2)

    Yields:
        (index, item) tuples where index is monotonically increasing from 0
    """
    queue: asyncio.Queue[tuple[int, T] | Exception | None] = asyncio.Queue(buffer_size)
    index_counter = 0

    async def producer():
        """Background task that iterates through the iterable and enqueues items."""
        try:
            def iterate():
                """Run blocking iteration in a thread."""
                yield from iterable

            iterator = iterate()
            for item in iterator:
                await queue.put((index_counter + len(queue._queue), item))
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(None)  # Signal end of iteration

    # Start the producer task
    producer_task = asyncio.create_task(producer())

    try:
        while True:
            # Get next item from queue
            item = await queue.get()

            # None signals end of iteration
            if item is None:
                break

            # Re-raise exceptions caught from the iterator
            if isinstance(item, Exception):
                raise item

            # Yield the indexed item
            index, value = item
            yield index, value
            index_counter = index + 1
    finally:
        # Ensure producer task is cancelled/cleaned up
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


class BoundedCallbackHandler:
    """Manages concurrent callback execution with bounded concurrency.

    Callbacks are submitted as coroutines and executed asynchronously with
    a semaphore-bounded maximum concurrency level. Exceptions in callbacks
    are logged but do not cancel the training loop.

    Args:
        max_concurrent: Maximum number of concurrent callback coroutines (default: 4)
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        """Initialize handler with bounded semaphore.

        Args:
            max_concurrent: Maximum concurrent callbacks (default: 4)
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending: set[asyncio.Task] = set()

    async def submit(self, coro: Coroutine) -> None:
        """Submit a callback coroutine for execution.

        Acquires the semaphore before starting the coroutine and releases
        on completion or exception. Exceptions in the coroutine are logged
        (not propagated) to avoid cancelling the training loop.

        Args:
            coro: Coroutine to execute as a callback
        """

        async def bounded_coro():
            async with self._semaphore:
                try:
                    await coro
                except Exception as e:
                    logger.exception(
                        f"Callback execution failed (logged, not propagated): {e}"
                    )

        task = asyncio.create_task(bounded_coro())
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def wait_all(self) -> None:
        """Block until all submitted coroutines complete.

        Must be called at the end of training to flush pending callbacks
        and ensure all async work completes before shutdown.
        """
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
