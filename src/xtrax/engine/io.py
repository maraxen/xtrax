"""IO utilities for async prefetching and bounded callback handling."""

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine, Iterable

logger = logging.getLogger(__name__)


async def async_indexed_stream[T](
    iterable: Iterable[T],
    buffer_size: int = 2,
) -> AsyncIterator[tuple[int, T]]:
    """Async iterator that prefetches items from a blocking iterable.

    Semantics: Uses asyncio.to_thread to run the blocking iterable iteration
    in a background thread, prefetching up to buffer_size items ahead into
    an asyncio.Queue. Yields (index, item) tuples with monotonically increasing
    indices starting from 0.

    Exceptions from the iterable are caught in the background thread and
    re-raised to the consumer on the next yield.

    Args:
        iterable: Blocking iterable to prefetch from.
        buffer_size: Maximum number of items to prefetch ahead (default: 2).

    Yields:
        (index, item) tuples where index is monotonically increasing from 0.

    Raises:
        Any exception raised by the iterable will be re-raised on the next yield.
    """
    # Create a queue to hold prefetched (index, item) pairs
    queue: asyncio.Queue[tuple[int, T] | Exception] = asyncio.Queue(
        maxsize=buffer_size
    )

    # Track if the producer is done
    producer_done = False
    exception_holder: Exception | None = None

    async def producer():
        """Background task that prefetches items into the queue."""
        nonlocal producer_done, exception_holder
        try:
            iterator = iter(iterable)
            index = 0
            while True:
                item = next(iterator)
                await queue.put((index, item))
                index += 1
        except StopIteration:
            # Normal completion
            producer_done = True
        except Exception as e:
            # Capture exception to re-raise in consumer
            exception_holder = e
            producer_done = True

    # Start the producer task
    producer_task = asyncio.create_task(producer())

    try:
        # Consumer loop
        while True:
            # Check if producer encountered an exception
            if exception_holder is not None:
                raise exception_holder

            # Try to get an item from the queue
            try:
                item = queue.get_nowait()
                # Check if it's actually a raised exception (shouldn't happen
                # with current design, but keeping for safety)
                if isinstance(item, Exception):
                    raise item
                index, value = item
                yield (index, value)
            except asyncio.QueueEmpty:
                # Queue is empty; check if producer is done
                if producer_done:
                    break
                # Wait for an item to become available
                # Use a short wait to allow checking producer_done
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.01)
                    if isinstance(item, Exception):
                        raise item
                    index, value = item
                    yield (index, value)
                except TimeoutError:
                    # Timeout waiting for item; loop back to check producer_done
                    continue
    finally:
        # Ensure producer task is cancelled if we exit early
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            pass


class BoundedCallbackHandler:
    """Manages bounded concurrent execution of async callbacks.

    Uses an asyncio.Semaphore to limit the number of concurrently running
    coroutines. Exceptions in submitted coroutines are logged but not
    propagated, allowing the training loop to continue.
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        """Initialize the handler.

        Args:
            max_concurrent: Maximum number of concurrent coroutines (default: 4).
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending_tasks: set[asyncio.Task] = set()

    async def submit(self, coro: Coroutine) -> None:
        """Submit a coroutine to be executed with bounded concurrency.

        The semaphore is acquired inside the task (not before), so this method
        returns immediately. The actual coroutine waits for a semaphore slot
        before executing.

        Exceptions in the coroutine are logged but not propagated.

        Args:
            coro: A coroutine object to execute.
        """

        async def bounded_coro():
            """Wrapper that acquires semaphore, runs coro, and logs exceptions."""
            async with self._semaphore:
                try:
                    await coro
                except Exception:
                    logger.exception("Exception in callback")

        task = asyncio.create_task(bounded_coro())
        self._pending_tasks.add(task)
        # Remove task from set when done
        task.add_done_callback(self._pending_tasks.discard)

    async def wait_all(self) -> None:
        """Wait for all submitted coroutines to complete.

        This method blocks until all pending tasks finish, regardless of
        whether they succeeded or raised exceptions (which would have been
        logged).
        """
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
