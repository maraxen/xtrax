"""IO utilities for async prefetching and bounded callback handling."""

import asyncio
import logging
from collections.abc import AsyncIterator, Coroutine, Iterable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Sentinel values for queue signaling
_DONE = object()  # Signals end of iteration
_ERROR = object()  # Marker for exceptions in queue


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
    # Create a queue to hold prefetched items or sentinel values
    queue: asyncio.Queue[tuple[int, T] | tuple[object, BaseException] | object] = (
        asyncio.Queue(maxsize=buffer_size)
    )

    exception_holder: BaseException | None = None

    async def producer():
        """Background task that prefetches items into the queue.

        Uses asyncio.to_thread to run the blocking iterator in a thread pool,
        preventing the event loop from being blocked by slow iterables (e.g., file I/O).
        """
        nonlocal exception_holder

        def fetch_next(iterator):
            """Fetch next item from iterator; can be run in thread pool.

            Returns the item or _DONE sentinel for iteration end.
            This avoids raising StopIteration which can't cross thread boundaries.
            """
            try:
                return next(iterator)
            except StopIteration:
                return _DONE

        try:
            iterator = iter(iterable)
            index = 0
            while True:
                # Run blocking next() in thread pool via asyncio.to_thread
                item = await asyncio.to_thread(fetch_next, iterator)

                # Check if iteration is complete
                if item is _DONE:
                    await queue.put(_DONE)
                    break

                # Successfully got item; add indexed item to queue
                await queue.put((index, item))
                index += 1
        except BaseException as e:
            # Capture exception to re-raise in consumer
            exception_holder = e
            # Signal exception with sentinel tuple
            await queue.put((_ERROR, e))

    # Start the producer task
    producer_task = asyncio.create_task(producer())

    try:
        # Consumer loop
        while True:
            # Get next item from queue
            item = await queue.get()

            # Check for end of iteration
            if item is _DONE:
                break

            # Check for exception marker
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _ERROR:
                raise item[1]

            # Normal item - unpack and yield
            index, value = item
            yield (index, value)
    finally:
        # Ensure producer task is cleaned up
        if not producer_task.done():
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

    Example:
        Import from the public surface and use with asyncio:

        >>> from xtrax.io import BoundedCallbackHandler
        >>> import asyncio
        >>> async def example():
        ...     handler = BoundedCallbackHandler(max_concurrent=2)
        ...     values = []
        ...     async def callback(x):
        ...         values.append(x)
        ...     await handler.submit(callback(1))
        ...     await handler.submit(callback(2))
        ...     await handler.wait_all()
        ...     return sorted(values)
        >>> asyncio.run(example())
        [1, 2]
    """

    def __init__(self, max_concurrent: int = 4) -> None:
        """Initialize the handler.

        Args:
            max_concurrent: Maximum number of concurrent coroutines (default: 4).
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending_tasks: set[asyncio.Task[None]] = set()

    async def submit(self, coro: Coroutine[None, None, None]) -> None:
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
                    logger.exception("callback error")

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
