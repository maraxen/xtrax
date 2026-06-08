"""Tests for async_indexed_stream and BoundedCallbackHandler from io/callbacks.py."""

import asyncio

import pytest

from xtrax.io.callbacks import BoundedCallbackHandler, async_indexed_stream


class TestAsyncIndexedStream:
    """Tests for async_indexed_stream prefetching async iterator."""

    @pytest.mark.asyncio
    async def test_async_indexed_stream_normal(self):
        """Normal list iterable yields (index, item) tuples in order."""
        items = [10, 20, 30]
        result = []

        async for idx, item in async_indexed_stream(items):
            result.append((idx, item))

        assert result == [(0, 10), (1, 20), (2, 30)]

    @pytest.mark.asyncio
    async def test_async_indexed_stream_buffer_size_1(self):
        """With buffer_size=1, all items yield in order."""
        items = [1, 2, 3, 4, 5]
        result = []

        async for idx, item in async_indexed_stream(items, buffer_size=1):
            result.append((idx, item))

        assert len(result) == 5
        assert result == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]

    @pytest.mark.asyncio
    async def test_async_indexed_stream_exception_propagates(self):
        """Exception from iterable propagates from async for loop."""

        def failing_iter():
            yield 1
            yield 2
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError, match="oops"):
            async for idx, item in async_indexed_stream(failing_iter()):
                pass

    @pytest.mark.asyncio
    async def test_bounded_handler_submit_normal(self):
        """Submit a normal coroutine that sets a flag."""
        handler = BoundedCallbackHandler()
        flag = []

        async def task():
            flag.append(True)

        await handler.submit(task())
        await handler.wait_all()

        assert flag == [True]

    @pytest.mark.asyncio
    async def test_bounded_handler_submit_exception_logged(self, caplog):
        """Submit a coroutine that raises; exception logged at ERROR level."""
        import logging

        handler = BoundedCallbackHandler()

        async def failing_task():
            raise ValueError("test error")

        with caplog.at_level(logging.ERROR, logger="xtrax.io.callbacks"):
            await handler.submit(failing_task())
            await handler.wait_all()

        # Check that the exception was logged
        assert any("test error" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_bounded_handler_wait_all_empty(self):
        """wait_all() with no tasks completes immediately."""
        handler = BoundedCallbackHandler()
        # Should not hang
        await handler.wait_all()

    @pytest.mark.asyncio
    async def test_bounded_handler_max_concurrent_1(self):
        """With max_concurrent=1, two submitted tasks complete without deadlock."""
        handler = BoundedCallbackHandler(max_concurrent=1)
        completed = []

        async def task(task_id):
            completed.append(task_id)
            await asyncio.sleep(0.001)

        await handler.submit(task(1))
        await handler.submit(task(2))
        await handler.wait_all()

        assert len(completed) == 2
        assert set(completed) == {1, 2}
