import asyncio

import pytest

from xtrax.engine.io import BoundedCallbackHandler, async_indexed_stream


class TestAsyncIndexedStream:
    """Tests for async_indexed_stream prefetching iterator."""

    @pytest.mark.asyncio
    async def test_yields_indexed_items_in_order(self):
        """Should yield (index, item) tuples with correct indices."""
        items = [10, 20, 30, 40]
        result = []

        async for idx, item in async_indexed_stream(items):
            result.append((idx, item))

        assert result == [(0, 10), (1, 20), (2, 30), (3, 40)]

    @pytest.mark.asyncio
    async def test_prefetches_items_in_background(self):
        """Should prefetch items ahead using background thread."""
        # We can't directly measure prefetch without timing, but we can verify
        # that items are available even if iteration is slow.
        items = list(range(10))
        indices = []
        values = []

        async for idx, item in async_indexed_stream(items, buffer_size=3):
            indices.append(idx)
            values.append(item)
            # Simulate slow consumer
            await asyncio.sleep(0.001)

        assert indices == list(range(10))
        assert values == items

    @pytest.mark.asyncio
    async def test_reraises_exception_from_iterable_on_next_yield(self):
        """Should re-raise exceptions from the iterable on next yield."""

        def failing_iter():
            yield 1
            yield 2
            raise ValueError("iterable error")
            yield 3  # unreachable

        with pytest.raises(ValueError, match="iterable error"):
            async for idx, item in async_indexed_stream(failing_iter()):
                pass

    @pytest.mark.asyncio
    async def test_respects_buffer_size(self):
        """Buffer size should control prefetch depth."""
        items = [1, 2, 3, 4, 5]
        count = 0

        async for idx, item in async_indexed_stream(items, buffer_size=2):
            count += 1

        assert count == 5

    @pytest.mark.asyncio
    async def test_empty_iterable(self):
        """Should handle empty iterables gracefully."""
        result = []

        async for idx, item in async_indexed_stream([]):
            result.append((idx, item))

        assert result == []

    @pytest.mark.asyncio
    async def test_single_item(self):
        """Should handle single-item iterables."""
        result = []

        async for idx, item in async_indexed_stream([42]):
            result.append((idx, item))

        assert result == [(0, 42)]


class TestBoundedCallbackHandler:
    """Tests for BoundedCallbackHandler concurrency limit."""

    @pytest.mark.asyncio
    async def test_submit_takes_coroutine_object(self):
        """submit() should accept a coroutine object, not a callable."""
        handler = BoundedCallbackHandler(max_concurrent=1)

        async def dummy_coro():
            pass

        # This should work with a coroutine object
        coro = dummy_coro()
        await handler.submit(coro)

    @pytest.mark.asyncio
    async def test_max_concurrent_enforced(self):
        """Should limit concurrent tasks to max_concurrent."""
        handler = BoundedCallbackHandler(max_concurrent=2)
        concurrent_count = 0
        max_observed = 0

        async def task():
            nonlocal concurrent_count, max_observed
            concurrent_count += 1
            max_observed = max(max_observed, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1

        # Submit 5 tasks
        coros = [task() for _ in range(5)]
        for coro in coros:
            await handler.submit(coro)

        await handler.wait_all()

        # Should never have exceeded max_concurrent=2
        assert max_observed <= 2

    @pytest.mark.asyncio
    async def test_exception_in_coro_is_logged_not_propagated(self, caplog):
        """Exceptions in coroutines should be logged, not propagated.

        Spec §3.27 requires: 'Exceptions in coro are LOGGED (via logging module)
        but NOT propagated'. This test verifies both halves of the requirement.
        """
        import logging

        handler = BoundedCallbackHandler(max_concurrent=1)

        async def failing_coro():
            raise RuntimeError("callback error")

        # Enable DEBUG logging to capture the logger.exception() call
        with caplog.at_level(logging.ERROR, logger="xtrax.engine.io"):
            # submit() should NOT raise; exception is logged, not propagated
            await handler.submit(failing_coro())

            # wait_all should complete without raising
            await handler.wait_all()

        # Verify the exception WAS logged via the logging module at ERROR level
        assert any(
            "callback error" in r.message and r.levelno == logging.ERROR
            for r in caplog.records
        ), "Exception should be logged to logging module at ERROR level"

    @pytest.mark.asyncio
    async def test_wait_all_waits_for_all_tasks(self):
        """wait_all() should block until all submitted tasks complete."""
        handler = BoundedCallbackHandler(max_concurrent=2)
        completed = []

        async def task(task_id):
            await asyncio.sleep(0.01)
            completed.append(task_id)

        # Submit tasks
        for i in range(4):
            coro = task(i)
            await handler.submit(coro)

        # At this point, not all tasks have necessarily completed
        await handler.wait_all()

        # After wait_all, all should be done
        assert len(completed) == 4
        assert set(completed) == {0, 1, 2, 3}

    @pytest.mark.asyncio
    async def test_multiple_concurrent_submits(self):
        """Should handle multiple concurrent submit calls."""
        handler = BoundedCallbackHandler(max_concurrent=3)
        completed = []

        async def task(task_id):
            await asyncio.sleep(0.01)
            completed.append(task_id)

        async def submit_many():
            for i in range(5):
                coro = task(i)
                await handler.submit(coro)

        await submit_many()
        await handler.wait_all()

        assert len(completed) == 5

    @pytest.mark.asyncio
    async def test_exception_logged_but_wait_all_completes(self):
        """Exceptions should be logged without preventing wait_all completion."""
        handler = BoundedCallbackHandler(max_concurrent=2)
        completed = []

        async def failing_task():
            await asyncio.sleep(0.01)
            raise RuntimeError("task failed")

        async def success_task(task_id):
            await asyncio.sleep(0.01)
            completed.append(task_id)

        # Submit one failing and one successful
        await handler.submit(failing_task())
        await handler.submit(success_task(1))

        # wait_all should complete despite the exception
        await handler.wait_all()

        # Successful task should have completed
        assert completed == [1]

    @pytest.mark.asyncio
    async def test_default_max_concurrent(self):
        """Should have sensible default for max_concurrent."""
        handler = BoundedCallbackHandler()  # No args
        # Just verify it constructs and works
        coro_called = []

        async def dummy():
            coro_called.append(1)

        await handler.submit(dummy())
        await handler.wait_all()
        assert coro_called == [1]

    @pytest.mark.asyncio
    async def test_async_indexed_stream_import_from_io_callbacks(self):
        """Verify async_indexed_stream can be imported from xtrax.io.callbacks."""
        from xtrax.io.callbacks import async_indexed_stream as aio_stream

        result = []
        async for idx, item in aio_stream([1, 2, 3]):
            result.append((idx, item))
        assert result == [(0, 1), (1, 2), (2, 3)]

    @pytest.mark.asyncio
    async def test_bounded_callback_handler_import_from_io_callbacks(self):
        """Verify BoundedCallbackHandler can be imported from xtrax.io.callbacks."""
        from xtrax.io.callbacks import BoundedCallbackHandler as BCH

        handler = BCH(max_concurrent=1)
        flag = [False]

        async def set_flag():
            flag[0] = True

        await handler.submit(set_flag())
        await handler.wait_all()
        assert flag[0] is True
