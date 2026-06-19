"""Async IO utilities — thin re-export from canonical engine.io implementation.

This module re-exports the canonical async I/O implementations from xtrax.engine.io.
Use this as the public import surface.

Example:
    Import and use the canonical callback handler:

    >>> from xtrax.io import BoundedCallbackHandler
    >>> import asyncio
    >>> async def test():
    ...     handler = BoundedCallbackHandler(max_concurrent=1)
    ...     executed = []
    ...     async def callback():
    ...         executed.append(True)
    ...     await handler.submit(callback())
    ...     await handler.wait_all()
    ...     return len(executed)
    >>> asyncio.run(test())
    1

    Stream blocking iterables without blocking the event loop:

    >>> from xtrax.io import async_indexed_stream
    >>> async def collect():
    ...     return [item async for item in async_indexed_stream(["a", "b"])]
    >>> asyncio.run(collect())
    [(0, 'a'), (1, 'b')]
"""

from xtrax.engine.io import BoundedCallbackHandler, async_indexed_stream

__all__ = ["async_indexed_stream", "BoundedCallbackHandler"]
