"""Async IO utilities — thin re-export from canonical engine.io implementation."""

from xtrax.engine.io import BoundedCallbackHandler, async_indexed_stream

__all__ = ["async_indexed_stream", "BoundedCallbackHandler"]
