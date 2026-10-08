"""Host sink session for ordered writes from traced code (#2587).

``sink_session`` opens a sink, yields a callback traced code can call, and
closes the sink on the way out. The callback is ``xtrax.stages._callback``'s
pinned ``io_callback`` with ``ordered=True``.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import jax
from jaxtyping import Shaped

from xtrax.stages._callback import io_callback as pinned_io_callback


class SinkSession:
    """Ordered host callback bound to one sink.

    Traced code calls :meth:`io_callback`. Each call runs ``sink(value)`` on
    the host, in program order. ``ordered=True`` cannot run under ``vmap``;
    call it from a scan or a sequential map.

    ``receipt`` is the return value of ``sink.finalize`` after a clean exit,
    or None when finalize did not run.
    """

    def __init__(self, sink: Callable[..., Any]) -> None:
        self._sink = sink
        self.receipt: Any = None

    def io_callback(self, value: Shaped[jax.Array, "..."]) -> None:
        """Hand ``value`` to the host sink, in program order."""
        sink = self._sink

        def _host(host_value: Any) -> None:
            sink(host_value)

        pinned_io_callback(_host, None, value, ordered=True)


def _call_if_present(obj: object, name: str) -> Any:
    """Call ``obj.name()`` when that attribute exists."""
    method = getattr(obj, name, None)
    if method is None:
        return None
    return method()


@contextmanager
def sink_session(sink: Callable[..., Any]) -> Iterator[SinkSession]:
    """Open ``sink``, yield an ordered callback, then finalize and close.

    Calls ``sink.open`` on entry when that method exists. On a clean exit,
    calls ``sink.finalize`` when it exists, stores the return value on
    :attr:`SinkSession.receipt`, then calls ``sink.close`` when it exists.
    An exception inside the block, including one raised while tracing, skips
    finalize and still calls close.

    ``sink`` is invoked as ``sink(value)`` once per :meth:`SinkSession.io_callback`.
    """
    _call_if_present(sink, "open")
    session = SinkSession(sink)
    try:
        yield session
        session.receipt = _call_if_present(sink, "finalize")
    finally:
        _call_if_present(sink, "close")


__all__ = ["SinkSession", "sink_session"]
