"""Public sink session: ordered pinned host writes, closed on traced failures (#2587)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xtrax.stages import SinkSession, sink_session
from xtrax.stages._callback import io_callback
from xtrax.stages.session import pinned_io_callback


class _LifecycleSink:
    """Callable host sink with explicit open, finalize, and close."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.finalized = False
        self.values: list[int] = []
        self.receipt = "receipt-2587"

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    def finalize(self) -> str:
        self.finalized = True
        return self.receipt

    def __call__(self, value: object) -> None:
        self.values.append(int(np.asarray(value)))


class _BareSink:
    """Callable sink with no lifecycle methods."""

    def __init__(self) -> None:
        self.values: list[int] = []

    def __call__(self, value: object) -> None:
        self.values.append(int(np.asarray(value)))


def test_jitted_pipeline_delivers_values_in_order() -> None:
    assert pinned_io_callback is io_callback
    sink = _LifecycleSink()
    xs = jnp.arange(8)

    with sink_session(sink) as session:
        assert isinstance(session, SinkSession)
        assert sink.opened
        assert not sink.closed

        @jax.jit
        def run(values: jax.Array) -> jax.Array:
            def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
                session.io_callback(value)
                return carry, value

            _carry, ys = jax.lax.scan(step, jnp.int32(0), values)
            return ys

        out = run(xs)
        jax.block_until_ready(out)
        assert sink.values == list(range(8))
        assert not sink.finalized

    assert sink.finalized
    assert sink.closed
    assert session.receipt == "receipt-2587"


def test_bare_sink_receives_jitted_values_in_order() -> None:
    sink = _BareSink()
    xs = jnp.arange(5)

    with sink_session(sink) as session:

        @jax.jit
        def run(values: jax.Array) -> jax.Array:
            def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
                session.io_callback(value)
                return carry, value

            return jax.lax.scan(step, jnp.int32(0), values)[1]

        out = run(xs)
        jax.block_until_ready(out)

    assert sink.values == list(range(5))
    assert session.receipt is None


def test_exception_inside_traced_region_closes_session() -> None:
    sink = _LifecycleSink()

    with pytest.raises(ValueError, match="trace boom"):
        with sink_session(sink) as session:

            @jax.jit
            def run(value: jax.Array) -> jax.Array:
                session.io_callback(value)
                msg = "trace boom"
                raise ValueError(msg)

            run(jnp.int32(3))

    assert sink.opened
    assert sink.closed
    assert not sink.finalized
    assert session.receipt is None


def test_host_callback_exception_closes_session() -> None:
    class _Boom(_LifecycleSink):
        def __call__(self, value: object) -> None:
            super().__call__(value)
            msg = "host boom"
            raise RuntimeError(msg)

    sink = _Boom()

    with pytest.raises(Exception, match="host boom"):
        with sink_session(sink) as session:

            @jax.jit
            def run(values: jax.Array) -> jax.Array:
                def step(carry: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
                    session.io_callback(value)
                    return carry, value

                return jax.lax.scan(step, jnp.int32(0), values)[1]

            jax.block_until_ready(run(jnp.arange(4)))

    assert sink.closed
    assert not sink.finalized
    assert session.receipt is None
    assert sink.values == [0]


def test_session_callback_is_an_ordered_effect() -> None:
    """Order comes from ordered=True, not from CPU scheduling luck."""
    sink = _BareSink()
    with sink_session(sink) as session:

        def step(carry: jax.Array, x: jax.Array) -> tuple[jax.Array, None]:
            session.io_callback(x)
            return carry, None

        jaxpr = jax.make_jaxpr(lambda xs: jax.lax.scan(step, 0, xs))(jnp.arange(3))

    def callback_eqns(jaxpr_like):
        for eqn in jaxpr_like.eqns:
            if eqn.primitive.name == "io_callback":
                yield eqn
            for value in eqn.params.values():
                inner = getattr(value, "jaxpr", value)
                if hasattr(inner, "eqns"):
                    yield from callback_eqns(inner)

    eqns = list(callback_eqns(jaxpr.jaxpr))
    assert len(eqns) == 1
    assert eqns[0].params["ordered"] is True
