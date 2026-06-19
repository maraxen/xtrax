"""Per-axis boundary operations for composable pipeline stages.

Three distinct boundary op types — keep them separate:
  Fuse[S, O]  — pure axis reducer: stacked S -> single O. Stays in pipeline.
  Tap[T]      — identity + side effect: T -> T. Stays in pipeline.
  Sink[T]     — terminal side effect: T -> None. Leaves pipeline.

The Fuse/Tap/Sink distinction is the property the type checker enforces:
whether the value continues downstream or leaves the pipeline entirely.

AxisBoundary bundles optional fuse, tap, and sink for one named axis.
All fields are eqx.field(static=True) — no JAX arrays; all are callables.

Note: Fuse is distinct from FuseFn in xtrax.stages.protocols. Fuse is for
axis-level reduction after stacking; FuseFn is a more generic reduction protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import equinox as eqx

S = TypeVar("S")  # stacked input type (pre-fuse)
Out_co = TypeVar("Out_co")  # output type (post-fuse)
T = TypeVar("T")  # passthrough type (tap/sink)

BoundaryCallable = Callable[..., Any]


@runtime_checkable
class Fuse(Protocol, Generic[S, Out_co]):  # noqa: UP046
    """Pure axis-reducing transform. Stacked S -> single Out_co.

    Called once per axis completion, after all steps have run.
    Must be a pure JAX function — no side effects, no io_callback.
    Example: ArithmeticMeanEncodingFusion (stacked EncoderOutput → single EncoderOutput).

    Note: This is distinct from FuseFn in xtrax.stages.protocols, which is a more
    generic reduction protocol. Fuse is specifically for axis-level stacking operations.
    """

    def __call__(self, stacked: S) -> Out_co: ...


@runtime_checkable
class Tap(Protocol, Generic[T]):  # noqa: UP046
    """Identity transform with side effect. T -> T.

    Value continues downstream unchanged; side effect fires at each step.
    `ordered`: if True, requires SafeMap or Scan strategy on this axis —
    vmap does not preserve step order. Validator enforces this.
    Implementations must use io_callback internally.
    """

    ordered: bool

    def __call__(self, x: T) -> T: ...


@runtime_checkable
class Sink(Protocol, Generic[T]):  # noqa: UP046
    """Terminal side effect. T -> None. Value leaves the pipeline.

    `ordered`: if True, requires SafeMap or Scan strategy on this axis.
    Implementations must use io_callback(ordered=self.ordered) internally.
    Example: IoCallbackEncoderSink (writes encoded tensors to H5).
    """

    ordered: bool

    def __call__(self, x: T) -> None: ...


class AxisBoundary(eqx.Module):
    """Per-axis pipeline boundary: optional fuse, tap, and sink.

    All fields are static (eqx.field(static=True)) since they are callables,
    not JAX arrays. Default: all None (no-op — axis passes through to next axis).

    Topology rules enforced by make_inference_plan validator:
    - tap.ordered=True or sink.ordered=True + Vmap strategy → PlanTopologyError
    - fuse on a Scan axis: fuse receives the stacked ys after the full scan
    """

    fuse: Fuse | BoundaryCallable | None = eqx.field(static=True, default=None)
    tap: Tap | BoundaryCallable | None = eqx.field(static=True, default=None)
    sink: Sink | BoundaryCallable | None = eqx.field(static=True, default=None)


__all__ = ["AxisBoundary", "Fuse", "Sink", "Tap"]
