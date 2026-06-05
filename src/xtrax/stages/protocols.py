"""Generic tier-1 protocols for composable stages.

These are the base protocol definitions used to define domain-specific
stage collections (e.g., in domain libraries like aminx). No concrete
implementations live in xtrax.
"""

from typing import Protocol, TypeVar, runtime_checkable

In = TypeVar("In")
Out = TypeVar("Out")
Carry = TypeVar("Carry")
PerItem = TypeVar("PerItem")
Combined = TypeVar("Combined")


@runtime_checkable
class TransformFn(Protocol[In, Out]):
    """Protocol for a simple stateless transform function.

    Transforms an input of type In to output of type Out.
    """

    def __call__(self, x: In) -> Out: ...


@runtime_checkable
class RollingFn(Protocol[Carry, In, Out]):
    """Protocol for a rolling/stateful accumulation function.

    Takes a carry state and an input, returns updated carry and output.
    Similar to scan transitions.
    """

    def __call__(self, carry: Carry, x: In) -> tuple[Carry, Out]: ...


@runtime_checkable
class FuseFn(Protocol[PerItem, Combined]):
    """Protocol for a fusion/reduction function.

    Combines multiple per-item results into a single combined result.
    """

    def __call__(self, items: PerItem) -> Combined: ...
