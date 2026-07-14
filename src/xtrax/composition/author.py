"""D6 generate-then-validate authoring front-end (T1-12, #3067).

The default authoring path free-generates a candidate `HostPrepGraph` and hands it to
`xtrax.composition.validate.validate_graph` (T1-10's gate) unchanged -- this module owns only
generation, not validation.

`Generator` is the pluggable-backend seam: any future generator (e.g. a real LLM call) can
implement `generate(seed, num_nodes) -> HostPrepGraph` without touching the CLI verb
(`xtrax.cli.graph_author_verb`) or the validation wiring. Only `TemplateGenerator` -- a
deterministic, dependency-free generator with no LLM call -- ships in this item. It is
deliberately the "lightest weight version": the AC's success metric is "authors >=1 graph
passing graph validate," not "LLM-authored," and a deterministic generator is a better fixture
for T1-13's bench harness (repeatable, non-flaky) than a real LLM call would be.

`callable_ref` must resolve to a real, importable top-level module attribute
(`xtrax.composition.serialize._callable_to_import_path` requires `__module__`/`__qualname__`
with no dots or closures, and re-importing must resolve back to the same object) -- so "free
generation" cannot invent arbitrary callables from nothing. `_EXAMPLE_CALLABLES` below is a
small, deliberately-labeled pool of toy top-level functions for `TemplateGenerator` to sample
from; they are placeholder authoring content, not production stages, and are kept trivial
(no jax usage, no branching) specifically so they stay jaxlint-clean -- `validate_graph`
attributes any JL-error-severity jaxlint finding in a callable's own source lines to that
node, which would make a generated candidate fail `graph validate` even though the AC only
asks for *a* graph that passes, not a semantically meaningful one.
"""

from collections.abc import Callable
from random import Random
from typing import Any, Protocol

from xtrax.composition.graph import GraphEdge, HostPrepGraph, HostPrepGraphNode


def example_increment(x: Any) -> Any:
    return x + 1


def example_double(x: Any) -> Any:
    return x * 2


def example_square(x: Any) -> Any:
    return x * x


def example_negate(x: Any) -> Any:
    return -x


_EXAMPLE_CALLABLES: tuple[Callable[..., Any], ...] = (
    example_increment,
    example_double,
    example_square,
    example_negate,
)


class Generator(Protocol):
    """The pluggable-backend seam for T1-12's default authoring path."""

    def generate(self, *, seed: int, num_nodes: int) -> HostPrepGraph: ...


class TemplateGenerator:
    """Deterministic free-generator: samples from `_EXAMPLE_CALLABLES`, wires a linear chain.

    The only `Generator` shipped in this item -- see module docstring.
    """

    def generate(self, *, seed: int, num_nodes: int) -> HostPrepGraph:
        if num_nodes < 1:
            msg = f"num_nodes must be >= 1, got {num_nodes}"
            raise ValueError(msg)

        rng = Random(seed)
        nodes = []
        for i in range(num_nodes):
            callable_ref = rng.choice(_EXAMPLE_CALLABLES)
            nodes.append(
                HostPrepGraphNode(
                    id=f"n{i}",
                    callable_ref=callable_ref,
                    metadata={
                        "nl_description": (
                            "auto-authored candidate node wrapping "
                            f"{getattr(callable_ref, '__module__', '?')}:"
                            f"{getattr(callable_ref, '__qualname__', '?')}"
                        ),
                    },
                )
            )

        edges = tuple(GraphEdge(src=f"n{i}", dst=f"n{i + 1}") for i in range(num_nodes - 1))
        return HostPrepGraph(nodes=tuple(nodes), edges=edges)


__all__ = ["Generator", "TemplateGenerator"]
