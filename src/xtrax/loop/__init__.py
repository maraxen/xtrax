"""#2181 agentic-autoresearch loop (T2-01, #3069): the autonomous evolve-loop package.

The first #2181 code in this repo. Distinct from `xtrax.composition` (the #2174 D1-D6
composition substrate: HostPrepGraph, serialization, IR schema, validation, authoring) --
this package owns the loop's own concerns (admission, ratchet, provenance, evaluator
sealing consumption) as they land, per the same ownership-boundary precedent
`xtrax.inference` established for the IR schema emitter (fork-13: a new conceptual owner
gets its own top-level package rather than being folded into an existing one).
"""

__all__: list[str] = []
