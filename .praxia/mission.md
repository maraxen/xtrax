# Mission

## North Star

**Write pure JAX again.** Domain authors compose functions from primitives; xtrax supplies the polished runtime around them — optimized host-side I/O, managed JIT boundaries, streaming, and memory — without polluting the numerical core.

The core xtrax compiler (tiling, batching, stages, run layer, training, checkpointing) and the **composition surface** above it are separate concerns. The compiler stays small, testable, and domain-agnostic. Everything about *how work is prepared, wired, presented, and operated* lives in a composable layer on top.

## Composition surface

1. **Auto-inference bundle** — Inspect pure JAX functions (and `jax.export` / compilation artifacts where appropriate) to infer signatures, shapes, and stage boundaries; emit a runnable CLI and typed host adapters without hand-written glue.

2. **Host-side input prep** — Users annotate and compose *pre-JIT* functions for data loading, normalization, batch assembly, and plugin hooks. These compose independently of the traced core; boundaries are explicit and auditable.

3. **Chain-map interface** — A visual, node-graph composer (Ableton Max / Unreal Blueprint / Blender inspired) for wiring JAX pipelines: math rendered with MathJax, natural-language node labels, literature citations on transforms, agentic validation and audit hooks, script usage metadata, and durable plugin state (e.g. bathos experiment tracking) attached to nodes rather than buried in application code.

## 12-month objectives

- **Separation of concerns** — Document and enforce the compiler vs. composition boundary in code and public API; no host/UI concepts leak into `src/xtrax/` core modules.
- **Pure-function authoring path** — A domain author can ship a traced function plus optional host-prep annotations and get a working CLI and batch plan without subclassing xtrax internals.
- **Export-aware tooling** — Leverage JAX compilation primitives (`jax.export`, stable HLO boundaries, inspection) for auto-bundle generation while keeping xtrax's own planner independent of any single JAX release detail.
- **Composition MVP** — A chain-map prototype that serializes to the same host-prep graph the CLI consumes; MathJax + NL labels + citation slots on nodes; audit/validation agents can walk the graph.
- **Plugin state** — Bathos (and similar) integrate as node-attached state, not ad hoc globals in user scripts.
- **Ship the foundation** — Close the audit-framework and distribution epics so the composition layer builds on a stable, releasable xtrax 1.x core.

## Strategic themes

| Theme | Intent |
|-------|--------|
| Pure JAX core | Numerical work stays in traceable functions; Equinox modules only where static/dynamic split demands it. |
| Host orchestration | xtrax owns streaming, sharding, checkpointing, and boundary placement — not business logic. |
| Auto CLI / bundle | Inspection + export drive interface generation; annotations refine, not replace, inference. |
| Visual composition | Node graphs are the authoring UX; the compiler consumes a lowered graph, not UI widgets. |
| Agentic audit | Validation, citations, and usage metadata are first-class graph properties, not afterthoughts. |
| Plugin state | Experiment tracking and external tool state bind to composition nodes, keeping scripts thin. |

## Backlog anchors

| Epic | Scope |
|------|--------|
| **#2174** Pure-JAX composition layer | Host-side input prep, chain-map UI, auto CLI from inspected/exported functions |
| **#2175** Agent composition tooling *(research)* | Agent identities, skills, knowledge bases, and evolvable tooling for JAX/xtrax composition — requires research and decomposition before implementation |
