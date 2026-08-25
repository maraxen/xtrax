# Tier 1: Host Boundary Mechanics

Change surface: how host callbacks fire -- never the math. Measured exemplar:
`scripts/prof_stage1_host_boundary.py` (ordered vs unordered vs no sink under
Scan, cheapest possible int-append callback).

## First measured results (CPU, jax 0.10.2, 64-step scan)

- no boundary: 0.214 ms/pass
- unordered sink: ~310 ms/pass (~1450x none)
- ordered sink: ~222 ms/pass (~1040x none)

Readings:

1. ANY per-step host callback inside a Scan is enormously expensive relative
   to pure-device work -- even the unordered variant that JAX may reorder.
   The mechanism cost is the round trips themselves, not only ordering.
2. Ordered was FASTER than unordered here (~0.72x). Do not over-read this:
   both are host-bound; their ratio is scheduler-dependent. The stable,
   citable claim is the orders-of-magnitude gap versus the boundary-free
   scan, not the ordered/unordered ratio.
3. Dispatch counts confirm mechanism: each sink variant fired n_executions =
   steps-per-execution * executions (195 over 3 traced runs at 64 steps),
   i.e. one callback dispatch per step; the boundary-free scan had 3.

## Decision rules (verify-paths)

- Default: keep boundaries OFF the hot path. Aggregate device-side (Fuse)
  and emit once post-iteration instead of per-step Tap/Sink.
- `ordered=True` ONLY when correctness depends on host-observed order --
  executor.py's own guidance. It buys strict serialization: N steps = N
  serialized device->host->device round trips
  (`src/xtrax/stages/executor.py` module docstring).
- Vmap cannot lower an ordered io_callback at all; topology rejects the plan
  up front (`src/xtrax/stages/topology.py::validate_plan_topology`).
- Ordered SafeMap silently degrades to one element at a time regardless of
  configured batch_size (`tests/stages/test_executor.py::TestSafeMapOrderedIgnoresBatchSize`)
  -- do not expect batching to amortize ordering.
- Batching the payload (fewer, larger callbacks -- e.g. ZarrStagingSink-style
  keyed staging drained later, `src/xtrax/run/zarr_sink.py`) is THE Tier-1
  optimization: same data, fraction of the round trips.

## Evidence protocol

Correctness gate BEFORE timing (ordered must preserve step order; any variant
must observe every step exactly once), untraced steady-state walls for the
money numbers, short separate traces per variant for dispatch counts only.
The stage-1 driver implements all three; extend it rather than improvising.
