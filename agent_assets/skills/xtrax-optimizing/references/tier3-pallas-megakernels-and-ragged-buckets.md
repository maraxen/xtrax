# Tier 3: Pallas Megakernels, Ragged Buckets, and Multi-GPU Optimization Playbook

When scaling multi-replica (`N_MOLS` / `B`) and ragged-bucket (`N_ATOMS`, `N_WATERS`) physics or simulation loops (e.g., `prolix` molecular dynamics) on GPU, individual operator optimizations saturate once HBM bandwidth, kernel launch overhead, and ghost-atom padding dominate. This playbook codifies the end-to-end engineering patterns (`MK-01` through `MK-35`) proven across Blackwell (`sm_120`), Hopper (`sm_90`), Ada (`sm_89`), and Ampere (`sm_80`).

---

## 1. Handling Ragged Axes (`N_ATOMS`, `N_WATERS`) and Ghost-Atom Compaction

Size bucketing (`ATOM_BUCKETS`, `WATER_BUCKETS`) prevents $O(K)$ XLA recompilations across heterogeneous systems by padding arrays to fixed bucket sizes (e.g., 6,658-atom solvated ubiquitin in an 8,192 or 25,000 bucket). However, naive padding introduces three severe performance and correctness hazards if ghost atoms (`~atom_mask`) participate in spatial data structures or tile loops:

### 1.1 Sentinel Cell Routing (`c_i = G_cells`) in 3D Cell-List Builders
- **Hazard**: Placing ghost atoms on a uniform grid inside the periodic box populates 3D spatial cells with thousands of non-interacting ghost particles. In a 6,658-atom system padded to a 25,000-atom bucket, un-filtered cell lists produce **1,899,431 ghost entries** inside real atoms' neighbor lists, wasting >50% of cell-list pruning and nonbonded tile evaluations.
- **Pattern (`prolix.physics.neighbor_list._build_cell_stencil_buffers_batched`)**:
  1. Route every ghost atom (`~atom_mask`) to an out-of-bounds sentinel cell index `c_i = G_cells` (`G_cells = Nx * Ny * Nz`).
  2. Size the host 27-neighbor stencil LUT (`_build_stencil_cell_lut_np`) to `((G_cells + 1) * 27,)` so sentinel cell `G_cells` is always in-bounds for gather/load operations.
  3. Zero out the sentinel cell's occupancy (`cell_counts = cell_counts.at[:, G_cells].set(0)`) so no real atom's 27-cell stencil ever scans ghost atoms.

### 1.2 Valid-Neighbor Compaction for Row-Level Early Exit
- **Hazard**: Even when ghost interactions are masked to zero energy/force, scattered sentinel indices (`idx == N`) across a neighbor-list row prevent CTA-level early-exit (`row_max_k`) from halting early.
- **Pattern (`prolix.physics.neighbor_list.encode_nl_exclusions`)**:
  1. Replace any neighbor entry where `~atom_mask[i]` or `~atom_mask[j]` with sentinel `N`.
  2. Compact all valid (`< N`) neighbors to the front of each row via `order = jnp.argsort((out == N).astype(jnp.int32), axis=-1)` and `out = jnp.take_along_axis(out, order, axis=-1)`.
  3. Precompute `row_max_k` (first column where all rows in a `BLOCK_N` tile are sentinel `N`); `pallas_nonbonded_nl` halts its inner `lax.fori_loop(0, num_k_tiles)` immediately after the last real neighbor tile (**1.94x–2.24x cell-prune speedup** and **1.19x–1.34x force-eval speedup** on ragged buckets, matching compact-bucket latency within 1%).

### 1.3 CTA-Level O(1) Empty-Tile Early Exit & the Triton `scf.yield` MLIR `Value` Rule
- **Pattern**: In 1D/2D Pallas kernels over padded atom tiles (`_spread_kernel`, `_gather_kernel`, `_make_prune_and_encode_kernel`) or padded bonded term tiles (`_bonded_all_fused_kernel`), wrap each tile body in `jax.lax.cond(has_any, _body, _empty)`.
- **Critical Triton Lowering Rule (`MK-32`)**:
  - Inside a Pallas Triton kernel, returning a Python scalar (`0`, `0.0`) or a literal JAX array (`jnp.int32(0)`, `jnp.zeros_like(x)`) from a `jax.lax.cond` branch raises:
    `ValueError: Operand 0 of operation "scf.yield" must be a Value (was TypedNdArray)`.
  - **Fix**: Always derive the zero carry operand from an in-kernel traced MLIR `ir.Value` via algebraic cancellation before `lax.cond` and pass it as the carry argument:
    ```python
    # Scalar int32 carry (e.g., hit counter in cell-list prune):
    zero_hits = c_i - c_i
    total_hits = jax.lax.cond(c_i < np.int32(G_cells), _scan_27_cells, lambda z: z, zero_hits)

    # Vector float32 carry (e.g., force accumulator in PME gather):
    zero_vec = (offs - offs).astype(dtype)
    fx, fy, fz = jax.lax.cond(
        jnp.max(is_real.astype(jnp.int32)) > 0,
        _do_gather,
        lambda z: (z, z, z),
        zero_vec,
    )
    ```

---

## 2. Heterogeneous Multi-System & Multi-Bucket Batches (`N_MOLS` / `B`)

### 2.1 Native 2D Pallas Grid `(B, grid_1d)` & Stride-0 Shared Topology
- **Avoid Serial `jax.lax.map` Loops in `custom_vmap`**: Implementing `@jax.custom_batching.custom_vmap` via `jax.lax.map` launches $B$ sequential GPU kernels per step. Instead, promote the Pallas grid to 2D `(B, grid_1d)` (`b = pl.program_id(0)`, `pid = pl.program_id(1)`) so all $B$ replicas/systems execute concurrently in a single kernel launch.
- **Stride-0 Shared Topology vs Per-Replica Parameters**: When `bdim` is `False` (shared across replicas, e.g., topology indices in homogeneous ensembles), pass the unbatched `(N, ...)` buffer with `pl.BlockSpec((None, ...), lambda b, i: (0, ...))` (stride-0 broadcast). When `bdim` is `True` (e.g., positions, or per-mutant charges/sigmas in WT+mutant batches), broadcast along `B` and index via `lambda b, i: (b, ...)`.

### 2.2 The `BatchTracer` Host-Fallback Box-Size Trap Under `jax.vmap` (`MK-31`)
- **Hazard**: Energy and neighbor-list factories often extract static box dimensions on the host via a helper like `_host_box_size(bundle, default=(30.0, 30.0, 30.0))` that catches `TypeError` (`TracerArrayConversionError`) when `np.asarray(bundle.box)` is called. Under `jax.vmap` (`B >= 2`), `bundle.box` is a `BatchTracer`, so `_host_box_size` silently falls back to the `(30.0, 30.0, 30.0)` default! On a 61.645 A solvated DHFR box, Pallas kernels (which close over static `sys.box_size`) wrap coordinates across a 30 A boundary and produce `NaN` on Step 0.
- **Rule**: Never allow a `TracerArrayConversionError` catch block to silently substitute a physical constant (like box size or cutoff) that differs from the actual system. Extract concrete `host_box_vec` (`np.diag(np.asarray(bundles[0].box))`) on the host *before* entering `@jax.jit` / `jax.vmap`, pass `host_box_vec` into the integrator setup, and replace `bundle.box` on the energy-construction bundle with a concrete `np.ndarray`.

### 2.3 Densest-Bundle + Union-Mask Shared `NeighborList` Seed Sizing (`MK-32`)
- **Hazard**: In a heterogeneous same-bucket batch (e.g., `[sys_4200, ubq_6658]` in an `8,192` bucket), allocating the shared `NeighborList` seed from `bundles[0]` under-sizes `max_occupancy` and `cell_list_capacity` for the denser `bundles[1]`. Conversely, calling `neighbor_fn.allocate` separately per bundle creates incompatible `cell_list_capacity` shapes that cannot be stacked via `jax.tree.map(jnp.stack, ...)`.
- **Rule (`EnsemblePlan._build_stacked_neighbor_seed`)**:
  1. Compute the union `atom_mask` (`jnp.any(jnp.stack([b.atom_mask for b in bundles]), axis=0)`) and identify `densest_bundle` (`max(bundles, key=lambda b: int(jnp.sum(b.atom_mask)))`).
  2. Call `neighbor_fn.allocate(densest_positions)` **exactly once** with `atom_mask=union_mask`.
  3. Pad `nbr.idx` columns in $O(1)$ (`jnp.concatenate([nbr.idx, pad_cols], axis=-1)`) up to `min(target_capacity, n_rows - 1, 27 * cell_list_capacity)`.
  4. For every other bundle in the stack, populate its initial neighbor list via `seed_invalidated.update(b_pos)` (where `reference_position = b_pos + 2 * cutoff`) so all replicas share identical PyTree structure and `cell_list_capacity`.

### 2.4 Caching Stacked Multi-Replica Executables with Dynamic `n_steps` (`MK-34`)
- Pass `n_steps` as a dynamic `jnp.int32` scalar (`jnp.asarray(n_steps, dtype=jnp.int32)`) into `@jax.jit def _batched(sb, sd, snb, n_steps_in)` and `jax.lax.while_loop`, and cache `(stacked, stacked_neighbor, neighbor_fn, _batched)` in `_STACKED_INFERENCE_CACHE`. Warm `EnsemblePlan.run()` calls with different `n_steps` or `seed` execute in **zero recompiles** and zero host NL allocations.

---

## 3. Composing Processes into Fused Megakernels

When `xtrax.profiling` shows that multiple small kernels or multi-pass reductions dominate step time, fuse them along the entity axis they share:

1. **Fused Bonded Megakernel (`pallas_bonded_all_fused`)**:
   - Replaces 6 separate XLA energy/force terms (bonds, harmonic/Urey-Bradley angles, periodic dihedrals, harmonic impropers, 1-4 nonbonded exceptions, and CMAP torsions) with a single 2D Pallas launch `(B, max_tiles)` that accumulates directly into `(fx, fy, fz)` via `pl.atomic_add` and `input_output_aliases` (aliasing distinct zero-initialized input buffers `zero_fx, zero_fy, zero_fz` to outputs `1, 2, 3`).
2. **Fused Cell-List Pruning + 2-Bit Exclusion Encoding (`pallas_prune_and_encode_cell_list`)**:
   - Combines 27-cell stencil distance pruning ($r^2 < r_{\text{cut}}^2$) with binary search in the sorted dense exclusion table (`excl_indices`) and 2-bit topological scale encoding (`bits 31..30` of `nbr.idx`) in a single Pallas kernel during amortized neighbor-list rebuilds.
3. **Single-Pass SETTLE Water Constraint Solver (`prolix.physics.settle`)**:
   - Fuses the duplicate water center-of-mass and regularized $3\times 3$ inertia-tensor reductions in `_langevin_step_o_constrained` into a single pass per water molecule, and passes `dv_tuple=(dv_O, dv_H1, dv_H2)` directly into `_remove_angular_momentum_from_impulse`, eliminating a full `(N, 3)` scatter + gather round-trip per step.

---

## 4. Runtime GPU Architecture Adaptation (`resolve_pallas_compiler_params`)

Never hardcode `num_warps` or `num_stages` for a single GPU SKU. Use a capability-aware resolver (`prolix.pallas_kernels.resolve_pallas_compiler_params`) inspected at trace time via `jax.devices()[0]` (`device_kind` and `compute_capability`):

| Architecture | Examples | Compute Capability | Nonbonded NL (`BLOCK_N=64, BLOCK_K=64`) | Rationale |
|---|---|---|---|---|
| **Blackwell** | B200, B300, RTX PRO 6000 | `sm_100` / `sm_120` | `num_warps=4`, `num_stages=3` (`2` for small grids `<256` blocks) | Deeper tensor/shared-memory pipelines benefit from 3 stages once grid occupancy saturates SMs |
| **Hopper** | H100, H200 | `sm_90` | `num_warps=4`, `num_stages=3` (`2` for small grids `<256` blocks) | High SM count (132 SMs); stage=2 on small grids preserves multi-CTA occupancy per SM |
| **Ada Lovelace** | L40S, L4 | `sm_89` | `num_warps=4`, `num_stages=2` | Tighter per-SM shared memory capacity; `num_stages=2` prevents shared-memory spilling |
| **Ampere** | A100 | `sm_80` | `num_warps=4`, `num_stages=2` | 108 SMs, 164 KB shared memory/SM; `num_stages=2` balances register pressure and occupancy |

Always expose environment overrides (`PROLIX_PALLAS_NUM_WARPS`, `PROLIX_PALLAS_NUM_STAGES`) for targeted autotuning sweeps without code edits.

---

## 5. Leveraging `xtrax.profiling` & Rigorous Ground-Truth A/B Testing

### 5.1 Pinpointing Bottlenecks with `xtrax.profiling`
1. **Structural Jaxpr Audit (Stage 0)**: Use `xtrax.profiling.loop_scaling.loop_bodies` and `iter_jaxpr_eqns` on the compiled `while_loop` / `scan` body to verify that all intended Pallas kernels (`pallas_call`) are present in the loop body and no accidental host callbacks (`io_callback`) or un-fused $O(N^2)$ XLA broadcast subgraphs remain.
2. **Compile-Count Guard**: Wrap warmup + warm production runs in `xtrax.profiling.count_backend_compiles()` to prove that dynamic `n_steps` and `_STACKED_INFERENCE_CACHE` trigger **0 backend compiles** on warm invocations.
3. **Sub-Step Decomposition via `xp.probe_unit`**: Probe each sub-step (`bonded_pallas`, `nonbonded_nl_pallas`, `pme_spme_pallas`, `settle_positions`, `settle_velocities`, `cell_prune_pallas`, and full `EnsemblePlan.run`) with `xp.probe_unit(fn, args, n_steady=3, n_wall=7)` on the exact same node and seed before and after every Tier-3 edit.

### 5.2 Ground-Truth A/B Parity Battery (Mandatory Before Keeping Any Kernel Edit)
Every Pallas kernel or megakernel change must pass four orthogonal accuracy gates:
1. **XLA Reference & OpenMM CUDA Single-Point Parity**:
   - Compare total potential energy ($|\Delta E| < 0.02\text{ kcal/mol}$ on 6,658–23,558 atoms) and per-atom forces ($\max |\Delta F| < 5\times 10^{-4}\text{ kcal/(mol}\cdot\text{\AA)}$, cosine similarity $> 0.999999$) against the pure-XLA reference and OpenMM CUDA.
2. **Ragged-Bucket Padding Invariance**:
   - Run the exact same 6,658-atom solvated system inside an `8,192` bucket and a `25,000` bucket. Assert $|E_{8\text{k}} - E_{25\text{k}}| = 0.0\text{ kcal/mol}$ and $\max |F_{8\text{k}} - F_{25\text{k}}| < 5\times 10^{-5}\text{ kcal/(mol}\cdot\text{\AA)}$ on real atoms `[:6658]`, and `ghost_neighbors_in_real_rows == 0`.
3. **Multi-Replica Batch Invariance (`B=1` vs `B>1`)**:
   - Assert that running a bundle inside a heterogeneous or homogeneous stacked `EnsemblePlan` batch (`B >= 2`) produces identical single-step energies/forces and statistically indistinguishable finite trajectories compared to running each bundle independently (`B = 1`).
4. **Analytical `custom_vjp` vs Finite Differences**:
   - Verify gradients w.r.t. coordinates (`positions`) and force-field parameters (`charges`, `sigmas`, `epsilons`) against central finite differences on CPU (`interpret=True`) and GPU.
