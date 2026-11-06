# Tier 3: Pallas Megakernels, Ragged Buckets, and Multi-GPU Optimization Playbook

When scaling multi-replica (`N_MOLS` / `B`) and ragged-bucket (`N_ATOMS`, `N_WATERS`) physics or simulation loops (e.g., `prolix` molecular dynamics) on GPU, individual operator optimizations saturate once HBM bandwidth, kernel launch overhead, and ghost-atom padding dominate. This playbook codifies the end-to-end engineering patterns (`MK-01` through `MK-45`) proven across Blackwell (`sm_120`), Hopper (`sm_90`), Ada (`sm_89`), and Ampere (`sm_80`).

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

## 3. Composing Processes into Fused Megakernels (`MK-36`–`MK-45`)

When `xtrax.profiling` shows that multiple small kernels or multi-pass reductions dominate step time, fuse them along the entity axis they share:

1. **Single-Launch Fused Bonded + Direct-Space Nonbonded NL Megakernel (`pallas_bonded_and_nonbonded_nl_fused`, `MK-36`, `MK-37`, `MK-39`, `MK-43`)**:
   - **Single 2D-Grid Launch `(B, total_blocks)`**: Fuses all 6 bonded CTA groups (`0 <= pid < total_bonded_blocks`: harmonic bonds, harmonic/Urey-Bradley angles, periodic dihedrals, harmonic impropers, 1-4 nonbonded exceptions, CMAP torsions) and direct-space nonbonded NL CTA blocks (`total_bonded_blocks <= pid < total_blocks`) into a single Pallas launch (`1.10x–1.21x` faster than separate launches on `H200` and `L40S`), accumulating forces into `(fx_out, fy_out, fz_out)` via `input_output_aliases={55: 1, 56: 2, 57: 3}` and writing per-atom nonbonded + per-block bonded energies into `e_out_ref` (upcast `.astype(orig_dtype)` before `jnp.sum` on host to prevent `float32` summation quantization across 25,000 atoms, `MK-39`).
   - **In-Condition Force Writeback (`MK-36`)**: Perform the force writeback inside `_run_k_tiles` so trailing ghost-atom blocks (`jnp.max(mask_i) == 0`) skip global memory writes completely and `jax.lax.cond(has_any_valid, _run_k_tiles, _skip_k_tiles, None)` yields `None` instead of carrying 128 vector registers across `scf.yield`.
   - **Never Use `jax.lax.while_loop` (`scf.while`) for Vector Accumulators in Triton (`MK-37`)**: Carrying 1D `(32,)` vector accumulators alongside scalar `i32` condition carries in `scf.while` crashes the C++ Triton compiler (`TritonGPURemoveLayoutConversionsPass` SIGSEGV). Always use `jax.lax.fori_loop(0, k_blocks, _k_tile_body, c0)` with an inner `jax.lax.cond(has_any_valid, _eval_nonempty_tile, lambda c: c, carry)` to skip empty sentinel tiles in $O(1)$.
   - **Branchless Writeback & `num_stages=1` in Multi-Branch Megakernels (`MK-43`)**: While standalone single-branch 2D kernels benefit from `num_stages=2` on Hopper/Ada, a 58-buffer, 7-branch megakernel with `num_stages=2` or a nested runtime `jax.lax.cond` around vector force stores inside `_run_k_tiles` spills registers to per-thread local memory on Blackwell (`sm_120`, causing a `5.5x` slowdown and `CUDA_ERROR_ILLEGAL_ADDRESS` stack overflow at `B >= 4`). Always use `num_warps=4, num_stages=1` for `kernel_kind="megakernel"` (and for all kernels on `sm_100`/`sm_120`) and keep `_run_k_tiles` writeback branchless at runtime (`if total_bonded_blocks == 0: _pl_store(...) else: _pl_atomic_add_1d(...)`).
2. **Fused Cell-List Pruning + 2-Bit Exclusion Encoding + Per-Replica Rebuild Masking (`pallas_prune_and_encode_cell_list`, `MK-47`)**:
   - Combines 27-cell stencil distance pruning ($r^2 < r_{\text{cut}}^2$) with binary search / range-gated lookup in the sorted dense exclusion table (`excl_indices`) and 2-bit topological scale encoding (`bits 31..30` of `nbr.idx`) in a single Pallas kernel during amortized neighbor-list rebuilds (`1.99x` faster on `H200` and `1.68x` faster on `L40S` with ghost-atom sentinel routing).
   - **Per-Replica Independent Rebuild Masking (`rebuild_mask: bool[B]`, `MK-47`)**: In multi-replica `vmap` batches (`B = 2..32`), pass `rebuild_mask: bool[B]` and `old_idx: int32[B, N, K]` into `_build_cell_stencil_buffers_batched` and `_pallas_prune_and_encode_cell_list_batch` with `input_output_aliases={15: 0}`. Replicas with `~rebuild_mask[b]` receive sentinel cell ID `G_cells` so their CTAs exit in $O(1)$ at the top of the Pallas kernel without touching global memory, preserving `old_idx[b]` in-place while only rebuilding replicas that crossed the Verlet skin threshold.
3. **Mixed-Precision `float32` / `complex64` SPME 3D FFT, Active-AD Gating, & Separable `(32, 16)` YZ-Plane B-Spline Factorization (`MK-38`, `MK-41`, `MK-49`)**:
   - **Active-AD `compute_dq` Gating (`MK-38`)**: Record `charges_perturbed = _is_perturbed(charges)` in `_spme_fwd` residuals and pass `compute_dq=bool(charges_perturbed)` to `gather_pallas` so standard position-only MD steps skip `dE/dq` potential accumulation and global memory stores (`1.05x–1.11x` faster PME gather).
   - **Eliminating the `float64` / `complex128` 3D FFT Trap (`MK-41`)**: Under `jax_enable_x64=True` (`mixed` precision mode: `float64` positions/velocities + `float32` forces), allowing `spread_charges_pallas` to cast `charge_grid` back to `float64` forces `jnp.fft.rfftn` / `irfftn` to execute double-precision (`float64`/`complex128`) 3D cuFFT plans on `(B, 64, 64, 64)` grids every step. Whenever `use_pallas_pme=True` and `_is_ortho_box(positions, box_size)` is `True` (covering both 1D `(3,)` and 2D batched `(B, 3)` orthorhombic boxes), keep `Q`, `G_cached`, `Q_hat`, and `theta_norm` strictly in `float32`/`complex64` and cast only the final scalar energy and `(N, 3)` forces back to `positions.dtype` (`1.28x` end-to-end DHFR speedup on `H200`).
   - **Separable `(BLOCK_N=32, 16)` YZ-Plane Factorized & Charge-Gated SPME (`MK-49`)**: Factorize the 64-point $(4\times 4\times 4)$ B-spline stencil in `_spread_kernel` and `_gather_kernel` into a precomputed `(32, 16)` YZ-plane outer product (`wyz = wy[:, :, None] * wz[:, None, :]`, `dwyz_dy`, `dwyz_dz`) outside the 4-iteration X-axis loop (`u = 0..3`), and gate global `atomic_add` writes in `_spread_kernel` on `(mask_n > 0) & (jnp.abs(q) > 0)` with early CTA exit when `jnp.max(active_i) == 0`.
4. **Fused 2D-Grid Pallas BAOAB + Analytical Miyamoto–Kollman / Horn Quaternion SETTLE + RATTLE Integrator Kernels (`prolix.physics.settle`, `MK-44`, `MK-45`, `MK-46`)**:
   - **Fused Pre-Force & Post-Force Pallas Integrators (`MK-46`)**: Fuse the ~18 per-step XLA scatter/gather/elementwise kernels in BAOAB + SETTLE into two register-resident 2D-grid `(B, cdiv(N_atoms, 64) + cdiv(N_waters, 32))` Pallas kernels (`pallas_baoab_pre_force` and `pallas_baoab_post_force`, **`2.26x` faster** than XLA BAOAB+SETTLE on `L40S`). Precompute the loop-invariant `(B, N_atoms)` water/ghost exclusion mask (`_iwg_precomputed`) once outside the `while_loop`, pin ghost atoms inside the Pallas kernels (`_pallas_settle_pins_ghosts = True`), and fall back automatically to pure-JAX/StableHLO SETTLE inside `export_context()`, under JVP tracers (`_is_ad`), or when `settle_velocity_iters == 0`.
   - **Triton `lax.neg_p` `TypedNdArray` Constant Rule (`MK-46`)**: In JAX Pallas Triton lowering (`_minus`), `lax.neg_p` inspects `x.type`, which raises `AttributeError: 'TypedNdArray' object has no attribute 'type'` whenever unary `-` is applied to a scalar constant `jnp.asarray(c, dtype=dtype)` inside a Pallas kernel. Always negate Python floats before `jnp.asarray(-c, dtype=dtype)` or write `zero - expr`.
   - **Vectorized `float32` Threefry Thermal Noise (`MK-44`)**: Under `jax_enable_x64=True`, sampling per-water `jax.random.normal` without `dtype=jnp.float32` executes $3 \times N_{\text{waters}}$ PRNG splits and $2 \times N_{\text{waters}}$ `float64` Box-Muller calls per step (`24,577` Threefry calls per replica!). Sample a single contiguous `z6_all = jax.random.normal(split_w, (n_w, 6), dtype=jnp.float32).astype(r_water.dtype)` tensor once per step and slice `z_trans = z6_w[:3]`, `z_rot = z6_w[3:]`.
   - **Uniform Bundle Precision (`MK-45`)**: Normalize all floating-point fields on `MolecularBundle` (`positions`, `masses`, `charges`, `sigmas`, `epsilons`, bonded params) to a uniform dtype (`float64` for `mixed`, `float32` for `single`) before integration, and cast `p_w.astype(r_o.dtype)` in `_maybe_rebuild_nl` so `lax.cond` branches never mismatch `reference_position.dtype`.

---

## 4. Runtime GPU Architecture Adaptation (`resolve_pallas_compiler_params`, `MK-35`, `MK-43`, `MK-46`)

Never hardcode `num_warps` or `num_stages` for a single GPU SKU. Use a capability-aware resolver (`prolix.export.resolve_pallas_compiler_params`) inspected at trace time via `jax.devices()[0]` (`device_kind` and `compute_capability`):

| Kernel Family / Architecture | Examples | Compute Capability | `CompilerParams` (`num_warps`, `num_stages`) | Rationale |
|---|---|---|---|---|
| **`megakernel`** & **`settle_integrator`** (All GPUs) | H200, L40S, B300, RTX PRO 6000, A100 | `sm_80`–`sm_120` | `num_warps=4`, `num_stages=1` | 58 buffer refs + 7 `pid` branches (`megakernel`) or dense 4x4 quaternion + 3x3 inertia algebra (`settle_integrator`); `num_stages=1` prevents register spill and per-thread stack overflow (`MK-43`, `MK-46`) |
| **`bonded_fused`** (All GPUs) | H200, L40S, B300, RTX PRO 6000, A100 | `sm_80`–`sm_120` | `num_warps=2`, `num_stages=1` | 1D elementwise gathers/atomics with low register footprint |
| **Blackwell** (`nonbonded_nl`, `cell_prune`, `pme`) | B200, B300, RTX PRO 6000 | `sm_100` / `sm_120` | `num_warps=4`, `num_stages=1` | Prevents `PTX 8.8` / `sm_120` software-pipeliner loop-pointer corruption at large 2D grids (`MK-43`/`MK-45`) |
| **Hopper / Ada / Ampere** (`nonbonded_nl`, `cell_prune`, `pme`) | H100, H200, L40S, L4, A100 | `sm_80` / `sm_89` / `sm_90` | `num_warps=4`, `num_stages=2` (`1` for small grids `<256` blocks) | `num_stages=1` on small grids maximizes multi-CTA SM occupancy; `num_stages=2` overlaps global loads on large grids |

Always expose environment overrides (`PROLIX_PALLAS_NUM_WARPS`, `PROLIX_PALLAS_NUM_STAGES`) and bucket-mapped PME FFT grid overrides (`PROLIX_PME_GRID_POINTS="8192:48,25000:64"`, `MK-42`) for targeted autotuning sweeps without code edits.

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
