# Closed-Loop GPU Kernel Optimization Playbook

This playbook operationalizes data-driven GPU kernel optimization across
`xtrax.profiling`, `xtrax.testing.kernel_parity`, and `xtrax.telemetry.profiling_events`:

1. **Observe & Algedonic Check**: Run `xtrax profile --telemetry-summary` before starting a cycle. Halt via `algedonic-escalation` if `instrument_degraded`, `noise_floor_blown`, `guardrail_erosion`, `parity_escape`, `heuristic_miscalibrated`, or `triage_gap` fires.
2. **Capture Dual-Pass Baseline**: Run `probe_kernel_dual_pass` on a clean commit (`sha_A`) to persist `prod` (`cmdbuf=on`, `diagnostic_mode="none"`) plus `.diag.json` (`cmdbuf=off`, normalized `1/n_steady` scopes, PTX/`ptxas`/Triton resource stats).
3. **Triage & Preregister**: Run `xtrax profile --triage prod.json --diag prod.diag.json` (`--precision-baseline` when comparing `single` vs `mixed`), note `shadow_subject_key`, and commit to a single Tier before editing.
4. **Implement & Parity-Gate**: Verify numerics with `xtrax.testing.kernel_parity.evaluate_kernel_parity_for_probe` (must include a corruption control that fails red and `work_active_count` preservation `>= 0.99`).
5. **Ratchet Compare & Shadow Label**: Measure clean commit (`sha_B`), run `xtrax profile --compare base.json cand.json --ratchet-pointer ptr.json --rejected-dir rejected --from-triage-probe prod.json --triage-category <cat>`. Keep **only** `KEEP` (`exit 0`); revert all `REVERT_*`, `REJECT_{PARITY,DISPATCH,VRAM}`, and `ADVISORY_ONLY` verdicts (`exit 1`), and fix `REJECT_UNVERIFIABLE` (`exit 2`).

---

## FP64 Pipe Saturation

Triggered when `mixed / single >= 1.35x` **AND** either running on a low-FP64 GPU (`L40S`, `L4`, `RTX`/`5090`/`RTX PRO 6000`) or compiled PTX reports residual `.f64` virtual registers (`ptx_f64_vregs_max > 0`).

- **Root Cause**: Workstations/inference GPUs throttle FP64 throughput to `1:64` vs FP32. Under `jax_enable_x64=True`, accidental float64 promotion in force kernels, 3D FFT charge grids, RNG draws, or pre-step trigonometry stalls the GPU pipe (`MK-41`, `MK-44`, `MK-58`).
- **Remediations**:
  1. Keep inner Pallas force arithmetic, transcendental approximations (`rsqrt`, `erfc`, B-splines), and 3D `rfftn`/`irfftn` FFTs strictly in `float32`/`complex64`; cast only state accumulators (`positions`, `momenta`) to `float64` at the integrator boundary (`MK-58`).
  2. Emit dual pre-cast `float32` coordinates directly from integrator registers (`return_pos_f32=True`) to eliminate stand-alone XLA `f64 <-> f32` cast kernels (`MK-58`).
  3. Generate thermal noise in `float32` (or in-register Philox-2x32 + Box-Muller, `MK-55`) rather than float64 Box-Muller (`MK-44`).
- **Verify**: Re-measure with `probe_kernel_dual_pass(capture_ptx=True)` and assert `ptx_f64_vregs_max` drops and `compute_precision_ratio(single, mixed)` closes toward `1.0x–1.2x` with `ParityContract` passing.

---

## Register Spill Pressure

Triggered when `ptxas_spill_bytes_max > 0`, `triton_spills_max > 0`, `ptx_lmem_bytes_max > 0`, `ptxas_regs_max >= 160`, or fallback `ptx_vregs_max >= 256`.

- **Root Cause**: Software-pipelined `num_stages >= 2` across large multi-branch kernels or unrolled inner loops exhausts the GPU's register file, spilling to per-thread local memory (`lmem` / L1/HBM traffic) or crashing with `CUDA_ERROR_ILLEGAL_ADDRESS` on `sm_120` (`MK-43`, `MK-46`, `MK-58`).
- **Remediations**:
  1. Pin `num_stages=1` and `num_warps=4` for multi-branch megakernels and complex integrator kernels (`resolve_pallas_compiler_params`, `MK-43`, `MK-46`).
  2. Convert Python-unrolled loops inside Pallas kernels into `jax.lax.fori_loop` (`scf.for`), avoiding live 1D vector carries across nested `jax.lax.cond` (`MK-36`, `MK-58`).
  3. Keep hot inner force-tile writebacks branchless inside the inner loop and split cold/rare paths outside the megakernel (`MK-43`).
- **Verify**: Confirm `ptxas_spill_bytes_max == 0`, `triton_spills_max == 0`, and `ptxas_regs_max` drops without losing `KEEP` speedup.

---

## Atomic Contention

Triggered when `ctas_per_atomic_target >= 64` or multi-replica per-replica scaling efficiency drops below `0.50` (`eta_rep = t_rep(B_lo) / t_rep(B_hi) < 0.50`).

- **Root Cause**: Unsorted atom tiles, high CTA contention onto shared output buffers, or `B`-replicated index tables saturating L2 atomic units and HBM bandwidth as replica count `B` increases (`MK-53`, `MK-56`, `MK-63`, `MK-64`).
- **Remediations**:
  1. Accumulate partial tile forces in Triton registers across the full K-tile / column stripe and issue ONE coalesced vector `plt.atomic_add` per tile instead of per-pair atomics (`MK-51`, `MK-56`).
  2. Use direct `_pl_store` (`atomic_add`-free) on disjoint output partitions such as pure-solvent CTAs (`nb_pid >= first_pure_water_block`, `MK-56`) or spatially compacted tiles (`MK-57`).
  3. Gate zero-charge/inactive lanes before atomic scatters (`active_q`, `MK-53`), hoist shared topology tables with `stride=0` across congeneric replicas (`MK-64`), and eliminate host-side buffer copies around `input_output_aliases` (`MK-52`, `MK-54`, `MK-55`).
- **Verify**: Re-run low-B vs high-B `batch_scaling_pair` probes and verify via `triage_probe_bundle(..., batch_scaling_pair=...)` that `eta_rep >= 0.50` alongside `work_active_count` preservation (CLI `--triage` currently accepts `--diag` and `--precision-baseline`).
