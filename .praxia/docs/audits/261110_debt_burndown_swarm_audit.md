# Swarm Code Review Audit Report — Task 261110_xtrax-debt-burn-down-and-consumer-adoption

**Date**: 2026-11-10T20:25:00Z  
**Task ID**: `261110_xtrax-debt-burn-down-and-consumer-adoption`  
**Consolidated Commit**: `2efcbfa7b9caf52be6bf4e97c8fcbd85b88c7782` on `feat/261110-xtrax-debt-burndown`  
**Participating Critics**:
1. Lens 1 (JAX & Mathematical Rigor): `critic_p5_jax_math` (`8e0a9035-3c31-4b56-bc99-3f27a735ba8c`) — `REQUEST_CHANGES`
2. Lens 2 (Cross-Repo Contracts & Systems): `critic_p5_contracts_systems` (`d2baa716-7ae6-4ac3-9a0c-2c4059453e13`) — `APPROVE`
3. Lens 3 (CI Hygiene, Packaging & Tooling): `critic_p5_ci_hygiene` (`78a0b1f1-0f4f-415e-ab54-194bbfcfd825`) — `REQUEST_CHANGES`

---

## 1. Executive Summary

An adversarial 3-lens swarm review was conducted across the consolidated implementations resolving the 19 technical debts in `xtrax` and its consumer seams in `aminx` and `demistify`.

- **Lens 2 (Contracts & Systems)**: **APPROVED**. Lease fencing attempt staging, Zarr commit chunking/dual-quarantine, consumer adoption in `aminx` and `demistify`, and workspace git reset safety guards passed all 365 tests with zero blocking issues.
- **Lens 1 (JAX & Mathematical Rigor)**: **REQUEST_CHANGES**. Identified 2 Blockers, 1 Critical, and 3 Major issues in autodiff and op naming.
- **Lens 3 (CI Hygiene & Packaging)**: **REQUEST_CHANGES**. Identified 1 Blocker, 1 Critical, and 1 Major issue in dependency isolation and snapshot invocations.

---

## 2. Master Findings & Remediation Plan

| ID | Lens | Severity | Subsystem / File | Issue Summary | Remediation Required | Status |
|---|---|---|---|---|---|---|
| **L1-B1** | JAX/Math | BLOCKER | `src/xtrax/kernel_seam.py:581` | Higher-order reverse/forward autodiff broken because `fn_fwd` bypassed `fwd_pair` and invoked `vk.fwd` directly. | Restore `fwd_pair(*unwrapped_args)` in `fn_fwd`. Verify `tests/op/test_op.py` passes. | RESOLVED (Restored `fwd_pair(*unwrapped_args)`; 41/41 `test_op.py` tests pass) |
| **L1-B2** | JAX/Math | BLOCKER | `src/xtrax/op/__init__.py` | Package directory `src/xtrax/op/` shadows top-level callable `xtrax.op(...)`. | Relocate `make_2d_grid_specs` to `src/xtrax/tiling/grid.py`, remove directory `src/xtrax/op/`, update imports and exports. | RESOLVED (Relocated to `src/xtrax/tiling/grid.py`, deleted `src/xtrax/op/`, verified `xtrax.op is op`) |
| **L1-C1** | JAX/Math | CRITICAL | `src/xtrax/kernel_seam.py:577` | Shallow tuple unwrapping of `CustomVJPPrimal` leaves nested PyTree array leaves wrapped, causing arithmetic TypeError. | Use `jax.tree_util.tree_map` with `is_leaf=lambda x: isinstance(x, CustomVJPPrimal)` to recursively unwrap. | RESOLVED (Recursive PyTree unwrapping in `fn_fwd` and `fn_bwd`) |
| **L1-M1** | JAX/Math | MAJOR | `src/xtrax/kernel_seam.py:588` | All-SymbolicZero VJP backward return does not handle PyTree arguments, returning `None` instead of PyTree of zeros. | Map `_to_symbolic_zero` over PyTrees via `jax.tree_util.tree_map`. | RESOLVED (Mapped `_to_symbolic_zero` over PyTrees; verified in `test_vjp_symbolic_zeros.py`) |
| **L1-M2** | JAX/Math | MAJOR | `src/xtrax/kernel_seam.py:624` | `Op.fused_primal_and_grad` ignores caller-supplied `cotangent` on kernel path. | Support/validate `cotangent` argument in kernel fused path. | RESOLVED (Forwarded `cotangent` to `vk.fused_primal_and_grad` with fallback; tested) |
| **L1-M3** | JAX/Math | MAJOR | `src/xtrax/tiling/carry.py:97` | `validate_carry_transition` skips non-array leaves, letting scalar type transitions bypass validation. | Check array status parity (`hasattr(in_leaf, "shape") == hasattr(out_leaf, "shape")`) and raise `CarryCanonicalizationError`. | RESOLVED (Added array status parity check; verified in `test_while_canonicalization.py`) |
| **L3-F1** | CI/Hygiene | BLOCKER | `tests/scripts/test_check_dependency_floors.py` | `test_repo_floors_parse` failed because `optax` and `orbax-checkpoint` moved to optional `[training]`. | Update floor test and `.github/workflows/ci.yml` dependency-floors job to handle training optional extra. | RESOLVED (Updated test and `.github/workflows/ci.yml` for `training` extra; 7/7 tests pass) |
| **L3-F2** | CI/Hygiene | CRITICAL | `src/xtrax/training/optim.py`, `checkpoint/orbax.py` | Missing `from __future__ import annotations` causes `AttributeError` on import when optional dependencies are absent. | Add future annotations, call `require_optax()`/`require_orbax()` in functions, guard `state.py`, `step.py`, `trainer.py`. | RESOLVED (Added future annotations, import guards, and `test_import_without_optax.py`) |
| **L3-F3** | CI/Hygiene | MAJOR | `Justfile:383` | Missing `--extra plugin` in `uv run python scripts/generate_plugin_snapshot.py` fails in unsynced environments. | Add `--extra plugin` to `Justfile:383`. | RESOLVED (Added `--extra plugin` to `Justfile:383`; `just plugin-snapshot` exits 0) |
| **L3-F4** | CI/Hygiene | MINOR | `scripts/assemble_changelog.py:270` | Hardcoded `changelog.d/` ignores `fragments_dir`. | Use `fragments_dir`. | RESOLVED (Used relative `fragments_dir`; 10/10 changelog gate tests pass) |
| **L3-F5** | CI/Hygiene | MINOR | `src/xtrax/eda/render_core.py:88` | Docstring states `bundle_js=True` raises NotImplementedError. | Update docstring to reflect working configurable implementation. | RESOLVED (Updated docstring for `bundle_js`; verified in `test_bundle_js.py`) |

---

## 3. Remediation Verification & Sign-Off

**Remediation Commit**: `579bdd4d021187c728ba6cb80996991a079cb6ac` on branch `feat/261110-xtrax-debt-burndown`  
**Date**: 2026-11-10T20:50:00Z  
**Worker**: `worker_p5_audit_remediation`

All 10 findings resolved and independently verified:
- **Test Verification Suite**: 812 tests passed across `tests/op/`, `tests/tiling/`, `tests/scripts/test_check_dependency_floors.py`, `tests/training/`, and `tests/eda/` with zero failures.
- **Autodiff & Higher-Order Math**: Verified 41/41 passing tests in `tests/op/test_op.py` (including second-order JVP over reverse VJP), plus symbolic-zero and primal+grad fusion tests in `tests/op/test_vjp_symbolic_zeros.py`.
- **Namespace Integrity**: Removed `src/xtrax/op/`, relocated `make_2d_grid_specs` to `src/xtrax/tiling/grid.py`, and confirmed `assert xtrax.op is op` succeeds.
- **Dependency Floor Testing**: Verified floor check in `tests/scripts/test_check_dependency_floors.py` (7/7 passed) and updated CI workflow `.github/workflows/ci.yml:132,135` for `[training]` extra.
- **Optional Dependency Guards**: Verified clean imports with absent `optax`/`orbax` and actionable errors in `tests/training/test_import_without_optax.py`.
- **Static Gates**:
  - `uv run ruff check .` passed with 0 errors.
  - `uv run ruff format --check .` passed across 4081 files.
  - `uv run ty check` passed cleanly on all modified modules.
  - `just plugin-snapshot` exited 0 and updated `src/xtrax/agent_plugin.json`.
- **Final Audit Status**: 100% findings RESOLVED. Ready for Sentinel Victory Auditor sign-off.
