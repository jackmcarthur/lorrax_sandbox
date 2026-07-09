# solvers/sternheimer_solve.py + solvers/sternheimer_precond.py

Group notes for the GW refactor map (2026-07-01). Repo root: `sources/lorrax_D`.

---

## src/solvers/sternheimer_solve.py (386 LOC)

### Purpose
The Sternheimer primitive: a single JIT-stable, level-shifted, preconditioned CG
solve `A_v · δu = −b` with `A = H_{k−q} − ε_{v,k} + α_pv·P_val`. Designed so ONE
compiled kernel is reused across all (k, q) — operator data flows as a pytree
(`SternheimerOp`) instead of fresh closures (which caused ~9× JIT retraces via
the generic `cg_posdef`). Wrapped in `jax.custom_jvp` so q-derivatives of δu
(→ χ_{G'0}) are obtained by implicit differentiation: one extra CG solve with
tangent RHS `ḃ − Ȧx`, no autodiff through iterations (Cancès et al. 2023 pattern).

Category: **physics: Sternheimer/DFPT linear solver primitive (chi0 via linear response)**.

### Function table
| Symbol | Role |
|---|---|
| `SternheimerOp` (pytree class, `__slots__`) | Bundle of operator arrays: `T_diag, V_scf, Gx/Gy/Gz, vnl_Z, vnl_E, mask, U_val, eps_v, alpha_pv, precond_diag`, optional Schur fields `U_extra/eps_extra`, static `fft_grid`. `tree_flatten/unflatten` preserve `None` Schur fields via aux flag. |
| `_apply_A_inline(op, x)` | Matvec `H_{k−q}x − ε_v x + α_pv P_val x`; uses `psp.dft_operators.apply_H_k_from_G` (G-sphere entry, avoids scatter/gather round-trip). Inlined for a single fused trace. |
| `_precond_inline(op, r)` | Elementwise multiply by `op.precond_diag` (TPA weights). |
| `_batched_dot`, `_batched_real_norm` | Per-band inner product / norm helpers (einsum `vsG,vsG->v`). |
| `cond_subspace_sos_solve(op, b)` | Public alias: sum-over-states closed-form solution restricted to `U_extra` Ritz subspace; used as SoS-only chi mode. Body is literally `return _schur_initial_guess(op, b)`. |
| `_schur_initial_guess(op, b)` | Explicit T-block solution `x = −Σ_m ⟨U_m|b⟩/(ε_m−ε_v)·U_m` over `U_extra`; CG warm-start when `use_schur=True`. Denominator clamp at 1e-8. |
| `_sternheimer_core(op, b, tol, max_iter, use_schur)` | `@jax.jit(static_argnames=('max_iter','use_schur'))`. Batched PCG with per-band convergence freeze (`alive` mask, mirrors QE `conv(ibnd)`), dead-band mask (‖b‖ < 1e-14 rows stay 0), `lax.while_loop` (replaced fixed `fori_loop`; ~50% iteration savings). Internally negates b: solves `A·δu = −b`. Optional `jax.debug.print` diagnostics via env `STERN_DEBUG`. |
| `sternheimer_solve(op, b, tol=1e-6, max_iter=100, use_schur=False)` | Public primitive, `@jax.custom_jvp(nondiff_argnums=(2,3,4))`; thin wrapper over `_sternheimer_core`. |
| `_sternheimer_solve_jvp` | JVP rule: primal solve, then `Ȧ·x` via `jax.jvp` of `_apply_A_inline` wrt op, then tangent solve on `rhs_tangent = −(ḃ − Ȧx)` (double-negation because the core negates its input). |

`__all__ = ["SternheimerOp", "sternheimer_solve"]` — note `cond_subspace_sos_solve` is used externally but NOT in `__all__`.

### Entry points / callers (grepped `sternheimer_solve|SternheimerOp|cond_subspace_sos_solve|_apply_A_inline` across src, tests, tools, scripts)
- `sternheimer_solve` <- `src/psp/run_sternheimer.py` (lines 78, 314, 363, 465, 553, 583, 618, 730, 779), `src/psp/tests/test_sternheimer_jvp.py:243`
- `SternheimerOp` <- `src/psp/run_sternheimer.py` (78, 315, 352, 571, 613, 653–654)
- `cond_subspace_sos_solve` <- `src/psp/run_sternheimer.py:776-777` (sos_only chi mode)
- `_apply_A_inline` (private!) <- imported by `src/psp/run_sternheimer.py:315, 415` — private-symbol cross-module import.
- No callers in tools/ or scripts/. `tests/test_sternheimer_solvers.py` covers the precond module but NOT this file's CG core directly (JVP test lives at `src/psp/tests/test_sternheimer_jvp.py`, an odd test location inside src/).

### Cross-module deps
- `psp.dft_operators.apply_H_k_from_G` (Hamiltonian matvec)
- Consumed by `psp.run_sternheimer` (the Sternheimer χ driver); doc references `solvers.cg_posdef` as the generic predecessor.

### I/O
None. Pure in-memory JAX compute. Only env var `STERN_DEBUG` (int-parsed) toggles `jax.debug.print`.

### Suspects
- **weird_code**
  - `_sternheimer_solve_jvp` lines 372–377: double-negation dance — passes `−(ḃ − Ȧx)` because the core internally negates. Correct per the comments, but a classic sign-convention trap; a refactor should make the core take the literal RHS.
  - `_schur_initial_guess` line 214: magic clamp `|denom| > 1e-8` for occupied/extra-Ritz crossings; silently zeroes those components.
  - `_sternheimer_core` line 262: magic dead-band threshold `1e-14` (absolute, dtype-independent — questionable for complex64/float32 path which the code otherwise supports via `dtype_r = float32`).
  - `SternheimerOp.fft_grid` is aux (static) while everything else is leaves; fine, but `tree_unflatten` rebuilds `None` from a boolean aux — subtle pytree trick worth flagging for refactor.
  - Docstring references "the other-agent gave in the Stage-3 writeup" (line 26) — agent-conversation residue in production docstring.
- **redundancy_suspects**
  - `cond_subspace_sos_solve` is a pure one-line alias of `_schur_initial_guess` — two names for one routine (the known "fetch_X vs fetch_X_dyn" pattern; keep one public name).
  - This module is explicitly a JIT-stable re-implementation of the `solvers/cg_posdef.py` closure-based CG (docstring says so); both PCG loops coexist. If `cg_posdef` has no other live consumers this is a parallel old/new path.
- **dead_suspects**: none — all public symbols have live callers in `run_sternheimer.py`.

---

## src/solvers/sternheimer_precond.py (139 LOC)

### Purpose
Teter–Payne–Allan (TPA) rational preconditioner for Sternheimer/DFPT CG solves
(PRB 40, 12255; QE `cgsolve_all.f90` analogue). Rescales by
`TPA(x)`, `x = T_G/K̄²_v`, with `T_G = |k−q+G|²` and per-band kinetic scale
`K̄²_v = ⟨ψ_{v,k}|T_k|ψ_{v,k}⟩` at the source k; smooth, sign-flip-free (unlike
the Davidson `1/(h_diag − ε)` in `psp/dft_precond.py`).

Category: **physics: DFPT preconditioner (numerics helper for Sternheimer solver)**.

### Function table
| Symbol | Role |
|---|---|
| `_tpa(x)` | The rational polynomial `(27+18x+12x²+8x³)/(same+16x⁴)`; jitted. |
| `compute_per_band_kinetic(U_val, T_diag)` | `K̄²_v = Σ_{s,G} T_G·|ψ_{v,s,G}|²`, jitted; (nv,) output. |
| `tpa_preconditioner_diag(T_diag_kminq, K_bar_sq)` | Returns the (nv, 1, nG) weight ARRAY — the form the JIT-stable `SternheimerOp.precond_diag` wants. Guards `K̄² ≤ 0 → 1.0`. |
| `make_tpa_preconditioner(T_diag_kminq, K_bar_sq)` | Factory returning a jitted CALLABLE closure doing the same elementwise multiply. |

`__all__ = ["compute_per_band_kinetic", "make_tpa_preconditioner", "tpa_preconditioner_diag"]`

### Entry points / callers (grepped across src, tests, tools, scripts)
- `compute_per_band_kinetic` <- `src/psp/run_sternheimer.py:76,1239`, `src/psp/tests/test_sternheimer_jvp.py:131`, `tests/test_sternheimer_solvers.py:132`
- `tpa_preconditioner_diag` <- `src/psp/run_sternheimer.py:76,979 (vmapped),1386`, `src/psp/tests/test_sternheimer_jvp.py:132`
- `make_tpa_preconditioner` <- ONLY `tests/test_sternheimer_solvers.py:143,230`; referenced in docstrings of `solvers/cg_posdef.py:100` and `solvers/minres.py:102`. Zero production callers.
- `_tpa` <- `tests/test_sternheimer_solvers.py:116,125` (private import in test).

### Cross-module deps
- None at import time (only jax). Semantically paired with `solvers/sternheimer_solve.py` (produces `precond_diag`) and contrasted with `psp/dft_precond.py`.

### I/O
None. Pure compute.

### Suspects
- **redundancy_suspects**: `make_tpa_preconditioner` (closure factory) vs `tpa_preconditioner_diag` (array form) — parallel old/new paths computing the identical weights; the array form's own docstring says "Equivalent to the factory below". Production (`run_sternheimer.py`) only uses the array form; the factory survives only in tests and in docstrings of the older closure-based solvers (`cg_posdef.py`, `minres.py`). Candidate for deletion together with whichever of those solvers is dead.
- **dead_suspects**: `make_tpa_preconditioner` — grep of src/tests/tools/scripts finds no production caller (tests + two docstring mentions only).
- **weird_code**: nothing alarming; magic constants (27/18/12/8/16, TPA(1)=65/81) are the literature-standard TPA polynomial and documented as such. `K_bar_sq > 0` clamp to 1.0 in both variants is a duplicated guard (line 93 and 123).

---

## Refactor-relevant summary for the pair
- Single production consumer: `psp/run_sternheimer.py`. These two files ARE the solver kernel of the Sternheimer χ pipeline (a separate stage from the ISDF/zeta GW main line in `gw/`).
- Consolidation targets: (1) drop `cond_subspace_sos_solve` alias or fold `_schur_initial_guess` into it; (2) delete `make_tpa_preconditioner` + retarget its tests to `tpa_preconditioner_diag`; (3) decide fate of legacy closure solvers `cg_posdef.py`/`minres.py` which this module superseded; (4) export `_apply_A_inline` properly since `run_sternheimer` imports the private name; (5) make the core take literal RHS to kill the JVP double negation.
