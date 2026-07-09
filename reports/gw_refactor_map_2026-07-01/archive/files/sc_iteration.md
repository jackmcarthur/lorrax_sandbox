# src/gw/sc_iteration.py — deep-read notes (2026-07-01)

Repo: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`
LOC: 756. Category: **physics: QSGW self-consistency driver** (iteration map + fixed-point solvers + post-SC artifact writers).

## Purpose

Implements the self-consistent QSGW loop as a pure `state → state` iteration map
(`gw_iteration_map`) plus a Python-loop driver (`run_self_consistency`) with two
fixed-point accelerators (linear α-mixing, rCROP/Anderson). The carry is a single
tensor `H_qp_dft` — the static QP Hamiltonian on the active band subspace
(`band_slices.sigma = [b0, b3)`), expressed in the **original DFT basis** so the
coordinate system is fixed across iterations (no cumulative U-product drift).
Each step: eigh(H) → (E_qp, U_qp); rotate the original DFT wfn bundle by U_qp;
recompute χ₀ → W → Σ_xc via the mode-orthogonal `sigma_dispatch.compute_sigma_xc`;
rotate (V_H + Σ_xc) back to DFT basis; H_new = kin_ion_dft + (V_H+Σ_xc)_dft, then
band-partition masking + per-iteration scissor for out-of-ω-grid bands.

Physics: `H_qp_dft[k] = T+V_ion (DFT basis) + U (V_H + Σ_xc)_QP U†`,
one iteration from H⁰=diag(E_DFT) ≡ one-shot G0W0 at E=E_DFT.
Scissor: `ΔE = (α−1)·E_DFT + β` fit per val/cond on in-range bands.

## Entry points (grep over src/, tests/, tools/, scripts/)

Grep: `grep -rn "sc_iteration|SCInputs|SCState|gw_iteration_map|make_initial_state_from_dft|run_self_consistency|final_qp_eigenstates|dump_qp_wfn_artifacts|dump_sigma_omega_h5_final|_rotate_to_dft_basis" src tests tools scripts`

| symbol | callers |
|---|---|
| `SCInputs` | gw_jax.py:520 (constructed in main SC block); mentioned in docs of band_partition.py:77, common/chi_from_dipole.py:80 |
| `make_initial_state_from_dft` | gw_jax.py:533 |
| `run_self_consistency` | gw_jax.py:545 |
| `final_qp_eigenstates` | gw_jax.py:571 |
| `dump_qp_wfn_artifacts` | gw_jax.py:575 (gated on `config.debug.write_wfn_h5`) |
| `dump_sigma_omega_h5_final` | gw_jax.py:580 |
| `_rotate_to_dft_basis` | gw_jax.py:600-610 (rotates SC SigmaResult fields v_h/sigma_x/sigma_xc/sigma_sx/sigma_coh QP→DFT); doc-referenced in common/chi_from_dipole.py:37 ("OPPOSITE direction") |
| `SCState` | internal only (no external import found) |
| `gw_iteration_map` | internal only (called by run_self_consistency / _run_linear_mixing / _run_rcrop); in `__all__` but no external caller found |

**No test file references sc_iteration** (grep over tests/ found nothing; only
tests/test_band_partition.py covers the adjacent band_partition module).

## Function table

| function | lines | role |
|---|---|---|
| `SCInputs` (dataclass, frozen) | 82-117 | Iteration-invariant bundle: `wfns_dft` (Wavefunctions, never mutated), `V_q`, `kin_ion_dft` (nk,nb_act,nb_act) Ry, `quad` (minimax quadrature), `e_ref`, `static_head_terms`, `head_resolver`, `config`, `meta`, `mesh_xy`, `sym`, `wfn` (WFNReader), `band_slices`, `input_dir`, `partition` (BandPartition), `e_dft_active_kn_ry` (nk,nb_act), `valence_mask_active_kn` (nk,nb_act bool), `print_fn`. NOTE: docstring claims `partition` defaults to `BandPartition.all_protected(nb_active)` but the field has **no default** — caller must always supply it (gw_jax builds `protected = in_range` from `classify_bands_in_grid`). |
| `SCState` (dataclass, frozen) | 120-138 | Carry: `H_qp_dft` (nk,nb_active,nb_active) complex128 Ry, DFT basis, replicated sharding; `iteration: int`; `last_sigma_result: SigmaResult|None` (output-writer only, not fed forward). |
| `make_initial_state_from_dft` | 145-167 | H⁰ = per-k diag(E_DFT) on active subspace. Reads energies via `common.load_wfns.get_enk_bandrange(wfn, sym, sigma_range, sigma_range, nspinor)`. device_put with `NamedSharding(mesh_xy, P(None,None,None))` (fully replicated). |
| `_make_kshard_eigh` | 174-204 | Factory for jit'd per-k eigh/eigvalsh: `with_sharding_constraint` to `P(('x','y'),None,None)` (k-shard), Hermitise `H_h = 0.5*(H_k + H_k^†)`, `vmap(eigh)`, allgather back to replicated. Requires `mesh_xy.size | nk`. Pure perf hint. |
| `_KSHARD_EIGH_CACHE` / `_kshard_eigh_kernels` | 209-221 | Module-level dict keyed by `id(mesh_xy)` caching (eigh, eigvalsh) jit pair per mesh. |
| `_diagonalize_and_get_efermi` | 224-238 | eigh + midgap E_F: `vbm = max(E[:, :n_occ])`, `cbm = min(E[:, n_occ:])` (or vbm if no conduction bands), `efermi = (vbm+cbm)/2`. Returns (E, U, efermi_ry). |
| `_rotate_to_dft_basis` | 241-244 | jit. `O_DFT[m,n] = Σ_pq U[m,p]·O_QP[p,q]·U[n,q]^*` per k. Einsum VERBATIM: `'kmp,kpq,knq->kmn'` over (U, O_qp, conj(U)), optimize=True. |
| `gw_iteration_map` | 247-332 | One QSGW step. (1) eigh(H_qp_dft) → E_qp, U_qp, efermi. (2) `rotate_wavefunctions(wfns_dft, U_qp, enk_active_new=E_qp_ry, efermi, mesh_xy, active_slice=band_slices.sigma)`. (3) `screening_requests_for(config.compute_mode, config)` → `compute_screening(wfns_qp, V_q, requests, quad, e_ref, ...)` → `{role → W_q}` dict. (4) `compute_sigma_xc(config.compute_mode, wfns=wfns_qp, V_q, W_by_role, e_qp_ev=E_qp_ry*RYD_TO_EV, ..., write_sigma_omega_h5=False)`. (5) `delta_h_qp = v_h_kij_ry + sigma_xc_kij_ry`; rotate to DFT; `H_full = kin_ion_dft + delta_h_dft`; per-iteration scissor for out-of-range diagonals; `apply_band_partition(H_full, protected_mask, in_range_mask, scissor_E_qp_kn)`. Local imports of screening to avoid cycles. |
| `_scissor_E_qp_for_outofrange` | 339-377 | Takes diag(H_full) as candidate E_QP, restricts fit to in-range bands, `fit_scissor(E_DFT_eV, H_diag_eV, valence_mask_kn, fit_mask_kn)`, evaluates `ΔE=(α−1)E+β`; returns `E_DFT + ΔE/RYD_TO_EV` (Ry). Fast path: returns e_dft when `in_range.all()`. Host-side numpy (`np.asarray` pulls from device). |
| `run_self_consistency` | 384-456 | Driver. Fast path `max_iter==1` → single `gw_iteration_map`, empty rms_history (one-shot G0W0). Else dispatch `accelerator="rcrop"` (default, m=5 = BGW QSGW default) or `"linear"`. Reads env `LORRAX_SC_DUMP_DIR`. Rationale in docstring: QSGW Jacobian cycle eigenvalue ≲ −3 on dense band manifolds ⇒ plain fixed point 2-cycles; rCROP required. |
| `_run_linear_mixing` | 459-502 | `H_{n+1} = α·map(H_n) + (1−α)·H_n`. Convergence: RMS ΔE (eV) between consecutive eigvalsh, also prints ΔE_{k,k-2} (2-cycle detector). Break at `rms < tol_ev`. |
| `_run_rcrop` | 505-596 | Wraps `mixing.acceleration.rcrop_nojit(residual_fn, H0.flatten(), m, maxit, tol)`. residual_fn: reshape, **re-Hermitise** (rCROP combos drift), run map, return `(H_out − H_in).flatten()`. Side-effect bookkeeping via closure lists `_last_sigma`, `_iter_idx`, `_e_history` (captures the last SigmaResult for the writer). Tol conversion: `‖f‖₂ ≤ √(nk·nb²)·tol_ev/RYD_TO_EV` (per-element RMS ≈ RMS ΔE). Notes 2 map calls per rCROP iteration. Final: `H_final = Hermitise(result.x)`. |
| `_maybe_dump_e_history` | 599-612 | If `LORRAX_SC_DUMP_DIR` set: writes `e_history_kn_ev.npy`, shape (iter, k, n), eV. |
| `final_qp_eigenstates` | 615-638 | Post-convergence eigh; returns host numpy `(enk_qp_ry (nk,nb_act) f64, U_kmn (nk,nb_act,nb_act) c128 with U[k,m,n]=⟨DFT_m|QP_n⟩, efermi_ry float)`. |
| `dump_sigma_omega_h5_final` | 641-683 | Single end-of-run `sigma_mnk.h5` write from `state.last_sigma_result.sigma_c_omega_kij_ry` (intermediate iterations skip via `write_sigma_omega_h5=False`; "replaces ~30 redundant writes"). Calls **private** `ppm_pipeline._write_sigma_omega_h5` with a `SimpleNamespace(sigma_kij_h5_path=None)` stub for the `sigma_omega` arg; `build_ppm_sigma_runtime_options(config, input_dir)` for ppm_options. Returns None for static modes (no Σ_c(ω) tensor). |
| `dump_qp_wfn_artifacts` | 686-743 | Writes `WFN_qp.h5` (BGW-format WFN, active block ψ rotated by U, active energies → E_qp; drop-in for BSE/restart) and `qp_wfn_rotations.h5` ((U, E_qp) companion) via `file_io.qp_wfn.{write_qp_wfn_h5, write_qp_rotations_h5}`. Rank-0-only write + `multihost_utils.sync_global_devices("qp_wfn_h5_write")` barrier inside `try/except Exception: pass`. Passes `E_qp_nk = enk_qp_ry * 0.5` (Ry → Hartree) to the rotations file. |
| `__all__` | 746-756 | Exports incl. the underscored `_rotate_to_dft_basis` (comment: "used by main() to rotate SC SigmaResult fields"). |

## Key arrays crossing boundaries

- `H_qp_dft`: (nk, nb_active, nb_active) complex128 Ry, DFT basis, replicated (`P(None,None,None)`); briefly k-sharded `P(('x','y'),None,None)` inside eigh kernels. `mesh_xy.size` must divide nk.
- `E_qp_ry` (nk, nb_active) replicated; `U_qp` (nk, nb_act, nb_act) replicated, `U[k,m,n]=⟨DFT_m|QP_n⟩`.
- `e_qp_ev` handed to compute_sigma_xc as **host numpy** (`np.asarray(E_qp_ry)*RYD_TO_EV`).
- Scissor fit runs entirely host-side numpy on diag(H) pulled from device.
- rCROP flattens H to a 1-D host vector (`np.asarray(H0).flatten()`), so the whole fixed-point history (m=5 depth) lives in `rcrop_nojit` on host — nk·nb² per snapshot.

## Config flags / env vars consumed

Inside this file:
- `inputs.config.compute_mode` (gw_iteration_map — screening_requests_for + compute_sigma_xc dispatch).
- `config` passed opaquely into `compute_screening`, `compute_sigma_xc`, `build_ppm_sigma_runtime_options`.
- env `LORRAX_SC_DUMP_DIR` (eigenvalue-history npy dump).

At the caller (gw_jax.py:481-556, relevant because there is a live TODO "plumb through config"):
- `config.self_consistent` gates the whole path; `config.debug.write_wfn_h5` gates dump_qp_wfn_artifacts; `config.ppm.omega_min_ev/omega_max_ev` build the in-range mask.
- env `LORRAX_SC_MAX_ITER` (default 20), `LORRAX_SC_TOL_EV` (1e-4), `LORRAX_SC_ACCEL` (rcrop), `LORRAX_SC_DEPTH` (5), `LORRAX_SC_MIXING` (1.0) — **SC hyperparameters are env vars, not cohsex.in keys**.

## Cross-module deps

`gw.band_partition` (BandPartition, apply_band_partition), `gw.scissor` (fit_scissor),
`gw.sigma_dispatch` (SigmaResult, compute_sigma_xc), `gw.wavefunction_bundle`
(BandSlices, Wavefunctions, rotate_wavefunctions), `gw.screening`
(compute_screening, screening_requests_for — local import), `gw.ppm_pipeline`
(`_write_sigma_omega_h5`, private), `gw.gw_driver_helpers`
(build_ppm_sigma_runtime_options), `mixing.acceleration` (rcrop_nojit, local
import), `file_io.qp_wfn` (write_qp_wfn_h5, write_qp_rotations_h5),
`common.units` (RYD_TO_EV), `common.load_wfns` (get_enk_bandrange, local import).
All verified to exist (e.g. rcrop_nojit at src/mixing/acceleration.py:812).

## I/O

| file | dir | format | notes |
|---|---|---|---|
| `$LORRAX_SC_DUMP_DIR/e_history_kn_ev.npy` | write | .npy | (n_iter+1, nk, nb_active) eigenvalue snapshots, eV |
| `WFN_qp.h5` | write | BGW WFN HDF5 | via file_io.qp_wfn.write_qp_wfn_h5; active block ψ rotated, energies replaced; rank-0 only |
| `qp_wfn_rotations.h5` | write | HDF5 | (U, E_qp in **Hartree**, band range, k-points, k-grid dims); rank-0 only |
| `sigma_mnk.h5` | write | HDF5 | via ppm_pipeline._write_sigma_omega_h5 (Σ_c(ω) grid + Σ_x + V_H); PPM/dynamic modes only |

Reads: none directly (DFT energies come through the WFNReader object via get_enk_bandrange).

## Dead suspects

- None strictly dead. Grepped every public name over src/tests/tools/scripts.
  Externally-unused exports: `SCState` and `gw_iteration_map` are in `__all__`
  but only consumed inside this file (gw_jax imports neither). Not dead —
  they are the type of the return value and the map the drivers wrap — but
  the `__all__` advertises a wider API than is used.
- **Zero test coverage**: no file under tests/ imports or mentions sc_iteration.

## Redundancy suspects

- `_run_linear_mixing` (459-502) and `_run_rcrop` (505-596) duplicate the
  RMS-ΔE bookkeeping block verbatim (eigvalsh → rms → rms2 vs `_e_history[-3]`
  → print → append). ~15 duplicated lines; a shared "record_and_print(E_new)"
  helper would collapse it.
- `final_qp_eigenstates` (615-638) is a thin numpy-casting wrapper over
  `_diagonalize_and_get_efermi`; gw_jax.py calls it AND `dump_qp_wfn_artifacts`
  calls it again internally → when `config.debug.write_wfn_h5` is on, the
  converged H is diagonalised twice (cheap, but a duplicate compute path).
- `dump_sigma_omega_h5_final` reaches into `ppm_pipeline._write_sigma_omega_h5`
  (private) and fakes its `sigma_omega` argument with a SimpleNamespace stub —
  a seam that duplicates knowledge of ppm_pipeline internals ("only consulted
  attribute is sigma_kij_h5_path").

## Weird code

1. **Docstring/default mismatch** (SCInputs, lines 89-94 vs 114): docstring
   says `partition` defaults to `BandPartition.all_protected(nb_active)`;
   the field is required with no default. Hypothesis: default was removed
   (dataclass field-ordering constraint after non-default fields were added)
   and the docstring was never updated.
2. **`id(mesh_xy)`-keyed global cache** (line 209-221): if a Mesh is GC'd and
   a new Mesh reuses the id, a stale jit pair keyed to dead shardings could be
   returned. Benign in practice (one mesh per run) but a footgun for
   multi-mesh tests.
3. **`enk_qp_ry * 0.5` magic constant** (line 730): inline Ry→Hartree
   conversion with only a trailing comment; every other unit conversion in
   the file uses named `RYD_TO_EV`. qp_wfn_rotations.h5 stores Hartree while
   everything else in the module is Ry/eV.
4. **rCROP residual_fn is impure** (lines 539-565): mutates closure lists
   (`_last_sigma`, `_iter_idx`, `_e_history`, `rms_history`) inside the
   function handed to the fixed-point solver; iteration counter semantics
   ("2 map calls per rCROP iteration") mean `state.iteration` counts pipeline
   calls, not rCROP iterations.
5. **`try/except Exception: pass` around multihost barrier** (735-739): a
   failed `sync_global_devices` is silently swallowed — non-rank-0 processes
   could proceed before WFN_qp.h5 exists, contradicting the docstring's
   guarantee ("caller can rely on both files existing on every rank").
6. **SC hyperparameters via env vars** (gw_jax.py:534-540, this file line
   430): explicit TODO "plumb max_iter / tol_ev through config; env vars for
   now". Refactor target: cohsex.in keys.
7. **Design TODO in module docstring** (lines 49-55): diagonal-Σ updates for
   inactive bands inside the ω-grid are unimplemented; out-of-range bands get
   scissor only; metals explicitly NOT robust (lines 40-45, needs re-sort +
   re-occupy).
8. **Double Hermitisation**: residual_fn Hermitises before the map (544) and
   the eigh kernels Hermitise again (191/199), plus final x Hermitised (589).
   Harmless but triple-redundant; indicates uncertainty about where drift
   enters.
9. **`_rotate_to_dft_basis` exported in `__all__` despite underscore** (755)
   — gw_jax main() depends on a nominally-private helper; chi_from_dipole.py
   docs note its rotation direction is the OPPOSITE of what dipole code
   needs (convention trap for future callers).
