# src/gw/gw_jax.py — deep-read notes (2026-07-01)

956 LOC. Top-level GW driver ("orchestration" module). Read fully; all three top-level
functions cataloged below.

## Purpose

`gw.gw_jax` is the **main entry-point driver** for a LORRAX GW/COHSEX run. It owns no
physics kernels itself: it parses `cohsex.in` into `LorraxConfig`, builds the JAX device
mesh, wires together ISDF fit → χ₀ → W → Σ (static COHSEX and/or dynamic PPM and/or QSGW
self-consistency), applies q→0 head corrections and degenerate-set averaging, builds and
diagonalizes H_QP, and dispatches all output writers. The physics is delegated to
`gw_init`, `w_isdf`, `cohsex_sigma`, `ppm_pipeline`, `sc_iteration`, `qsgw_utils`,
`head_correction`, `scissor`, `degen_average`, `eqp_bgw`, `gw_output`, `screening`.

Category: **driver/orchestration: top-level GW pipeline (main())**.

## Module-level side effects (lines 1–19)

- L1–2: `set_default_env()` from `runtime` runs **before `import jax`** (env vars read at
  JAX import). Import-order-load-bearing.
- L11: `jax.config.update("jax_enable_x64", True)` — global f64.
- L13–15: `init_jax_distributed()`; `fallback_to_cpu_if_no_gpu_backend()` — run at
  **import time**, so merely importing `gw.gw_jax` initializes distributed JAX (the
  profiler tools rely on this: `tools/profile_gw_xprof.py` sets `sys.argv` then imports).
- L19: `_maybe_init_jax_distributed = init_jax_distributed` — back-compat alias; comment
  says "a few sandbox scripts import this name". Confirmed: sandbox scripts
  (`lorrax_sandbox/scripts/debug_zct_ws.py`, `t6b_hlo_si10_4gb.py`,
  `validate_si10_4gb_dump.py`, `phdf5_async_bench.py`, `dump_fit_bufasgn.py`) use the
  name (they define/import it; not from this repo's src/tests/tools/scripts). No caller
  inside lorrax_D itself (`src/common/cusolvermp_eigh_test.py` defines its own copy).

## Entry points / callers

- `main` ← console scripts `gw_jax` and `lorrax-gw` (`src/lorrax.egg-info/entry_points.txt`:
  `gw_jax = gw.gw_jax:main`, `lorrax-gw = gw.gw_jax:main`); `python -m gw.gw_jax -i cohsex.in`
  via `__main__` guard (L955–956); subprocess in
  `tests/test_gw_jax_regression.py:131` (`[sys.executable, "-m", "gw.gw_jax", "-i", ...]`);
  in-process in `tools/profile_gw_xprof.py:70` and `tests/tools/profile_gw_xprof.py:70`
  (`from gw import gw_jax; gw_jax.main()`).
- `_build_mesh` ← only `main` (L124). Two other modules **copy the idiom instead of
  calling it**: `src/bandstructure/htransform.py:32` ("matches gw_jax._build_mesh idiom")
  and `src/centroid/kmeans_cli.py:160` ("same recipe as ``gw_jax._build_mesh``").
- `_compute_static_head` ← only `main` (L353).
- `_maybe_init_jax_distributed` ← no in-repo caller (grep of src/tests/tools/scripts
  found only its definition and copies in blitz workspaces / sandbox scripts outside
  this checkout).

## Function-by-function

### `_build_mesh()` — L71–77
Role: construct 2D `Mesh` over all devices with the most-square factorization:
`gx = floor(sqrt(total))`, decrement until `total % gx == 0`, axes named `('x','y')`.
No physics. Consumers: `main` only. Duplicated (as copy-paste, not call) in
`bandstructure/htransform.py` and `centroid/kmeans_cli.py` → redundancy suspect.

### `_compute_static_head(head_resolver, meta, do_screened, print0)` — L80–88
Role: resolve q→0 head sample at ω=0 (`head_resolver.at(0.0+0.0j)`), build occupancy mask
`occ = arange(nb_sigma) < nelec`, and call
`compute_static_head_terms_from_sample(head, occ, cell_volume, nk_tot)` (in
`head_correction.py`). Physics: exact band-diagonal q→0 head terms for COHSEX Σ_X /
Σ_SX / Σ_COH — magnitude noted elsewhere in the file as ±W^c(0)/(2·V_cell·N_k) on-shell.
Prints diagnostics via `format_head_sample_diagnostics` / `format_static_head_diagnostics`.
Consumers: `main` only (gated on `config.do_G0`, L352).

### `main(argv=None)` — L91–952
Monolithic driver, ~860 lines. Stage map (all sub-steps inline in one function body):

| Stage | Lines | What happens |
|---|---|---|
| argparse + `print0` | 91–108 | `-i/--input` (default `cohsex_test.in`); rank-0-gated print closure (deliberately does NOT clobber `builtins.print`). |
| Config | 113–114 | `LorraxConfig.from_input_file(args.input)`. |
| Mesh + runtime | 119–134 | `_build_mesh()`, `setup_runtime(config, mesh_xy)`, `print_banner`. Note L129: `gw_output` imported **inside main** (deferred import). |
| Readers | 136–145 | `WFNReader(config.paths.wfn_file, mesh=mesh_xy)`, `SymMaps(wfn)`, `load_centroids(config.paths.centroids_file, wfn.fft_grid)`; makes `{input_dir}/tmp/`; restart file name `tmp/isdf_tensors_{n_rmu}.h5`. |
| Meta + bands | 147–154 | `Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)`; mutates `meta.rank/n_proc/sys_dim/bispinor/chunk_size` post-construction; `BandSlices.from_band_edges(*meta.band_edges)`. |
| Head resolver | 160 | Single `HeadResolver(config, input_dir, wfn, sym, meta, print0)` shared by COHSEX static head, W0 restart flush, and PPM dynamic head (one cache; resolution order overrides → epshead → s_tensor). |
| BGW vcoul override | 164–165 | `build_bgw_v_grid_fn(...)` — diagnostic only, None unless `use_bgw_vcoul`. |
| ISDF fit / restart | 168–191 | `prepare_isdf_and_wavefunctions(...)` → `isdf.V_qmunu`, `isdf.wf_bundle`; bispinor extras via `getattr(isdf, 'wf_bundle_transverse', None)`; `bispinor_v_q_path = tmp/v_q_bispinor.h5` when transverse bundle exists. |
| V flatten | 198 | `V_q = flatten_V_qmunu(V_qmunu)` — documented back-compat **no-op** (V is already flat-q `(nq, μ, μ)`); kept for legacy 8-D restart layouts. |
| χ₀ → W (screened) | 199–289 | See below. |
| Unscreened | 291–292 | `W_q = V_q` when `not do_screened`. |
| W0/head persist | 301–340 | Writes `W0_qmunu` + head scalars into `isdf_tensors_*.h5` (see I/O). |
| `Gij` | 342 | `build_Gij(meta, mesh_xy)` (cohsex_sigma). |
| Static head | 351–354 | `_compute_static_head` iff `config.do_G0`. Comment documents the historical "Bare Σ_X missing q→0 head" bug (compare SKILL §4i): head must NOT be gated on `not use_ppm_sigma`. |
| Static COHSEX Σ | 356–371 | `compute_cohsex_sigma(wfns, V_q, W_q, meta, mesh_xy, Gij=, do_screened=, static_head_terms=, compute_bare_x=True, wfns_transverse=, bispinor_v_q_path=, backend=config.backend.slab_io)` → dict `sig_sx, sig_coh, sig_h, sig_x` (all `(nk, nb, nb)` Ry, replicated per L458-466 comment). `import gc; gc.collect()` immediately before (L357). |
| Σ_X print + degen avg | 378–390 | `average_within_degenerate_sets` on diag(Σ_X) (mirrors BGW `Sigma/shiftenergy.f90`), tol `config.degen_avg_tol_ry`, unless `no_degen_averaging`. |
| One-shot dynamic Σ | 406–437 | `mode.is_dynamic and not self_consistent`: `screening_requests_for(mode, config)` minus role=="static", `compute_screening(...)` → `_W_extra["probe"]`, then `compute_ppm_sigma_pipeline(wfns, V_q, W_static_q=W_q, W_probe_q=_W_extra["probe"], sig_x, sig_h, quad, e_ref, ...)`. |
| Post-PPM seam | 438–456 | Copies every downstream field of `ppm_outputs` into bare locals (`sigma_omega_h5_path`, `sigma_c_at_dft_ev`, `sigma_xc_at_dft_ev`, `omega_dft_rel_ev`, `efermi_dft_ev`, `sigma_c_omega`, `head_sigma_diag_w_kn_ry`, `omega_grid_ev/ry`), then `del ppm_outputs` — so SC / one-shot / static branches all feed identical names to writer & freq-debug. |
| kin_ion load | 467–470 | `load_kin_ion_submatrix(config.paths.kin_ion_file, b0, b3, mesh, backend=slab_io)` — fully replicated `(nk, nb, nb)` Ry. |
| SC (QSGW) branch | 474–627 | See below. |
| Diag-Σ(E) branch | 628–709 | See below. |
| Static branch | 710–712 | `sigma_total = sig_sx + sig_coh + sig_h`. |
| Degen avg at H seam | 714–742 | `apply_to_matrix_diagonals` on `sigma_total, sig_sx, sig_coh, sig_h, sig_x` + `average_within_degenerate_sets` on `sigma_c_at_dft_ev`; comment: replaces earlier per-component averaging at the writer; off-diagonals preserved. |
| H build + eigh | 744–746 | `H = 0.5·((kin_ion+Σ_tot) + (kin_ion+Σ_tot)^†)`; `E_full, U_full = vmap(eigh)(H)`. Physics: `H_QP = (H_DFT − V_xc) + V_H + Σ_xc` (kin_ion file = H_DFT − V_xc). |
| WFN_qp.h5 (one-shot) | 757–772 | `write_qp_wfn_h5` on rank 0 iff `debug.write_wfn_h5 and not self_consistent`; `multihost_utils.sync_global_devices("oneshot_qp_wfn_h5_write")` wrapped in bare `try/except Exception: pass`. |
| Σ_c diag extract | 780–792 | `extract_sigma_diag_replicated(sigma_c_omega, mesh_xy)` → eV; Ry copy for writer. |
| freq-debug table | 805–909 | Rank-0 + `debug.sigma_freq_debug_output`: builds columns `E_dft, Edft-Ef, kin_ion, V_H, x_bare[, x_head][, sig_c(Edft)[, sig_c_head(Edft)] | sex_0, coh_0[, sex_head, coh_head]], eqp0, eqp1` via `compute_z_factor_from_omega_grid`, `interp_along_omega`, `compute_eqp_diag`; writes with `write_sigma_freq_debug_table(config.debug.sigma_freq_debug_file, cols)`. Invariants documented: `eqp0 = kin_ion + V_H + x_bare + Re Σ_c(E_DFT)` exactly; `eqp1 = E_DFT + Z·(eqp0 − E_DFT)`, Z=1 static. |
| Output | 911–948 | `GWResults(...)` (all components host-copied via `np.array`), rank-0 `write_results(results, sigma_diag_file, eqp0_file, eqp1_file, input_dir, kpoints_crys=sym.unfolded_kpts, kgrid, kpoints_irr_frac=wfn.kpoints, kpoints_reduced=wfn.kpoints, kirr_to_kfull=sym.kirr_fullids, ...)`. |
| Timing | 949–952 | `timing.report(...)`; `return 0`. |

#### χ₀ → W sub-stage detail (L199–289)
Physics: `W = (1 − Vχ₀)⁻¹ V` (static quadrature χ₀). Steps:
- `_use_ibz_w_requested = not int(os.environ.get('LORRAX_FORCE_FULL_BZ','0'))`; if
  requested and `sym.q_irr_full_idx` exists, calls
  `v_q_g_flat._resolve_ibz_q_list(sym=, centroid_indices=, kgrid=, fft_grid=, verbose=False)`
  → tuple `(_, q_irr_frac, full_to_irr_idx, full_to_irr_sym, sym_perm, L_table, use_ibz_w)`
  (activation gated on centroid orbit closure, checked inside that helper).
- `build_static_quadrature(wfns, config.minimax_config, print_fn)` → `(quad, e_ref)`.
- `precompile_chi0` / `compute_chi0(wfns, quad, meta, mesh_xy, energy_reference=e_ref)` →
  `chi0_q` `(nq, μ, μ)` sharded `P(None,'x','y')`.
- IBZ slice: `slice_q_full_to_ibz(V_q, sym.q_irr_full_idx, out_sharding=NamedSharding(mesh, P(None,'x','y')))`
  for both V_q and χ₀_q.
- `precompile_solve_w` / `solve_w(V_q_solve, chi0_q_solve, meta, mesh_xy, solver=config.backend.screening_solver)`;
  **χ₀ buffer is donated inside solve_w** — explicit `del chi0_q_solve` and the
  `block_until_ready()` calls are documented as load-bearing for donation (comment L228–236
  explicitly forbids `_chi_sec.watch(...)` because a bound method would keep χ₀ alive).
- IBZ→full unfold: `unfold_v_q(W_q_solve, irr_idx=, sym_idx=, sym_perm=, L_table=, q_irr_frac=, mesh_xy=, n_sym_spatial=sym_perm.shape[0]//2)`
  — comment: "same helper V_q uses (centroid double-permute + L-phase + TRS conj)";
  `n_sym_spatial = len(sym_perm)//2` encodes the TRS-augmented table convention.
- Timing sections: `gw_jax.chi0_W.{chi,W}.{compile,exec}`, `W.slice_to_ibz`, `W.unfold_to_full_bz`.

#### W0 + head persist (L301–340)
Gate: `config.do_screened and os.path.exists(tensors_filename)`. Writes into
`tmp/isdf_tensors_{n_rmu}.h5`:
- `write_w0_qmunu_to_h5(tensors_filename, W_q, mesh=, backend=config.backend.slab_io)` —
  dataset `/W0_qmunu`, flat-q rank-3 `(nq, μ, μ)`. Comment documents the fixed bug: the
  legacy 8-D reshape tripped `phdf5 async write: dataset rank mismatch ds=/W0_qmunu
  file_rank=3 write_rank=8`; BSE consumers updated to flat-q in commit `a052a1c`.
- `write_head_scalars_to_h5(tensors_filename, vhead=head_static.vc0, whead=whead_arr, omega_grid=)`.
  `whead` axis length 1 (COHSEX static) or 2 (dynamic: static + probe). Probe frequency:
  HL-PPM → real Ω = `config.ppm.omega_p` on the real axis; GN-PPM → `1j·omega_p`
  imaginary axis (L316–321). Downstream applies rank-1 head update via
  `head_correction.apply_q0_head_rank1`.

#### SC / QSGW branch detail (L474–627)
- Builds `BandPartition(protected_mask, in_range_mask)` from
  `classify_bands_in_grid(E_dft_ev, ω_min+E_F, ω_max+E_F)`; ω-window from
  `config.ppm.omega_min_ev/omega_max_ev` + `wfn.efermi·RYD_TO_EV`. Out-of-window bands get
  per-iteration scissor (comment: otherwise clamped Σ_c "explodes the iteration").
- `SCInputs(wfns_dft=wfns, V_q=V_q, kin_ion_dft=kin_ion, quad=quad, e_ref=e_ref, static_head_terms, head_resolver, config, meta, mesh_xy, sym, wfn, band_slices, input_dir, partition, e_dft_active_kn_ry, valence_mask_active_kn, print_fn)`.
- SC knobs from **env vars, not config** (TODO at L534): `LORRAX_SC_MAX_ITER` (20),
  `LORRAX_SC_TOL_EV` (1e-4), `LORRAX_SC_ACCEL` ("rcrop"), `LORRAX_SC_DEPTH` (5),
  `LORRAX_SC_MIXING` (1.0).
- `run_self_consistency(state_init, sc_inputs, max_iter, tol_ev, accelerator, history_depth, mixing)`
  → `(_state_final, sc_rms_history)`. Iteration map (comment L476–481): rotate ψ by
  U_qp from `eigh(H_qp_dft)`, recompute χ₀→W→Σ_xc each step; convergence = RMS ΔE between
  consecutive `eigvalsh`.
- Post-SC: `final_qp_eigenstates(state_final, n_occ, mesh_xy)` → `(E_qp, U_kmn, efermi)`;
  `dump_qp_wfn_artifacts` (WFN_qp.h5 + qp_wfn_rotations.h5, gated `debug.write_wfn_h5`);
  `dump_sigma_omega_h5_final` (single end-of-run sigma_mnk.h5 write).
- **Basis fix** (comments L586–599, load-bearing): `SigmaResult.{v_h,sigma_x,sigma_xc,sigma_sx,sigma_coh}_kij_ry`
  live in the **QP basis** but `kin_ion` is DFT-basis; every field is rotated back via
  `sc_iteration._rotate_to_dft_basis(field, U_kmn)` (a private-name cross-module import),
  and `sigma_total = Σ_xc(DFT) + V_H(DFT)` overrides the generic H-build inputs. Without
  it, "eigh sees kin_ion (DFT) + sigma_xc (QP) → nonsense eigenvalues... off by tens of
  eV per band on MoS2".

#### Diagonal Σ(E) / QSGW-collapse branch detail (L628–709)
Gate: `mode.is_dynamic and sigma_c_omega is not None` (one-shot dynamic). Physics:
- `sigma_xc_diag_w_kn_ry = Σ_c(ω,k,n) + Σ_x(k,n)`; fixed point
  `E = h0 + Re Σ_xc(E)` solved by `solve_diagonal_sigma_fixed_point(h0_diag − E_F, Σ_xc(ω), ω_grid, max_iter=120, tol_ev=1e-7/RYD_TO_EV, mixing=0.6)` (qsgw_utils).
- Per-band scissor for out-of-grid bands: `classify_bands_in_grid` + optional
  `fit_scissor(E_dft_eV, E_sc_eV, valence_mask, fit_mask)` when
  `config.ppm.sigma_at_dft_extrapolate` and `0 < n_in < n_total`; fallback `E_DFT`
  (comment: older `eigvalsh(H_qp)` fallback "was unreliable for pseudobands").
- `build_qsgw_sigma_xc(sigma_c_omega, sig_x_rep, ω_grid_eV, E_sc_eV, mesh_xy)` →
  replicated `Σ_xc^QSGW(k,i,j)` (Ry) + clip diagnostics (`n_clipped`, `frac_clipped`);
  `sigma_total = Σ_xc^QSGW + sig_h`. Unit seams documented meticulously (Ry internally,
  eV only at kernel boundary and prints).

## Cross-module dependencies (imports)

`runtime` (env/dist init), `file_io` (WFNReader, write_sigma_omega_h5 [unused, see
suspects], load_kin_ion_submatrix, load_centroids; deferred: write_w0_qmunu_to_h5,
write_head_scalars_to_h5, write_sigma_freq_debug_table, qp_wfn.write_qp_wfn_h5),
`common.symmetry_maps` (SymMaps, slice_q_full_to_ibz, unfold_v_q),
`common.load_wfns.get_enk_bandrange`, `common` (Meta, RYD_TO_EV, jax_profile, timing),
`gw.gw_config` (ComputeMode, LorraxConfig), `gw.gw_init`, `gw.gw_driver_helpers`
(build_bgw_v_grid_fn, setup_runtime), `gw.w_isdf`, `gw.ppm_pipeline`, `gw.cohsex_sigma`,
`gw.qsgw_utils` (+ deferred interp_along_omega), `gw.head_correction`,
`gw.wavefunction_bundle.BandSlices`, and deferred/in-function imports:
`gw.gw_output`, `gw.v_q_g_flat._resolve_ibz_q_list`, `gw.degen_average`, `gw.screening`,
`gw.sc_iteration` (incl. private `_rotate_to_dft_basis`), `gw.band_partition`,
`gw.scissor`, `gw.eqp_bgw`, `jax.experimental.multihost_utils`.

## Config flags / env vars consumed (grep-verified in this file)

cohsex.in via LorraxConfig: `paths.wfn_file`, `paths.centroids_file`,
`paths.kin_ion_file`, `paths.sigma_diag_file`, `paths.eqp0_file`, `paths.eqp1_file`,
`input_dir`, `nval`, `ncond`, `nband`, `bispinor`, `sys_dim`, `memory.chunk_size`,
`minimax_config`, `do_screened`, `do_G0`, `compute_mode` (drives X_ONLY/COHSEX/GN_PPM/HL_PPM),
`self_consistent`, `ppm.omega_p`, `ppm.omega_min_ev`, `ppm.omega_max_ev`,
`ppm.sigma_at_dft_extrapolate`, `backend.screening_solver`, `backend.slab_io`,
`no_degen_averaging`, `degen_avg_tol_ry`, `debug.write_wfn_h5`,
`debug.sigma_freq_debug_output`, `debug.sigma_freq_debug_file`.
(Also indirectly: `use_bgw_vcoul` via build_bgw_v_grid_fn, head override fields via HeadResolver.)

Env vars: `LORRAX_FORCE_FULL_BZ`, `LORRAX_SC_MAX_ITER`, `LORRAX_SC_TOL_EV`,
`LORRAX_SC_ACCEL`, `LORRAX_SC_DEPTH`, `LORRAX_SC_MIXING`.

## Key arrays crossing boundaries

| Array | Shape | Residency / sharding |
|---|---|---|
| `V_qmunu` / `V_q` | `(nq, μ, μ)` flat-q | device, `P(None,'x','y')` |
| `chi0_q` | `(nq, μ, μ)` | device, `P(None,'x','y')`; **donated into solve_w** |
| `W_q` | `(nq, μ, μ)` | device, `P(None,'x','y')`; IBZ-solved then unfolded |
| `kin_ion`, `sig_sx/coh/h/x`, `sigma_total` | `(nk, nb_sigma, nb_sigma)` Ry | device, fully **replicated** `P(None,None,None)` |
| `sigma_c_omega` | `(nω, nk, i, j)` Ry | device, sharded (only sharded object past Σ stage; collapsed by build_qsgw_sigma_xc / extract_sigma_diag_replicated) |
| `E_full, U_full` | `(nk, nb)`, `(nk, nb, nb)` | replicated (vmap eigh) |

No einsums appear in this file (all einsum kernels live in the imported modules).

## I/O

Reads:
- `cohsex.in`-style input (arg `-i`, text, via LorraxConfig).
- `WFN.h5` (`config.paths.wfn_file`) via WFNReader.
- centroids file (`config.paths.centroids_file`) via `load_centroids`.
- kin_ion file (`config.paths.kin_ion_file`) via `load_kin_ion_submatrix` (band window b0:b3).
- (indirect) epshead / s_tensor via HeadResolver; BGW vcoul file when `use_bgw_vcoul`.

Writes:
- `{input_dir}/tmp/isdf_tensors_{n_rmu}.h5` — restart: `/W0_qmunu` (flat-q rank-3), head
  scalars `vhead`, `whead[nω]`, `omega_grid` (via write_w0_qmunu_to_h5 / write_head_scalars_to_h5).
- `{input_dir}/tmp/v_q_bispinor.h5` — path constructed here (L188–191) and passed to
  compute_cohsex_sigma; written elsewhere (gw_init bispinor path), read by cohsex Σ^B.
- `WFN_qp.h5` in input_dir (one-shot path, `debug.write_wfn_h5`); SC path writes its own
  WFN_qp.h5 + qp_wfn_rotations.h5 via dump_qp_wfn_artifacts.
- `sigma_mnk.h5` (sigma_omega): via ppm_pipeline (one-shot) or dump_sigma_omega_h5_final (SC).
- freq-debug text table `config.debug.sigma_freq_debug_file`.
- `sigma_diag.dat`, `eqp0.dat`, `eqp1.dat` via `gw_output.write_results` (rank 0).

## Suspects

### dead_suspects
1. `_maybe_init_jax_distributed` (L19) — grepped
   `grep -rn "_maybe_init_jax_distributed" src tests tools scripts`: only the definition
   here plus an unrelated same-named local function in `src/common/cusolvermp_eigh_test.py`.
   The claimed consumers are sandbox scripts *outside* the repo
   (`lorrax_sandbox/scripts/*.py`) — dead within the package proper, kept as external shim.
2. `write_sigma_omega_h5` import (L21) — imported from `file_io` but **never used in this
   file** (grep: only the import line; the actual writes go through
   `ppm_pipeline._write_sigma_omega_h5` and `sc_iteration.dump_sigma_omega_h5_final`).
   Leftover from the pre-`ppm_pipeline` monolith.

### redundancy_suspects
1. `_build_mesh` most-square factorization is copy-pasted (not imported) in
   `src/bandstructure/htransform.py:32` and `src/centroid/kmeans_cli.py:147-160` — three
   textual copies of the same mesh recipe.
2. `flatten_V_qmunu` call at L198 is a self-documented **no-op back-compat shim** for a
   legacy 8-D layout ("kept as a back-compat no-op for restart paths").
3. Dual WFN_qp.h5 writers: SC path (`dump_qp_wfn_artifacts`, L574-579) vs one-shot path
   (`write_qp_wfn_h5`, L757-772) — comment admits "same physics, slightly different
   numerics", guarded only by `not config.self_consistent` to avoid clobbering.
4. Degenerate-set averaging applied twice: once to `sig_x_diag` for the console print
   (L380-388) and again to all matrices at the H-build seam (L723-742) — comment
   acknowledges "the redundancy across components is not a perf concern".
5. freq-debug eqp0/eqp1 recomputes the writer's math (L886-906) to stay "bit-consistent"
   with eqp{0,1}.dat — deliberate duplication of the eqp recipe with the pipeline's own
   interp noted as differing by ~10 meV.

### weird_code
1. L1-2 / L13-15: import-order-load-bearing module side effects (`set_default_env()`
   before `import jax`; distributed init at import). Any refactor that reorders imports
   breaks GPU/dist setup silently.
2. L228-236 + L264-269: buffer-donation choreography — `block_until_ready()` +
   `del chi0_q_solve` are semantically load-bearing (XLA donation), and a comment
   explicitly bans `timing` `.watch(...)` because a bound method reference blocks donation.
   Fragile hidden contract between driver and `solve_w`.
3. L276-277: `_n_sym_spatial = sym_perm.shape[0] // 2` — magic halving encoding the
   convention that the sym_perm table is TRS-augmented (spatial ops then TRS partners).
   Directly adjacent to the historical TRS-blind unfold bug territory.
4. L521 & L237: `quad`/`e_ref` are only defined inside the `if config.do_screened:` block
   but are referenced by the SC branch (`SCInputs(quad=quad, e_ref=e_ref)`, L522) and the
   one-shot dynamic branch (L421, L431). Dynamic modes are config-validated to require
   `do_screened=true` (gw_config.py:746-748), but `self_consistent=True` with
   `do_screened=False` (SC X_ONLY) would hit a **NameError on `quad`** — latent crash if
   that combination is ever legal.
5. L534-540: SC iteration hyperparameters read from env vars with an in-line TODO
   ("plumb max_iter / tol_ev through config; env vars for now").
6. L600: cross-module import of a private function
   `from .sc_iteration import _rotate_to_dft_basis` — QP→DFT basis rotation is
   correctness-critical (tens of eV wrong without it) yet lives behind an underscore name.
7. L767-771: bare `try: multihost_utils.sync_global_devices(...) except Exception: pass`
   — swallows all sync failures after rank-0 WFN_qp.h5 write.
8. L400-405: history note embedded in code — analytic q→0 PPM head removed in commit
   1542342 (Apr-10) then re-added 2026-04-25; magnitude ±W^c(0)/(2·V_cell·N_k)
   (~1.24 eV/band Si 4×4×4). Comment says this decision "is not yet captured anywhere else".
9. L649-651: hard-coded fixed-point solver constants `max_iter=120, tol_ev=1e-7/RYD_TO_EV,
   mixing=0.6` — not config-exposed.
10. L830: `float(efermi_dft_ev or 0.0)` — silently maps a legitimately-zero (or None)
    Fermi energy to 0.0 for the `Edft-Ef` debug column; `or` on a float is a footgun if
    E_F is exactly 0.
11. L187: `getattr(isdf, 'wf_bundle_transverse', None)` — duck-typed optional attribute
    instead of an explicit field on the isdf result.
12. L94-98: default input filename is `cohsex_test.in` (test-flavored name as production
    default).
