# psp dipole/G-space group — refactor map notes (2026-07-01)

Group: `src/psp/get_dipole_mtxels.py`, `src/psp/ionic_gspace.py`, `src/psp/gvec_utils.py`, `src/psp/orbital_magnetization.py`
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`

---

## src/psp/get_dipole_mtxels.py (755 loc)

**Purpose.** Standalone CLI driver (`python -m psp.get_dipole_mtxels`) computing velocity/dipole matrix elements ⟨mk|v|nk⟩ = p + i[r, V_NL] at q=0 for every full-BZ k, plus (optionally, `--with-finite-q`) finite-q cell overlaps ρ_cv(k,q) and symmetric velocity matrix elements v_cv(k,q) for the SOS χ head/wing pipeline. Writes `dipole.h5`. Category: **physics: optics/dipole matrix elements (BSE + SOS-chi front-end)**.

**Function table**
| function | role |
|---|---|
| `compute_p_operator_k(wfn_k, Gk_crys, kpoint, bdot, bvec, blat)` | kinetic momentum (3,nb,nb): Σ_G (k+G)_cart c*c via `dft_operators.momentum_matrix_k` |
| `compute_vnl_matrix_from_setup(...)` | ⟨m|V_NL(k)|n⟩ via `vnl_ops.build_vnl_kdata_from_kvec` + `vnl_matrix`; used by numeric-FD vNL mode only |
| `compute_vnl_velocity_cart(...)` | analytic dV_NL/dK_cart via `vnl_ops.vnl_velocity_matrix` (compute_dZ=True); slices bispinor ψ to physical spinors |
| `compute_projected_momentum_bgw_like(*a, **k)` | **stub — raises NotImplementedError** ("Debug-only momentum projection removed") |
| `compute_block_direct_cnk(*a, **k)` | **stub — raises NotImplementedError** ("Debug-only direct cnk path removed") |
| `_build_g_lookup(Gk_int_kmq, Gk_int_k, G_wrap)` | host-side python-dict integer lookup μ_k → μ_kmq under umklapp; returns (map_arr, mask), -1→0+mask for out-of-sphere |
| `_cell_overlap_with_lookup(...)` (@jit) | G-sphere ρ_mn and symmetrized v_sym = ½(v_R + v_L) einsum with umklapp gather |
| `_apply_kinetic_velocity_Gbox(psi_Gbox_k, kvec, bvec_blat, fft_grid)` (@jit) | **DEAD** — kinetic velocity in FFT-box layout; zero callers |
| `_cell_overlaps_at_q_Gbox(...)` (@jit) | **DEAD** — G-box cell overlaps with jnp.roll umklapp (kinetic-only); superseded by the G-sphere lookup path |
| `compute_finite_q_mtxels(wfn, sym, meta, vnl_setup, wfn_k_sharded, Gk_crys_all, *, iq_list, nv_block, nc_block)` | driver: per-k v_kin+V_NL applies (python lists, ragged G-spheres), then per-(k,q) loop → rho_cvkq (nc,nv,nk,nq), v_cvkq (3,nc,nv,nk,nq), kminq_idx |
| `main(argv)` | full CLI: read cohsex.in, load WFN + SymMaps + Meta, load ψ to devices, build vnl_setup, per-k dipole+deltaE loop, optional finite-q, write dipole.h5 |

**Entry points / callers** (grep over src, tests, tools, scripts):
- `main` ← CLI only (`python -m psp.get_dipole_mtxels`, documented in `src/bse/BGW_COMPARE.md`, `src/bse/STATUS.md`). No in-code importer of `main`.
- `compute_p_operator_k`, `compute_vnl_velocity_cart` ← zero callers in lorrax_D; imported by `psp/orbital_magnetization.py` **in the lorrax_B_orbmag_wt worktree only** (see below).
- Output consumers: `src/bse/absorption_common.py:load dipole.h5` (→ `absorption_eigvecs.py`, `absorption_haydock.py`), `src/common/chi_sos.py` (reads `finite_q/{rho_cvkq, v_cvkq, kminq_idx}`), `scripts/checks/sigma_direct_check.py` via `common.chi_from_dipole.read_dipole_h5`.
- `compute_vnl_matrix_from_setup`, `compute_finite_q_mtxels`, `_build_g_lookup`, `_cell_overlap_with_lookup` ← internal to this module only.

**Flags/config consumed.** `cohsex.in` [cohsex] block via `psp.get_DFT_mtxels.read_cohsex_input`: `wfn_file`, `nval`, `ncond`, `nband`, `bispinor`. CLI: `--divide-energy`, `--debug`, `--debug-kindex`, `--vnl-mode {analytic,numeric}`, `--vnl-h`, `--vnl-h-rel`, `--vnl-num-scheme {naive,richardson}`, `--skip-vnl` (match BGW `use_momentum`), `--out`, `--with-finite-q`, `--iq-list`. Env: `JAX_ENABLE_X64=1` (setdefault).

**I/O.**
- Reads: `cohsex.in` (INI), `WFN.h5` (BGW HDF5 via WfnLoader), `*.upf` pseudos (input dir, fallback `../qe/{scf,nscf}`).
- Writes: `dipole.h5` — datasets `dipole_cart` (3,nk,nb,nb) c128, `deltaE` (nk,nb,nb) f64; attrs `nbands, nk, skip_vnl, note`; optional group `finite_q/` with `rho_cvkq` (nc,nv,nk,nq), `v_cvkq` (3,nc,nv,nk,nq), `kminq_idx` (nk,nq), `iq_list`, attrs `n_occ, v_lo, c_hi, note`.

**Cross-module deps.** `file_io.WfnLoader`, `common.symmetry_maps.SymMaps`, `common.load_wfns.read_Gvecs_to_devices`, `common.Meta`, `common.kq_mapping.{kminq_idx_for_iq, umklapp_G_wrap}` (umklapp_G_wrap imported but unused — G_wrap recomputed inline at line 400), `psp.get_DFT_mtxels.read_cohsex_input`, `psp.pseudos`, `psp.dft_operators.{generate_gvectors_k, gather_psi_G_from_crys, momentum_matrix_k, apply_kinetic_velocity_to_ket}`, `psp.vnl_ops`.

**Dead suspects.**
- `_apply_kinetic_velocity_Gbox` (L182), `_cell_overlaps_at_q_Gbox` (L225): grep `_apply_kinetic_velocity_Gbox|_cell_overlaps_at_q_Gbox` across src/tests/tools/scripts → only their definitions. Old FFT-box finite-q path replaced by the G-sphere `_cell_overlap_with_lookup` path (compute_finite_q_mtxels docstring even says "G-sphere throughout (no FFT box)").
- `compute_projected_momentum_bgw_like`, `compute_block_direct_cnk` (L95-100): NotImplementedError tombstones, zero callers.
- `compute_p_operator_k`, `compute_vnl_velocity_cart`: zero callers inside lorrax_D beyond `main`; external caller is the unmerged orbital_magnetization script in lorrax_B_orbmag_wt. Keep if orbmag merges, else fold into main.

**Weird code.**
- L636 `vNL_cart = -vNL_cart` with comment "Liu-2024 Eq. 17 / BGW k·p ... flip here": global sign flip to BGW convention; L366-369 in `compute_finite_q_mtxels` repeats the same flip with a cross-reference comment. Per project memory the *physical* dH/dk is p+vNL — so `dipole.h5` deliberately stores the BGW-convention p−vNL. Convention split across two code sites; unify in refactor.
- Mixed indentation: tabs in `compute_p_operator_k`…`compute_vnl_velocity_cart` and `main`, 4-space elsewhere (stubs + finite-q block appended later).
- `_build_g_lookup` is an O(nG) pure-Python dict loop per (k,q) pair — nk²·nG host-side; fine for small grids, a wall-clock landmine for dense meshes.
- L400: G_wrap recomputed via `np.round((k − q) − k_can)` inline while `umklapp_G_wrap` is imported and unused.
- Ragged per-k Python lists (`vket_v_per_k` etc.) hold (3,nb,ns,nG_k) JAX arrays for ALL full-BZ k simultaneously — memory scales nk·nb·nG with no chunking.
- Bispinor handling: pads V_NL result with zero spinor components (L373-378) — silent physics assumption that V_NL acts only on the first nspinor_E components.
- `deltaE` fallback `except Exception: k_red = int(i)` (L640-643) silently mislabels energies if `irr_idx_k` missing.

**Redundancy suspects.** q=0 path (main loop, `compute_vnl_velocity_cart` → `vnl_velocity_matrix`) vs finite-q path (`apply_vnl_velocity_to_ket` + einsum overlaps) assemble the same v = kin + VNL physics through two different vnl_ops entry points with independently-applied sign flips. Dead G-box duo is a third copy of the overlap logic.

---

## src/psp/ionic_gspace.py (273 loc)

**Purpose.** Unified jittable G-space builder for ionic local potential V_loc(r) and NLCC core density ρ_core(r) from UPF pseudopotentials: structure factors → radial-table interpolation → species accumulation → long-range Coulomb tail (optional 2D truncation) → IFFTs, all inside one XLA module. Replaces older per-atom Python loops (QE setlocal/struct_fact/set_rhoc analogue). Category: **physics: DFT potential setup (SCF/NSCF/Sternheimer support)**.

**Function table**
| function | role |
|---|---|
| `species_structure_factors(species_tau, species_natoms, G_crys_flat, max_atoms)` (@jit) | S_s(G) = Σ_a e^{-2πiG·τ} via nested lax.scan (species × padded atoms), O(N) memory |
| `accumulate_species_on_G(tables, prefactors, S_species, G_norm_flat, q0, dq)` (@jit) | Σ_s pf·interp(table_s,|G|)·S_s(G) via lax.scan; uses `radial_jax.interp_uniform_jax` |
| `build_fft_G_data(fft_grid, bvec, blat)` | host: Miller indices + |G| on full FFT grid |
| `build_ionic_and_core(wfn, pseudos, fft_grid, *, truncation_2d, n_q=4000)` | high-level: species extraction, radial tables, alpha-Z q=0 override, dispatch to `_ionic_gspace_jit`; returns (V_loc_r, rho_core_r, rho_core_G_scaled) |
| `_ionic_gspace_jit(...)` (@jit, static max_atoms/nx/ny/nz/truncation_2d) | one-shot: S(G) → NLCC ρ_core → V_loc SR + LR Coulomb tail (+2D cutoff) → two IFFTs |

**Entry points / callers.**
- `build_ionic_and_core` ← `psp/scf_potential.py:81` (build_dft_potentials) and `psp/kpm_dos.py:163`. Transitively feeds `run_nscf`, `run_sternheimer`, orbmag-sternheimer.
- `species_structure_factors`, `accumulate_species_on_G`, `build_fft_G_data` ← internal only (in `__all__` but no external importer found via grep of the four names across src/tests/tools/scripts).

**Flags consumed.** kwargs only: `truncation_2d` (2D slab Coulomb cutoff for V_loc LR), `n_q` (radial table resolution, default 4000).

**I/O.** None directly (pseudos dict passed in; UPF parsing lives in `psp.pseudos`/`psp.radial_tables`). Prints core-density integral diagnostic.

**Cross-module deps.** `psp.radial.radial_jax.interp_uniform_jax`, `psp.species.{extract_species, build_atom_species_map}`, `psp.radial_tables.{build_all_tables, alpha_z}`.

**Weird code.**
- L251-252: LR tail multiplied by `exp(-0.25·G²)` — hard-coded Gaussian-smeared ionic charge (Ewald η=1); the compensating real-space term must live in alpha_z/table conventions; no comment tying the 0.25 to the vloc short-range table split. Magic constant.
- L176-177: `tables["vloc"][i, 0] = az * vol / (4π)` — q=0 alpha-Z override mutates the shared table dict in place, and is skipped when `truncation_2d=True` (correct for 2D but implicit).
- L258: 2D truncation `lz = π / B_cart[2,2]` — assumes c-axis ⟂ ab-plane and z-aligned; silently wrong for oblique cells.
- Mixed FFT normalizations: ρ_core via `ifftn(...)·N` (unnormalized) while V_loc via `ifftn(norm="ortho")` with `sqrtN` baked into `vloc_pf` and the LR prefactor — two conventions in one function, easy refactor trap.
- `q0` hardcoded 0.0 but threaded as a traced scalar argument through jit.

**Dead/redundancy.** None internal. Note the docstring claims these primitives "replace the Python loops in build_core_density and build_local_ionic_potential_on_G_total" — grep: those functions no longer exist in lorrax_D (cleanly deleted; docstring reference is stale).

---

## src/psp/gvec_utils.py (93 loc)

**Purpose.** Small host-side numpy helpers for plane-wave G-vector bookkeeping in the standalone NSCF/KPM solvers: master ecutrho G-list in QE ordering, per-k G selection, ngkmax, and Davidson→QE coefficient reordering. Category: **preprocessing/utility: PW basis bookkeeping (NSCF solver support)**.

**Function table**
| function | role |
|---|---|
| `build_master_gvec_list(crystal)` | ecutrho-filtered G list sorted by discretized |G|² then lexicographic (QE convention) |
| `select_gvecs_for_k(kvec, G_master, bdot, ecutwfc)` | per-k sphere mask on master list |
| `compute_ngkmax(kpoints, bdot, ecutwfc, fft_grid)` | max nG over k for uniform JIT padding |
| `reorder_to_qe(evecs_np, H_k, Gk_qe)` | Davidson (|k+G|²-sorted) → QE master-order coefficient permutation via python dict |

**Entry points / callers.**
- `build_master_gvec_list` ← `psp/run_nscf.py:83`, `psp/kpm_dos.py:186`.
- `select_gvecs_for_k` ← `psp/run_nscf.py:86`.
- `compute_ngkmax` ← `psp/run_nscf.py:85`, `psp/kpm_dos.py:188` (this copy); `psp/run_sternheimer.py:1150` and orbmag import the **duplicate in dft_operators.py:186**.
- `reorder_to_qe` ← `psp/run_nscf.py:117,235`.

**Flags consumed.** None.

**I/O.** None.

**Cross-module deps.** numpy only; consumes a `crystal` object (fft_grid, bdot, ecutrho) and `HamiltonianK` (Gx/Gy/Gz/mask).

**Redundancy suspects.**
- `compute_ngkmax` is duplicated **verbatim** (identical body, ~15 lines) in `src/psp/dft_operators.py:186`. Two live copies with disjoint caller sets (gvec_utils copy: run_nscf, kpm_dos; dft_operators copy: run_sternheimer, setup_H_k docs, orbmag). Classic fetch_X/fetch_X_dyn-style cruft — delete one.

**Weird code.**
- L31: `G2_int = np.round(G2_f * 1e8)` — magic 1e8 discretization so lexsort groups |G|² shells despite float noise; correctness depends on ecutrho scale.
- `reorder_to_qe` raises bare KeyError if a QE G-vector is absent from the Davidson mask — no diagnostic.

---

## src/psp/orbital_magnetization.py — **ABSENT from lorrax_D**

**Status.** Assigned path `src/psp/orbital_magnetization.py` does **not exist** in lorrax_D (checked worktree + full `git log --all --name-only`: no branch in the D clone ever contained it). The live copy is at:
- `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B_orbmag_wt/src/psp/orbital_magnetization.py` (687 loc, + `orbital_magnetization_THEORY.md` alongside)
- report copy: `/pscratch/sd/j/jackm/lorrax_sandbox/reports/B_orbital_magnetization_cri3_2026-06-16/orbital_magnetization_B.py`

It is an unmerged lorrax_B worktree feature. Notes below refer to the B-worktree copy; **refactor-map implication: merge or explicitly orphan this file**.

**Purpose.** Standalone CLI (`python -m psp.orbital_magnetization --wfn WFN.h5`) computing the per-cell orbital magnetic moment of a spinor (SOC) crystal via the modern theory, with three evaluation routes: full-BZ sum-over-states, IBZ + magnetic-point-group axial-vector symmetrization, and band-sum-free Sternheimer covariant derivative. Uses analytic dH/dk = 2(k+G) + dV_NL/dk; also reports the spin moment ⟨σ_z⟩ as calibration. Category: **physics: magnetism diagnostic / postprocessing tool**.

**Function table**
| function | role |
|---|---|
| `velocity_at_k(wfn, sym, meta, vnl_setup, ik, nb)` | per-k (v_kin, v_nl, eps, sz) via FFT-box load + `compute_p_operator_k`/`compute_vnl_velocity_cart` from get_dipole_mtxels |
| `orbital_pieces_at_k(v, eps, nocc, deps_tol)` | μ-independent SOS summand pieces PA/PB (3,nb,nb): cross_g = ε_gab v^a v^b with 1/Δε² masked at deps_tol |
| `magnetic_point_group(sym, m_axis_cart, tol)` | keep spatial ops with det(R)·R@m == m (drops T and field-reversing ops) |
| `axial_projector(R_mpg, det_mpg)` | (3,3) trivial-rep projector (1/|G|)Σ det(R)·R for axial vectors |
| `run_ibz(...)` | IBZ loop on stored k (raw G-flat ψ, no unfold), accumulate weighted PA/PB, symmetrize with projector |
| `run_sternheimer_orbmag(...)` | band-sum-free route: rebuild V_scf from WFN density, solve Sternheimer for covariant derivative per occ band, m from (H+ε−2μ) sandwich |
| `hf_group_velocity_check(Vp, Vnl, eps_grid, kcrys_grid, B, kgrid)` | Hellmann-Feynman FD band-slope test choosing the kinetic/nonlocal relative sign |
| `main(argv)` | CLI dispatch of the three branches + shared μ/reporting/npz dump |

**Entry points / callers.** CLI only; zero importers anywhere in the sandbox (grep across lorrax_D src/tests/tools/scripts and lorrax_B_orbmag_wt). It *imports from* `psp.get_dipole_mtxels` (compute_p_operator_k, compute_vnl_velocity_cart) — the only external consumer of those two functions.

**Flags consumed.** `--wfn` (required), `--nbnd`, `--nocc`, `--mu`, `--mu-scan`, `--deps-tol` (default 1.4e-3 eV), `--pseudo-dir`, `--vnl-sign {auto,plus,minus}`, `--skip-vnl`, `--ibz`, `--mag-axis`, `--method {sos,sternheimer}`, `--truncation-2d` (default True — monolayer CrI3 baked in), `--cpu`, `--convergence`, `--per-band`, `--out`. Env: JAX_ENABLE_X64, JAX_PLATFORMS=cpu + OMP_NUM_THREADS=32 for `--cpu`.

**I/O.** Reads `WFN.h5` (BGW), `*.upf` (auto-discover near WFN or `--pseudo-dir`). Writes optional `--out` .npz (E, mu, nocc, m_orb, m_spin_z, mode, colA_z/colB_z + branch extras Vp/Vnl/SZ/Kc/cA/cB). Console report is the primary output.

**Cross-module deps.** `file_io.WfnLoader`, `common.symmetry_maps`, `common.Meta`, `common.load_wfns.load_kpoint_fftbox`, `psp.dft_operators` (generate_gvectors_k, gather_psi_G_from_crys, momentum_matrix_k, setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax — the dft_operators duplicate), `psp.get_dipole_mtxels`, `psp.vnl_ops`, `psp.pseudos`, `psp.run_sternheimer` (_psi_box_to_G_sphere — private-name import across modules, compute_kp_tangent_at_kvec), `psp.scf_potential`, `solvers.sternheimer_precond`.

**Weird code.**
- Sign machinery: `MU_B_PREFACTOR = 0.5` with "sign handled below"; final `m_orb = -pref * Im C(μ)`; spin moment returned as `-1.0 * S_sum`; `frame = ±1` flip to report parallel/antiparallel to spin. Four sign sites for one convention.
- `--vnl-sign auto` heuristic: HF check with hard-coded `hf[-1] < 0.8 * hf[1]` margin, and long comment admitting the diagonal test is ~sign-blind (vNL ~pure off-diagonal, "ratio 1.000") — the auto mode effectively always chooses "plus". Interacts with get_dipole_mtxels' BGW-convention flip (this script uses the *unflipped* `vnl_velocity_matrix`, i.e. physical +dV_NL/dk).
- `hf_group_velocity_check` FD steps only in-plane (`steps = [(0,nkx),(1,nky)]` with comment "kz single layer") — 2D-monolayer assumption hard-coded; magic thresholds 2e-3 Ry degeneracy, 0.02 dispersion floor, 200-band cap.
- `sz` computed as |c_up|²−|c_dn|² assuming spinor axis ordering and z quantization — fine for the CrI3 use case, unlabeled assumption otherwise.
- `run_ibz` L187: silently renormalizes kweights if they don't sum to 1.
- Imports `run_sternheimer._psi_box_to_G_sphere` (underscore-private) — cross-module private dependency.
- Duplicated per-k pattern with get_dipole_mtxels' main loop (FFT-box load → p + vNL) — a merge should share one velocity-assembly helper.

**Dead suspects.** The whole file is orphaned relative to lorrax_D main (worktree-only, no in-repo callers) — "dead" in the sense of unmerged, not unused (it was the driver for the CrI3 orbmag reports of 2026-06-16).
