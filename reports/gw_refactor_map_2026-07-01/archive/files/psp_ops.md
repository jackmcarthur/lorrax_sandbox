# psp DFT-operator group — refactor map notes

Group: `src/psp/{dft_operators, get_DFT_mtxels, vnl_ops, h_dft, dft_precond, xc, scf_potential, operator_checks}.py`
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. Caller greps ran over `src/`, `tests/`, `tools/`, `scripts/` (top-level `tests/`, `tools/`, `scripts/` contain essentially no psp consumers; only `tests/archive/plot_*.py` touch `psp.load_upf`).

Big-picture: this group is the **plane-wave DFT Hamiltonian layer** — build V_scf (V_loc+V_H+V_xc), Kleinman–Bylander V_NL, apply H|ψ⟩, ⟨m|H|n⟩ matrices, velocity/dipole operators. Consumers: Davidson NSCF (`psp/run_nscf.py`), Sternheimer (`psp/run_sternheimer.py`, `solvers/sternheimer_solve.py`), KPM DOS (`psp/kpm_dos.py`), dipole driver (`psp/get_dipole_mtxels.py`), GW kin+ion I/O (`gw/kin_ion_io.py`), centroid density (`centroid/charge_density.py`). Core `gw.gw_jax` mostly does NOT go through here (memory note: V_xc here is spin-blind; core GW unaffected).

---

## 1. src/psp/dft_operators.py (1138 loc)

**Purpose.** Canonical module for the plane-wave DFT Hamiltonian: `HamiltonianK` dataclass, per-component builders (T_diag, V_scf, h_diag, VNL kdata), fused JIT kernels `apply_H_k` / `apply_H_k_from_G` / `build_matrix_k`, Poisson solver with Ismail-Beigi 2D truncation, and velocity/dipole (dH/dk) machinery. Two build paths: SymMaps/WFN.h5 (`setup_H_k`) and standalone k-vector (`setup_H_k_from_kvec`, used by Davidson/Sternheimer with ngkmax padding for one-JIT-for-all-k).

**Category.** physics: DFT Hamiltonian operators (H-apply / velocity kernels).

**Function table.**
| function | role | callers |
|---|---|---|
| `HamiltonianK` (dataclass) | per-k H data bundle | h_dft, kpm_dos, run_nscf/run_sternheimer via setup_* |
| `poisson_potential_from_rhoG` | V_H(G)=8πρ/G², optional 2D slab truncation | compute_V_H_and_V_xc, get_DFT_mtxels.compute_hartree_potential_real (re-export), test_dft_hamiltonian |
| `generate_gvectors_k` | per-k G-list via WfnLoader (post-P5: legacy WFNReader fallback rebuilds loader) | get_DFT_mtxels, get_dipole_mtxels, gw/kin_ion_io, vnl_ops, internal |
| `build_G_cart` | Cartesian G grid | scf_potential, kpm_dos, test_dft_hamiltonian |
| `compute_ngkmax` | max nG over k (padding size) | run_sternheimer (NOTE: run_nscf & kpm_dos import a *different* `compute_ngkmax` from `psp.gvec_utils`!) |
| `build_T_diag` | \|k+G\|² via SymMaps | setup_H_k only |
| `build_T_diag_from_kvec` | \|k+G\|² from cutoff sphere | setup_H_k_from_kvec only |
| `_T_diag_from_G` | shared core | both above |
| `compute_V_H_and_V_xc` (@jit) | Poisson + PBE V_xc | scf_potential.build_dft_potentials, kpm_dos, get_DFT_mtxels(indirect), test |
| `build_V_scf` | V_loc+V_H+V_xc sum | scf_potential, kpm_dos, test |
| `build_vnl_kdata` | thin wrapper → vnl_ops.build_vnl_kdata | **no callers** (dead) |
| `gather_psi_G` (@jit) | FFT-box → G-sphere gather | gather_psi_G_from_crys |
| `gather_psi_G_from_crys` | int G-list wrapper | get_dipole_mtxels (x4), vnl_matrix_from_kdata |
| `vnl_matrix_from_kdata` | ⟨m\|V_NL\|n⟩ from box+kdata; slices bispinor 4→2 spinors | get_DFT_mtxels, gw/kin_ion_io |
| `vnl_velocity_from_kdata` | dV_NL/dk from box+kdata | **no callers** (dead) |
| `build_h_diag` | Davidson precond diagonal (QE g_psi) | setup_H_k, setup_H_k_from_kvec, re-exported by h_dft |
| `setup_H_k` | assemble HamiltonianK (SymMaps) | test_dft_hamiltonian only |
| `setup_H_k_from_kvec` | assemble HamiltonianK (standalone, ngkmax pad) | run_sternheimer, run_nscf (via h_dft re-export), kpm_dos, test_sternheimer_jvp |
| `apply_H_k` (@jit, donate psi_box) | fused H\|ψ⟩ box→G | h_dft._apply_H_sparse, `apply` |
| `apply` | HamiltonianK convenience wrapper | **no callers** (dead) |
| `apply_H_k_from_G` (@jit) | H\|ψ⟩ G-sphere→G-sphere (saves scatter/gather; `.add()` not `.set()` for padded-G collision at (0,0,0)) | solvers/sternheimer_solve |
| `build_matrix_k` (@jit) | full ⟨m\|H\|n⟩ | `matrix` |
| `matrix` | convenience wrapper | test_dft_hamiltonian only |
| `VNLChannelData` + `extract_vnl_channel_data` | autodiff-ready channel extraction from "projector plan" dict | compute_dipole_all only |
| `_build_reduced_tables` | G_l=F_l/q^l + FD derivative | extract_vnl_channel_data only |
| `_build_Z_channel_jax`, `_radial_times_solid_harm{,_impl,_jvp}` | k-traceable KB projector w/ custom JVP | vnl_matrix_at_k only |
| `vnl_matrix_at_k` | k-traceable V_NL matrix | vnl_velocity_autodiff only |
| `build_Z_and_dZ` | precompute Z, dZ per channel | velocity_matrix_k, compute_dipole_all |
| `vnl_velocity_from_dZ` | velocity from precomputed Z/dZ | velocity_matrix_k only |
| `vnl_velocity_autodiff` | jacfwd velocity (verification path) | **no callers** (dead) |
| `apply_kinetic_velocity_to_ket` (@jit) | 2(k+G)ψ | momentum_matrix_k; get_dipole_mtxels (SOS path) |
| `momentum_matrix_k` (@jit) | kinetic velocity matrix | velocity_matrix_k; get_dipole_mtxels |
| `velocity_matrix_k` | full dH/dk = p + dV_NL/dk (physical sign) | compute_dipole_all only |
| `compute_dipole_all` | batch velocity mtxels all k | **no callers in this checkout** (dead; orbital_magnetization tool lives outside lorrax_D) |

**I/O.** None directly (reads WFN.h5 indirectly through `file_io.WfnLoader` in `generate_gvectors_k`).

**Flags consumed.** None (pure library).

**Dead suspects (grep evidence: `grep -rn "\bNAME\b" src tests tools scripts`):**
- `apply` — only its own definition and docstrings.
- `build_vnl_kdata` (dft_operators wrapper, line 351) — no callers; and it's the ONLY caller of `vnl_ops.build_vnl_kdata`, so both are dead as a chain.
- `vnl_velocity_from_kdata` — no callers.
- `compute_dipole_all`, `velocity_matrix_k`, `vnl_velocity_autodiff`, `vnl_matrix_at_k`, `vnl_velocity_from_dZ`, `build_Z_and_dZ`, `extract_vnl_channel_data`, `VNLChannelData`, `_build_reduced_tables`, `_build_Z_channel_jax`, `_radial_times_solid_harm*` — the entire "Autodiff-compatible V_NL" section (lines ~753–1138 minus the two kinetic helpers) has zero external callers in lorrax_D. `get_dipole_mtxels` uses the parallel `vnl_ops` dZ path instead. Caveat: memory notes call `velocity_matrix_k` the "canonical physical dH/dk" used by an orbital-magnetization tool — that tool is not in this checkout.
- `matrix`, `setup_H_k` — test-only callers (`src/psp/tests/test_dft_hamiltonian.py`).

**Redundancy suspects:**
- **Two full VNL implementations**: the autodiff section here (`VNLChannelData`/`extract_vnl_channel_data`/`_build_Z_channel_jax`/`build_Z_and_dZ`) duplicates `vnl_ops.py` (`VNLSetup`/`_assemble_Z_jit`/`compute_dZ` path). Same math (G_l=F_l/q^l reduction, solid harmonics, phase, dphase, custom JVP) written twice with different data structures ("projector plan" dict vs VNLSetup dataclass) and different derivative tables (FD `differentiate_uniform_table` here vs analytic Bessel-recurrence in vnl_ops — vnl_ops comments say the FD version has ~10% velocity errors!).
- `generate_gvectors_k` legacy WFNReader fallback rebuilding a WfnLoader from `wfn._filename` (kept "for tests").
- `compute_ngkmax` exists here AND in `psp/gvec_utils.py`; run_sternheimer uses this one, run_nscf/kpm_dos use gvec_utils'.
- `setup_H_k` vs `setup_H_k_from_kvec`: ~70 lines of copy-paste padding/masking logic differing only in T_diag source.
- `apply_H_k` vs `apply_H_k_from_G` vs `build_matrix_k` share the T/V_scf/V_NL triple with copy-pasted bodies.

**Weird code:**
- `poisson_potential_from_rhoG` default `truncation_2d=True` — a silently dangerous default for a bulk code; callers must remember to pass False for 3D. (`compute_hartree_potential_real` in get_DFT_mtxels defaults it False.)
- 2D truncation `zc = jnp.pi / B[2,2]` assumes c-axis ⟂ ab-plane and z-diagonal B.
- Magic constants: `1e10` padding sentinel for T_diag/h_diag; `_EPS2 = 1e-60` q-regularizer here vs `1e-8` in vnl_ops `_assemble_Z_jit` — same purpose, 52 orders of magnitude apart (1e-8 in q² shifts q by ~1e-4 at small q; possibly deliberate but undocumented divergence between paths).
- `compute_dipole_all` energies indexing: `energies[0, k_red, :nb]` with try/except fallback `k_red = ik` — silently wrong if `irr_idx_k` missing.
- Comment says velocity = "2(k+G)" (Rydberg factor-of-2 convention) — sign/convention trap vs the dipole driver's p−vNL BGW convention (see get_dipole_mtxels).

---

## 2. src/psp/get_DFT_mtxels.py (995 loc)

**Purpose.** Legacy-ish driver + helper library for DFT Hamiltonian matrix elements ⟨mk|T+V_ion+V_H+V_NL|nk⟩: valence density from occupied states, Hartree potential, per-k kinetic/local-V matrix element kernels, `get_kin_ion` (nk,nb,nb) builder consumed by GW's exchange pipeline, and a standalone `main()` diagnostic that compares k=0 diagonals against a `k0_diag_check` reference. The reusable parts (`compute_kinetic_k`, `compute_local_V_k`, `compute_valence_density`, `get_kin_ion`) are imported by `gw/kin_ion_io.py` and `centroid/charge_density.py`.

**Category.** mixed: physics library (H matrix elements / density) + stale diagnostic driver.

**Function table.**
| function | role | callers |
|---|---|---|
| `_wfn_loader_for_path` (@lru_cache) / `_gvecs_full_cache` / `_ngk_full_cache` | per-path WfnLoader G-vec cache | compute_valence_density |
| `read_cohsex_input` | parse [cohsex] INI stripping K_POINTS block | own main, get_dipole_mtxels, gw/kin_ion_io, run_sternheimer |
| `get_bandranges` | band-range tuples | own main only (gw uses `gw.gw_init.get_bandranges`) |
| re-exports from psp.pseudos: `load_pseudopotentials`, `_symbol_to_Z`, `AtomPP`, `build_atom_pp_assignments`, `print_atomic_structure` | back-compat shims | various |
| `compute_valence_density` | ρ_val(r) from occupied bands, optional ecutrho pad grid | get_H_matrix_elements, get_kin_ion, centroid/charge_density |
| `compute_core_density` | **placeholder returning zeros, TODO body** | only a commented-out call site (line 541) |
| `compute_hartree_potential_real` | thin wrapper on poisson solver | get_H_matrix_elements, get_kin_ion |
| `compute_kinetic_k` / `_compute_kinetic_k_jit` | ⟨m\|T\|n⟩ per k | get_H_matrix_elements, get_kin_ion, gw/kin_ion_io.get_kin_ion_k |
| `compute_local_V_k` / `_compute_local_V_k_jit` | ⟨m\|V_loc-ish\|n⟩ per k | same as above |
| `get_H_matrix_elements` | full H build — **only loops `for i in range(1)` (k=0 debug only)** | main only |
| `get_kin_ion` | (nk,nb,nb) T+V_ion(+V_H)+V_NL | main; (gw pipeline uses kin_ion_io's chunked variant) |
| `write_kin_ion_h5` | write kin_ion.h5 | main only |
| `main` | standalone CLI driver + k0_diag comparison | `python -m psp.get_DFT_mtxels` (no script/docs references found) |

**I/O.**
- Reads: WFN.h5 (via WFNReader/WfnLoader), `cohsex.in`-style INI, `*.upf` pseudos from input dir, optional `k0_diag_check[.txt]` reference.
- Writes: `k0_diag.txt` (band-diagonal table), `kin_ion.h5` (dataset `kin_ion`, (nk,nb,nb) complex).

**Flags consumed** (from `[cohsex]` section): `wfn_file`, `nval`, `ncond`, `nband`, `bispinor`, `ecutrho` (parsed as `ecutrho_eV` but "field is actually Ry despite the name"), `sys_dim`.

**Dead suspects:**
- `compute_core_density` — TODO placeholder returning zeros; only call site is commented out (line 541). Grep: no other callers.
- `get_H_matrix_elements` / `write_kin_ion_h5` / `get_bandranges` / `main` — only reachable via `main()`; nothing in src/tests/tools/scripts invokes the module as a program except its own `__main__`. The whole driver looks superseded by `gw/kin_ion_io.py`.
- Module-level env-var setup (JAX_PLATFORMS etc., lines 21–28) — driver-era side effects that fire on *import* by gw/kin_ion_io and centroid/charge_density.

**Redundancy suspects:**
- `read_cohsex_input` — docstring admits it "mirrors the robust parser used in `gw.gw_init.read_cohsex_input`" — two copies of the same parser, different fallback defaults.
- `get_bandranges` duplicated in `gw/gw_init.py` (gw/__init__ exports that one).
- `get_kin_ion` here vs `gw/kin_ion_io.get_kin_ion_k` + its chunked main — parallel old/new kin+ion paths; kin_ion_io imports the per-k kernels from here but re-implements the k-loop.
- `compute_hartree_potential_real` is a 3-line wrapper over the re-exported `poisson_potential_from_rhoG`.
- G-pad/mask construction (lines 490–509 and 668–687) copy-pasted twice within the file, and again in dft_operators.setup_H_k*.
- Try/except dual import block (relative vs sys.path hack) for `python -m` vs direct execution.

**Weird code:**
- **Unreachable code** in `_compute_local_V_k_jit` (lines 417–418): a `raise NotImplementedError(...)` AFTER the `return` — leftover from removed `compute_V_NL_k` legacy path, currently dead text inside the function body.
- `compute_local_V_k` normalization gymnastics: `scale = sqrt(ngrid/vol)`, `deltaV*fft_norm` post-factor, then final `* jnp.sqrt(1.0/volume)` — three separate volume factors that presumably compose to ortho convention; single most confusing normalization stack in the group.
- Hardcoded `truncation_2d=True` in `get_H_matrix_elements` ("Match QE's assume_isolated='2D'") but `truncation_2d=False` in `get_kin_ion`'s Hartree branch with comment "Set to True for 2D slab truncation matching ISDF" — inconsistent, non-flag-driven truncation choices; `get_kin_ion`'s V_loc correctly uses `ctx.truncation_2d` from sys_dim but its V_H does not.
- `ecutrho_eV` field actually in Ry (line 850 comment) and only ever used to set grid_rho = 2× wfn grid regardless of value.
- `main()`'s k0_diag_check block: two least-squares fits (constrained b,d then unconstrained A..E) of reference diagonals against component columns — debugging archaeology, ~60 lines.
- `for i in range(1)` in get_H_matrix_elements — "first k-point for debug" hardwired.

---

## 3. src/psp/vnl_ops.py (526 loc)

**Purpose.** Production Kleinman–Bylander nonlocal-potential engine: build k-independent `VNLSetup` (radial tables on uniform q-grid, flattened per-row metadata, block-diagonal E_super), assemble dense per-k projectors Z (single vectorized JIT `_assemble_Z_jit`), and apply/matrix/velocity einsum kernels. Analytic dG_l/dq via Bessel recurrence with `_interp_with_deriv` custom-JVP so `jax.jvp` through form factors gives physical derivatives.

**Category.** physics: nonlocal pseudopotential operator kernels.

**Function table.**
| function | role | callers |
|---|---|---|
| `ChannelMeta`, `VNLSetup`, `VNLKData` | dataclasses | throughout |
| `build_vnl_setup` | k-independent setup (tables, E_super, row metadata) | scf_potential, get_DFT_mtxels (x2), get_dipole_mtxels, kpm_dos, gw/kin_ion_io, test_dft_hamiltonian |
| `_table_interp` (@jit) | linear interp on uniform grid | internal |
| `_interp_with_deriv` (@custom_jvp) | interp with physical-derivative tangent | _assemble_Z_jit, _build_vnl_kdata_core |
| `build_vnl_kdata` | SymMaps-path per-k Z | ONLY dft_operators.build_vnl_kdata (itself dead) |
| `build_vnl_kdata_from_kvec` | explicit-k per-k Z (+dZ) | get_DFT_mtxels, get_dipole_mtxels, dft_operators.setup_H_k*, gw/kin_ion_io |
| `_assemble_Z_jit` (@jit, static l_max) | vectorized Z assembly, module-scope for shared compile cache | _build_vnl_kdata_core |
| `_build_vnl_kdata_core` | dispatch: jit Z path or eager Z+dZ path | build_vnl_kdata*, run_sternheimer (imports `_build_vnl_kdata_core` directly — private-name leak) |
| `apply_vnl` (@jit) | V_NL\|ψ⟩ = Z E Z†ψ | **no callers** (apply_H_k inlines the same einsums) |
| `vnl_matrix` (@jit) | ⟨m\|V_NL\|n⟩ | dft_operators (vnl_matrix_from_kdata, build_matrix_k), get_dipole_mtxels |
| `apply_vnl_velocity_to_ket` (@jit) | ∂V_NL/∂k applied to ket (bra-free, for finite-q SOS) | vnl_velocity_matrix, get_dipole_mtxels (x2, with **minus sign**) |
| `vnl_velocity_matrix` (@jit) | (3,nb,nb) velocity mtxels | dft_operators.vnl_velocity_from_kdata (dead), get_dipole_mtxels |

**I/O.** None (consumes parsed UPF objects via psp.species / psp.radial_tables).

**Dead suspects:**
- `apply_vnl` — grep over src/tests/tools/scripts: definition + docstring mentions only. `apply_H_k`/`apply_H_k_from_G` re-inline the identical three einsums rather than calling it.
- `build_vnl_kdata` (SymMaps path) — sole caller is the dead `dft_operators.build_vnl_kdata` wrapper.

**Redundancy suspects:**
- Entire dZ/velocity machinery duplicated in dft_operators' autodiff section (see above). vnl_ops is the production one (analytic derivative table); dft_operators' is the FD-table one flagged in comments as ~10% biased.
- The `compute_dZ=True` branch of `_build_vnl_kdata_core` recomputes Z eagerly (doesn't reuse `_assemble_Z_jit`) — near-duplicate Z assembly within one file, marked TODO.
- P/D/unproject einsum triple appears in: apply_vnl, vnl_matrix, apply_H_k, apply_H_k_from_G, build_matrix_k, vnl_matrix_at_k — six inlined copies of Z E Z†.

**Weird code:**
- q regularizer `jnp.sqrt(sum(K²) + 1e-8)` (lines 363, 416) vs `1e-60` in dft_operators — inconsistent magic constant, same physics.
- `q_max *= 1.01` fudge factor (line 132); scf_potential independently applies `*1.01` at its call site (double-application risk if both paths ever combine).
- Tail-G contract: padded Z entries computed at K=kvec are *finite garbage*; correctness relies on every caller masking afterwards (documented but fragile — three call sites each re-implement the masking).
- `compute_dZ` TODO comments (lines 409–410, 428): eager per-channel loop "TODO: jit this too once the per-channel for-loop is vectorised".
- `run_sternheimer.py:72` imports the underscore-private `_build_vnl_kdata_core`.
- `ChannelMeta.E` documented as "(2, 2, R, R)" but sliced `[:nspinor,:nspinor]` — hardcoded max-2-spinor assumption baked into the comment.

---

## 4. src/psp/h_dft.py (52 loc)

**Purpose.** Tiny adapter exposing the DFT Hamiltonian as a black-box callable for Davidson/solvers: `make_apply_H(H_k)` returns sparse-G → sparse-G ψ↦Hψ using one shared JIT kernel (`_apply_H_sparse` scatters G-sphere → FFT box then calls `apply_H_k`).

**Category.** glue: solver-facing operator adapter.

**Functions.** `_apply_H_sparse` (jit, static grid dims; scatter + apply_H_k); `make_apply_H` (closure factory). Re-exports `setup_H_k_from_kvec`, `build_h_diag`, `HamiltonianK` in `__all__`.

**Callers.** `make_apply_H` ← run_nscf, run_sternheimer, kpm_dos. `setup_H_k_from_kvec` re-export ← run_nscf (imports it from h_dft while run_sternheimer imports the same symbol from dft_operators — inconsistent import origin).

**I/O / flags.** None.

**Dead suspects.** None (all used), though the re-export layer itself is redundant indirection.

**Redundancy suspects.** `_apply_H_sparse` = scatter + `apply_H_k` (box path: scatter→gather→FFT) while `apply_H_k_from_G` in dft_operators is the same operation done smarter (single scatter). Two sparse-G H-apply entry points; the h_dft one pays an extra round trip and could be replaced by `apply_H_k_from_G`.

**Weird code.** `.add()` scatter (line 25) for the padded-G (0,0,0) collision — same subtle convention as apply_H_k_from_G, documented only in the latter.

---

## 5. src/psp/dft_precond.py (83 loc)

**Purpose.** DFT-specific plug-ins for the generic Davidson solver: `make_dft_preconditioner(h_diag)` (QE g_psi diagonal preconditioner) and `make_pw_init(T_diag, n_channels)` (lowest-|k+G|² plane-wave subspace initial guess).

**Category.** numerics: eigensolver preconditioning/init.

**Functions.** `make_dft_preconditioner` ← run_nscf:226; `make_pw_init` ← run_nscf:227. Both closure factories returning @jit callables.

**I/O / flags.** None.

**Dead suspects.** None; but the only consumer is run_nscf (Davidson). Single-caller module.

**Redundancy suspects.** None internal.

**Weird code.** `_EPS = 1e-2` denominator clamp and the double-guard `where(|denom|<EPS)` then `where(denom==0)` — second where is unreachable given the first (sign(0)*EPS=0 case is the only path, i.e. denom exactly 0 → first where gives 0 → caught by second; subtle but intentional-looking); norm floor `1e-30` magic.

---

## 6. src/psp/xc.py (169 loc)

**Purpose.** Exchange-correlation potential via autodiff: functional ε_xc(ρ[,σ[,τ]]) → V_xc(r) with one code path for LDA/GGA/mGGA; GGA divergence correction −2∇·(∂f/∂σ ∇ρ) done in G-space. PBE via `jax_xc_local.pbe`.

**Category.** physics: XC functional evaluation.

**Functions.** `XCLevel` enum; `pbe_functional` ← dft_operators.compute_V_H_and_V_xc; `_compute_sigma`, `_compute_grad_components`; `compute_V_xc` ← dft_operators.compute_V_H_and_V_xc (only external caller); `_vxc_lda`, `_vxc_gga`, `_vxc_mgga`.

**I/O / flags.** None.

**Dead suspects:**
- `_compute_grad_components` — grep: defined, never called (the GGA divergence loop recomputes drho_i inline).
- `_vxc_mgga` — reachable only via `XCLevel.MGGA`, which nothing constructs (only `pbe_functional` → GGA exists); τ is a zeros placeholder anyway.
- `_vxc_lda` — reachable only via XCLevel.LDA, never constructed in-repo.

**Redundancy suspects.** GGA divergence loop copy-pasted between `_vxc_gga` and `_vxc_mgga`.

**Weird code:**
- **Spin-blind** (known project issue): no spin-polarized branch at all; V_xc wrong for magnetic systems, omits B_xc (memory: 1.4 eV H-residual on CrI3 vs 0.2 meV MoS2).
- Magic masking constants: `rho = max(rho, 1e-10)`; LDA-fallback mask `(rho_raw > 1e-6) & (sigma > 1e-10)` — silent LDA substitution in low-density regions.
- mGGA τ term added as multiplicative potential (`+ df_dtau`) with a comment admitting the non-multiplicative Hamiltonian part is missing.

---

## 7. src/psp/scf_potential.py (169 loc)

**Purpose.** Build the DFT self-consistent potential from scratch (no QE potential file): `build_dft_potentials(mf, pseudos, rho_val)` → (V_scf, V_loc, vnl_setup); `build_rho_val_from_wfn` → ρ_val(r) by full-BZ sum over occupied bands (deliberately avoids IBZ+symmetrise, whose `_symmetrise_density` in `psp/archive/charge_density.py` is documented broken). Lifted out of run_nscf so Sternheimer shares it.

**Category.** physics: SCF potential reconstruction (standalone-DFT path).

**Functions.** `build_dft_potentials` ← run_nscf._build_potentials wrapper, run_sternheimer:1122, test_sternheimer_jvp; `build_rho_val_from_wfn` ← run_sternheimer:1119, test_sternheimer_jvp.

**I/O.** Reads WFN.h5 indirectly via `common.load_wfns.load_kpoint_fftbox` (full-BZ ψ per k). No writes.

**Flags.** `truncation_2d` (keyword, caller-supplied).

**Dead suspects.** None.

**Redundancy suspects:**
- `build_rho_val_from_wfn` vs `get_DFT_mtxels.compute_valence_density` vs `psp/archive/charge_density.build_density_from_ibz` — THREE valence-density builders. This one loops full-BZ `load_kpoint_fftbox`; compute_valence_density loops a pre-sharded array with kweights logic and ecutrho pad-grid support. Different normalizations (spin_factor·N_grid/vol here vs sqrt(ngrid/vol)² there).
- `q_max=sqrt(ecutwfc)*1.01` at line 95 duplicates the internal `*1.01` logic of build_vnl_setup.

**Weird code:** insulator-only assumption (`n_occ` fixed per k, no occupations); spin_factor=2/1 by nspinor with no nspin=2 (spin-polarized) case — same spin-blindness as xc.py.

---

## 8. src/psp/operator_checks.py (114 loc)

**Purpose.** Pre-flight validation before operator matrix-element computation: pseudos non-empty, every atomic species covered by a loaded PP, sys_dim ∈ {0,2,3} → derives `truncation_2d = (sys_dim==2)`. Returns frozen `OperatorContext`.

**Category.** validation/guard utility.

**Functions.** `OperatorContext` (frozen dataclass); `validate_operator_inputs` ← gw/kin_ion_io:129, get_DFT_mtxels.get_kin_ion:629.

**I/O / flags.** None directly; interprets `sys_dim`.

**Dead suspects.** None.

**Redundancy suspects.** None; small and single-purpose. (`ctx.pseudos` is just the input dict passed back — mild API padding.)

**Weird code:** `sys_dim=0` (molecule) maps to `truncation_2d=False`, i.e. molecules get *bulk* Coulomb — accepted-but-wrong dimensionality handling (no 0D truncation exists); callers that hardcode truncation_2d bypass this check entirely (get_H_matrix_elements, get_kin_ion's Hartree branch).

---

## Cross-cutting refactor targets for this group

1. **Delete/merge the duplicate VNL velocity stack**: dft_operators lines ~753–1138 vs vnl_ops — keep vnl_ops (analytic derivatives), port `velocity_matrix_k`'s physical-sign wrapper if the orbital-mag tool needs it.
2. **One kin+ion path**: retire get_DFT_mtxels.main/get_H_matrix_elements/get_kin_ion in favor of gw/kin_ion_io; move `compute_kinetic_k`/`compute_local_V_k`/`compute_valence_density` into dft_operators or a small mtxels module without the driver's import-time env-var side effects.
3. **One cohsex parser, one get_bandranges** (gw.gw_init already canonical).
4. **One valence-density builder** (three exist).
5. **Unify truncation_2d plumbing** through operator_checks instead of hardcoded booleans.
6. **Unify q-regularizer constant** (1e-8 vs 1e-60) and the padding sentinel/mask idiom (5+ copies).
7. Fix `run_sternheimer`'s import of private `_build_vnl_kdata_core`; unify `compute_ngkmax` (dft_operators vs gvec_utils).
