# src/bse/absorption_{common,eigvecs,haydock}.py + eigvals_to_eps2.py — deep-read notes (958 LOC)

Audit date: 2026-07-15, lorrax_D checkout. Nominal base: agent/slate-linalg-ffi
@ e18d0e5; working HEAD during audit was adc2197 (agent/ppm-fit-conditioning).
`git diff e18d0e5 adc2197 -- src/bse/absorption_*.py src/bse/eigvals_to_eps2.py`
is **empty** — notes valid for both.

## Purpose

The BSE absorption-spectrum output stage: turn a solved (or implicitly
solved) TDA BSE Hamiltonian into ε₂(ω)/ε₁(ω)/JDOS curves and BGW-format
`.dat` files. Four files, three routes:

- `absorption_common.py` — shared host-side toolbox: h5 readers
  (eigenvectors.h5, dipole.h5), dipole window slicing, Lorentzian
  broadening, naive Kramers-Kronig, BGW-format `.dat`/`.h5` writers.
- `absorption_eigvecs.py` — sum-over-states route from explicit Ritz
  vectors (mirrors BGW `BSE/diag.f90 → absp.f90`). **Pure numpy, no jax
  import, single process.**
- `absorption_haydock.py` — eigenvector-free continued-fraction route
  (mirrors BGW `BSE/haydock.f90 → iterate.f90 → absh.f90`). Runs the
  sharded BSE matvec under jit; the only device-heavy file of the four.
- `eigvals_to_eps2.py` — re-broaden any BGW-format `eigenvalues.dat`
  (BGW's or ours) at arbitrary η / truncation; the canonical
  "fair-comparison" tool of `BGW_COMPARE.md`.

Physics as written in code (explicit indices):

```
# absorption_common.slice_dipole_to_bse_window (97-101), r-form dipole:
#   v_cv[α,k,c,v] = dipole_cart[α, k, n_occ+c, (n_occ-n_val)+v]        c=0..nc-1, v=0..nv-1
#   de_cv[k,c,v]  = deltaE[k, n_occ+c, (n_occ-n_val)+v]  (= E_c - E_v, Ry, > 0)
#   d_alpha[α,k,c,v] = v_cv[α,k,c,v] / de_cv[k,c,v]
# NOTE v axis is LORRAX-internal lowest-first (v=0 = deepest valence in window).

# absorption_eigvecs.compute_dipole_projections (47):
#   einsum("Nkcv,akcv->Na"):  proj[S,α] = Σ_{k,c,v} A[S,k,c,v] · d_alpha[α,k,c,v]
#   (NO conjugate on A — matches the module-docstring formula
#    ⟨0|r̂_α|S⟩ = Σ_{cvk} A^S_{cvk} d^α_{cvk}; consistent with our own
#    eigenvectors.h5 writer convention and numerically validated vs BGW,
#    STATUS.md "Results".)

# absorption_eigvecs.compute_eps2 (64-75) + absorption_common.lorentzian_broaden (119-126):
#   f_Sα        = |proj[S,α]|²
#   pref        = 16π² / (V_cell · N_k · n_spin · n_spinor)
#   ε₂^α(ω_i)  = pref · Σ_S f_Sα · (η/π) / ((ω_i − E_S)² + η²)        [all in Ry]

# absorption_haydock.haydock_recursion_block (54-94), per polarisation block α:
#   ‖d^α‖ = sqrt(Σ_{cvk} |d[α,c,v,k]|²);   s_1 = d^α/‖d^α‖,  s_0 = 0, β_0 = 0
#   loop n = 1..n_iter (lax.scan):
#     α_n = Re Σ_{cvk} conj(s_n)·(H s_n)          einsum("a...,a...->a").real
#     r   = H s_n − α_n s_n − β_{n−1} s_{n−1}
#     β_n = ‖r‖;   s_{n+1} = r / β_n              (no reorthogonalisation, no β=0 guard)

# absorption_haydock.absorption_from_haydock (97-118), backward CF (0-based code idx):
#   cf = 0;  for n = n_iter−1 .. 1:  cf ← β_{n−1}² / (z − α_n − cf),  z = ω + iη
#   g(z) = 1 / (z − α_0 − cf)
#   ε₂^α(ω) = −pref · ‖d^α‖² · Im g(ω+iη) / π      (Im g < 0 for Hermitian H ⇒ ε₂ ≥ 0)
#   [uses α_0..α_{n_iter−1} and β_0..β_{n_iter−2}; last β discarded — standard truncation]

# "JDOS" column (absorption_eigvecs.compute_jdos 78-87; haydock jdos_from_dipole 121-132):
#   jdos^α(ω_i) = pref · Σ_{cvk} |d_alpha[α,k,c,v]|² · (η/π)/((ω_i − de_cv[k,c,v])² + η²)
#   i.e. dipole-weighted INDEPENDENT-PARTICLE ε₂⁰ at (possibly DFT) ΔE — see
#   BGW-convention divergence under Suspects.

# absorption_common.kramers_kronig_eps1 (129-145), uniform grid, PV by point-exclusion:
#   ε₁(ω_i) = 1 + (2/π) · Σ_{j≠i} ω_j ε₂(ω_j) / (ω_j² − ω_i²) · Δω

# eigvals_to_eps2.compute_eps2 (74-94):
#   pref = 16π² / (V_super · n_spin · n_spinor)   with V_super = V_cell·N_k from
#   the file header (so N_k is absorbed); broadening done in Ry space, kernel
#   Lorentzian (58-61) or Gaussian (64-71): G = exp(−½((ω−E)/σ)²)/(σ√(2π)).
```

The prefactor matches BGW `BSE/absh.f90:46` verbatim
(`pref = 16.d0*PI_D**2/(mmts%vol*dble(nspin)*dble(nspinor))`, checked in
`~/software/BerkeleyGW/BSE/absh.f90`); `mmts%vol` = supercell volume =
`V_cell·N_k`. All broadening happens in **Ry**; file I/O of ω and E is in
**eV** (RYD2EV = 13.6056980659, `absorption_common.py:23`).

Category: **pipeline stage — post-BSE optics/spectra output**, with
`eigvals_to_eps2.py` doubling as a **BGW-compat diagnostic**.

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/ + skills/)

| symbol | callers (grep evidence) |
|---|---|
| `python -m bse.absorption_haydock` (`main`, 298) | documented CLI: `src/bse/STATUS.md:108`, `src/bse/BGW_COMPARE.md:49-53` (canonical cookbook); outputs consumed by `runs/Si/C_bse_davidson_profile_2026-04-28/compare/make_fair_comparison.py:32` |
| `python -m bse.absorption_eigvecs` (`main`, 90) | documented CLI: `src/bse/STATUS.md:114`, `src/bse/BGW_COMPARE.md:63-66` |
| `python -m bse.eigvals_to_eps2` (`_main`, 113) | documented CLI: `src/bse/BGW_COMPARE.md:74-78`; library imports: `runs/Si/C_bse_davidson_profile_2026-04-29/make_comparison.py:18` (`read_bgw_eigvals, compute_eps2`), `runs/Si/C_bse_davidson_profile_2026-04-28/compare/make_fair_comparison.py:25` (`compute_eps2_from_files`) |
| `absorption_common.*` | `absorption_eigvecs.py:26-36`, `absorption_haydock.py:41-49`, `src/bse/davidson_absorption.py:32` (`load_dipole_h5, slice_dipole_to_bse_window, write_eigenvalues_dat`), sandbox scripts `runs/Si/C_bse_davidson_profile_2026-04-28/compare/{dipole_isolate.py:20, absorption_compare.py:27}` (`lorentzian_broaden, slice_dipole_to_bse_window`) |
| `run_haydock` (135) | only `main` in the same file (339); no other repo callers (`grep -rn run_haydock src tests tools scripts` → this file only) |
| `load_eigenvectors_h5` | only `absorption_eigvecs.py:119` (repo-wide grep) |
| `build_dipole_vector_bse` | only `absorption_haydock.py:198` (repo-wide grep) |

`src/bse/__init__.py` is a bare docstring — no re-exports. **No pytest
coverage anywhere**: `grep -rn "absorption_\|eigvals_to_eps2" src/bse/test_bse.py
src/bse/test_davidson_bse.py tests/` → empty. These four files are exercised
only by the documented CLIs and sandbox run scripts.

## Function table

### absorption_common.py (239 LOC, host-only, numpy+h5py)

| function | lines | role |
|---|---|---|
| `load_eigenvectors_h5(path)` | 26-67 | Read BGW-format eigenvectors.h5. Asserts nQ=1, ns=1 (raise NotImplementedError, 54-56). Builds complex `A = evecs[...,0]+1j·evecs[...,1]`, drops nQ/ns axes → `(N, nk, nc, nv)`. **Flips valence axis** `A[:,:,:,::-1]` (63): BGW v=0 = highest valence → LORRAX-internal lowest-first (BGW-compat convention, see STATUS.md "Index ordering" — NOT a bug). Converts eigenvalues eV→Ry (66). Infers `nspinor = 2 if spin_kernel==3 else 1` (47). |
| `load_dipole_h5(path)` | 70-83 | Read `dipole.h5` (psp.get_dipole_mtxels): `dipole_cart (3,nk,nb,nb)` complex, `deltaE (nk,nb,nb)` = E_m−E_n (Ry), attrs {nbands, nk}. Full (nb,nb) table on host, every process. |
| `slice_dipole_to_bse_window(dipole_cart, deltaE, n_occ, n_val, n_cond)` | 86-102 | m=c ∈ [n_occ, n_occ+n_cond), n=v ∈ [n_occ−n_val, n_occ) Python slices; `d_alpha = v_cv/de_cv` (r-form). **Slices clamp silently** if dipole.h5 nb < n_occ+n_cond (see Suspects). Negative `val_lo` (n_val > n_occ) would wrap — loader clamps its own copy but callers pass CLI values. |
| `build_dipole_vector_bse(d_alpha, n_cond_pad, n_val_pad)` | 105-116 | `(3,nk,nc,nv)` → `(3,nc_pad,nv_pad,nk)` via `transpose(0,2,3,1)` into a zero block; pads c/v axes for mesh divisibility. Zero-pad = silent truncation absorber (see Suspects). |
| `lorentzian_broaden(omegas, energies, weights, eta)` | 119-126 | `L[i,j]=(η/π)/((ω_i−E_j)²+η²)`; returns `L @ weights`. Dense (n_omega × n_states) intermediate. |
| `kramers_kronig_eps1(omegas, eps2)` | 129-145 | Naive PV KK on uniform grid, Python loop O(N²), rebuilds a bool mask per row. Assumes uniform spacing (`do = omegas[1]-omegas[0]`, 139); truncation at omega_max unwindowed. |
| `write_absorption_dat(path, ...)` | 148-163 | 4-column BGW-style `.dat` (ω eV, ε₂, ε₁, JDOS). Docstring claims "Matches BGW's BSE/absp.f90:182 format" — true for the float format, **not** for column-4 semantics (see Suspects). |
| `write_eigenvalues_dat(path, eigvals_eV, dipoles_pol, *, n_spin, n_spinor, vol_supercell)` | 166-190 | BGW `eigenvalues_b{1,2,3}.dat` (absp_io.f90:122): rows `E_S(eV), |d_S|², Re d_S, Im d_S`; header carries `neig`, `vol = V_cell·N_k`, `nspin nspinor` — this is the contract `eigvals_to_eps2.read_bgw_eigvals` parses back. |
| `write_absorption_h5(path, ...)` | 193-225 | Combined artifact h5: ω grid, (n_omega,3) ε₂/ε₁/JDOS, optional eigvals/oscillators (eigvec route) or α/β/norms (Haydock route), metadata attrs. LORRAX-only format. |

### absorption_eigvecs.py (198 LOC, host-only — **no jax import at all**)

| function | lines | role |
|---|---|---|
| `compute_dipole_projections(A, d_alpha)` | 39-47 | `proj[S,α] = Σ_{kcv} A[S,k,c,v]·d[α,k,c,v]` (einsum above; no conj on A). |
| `compute_oscillator_strengths(A, d_alpha)` | 50-52 | `|proj|²` wrapper. **Zero repo callers** outside this file (main uses `compute_eps2`'s returned `f_Sa` instead) — dead-ish convenience export. |
| `compute_jdos_oscillators(d_alpha)` | 55-61 | `|d|²` per (α,k,c,v). Only used by `compute_jdos` (80). |
| `compute_eps2(eigvals_Ry, A, d_alpha, omegas_Ry, V_cell, eta_Ry, n_spin, n_spinor, n_k)` | 64-75 | pref·Σ_S f_Sα·L; returns (eps2, f_Sa, proj). |
| `compute_jdos(d_alpha, de_cv, ...)` | 78-87 | IP ε₂⁰ from flattened (cvk) transitions at `de_cv` (DFT ΔE from dipole.h5 — never eqp-corrected on this route). |
| `main(argv)` | 90-194 | argparse → load h5s → slice window using `nc, nv, nk` **from eigenvectors.h5 header** (122-131) and CLI `--n-occ` → ε₂/JDOS/KK → write per-polarisation `absorption_{b1,b2,b3}_eh.dat` + `eigenvalues_{b}.dat` + one `.h5`. `n_spinor` default inferred from file `spin_kernel` (124) — contrast Haydock route. |

### absorption_haydock.py (343 LOC, sharded jit route)

Import-time side effect (25-26): `from runtime import init_jax_distributed;
init_jax_distributed()` **before** other imports — same pattern as
`bse_jax.py:11-12` / `gw_jax` (gw_refactor_map runtime.md:169). x64 +
distributed bootstrap on module import.

| function | lines | role |
|---|---|---|
| `haydock_recursion_block(matvec_block, d_block, n_iter)` | 54-94 | 3-block Lanczos recurrence under `lax.scan` (91-92), recurrence math above. `α_n` via `.real` (82) — **assumes Hermitian H, i.e. TDA-only**. No reorth (documented; matches BGW haydock), no β→0 guard (FP keeps β>0 in practice → ghost states, not NaN). Returns `(alphas (3,n_iter), betas (3,n_iter), norms (3,))` after transposing scan's leading-axis stack (94). |
| `absorption_from_haydock(omegas_Ry, eta_Ry, alphas, betas, norms, V_cell, n_k, n_spin, n_spinor)` | 97-118 | Backward CF per polarisation (host numpy); formula above. Loops `range(n_pol)` but writes into a hard-coded `(n_omega, 3)` array (111) — fine while block=3. |
| `jdos_from_dipole(d_alpha, de_cv, omegas_Ry, V_cell, n_k, eta_Ry, n_spin, n_spinor)` | 121-132 | IP ε₂⁰ — **re-implements the Lorentzian inline (129-131) instead of calling `absorption_common.lorentzian_broaden`** (see Suspects). Uses DFT `de_cv` even when `--eqp` is passed to the BSE H. |
| `run_haydock(*, input_file, n_val, n_cond, n_occ, eqp_file, dipole_file, V_cell, n_iter, eta_eV, n_spin, n_spinor, omega_min_eV, omega_max_eV, n_omega, out_prefix, no_eps1, matvec_kind="ring")` | 135-295 | End-to-end driver, detailed walk below. |
| `main(argv)` | 298-339 | argparse → `run_haydock`. |

`run_haydock` walk:
- 157: restart bundle = `bse_io._find_restart_file(input_file)` → glob
  `{tmp/,}isdf_tensors_*.h5` next to cohsex.in (private reach-in into bse_io).
- 158-161: **raises RuntimeError if `jax.device_count() < 2`** ("requires the
  sharded matvec") — see Suspects.
- 162-164: `create_mesh_2d()` (bse_ring_comm.py:31 — px = largest int ≤ √n
  dividing n) + `make_bse_shardings` (bse_ring_comm.py:46).
- 167-170: `bse_io.load_bse_data_from_restart_sharded(restart_file, n_val,
  n_cond, mesh_xy, input_file, n_occ)` → dict of sharded device arrays
  (psi_c/v X/Y, eps_c/v, W_q, V_q0, pads; bse_io.py:358-537). The loader
  resolves n_occ, clamps n_val/n_cond to availability with a printed warning
  (bse_io.py:424-431), pads band axes to mesh multiples with zeros
  (bse_io.py:453-459) and injects the q=0 rank-1 head into V_q0/W_q via
  `gw.head_correction.apply_q0_head_rank1_sharded` (bse_io.py:504-508),
  honoring `vhead`/`whead_0freq` overrides parsed from cohsex.in.
- 172-176: re-reads `enk_full` and calls `resolve_n_occ(..., n_occ=n_occ)` —
  a **no-op** since argparse makes `--n-occ` required (see Suspects).
- 177-185: `--eqp` branch: `apply_eqp_corrections(enk_full_np, eqp_file,
  input_file)` (bse_io.py:698: IBZ→full-BZ via `SymMaps.irr_idx_k`, eV→Ry),
  then **overwrites** `data["eps_v"]/["eps_c"]` from CLI `n_val/n_cond`
  windows and re-pads to grid multiples. Uses CLI values, not the
  loader-clamped `data["n_val"]/["n_cond"]` (see Suspects).
- 195-199: dipole load + window slice + `build_dipole_vector_bse` →
  `d_block (3, nc_pad, nv_pad, nk)` on device.
- 202-208: matvec select: `"simple"` → `bse_simple.build_bse_simple_matvec`
  (plain-jit, XLA-partitioned); else `bse_ring_comm.build_bse_ring_matvec(
  ..., include_W=True, low_mem=(matvec_kind=="ring"))` — `"gather"` selects
  the all-gather encode path inside the same builder. All are **TDA**
  matvecs `HX = D + V − W` (bse_ring_comm.py:443-450, bse_simple.py:175);
  `build_bse_ring_matvec_full` (bse_ring_comm.py:487) is not reachable here.
- 210-211: `W_R = make_sharded_ifftn_3d(mesh, sh.W.spec, sh.W.spec,
  axes=(2,3,4), norm='ortho')(W_q)` — W(q)→W(R) **once**, outside the
  Lanczos loop (both matvec kinds consume W_R).
- 214-233: `_full_run` jit: `in_shardings=(sh.X, sh.psi_x, sh.psi_y, sh.psi_x,
  sh.psi_y, sh.eps, sh.eps, sh.W, sh.V)`, `out_shardings` fully replicated
  `P()` for (alphas, betas, norms), `donate_argnums=(7,)` **donates W_q** —
  its buffer is reused/freed once W_R exists. `matvec_block` pins the
  Lanczos vector to `sh.X` every application (227).
- 236-295: pull α/β/‖d‖ to host, evaluate CF + JDOS + optional KK, write
  `absorption_haydock_{b1,b2,b3}_eh.dat` + combined `.h5`. **No
  `jax.process_index()==0` guard on any write** (see Suspects).

### eigvals_to_eps2.py (178 LOC, host-only)

| function | lines | role |
|---|---|---|
| `read_bgw_eigvals(path)` | 43-55 | Regex-parse header (`neig`, `vol`, `nspin, nspinor`) + `np.loadtxt` rows → `(E_eV, |d|², V_super, n_spin, n_spinor, n_eig)`. Works on both BGW `eigenvalues.dat` and our `eigenvalues_b*.dat` (shared format contract with `write_eigenvalues_dat`). |
| `lorentzian(omegas, energies, weights, eta)` | 58-61 | third copy of the Lorentzian kernel (see Suspects). |
| `gaussian(...)` | 64-71 | area-normalised Gaussian, σ=eta. |
| `compute_eps2(E_eV, f_S, V_super, n_spin, n_spinor, *, omegas_eV, eta_eV, n_max, kernel)` | 74-94 | truncate to n_max, convert eV→Ry (local `ryd2ev` literal, 85), pref = 16π²/(V_super·ns·nspinor), broaden in Ry. |
| `compute_eps2_from_files(paths, ...)` | 97-110 | loop wrapper; dict keyed `str(path)` (duplicate paths collide silently); `n_used = min(n_max, n_in)` uses header `neig`, truncation uses actual rows — cosmetic mismatch if header lies. |
| `_main()` | 113-174 | argparse → curves → matplotlib Agg plot or `.npz`; prints per-file peak ε₂/ω. |

## Cross-module dependencies

- **bse_io** (haydock only): `_find_restart_file` (private), 
  `load_bse_data_from_restart_sharded`, `resolve_n_occ`,
  `_pad_axis_to_multiple` (private), `apply_eqp_corrections`
  (absorption_haydock.py:50, 175, 178).
- **bse_ring_comm**: `build_bse_ring_matvec`, `create_mesh_2d`,
  `make_bse_shardings` (absorption_haydock.py:51).
- **bse_simple**: `build_bse_simple_matvec` (lazy import, 203).
- **common.fft_helpers**: `make_sharded_ifftn_3d` (39, 210).
- **runtime**: `init_jax_distributed` at import (25-26).
- **gw/**: no direct import from the four files. Indirect: the restart bundle
  *is* gw_jax's `isdf_tensors_*.h5` (V_qmunu/W0_qmunu/psi_full_y/enk_full/
  G0_mu_nu/vhead/whead), and the loader calls
  `gw.head_correction.apply_q0_head_rank1_sharded` (bse_io.py:504).
- **isdf/**: no direct import; the μ/ν axes of V_q0/W_q/ψ are the ISDF
  centroid basis produced upstream by gw_jax.
- **psp/**: `dipole.h5` is `psp.get_dipole_mtxels` output (BGW
  `use_momentum` parity via `--skip-vnl`; see STATUS.md fair-comparison
  table). Consumer relationship documented in
  gw_refactor_map psp_dipole.md:30.
- Inbound: `davidson_absorption.py:32` imports three absorption_common
  helpers (a fourth absorption route outside this note's scope).

## Flags / CLI args

`absorption_eigvecs`: `--eigenvectors` (eigenvectors.h5), `--dipole`
(dipole.h5), `--n-occ` (required), `--V-cell` (required, bohr³), `--n-spin`
(1), `--n-spinor` (None → infer from file `spin_kernel`), `--eta-eV` (0.1),
`--omega-min-eV` (0), `--omega-max-eV` (15), `--n-omega` (1500),
`--out-prefix` (absorption_eigvecs), `--no-eps1`.

`absorption_haydock`: `-i/--input` (required cohsex.in), `--n-val`/`--n-cond`
/`--n-occ` (required), `--eqp` (None), `--dipole` (dipole.h5), `--V-cell`
(required), `--n-iter` (200), `--n-spin` (1), **`--n-spinor` (hard default
2)**, `--eta-eV` (0.1), `--omega-{min,max}-eV` (0/15), `--n-omega` (1500),
`--out-prefix` (absorption_haydock), `--no-eps1`, `--matvec-kind`
(ring|gather|simple, default ring — note bse_jax's default is simple).
Indirect config reads via bse_io from cohsex.in: `wfn_file`
(bse_io.py:588-601), `vhead`, `whead_0freq` (bse_io.py:669-696).
Indirect env via `runtime.init_jax_distributed` (SLURM vars,
CUDA_VISIBLE_DEVICES, `_LORRAX_JAX_DISTRIBUTED_DONE`).

`eigvals_to_eps2`: `--files` (required, n paths), `--eta-eV` (0.05),
`--kernel` (lorentzian|gaussian), `--n-max` (None), `--omega-{min,max}-eV`
(0/8), `--n-omega` (4001), `--out` (eps2_compare.png), `--no-plot`,
`--label`.

`absorption_common`: none (library only).

## Sharding / residency of large arrays (haydock route)

Shardings from `make_bse_shardings` (bse_ring_comm.py:46-63), mesh axes
("x","y") from `create_mesh_2d`:

- `d_block`/Lanczos vectors `X (3, nc_pad, nv_pad, nk)`: `sh.X =
  P(None,'x','y',None)` — conduction on x, valence on y, k replicated.
  Built replicated on host then resharded by `_full_run`'s in_shardings.
- `psi_{c,v}_X (nk, nb_pad, nspinor, μ_pad)`: `P(None,None,None,'x')`;
  `psi_{c,v}_Y`: same on 'y' — dual copies, loaded shard-aligned by
  `bse_io._read_psi_mu_sharded`.
- `W_q → W_R (μ_pad, ν_pad, nkx, nky, nkz)`: `sh.W = P('x','y',None,None,None)`
  — FFT axes local, so `make_sharded_ifftn_3d` runs devicewise with no
  collective. W_q donated after the transform (donate_argnums=(7,)).
- `V_q0 (μ_pad, ν_pad)`: `sh.V = P('x','y')`.
- `eps_c/eps_v (nk, nb_pad)`: `sh.eps = P(None,None)` — replicated.
- Outputs `alphas/betas/norms`: replicated `P()`, then host numpy.
- Host-resident: `dipole_cart (3,nk,nb,nb)` complex128 — **full table read
  by every process** (absorption_common.py:80), sliced immediately; enk_full;
  all spectra, CF evaluation, KK, and file writers are host numpy.
- Band/μ zero-padding is benign for the physics: padded ψ rows are zero ⇒
  V/W terms contribute nothing there; padded eps entries (0 Ry) multiply
  only X entries that start zero (seed padded with zeros) and stay zero
  under the matvec, so the Krylov space never populates pad rows.

`absorption_eigvecs` and `eigvals_to_eps2` are entirely host/numpy;
`A (N, nk, nc, nv)` complex128 is the biggest buffer of the eigvec route.

## TDA vs full BSE

TDA-only, throughout: `load_eigenvectors_h5` docstring pins "TDA, single Q,
single spin" (27); the Haydock recurrence takes `⟨s|Hs⟩.real` (82) and a
real symmetric-tridiagonal CF — valid only for the Hermitian TDA H; both
matvec builders used here produce `D + V − W` TDA kernels
(bse_ring_comm.py:443-450, bse_simple.py:175). BGW's non-TDA branches in
absp.f90 (`.not.xct%tda`, emission terms with en<0 skips) have no
counterpart. `build_bse_ring_matvec_full` exists (bse_ring_comm.py:487)
but is unreachable from any absorption CLI.

## Spin / nspinor

- Spinor structure lives inside the matvec (ψ arrays carry an nspinor axis;
  spinor sums happen in the T/V einsums, e.g. `'kcsN,kvsN->kcvN'`
  bse_simple.py:90). The absorption modules only use `n_spin`/`n_spinor` as
  scalar prefactor divisors — exactly like BGW absh/absp.
- Collinear ns=2 unsupported on the eigvec route (hard raise,
  absorption_common.py:54-56); `--n-spin` exists but nothing validates it
  against the data.
- `absorption_eigvecs` infers `n_spinor` from `spin_kernel==3` in the h5
  (47, 124); `absorption_haydock` cannot (no eigvec file) and hard-defaults
  `--n-spinor 2` (317-318) even though the true value sits in
  `data["psi_c_X"].shape[2]` — see Suspects.

## Suspects

### Bugs / hazards

1. **2× ε₂ error following the canonical cookbook on non-spinor systems.**
   `absorption_haydock` defaults `--n-spinor 2` (317-318). BGW_COMPARE.md's
   cookbook command (49-53) runs Si (nspinor=1, `--n-occ 8`) **without**
   `--n-spinor`, so pref = 16π²/(V·N_k·1·2) — exactly half of BGW's
   16π²/(vol·1·1) (absh.f90:46). ε₂ and JDOS come out 2× too small; the
   validated 1.5% agreement (STATUS.md) is unreproducible from the
   documented command. The correct nspinor is available in-band as
   `data["psi_c_X"].shape[2]` but ignored. Fix: infer like the eigvec route
   does, or at least cross-check CLI vs ψ shape; and patch the cookbook.
2. **Unguarded multi-process writes.** `run_haydock` prints and writes
   `.dat`/`.h5` on **every** rank — zero `jax.process_index()` hits in the
   file, vs `davidson_absorption.py:80` which guards with `rank0`. Under a
   multi-process srun launch (the very case `init_jax_distributed` at
   import supports, and the ≥2-device gate implies), N ranks concurrently
   `h5py.File(path, "w")` the same output (absorption_common.py:208) →
   HDF5 lock error or truncation race; the `.dat` writers race too.
3. **Silent oscillator-strength loss when dipole.h5 is band-short.**
   `slice_dipole_to_bse_window` (97-100) uses clamping Python slices: if
   dipole.h5 `nbands < n_occ + n_cond`, `v_cv` silently has nc′ < n_cond;
   the Haydock seed builder `build_dipole_vector_bse` (114-115) then
   zero-fills the missing conduction rows — the run completes with no
   warning and a spectrum missing the top-band transitions. (The eigvec
   route instead crashes on einsum shape mismatch at
   absorption_eigvecs.py:47 — inconsistent failure modes.)
4. **eqp branch ignores loader clamping.** The loader clamps n_val/n_cond
   to availability (bse_io.py:424-431, returned as `data["n_val"]/["n_cond"]`,
   unused here); `run_haydock:180-183` re-slices `enk_full_np[:, cond_idx]`
   with the **CLI** values → numpy fancy-index IndexError when the user
   over-asks with `--eqp` set (a path the loader had already "handled" with
   a warning). Same CLI values feed the dipole slice (196).

### Redundancy

- **Three copies of the Lorentzian kernel**: `absorption_common.
  lorentzian_broaden` (119-126), inlined in `absorption_haydock.
  jdos_from_dipole` (129-131), and `eigvals_to_eps2.lorentzian` (58-61).
  Likewise `compute_jdos` (eigvecs 78-87) vs `jdos_from_dipole` (haydock
  121-132) are the same function modulo the inlining. Violates the
  single-source-of-truth rule.
- `RYD2EV` duplicated as a literal (`eigvals_to_eps2.py:85` local `ryd2ev`)
  despite `absorption_common.RYD2EV:23` — part of the known bse/ units
  sprawl (gw_refactor_map common_utils.md:56).
- `compute_oscillator_strengths` (eigvecs 50-52): zero callers anywhere
  (repo-wide grep); `main` gets `f_Sa` from `compute_eps2` instead. Dead
  convenience wrapper.

### Cruft / weird

- **No-op n_occ re-resolution + stale comment** (haydock 172-176): argparse
  requires `--n-occ`, so `resolve_n_occ(..., n_occ=n_occ)` returns the CLI
  value at its first branch (bse_io.py:636-637); the comment advertises a
  "largest-gap fallback" that bse_io explicitly removed (bse_io.py:621-624
  docstring: heuristic was "silently broken... We now require an explicit
  source"). The `enk_full` read is still needed for the eqp branch, but the
  resolve call is dead.
- **≥2-device gate** (158-161) blocks 1-GPU runs though `create_mesh_2d`
  yields a working 1×1 mesh and both matvec kinds run under shard_map on a
  single device — conflicts with the project rule that nothing may require
  multi-GPU hardware. Untestable on the standard 1-GPU MoS2 rig as-is.
- Private reach-ins: `_find_restart_file`, `_pad_axis_to_multiple` imported
  from bse_io (50, 178).
- No β→0 guard in the recurrence (87): exact-arithmetic breakdown at
  n_iter > dim would divide by zero; FP noise prevents it in practice
  (ghost states instead — documented in STATUS.md).

### BGW-convention divergences (NOT bugs — flagged so nobody "fixes" them)

- Valence-axis flip on eigenvectors.h5 read (absorption_common.py:59-63)
  and the lowest-first window slice (97-99) are the documented
  LORRAX↔BGW mapping (STATUS.md "Index ordering" items 1/3).
- Ry↔eV: files carry eV, all math in Ry (66, 136-137); matches BGW's
  Ry-space broadening.
- **Column-4 "JDOS" semantics differ from BGW**: LORRAX writes
  dipole-weighted independent-particle ε₂⁰ at ΔE_cv from dipole.h5;
  BGW absp.f90:155 writes `dos = (nspin/neig)·Σ_S fac2` — unit-weight
  exciton DOS. The two column-4s are not comparable numbers; only
  columns 1-3 are. `write_absorption_dat`'s "Matches BGW" docstring
  (150-152) refers to the float format only. Also: BGW's Haydock
  absorption_eh.dat (absh.f90) has only 3 columns.
- LORRAX broadening keeps only the resonant term; BGW absp.f90:157-180
  subtracts the mirrored (emission) kernel at −ω so ε₂(0)=0 exactly.
  O(η²/E²) effect (≈0.06% at the Si validation settings), visible only
  near ω→0. Same for the Haydock CF (resolvent at ω+iη only). BGW also
  builds ε₁ from a broadened sum (absp.f90:153) where LORRAX uses grid
  KK — `--no-eps1` is the recommended comparison mode anyway.
- JDOS stays at DFT ΔE even when `--eqp` shifts the BSE H (haydock 196 vs
  178-183); BGW's IP spectrum follows eqp. Deliberate? Undocumented either
  way.
