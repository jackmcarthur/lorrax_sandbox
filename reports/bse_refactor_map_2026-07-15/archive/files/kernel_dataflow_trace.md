# BSE end-to-end kernel dataflow trace — deep-read notes

Audit date: 2026-07-15, lorrax_D checkout. Requested base: agent/slate-linalg-ffi
(e18d0e5); checkout HEAD is adc2197 (agent/ppm-fit-conditioning), but
`git diff --stat e18d0e5..HEAD -- src/bse src/solvers` is **empty** — the BSE and
solvers trees audited here are byte-identical to e18d0e5. (Only `src/gw/gw_config.py`
differs between the two commits; it is not on any BSE path.)

Scope: the MAP.md-§1-style dataflow spine for the BSE stack — where H^BSE = D + K
comes from, every boundary array, sharding, host/device residency, TDA vs full,
spin handling, gw/isdf/common coupling, and every coarse-k bake-in point.
Files: `src/bse/*` (~10.2 kLOC total), plus the producer side in
`src/gw/gw_init.py`, `src/gw/gw_output.py`, `src/gw/head_correction.py`,
`src/file_io/tagged_arrays.py`, and the consumers in `src/solvers/`.

## Purpose

The BSE stack diagonalises (or Krylov-resolves) the exciton Hamiltonian in the
ISDF centroid basis on the **coarse GW k-grid**, entirely from the GW restart
bundle `tmp/isdf_tensors_{n_rmu}.h5` — no plane-wave data is touched at BSE
time except `WFN.h5` metadata (kgrid, ifmax, cell_volume). The TDA Hamiltonian
as implemented (all four matvec variants agree):

```
H X = D X + V X − W X                       (Ry throughout; bse_serial.py:80)

D:  (ΔE X)[b,c,v,k]  = (ε_c[k,c] − ε_v[k,v]) · X[b,c,v,k]          # QP or DFT ε

V (exchange, q=0):                                                  # bse_serial.py:62-64
    M_cvk(ν)        = Σ_s ψ*_c[k,c,s,ν] ψ_v[k,v,s,ν]                # spin-traced
    S[b,ν,k]        = Σ_{c,v} M_cvk(ν) X[b,c,v,k] / √Nk
    U[b,μ,k]        = Σ_ν V_q0[μ,ν] S[b,ν,k]
    (V X)[b,c,v,k]  = Σ_μ M*_cvk(μ) U[b,μ,k] / √Nk
                    = (1/Nk) Σ_{μν} M*_cvk(μ) V_0(μν) Σ_{c'v'} M_c'v'k(ν) X[b,c',v',k]
    →  k is a BATCH index end-to-end (see Suspects §B1: no Σ_{k'}).

W (direct, screened):                                               # bse_serial.py:69-78
    T[b,μ,ν,t,s,k'] = Σ_{c',v'} ψ_c[k',c',t,μ] ψ*_v[k',v',s,ν] X[b,c',v',k']
    U[b,μ,ν,t,s,k]  = (1/√Nk) Σ_q W_q[μ,ν,q] T[b,μ,ν,t,s,k−q]      # ortho ifftₖ·mul·fftₖ
    (W X)[b,c,v,k]  = (1/√Nk) Σ_{μνts} ψ*_c[k,c,t,μ] ψ_v[k,v,s,ν] U[b,μ,ν,t,s,k]
                    = (1/Nk) Σ_{k'} Σ_{c'v'} [Σ_{tμ} ψ*_c(k)ψ_c(k')]_μ
                                     W_{μν}(k−k')
                                     [Σ_{sν} ψ_v(k)ψ*_v(k')]_ν  X[b,c',v',k']
```

The direct term is the standard ISDF-factorised BGW kernel
`⟨cvk|K^d|c'v'k'⟩ = Σ_{μν} [ψ*_c(k)ψ_c(k')](μ) W_{μν}(k−k') [ψ_v(k)ψ*_v(k')](ν)/Nk`,
with the q = k−k' convolution done as a 3-D ortho-normalised FFT pair over the
coarse k-grid (numpy ifft sign convention ⇒ `U[k] = (1/√Nk) Σ_q W[q]·T[k−q]`,
so the W_q index is exactly q = k−k'). Spinor indices t (conduction line) and s
(valence line) are traced separately — W is treated as a spin-independent
charge-channel interaction. The exchange as coded is **k-block-diagonal**; the
textbook exchange is dense in (k,k') — flagged as the top physics suspect below.

Category: **pipeline stage (post-GW excited-state solve) + spectra post-processing**.

## Dataflow spine (analogue of GW MAP.md §1)

```
                 ┌── GW SIDE (producers) ─────────────────────────────────────────┐
 WFN.h5 ─┬─► gw_init.setup_isdf_tensors (gw_init.py:545-716)
         │      kmeans centroids → fit_zeta → compute_V_q
         │      write_restart_state_to_h5(mode="w")  [tagged_arrays.py:29-127]
         │        V_qmunu (nq,μ,ν) flat-q, LOGICAL μ on disk
         │        G0_mu_nu (μ,) = ζ(q=0,μ,G=0)   enk_full (nk,nb) Ry
         │        kgrid → TOP-LEVEL DATASET (write_attr, _slab_io_allgather.py:120-137)
         │        W0_qmunu zero placeholder, attr W0_ready=False
         │      append psi_full_y = wfns.psi_yr (nk,nb,nspinor,μ)   [gw_init.py:692-697]
         │   gw_jax → compute_screening → persist_w0_and_head (gw_output.py:175-241)
         │        W0_qmunu ← W_q(ω=0) flat-q, W0_ready=True
         │        vhead = v(q→0,G=G'=0), whead[ω] = wcoul0   (write_head_scalars_to_h5)
         │
         └─► psp.get_dipole_mtxels (-i cohsex.in [--skip-vnl])
                dipole.h5: dipole_cart (3,nk,nb,nb) ⟨m|p̂(+i[r,V_NL])|n⟩, deltaE (nk,nb,nb)
                                                    [get_dipole_mtxels.py:723-740]
                 └───────────────────────────────────────────────────────────────┘
 tmp/isdf_tensors_*.h5          eqp1.dat (BGW)              cohsex.in
        │                            │                          │ (wfn_file, vhead/whead_0freq overrides)
        ▼                            ▼                          ▼
 ┌─ LOADERS (bse_io.py) ──────────────────────────────────────────────────────────┐
 │ _find_restart_file (756)  → tmp/isdf_tensors_*.h5 glob                          │
 │ ≥2 devices: load_bse_data_from_restart_sharded (358-536)                        │
 │    resolve_n_occ (603): --n-occ ▸ WFN ifmax (WfnLoader.nelec=max(ifmax))        │
 │    val_idx = [n_occ−n_val, n_occ) (v=0 = DEEPEST valence — LORRAX internal)     │
 │    _read_psi_mu_sharded ×2 → psi_{v,c}_X  P(None,None,None,'x') (+Y copy on 'y')│
 │    _read_vq0_sharded → V_q0 (μ,ν)  P('x','y')          [8-D-ONLY reader — B3]   │
 │    _read_wq_sharded  → W_q (μ,ν,nkx,nky,nkz) P('x','y',None,None,None)  [B4]    │
 │    apply_q0_head_rank1_sharded (gw/head_correction.py:779):                     │
 │        V_q0 += (vhead/V_cell)·g0*⊗g0 ;  W_q[:,:,0,0,0] += (whead[0]/V_cell)·g0*⊗g0│
 │ 1 device: _load_ring_subset (767-932): same, host-numpy, apply_q0_head_rank1    │
 │        BEFORE the layout shim [B5]; W_src = W0_qmunu if W0_ready else V_qmunu   │
 │ eqp: apply_eqp_corrections (698): eqp1.dat (eV, IBZ) → SymMaps.irr_idx_k        │
 │        unfold → enk_qp/13.6057 (Ry, full BZ) → re-slice eps_v/eps_c             │
 └──────────────────────────────────────────────────────────────────────────────┘
        │  data dict: psi_{c,v}_{X,Y}, eps_c/eps_v, W_q, V_q0, g0_X/g0_Y,
        │             nkx/nky/nkz, n_rmu(_pad), n_val/cond(_pad)
        ▼
 ┌─ MATVEC (one H·X, all TDA-hermitian) ──────────────────────────────────────────┐
 │ W_R = ifftₖ(W_q) ONCE per solve (make_sharded_ifftn_3d, common/fft_helpers)     │
 │ "simple": bse_simple.build_bse_simple_matvec — plain jit + einsum +             │
 │           with_sharding_constraint, XLA auto-SPMD (fastest, recommended)        │
 │ "ring":   bse_ring_comm.build_bse_ring_matvec — shard_map, ppermute rings       │
 │           over band chunks (low memory)                                         │
 │ "gather": same but lax.all_gather T-encode                                      │
 │ serial:   bse_serial.apply_bse_hamiltonian_single_device (1-device reference)   │
 │ full BSE: bse_ring_comm.build_bse_ring_matvec_full — S=[[A,B],[−B†,−A†]]        │
 │           (B-encode einsum fatally malformed when include_W [B2])               │
 └──────────────────────────────────────────────────────────────────────────────┘
        ▼
 ┌─ SOLVERS ──────────────────────────────────────────────────────────────────────┐
 │ bse_jax CLI default        → bse_feast.main   (FEAST windows + GMRES + Ritz)    │
 │ --lanczos                  → bse_lanczos.solve_bse_sharded (block-)Lanczos      │
 │ --lanczos --solver davidson→ solvers.davidson + bse_davidson_helpers            │
 │ --kpm-dos                  → bse_kpm (Chebyshev DOS, solvers.chebyshev)         │
 │ 1 device                   → solve_bse (jit Lanczos; CLI knobs ignored [C2])    │
 │ diagnostics: bse_w_exact (Casida shifted-solve W_c(ω) columns),                 │
 │   bse_pseudopoles, feast_*_sweep, bse_feast_dense_debug, test_bse,              │
 │   test_davidson_bse                                                             │
 └──────────────────────────────────────────────────────────────────────────────┘
        │ eigenvalues (Ry, replicated), eigenvectors (n_eig,1,nc,nv,nk) replicated
        ▼
 ┌─ OUTPUT / SPECTRA ─────────────────────────────────────────────────────────────┐
 │ bse_io.write_eigenvectors_stream → eigenvectors.h5 (BGW spec: eV, valence axis  │
 │   FLIPPED on write, spin_kernel=3, ns=1, use_tda=1)                             │
 │ absorption_haydock: seed |d^α⟩ (α=x,y,z) → 3-block CF recursion (α_n,β_n)       │
 │   ε₂^α(ω) = (16π²/(V_cell·Nk·n_spin·n_spinor))·‖d^α‖²·(−Im g(ω+iη)/π)           │
 │ absorption_eigvecs: ⟨0|r̂_α|S⟩ = Σ_cvk A^S_cvk d^α_cvk; Lorentzian sum-over-states│
 │ davidson_absorption: exact low eigvecs + per-state oscillators                  │
 │ eigvals_to_eps2: re-broaden any BGW/LORRAX eigenvalues.dat at custom η/n_max    │
 │ outputs: absorption_*_{b1,b2,b3}_eh.dat (BGW 4-col), eigenvalues_b*.dat, .h5    │
 └──────────────────────────────────────────────────────────────────────────────┘
```

Where each H^BSE ingredient comes from:

| Ingredient | Source | File / dataset |
|---|---|---|
| Diagonal D | `enk_full` (DFT, Ry) from GW restart; optionally overwritten by BGW `eqp1.dat` via `--eqp` (IBZ→full-BZ via `common.symmetry_maps.SymMaps.irr_idx_k`, eV→Ry) | `bse_io.py:436-437, 698-753` |
| Direct K^d (W) | `W0_qmunu` = static W(ω=0) in ISDF μν basis, written by GW `persist_w0_and_head` after `compute_screening`; **falls back silently to bare `V_qmunu`** when `W0_ready` attr is False (`bse_io.py:376-379, 906`) | `gw_output.py:175-241` |
| Exchange K^x (v) | `V_qmunu[q=0]` tile (bare Coulomb, G=G'=0 zeroed by `compute_vcoul`) | `bse_io.py:460, 904` |
| q→0 head | rank-1 `(head/V_cell)·conj(G0)⊗G0` with `G0_mu_nu = ζ(0,μ,G=0)`; head scalars from restart `vhead`/`whead` or `cohsex.in` overrides `vhead=`/`whead_0freq=` (`_parse_head_overrides`, `bse_io.py:667-695`) | `gw/head_correction.py:743-820` |
| Dipole (spectra) | `dipole.h5` from `psp.get_dipole_mtxels` (BGW-compat = `--skip-vnl`, bare p̂); r-form d = ⟨c|v̂|v⟩/ΔE | `absorption_common.py:70-102` |
| epsmat / zeta_q.h5 | **not read by bse/** — all screening enters via W0_qmunu; `bse_w_exact.py` is a *diagnostic* that recomputes W_c(ω) columns from the Casida matrix, not a kernel source | `bse_w_exact.py:1-178` |

## Entry points (grep over src/, tests/, tools/, scripts/, sandbox skills+runs)

| symbol / module | callers (grep evidence) |
|---|---|
| `python -m bse.bse_jax` | STATUS.md:100-105, BGW_COMPARE.md:58-62 (recommended commands); no run *.sh found under sandbox runs/ (runs driven interactively; `runs/Si/04_si_4x4x4_bse/00_lorrax_bse/` holds outputs only) |
| `python -m bse.absorption_haydock` | STATUS.md:108-111, BGW_COMPARE.md:49-53 |
| `python -m bse.absorption_eigvecs` | STATUS.md:114-116, BGW_COMPARE.md:63-66 |
| `python -m bse.eigvals_to_eps2` | BGW_COMPARE.md:78-83 |
| `python -m bse.davidson_absorption` | own docstring usage block (davidson_absorption.py:12-19); no external refs |
| `python -m bse.test_bse`, `bse.test_davidson_bse` | test_bse.py docstring:3-4; not collected by pytest (live under src/, need `-i` + GPU data) |
| `python -m bse.bse_w_exact / bse_pseudopoles / feast_sweep / feast_zolo_sweep / feast_ellipse_mixed_sweep / pseudopoles_eval / bse_feast_dense_debug / write_eigenvectors / bse_kpm / bse_feast` | `if __name__ == "__main__"` mains; `tests/archive/projects/test_isdf/sweep_12v12c_plot.py:9,33,38` references the *old package name* `bse_isdf.pseudopoles_eval` (stale) |
| `bse_io.read_bgw_eqp` | `tests/test_eqp_bgw.py:78` — the only pytest-collected import of the bse package |
| `bse_feast.main`, `bse_kpm.main` | delegated from `bse_jax.__main__` (bse_jax.py:536, 543) |
| `solve_bse_sharded` | bse_jax.py:283 |
| `load_bse_data_from_restart_sharded` | bse_jax.py:234, absorption_haydock.py:167, davidson_absorption.py:~90, bse_feast.py:1169, bse_w_exact.py:86, bse_kpm.py (via bse_feast helpers), test_davidson_bse.py |
| `_load_ring_subset` | bse_jax.py:295 (1-device), bse_ring_comm.py:919 (correctness check) |
| `gw/head_correction.apply_q0_head_rank1{,_sharded}` | bse_io.py:818, 504 — the **only** gw→bse code imports |

## Boundary arrays (name, shape, sharding, residency)

| Array | Shape | Sharding (mesh ('x','y')) | Residency / notes |
|---|---|---|---|
| `V_qmunu` on disk | (nq, μ, ν) flat-q (modern); legacy (1,npol,npol,nkx,nky,nkz,μ,ν) 8-D and (1,npol,npol,nq,μ,ν) 6-D still in fleet | n/a | HDF5; LOGICAL μ extent (SHARDING_RULES §2, tagged_arrays.py:14-27); re-padded on read via `runtime.padding.padded_mu_extent(n_rmu, grid_x·grid_y)` |
| `W0_qmunu` on disk | (nq, μ, ν) | n/a | zero placeholder until `W0_ready=True` (tagged_arrays.py:98-121) |
| `psi_full_y` on disk | (nk, nb, nspinor, μ) | n/a | ψ at charge centroids (= `wfns.psi_yr`); h5py **partial reads** per μ-shard (`_read_psi_mu_sharded`, bse_io.py:174-218) — never materialised whole on one host in the sharded path |
| `psi_{c,v}_X` / `_Y` | (nk, nb_pad, nspinor, μ_pad) | P(None,None,None,'x') / (…,'y') | device; TWO copies of each ψ (μ on x for encode/decode, ν on y) |
| `V_q0` | (μ_pad, ν_pad) | P('x','y') | device |
| `W_q` | (μ_pad, ν_pad, nkx, nky, nkz) | P('x','y',None,None,None) | device; k axes replicated so ifftₖ is device-local (`make_sharded_ifftn_3d`); donated into the solver jit (bse_lanczos.py:244) |
| `W_R` | same as W_q | same | built once per solve inside jit |
| `eps_c`, `eps_v` | (nk, nb_pad) | P(None,None) replicated | device (tiny) |
| `g0_X`, `g0_Y` | (μ_pad,) | P('x') and P('y') | dual-sharded so rank-1 head is comm-free (bse_io.py:463-484) |
| trial X | (block, nc_pad, nv_pad, nk) | `sh.X` = P(None,'x','y',None) | device; c on x, v on y, k replicated |
| non-TDA X_full | (2, block, nc, nv, nk) | `sh.X_full` = P(None,None,'x','y',None) | device |
| T (W-encode) | (b, μ_loc, ν_loc, ns, ns, nk) | `sh.T` = P(None,'x','y',None,None,None) | device; the memory hog; donated to `apply_W_from_T` (bse_ring_comm.py:430) |
| eigenvalues / eigenvectors | (n_eig,), (n_eig,1,nc_pad,nv_pad,nk) | replicated P() | device→host at write time, streamed per-vector (`write_eigenvectors_stream` loop, bse_io.py:87-103) |
| dipole_cart / deltaE | (3,nk,nb,nb) / (nk,nb,nb) | n/a | host numpy (absorption_common.py:70-83); sliced then `jnp.asarray` as (3,nc_pad,nv_pad,nk) seed |
| Haydock (α_n, β_n, ‖d‖) | (3,n_iter) ×2, (3,) | replicated out_shardings | continued fraction evaluated on host numpy (absorption_haydock.py:97-118) |

All shardings are declared in ONE place: `make_bse_shardings` (bse_ring_comm.py:46-63).
`create_mesh_2d` (bse_ring_comm.py:31-43) picks px = largest divisor ≤ √ndev.
`_assert_local_block` (bse_io.py:164-171) requires each process's local devices to
form a Cartesian x×y block (multi-node constraint).

## Function tables (core dataflow files)

### bse_jax.py (626 LOC) — CLI + dead module-level matvec copy
- `apply_bse_hamiltonian` 67-86, `apply_D` 89-91, `apply_V` 98-121, `apply_W`
  124-160: module-level copy of the shard_map internals using
  `lax.psum(axis_name=…)` — **uncallable outside shard_map, zero callers** (dead, D1).
- `_main_random_demo` 163-200 (`--debug-parallelism`).
- `_preview_lanczos` 203-345: sharded-vs-1-device branch (221-222 on
  `jax.device_count()`), eqp re-slice 241-261, solver dispatch 283-288 / 325-328,
  eigenvector write 333-345.
- `__main__` 349-626: argparse; routes → kpm (512-537) → feast (539-604, the
  **default** when `--lanczos` absent) → TDA-only Lanczos preview (606-626).

### bse_io.py (932 LOC) — all restart I/O + eqp + heads
- `write_eigenvectors_stream` 23-105: BGW-compliant eigenvectors.h5 — Ry→eV
  (40-41), valence flip on write (100), ns=1/spin_kernel=3/use_tda=1 hardcoded (45-47, 71).
- pad helpers 108-146; **`_pad_axis_to_multiple` returns the ORIGINAL size** (B6).
- mesh-coord helpers 149-171; `_read_psi_mu_sharded` 174-218; `_read_vq0_sharded`
  221-265 (8-D-only, B3); `_read_wq_sharded` 268-355 (3-layout shim; flat-q
  requires `dset.attrs['kgrid']`, B4).
- `load_bse_data_from_restart_sharded` 358-536 (spine above).
- `read_bgw_eqp` 539-581; `_parse_wfn_path` 584-600; `resolve_n_occ` 603-664
  (explicit → WFN `ifmax` via `WfnLoader.nelec = max(ifmax)`, wfn_loader.py:162 —
  name says electrons, value is highest-occupied-band index, correct for the
  purpose) ; `_parse_head_overrides` 667-695; `apply_eqp_corrections` 698-753
  (SymMaps path 711-725, energy-matching fallback 726-751, tol 0.01 eV);
  `_find_restart_file` 756-764; `_load_ring_subset` 767-932 (head injection
  804-836 **before** layout shim 838-874, B5; random X 914-917).

### bse_serial.py (99) / bse_simple.py (184) / bse_ring_comm.py (996)
- `bse_serial.symmetrize_W_q` 12-24: enforces W(q)=W(−q)†; **never called** (D2) —
  H hermiticity relies on the GW-side W being symmetric.
- `bse_serial.apply_bse_hamiltonian_single_device` 38-80: reference TDA H (formulas §Purpose).
- `bse_simple.build_bse_simple_matvec` 46-184: same math as plain einsums +
  `with_sharding_constraint`; jit with explicit in/out shardings 177-184.
- `bse_ring_comm`: `_ring_sum_valence` 70-96 / `_ring_sum_conduction` 99-126
  (A-block T-encode rings); `_ring_sum_conduction_first` 129-157 /
  `_ring_sum_valence_second` 160-188 (**B-block; malformed einsum, B2**);
  `apply_W_ring` 191-225; `apply_V_ring` 228-300 (k-batch exchange, B1);
  `apply_bse_hamiltonian_ring` 303-337 (**dead**, D3); `build_bse_ring_matvec`
  340-484; `build_bse_ring_matvec_full` 487-716 (non-TDA
  S=[[A,B],[−B†,−A†]], Y_out = −B†X − AY at 659-670);
  `build_realspace_random_transition_generator` 719-772 and
  `build_density_snapshot_operator` 775-850 (χ-probe ops for bse_w_exact:
  these DO sum over k — `d_mu = Σ_k U` at 842);
  `ring_matvec_smoke_test` 853-898; `ring_matvec_correctness_check` 901-995
  (ring vs serial only — **no independent dense reference anywhere**).

### bse_lanczos.py (313)
- `solve_bse` 30-97: 1-device wrapper; **defaults n_reorth=10, no plumbing from CLI** (C2).
- `solve_bse_sharded` 100-313: builds matvec by `data["matvec_kind"]` (155-165);
  `W_R` once inside jit via `make_sharded_ifftn_3d` (173-181); Davidson branch
  188-231 (diag (ΔE−λ+ε)⁻¹ precond + energy-sorted init subspace +
  `warmup_davidson_jit` pre-compiles Ritz solves at m ∈ {n_eig,…,4·n_eig});
  Lanczos branch 233-313: one end-to-end jit, in_shardings from
  `make_bse_shardings`, `donate_argnums=(6,)` (W_q), rtol>0 →
  `block_lanczos_eig_jit_converged` while_loop early exit.

### Absorption stage
- `absorption_common.py` (239): h5 readers (BGW valence flip on READ,
  load_eigenvectors_h5:63), dipole slice ⟨c|v̂|v⟩/ΔE (86-102), Lorentzian,
  O(N²) KK, BGW-format writers (absorption `4f16.9` per BSE/absp.f90:182;
  eigenvalues per absp_io.f90:122).
- `absorption_haydock.py` (343): 3-block CF recursion under one jit (214-233),
  **hard-requires ≥2 devices** (158-161); prefactor 16π²/(V·Nk·ns·nspinor)
  matches BGW absh.f90:46 (per STATUS.md:122); `--n-spinor` defaults to 2.
- `absorption_eigvecs.py` (198): sum-over-states from eigenvectors.h5;
  n_spinor inferred from spin_kernel==3.
- `davidson_absorption.py` (216): BGW-diagonalization-mode analogue.
- `eigvals_to_eps2.py` (178): re-broaden eigenvalues.dat, matched-truncation compare.
- `write_eigenvectors.py` (236): older non-streaming writer + `generate_kpts_grid`
  (C-order [ix/nkx, iy/nky, iz/nkz]) — used by test_bse and by the streaming writer
  for kpts.

### Solver/diagnostic satellites (skimmed, categorized)
- `bse_feast.py` (1320): spectral bounds via short Lanczos, KPM-weighted windows,
  FEAST ellipse/Zolotarev quadrature, GMRES shifted solves (fp32 option), Ritz
  extraction. Consumes the same matvecs. No `--eqp`/`--n-occ` flags (C3).
- `bse_kpm.py` (407): Jackson-damped Chebyshev DOS, stochastic trace.
- `bse_pseudopoles.py` (699) + `pseudopoles_eval.py` (179): pseudo-pole fit of
  W_c(ω) from Casida solves (research thread; `tests/archive` references its old
  package name only).
- `feast_sweep.py` (633), `feast_zolo_sweep.py` (333),
  `feast_ellipse_mixed_sweep.py` (321), `bse_feast_dense_debug.py` (178):
  parameter sweeps / dense-vs-FEAST debug; diagnostics.
- `context/` — design notes (Henneke 2020, parallel algos, FEAST notes), not code.

## Flags / config consumed

CLI (bse_jax): `-i/--input` (cohsex.in; used for restart glob dir + `wfn_file`
+ `vhead`/`whead_0freq` overrides), `--n-val` (4), `--n-cond` (4), `--px/--py`
(1; **only used by ring-check path** — production mesh comes from
`create_mesh_2d()` over all visible devices), `--n-eig` (5), `--tda`, `--rpa`,
`--bse` (**default kernel is RPA D+V**; `--bse` enables −W), `--lanczos`
(**default route is FEAST**), `--eqp` (BGW eqp1.dat), `--n-occ`,
`--write-eigs [N]`, `--max-lanczos-iter`, `--block-size`, `--lanczos-rtol`,
`--lanczos-check-every`, `--n-reorth` (−1 = full), `--matvec-kind`
(ring|gather|simple), `--gather-t` (deprecated alias), `--solver`
(lanczos|davidson), `--kpm-*` (7 flags), `--feast-*` (9 flags), `--gmres-*`
(4), `--ring-test/--ring-check/--ring-timing/--components/--debug-parallelism`.

cohsex.in keys read by bse_io: `wfn_file` (default WFN.h5, relative to input
dir), `vhead`, `whead_0freq` (head overrides; BGW-compare convention #3 says
populate these from BGW's vcoul when matching BGW). Env: `JAX_ENABLE_X64=1`
setdefault + `runtime.init_jax_distributed()` at import (bse_jax.py:10-12);
`ISDF_JAX_PROFILE_DIR` (test_bse).

## TDA vs full-BSE

- All production routes (Lanczos, Davidson, Haydock, eigvec absorption) are
  **TDA-only**; `_preview_lanczos` refuses non-TDA (bse_jax.py:606-607);
  eigenvectors.h5 writes `use_tda=1` unconditionally.
- Non-TDA exists only as `build_bse_ring_matvec_full` (FEAST / KPM / pseudopoles
  / w_exact when `--tda` absent — note their default IS non-TDA), applying
  S = [[A,B],[−B†,−A†]] with B built by conjugated-ψ exchange
  (`apply_V_ring_B`, bse_ring_comm.py:572-583) and a B-side T-encode that is
  **fatally malformed** whenever `include_W=True` (B2) — i.e. full-BSE has only
  ever worked as non-TDA **RPA** (D+V casida). No Y-component I/O exists.

## Spin / nspinor

- nspinor inferred from `psi_full_y.shape[2]`; no collinear nspin=2 path
  anywhere in bse/ (restart has no spin axis).
- Exchange M is spinor-traced at same k (`kcsm,kvsm->kcvm`); direct term traces
  conduction (t) and valence (s) spinor lines separately against a **scalar**
  W_{μν} — charge-channel screening only (consistent with the known
  bispinor-screened-W roadmap: no spin-flip/transverse W in BSE).
- No singlet factor 2 on K^x and no n_spin switch in any matvec: correct for
  nspinor=2 spinor runs (the validated Si-SOC case), but a scalar nspinor=1 run
  would produce the triplet-like kernel unless the user knows to interpret it
  (C4). n_spin/n_spinor enter only as ε₂ prefactors in the absorption stage
  (user-supplied, defaults n_spin=1, n_spinor=2).
- BGW-compat conventions (NOT bugs): valence axis flipped only at file
  boundaries (write: bse_io.py:100; read: absorption_common.py:63; internal
  v=0 = deepest valence); Ry internal / eV in eigenvectors.h5+eigenvalues.dat;
  BGW `iv=1` = highest valence per STATUS.md "Index ordering".

## Coupling to gw/ isdf/ common/ (import graph, grep evidence)

- `gw.head_correction.apply_q0_head_rank1{,_sharded}` (bse_io.py:504, 818) —
  the ONLY `from gw.` imports in bse/.
- **No `isdf` imports**: all ISDF data arrives via the restart h5 (the h5 is the
  module boundary). Writers: `file_io.tagged_arrays.write_restart_state_to_h5`
  (gw_init.py:661, 692), `write_w0_qmunu_to_h5` + `write_head_scalars_to_h5`
  (gw_output.py:209-233).
- `common.fft_helpers.make_sharded_{i,}fftn_3d` (bse_lanczos.py:17,
  bse_ring_comm.py:20-23, bse_simple.py:39-42, absorption_haydock.py:39,
  davidson_absorption.py:40) — the custom-partitioned local FFT (avoids XLA
  all-gather around sharded-tensor FFTs).
- `common.symmetry_maps.SymMaps` (bse_io.py:713) — IBZ→full-BZ eqp unfold.
- `common.timing`, `common.jax_profile`, `common.jax_compile_cache`.
- `file_io.WfnLoader` (kgrid, ifmax→nelec, cell_volume), `runtime.init_jax_distributed`,
  `runtime.padding.padded_mu_extent` (the ONE μ-pad convention).
- `solvers.lanczos` (lanczos_eig_jit, block_lanczos_eig_jit{,_converged}),
  `solvers.davidson`, `solvers.chebyshev`, `solvers.quadrature`.

## Coarse-k bake-in points (where a fine-grid/interpolation layer must hook)

1. `psi_full_y` exists only at coarse k AND only at ISDF centroids r_μ — a
   BGW-style WFN_fi interpolation cannot be retrofitted at the loader without a
   new ψ source (bse_io.py:385, 446-447).
2. kgrid resolution chain (V_qmunu shape / `kgrid` dataset / WFN.kgrid):
   bse_io.py:389-411, 846-874 — nq must equal nkx·nky·nkz exactly (checked 318-320).
3. W(q=k−k') convolution as 3-D FFT over exactly (nkx,nky,nkz)
   (bse_serial.py:71-75, bse_ring_comm.py:214-218, bse_simple.py:147-157) —
   fine-k W would need interpolation of W_{μν}(q) plus a larger FFT grid.
4. D diagonal from enk_full[:, idx] at coarse k; eqp unfold maps IBZ→coarse
   full BZ (bse_io.py:721-725).
5. dipole.h5 is (3, nk_coarse, nb, nb) (absorption_common.py:75).
6. `generate_kpts_grid` regenerates the C-order MP grid for eigenvectors.h5
   (write_eigenvectors.py:156-173). **Implicit global assumption**: the k-axis
   order of psi_full_y/enk_full (GW SymMaps full-BZ order), the flat-q order of
   V_qmunu/W0_qmunu, and C-order (ix,iy,iz) must all coincide — the FFT
   convolution and the `reshape(nkx,nky,nkz,…)` at bse_io.py:911 silently
   depend on it (validated implicitly by the Si-vs-BGW eigenvalue agreement).

## Suspects

### B — bugs (explicit failure math)

- **B1 — exchange kernel is k-block-diagonal in every implementation.**
  Per-element (from `S = einsum("kcvN,bcvk->bNk", M_Y, X)` — k appears in both
  operands AND the output ⇒ batch, never summed; then `U = einsum("MN,bNk->bMk")`,
  `VX = einsum("kcvM,bMk->bcvk")`):
  `(VX)[b,c,v,k] = (1/Nk) Σ_{μν} M*_cvk(μ) V_0(μν) Σ_{c'v'} M_c'v'k(ν) X[b,c',v',k]`.
  The physical exchange (Rohlfing-Louie Eq. 42; BGW bsexmtx)
  `⟨cvk|K^x|c'v'k'⟩ = (1/Nk) Σ_{μν} M*_cvk(μ) V_0(μν) M_c'v'k'(ν)` is **dense in
  (k,k')** — both pair densities are lattice-periodic (Q=0) and couple through
  the same v(G). The code keeps only the k'=k block, suppressing the exchange by
  ~1/Nk for delocalised excitons. Sites: bse_jax.py:108-121, bse_serial.py:62-64,
  bse_simple.py:101-131, bse_ring_comm.py:276-300 (S_total keeps k; psum/psum_scatter
  run only over mesh axes which shard c/v/μ/ν, never k). All four agree with each
  other, and `ring_matvec_correctness_check` + test_bse only compare them against
  each other — no independent dense reference exists. The Si-SOC validation
  (STATUS.md: ~3 meV eigenvalues, Haydock peaks 1.5% at η=150 meV) is weakly
  sensitive: for Si the G=0 exchange element vanishes by orthogonality and the
  lowest manifold is triplet-dominated. Wrong-output scenario: any system/state
  with significant exchange (singlet-triplet splitting, LT splitting) gets
  exchange ≈ diag-only ≈ 1/Nk of physical. Decisive check: build dense H for a
  2-k toy and compare K^x off-diagonal k-blocks against a direct quadrature (or
  BGW bsemat).
- **B2 — non-TDA B-block T-encode einsum output index `t` unbound.**
  `jnp.einsum("kvsM,bvksN->bMNtsk", …)` at bse_ring_comm.py:183
  (`_ring_sum_valence_second`) and 551 (`_encode_T_B_gather`): `t` appears in
  the output but in neither operand → ValueError at trace time (numpy/opt_einsum
  semantics). Should be `"kvtM,bvksN->bMNtsk"` (valence spinor index t).
  Failure: any `build_bse_ring_matvec_full` matvec with `include_W=True` — e.g.
  `python -m bse.bse_w_exact -i … ` **without** `--tda`/`--rpa` (its defaults) —
  crashes at first B-apply. Explains itself: non-TDA has only ever been run RPA.
- **B3 — sharded V_q0 reader is 8-D-only; current writer emits rank-3 flat-q.**
  `_read_vq0_sharded`: `n_rmu = dset.shape[6]` and
  `dset[0,0,0,0,0,0,μ0:μ1,ν0:ν1]` (bse_io.py:233-234, 250). Current gw_jax
  writes V_qmunu as (nq, μ, ν) (gw_output.py:202-208 "W0 placeholder … rank-3
  (sized from V_qmunu)"; gw_init.py:661-668). Failure: multi-device BSE on any
  fresh restart → `IndexError: tuple index out of range` at load. The header
  shim (bse_io.py:389-411) learned flat-q; this reader did not.
- **B4 — `_read_wq_sharded` flat-q branch demands `dset.attrs['kgrid']` that no
  writer sets.** bse_io.py:309-317 raises ValueError for a rank-3/6 dataset
  without the attr; the writer stores kgrid as a **top-level dataset**
  (tagged_arrays.py:76-82 via `write_attr` → `create_dataset`,
  _slab_io_allgather.py:120-137). The caller already resolved kgrid
  (bse_io.py:396-411 reads `f['kgrid']`) but never passes it down. Failure:
  same fresh-restart multi-device load, one line after B3 is fixed.
- **B5 — 1-device head injection applied before the layout shim.**
  `_load_ring_subset` calls `apply_q0_head_rank1` (bse_io.py:828) on the RAW
  on-disk array; that function indexes `V_qmunu.at[..., 0, 0, 0, :, :]`
  (head_correction.py:775-777) — 5 explicit indices, valid for the 8-D legacy
  (and coincidentally 6-D) layouts, **IndexError on rank-3 flat-q** (5 indices >
  3 dims). Triggered whenever the restart has `G0_mu_nu` + `vhead`/`whead`
  (i.e. every `do_screened` GW run). Fix direction: normalise to flat-q first,
  then inject at `[0, :, :]`. — Net effect of B3+B4+B5: **the entire BSE
  pipeline only runs against legacy-era restart files at HEAD**; every path is
  broken for files written by current gw_jax (sharded: B3→B4; 1-device: B5).
- **B6 — `n_val_pad`/`n_cond_pad` carry the UNPADDED counts.**
  `_pad_axis_to_multiple` returns `(padded_array, size_before_padding)`
  (bse_io.py:139-146: `size = x.shape[axis]` captured before `jnp.pad`, returned
  in both branches). Callers name it `n_val_pad` (bse_io.py:450-451, 899-900)
  and size trial vectors with it: `shape = (bs, nc_pad, nv_pad, nk)`,
  `n_flat = nc_pad·nv_pad·nk` (bse_lanczos.py:140-148), X in `_load_ring_subset`
  (bse_io.py:915-917), Haydock seed padding (absorption_haydock.py:189-198).
  Whenever `n_cond % grid_x ≠ 0` or `n_val % grid_y ≠ 0` the psi band axes ARE
  padded but X is NOT → einsum dim mismatch at trace ("kcsN,bcvk" c-extent
  conflict). Concrete: `--n-val 3 --n-cond 5` on a 2×2 mesh crashes; every
  validated run used divisible counts (4 or 8 on 2×2) so it never fired.
  Secondary hazard once fixed: eps pads are ZEROS, so padded transitions get
  ΔE = ±ε ordering artifacts unless masked (bse_kpm masks; Lanczos doesn't).
- **B7 — `_preview_lanczos` eqp branch uses CLI band counts, not loader-clamped.**
  bse_jax.py:254-261 slices `arange(n_occ_eff − n_val, …)` with the raw CLI
  `n_val` while the loader may have clamped (`bse_io.py:423-427`). If
  `--n-val` > available, `np.arange` gets a negative start → fancy-indexing
  wraps to the TOP bands (silently wrong ε_v) and/or eps-vs-X band-extent
  mismatch in the D term. `run_haydock` (absorption_haydock.py:177-185) shares
  the flaw; `davidson_absorption` uses `n_val_eff` (davidson_absorption.py:98-103)
  — the one careful copy of a thrice-duplicated block.

### D — dead (all call mechanisms grepped at HEAD: src, tests, tools, scripts, sandbox skills/runs, `-m` invocations, __all__ re-exports, getattr-style strings)

- `bse_jax.apply_bse_hamiltonian` + its private `apply_D`/`apply_V`/`apply_W`
  (bse_jax.py:67-160): grep `apply_bse_hamiltonian\b` → only def/__all__/context
  doc. Uses `lax.psum(axis_name)` with no shard_map wrapper anywhere ⇒ also
  uncallable as exported. (`compute_pair_amplitude` in the same file IS alive via
  test_bse.py:34-38.)
- `bse_ring_comm.apply_bse_hamiltonian_ring` (303-337): imported/re-exported by
  bse_jax only; nothing shard_maps it (production goes through
  `build_bse_ring_matvec`, which assembles from the same pieces).
- `bse_serial.symmetrize_W_q` (12-24): re-exported by bse_jax, zero calls.
  Physics consequence: W(q)=W(−q)† is assumed, never enforced, and Lanczos
  assumes Hermitian H.
- `bse_io.BSEData` (18-20): zero references.
- `test_bse.load_bse_data_from_restart` (42-168): reads `psi_l/psi_r/enk_l/enk_r`
  datasets and 8-D V_qmunu — datasets the canonical writer no longer produces
  (tagged_arrays.py:91-96 writes V_qmunu/G0/psi_full_y/enk_full only) ⇒ the
  module self-brands as a test but cannot load any current restart. Stale.
- `tests/archive/projects/test_isdf/sweep_12v12c_plot.py:9` imports
  `bse_isdf.pseudopoles_eval` — pre-rename package path, import would fail.

### C — conventions / cruft / refactor targets

- **C1 — default routing surprises**: bare `python -m bse.bse_jax -i …` runs
  **FEAST** with the **RPA kernel** (bse_jax.py:539-542: `use_rpa = args.rpa or
  not args.bse`); you must pass `--bse --lanczos --tda` to get the documented
  TDA screened solve. STATUS.md always spells this out, but the defaults invert
  the physical expectation.
- **C2 — 1-device path silently drops CLI solver knobs**: `_preview_lanczos`
  unsharded branch (bse_jax.py:294-328) calls `solve_bse` without `n_reorth`,
  `block_size`, `rtol`, `solver_kind`, or `matvec_kind` — so `--n-reorth -1`
  (documented as "the right default for spinor BSE", STATUS.md:15) is ignored on
  1 GPU and the hardcoded partial reorth window 10 (bse_lanczos.py:44) applies —
  the exact ghost-eigenvalue regime STATUS warns about. Clashes with the
  no-multi-GPU-gating policy; so does `absorption_haydock`'s hard ≥2-device
  requirement (absorption_haydock.py:158-161).
- **C3 — `--n-occ`/`--eqp` not forwarded to FEAST/KPM delegations**
  (bse_jax.py:516-536 kpm_argv, 543-604 feast argv contain neither; bse_feast
  has no such flags) — the default route can only use WFN-ifmax n_occ and DFT
  energies; convention #2/#5 of BGW_COMPARE silently unavailable there.
- **C4 — no singlet factor for nspinor=1** (see Spin section): fine for the
  spinor-default codebase, but nothing guards or documents a scalar-WFN run
  inside the matvec (kernel is then the triplet-like D+K^x−K^d with K^x
  unscaled). Worth an assert on nspinor or a factor-2 knob.
- **Redundancy** (violates no-redundancy rule): `compute_pair_amplitude` ×3
  (bse_jax.py:94, bse_serial.py:27, bse_preconditioner.py:44-46), `apply_D` ×2
  (+ inline delta_E in bse_simple:82, bse_ring_comm:434, 632, davidson helpers),
  eqp-override block ×3 (bse_jax.py:241-261, absorption_haydock.py:177-185,
  davidson_absorption.py:96-105), `_encode_T`/FFT-helper/W-decode blocks
  duplicated verbatim between `build_bse_ring_matvec` and
  `build_bse_ring_matvec_full` (bse_ring_comm.py:353-441 vs 501-640), two
  eigenvector writers (write_eigenvectors.py vs bse_io.write_eigenvectors_stream),
  two restart loaders in test_bse vs bse_io.
- **Weird**: `_load_ring_subset` builds its random probe X with the SAME PRNGKey
  for real and imaginary parts → X = (1+1j)·N (bse_io.py:914-917); the smoke
  test does the same for ψ (bse_ring_comm.py:867-874). Harmless for smoke
  checks, but a correlated probe can mask phase-sensitive matvec errors.
  `--px/--py` look like production mesh controls but only reach the
  `--ring-check` path; the real mesh is `create_mesh_2d()` over all devices.
  `_ring_sum_valence` einsum string contains a stray space: `"kv sN,bcvk->bcksN"`
  (bse_ring_comm.py:91, also bse_jax.py:142) — legal (spaces ignored) but reads
  like a typo next to the genuinely broken B-side string.
