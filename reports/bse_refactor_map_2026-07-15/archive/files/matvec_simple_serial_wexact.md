# src/bse/{bse_simple,bse_serial,bse_w_exact}.py — deep-read notes (184 + 99 + 178 LOC)

Audit date: 2026-07-15, lorrax_D checkout, branch agent/slate-linalg-ffi (e18d0e5).

## Purpose

Three files in the BSE matvec family:

- **`bse_serial.py`** — single-device reference kernels: the TDA BSE
  Hamiltonian `H = D + V − W` applied to a trial block `X(b,c,v,k)`, plus
  the pair-amplitude and W(q)-hermitization helpers. Ground truth for the
  distributed implementations (ring correctness check compares against it).
- **`bse_simple.py`** — the production distributed TDA matvec: plain
  `jax.jit` + einsums + `with_sharding_constraint` on the (x,y) mesh; XLA's
  SPMD partitioner generates all collectives (no shard_map/ppermute). This
  is `--matvec-kind=simple`, the default per STATUS.md:16.
- **`bse_w_exact.py`** — standalone CLI (`python -m bse.bse_w_exact`) that
  computes columns of the "exact" screened interaction `W_c(ω)` in the
  ISDF centroid basis via shifted GMRES solves with the Casida/BSE matrix
  (TDA or full non-TDA), one column per unit source `δ_{ν,ν0}`. Debug/
  reference tool for the pseudopole-W program (`bse_pseudopoles.py`).

Physics as written (per-element, code index names; `nk = nkx·nky·nkz`,
`√nk = sqrt_nk`, μ/ν = ISDF centroids, s/t = spinor, b = trial batch):

```
D term   (bse_serial:56-57, bse_simple:82-84):
  ΔE[c,v,k]      = eps_c[k,c] − eps_v[k,v]           (Ry; QP energies)
  D[b,c,v,k]     = ΔE[c,v,k] · X[b,c,v,k]

pair amplitude (bse_serial:27-29 compute_pair_amplitude; simple builds
M_Y/M_X inline at :89-92/:118-121):
  M[k,c,v,μ]     = Σ_s conj(ψ_c[k,c,s,μ]) · ψ_v[k,v,s,μ]

V term  (bse_serial:62-64; bse_simple:101-131 — identical index math):
  S[b,ν,k]       = Σ_{c,v} M[k,c,v,ν] · X[b,c,v,k] / √nk     ← k is a BATCH axis
  U[b,μ,k]       = Σ_ν V_q0[μ,ν] · S[b,ν,k]
  HX_V[b,c,v,k]  = Σ_μ conj(M[k,c,v,μ]) · U[b,μ,k] / √nk

W term  (bse_serial:69-78; bse_simple:141-173 — identical index math):
  T[b,μ,ν,t,s,k] = Σ_{c,v} ψ_c[k,c,t,μ] · conj(ψ_v[k,v,s,ν]) · X[b,c,v,k]
  T_R            = ifftn_{k→R}(T, axes=k-grid, norm='ortho')
  U_R            = W_R[μ,ν,R] · T_R[b,μ,ν,t,s,R]     (pointwise; W spinor-scalar)
  U              = fftn_{R→k}(U_R, norm='ortho')
                 ⇒ U[b,μ,ν,t,s,k] = (1/√nk) Σ_{k'} W_{k−k'}[μ,ν] · T[b,μ,ν,t,s,k']
  A[b,c,ν,s,k]   = Σ_{t,μ} conj(ψ_c[k,c,t,μ]) · U[b,μ,ν,t,s,k]
  HX_W[b,c,v,k]  = Σ_{s,ν} ψ_v[k,v,s,ν] · A[b,c,ν,s,k] / √nk

  H X = D + HX_V − HX_W                    (bse_serial:80, bse_simple:175)
```

`bse_serial` computes `W_R = ifftn(W_q, axes=(2,3,4), 'ortho')` inside every
call (line 73); `bse_simple` takes precomputed `W_R` as an argument (callers
do the IFFT once: bse_lanczos.py:172-180, absorption_haydock.py:210-225,
davidson_absorption.py:124-127, test_davidson_bse.py:106-111).

`bse_w_exact.main` per column ν0 at complex shift `z = (ω + iη)/ry_to_ev`:

```
r[0,ν,k]   = δ_{ν,ν0}  ∀k                                  (bse_w_exact:127-129)
f[0,c,v,k] = (1/√nk) Σ_μ conj(M_X[k,c,v,μ]) Σ_ν V_q0[μ,ν] r[0,ν,k]
             (gen = build_realspace_random_transition_generator,
              bse_ring_comm:742-765)
x          = GMRES solve of (z − H) x = f        (TDA)      (bse_w_exact:139-147)
             or (z − H_full) x = [f; f]          (non-TDA), s = x[0] + x[1]
Wc[:,ν0]   = d_μ[0,:],  d_μ[b,μ] = Σ_ν V_q0[μ,ν] (1/√nk) Σ_k Σ_{c,v}
             M[k,c,v,ν] · s[b,c,v,k]
             (snapshot_op = build_density_snapshot_operator,
              bse_ring_comm:789-843)
```

i.e. `Wc(ω) ≈ V·χ(ω)·V` columns read out in the (μ) centroid basis, with the
resonant+antiresonant structure supplied by the full (non-TDA) matvec.

Category: **physics: BSE matvec kernels (serial reference + production
simple) and a diagnostics CLI (exact-W columns)**.

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox runs/skills/scripts)

| symbol | callers (grep evidence) |
|---|---|
| `bse_simple.build_bse_simple_matvec` | `bse_lanczos.py:157-158` (solve_bse_sharded, `matvec_kind=="simple"`), `absorption_haydock.py:203-204`, `davidson_absorption.py:39,122`, `test_davidson_bse.py:42,101`, referenced by `bse_davidson_helpers.py:6,92,184` (docs/comments) |
| `bse_serial.apply_bse_hamiltonian_single_device` | `bse_ring_comm.py:25,934` (ring correctness reference), `bse_lanczos.py:26,57,77` (single-device solve_bse), `feast_sweep.py:27,102`, `bse_jax.py:29,186` (re-export + `_main_random_demo`), `test_bse.py:35,191,202,276` |
| `bse_serial.compute_pair_amplitude` | `bse_ring_comm.py:25,958`, `bse_jax.py` (re-exported name but defines its own copy at :94-95), `test_bse.py:37,226` |
| `bse_serial.apply_D` | `bse_ring_comm.py:25,959,977` (correctness check only) |
| `bse_serial.apply_bse_hamiltonian_single_device_jit` | **import/re-export only**: `bse_jax.py:30,49`; zero call sites repo-wide (`grep -rn single_device_jit src tests` → def + import + `__all__` only) |
| `bse_serial.symmetrize_W_q` | **import/re-export only**: `bse_jax.py:31,63`; zero call sites repo-wide (`grep -rn 'symmetrize_W_q(' src tests tools scripts` → definition only) |
| `bse_w_exact.main` | `if __name__ == "__main__"` (:177-178) → `python -m bse.bse_w_exact`. No entry in pyproject.toml (`lorrax-bse` removed, see pyproject.toml:36-41). No invocations found in sandbox `runs/**/*.sh|yaml`, `skills/`, `scripts/` (targeted grep → empty). Referenced by `pseudopoles_eval.py:23-29,131` (output-format contract) and `tests/archive/projects/test_isdf/sweep_12v12c_plot.py:38` ("Exact Wc(0) columns H5 (from bse_w_exact)"); listed in `docs/architecture/codebase.md:115` |

`src/bse/__init__.py` is a bare docstring (no re-exports). Sandbox runs
contain production artifacts referencing `bse_serial.py` line numbers
(`runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse/compile_summary.json:625-633`,
xla_dump modules) — the serial kernel executes inside solve_bse in real runs.

## Function tables

### bse_serial.py

| function | lines | role |
|---|---|---|
| `symmetrize_W_q(W_q, nkx, nky, nkz)` | 12-24 | `W(q) ← (W(q) + W(−q)†)/2` with per-element `out[μ,ν,q] = (W[μ,ν,q] + conj(W[ν,μ,(n−q)%n]))/2`; correct hermitization for real-symmetric W(r,r'). **No callers.** |
| `compute_pair_amplitude(psi_c, psi_v)` | 27-29 | `M[k,c,v,m] = Σ_s conj(ψ_c[k,c,s,m])·ψ_v[k,v,s,m]` (spinor-traced) |
| `apply_D(X, eps_c, eps_v)` | 32-35 | `(ε_c−ε_v)·X` via `bse_preconditioner.energy_diff_cv_k` ((nc,nv,nk) layout) |
| `apply_bse_hamiltonian_single_device(...)` | 38-80 | full TDA H·X as in Purpose; `include_W=False` returns `D + V` (RPA kernel); re-derives ΔE inline (56-57) instead of calling `apply_D` |
| `apply_bse_hamiltonian_single_device_jit(...)` | 83-99 | jit wrapper, `static_argnums=(7,8,9,10)` = nkx,nky,nkz,include_W. **No callers.** |

### bse_simple.py

| function | lines | role |
|---|---|---|
| `build_bse_simple_matvec(mesh_xy, nkx, nky, nkz, *, include_W=True)` | 46-184 | factory; returns one jit'd 9-arg matvec `(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0) → HX` — same protocol as `build_bse_ring_matvec`, so `--matvec-kind` can switch implementations |
| inner `_matvec` | 77-175 | D/V/W terms per Purpose; `include_W` is build-time static (V-only ⇒ RPA kernel) |

Details:
- FFT helpers built once at factory time (lines 71-75):
  `make_sharded_{i,}fftn_3d(mesh_xy, _T_8d_spec, _T_8d_spec, axes=(5,6,7),
  norm='ortho')` with `_T_8d_spec = P(None,'x','y',None,None,None,None,None)`
  — k-grid axes replicated, so the 3D cuFFT is device-local (single
  CustomCall inside shard_map). Docstring (61-65) records that fusing
  ifft·mul·fft was benchmarked slower; commit beabb1a.
- `sqrt_nk` is a strong-typed f64 constant (line 69) — complex64 inputs
  would silently promote to complex128 under x64 (serial version uses
  `X.real.dtype`, line 60). fp32 GMRES paths use the ring matvec, so
  currently latent.
- Comment lines 94-97 documents the conjugation convention: forward
  contraction uses `M_Y` itself (`conj(ψ_c)·ψ_v·X`, Rohlfing-Louie side),
  back-contraction uses `conj(M_X)`; matches apply_V_ring and serial.

### bse_w_exact.py

| function | lines | role |
|---|---|---|
| `_create_mesh_xy(px, py)` | 28-34 | builds (px,py) Mesh from `jax.devices()[:px·py]`; raises if too few devices |
| `_parse_cols(col_str, n_mu, n_cols, seed)` | 37-47 | column selection: explicit comma list (silently drops out-of-range) → random `n_cols` sample (`default_rng(seed)`) → all μ |
| `main(argv=None)` | 50-174 | argparse CLI; load restart bundle sharded; build TDA (`build_bse_ring_matvec`) or full (`build_bse_ring_matvec_full`) matvec per `--tda`; `include_W = not --rpa`; per-column shifted GMRES + density snapshot; writes `Wc_exact.h5` (`columns` int32, `Wc` (ncols, n_rmu) complex, attrs omega_ev/eta_ev/ry_to_ev/use_tda/include_W/gmres_max_iter/gmres_tol/seed) |

## Flags / CLI args consumed

`bse_serial.py`, `bse_simple.py`: none (library modules; no config reads).

`bse_w_exact.py` (argparse, lines 53-78):

| flag | meaning | default |
|---|---|---|
| `-i/--input` | cohsex input file (restart discovery via `_find_restart_file`; `wfn_file` key read indirectly by loader) | required |
| `--n-val`, `--n-cond` | BSE window sizes (capped at availability by loader) | 4, 4 |
| `--px`, `--py` | device mesh shape | 1, 1 |
| `--omega-ev`, `--eta-ev` | real frequency and broadening, eV; `z=(ω+iη)/ry_to_ev` | 0.0, 0.0 |
| `--cols` | explicit comma-separated 0-based μ column indices | None |
| `--n-cols` | random sample size (used only if `--cols` absent) | None (= all) |
| `--seed` | RNG seed for `--n-cols` | 0 |
| `--gmres-max-iter`, `--gmres-tol` | shifted-solve controls | 10, 1e-2 |
| `--gmres-fp32` | "Use FP32 data/GMRES" — **only casts the RHS dtype** (lines 120-121, 134-136); `data` stays complex128 (no `_build_gmres_data_fp32` call, unlike bse_feast.py:514, bse_kpm.py:150-155) so the matvec runs in f64 | off |
| `--rpa` | drop W term (D+V kernel only) | off |
| `--tda` | TDA matvec instead of full non-TDA | off (full) |
| `--ry-to-ev` | unit conversion | 13.6056980659 (`bse_feast.RY_TO_EV_DEFAULT`) |
| `--out` | output H5 path | `Wc_exact.h5` |

Units: LORRAX-internal Ry throughout the solve; eV appears only at the CLI
boundary (`--omega-ev/--eta-ev` divided by `ry_to_ev` at line 118) and in
the H5 attrs. No BGW valence-axis flip in any of these files (loader
`bse_io.load_bse_data_from_restart_sharded` returns LORRAX-internal band
order; the flip lives in `bse_io.write_eigenvectors_stream`, per
STATUS.md "Index ordering").

## Sharding / PartitionSpec assumptions

All from `bse_ring_comm.make_bse_shardings(mesh_xy)` (bse_ring_comm.py:46-63)
on the 2D `("x","y")` mesh:

| array | spec | notes |
|---|---|---|
| `X` (b,c,v,k) | `P(None,'x','y',None)` | c on x, v on y, k replicated |
| `X_full` (2,b,c,v,k) | `P(None,None,'x','y',None)` | non-TDA stacked |
| `psi_*_X` (k,b,s,μ) | `P(None,None,None,'x')` | μ on x; dual copy |
| `psi_*_Y` (k,b,s,ν) | `P(None,None,None,'y')` | ν on y; dual copy |
| `V_q0` (μ,ν) | `P('x','y')` | tiled |
| `W_R` (μ,ν,kx,ky,kz) | `P('x','y',None,None,None)` | k-grid axes replicated (FFT locality requirement) |
| `T/U` (b,μ,ν,t,s,k) | `P(None,'x','y',None,None,None)`; 8D reshape `P(None,'x','y',None,None,None,None,None)` | bse_simple:71 |
| `S` (b,ν,k) | `P(None,'y',None)` | V-term intermediate; also the density-space vector layout in w_exact |
| `U_mu` (b,μ,k) | `P(None,'x',None)` | |
| `A` (b,c,ν,s,k) | `P(None,'x','y',None,None)` | |
| `eps_c/eps_v` (nk,nb) | `P(None,None)` | replicated |
| `d_mu` (b,μ) | `P(None,'x')` | snapshot output |

`bse_simple` pins these via `jax.jit(in_shardings=..., out_shardings=sh.X)`
(lines 177-184) plus interior `with_sharding_constraint`s; the docstring
(17-30) states the specs are intentionally identical to the ring
implementation. The V-term reduce order follows the
`ppm_sigma._make_project_ri_reduce_scatter` psum_scatter idea (docstring
27-30). `bse_serial` is sharding-free (single device). `bse_w_exact`
device_puts the source `r` to `sh.S` (:129) and constrains `f`/`rhs`/`s` to
`sh.X`/`sh.X_full` (:132-137, 153); band-padding divisibility (c by px, v by
py, μ/ν by mesh product) is guaranteed upstream by
`bse_io.load_bse_data_from_restart_sharded` (`padded_mu_extent`,
`_pad_axis_to_multiple`, bse_io.py:442-458) and re-checked by the transition
generator (bse_ring_comm.py:736-737).

## Host vs device residency

- All heavy tensors (ψ dual copies, V_q0, W_q/W_R, X blocks, GMRES Krylov
  buffers) are device-resident, sharded as above. No io_callback use.
- `bse_w_exact`: per-column `Wc` vector pulled to host via
  `jax.device_get(w_c[0])` (:156), accumulated in a Python list, stacked
  with numpy and written by h5py on the host (:159-171). Columns loop is
  Python-level (one GMRES compile reused via `_GMRES_SOLVER_CACHE` keyed on
  (max_iter, tol, dtype), bse_feast.py:128-130).
- `bse_serial` assumes everything fits one device (verification scale).

## TDA vs full BSE

- `bse_serial`, `bse_simple`: **TDA-only** — single-block `X(b,c,v,k)`,
  Hermitian `H = D + V − W`. No coupling (B) blocks anywhere in these files.
- `bse_w_exact`: both. `--tda` → `build_bse_ring_matvec`; default full
  non-TDA → `build_bse_ring_matvec_full` (2-block X, `H_full` with ±
  structure; preconditioner diag stacked `[diag_h, −diag_h]`,
  bse_feast.py:115-117). Response channel: TDA reads `V·(d·X)`; non-TDA sums
  `s = x[0] + x[1]` (:149-152) then applies the **same** TDA snapshot
  operator — note `bse_pseudopoles._compute_density_snapshots` (:139-153)
  documents the non-TDA readout as `w = V(dX + d*Y)` and uses a distinct
  `readout_full` operator; `bse_w_exact` instead applies `V·d·(X+Y)` with no
  conjugated pair density on the Y block. Whether the two agree depends on
  the reality of the pair densities; not independently verified here (the
  historical SOS cross-check script `compare_w_exact_to_sos.py` referenced
  in pseudopoles_eval.py:29 is absent at HEAD).

## Spin / nspinor

- Pair amplitudes trace the spinor axis (`Σ_s conj(ψ_c)ψ_v`); works for
  nspinor ∈ {1,2}; nspinor read from `psi_c.shape[2]` (bse_serial:53,
  bse_simple via T shape).
- W term keeps two spinor axes (t on the ψ_c/μ side, s on the ψ_v/ν side)
  through the FFT and decodes both — W is spinor-scalar (charge channel
  only; consistent with the bispinor screened-W roadmap where transverse
  channels exist only in GW Σ^B, not BSE).
- No singlet factor: `H = D + V − W`, not Henneke's spin-unpolarized
  singlet `D + 2V_A − W_A` (Henneke context doc line 165 and Eq. 4-4 block;
  gpt5.2suggestion.md §4 also writes `+2(VX)`). For nspinor=2 (all LORRAX
  FR-pseudopotential WFNs) factor 1 with explicit spinor sums is the
  correct convention; a future nspinor=1 spin-singlet run would need the
  2× (convention hazard, not a present bug).

## Coupling to gw/ and isdf/

- `bse_simple` → `common.fft_helpers.make_sharded_{i,}fftn_3d` (shared with
  gw flat-k kernels) and `bse_ring_comm.make_bse_shardings`.
- `bse_serial` → `bse_preconditioner.energy_diff_cv_k` only.
- `bse_w_exact` → `bse_feast` (GMRES + preconditioner diag + RY_TO_EV),
  `bse_io` (restart loader; reads gw_jax restart datasets `V_qmunu`,
  `W0_qmunu` (`W0_ready` attr), `psi_full_y`, `enk_full`, `G0_mu_nu`,
  `vhead`/`whead`; head injection calls
  `gw.head_correction.apply_q0_head_rank1_sharded`, bse_io.py:504-507),
  `bse_ring_comm` (matvecs + transition generator + snapshot),
  `common.timing`. No direct isdf/ imports — ISDF quantities arrive via the
  restart file.

## Suspects

### Bugs

- **`bse_w_exact.main` crashes with `KeyError: 'W_R'` on the first solve —
  in every mode.** The loader's return dict (bse_io.py:515-536) has no
  `"W_R"`; `main` never adds it; `gmres_solve_sharded_jit` →
  `_apply_shifted_matvec` unconditionally reads `data["W_R"]`
  (bse_feast.py:69, reached from :139 during trace) — before `include_W` can
  matter, so `--rpa` crashes too. Every other consumer sets it after load
  (`bse_feast.py:505-507`, `bse_pseudopoles.py:252-254`,
  `feast_zolo_sweep.py:123-124`, `feast_ellipse_mixed_sweep.py:98-99`,
  `bse_kpm.py:153-155`); `git log -S 'W_R' -- src/bse/bse_w_exact.py` is
  empty — the file never set it in available history. Fix is one line:
  `data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2,3,4), norm="ortho")`.
  Consistent with the a0da0a5→fe5e3e8 delete/restore cycle ("restore ...
  W-exact", restored as-was without a run).
- **V (exchange) term is applied k-block-diagonally in every
  implementation — the Σ_k' of the reference algorithm is missing.**
  In `S[b,ν,k] = Σ_{c,v} M[k,c,v,ν]·X[b,c,v,k]` (einsum
  `'kcvN,bcvk->bNk'`, bse_serial:62, bse_simple:101-104; same structure in
  bse_jax.apply_V:108-119 and apply_V_ring bse_ring_comm:269-297) the index
  k appears in both operands *and* the output ⇒ batch axis, never summed;
  the back-contraction `'kcvM,bMk->bcvk'` reuses the same k. Net operator:
  `V_code[cvk, c'v'k'] = δ_{kk'} (1/nk) Σ_{μν} conj(M[k,cv,μ]) V_q0[μν]
  M[k,c'v',ν]`. The module's own reference (Henneke 2020, context copy) Eq.
  (2-17)/(4-3) defines `V_A(i_v i_c k, j_v j_c k') = (1/N_k) Σ_{μν}
  ū_{ick}(μ)u_{ivk}(μ) Ṽ_{μν} ū_{jvk'}(ν)u_{jck'}(ν)` — dense in (k,k') —
  and the regrouped matvec Eq. (4-5) contains an explicit `Σ_{k'}` inside
  the braces; the module's design note gpt5.2suggestion.md §3.1 likewise
  writes `p(b,ν) = Σ_k P(b,k,ν)` ("Now sum over k locally"). BGW's kernel-x
  is also dense in (k,k'). Effect: exchange/local-field coupling between
  different k-blocks is dropped; for k-coherent low excitons the
  expectation error is second order in the k-variation of `M_k·a` (which is
  consistent with the reported ~3 meV Si eigenvalue agreement and 1.5%
  Haydock peak match, both exchange-insensitive at that scale), but
  observables that hinge on the k-summed exchange (singlet–triplet/
  dark–bright structure, RPA resummation inside `bse_w_exact`'s
  `(z−D−V)^{-1}` — each bubble insertion is suppressed to its k-diagonal)
  are structurally wrong. All five implementations share the pattern and
  are mutually consistency-checked, so no internal test can catch it.
  (Conjugation sides — code contracts `M = conj(ψ_c)ψ_v` with X and
  back-projects `conj(M)` — are the Rohlfing-Louie convention, the
  elementwise conjugate of Henneke's (4-5); with real V_q0/W kernels the
  whole H is the transpose of the Henneke-convention H, so that part is a
  convention, not a bug.)
- **`bse_w_exact` treats the padded μ extent as the logical one**:
  `n_rmu = int(data["V_q0"].shape[0])` (:98) is `n_rmu_pad`
  (loader pads to the mesh-product multiple with zero rows,
  bse_io.py:442-460, trim=False) while the logical count sits unused in
  `data["n_rmu"]`. `--n-cols` random sampling and the
  `"Computing ... out of N_mu"` print (:115-116) therefore include padding
  columns whenever `n_rmu % (px·py) ≠ 0`; those columns give identically
  zero `Wc` columns (zero V_q0 column → zero rhs → zero solution) silently
  wasting solves and skewing downstream column statistics.

### Dead / test-only

- `bse_serial.symmetrize_W_q` (12-24): zero call sites; only imported and
  re-exported by bse_jax.py:31,63. Correct math, orphaned utility.
- `bse_serial.apply_bse_hamiltonian_single_device_jit` (83-99): zero call
  sites; import/`__all__` only (bse_jax.py:30,49). Callers jit the plain
  version themselves (bse_lanczos.py:54-60).
- `bse_w_exact` unused imports: `dataclass` (:4), `math` (:5),
  `NamedSharding`, `P` (:11) — none referenced in the file body; plus two
  separate `from .bse_feast import` statements (:13, :22).
- `test_bse.py` / `test_davidson_bse.py` (consumers of these modules) are
  manual drivers, not pytest-collected (a0da0a5 commit message: "test_bse.py
  — old test script, not in pytest path").

### Redundancy / refactor

- Four+ parallel V/W matvec implementations (serial, simple, ring,
  ring-full, bse_jax.apply_V/apply_W with hand-rolled psums) encode the
  same contractions; `bse_jax.py:94-95` even duplicates
  `compute_pair_amplitude` while importing the serial one's name for
  re-export. Per the simple-module docstring the ring versions are
  hand-rolled collectives the XLA partitioner generates anyway; STATUS.md
  keeps ring "for memory-tight runs" — a candidate for the sandbox's
  no-parallel-paths rule once memory parity is demonstrated.
- `apply_bse_hamiltonian_single_device` re-derives ΔE inline (:56-57)
  instead of using its sibling `apply_D` (:32-35), and re-FFTs `W_q → W_R`
  on every call (:73) — wasteful inside Lanczos loops (serial solve_bse
  jits it per-iteration with W_q as an argument, bse_lanczos.py:55-60).
- `bse_w_exact --gmres-fp32` half-implemented: casts only the RHS; the
  matvec data stays c128 (contrast bse_feast.py:512-518 which builds
  `_build_gmres_data_fp32(data)` + fp32 `W_R` + fp32 diag). Little memory
  or speed benefit is actually delivered; help text overpromises.

### Weird

- `pseudopoles_eval.py:28` documents "`bse_w_exact.py --write-kind Wc`" —
  no such flag exists in bse_w_exact (stale doc in a neighbor file).
- Neighbor-file corroboration of the restore-without-validation story:
  `bse_pseudopoles.py:32` imports `build_density_readout_operator_full`
  from `.bse_ring_comm`, which is **not defined anywhere at HEAD**
  (`grep -rn 'def build_density_readout_operator_full' src` → empty) — the
  a0da0a5 commit message itself called bse_pseudopoles "broken on main
  (import error)". The bse_w_exact W_R KeyError is the same class of rot.
- `bse_simple`'s f64 `sqrt_nk` constant (line 69) vs serial's
  dtype-following version (line 60) — silent c64→c128 promotion hazard if
  the simple matvec is ever used on fp32 data.
