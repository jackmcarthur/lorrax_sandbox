# src/bse/bse_feast.py — deep-read notes (1321 LOC)

Audit date: 2026-07-15, lorrax_D checkout. Named audit base e18d0e5
(agent/slate-linalg-ffi); working checkout is adc2197 (its direct parent) —
`git diff adc2197 e18d0e5 -- src/bse/` is empty, so these notes hold at both.

## Purpose

FEAST interior-eigenvalue solver for the sharded ISDF-BSE Hamiltonian:
spectral-bound estimation (short Lanczos), energy-window definition (eV),
contour quadrature (ellipse-trapezoid or Zolotarev rational), the
GMRES-based FEAST filter + Rayleigh-Ritz extraction, and a stdout
"physics-style report". Despite the docstring calling it "setup
utilities", this is the **default solver path** of the BSE driver:
`bse_jax.main` dispatches here whenever `--lanczos` is *not* given
(bse_jax.py:539-543).

Physics as written in code (explicit indices):

```
H (TDA, from bse_ring_comm._matvec_impl):  H X = D_term + V_term − W_term
  D_term[b,c,v,k] = (eps_c[k,c] − eps_v[k,v]) · X[b,c,v,k]
FEAST projector:  P = (1/2πi) ∮ (zI − H)⁻¹ dz  (solvers/quadrature.py:30-31)
Filter accumulation (_get_feast_runner, lines 243-256), y_j = (z_j/s − H)⁻¹ x,
  s = ry_to_ev (nodes/weights are in eV, H in Ry; both z and w divided by s):
    TDA  (use_conjugate_symmetry): filt += 2·Re(w_j · y_j)      [elementwise]
    full (non-TDA):                filt += w_j · y_j  over nodes ∪ conj(nodes)
GMRES (right-preconditioned, _get_gmres_solver lines 132-196):
  M⁻¹ = 1/(z − diag_h);  x0 = M⁻¹b;  z_k = M⁻¹v_k;  w = (z − H)z_k;
  Arnoldi h = ⟨V_i, w⟩;  y from normal equations (HᴴH + jitter·I)y = Hᴴg;
  x = x0 + Σ_k y_k Z_k;  rel = ‖g − Hy‖/‖r0‖
Jacobi diagonal (build_preconditioner_diagonal_sharded, lines 90-110):
  M_X[k,c,v,m] = Σ_s conj(psi_c_X[k,c,s,m])·psi_v_X[k,v,s,m]   (spin-traced)
  V_diag[k,c,v] = Σ_{M,N} conj(M_X[k,c,v,M])·V_q0[M,N]·M_Y[k,c,v,N] / nk
  rho_c[k,c,m] = Σ_s |psi_c_X[k,c,s,m]|²;  rho_v likewise on psi_v_Y
  W_diag[k,c,v] = Σ_{M,N} rho_c[k,c,M]·W_q[M,N,0,0,0]·rho_v[k,v,N] / nk
  diag_h = ΔE + V_diag − W_diag,  ΔE(c,v,k) = eps_c.T[:,None,:] − eps_v.T[None,:,:]
Rayleigh-Ritz (_rayleigh_ritz, lines 349-412), rows of V_flat are vectors:
  S[i,j] = Σ_x conj(V[i,x])V[j,x];  H[i,j] = ⟨v_i|Hv_j⟩;  G[i,j] = ⟨Hv_i|Hv_j⟩
  whitening W = U_S · diag(1/√max(s, s_floor)), s_floor = s_cutoff·max(s)
  A = Wᴴ H W → eigh (TDA) / eig-then-Re (full);  coeffs = W·evecs
  rel_res_i² = Re[cᴴ(G − 2λH + λ²S)c] / Re[cᴴSc]      (valid for real λ)
Zolotarev step (_zolotarev_step_poles_weights, lines 858-913):
  h_step(λ) = 1/2 + Σ_j 2Re[w_j/(z_j − λ)];  z_j = edge + iρ√d_j;
  w_j = −β_j·ρ/4;  d_j = ε²sn²/cn² at u = (2j−1)K'(1−ε²)/2n, ε = 1/G,
  G = max(|λ_min−edge|, λ_max−edge)/ρ;  indicator = step(a) − step(b)
  (the 1/2 constants cancel between edges, feast_zolotarev_quadrature:926-928)
Lanczos bounds (estimate_spectral_bounds_sharded, lines 777-826):
  α = Re⟨q, Hq⟩;  z ← Hq − αq − β_prev q_prev;  β = ‖z‖;  E_max from
  eigvalsh(tridiag);  adaptive stop when |ΔE_max| ≤ atol + rtol·|E_max| for
  `patience` consecutive steps;  E_min = min(eps_c[:, :n_cond]) − max(eps_v[:, :n_val]);
  non-TDA: E_min := −|E_max| (line 825-826)
```

Units: eps and H in Ry; windows, quadrature nodes/weights, and all printed
eigenvalues in eV via `ry_to_ev = 13.6056980659` (line 29; matches the BGW
factor in src/bse/STATUS.md "Index ordering" item 2). The valence axis here
is LORRAX-internal (v=0 = deepest requested valence band); no BGW flip
happens in this module — the flip lives in `bse_io.write_eigenvectors_stream`.

Category: **pipeline stage** (BSE eigensolver, default via bse_jax) with a
diagnostics/report layer bolted on.

## Entry points (grep over src/, tests/, tools/, scripts/, docs/, sandbox)

| symbol | callers (grep evidence) |
|---|---|
| `main` | `src/bse/bse_jax.py:540-543` (default dispatch when `--lanczos` absent — the primary production path); `__main__` guard line 1319-1320; documented `uv run python -m bse.bse_feast` in `src/bse/context/feast_accuracy_notes.md:75`; sandbox patch copy `lorrax_sandbox/scripts/bse_src_patches/bse_jax.py:392-395` |
| `run_feast_ritz` | internal `main` (line 1282); `src/bse/feast_sweep.py:31,173` |
| `estimate_spectral_bounds_sharded` | `bse_kpm.py:28`, `feast_sweep.py:33`, `feast_zolo_sweep.py:29`, `feast_ellipse_mixed_sweep.py:25`, `bse_pseudopoles.py:23` |
| `build_preconditioner_diagonal_sharded` | `bse_w_exact.py:22`, `bse_pseudopoles.py:22`, `feast_zolo_sweep.py:30`, `feast_ellipse_mixed_sweep.py:26` |
| `gmres_solve_sharded_jit` | `bse_w_exact.py:13,139` |
| `_get_feast_runner` (private) | reach-ins: `bse_pseudopoles.py:19`, `feast_zolo_sweep.py:31`, `feast_ellipse_mixed_sweep.py:27` |
| `_build_gmres_data_fp32` (private) | `bse_kpm.py:28`, `bse_pseudopoles.py:20` |
| `_cast_with_sharding` (private) | `bse_pseudopoles.py:21` |
| `_rayleigh_ritz`, `_build_ritz_vectors` (private) | `feast_zolo_sweep.py:32-33`, `feast_ellipse_mixed_sweep.py:28-29` |
| `_zolotarev_step_poles_weights` (private) | `feast_zolo_sweep.py:34` |
| `feast_zolotarev_quadrature` | `bse_pseudopoles.py:17`; internal (592) |
| `feast_ellipse_quadrature` | `bse_pseudopoles.py:16`; internal (592, 1243) |
| `_create_mesh_xy` (private) | `bse_kpm.py:28,361`, `feast_sweep.py:32,570` |
| `build_default_windows_eV` | `bse_pseudopoles.py:24,618`; internal (1197) |
| `_parse_window_arg` (private) | `bse_pseudopoles.py:25,645-646`; internal (1227-1228) |
| `WindowSpec`, `RY_TO_EV_DEFAULT` | all four sweep/pseudopole modules + `bse_w_exact.py:13` |
| `QuadratureSpec`, `RitzResult`, `format_feast_report`, `_apply_shifted_matvec`, `_get_gmres_solver`, `_build_gmres_data_fp32` | module-internal only (repo-wide grep finds no external use of these names) |

No pytest coverage: `grep -rn feast tests/ src/bse/test_bse.py src/bse/test_davidson_bse.py` → empty.
`src/bse/__init__.py` is a bare docstring — no re-exports. No hits in
depth-≤3 `runs/**/*.sh` (38 files scanned).

## Function table

| function | lines | role |
|---|---|---|
| `WindowSpec` / `QuadratureSpec` | 37-52 | frozen dataclasses: eV window; nodes+weights+type per window |
| `_apply_shifted_matvec(matvec, x, z, data)` | 55-72 | `z·x − H·x`; unpacks the 8 data-dict arrays into the matvec positional signature |
| `build_preconditioner_diagonal_sharded(data, mesh_xy, include_W, use_tda)` | 75-117 | Jacobi diagonal (formulas above). TDA output `(nc_pad, nv_pad, nk)` pinned `P("x","y",None)`; full: `stack([d, −d])[:,None]` → `(2,1,nc,nv,nk)` pinned `P(None,None,"x","y",None)` |
| `_get_gmres_solver(matvec, data, max_iter, tol, dtype)` | 120-200 | cached jit FGMRES; **cache key `(max_iter, tol, dtype)` omits matvec/data identity** (line 128) though both are captured in the closure; while_loop stopping; normal-equations LSQ with 1e-14 trace-scaled jitter (180-185) |
| `gmres_solve_sharded_jit(...)` | 203-214 | thin public wrapper → `(x, k_iters)` |
| `_get_feast_runner(matvec, data, n_quad, n_ritz, max_iter, tol, ry_to_ev, dtype, use_conjugate_symmetry)` | 217-270 | cached jit runner: fori over n_ritz × fori over n_quad of GMRES solves; TDA accumulates `2·Re(w·y)`, full accumulates `w·y`; returns (filtered_batch, per-pole iteration counts). **Cache key omits matvec/data** (line 229) |
| `_cast_with_sharding(x, dtype)` | 273-278 | dtype cast re-pinning `x.sharding` |
| `_build_gmres_data_fp32(data)` | 281-291 | complex64/float32 copies of the 8 heavy arrays (psi×4, eps×2, V_q0, W_q); shardings preserved; does **not** carry `W_R` or scalar keys |
| `RitzResult` | 294-303 | frozen result: evals (Ry), coeffs, overlap spectrum, RQs, full-space rel residuals, n_total, s_floor |
| `_rayleigh_ritz(matvec, vectors, data, s_cutoff, use_tda)` | 306-422 | n matvecs (full-precision `data`), S/H/G Gram matrices on device, `device_get` to host numpy; overlap-clip whitening (no dimension truncation, per docstring 313-319); TDA `eigh`, full `eig`→`.real` coercion (397-401); residual identity above |
| `_build_ritz_vectors(filtered, coeffs, sharding, use_tda)` | 425-460 | Ritz vecs = Σ_j c_j·filtered_j; **TDA branch keeps only `c[j].real` and takes `v.real`** ("BSE-TDA eigenvectors are real", 452-453); normalized, re-pinned |
| `run_feast_ritz(data, mesh_xy, windows, n_quad, gamma, n_ritz, gmres_max_iter, gmres_tol, seed, ...)` | 463-684 | driver: builds TDA/full matvec (486-500), `data["W_R"] = jnp.fft.ifftn(data["W_q"], axes=(2,3,4), norm="ortho")` **mutating the caller's dict** (505), fp32 shadow data (513-520), per-window loop with `feast_iter` subspace iterations, per-iteration n_quad schedule (528-534), quadrature choice (582-594), non-TDA node doubling `z ∪ conj(z)` (595-597), GMRES stats print (606-613), Rayleigh-Ritz (620-624), next-iteration restart from in-window Ritz vectors padded with fresh randoms (649-678) |
| `_create_mesh_xy(px, py)` | 686-692 | first px·py devices reshaped to `Mesh(("x","y"))` |
| `estimate_spectral_bounds_sharded(data, mesh_xy, n_lanczos, n_lanczos_max, emax_rtol, emax_atol, emax_patience, seed, include_W, use_tda)` | 695-846 | fp32 Lanczos for E_max (adaptive stop), diag-gap E_min on the *unpadded* band slices (723-727); **always the TDA matvec** (734-740) even for `use_tda=False`, then `E_min := −|E_max|` (825-826); returns bounds dict + host array of all unpadded diagonal transitions (829-832) |
| `build_default_windows_eV(e_max_eV)` | 849-855 | `[0,2]` + `[2,e_max]`, collapsed if e_max < 2 |
| `_zolotarev_step_poles_weights(edge, n_poles, λ_min, λ_max, rho)` | 858-913 | Jacobi-elliptic pole/weight construction (Güttel-Polizzi-Tang-Viaud); lazy `scipy.special` import |
| `feast_zolotarev_quadrature(window, n_quad, λ_min_eV, λ_max_eV, rho_scale)` | 916-962 | indicator = step(a) − step(b); poles split n_quad//2 / rest between edges; right-edge weights negated (961); ρ = rho_scale·(b−a)/2 |
| `feast_ellipse_quadrature(window, n_quad, gamma)` | 965-973 | wrapper → `solvers.quadrature.feast_ellipse_quadrature(center, half_width, n_quad, gamma)` (upper-half midpoint nodes) |
| `format_feast_report(...)` | 980-1058 | text report: bounds, windows, nodes/weights, sampled filter response `f(E) = 2ReΣw/(z−E)` (1051-1055) |
| `_parse_window_arg(values, default)` | 1061-1070 | `A B` pair, `B='auto'` → NaN → replaced by buffered E_max |
| `main(argv)` | 1073-1316 | argparse (below); compile-cache enable (1158-1164); restart load via `bse_io` (1167-1175); bounds → windows (default / KPM / user, 1197-1236); report print; per-window diagonal-transition count (1263-1268); optional `--feast-ritz` solve with schedule `[n_quad1, n_quad2, ...]` (1270-1314); `timing.report` (1316) |

Module-level side effect: `jax.config.update("jax_enable_x64", True)` at
import time (line 26) — any process importing bse_feast flips global x64.

## Flags / CLI args consumed (main, lines 1076-1152)

All defaults as written. `-i/--input` (required, cohsex.in; used for restart
discovery, q=0 head overrides, WFN kgrid fallback), `--n-val` 4, `--n-cond` 4,
`--px` 1, `--py` 1, `--n-lanczos` 10, `--n-lanczos-max` 50, `--lanczos-rtol`
5e-4, `--lanczos-atol` 0.0, `--lanczos-patience` 2, `--buffer` 0.05,
`--n-quad1` 4 (iteration-1 poles), `--n-quad2` 8 (iterations 2+ *and* the
report), `--feast-ritz` (off: report only), `--feast-ritz-count` 4, `--rpa`
(D+V kernel, skip W), `--windows-kpm`, `--windows-kpm-count` 4,
`--kpm-n-moments` 100, `--kpm-n-random` 4, `--kpm-seed` 0,
`--kpm-n-energy-pts` 2000, `--kpm-n-lanczos` 100, `--s-cutoff` 1e-6,
`--feast-iter` 2, `--quadrature` ellipse|zolotarev (default **ellipse**),
`--gmres-max-iter` 10, `--gmres-tol` 1e-2, `--gmres-seed` 0, `--gmres-fp32`,
`--tda` (default **full non-TDA**), `--units-ev-per-ry` 13.6056980659,
`--window1 A B`, `--window2 A B` (`B='auto'` allowed).

Ellipse aspect ratio is **not** a flag: `ELLIPSE_GAMMA_FIXED = 0.2` (line 30)
is hard-coded (the sweep modules vary gamma by calling internals directly).
No LorraxConfig / env reads in this file. When invoked through `bse_jax`,
flags arrive re-spelled (`--feast-n-quad1` → `--n-quad1`, etc.,
bse_jax.py:543-604) and `--windows-kpm` is appended **unconditionally**
(bse_jax.py:579).

## Sharding / PartitionSpec assumptions

Mesh is 2D `("x","y")` = (conduction, valence) band shards. From
`bse_ring_comm.make_bse_shardings` (bse_ring_comm.py:46-62):
`X = P(None,"x","y",None)` on `(b, nc_pad, nv_pad, nk)`;
`X_full = P(None,None,"x","y",None)` on `(2, b, nc_pad, nv_pad, nk)`;
`psi_x/psi_y = P(None,None,None,"x"/"y")` on `(nk, nb, ns, μ_pad)`;
`V = P("x","y")` on `(μ_pad, ν_pad)`; `W = P("x","y",None,None,None)` on
`(μ_pad, ν_pad, nkx, nky, nkz)`; `eps` replicated `(nk, nb_pad)`.
bse_feast pins: TDA diag `P("x","y",None)` (112-113), full diag
`P(None,None,"x","y",None)` (116-117), start/Ritz vectors `sh.X` / `sh.X_full`
(560, 567, 657, 669, 676, 750). GMRES Krylov storage `V/Z` (shape
`(max_iter+1,)+b.shape`, lines 145-148) and the Rayleigh-Ritz Gram products
(352-357) carry **no explicit constraints** — sharding left to XLA
propagation. `_cast_with_sharding` preserves whatever `x.sharding` was.
k axes are never sharded (each device holds all k); band padding
(`n_cond_pad` multiple of px, `n_val_pad` multiple of py) is assumed done by
`bse_io.load_bse_data_from_restart_sharded` (bse_io.py:449-453).

## Host vs device residency

Everything heavy is **device-resident and sharded** (psi ~ nk·nb·ns·μ_pad,
W_q ~ μ_pad²·nk, V_q0 ~ μ_pad²) — loaded sharded by bse_io; note this is the
opposite convention from gw/ (where ψ(G)-scale caches live on host behind
io_callback). fp32 GMRES mode holds a *second* complex64 copy of all eight
arrays plus its own W_R (513-520) alongside the fp64 originals (kept for
Rayleigh-Ritz). Host-side: n×n Gram matrices (device_get at 359-361),
Lanczos tridiagonal scalars (794-797), all quadrature construction
(numpy/scipy), window logic, report.

## TDA vs full-BSE handling

`use_tda` selects: matvec `build_bse_ring_matvec` (Hermitian `A = D+V−W`) vs
`build_bse_ring_matvec_full` (`S = [[A,B],[−Bᴴ,−Aᴴ]]`, ring_comm:487+);
vector shape `(1,nc,nv,nk)` vs `(2,1,nc,nv,nk)`; diag vs `[diag, −diag]`;
contour = upper-half nodes + elementwise `2Re(w·y)` vs conjugate-doubled
node set with plain `w·y` (595-597, 250-254); Ritz `eigh` + real vectors vs
`eig` + real-part eigenvalue coercion (397-401) + complex vectors.
CLI default is **full non-TDA** (`--tda` is opt-in, line 1131-1132), but the
Python API defaults are `use_tda=True` and `quadrature="zolotarev"`
(463-484) — opposite of the CLI (`ellipse`). Spectral bounds always use the
TDA operator; the full-spectrum lower bound is approximated as −|E_max|.

## Spin / nspinor handling

psi arrays carry an explicit spinor axis `s` (`(nk, nb, ns, μ)`); every
pair-density einsum traces it (`"kcsm,kvsm->kcvm"`, lines 90-91, 99-100),
so nspinor 1 and 2 flow through identical code. No singlet factor-of-2
anywhere — spin structure is whatever the restart ψ carries (matches the
matvec convention in bse_ring_comm, validated against BGW per
src/bse/STATUS.md). Caveat: the TDA-only realness assumptions (below) are
in tension with spinor wavefunctions.

## Coupling to gw/ and isdf/

No direct gw/ or isdf/ imports. Coupling is via the **gw_jax restart-bundle
contract** consumed through `bse_io.load_bse_data_from_restart_sharded`
(bse_io.py:358-537): datasets `V_qmunu`, `W0_qmunu` (+`W0_ready` attr, else
V is reused as W!), `psi_full_y`, `enk_full`, `G0_mu_nu`, `vhead`, `whead`;
q=0 head injected by `gw.head_correction.apply_q0_head_rank1_sharded`
(bse_io.py:504-506). The μ/ν axes are the ISDF centroid basis produced by
the gw zeta-fit pipeline; padding convention `runtime.padding.padded_mu_extent`
(bse_io.py:15,442). Other imports: `solvers.quadrature` (generic ellipse
rule), `.bse_ring_comm`, `.bse_preconditioner.energy_diff_cv_k`,
`.bse_io`, `common.timing`, lazy `common.jax_compile_cache` and `.bse_kpm`.

## Suspects

### Bugs

1. **`--window1/--window2` are dead through the primary driver** (control
   flow, confirmed by reading). `main` line 1199 `if args.windows_kpm:` /
   line 1224 `elif args.window1 is not None or args.window2 is not None:` —
   KPM wins whenever both are present. `bse_jax.py:579` appends
   `--windows-kpm` **unconditionally** while also forwarding
   `--feast-window1/2` (bse_jax.py:592-600). A user requesting explicit
   windows via `python -m bse.bse_jax --feast-window1 3 4` silently gets
   KPM-derived windows instead.

2. **Zero-padded band channels are exact in-window eigenpairs** (static
   math). `bse_io._pad_axis_to_multiple` zero-pads ψ *and* eps
   (bse_io.py:449-453, `mode="constant"`). A padded-valence channel has
   ψ_v = 0 ⇒ V_term = W_term = 0 and eigenvalue exactly
   ΔE = eps_c(k,c) − 0 = eps_c(k,c) (absolute Ry energy, e.g. ~7-20 eV —
   inside the W2 bulk window). FEAST start vectors are dense over the
   padded space (`jax.random.normal(subkey, (1, n_cond_pad, n_val_pad, nk))`,
   lines 558/563-567/667-677) with no pad mask, so on multi-device runs with
   px∤n_cond or py∤n_val the filter passes these spurious channels and
   Rayleigh-Ritz reports them as converged in-window states (residual = 0 —
   they *are* eigenpairs of the padded operator). Single-device and
   evenly-divisible runs are immune.

3. **Kernel caches keyed without matvec/data identity** (latent).
   `_GMRES_SOLVER_CACHE` key `(max_iter, tol, dtype)` (lines 33, 128-130)
   and `_FEAST_RUNNER_CACHE` key `(n_quad, n_ritz, max_iter, tol, ry_to_ev,
   dtype, conj_sym)` (lines 34, 229-231) both return jitted closures that
   captured `matvec` and the concrete `data` arrays as constants. Second
   call in the same process with the same key but different physics — e.g.
   `run_feast_ritz(include_W=True)` then `include_W=False` (RPA), or a new
   restart file — silently reuses the first call's matvec *and* data:
   RPA output is actually BSE output. Not currently triggered by in-repo
   drivers (feast_sweep varies only n_quad/gamma/n_ritz on one dataset),
   but any sweep across kernels/datasets will hit it. Same hazard family as
   the `id(mesh_xy)` cache flagged in gw/cohsex_sigma.

4. **TDA path hard-codes an elementwise-real Hamiltonian** (plausible;
   input-dependent). The conjugate-symmetry accumulation
   `filt += 2·jnp.real(w·y)` (251-252) equals the true two-half-plane sum
   `w·y + conj(w)·(z̄−H)⁻¹x` only if `conj(y) = (z̄−H)⁻¹x`, i.e.
   `conj((z̄−H̄)⁻¹x̄)` with H̄ = H and x̄ = x — elementwise-real H and real x.
   `_build_ritz_vectors` then enforces it: `float(c[j].real)`, `v = v.real`
   ("BSE-TDA eigenvectors are real", 452-453). True for the validated Si
   runs (real-gauge ψ ⇒ real H), false for complex-Hermitian TDA H (SOC
   spinor ψ, general gauges), where the imaginary parts of the filtered
   subspace are silently discarded. The default non-TDA path (explicit
   conjugate nodes, complex vectors) does not have this assumption.

### Redundancy / cruft

- `_create_mesh_xy` exists in **five** copies: here (686-692),
  `bse_pseudopoles.py:203`, `bse_w_exact.py:28`, `feast_zolo_sweep.py:98`,
  `feast_ellipse_mixed_sweep.py:74` — while `bse_kpm.py:28` and
  `feast_sweep.py:32` import this module's copy, and
  `bse_ring_comm.create_mesh_2d` (ring_comm:30) is a sixth variant.
- W_R is computed up to three times per `main` run: fp32 for bounds
  (714-716), fp64 for FEAST (505), fp32 again if `--gmres-fp32` (516-518);
  all via **plain `jnp.fft.ifftn` on the sharded W_q** — exactly the
  pattern bse_ring_comm works around with `make_sharded_ifftn_3d` because
  it forces a full all-gather (ring_comm comment at 398-409). One-time
  cost, but the gather materializes the full μ²·nk W tensor per device
  (μ_pad=4000, nk=64 ⇒ ~16 GB) — an OOM hazard scaling to production.
- `run_feast_ritz` mutates the caller's `data` dict (`data["W_R"] = ...`,
  505-507); `feast_sweep.py:613-614` comments show callers already being
  confused about who owns W_R.
- API defaults contradict the CLI: `run_feast_ritz(quadrature="zolotarev",
  use_tda=True)` (476, 483) vs main's `--quadrature ellipse` / non-TDA
  default. `feast_sweep.py:173-186` calls with defaults and no
  lambda bounds ⇒ every config raises "Zolotarev quadrature requires
  spectral bounds" (584-587) inside its try/except — the whole sweep
  no-ops at HEAD (feast_sweep also reads `rr_data.n_physical`, a field
  `RitzResult` no longer has — caller drift, logged for the feast_sweep
  file).
- `QuadratureSpec.gamma` is filled with `ELLIPSE_GAMMA_FIXED` even for
  zolotarev specs (main:1244-1247) — meaningless for that type.
- `src/bse/context/feast_accuracy_notes.md:75-86` documents `--n-quad` and
  `--gamma` flags that no longer exist (now `--n-quad1/2`; gamma
  hard-coded) — stale doc.

### Weird but defensible

- Bounds Lanczos always runs the TDA operator, non-TDA lower bound taken as
  −|E_max| (734, 825-826) — approximation, printed as such.
- Non-TDA Ritz eigenvalues coerced real (`evals = evals.real`, 397-401) and
  the residual identity `Re[cᴴ(G − 2λH + λ²S)c]` is only exact for real λ —
  fine for positive-definite BSE, wrong if genuinely complex pairs appear.
- `--windows-kpm` runs a second, independent Lanczos inside
  `bse_kpm.run_kpm_dos` (1203-1218) after `estimate_spectral_bounds_sharded`
  already ran one — duplicated bound estimation on the default bse_jax path.
- `jax_enable_x64` flipped globally at import (26).
- GMRES `rel` is measured against ‖r0‖ (post-x0 residual), not ‖b‖ (187).
