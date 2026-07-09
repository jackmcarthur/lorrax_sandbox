# Group notes: pseudobands (src/solvers/pseudobands.py, src/solvers/pseudobands_v2.py)

Cataloged 2026-07-01 for the GW refactor map. Both files are physics-agnostic spectral
compression: they replace the high-energy tail of the eigenspectrum of any Hermitian
matvec `apply_H` with a small number of weighted "pseudoband" vectors, used to shrink
the empty-band sum in GW. Neither file touches disk; the consumer
(`src/psp/run_nscf.py`) writes `WFN_pseudobands.h5`. Downstream, the per-band `weights`
land in WFN band norms and are consumed by `common/isdf_fitting.py` (`band_norms`
pre-divide, see `isdf_fitting.py:1807`).

Call graph (grep over src/, tests/, tools/, scripts/):

- `ritz_pseudobands` (v1) <- `src/psp/run_nscf.py:341` (only when `pb_version == 1` in
  nscf.in; default is 2) and re-export `src/solvers/__init__.py:12`.
- `ritz_pseudobands_v2` <- `src/psp/run_nscf.py:339` (default path) and
  `src/solvers/__init__.py:13`.
- No test in tests/ imports either module (grep `ritz_pseudobands|solvers.pseudobands`
  over tests/ → 0 hits). `tests/test_zq_from_psi_sm_bit_identity.py` only tests the
  downstream band-norms pre-divide, not these modules.
- No callers in tools/ or scripts/.

Flags consumed (indirectly, via `psp/nscf_input.py` → `psp/run_nscf.py` kwargs; these
modules take plain Python kwargs, no LorraxConfig/cohsex.in):
`pseudobands` (bool), `pb_version` (1|2, default 2), `pb_k` (default 6), `pb_M_max`
(default 1500), `pb_F` (default 0.10), `pb_n_windows` (default 50), `pb_n_prot`
(default 0 = auto), plus CLI `--pseudobands`, `--pb-seed`, `--pb-wfn`, `--no-davidson`.

Cross-k consistency protocol in run_nscf: k=0 call determines `dos_result`,
`n_prot`/`n_protected` and (v2) `boundaries`, which are then passed as
`dos_result=pb0.dos`, `n_prot_override=n_prot_k0`, `boundaries_override=pb0.windows`
(v2) / `n_protected=n_prot_k0` (v1) to every other k-point, with per-k seed
`ik + pb_seed * 10000`.

---

## src/solvers/pseudobands.py (507 LOC) — v1 "hybrid stochastic + CJ-Ritz"

Purpose: v1 pseudoband generator. Three band regions: protected Davidson eigenstates
(weight 1), stochastic pseudobands (random-phase combos of exact eigenstates per
window, à la BGW stochastic pseudobands), and CJ-Ritz pseudobands (Chebyshev-Jackson
filtered Galerkin-Ritz vectors) for windows above the Davidson range.

### Function table

| Function | Lines | Role |
|---|---|---|
| `PseudobandsResult` (dataclass) | 47–59 | Output bundle: `Phi_out (n_total, dim)` np.complex128 host, `E_out (n_total,)`, `weights (n_total,)` (1.0 protected, sqrt(n_eff/k) pseudo), counts, `dos: DOSResult`, `windows: WindowPartition|None`. NOTE: same class name redefined with different fields in v2. |
| `_boundary_coefficients` | 66–94 | Jackson-damped Chebyshev coefficients for cumulative step filters. Per boundary b: `gamma_0 = 1 - arccos(b)/pi`, `gamma_n = -2/(pi n) sin(n arccos(b))`, `coeffs = gamma * g_Jackson`. Returns `(N_S+1, M_max)`. Called only by `ritz_pseudobands` (line 400). |
| `_telescoping_filter` | 101–145 | M_max block Chebyshev matvecs accumulating all N_S+1 boundary cumulatives at once; `apply_H_tilde(x) = (apply_H(x) - center x)/half_width`; recurrence `T_n = 2 H~ T_{n-1} - T_{n-2}` via `jax.lax.fori_loop`; telescope `Y_j = A_{j+1} - A_j` inside (returns `(N_S, k, dim)` device array). Called at line 409. |
| `_stochastic_pseudobands` | 152–190 | BGW-style stochastic pseudoband: `ξ_α = (1/√n) Σ_i exp(iθ_{α,i}) ψ_i`; pseudo-energy `<ξ|H|ξ> = mean(E_i)` (cross terms cancel in expectation). Uses `jnp.dot(coeffs, Phi)` not einsum. Returns host np arrays `(k, dim)`, `(k,)`. Called at line 438. |
| `_galerkin_ritz` | 197–243 | CJ window Ritz extraction with deflation. Einsums VERBATIM: deflate `overlap = jnp.einsum('nd,kd->nk', conj(Phi_det), Y_j)`, `Y_j -= jnp.einsum('nk,nd->kd', overlap, Phi_det)`; cross-window `overlap = jnp.einsum('pd,kd->pk', conj(Q_prev), Y_j)`, `Y_j -= jnp.einsum('pk,pd->kd', overlap, Q_prev)`; Galerkin `H_proj = jnp.einsum('kd,ld->kl', conj(Q), HQ)`, hermitize `0.5(H+H†)`, `eigh`; Ritz `Xi = jnp.einsum('kl,kd->ld', S.T, Q)`. Returns `(Xi, theta, Q)`; Q chains to next window (§7 cross-window orthogonalization — v2 dropped this). Called at line 446. |
| `ritz_pseudobands` | 250–507 | Top-level driver. Steps: KPM DOS (`solvers.dos.compute_dos`) unless `dos_result` given; CJ resolution `π·B/M_max`, crossover `eps_cross = C_m · cj_resolution / F`; window boundaries via `dos_weighted_windows` (if `n_windows_target`) else `geometric_windows(eps_cross, E_max-E_F, F) + E_F`; classify det bands protected/available (fixed `n_protected` override = first n by energy sort, for cross-k band-count consistency); classify windows stochastic (≥1 det eigenstate) vs CJ; run telescoping filter only if CJ windows exist (but computes ALL windows — shared recurrence); per-window loop builds Xi with weight `w_j = sqrt(n_states/k)` (stoch) or `sqrt(n_eff_j/k)` (CJ, `n_eff = win_part.n_eff * dim` from `compute_window_partition`); CJ leak check: if any Ritz θ outside `[e_lo - cj_res, e_hi + cj_res]` → w=0 ("CJ-0"); universal drop rule `n_eff < 0.5` → zero coefficients, θ = window midpoint; weight absorbed into wavefunction (`Xi * w_j`); output = protected ++ all pseudo. |

Key arrays: everything host np.complex128 except inside the filter/Ritz jit-free jnp
calls; `Y_all` is materialized to host `(N_S, k, dim)` at line 411 (for large dim and
many windows this is the memory hot spot). `apply_H` is vmapped to a block matvec.

### v1 suspects
- **Dead assignment** line 394: `Y_cj = None` — never read anywhere (the real
  variable is `Y_all`). Vestige of a refactor.
- `C_m` kwarg (default 1.0): never set by any caller (`run_nscf.py` does not pass it;
  grep `C_m` over src → only this file). Dead knob.
- Magic seed offsets: `seed + 42` (filter Omega, line 404), `seed + 100` (stochastic
  RNG, line 428).
- Omega convention: v1 draws real Gaussian then `+ 0j` (line 405-406); v2 draws
  random unit-modulus phases `exp(iθ)` (v2 lines 595-596). Silent convention
  difference between "identical" filters.
- Whole module is a legacy parallel path: only reachable via `pb_version = 1` in
  nscf.in; default is 2. Classic old/new pair per the no-redundancy rule.

---

## src/solvers/pseudobands_v2.py (755 LOC) — v2 "Galerkin-Ritz + Gauss quadrature"

Purpose: v2 pseudoband generator, current default (`pb_version=2`). Differences vs v1:
shifted-CJ boundaries giving a quadratic partition of unity (Σ w_j² ≈ 1); window
placement by an equal-error metric with an n_min state floor (`_place_windows_v2`)
instead of DOS-weighted/geometric windows; Davidson windows do rank-k Galerkin
compression on stored eigenvalues (no matvec) with Gauss-quadrature node/weight
compression when n_in > k; CJ n_eff uses ∫ w_j(E)² ρ dE. The originally-designed
Gauss weights for CJ windows were empirically ROLLED BACK (commit d1466ad) to uniform
`n_eff/k` — much of the Gauss machinery is now vestigial for the CJ path.

### Function table

| Function | Lines | Role |
|---|---|---|
| `PseudobandsResult` (dataclass) | 36–47 | Same name as v1, DIFFERENT fields: `n_prot` (v1: `n_det`), `n_dav_windows`/`n_cj_windows` (v1: `n_stochastic`/`n_cj`), `windows: np.ndarray` boundaries (v1: `WindowPartition|None`). Not exported from `solvers/__init__` (only v1's is). run_nscf must branch `pb0.n_prot if pb_version == 2 else pb0.n_det`. |
| `_shifted_boundary_coefficients` | 54–112 | Like v1 `_boundary_coefficients` but each internal boundary gets TWO cumulatives at ε ± δ, δ = π/(2M_max) (rescaled units); outer boundaries unshifted. Same gamma formulas as v1 (duplicated). Returns `(n_cum, M_max)` coeffs + per-CJ-window `(idx_hi, idx_lo)` pairs. Called at line 587. |
| `_telescoping_filter` | 119–147 | Near-verbatim copy of v1's (lines 101–145) minus the internal telescoping step — returns raw accumulators `A (n_cum, k, dim)`; the caller does `Y_j = A[hi] - A[lo]`. Duplicated code. Called at line 599. |
| `_gauss_from_moments` | 154–226 | k-point Gauss quadrature from 2k power moments: Hankel `H[i,j] = m_{i+j}`, Cholesky (or eigh fallback), then Stieltjes recurrence for Jacobi α/β, Golub-Welsch: `nodes = eigvals(J)`, `weights = m_0 · vecs[0,:]²`. Called only from the Davidson-window branch (line 652) when `n_in > k`. NOTE: the Cholesky `L` (lines 174–180) is computed but NEVER USED — the Stieltjes recurrence works directly on `moments`. Dead computation, including its try/except fallback. |
| `_compute_window_moments` | 229–254 | Continuous-grid power moments `m_n = ∫ x^n w_j(E) ρ(E) dE`, `x = (E-shift)/scale`, support-clipped at `abs(w_j) > 1e-6`. **DEAD**: zero call sites (grep `_compute_window_moments(` over src/tests/tools/scripts → definition only). Orphaned by the d1466ad Gauss-weights rollback. |
| `_compute_window_moments_discrete` | 257–268 | Power moments `m_n = Σ_i x_i^n` from discrete eigenvalues. Called at line 650 (Davidson window, n_in > k). |
| `_cj_window_on_grid` | 275–302 | Evaluates the CJ window indicator `w_j(E) = C(b_hi) - C(b_lo)` on the DOS grid via `T = cos(n · arccos(ε))`. Third duplication of the gamma/Jackson boundary formula. Awkward API: takes b_lo/b_hi in ABSOLUTE energy and rescales internally, while the call site (lines 673–676) converts stored rescaled boundaries back to absolute first (round-trip rescale). Called at line 673. |
| `_galerkin_ritz_cj` | 309–331 | v1 `_galerkin_ritz` minus the Q_prev cross-window orthogonalization (v2 relies on shifted-CJ POU instead). Einsums VERBATIM: `overlap = jnp.einsum('nd,kd->nk', conj(Phi_dav), Y_j)`; `Y_j -= jnp.einsum('nk,nd->kd', overlap, Phi_dav)`; `H_proj = jnp.einsum('kd,ld->kl', conj(Q), HQ)`; `Xi = jnp.einsum('kl,kd->ld', S.T, Q)`. Called at line 666. |
| `_galerkin_ritz_dav` | 334–375 | Davidson window, no matvec. n ≤ k: pass eigenstates through, zero-pad to k, θ pad = mean(E). n > k: random-phase projector `R = exp(iθ) (n,k)`, `Z = jnp.einsum('nd,na->ad', Phi_j, R)`, QR, then Galerkin from STORED eigenvalues: `overlap = jnp.einsum('nd,kd->nk', conj(Phi_j), Q)`; `H_proj = jnp.einsum('nk,n,nl->kl', conj(overlap), E_j, overlap)`; `Xi = jnp.einsum('kl,kd->ld', S.T, Q)`. Called at line 635. |
| `_place_windows_v2` | 382–453 | Boundary placement: equal-error rule + n_min floor. Per trial window: `σ = Δ/√12`, `metric = n_eff · σ⁴ / ε̄⁵` ("conservative effective-k=2"; the full σ^(2k)/ε^(2k+1) bound abandoned after d1466ad rollback — see comment lines 425–430); inner scan of 500 linspace trials; outer geometric bisection on τ (50 iters, accept when `abs(n_w - target) ≤ 1`). NOTE: parameter `k` is accepted (and passed `k=k` at line 555) but NEVER USED in the body — the exponent is hardcoded to the k=2 form. Called at line 553. |
| `ritz_pseudobands_v2` | 460–755 | Top-level driver. KPM DOS; protected count from `n_prot` / crossover `cj_resolution/F` / `n_prot_override` (three-way precedence, `n_prot_override` wins); windows from `_place_windows_v2` unless `boundaries_override`; classify windows Davidson (`boundaries[j+1] ≤ E_dav_max`) vs CJ; shifted-CJ filter for CJ windows; per-window: dav → `_galerkin_ritz_dav` + exact eigenvalues (n≤k) or discrete-moment Gauss compression (n>k); CJ → `_galerkin_ritz_cj`, `n_eff_j = ∫ w_E² max(ρ,0) dE · dim`, nodes = sorted Ritz θ, weights UNIFORM `n_eff_j/k` (Gauss weights rejected — comment lines 685–693 documents 14× weight imbalance amplifying per-seed noise to 240 meV std vs 7 meV for uniform); sort Ritz and nodes ascending and pair by order; drop rule `Σ gauss_w < 0.5` → zeros + midpoint energies; weight absorbed as `Xi * sqrt(w)` per band (`weights_sorted = sqrt(max(gauss_w,0))`). |

Key arrays: host np.complex128 throughout the assembly; `A_all (n_cum, k, dim)`
materialized to host at line 601 (2 cumulatives per CJ window — ~2× the v1 filter
output for the same window count). No sharding; single-process, vmapped matvec.

### v2 suspects
- **Dead function**: `_compute_window_moments` (lines 229–254), zero callers.
- **Dead imports**: `dos_weighted_windows`, `compute_window_partition` (line 29) and
  `functools.partial` (line 26) — none referenced in the body (grep in-file).
- **Dead computation**: Cholesky `L` in `_gauss_from_moments` (lines 173–180)
  computed, never used.
- **Unused parameter**: `k` in `_place_windows_v2` — metric hardcoded σ⁴/ε̄⁵.
- Magic constants: δ = π/(2M_max) boundary shift; support clip 1e-6; grid clip
  ±0.9999 (line 286); seeds `seed + 42` (filter), `seed + 200` (dav RNG; v1 used
  +100); 500 linspace trials / 50 bisection iters / τ ∈ [1e-30, 1e10] in
  `_place_windows_v2`; drop threshold 0.5 states (shared with v1).
- Sorting gymnastics lines 698–705: `nodes` is already `np.sort(theta_ritz)` in the
  CJ branch, then re-argsorted via `gauss_order`; `ritz_order`/`gauss_order` are both
  identity there — the double sort only matters for the dav/Gauss branch.
- The "Gauss-quadrature energies" advertised in the module docstring are only actually
  used for Davidson windows with n_in > k; CJ windows use Ritz θ + uniform weights
  after the rollback. Docstring overstates current behavior.

### Redundancy across the pair (refactor targets)
1. Two whole parallel modules selected by `pb_version` — exactly the old/new-path
   pattern the sandbox no-redundancy rule forbids. v1 is reachable but non-default and
   untested.
2. `_telescoping_filter` duplicated nearly verbatim (v1:101 vs v2:119; only the
   final telescoping subtraction moved to the caller in v2).
3. Jackson-damped step-function coefficient formula written THREE times:
   v1 `_boundary_coefficients`, v2 `_shifted_boundary_coefficients`,
   v2 `_cj_window_on_grid`.
4. `_galerkin_ritz` (v1) vs `_galerkin_ritz_cj` (v2): identical except Q_prev
   deflation.
5. `_stochastic_pseudobands` (v1) vs `_galerkin_ritz_dav` n>k branch (v2): both are
   random-phase compressions of exact window eigenstates; v2 adds the Galerkin eigh.
6. Two dataclasses named `PseudobandsResult` with incompatible field names, forcing
   the `pb_version == 2` attribute branch in run_nscf.py:375.
