# src/solvers/pseudobands.py (507 LOC) + src/solvers/pseudobands_v2.py (755 LOC) — deep-read notes

Audit date: 2026-07-15, lorrax_D checkout. Working tree is on branch
`agent/ppm-fit-conditioning` @ adc2197, **not** the stated audit base
`agent/slate-linalg-ffi` @ e18d0e5; however
`git diff --stat e18d0e5 HEAD -- src/solvers/pseudobands.py src/solvers/pseudobands_v2.py`
is empty — both files are byte-identical at the two refs, so these notes hold
for both.

## Purpose

Spectral compression of a Hermitian operator's high-energy manifold into a
small set of weighted "pseudobands": random/filtered vectors that reproduce
sum-over-states quantities (χ⁰, Σ band sums) without storing thousands of
exact eigenstates. This is the LORRAX-native replacement for BGW's
`parabands` stochastic pseudobands (Altman et al. convention, see
`src/solvers/docs/pseudobands_SI.md`, `pseudobands_bgw_convention.md`).
Both files are **physics-agnostic**: they see only a Hermitian matvec
`apply_H: (dim,) → (dim,)` and a flat vector space `dim = nspinor·ngkmax`.
Today the only operator ever passed in is H_DFT (from `psp.run_nscf`); the
v1 docstring's "works for H_DFT, H_BSE, or any Hermitian operator" is
aspirational — no BSE caller exists (grep below).

Pipeline (both versions):

1. **KPM DOS** (`solvers.dos.compute_dos`): stochastic-trace Chebyshev
   moments, Jackson damping, ρ(E) on a dense grid. Moments are normalized
   `mu /= n_random * dim` (chebyshev.py:171), so **∫ρ dE ≈ 1** and every
   state-count in these files multiplies by `dim`
   (pseudobands.py:418 `n_eff = win_part.n_eff * dim`;
   pseudobands_v2.py:406 `cdf … * dim`, :683 `trapezoid(w_E²·ρ)·dim`).
   (`compute_window_partition`'s docstring "total states in window" at
   dos.py:313 describes the *fraction* — the ·dim here is correct, the dos.py
   docstring is the misleading one.)
2. **Window partition** of `[E_start, E_max]` into N_S windows.
3. **Chebyshev–Jackson (CJ) cumulative step filters**. Coefficients of the
   smoothed indicator `1_{[-1,b]}(x)` as written (pseudobands.py:90–92,
   identical formula at v2:107–110 and v2:295–296):

   ```
   γ_0 = 1 − arccos(b)/π
   γ_n = −(2/(πn))·sin(n·arccos(b)),  n ≥ 1
   c_n = γ_n · g_n              (g = jackson_coefficients(M_max−1), shape (M_max,))
   ```

   Verified against the exact Chebyshev expansion of the step:
   c_n = (2−δ_{n0})/π ∫_{−1}^{b} T_n(x)/√(1−x²) dx = −2 sin(n·arccos b)/(πn),
   c_0 = 1 − arccos(b)/π. Correct.
4. **Telescoping filter** (`_telescoping_filter`, identical body in both
   files): one block Chebyshev recurrence
   `T_n = 2·H̃·T_{n−1} − T_{n−2}`, `H̃ = (H − center)/half_width`, run once
   for all boundaries simultaneously; accumulators
   `A_j = Σ_n c_n^{(b_j)} T_n(H̃) Ω`. Window block = difference of adjacent
   cumulatives: v1 telescopes internally `Y = A[1:] − A[:-1]`
   (pseudobands.py:144); v2 returns raw `A` and differencing happens at the
   shifted-pair level (v2:606 `Y_cj[j] = A_all[idx_hi] − A_all[idx_lo]`).
5. **Per-window Ritz extraction** (Galerkin): deflate against exact
   eigenvectors, QR, `H_proj = Q†HQ`, `eigh`, rotate. Per-element formulas
   as written (pseudobands.py:220–241):

   ```
   overlap[n,k] = Σ_d conj(Phi_det[n,d])·Y[k,d]          ('nd,kd->nk')  = ⟨φ_n|y_k⟩
   Y[k,d]      −= Σ_n overlap[n,k]·Phi_det[n,d]           ('nk,nd->kd')  (projector deflation)
   Q            = qr(Y.T).Q.T                              (k,dim) orthonormal rows
   H_proj[k,l]  = Σ_d conj(Q[k,d])·(HQ)[l,d]              ('kd,ld->kl')  = ⟨q_k|H|q_l⟩
   θ, S         = eigh( (H_proj+H_proj†)/2 )
   Xi[l,d]      = Σ_k (S.T)[k,l]·Q[k,d] = Σ_k S[l,k]·Q[k,d]   ('kl,kd->ld' on S.T)   ← see BUG-1
   ```

6. **Weights**: each window's k pseudobands carry weight `w = √(n_eff/k)`
   (spectral mass n_eff distributed uniformly), **absorbed into the stored
   wavefunction** (`Xi * w`, pseudobands.py:471; v2:719
   `Xi_sorted * weights_sorted[:,None]`), plus returned separately in
   `weights`. Protected (deterministic) bands keep weight 1.0. This is the
   BGW-parabands convention: downstream consumers "just see coefficients
   with larger norms" (`solvers/docs/pseudobands_bgw_convention.md:15,40`).
7. **Universal drop rule**: any window carrying < 0.5 states total gets its
   vectors zeroed and its energies set to the window midpoint, so phantom θ
   values can't corrupt downstream `cmin`/`vmax` band-edge searches
   (pseudobands.py:459–469; v2:707–717).

**v1** (`ritz_pseudobands`, hybrid stochastic + CJ-Ritz): windows that
contain ≥1 exact eigenstate are "stochastic" — pseudobands are random-phase
combinations `ξ_α = (1/√n) Σ_i exp(iθ_{α,i}) ψ_i` (pseudobands.py:180–184;
exactly unit norm, exact pseudo-energy `⟨ξ|H|ξ⟩ = mean(E_window)` since
cross terms vanish for orthonormal ψ, line 188); empty windows use the CJ
filter, with a leak check (any Ritz θ outside `[e_lo−res, e_hi+res]` →
whole window zeroed, "CJ-0", lines 450–453) and cross-window
orthogonalization against the previous CJ window's Q (Q_prev, lines
224–226, 457).

**v2** (`ritz_pseudobands_v2`, Galerkin-Ritz + Gauss-quadrature energies):
"Davidson windows" (fully below E_dav_max) compress n exact eigenstates into
k nodes/weights via power-moment → Jacobi-matrix (Golub-Welsch/Stieltjes)
Gauss quadrature (`_gauss_from_moments`, Wheeler recurrence verified:
σ_{j,l} = σ_{j−1,l+1} − α_{j−1}σ_{j−1,l} − β_{j−1}σ_{j−2,l};
α_j = σ_{j,j+1}/σ_{j,j} − σ_{j−1,j}/σ_{j−1,j−1}; β_j = σ_{j,j}/σ_{j−1,j−1};
weights = m_0·vecs[0,:]², all standard). "CJ windows" use **shifted**
boundary cumulatives at ε±δ, δ = π/(2M) (v2:71), giving overlapping window
indicators with a *quadratic* partition of unity (Σ_j w_j(E)² ≈ 1), so the
per-window captured mass is `n_eff = ∫ w_j(E)²ρ(E)dE · dim` (v2:679–683).
CJ energies = sorted Ritz θ with **uniform** weights n_eff/k — Gauss weights
for CJ windows were tried and reverted (commits 417e05a → 35ef270 → d1466ad;
rationale in the long comment v2:685–693: unbalanced Gauss weights amplified
per-seed stochastic noise, 240 meV vs 7 meV std). Window placement
(`_place_windows_v2`) is equal-error with an n_min=k state floor and a
hardcoded conservative σ⁴/ε̄⁵ metric (v2:424–430, rollback leftover — the
`k` parameter and the σ^{2k} docstring are stale, see suspects). No
cross-window orthogonalization in v2 (dropped deliberately: adjacent shifted
windows overlap by design).

Category: **pipeline stage (mean-field band-compression front-end)** — runs
between Davidson NSCF and WFN.h5 output, upstream of everything GW/BSE.

## Entry points (grep over src/, tests/, tools/, scripts/, sandbox runs/skills/scripts, docs)

| symbol | callers (grep evidence) |
|---|---|
| `ritz_pseudobands` (v1) | `src/psp/run_nscf.py:341` (only when `pb_version != 2`; calls at :367–372, :409–416); re-exported `src/solvers/__init__.py:12` |
| `ritz_pseudobands_v2` | `src/psp/run_nscf.py:339` (default `pb_version=2`; calls at :359–365, :397–407); re-exported `src/solvers/__init__.py:13` |
| `PseudobandsResult` (v1) | re-exported `src/solvers/__init__.py:12`; consumed structurally by run_nscf (`pb0.Phi_out/E_out/n_det/n_prot/dos/windows`, run_nscf.py:355–427) |
| `PseudobandsResult` (v2, same name, different fields) | module-local only; **not** re-exported (grep `solvers/__init__.py` — only the v1 import) |
| all `_`-private helpers (both files) | module-internal only; `grep -rn "_telescoping_filter\|_galerkin_ritz\|_gauss_from_moments\|_place_windows_v2\|_shifted_boundary_coefficients\|_cj_window_on_grid\|_boundary_coefficients\|_stochastic_pseudobands" src tests tools` → hits only inside the two files |
| `python -m solvers.pseudobands*` | none — neither file has an `if __name__` block (grep confirmed) |
| production invocation chain | `python3 … -m psp.run_nscf -i nscf.in` (documented at `skills/profiling_stack/SKILL.md:101–105`, `drilldowns.md:211`) → `run_nscf(do_pseudobands=…)` → writes `WFN_pseudobands.h5` |
| tests | **none for either module** (grep `pseudobands` over tests/ → only `test_zq_from_psi_sm_bit_identity.py:435` which tests the downstream ISDF *norms pre-divide*, not these solvers) |
| sandbox runs | `runs/Si_pseudobands/**` is a **BGW-parabands** comparison run (manifest "40/200 bands from parabands"), not an invocation of this module |

v1 is only reachable through the `pb_version = 1` key of `nscf.in`
(psp/nscf_input.py:74 `pb_version=sec.getint("pb_version", 2)`); it is a
config-selectable fallback, not dead.

## Function table

### pseudobands.py (v1)

| function | lines | role |
|---|---|---|
| `PseudobandsResult` | 47–59 | output dataclass: `Phi_out (n_total,dim)`, `E_out`, `weights`, `n_det`, `n_pseudo`, `n_windows`, `n_stochastic`, `n_cj`, `dos: DOSResult`, `windows: WindowPartition\|None` |
| `_boundary_coefficients(tilde_eps, M_max)` | 66–94 | CJ step coefficients per boundary, `(N_S+1, M_max)`; formula verified above |
| `_telescoping_filter(apply_H, Omega, coeffs, center, half_width, M_max)` | 101–145 | block Chebyshev recurrence via `jax.lax.fori_loop(2, M_max, …)`; returns telescoped `Y (N_S,k,dim)`; header claims "(JIT'd)" but there is no `@jit` — fori_loop stages the body regardless |
| `_stochastic_pseudobands(Phi_window, E_window, k, key)` | 152–190 | random-phase combos, exact unit norm and exact `E_pseudo = mean(E_window)` |
| `_galerkin_ritz(apply_H, Y_j, Phi_det, Q_prev)` | 197–243 | deflate (det manifold + previous window Q), QR, Galerkin eigh, rotate (BUG-1 at 241) |
| `ritz_pseudobands(...)` | 250–507 | driver: DOS → windows → classify stochastic/CJ → per-window build → assemble `[protected; pseudo]` |

Driver internals: window boundaries from `dos_weighted_windows`
(:328–330, when `n_windows_target` given — always, from run_nscf) or
`geometric_windows(eps_cross, E_max−E_fermi, F)+E_fermi` (:332–333, API-only
fallback); crossover `eps_cross = C_m·(πB/M_max)/F` (:317–318); protected =
det bands below `boundaries[0]`, or exactly `n_protected` lowest by energy
when fixed (:344–352, the k-point-consistency path used by run_nscf for
ik>0 via `n_protected=n_prot_k0`); stochastic = window holds ≥1 available
det state (:374–381); `w_j = √(n_states/k)` stochastic (:441),
`√(n_eff_j/k)` CJ (:455), 0 on leak (:450–453) or `n_eff<0.5` (:464).

### pseudobands_v2.py

| function | lines | role |
|---|---|---|
| `PseudobandsResult` | 36–47 | **same class name, different schema**: `n_prot`, `n_dav_windows`, `n_cj_windows`, `windows: np.ndarray` (boundaries) |
| `_shifted_boundary_coefficients(boundaries, center, half_width, M_max, cj_windows)` | 54–112 | per-CJ-window cumulative pairs at ε∓δ/ε±δ (outer boundaries unshifted, :84–93); returns `(coeffs (2·n_cj, M_max), window_pairs [(idx_hi, idx_lo)])` |
| `_telescoping_filter(...)` | 119–147 | verbatim copy of v1's recurrence, returns raw `A (n_cum,k,dim)` |
| `_gauss_from_moments(moments, k)` | 154–226 | 2k power moments → k-node Gauss rule (Wheeler + Golub-Welsch, verified); dead Cholesky `L` at 173–180 |
| `_compute_window_moments(E_grid, rho, w_j, n_moments, E_shift, E_scale)` | 229–254 | grid-weighted power moments — **zero callers, dead** (see suspects) |
| `_compute_window_moments_discrete(E_eig, n_moments, E_shift, E_scale)` | 257–268 | `m_n = Σ_i x_i^n`, x = (E−shift)/scale; used by dav windows with n_in > k (:650) |
| `_cj_window_on_grid(E_grid, b_lo, b_hi, center, half_width, M_max)` | 275–302 | evaluates `w_j(E) = C_{b_hi}(E) − C_{b_lo}(E)` on the DOS grid (takes *absolute* b's, rescales internally; caller at :673–676 converts its rescaled b's back to absolute — consistent). Builds a `(M_max, n_grid)` = 1500×10000 float64 T-matrix per call (~120 MB transient, once per CJ window) |
| `_galerkin_ritz_cj(apply_H, Y_j, Phi_dav)` | 309–331 | deflate against **all** Davidson states, QR, Galerkin eigh, rotate (BUG-1 at 330); no Q_prev |
| `_galerkin_ritz_dav(Phi_window, E_window, k, rng_key)` | 334–375 | n≤k: pass-through + zero-pad; n>k: random-phase project to rank k, Galerkin on **stored eigenvalues** `H_proj[k,l] = Σ_n conj(⟨φ_n|q_k⟩)·E_n·⟨φ_n|q_l⟩` ('nk,n,nl->kl', verified), rotate (BUG-1 at 373). No matvec needed |
| `_place_windows_v2(E_grid, rho, E_start, E_max, n_windows_target, n_min, dim, k=1)` | 382–453 | equal-error boundaries: grow each window until `n_eff·σ⁴/ε̄⁵ ≥ τ` **and** `n_eff ≥ n_min`, τ found by 50-step geometric bisection to hit the target window count; `k` param unused (suspects) |
| `ritz_pseudobands_v2(...)` | 460–755 | driver: DOS → protected split → place/override windows → classify dav/CJ (`is_dav[j] ⇔ boundaries[j+1] ≤ E_dav_max+1e−10`, :570) → CJ filter (random unit-modulus phase block Ω, :595–596) → per-window Ritz+Gauss → sort-and-pair (BUG-2) → assemble |

## Flags / CLI args / config keys consumed

**Neither file reads any CLI arg, config file, or environment variable
directly.** All knobs are keyword parameters, wired exclusively by
`psp.run_nscf`:

| knob | v1 / v2 param | wiring | default |
|---|---|---|---|
| `pseudobands` (nscf.in) / `--pseudobands` | — | gates the stage, run_nscf.py:477,494 | false |
| `pb_version` (nscf.in only) | selects module | run_nscf.py:338–341 | 2 |
| `pb_k` | `k` | pseudobands per window (block size) | 6 |
| `pb_M_max` | `M_max` | Chebyshev order | 1500 |
| `pb_F` | `F` | window ratio / crossover divisor | 0.10 |
| `pb_n_windows` | `n_windows_target` | target window count | 50 (nscf.in) vs 40 (v2 signature default) |
| `pb_n_prot` (0=auto) | v2 `n_prot` | fixed protected count | None |
| `--pb-seed` (CLI only, not in nscf.in) | `seed` | RNG base; ik>0 uses `ik + pb_seed·10000` (run_nscf.py:407,416) | 0 |
| `--no-davidson`, `--pb-wfn` / `rho_from_wfn`+`wfn_file` | — | pseudobands-from-existing-WFN mode, run_nscf.py:285–331 | — |

API-only (never wired by run_nscf): v1 `C_m` (filter sharpness, =1.0),
`n_kpm_moments` (500), `n_kpm_random` (10), v1 `n_windows_target=None`
geometric-window fallback, `E_fermi` (run_nscf hardcodes `E_fermi=0.0` —
absolute Ry zero, not the actual Fermi level; harmless for v2-with-`n_prot`
but shifts v1's auto protected/crossover split), v2 `dos_result` /
`n_prot_override` / `boundaries_override` (used internally by run_nscf for
k>0 consistency: k=0's DOS, protected count, and boundaries are reused at
every other k-point, run_nscf.py:404–406; v1's k>0 calls share `dos_result`
+ `n_protected` and get identical boundaries by determinism of
`dos_weighted_windows` on the shared DOS).

## Sharding / PartitionSpec assumptions

None. No `Mesh`, no `PartitionSpec`, no `shard_map`, no `device_put` in
either file — everything runs on the default (single) device with host
numpy staging. Block matvecs go through `jax.vmap(apply_H)`
(pseudobands.py:302, v2:506). Multi-process behavior lives entirely in the
caller: run_nscf's Davidson loop round-robins k-points over ranks
(run_nscf.py:220–221) but its **pseudobands loop does not**
(run_nscf.py:389 `for ik in range(1, nk)` with no rank filter) — every rank
redundantly recomputes every k-point's pseudobands with identical seeds and
only rank 0 writes (:418). Correct but wasteful at scale.

## Host vs device residency of large arrays

| array | shape | residency |
|---|---|---|
| `Phi_det`/`Phi_dav` (full Davidson manifold) | (nbnd, dim) c128 | host numpy input; **entire manifold pushed to device once** for deflation (`jnp.asarray`, pseudobands.py:422 / v2:619) — e.g. 500 bands × (2·50k) dim ≈ 800 MB; the dominant device buffer |
| filter accumulators `A` | v1 (N_S+1, k, dim); v2 (2·n_cj, k, dim) c128 | device during the M_max-step recurrence, then full host copy (`np.asarray`, pseudobands.py:411 / v2:601); ~0.5–0.8 GB at 50 windows, k=6, dim 10⁵ |
| per-window `Y_j`, `Q`, `Xi` | (k, dim) | bounced host↔device per window (`jnp.asarray` in, `np.asarray` out) |
| `Phi_out` | (n_prot + N_S·k, dim) | host numpy; caller reshapes to (n_total, nspinor, ngkmax), reorders to QE G-order, writes `WFN_pseudobands.h5` (run_nscf.py:107–118) |
| DOS grid artifacts | (10000,) grids; v2 `_cj_window_on_grid` T-matrix (M_max, n_grid) ≈ 120 MB | host numpy, transient per CJ window |

## TDA vs full-BSE handling

Not applicable in the BSE sense — but relevant forward-looking: everything
here assumes a **Hermitian** operator (`eigh`, Hermitian symmetrization
`0.5·(H+H†)`, real spectrum bounds, Chebyshev on [−1,1]). Usable for a
TDA/Hermitian-BSE Hamiltonian as-is; the full-BSE non-Hermitian block
structure would need a different filter/Ritz machinery. Also
conduction-side only: windows extend *upward* from
E_cross/E_start to E_max. Valence-side compression is specified in the
design doc (`solvers/docs/ritz_pseudobands_implementation.md:290` — run on
−H, negate energies) but **not implemented** in either file or in run_nscf.

## Spin / nspinor handling

Spin-structure-blind: operates on flat vectors of length
`dim = nspinor·ngkmax` (flattening/unflattening done by the caller:
`_make_flat_matvec` run_nscf.py:100–104, reshape back at :115). Works
unchanged for nspinor=1/2 (and would for bispinor). No collinear-nspin
channel loop anywhere (LORRAX is nspinor-based). Random probes: v1 CJ block
Ω is real-Gaussian cast to complex (pseudobands.py:405–406), v2 uses
unit-modulus random phases (v2:595–596) — deliberate v2 change, both valid
stochastic probes.

## Coupling to gw/ and isdf/ modules

No imports in either direction — coupling is entirely through the
`WFN_pseudobands.h5` file convention (weights absorbed into coefficient
norms):

- `gw/gw_init.py:64` `_band_norms = getattr(wfn, 'band_norms', None)` →
  `gw/isdf_fitting.py:282–283` / `isdf/core.py:1870–1890`
  `_band_norms_slice` divides ψ by `max(1, w_n)`;
  `centroid/pivoted_cholesky.py:929–940` same clamp for centroid selection.
- **Lost wiring**: no producer of `band_norms` exists at HEAD.
  `grep -rn "band_norms" src/` hits only the consumers above + design docs;
  `file_io/wfn_loader.py` (class `WfnLoader`, the object passed as `wfn`)
  has zero matches for `norm`. So `getattr(..., None)` always returns None
  and the entire max(1,w) renormalization chain is inert. The design doc
  explicitly specifies the missing piece
  (`solvers/docs/ritz_pseudobands_implementation.md:295`:
  "`WFNReader.band_norms`: clamp zero norms to 1.0"). Physics is still
  correct without it (absorbed norms = BGW parabands convention; GW sums
  weight themselves), but the ISDF conditioning refinement it was built for
  never activates. Per the parsed-but-unread rule: flag, don't delete.
  The only exerciser is the synthetic-norms test
  `tests/test_zq_from_psi_sm_bit_identity.py:435` (test-only).
- `solvers/dos.py` (compute_dos / windows / partition) and
  `solvers/chebyshev.py` (`jackson_coefficients`; `(M+1,)` for arg M, so
  `jackson_coefficients(M_max−1)` → `(M_max,)`, shapes consistent) are the
  only intra-package dependencies.
- No `bse/` module references pseudobands (grep over src/bse → none).

## Conventions (LORRAX-internal vs BGW-compat)

- Energies in **Ry** throughout (DOS prints, cj_resolution) — LORRAX
  internal; no eV anywhere in these files.
- Weight-absorbed coefficients = **BGW parabands compat** by design
  (`docs/pseudobands_bgw_convention.md`).
- Band ordering ascending in energy, protected-then-pseudo concatenation;
  no valence-axis flip involved (conduction-side only).
- The universal drop rule's "energy at window midpoint" for zeroed windows
  is a deliberate LORRAX convention to protect downstream cmin/vmax scans
  (comment pseudobands.py:459–463, v2:707–709).

## Suspects

### BUG-1 — Ritz rotation uses the transposed eigenvector matrix (3 sites)

`Xi = jnp.einsum('kl,kd->ld', S.T, Q)` at **pseudobands.py:241**,
**pseudobands_v2.py:330**, **pseudobands_v2.py:373**.

Per-element: `einsum('kl,kd->ld', A, Q)[l,d] = Σ_k A[k,l]·Q[k,d]`; with
`A = S.T`, `A[k,l] = S[l,k]`, so the code computes
`Xi[l,d] = Σ_k S[l,k]·Q[k,d]` — i.e. `(S @ Q)[l,d]`, combining **rows** of
S. The Ritz vector for eigenvalue θ_l is `ξ_l[d] = Σ_k S[k,l]·Q[k,d]`
(= `(Sᵀ @ Q)[l,d]`, columns of S: `eigh` returns eigenvectors in columns,
`H_proj @ S[:,l] = θ_l·S[:,l]`). Concrete 2×2 check: H_proj = [[0,1],[1,0]],
eigh → θ=(−1,+1), S = (1/√2)[[1,1],[−1,1]]; correct ξ_0 = (q0−q1)/√2 but
code yields Xi[0] = (q0+q1)/√2 — the θ=+1 eigenvector stored under θ=−1.
For complex Hermitian H_proj (the generic case) S is unitary and
⟨Xi_l|H|Xi_l⟩ = Σ_m θ_m·|(conj(S)·S)[l,m]|² — a convex mixture of all the
window's Ritz values, not θ_l. Consequences: Phi_out rows remain orthonormal
and span the correct filtered subspace, and total spectral weight is
untouched, but every CJ window's (and v2 dav n>k window's) stored
energy↔vector pairing is scrambled within the window — the stored E_out is
wrong for its vector by up to the window's spectral spread. The v1 leak
check (:450) uses θ directly and is unaffected. Fix is dropping the `.T`
(`einsum('kl,kd->ld', S, Q)` ⇒ `Σ_k S[k,l] Q[k,d]`). This silently degrades
the Ritz refinement back to "arbitrary orthonormal basis + sorted θ list",
which is bounded by window width — consistent with the module having passed
end-to-end GW comparisons.

### BUG-2 — v2 Davidson-window (n_in ≤ k) sort-pairing drops exact eigenstates

`_galerkin_ritz_dav` pads θ with `mean(E_window)` (v2:351) while the driver
pads `nodes` with the window **midpoint** (v2:643) and pads `gauss_w` with 0
(v2:644); rows are then paired by independent argsorts (v2:699–705:
`Xi_sorted = Xi[argsort(θ)]`, `nodes_sorted = nodes[argsort(nodes)]`,
`weights = √(gauss_w[argsort(nodes)])`). When any exact eigenvalue exceeds
the window midpoint the two sort orders differ and weight lands on
zero-padded Xi rows. Concrete: k=2, n_in=1, E=[9.0], window [4,10] →
θ=[9,9] (pad=mean=9), nodes=[9,7] (pad=mid=7), gauss_w=[1,0];
argsort(θ)=[0,1], argsort(nodes)=[1,0] → pairs (φ, E=7, w=0) and
(0-vector, E=9, w=1). Output: `Xi·w = 0` for both rows — the exact
eigenstate at 9 Ry is deleted and a weight-1 zero-coefficient band at E=9 is
written to WFN_pseudobands.h5 (silent loss of ~1 state of spectral weight
per affected window). Reachable in practice because `_place_windows_v2`'s
n_min=k floor uses the *smooth KPM* count while `n_in` is the *discrete*
count at this k-point, and because ik>0 reuses k=0's boundaries
(`boundaries_override`, run_nscf.py:406) against different eigenvalue sets.
Minimal fix: pad θ and nodes with the same value (or skip the double-sort
for the n_in ≤ k branch and pair positionally).

### Dead

- `_compute_window_moments` (grid version), **pseudobands_v2.py:229–254**:
  zero callers (`grep -n "_compute_window_moments" pseudobands_v2.py` → only
  the def; repo-wide grep → nothing else). Orphaned by the d1466ad Gauss
  rollback ("also revert Gauss weights"); its discrete sibling (:257) is
  still live.
- Cholesky block in `_gauss_from_moments`, **pseudobands_v2.py:173–180**:
  `L` is computed (with an eigendecomposition fallback) and never read —
  the Stieltjes recurrence below works on raw moments. Dead compute plus a
  no-op "stability" try/except.
- `Y_cj = None`, **pseudobands.py:394**: assigned, never read (v1's CJ path
  uses `Y_all`; `Y_cj` is only meaningful in v2). Copy-paste residue.
- `from functools import partial`, **pseudobands_v2.py:26**: `partial` never
  used in the file.
- `k` parameter of `_place_windows_v2`, **pseudobands_v2.py:390**: passed by
  the driver (:555 `k=k`) but unread in the body — the error metric is
  hardcoded `σ⁴/ε̄⁵` (:430) after the 11ac4f6 rollback; the docstring
  (:395–396) still advertises the σ^(2k) metric. Stale parameter + stale
  docstring.

### Redundancy / refactor

- **v1/v2 are parallel old/new paths** (house rule: no parallel routines).
  `_telescoping_filter` bodies are verbatim copies (pseudobands.py:123–141
  vs v2:131–146); the CJ step-coefficient γ formula exists **three times**
  (pseudobands.py:89–92, v2:106–110, v2:294–296); Jackson/QR/Galerkin
  boilerplate duplicated. If v2 stays the default, v1's unique value is only
  the stochastic-window construction + leak check.
- Two dataclasses named `PseudobandsResult` with incompatible schemas
  (v1:47–59 `n_det`/`n_stochastic`/`windows: WindowPartition`;
  v2:36–47 `n_prot`/`n_dav_windows`/`windows: ndarray`), forcing
  version-branching in the caller (run_nscf.py:375
  `pb0.n_prot if pb_version == 2 else pb0.n_det`). Rename or unify.

### Weird / conventions to know before refactoring

- The universal drop rule can discard **real** Davidson eigenstates: a
  stochastic window holding 1 exact state whose KPM DOS mass reads < 0.5
  is zeroed (pseudobands.py:464 tests the smooth-DOS `n_eff[j]`, not
  `n_states`). Documented as intentional in the comment (:459–463), but it
  trades exact spectral weight for KPM smoothness near sharp band edges.
- run_nscf hardcodes `E_fermi=0.0` (absolute Ry) at every call site
  (:361, :369, :400, :411) — v1's auto crossover and v2's `n_prot=None`
  auto-protect measure from absolute 0, not the real Fermi level. Works
  because production runs pass `pb_n_prot`/fixed counts, but the default is
  a trap.
- v2 header docstring says "Shifted CJ boundaries enforce a quadratic
  partition of unity (Σ w_j² ≈ 1)" — the n_eff integral (:683) is consistent
  with that (uses w², not w), with an explanatory comment (:679–682). Do not
  "simplify" it back to ∫w·ρ.
- v1's `windows` result field holds a `WindowPartition` whose `n_eff` is a
  *fraction of dim* (see Purpose §1); v2's `windows` holds raw boundary
  energies. Don't interchange.
- Seeds: k=0 uses `seed`, other k-points `ik + seed·10000`
  (run_nscf.py:407,416) — deliberate decorrelation across k with
  reproducibility per (ik, seed); keep when refactoring.
