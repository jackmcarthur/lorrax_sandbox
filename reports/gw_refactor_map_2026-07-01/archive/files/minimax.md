# src/common/minimax.py (1049 LOC)

Deep-read notes for the GW refactor map, 2026-07-01. Read-only audit; no code executed.

## Purpose

Pure-numpy/scipy minimax quadrature solvers that build exponential-sum and sine-sum
approximations to GW energy denominators. Three solver families:

1. **Non-crossing** (x > 0): `1/x ≈ Σ_l w_l exp(-t_l x)` on `[1, R]`, O(ln R) nodes.
   Empirical error model `ε ≈ 0.31·exp[-N(3.55/ln R + 0.68)]`.
2. **Crossing** (x changes sign, regularized): `G(u) ≈ Σ_l w_l sin(τ_l u)` on `[0, A]`,
   O(A) nodes. Error model `ε ≈ exp(-0.93 - 14.25·N/A)`.
3. **Imaginary-axis**: `x/(x²+ω̂²) ≈ Σ_l w_l exp(-t_l x)` on `[1, R]` (same VarPro+Lawson
   machinery as non-crossing with a modified target).

References cited in the docstring: Hackbusch CVS 21,1 (2019); Helmich-Paris & Visscher
JCP 321, 927 (2016); Kim/Martyna/Ismail-Beigi PRB 101, 035139 (2020) (HGL); Golub &
Pereyra SIAM JNA 10, 413 (1973) (VarPro).

**Category guess**: numerical support — minimax quadrature engine for the GW
frequency-integration stage (feeds `gw/minimax_screening.py`, i.e. the chi0/W screening
grids and GN-PPM parameter extraction).

## Consumers (grep evidence)

Grep over `src/`, `tests/`, `tools/`, `scripts/` for `common.minimax` /
`from common import minimax` finds exactly ONE importer:

- `src/gw/minimax_screening.py:25` — `from common import minimax as _minimax`

Per-symbol grep (all 20 public names, `\b<name>\b` across the four trees, excluding
this file):

| symbol | callers |
|---|---|
| `G_hgl` | minimax_screening.py:538 (`_solve_crossing_scaled_cached`, `target_kind=="hgl"`) |
| `G_fermi` | minimax_screening.py:541 (`target_kind=="fermi"`) |
| `tau_max_hgl` | minimax_screening.py:539 |
| `tau_max_fermi` | minimax_screening.py:542 |
| `noncrossing_grids` | minimax_screening.py:477 (`_solve_noncrossing_scaled_cached`, lru_cache + disk cache) |
| `noncrossing_imag_grids` | minimax_screening.py:505 (`_solve_noncrossing_imag_scaled_cached`) |
| `crossing_grids` | minimax_screening.py:545 (`_solve_crossing_scaled_cached`) |
| `predict_N_noncrossing` | **none** |
| `error_estimate_noncrossing` | **none** |
| `solve_noncrossing` | **none** (Remez path; internal-only subtree) |
| `evaluate_noncrossing` | **none** |
| `solve_noncrossing_imag` | internal only (noncrossing_imag_grids:568) |
| `evaluate_noncrossing_imag` | **none** |
| `solve_crossing` | internal only (crossing_grids:809) |
| `predict_N_crossing` | **none** |
| `build_crossing_quadrature` | **none** |
| `evaluate_crossing` | **none** |
| `rescale_noncrossing` | **none** |
| `rescale_crossing` | **none** |
| `rescale_noncrossing_imag` | **none** |

`tests/test_minimax_assets.py` and `tools/generate_minimax_assets.py` import
`gw.minimax_screening`, not this module directly. `tests/test_real_axis_quadrature.py`
uses `solve_laplace_minimax_interval` from minimax_screening (which wraps the cached
solvers above). So this module is exercised only through minimax_screening's three
cached wrappers.

## Function-by-function

### Crossing target functions (lines 35–65)

| function | lines | role / formula |
|---|---|---|
| `G_hgl(u)` | 35–46 | HGL crossing target `Im[√(π/2)·exp(-(u+i)²/2)·(1+i·erfi((u+i)/√2))]`, rewritten via Faddeeva `wofz` for stability: `√(π/2)·(2·exp(-z²/2) − w(−z/√2))`, `z = u+i`. Zeroed for |u|<1e-30. |
| `G_fermi(u)` | 49–55 | Fermi target `1/u − π/(2 sinh(πu/2))`; guard |u|<1e-14 → 0. |
| `tau_max_hgl(eps_q)` | 58–60 | Effective sine-frequency support `√(2 ln(1/ε_q))`. |
| `tau_max_fermi(eps_q)` | 63–65 | `0.5 ln(1/ε_q)`. |

### Non-crossing: 1/x on [1,R] (lines 68–426)

Module constants (72–75): `_NC_LN_C = ln(0.3112)`, `_NC_A = 3.5456`, `_NC_B = 0.6845`
— empirical regression (`R² = 0.995 on eps in [1e-5, 1e-2]` per comment).

| function | lines | role |
|---|---|---|
| `predict_N_noncrossing(R, target_error)` | 78–82 | Invert the empirical error model for N. **DEAD** (no callers). |
| `error_estimate_noncrossing(N, R)` | 85–88 | Forward error model `exp(_NC_LN_C − N·rate)`, `rate = _NC_A/ln R + _NC_B`. **DEAD**. |
| `_nc_varpro_residual(s, x_grid, g, W_sqrt)` | 93–99 | VarPro residual `(I − UUᵀ)(W·g)` with basis `Φ_il = exp(-x_i·e^{s_l})`, U from SVD. |
| `_nc_solve_once(...)` | 102–128 | One scipy `least_squares` TRF VarPro solve for log-nodes s with bounds `[s_lo,s_hi]`, tolerances 1e-14; then linear weights via `lstsq`. |
| `_nc_solve_at_R(N, Ri, s_init, lawson_iter=4)` | 131–152 | Solve at single R on log grid of M = max(200, 15N) points; 4 Lawson IRLS reweight rounds (`irls_w = 1/max(|e|, 1e-2·max|e|)`, normalized). |
| `_nc_solve_varpro(N, R, lawson_iter=4)` | 155–202 | Continuation in R (start `max(2, min(e^{0.7N}, R))`, ×4 per step); two inits: Hackbusch-style `s_hack = ln[π²(l−½)/(2 ln 4R)]` and uniform-in-s; keeps the better on M_eval = max(1000, 40N) eval grid. Returns sorted `(t, w, err)`. |
| `noncrossing_grids(R, eps, N_start=2, N_max=60)` | 205–216 | Minimal-N search: increments N until `err < eps`. **Live**: minimax_screening:477. NOTE: on exhaustion returns the N_max result silently (no warning). |

Remez-exchange path (219–420), header comment "enhanced, standalone use":

| function | lines | role |
|---|---|---|
| `_nc_hack_init_s(N,R)` | 221–222 | Same s_hack formula as line 190 (duplicated). |
| `_nc_loguni_init_s(N,R)` | 225–228 | Uniform-in-s init (duplicates lines 169–170+193). |
| `_nc_phi(x,s)` | 231–233 | Basis matrix `exp(-x⊗e^s)`. |
| `_nc_ls_weights(x,s)` | 236–238 | lstsq weights for target 1/x. |
| `_nc_err_curve(x,s,w)` | 241–243 | `1/x − Φw`. |
| `_nc_select_alternating_extrema(x,e,m)` | 246–287 | Pick m alternating-sign local extrema of the error curve (block-merge same-sign runs; pad/trim to exactly m; force alternating sign pattern `first·(−1)^i`). Classic Remez reference-set selection. |
| `_nc_newton_equioscillation(xr, signs, s0, w0, lam0, R, maxit=20)` | 290–340 | Damped Newton on the equioscillation system `1/x_r − Σ w_l e^{-t_l x_r} = σ_r·λ`; Jacobian blocks: `∂/∂s = (w·t)·x·E`, `∂/∂w = −E`, `∂/∂λ = −signs`; backtracking line search α ∈ {1,…,0.05}. |
| `_nc_remez_at_R(N, R, s_init, max_outer=6)` | 343–383 | Outer Remez loop: dense grid Md = max(2000, 120N), warm-start via `_nc_solve_at_R(lawson_iter=0)` (in try/except pass!), exchange + Newton until reference set stops moving or error stops improving by factor 0.9999. |
| `solve_noncrossing(N, R)` | 386–420 | R-continuation (×2 schedule from 2) over both inits, keep best. Docstring: "More accurate than the VarPro+Lawson solver used by noncrossing_grids, but slower. Use this for standalone high-accuracy solves." **DEAD** — nothing calls it, so the whole Remez subtree (~200 LOC, lines 219–420) is unreachable. |
| `evaluate_noncrossing(x, tau, w)` | 423–426 | `Σ_l w_l exp(−τ_l x)` evaluator. **DEAD**. |

### Imaginary-axis: x/(x²+ω̂²) on [1,R] (lines 429–577)

| function | lines | role |
|---|---|---|
| `_imag_target(x, omega_hat)` | 433–435 | `x/(x² + ω̂²)`. |
| `_imag_varpro_residual` | 438–444 | **Verbatim copy** of `_nc_varpro_residual` (only cosmetic diff: discards SVD sig/Vt). |
| `_imag_solve_once` | 447–473 | **Verbatim copy** of `_nc_solve_once` with `_imag_varpro_residual`. |
| `_imag_solve_at_R(N, Ri, omega_hat, s_init, lawson_iter=4)` | 476–497 | Copy of `_nc_solve_at_R` with `g = _imag_target(x, ω̂)` instead of `1/x`. |
| `solve_noncrossing_imag(N, R, omega_hat, lawson_iter=4)` | 500–559 | Copy of `_nc_solve_varpro` with imag target; `ω̂ = ω_p/x_min` dimensionless; docstring notes ω̂ = 0 recovers static 1/x "up to normalization". Called internally by `noncrossing_imag_grids` only. |
| `noncrossing_imag_grids(R, omega_hat, eps, N_start=2, N_max=60)` | 562–571 | Minimal-N search. **Live**: minimax_screening:505. Same silent-exhaustion return as noncrossing_grids. |
| `evaluate_noncrossing_imag(x, t, w)` | 574–577 | Duplicate of `evaluate_noncrossing`. **DEAD**. |

### Crossing: sin-sums on [0,A] (lines 580–1018)

| function | lines | role |
|---|---|---|
| `_cr_varpro_lm(tau, u_grid, g, max_iter=120, ...)` | 586–641 | Hand-rolled VarPro Levenberg-Marquardt in the sin basis `Φ_il = sin(u_i τ_l)`; SVD pseudo-inverse with cutoff `1e-14·σ₀`; μ ∈ [1e-15, 1e8], stall counter ≥10 exits; nodes clipped to `[1e-10, tau_hi]` and sorted each step. |
| `_cr_minimax_lp(Phi, g, method='highs-ipm')` | 644–673 | L∞ fit as LP: `min t s.t. |g − Φw| ≤ t`, w split into w⁺−w⁻; objective has tiny `1e-12` L1 penalty on w⁺,w⁻ (tie-breaking/sparsity nudge). Returns `(None, inf)` on LP failure. |
| `_cr_lp_backward_elim(N, A, G_func, tau_max_val)` | 676–715 | Dense candidate grid of K = min(500, max(30, ⌈3·A·1.3·τ_max/π⌉+20)) frequencies on `[0.01, 1.3·τ_max]`; full LP; keep top-3N by |w|; backward-eliminate smallest-|w| one at a time (re-solving LP) down to N. |
| `_cr_final_lp_weights(tau, u_eval, g_eval)` | 718–724 | Weights-only minimax LP at fixed τ; lstsq fallback. |
| `solve_crossing(N, A, G_func, tau_max_val, lawson_iter=5)` | 727–794 | 3-stage architecture: (1) LP + backward elim → N freqs; (2) VarPro-LM + 5 Lawson IRLS rounds; (3) final LP weights, keep min(err_lp, err_vp). Tries 3 starts: LP-eliminated, linear `linspace(0.1, 1.1·τ_max, N)`, Chebyshev `τ_max·½(1−cos(πk/(N+1)))`. Internal-only caller: `crossing_grids`. |
| `crossing_grids(A, eps, G_func, tau_max_func, eps_q=1e-3, N_max=500)` | 797–814 | Minimal-N search starting from analytic estimate `N_est = ⌈A·τ_max/π⌉ − 5`. **Live**: minimax_screening:545. Same silent-exhaustion return. |

Higher-level crossing builder (817–1012) — a second, PHYSICAL-UNITS crossing pipeline:

Constants (820–822): `_CR_INTERCEPT = −0.93`, `_CR_SLOPE = −14.25` (error model),
`_CR_TAU_MAX = √(2 ln 1e3)` — hardcodes ε_q = 1e-3 (inconsistent with the eps_q
parameter that the live path threads through).

| function | lines | role |
|---|---|---|
| `predict_N_crossing(xi_eff_target, E_bw, target_error, a_eff_est=1.35)` | 824–846 | Node-count predictor from error model, `A_est = E_bw/(ξ_eff/a_eff)`, `N = ⌈max(ratio,0.15)·A⌉`, min 5. **DEAD**. |
| `_cr_delta_from_sines(tau, w, A)` | 849–851 | `δ = A − Σ_l w_l (sin τ_l A − τ_l A cos τ_l A)/τ_l²` (∫ of the sine fit). |
| `_cr_a_eff_from_delta(delta, A)` | 854–864 | Solve `a·arctan(A/a) = δ` for effective Lorentzian width via brentq; manual bisection fallback (200 iter). |
| `_cr_solve_1overx(N, A_dim, u_min=5.0)` | 867–922 | Fit `1/u` on `[u_min, A]` with N sines: ζ-scan over 30 spacings `τ_n = nπ/(fit_len+ζ)`, then a THIRD hand-rolled VarPro-LM copy (100 iters, per-column Jacobian loop), final lstsq weights on 5000-pt grid. |
| `build_crossing_quadrature(N, xi_eff_target, E_bw, tol=0.05, verbose=True)` | 925–1012 | Binary-search window size A (30 iters) so that the fit's effective Lorentzian width `ξ_eff = a_eff·E_bw/A` hits target within 5%. `verbose=True` default prints progress to stdout. Returns `(tau, w, info-dict{xi_0, xi_eff, a_eff, u_min, A_dim, fit_err, xi_eff_target, N_over_A})`. **DEAD** — entire subtree (~190 LOC incl. helpers above) unreachable. |
| `evaluate_crossing(x, tau, w, xi_0)` | 1015–1018 | `F(x) = Σ w_l sin(τ_l x/ξ₀)`. **DEAD**. |

### Physical rescaling helpers (1021–1049)

| function | lines | formula |
|---|---|---|
| `rescale_noncrossing(t, w, E_gap)` | 1025–1030 | `τ_phys = t/E_gap, W_phys = w/E_gap`. **DEAD**. |
| `rescale_crossing(tau, w, xi)` | 1033–1038 | `t_phys = ξ·τ, W_phys = w/ξ`. **DEAD**. |
| `rescale_noncrossing_imag(t, w, E_gap)` | 1041–1049 | Same as rescale_noncrossing (identical body); target `Σ w exp(−tE) ≈ E/(E²+ω_p²)`, `ω_p = ω̂·E_gap`. **DEAD**. |

All three rescalers are dead — minimax_screening evidently does its own unit
rescaling around the cached solver wrappers (`solve_laplace_minimax_interval` etc.).

## Flags / config consumed

None directly. No LorraxConfig, no cohsex.in keys, no env vars. All parameters
(R, A, eps, eps_q, omega_hat, max_nodes, target_kind hgl/fermi) arrive as plain
floats/ints from `gw/minimax_screening.py` (which itself is driven by
`gw/minimax_config.py::MinimaxConfig`).

## I/O

None. Pure computation on numpy arrays. The disk caching of solver results
(`_load_minimax_disk_cache`/`_store_minimax_disk_cache`, plus the shipped minimax
asset files handled by `tools/generate_minimax_assets.py`) lives entirely in
`gw/minimax_screening.py`, not here.

## Arrays crossing the boundary

All tiny host-side numpy float64: node vectors `t`/`tau` (N ≤ 60 non-crossing,
N ≤ 500 crossing) and weight vectors `w` (same shape), plus scalar `err`.
Nothing touches JAX/device.

## Dead suspects (grep evidence: `grep -rn "\b<name>\b" src tests tools scripts --include=*.py`, zero hits outside this file)

- `predict_N_noncrossing`, `error_estimate_noncrossing` — error-model utilities, no callers.
- `solve_noncrossing` + the entire Remez subtree it gates (`_nc_hack_init_s`,
  `_nc_loguni_init_s`, `_nc_phi`, `_nc_ls_weights`, `_nc_err_curve`,
  `_nc_select_alternating_extrema`, `_nc_newton_equioscillation`, `_nc_remez_at_R`;
  lines 219–420, ~200 LOC).
- `evaluate_noncrossing`, `evaluate_noncrossing_imag`, `evaluate_crossing` — evaluators, no callers (tests evaluate via minimax_screening instead).
- `predict_N_crossing`, `build_crossing_quadrature` + helpers (`_cr_delta_from_sines`,
  `_cr_a_eff_from_delta`, `_cr_solve_1overx`, constants `_CR_INTERCEPT`, `_CR_SLOPE`,
  `_CR_TAU_MAX`; lines 817–1012, ~190 LOC).
- `rescale_noncrossing`, `rescale_crossing`, `rescale_noncrossing_imag` (lines 1021–1049).

Net: only ~7 of 20 public symbols are live; roughly 40% of the file (~430 LOC) is
unreachable from any caller in src/tests/tools/scripts.

## Redundancy suspects

1. **`_imag_*` family is a copy-paste of `_nc_*`** — `_imag_varpro_residual` ≡
   `_nc_varpro_residual`, `_imag_solve_once` ≡ `_nc_solve_once`,
   `_imag_solve_at_R`/`solve_noncrossing_imag` ≡ `_nc_solve_at_R`/`_nc_solve_varpro`
   with the sole substantive change `g = x/(x²+ω̂²)` instead of `g = 1/x`. Classic
   "fetch_X_dyn next to fetch_X" pattern; one target-parameterized solver would do.
2. **Three independent VarPro-LM implementations**: scipy-TRF VarPro (`_nc_solve_once`,
   exp basis), hand-rolled LM (`_cr_varpro_lm`, sin basis, vectorized Jacobian), and a
   second hand-rolled LM inside `_cr_solve_1overx` (sin basis, per-column Jacobian loop).
3. **Two parallel non-crossing solvers**: VarPro+Lawson (`_nc_solve_varpro`, live via
   `noncrossing_grids`) vs Remez (`solve_noncrossing`, dead) — explicitly acknowledged
   parallel old/new paths in the docstrings.
4. **Two parallel crossing pipelines**: dimensionless `solve_crossing`/`crossing_grids`
   (live) vs physical-units `build_crossing_quadrature` (dead).
5. Duplicated init formulas: `s_hack = ln[π²(l−½)/(2 ln 4R)]` written out at lines 190,
   222, 546; uniform-in-s init at 193 and 227/550; Lawson IRLS block repeated 4×
   (143–150, 488–495, 755–763, and within `_cr_varpro_lm` implicitly).
6. `rescale_noncrossing` and `rescale_noncrossing_imag` have identical bodies.
7. `evaluate_noncrossing` ≡ `evaluate_noncrossing_imag`.

## Weird code

- **Empirical magic constants** at lines 73–75 (`0.3112, 3.5456, 0.6845`, "R²=0.995")
  and 820–822 (`−0.93, −14.25`) — regression fits from some offline experiment; only
  consumed by dead predictor functions, so their provenance is unverifiable and they
  can rot silently.
- **`_CR_TAU_MAX = √(2·ln 1e3)`** (line 821) hardcodes ε_q = 1e-3 while the live
  crossing path takes eps_q as a parameter — a frozen default inside the dead subtree.
- **Silent N_max exhaustion**: `noncrossing_grids` (216), `noncrossing_imag_grids`
  (571), `crossing_grids` (814) all fall through and return the last (failing) result
  with no warning when eps is not reached; caller must check err itself
  (minimax_screening stores err in its cache, so it can, but nothing raises).
- **`try/except Exception: pass`** at `_nc_remez_at_R` lines 349–353 around the
  warm-start solve — swallows all errors (dead path, but still).
- **LP objective 1e-12 L1 nudge** in `_cr_minimax_lp` (650–652) — undocumented
  tie-breaking regularizer; magnitude interacts with `highs-ipm` tolerances.
- **Assorted magic slack factors** on τ bounds: candidate grid up to `1.3·τ_max`
  (678), clip to `1.5·τ_max` (742, 883, 907), linear start to `1.1·τ_max` (786),
  `u_min = 5.0` hardcoded (939, 867) — none derived or documented.
- **`print` diagnostics with `verbose=True` default** in `build_crossing_quadrature`
  (950–1008) — notebook-style output in a library module (dead code).
- `noncrossing_grids`-family loop variables `t, w, err` leak from the for-loop into
  the fallback return — works in CPython but NameErrors if N_start > N_max.
