# src/gw/scissor.py — deep-read notes (2026-07-01)

Repo: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
LOC: 269. Pure NumPy, no JAX, no I/O, no config imports. Stateless math module.

## Purpose

Affine ("scissor") extrapolation of QP corrections for bands whose DFT energy
falls outside the Σ_c(ω) frequency grid. The dynamic self-energy is stored on a
small ω-window (typically ±10 eV around E_F); bands outside it get

    ΔE(E) = (α − 1)·E + β        (separate α,β for valence vs conduction)

fit by OLS on in-grid bands, applied at the eigenvalue level (E_QP = E_DFT + ΔE).
Used both post-hoc in one-shot G0W0 (gw_jax.py) and refit per iteration in QSGW
self-consistency (sc_iteration.py). Deliberately exposes no matrix diagonal-add:
callers do `H.at[:, idx, idx].add(diag)` themselves (module docstring, lines 17–22).

Per-band in-grid rule: a band n is in-grid iff E_DFT[k,n] ∈ [ω_min, ω_max] for
EVERY k; otherwise the whole band takes the scissor uniformly (the diagonal
Σ(E) fixed point clips Σ_c at the ω-boundary for the offending k, contaminating
neighboring k through the band's dispersion — lines 24–33).

## Category

physics: post-Σ QP-eigenvalue stage (scissor extrapolation helper for G0W0/QSGW
band-energy assembly). Not chi0/W, not I/O, not resource mgmt.

## Function table

### `ScissorFit` (dataclass, frozen) — lines 55–116
Fields: `alpha_v, beta_v_ev, alpha_c, beta_c_ev, n_fit_v, n_fit_c, rmse_v_ev, rmse_c_ev`.
Semantics: fit is E_QP = α·E_DFT + β directly, so α is the "stretching factor"
(α=1 ⇔ rigid shift, α>1 ⇔ gap opens). RMSE is reported on ΔE = E_QP − E_DFT.

- `predict(E_ev (nk,nb) float64, valence_mask (nk,nb) bool) -> ΔE (nk,nb)` — lines 89–107.
  Equation: ΔE = (α−1)·E + β, `np.where(vm, delta_v, delta_c)`.
  Callers:
  - `src/gw/gw_jax.py:691` — `extrap_rel_ry = E_dft_rel_ry + fit.predict(E_dft_rel_ry * RYD_TO_EV, occ_mask_kn) / RYD_TO_EV`
  - `src/gw/sc_iteration.py:375` — `delta_ev = fit.predict(e_dft_np * RYD_TO_EV, valence_mask)`
- `summary() -> str` — lines 109–116. Human-readable one-liner. Caller: `src/gw/gw_jax.py:690` (`print0`).

### `classify_bands_in_grid(E_kn_ev (nk,nb), omega_min_ev, omega_max_ev)` — lines 123–151
Returns `(band_in_grid (nb,) bool, kn_in_grid_band (nk,nb) bool)`.
`band_in_grid = np.all((E >= ω_min) & (E <= ω_max), axis=0)`; second output is
the broadcast of the first (band-uniform mask for `np.where` consumers).
Callers (grep over src/tests/tools/scripts):
- `src/gw/gw_jax.py:506` — SC/QSGW path: builds `_in_range` mask for the
  `BandPartition` (protected = in-range bands carry full off-diag Σ; out-of-range
  bands take per-iteration scissor). Window here is `config.ppm.omega_min_ev/omega_max_ev`
  shifted by E_F to absolute eV (gw_jax.py:503–507).
- `src/gw/gw_jax.py:665` — one-shot diagonal-Σ(E) path: classification done in Ry
  against `omega_grid_ry[0]`/`omega_grid_ry[-1]` (docstring says eV but function is
  unit-agnostic; caller keeps units consistent).

### `_ols_line(x, y) -> (slope, intercept, rmse)` — lines 158–174
Closed-form OLS. Edge cases: n=0 → (0,0,0); n=1 → (0, y[0], 0);
degenerate x (Σdx² < 1e-30) → slope 0, intercept = mean(y), rmse of constant fit.
Private; only caller is `fit_scissor` (lines 247–248). Note the returned rmse
from `_ols_line` is DISCARDED by `fit_scissor` (bound to `_`), which recomputes
residuals on ΔE instead (lines 247–254).

### `fit_scissor(E_dft_kn_ev, E_qp_kn_ev, valence_mask_kn, fit_mask_kn) -> ScissorFit` — lines 177–261
All inputs (nk, nb); E_qp may be complex (real part taken, matching
`solve_diagonal_sigma_fixed_point` convention, line 220).
**Sort-and-pair semantics** (lines 229–241): per-k, E_DFT and E_QP are each
argsorted independently and paired by rank p:
  fit pair = (E_DFT_sorted[k,p], E_QP_sorted[k,p] − E_DFT_sorted[k,p]).
Rationale: robust to QSGW eigenvalue reorderings; band identity dropped in favor
of energy-rank matching. Masks (valence/fit) are reordered by the **DFT**
permutation only (`vm[rows, order_dft]`, lines 237–238) since occupation and
in-window-ness are DFT-band properties.
Then: `mask_v = vm_sorted & fm_sorted`, `mask_c = (~vm_sorted) & fm_sorted`;
two `_ols_line` fits of E_QP vs E_DFT; RMSE recomputed on
ΔE − ((α−1)·E + β) per channel.
Callers:
- `src/gw/gw_jax.py:684` — gated on `config.ppm.sigma_at_dft_extrapolate and
  0 < n_bands_in < n_bands_total`; fit in eV (E_dft_rel_ry*RYD_TO_EV vs
  E_sc_rel_ry*RYD_TO_EV), valence mask = `arange(nb) < meta.nelec`, fit mask =
  in-grid-band broadcast. Fallback when flag off or all/no bands in grid:
  out-of-grid E_QP := E_DFT (zero correction; older `eigvalsh(H_qp)` fallback
  removed as unreliable for pseudobands, gw_jax.py:660–662).
- `src/gw/sc_iteration.py:368` inside `_scissor_E_qp_for_outofrange`
  (sc_iteration.py:339–378): per-QSGW-iteration refit. Reference set = diag of
  `H_qp_dft_full` restricted to in-range bands; fast path returns E_DFT when all
  bands in-range. Output feeds `band_partition.build (scissor_E_qp_kn)` diagonal
  replacement for out-of-range bands.

## Entry points / callers summary

- `fit_scissor` <- gw_jax.py:684 (one-shot extrapolation), sc_iteration.py:368 (`_scissor_E_qp_for_outofrange`)
- `classify_bands_in_grid` <- gw_jax.py:506 (SC band-partition in-range mask), gw_jax.py:665 (one-shot per-band classification)
- `ScissorFit.predict` <- gw_jax.py:691, sc_iteration.py:375
- `ScissorFit.summary` <- gw_jax.py:690
- No imports found in tests/, tools/, scripts/ (grep for `ScissorFit|classify_bands_in_grid|fit_scissor|scissor` across those trees: only `tests/test_band_partition.py` matches, and only via the `scissor_E_qp_kn` argument of band_partition — it passes constant arrays, never imports gw.scissor). **No direct unit test for this module.**

## Flags consumed

None directly (module imports only numpy). Callers gate/parameterize with:
- `config.ppm.sigma_at_dft_extrapolate` (bool, default False — gw_config.py:326,589,956; also mirrored in gw_driver_helpers.py:33,275) — enables the one-shot fit+extrapolation in gw_jax.py.
- `config.ppm.omega_min_ev` / `omega_max_ev` — define the ω-window used for classification (gw_jax.py:503–507; and via `omega_grid_ry` endpoints at gw_jax.py:665).

## Arrays crossing the boundary

All host-side NumPy, small: (nk, nb) float64 energies (eV at fit time in both
call sites), (nk, nb) bool masks. sc_iteration converts jax.Array →
np.asarray on entry and returns jnp.asarray; the fit itself is O(nk·nb) scalars
(module docstring, lines 40–41). No sharding, no HDF5, no einsums.

## I/O

None. No files read or written.

## Suspects

### Dead
None. All three `__all__` symbols and both dataclass methods have callers in src
(grepped `ScissorFit|classify_bands_in_grid|fit_scissor|from .scissor|gw.scissor`
across src/, tests/, tools/, scripts/).

### Redundancy
- Two parallel scissor-application wrappers with near-identical
  fit→predict→Ry/eV-conversion boilerplate: gw_jax.py:663–697 (one-shot,
  in-grid mask from ω-grid endpoints in Ry) and
  sc_iteration.py:339–378 `_scissor_E_qp_for_outofrange` (per-iteration, in-range
  mask precomputed in gw_jax.py:486–512 in eV). Not copy-paste duplicates
  (different QP source: E_sc from diagonal fixed point vs diag(H_qp)), but a
  refactor could hoist the shared "fit in eV, predict, convert back to Ry,
  np.where(in_grid, qp, extrap)" pattern.
- Minor: in-grid classification happens TWICE in a full SC run — once in eV
  (gw_jax.py:506, window = ppm.omega_{min,max}_ev + E_F) and once in Ry
  (gw_jax.py:665, window = omega_grid_ry endpoints). Same physics, two unit
  conventions; consistent only if omega_grid_ry endpoints equal the config
  window exactly.

### Weird
- lines 169–170: magic epsilon `denom < 1.0e-30` guarding degenerate-x OLS;
  hypothesis: ad-hoc float64 underflow guard, fine for eV-scale data.
- lines 247–248: `_ols_line`'s RMSE return value is discarded (`_`) and
  residuals recomputed on ΔE (lines 249–254); deliberate per the comment
  ("RMSE is reported on the QP correction ... for human readability") but a
  refactor trap — the two RMSEs differ.
- lines 229–241: sort-and-pair index gymnastics — E_DFT and E_QP argsorted
  INDEPENDENTLY per k, masks reordered by the DFT permutation only. Correct by
  design for QSGW reorderings (documented lines 183–202), but silently assumes
  rank-p QP state corresponds to rank-p DFT state's occupation/in-window flags;
  breaks if QP reordering crosses the v/c boundary or the in-grid boundary.
- `classify_bands_in_grid` docstring says inputs in eV ("_ev" suffixes) but the
  gw_jax.py:665 call site passes Ry on both sides — unit-agnostic in practice,
  misleading names.
- Module docstring lines 17–22 documents a deliberately ABSENT feature
  ("No matrix-level diagonal-add is exposed") — signpost for refactorers, not a bug.
