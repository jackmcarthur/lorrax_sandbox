# src/gw/head_correction.py — deep-read notes (2026-07-01)

832 LOC. Repo: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D

## Purpose

Centralizes everything about the q→0, G=G'=0 Coulomb "head", which the ISDF body
tensors deliberately omit (`compute_vcoul` zeroes the G=G'=0 element at q=0):

1. Head-source resolution (`override` → `epshead` from eps0mat.h5 → `s_tensor` from
   dipole.h5) via a memoized `HeadResolver`.
2. Scalar two-point PPM pole fits for the head (GN, HL, analytic-HL, fixed-Ω_h).
3. Exact static COHSEX head shifts (Σ^X, Σ^SX, Σ^{SX−X}, Σ^COH) as band-diagonal terms.
4. Rank-1 (μ,ν)-basis head injection into V_qmunu/W_qmunu at q=0 (dense + sharded variants).

Category: **physics: q=0 Coulomb-head / finite-size correction (shared by COHSEX, PPM Σ_c, and BSE)**.

## Dataclasses

| Name | Lines | Fields | Notes |
|---|---|---|---|
| `HeadSample` | 33–41 | `vc0: complex, wcoul0: complex, source: str, omega: complex` | one resolved head sample at one frequency (a.u.) |
| `HeadGNParams` | 43–55 | `omega_h_sq, omega_h, B_h, R_h, wc_head_0, wc_head_iwp, vc0, omega_p` (all float) | fitted scalar PPM pole for the head |
| `StaticHeadTerms` | 57–74 | `sigma_x_diag, sigma_sx_diag, sigma_sx_minus_x_diag, sigma_coh_diag` (jnp (nb,)), `vc0, wcoul0, wc_head_0: complex, source: str` | band-diagonal shifts in Ry |

## Function table

### `_representative_entry(diag)` — 76–84
First nonzero diagonal value for diagnostics printing. Internal only (used by
`format_static_head_diagnostics`).

### `resolve_head_override(params, omega)` — 87–102
Returns a `HeadSample` when the user supplied *both* `vhead` and the ω-appropriate
`whead_*` key; else `None`. Key selection: `"whead_0freq" if |ω| <= 1e-14 else "whead_imfreq"`.
Callers: `resolve_head_sample` (internal), `tests/test_head_correction.py:68,69`.
Flags: `vhead`, `whead_0freq`, `whead_imfreq` (cohsex.in / HeadConfig).

### `resolve_head_sample(params, input_dir, wfn, sym, meta, print_fn, omega)` — 105–199
Resolves one q=0 head sample. Order: explicit override, then `wcoul0_source`
(`"epshead"` or `"s_tensor"`, default `s_tensor`), then falls through to the *other*
source if the preferred one fails/missing (line 191 `source_order` loop). Raises
RuntimeError if nothing works.

- `from_epshead()` (122–149): reads `eps0mat.h5` via `file_io.epsreader.EPSReader`,
  takes `eps0.epshead`, calls `gw.vcoul.compute_q0_averages(wfn, epshead, meta, S_cart=None)`
  → `(vc0_mean, wcoul0)`. Static-only: for ω≠0 it prints a warning and uses epshead(0).
  Broad `except Exception` returns None (pragma: no cover).
- `from_s_tensor()` (151–189): reads `dipole.h5` via `common.chi_from_dipole.read_dipole_h5`
  → `(dipole_cart, deltaE)`; builds occupations `occ[(nk_tot, nb)]` with
  `occ[:, :min(nelec, nb)] = 1.0` (first nelec bands occupied — spin handling delegated to
  `compute_S_omega`, which receives `nspin`, `nspinor`); computes head-screening tensor
  `S_cart_omega = compute_S_omega(dipole_cart, deltaE, f_nk, V_cell, nk_tot, nspin, nspinor, [ω], eta)`,
  then `compute_q0_averages(wfn, 0.0, meta, S_cart=S_cart_omega)` (mini-BZ / Voronoi
  average of v and W over the q→0 cell).

Physics: v(q→0) mini-BZ average and W(q→0) = ε⁻¹(q→0)·v head.
Callers: `HeadResolver.at` only (grep across src/tests/tools/scripts).
Flags: `wcoul0_source`, `wcoul0_eta`, plus override keys above.

### `class HeadResolver` — 202–252
Memoized head-sample resolver (dict keyed on ω rounded to 12 decimals). Built once in
`gw_jax.main` (`src/gw/gw_jax.py:160`) from `config.head`; `.at(omega)` used for ω=0
(static COHSEX) and probe ω (dynamic PPM). Docstring: without memoization the
eps0mat/dipole read + Voronoi integral ran 3× per run.
Callers: `gw.gw_jax:160`; `gw.ppm_pipeline` (`_fit_head_correction` arg, `run_ppm_sigma`
arg at line 294). Referenced in docs of `gw_config.py:529`, `gw/coulomb/base.py:43`.
Known staleness hazard (documented in `common/chi_from_dipole.py:68–78`): in eqp
self-consistency iterations the cache holds S(ω) computed at iteration-1 energies; the
comment there lists cache-invalidation as a TODO for updated-eigenvalue reruns.
Consumed config: `config.head.{wcoul0_source, wcoul0_eta, vhead, whead_0freq, whead_imfreq}`.

### `format_head_sample_diagnostics(head, *, include_screened=True)` — 255–273
Pretty-printer ("FINITE-SIZE CORRECTIONS" block). Caller: `gw_jax.py:83`.

### `fit_head_ppm(vc0, wcoul0_static, wcoul0_probe, probe_omega)` — 280–345
Model-agnostic two-point scalar pole fit. With `w1 = W(0) − v`, `w2 = W(ω_probe) − v`,
`z² = probe_omega²` (signed: negative for GN's imaginary probe, positive for HL's real probe):

    Ω_h² = −w2·z² / (w1 − w2)
    B_h  = −w1·Ω_h²
    R_h  = B_h / (2 Ω_h)

Degenerate guard `|w1−w2| < 1e-30` → returns placeholder `Ω_h²=1.0, Ω_h=1.0, B_h=0, R_h=0`.
Negative Ω_h² branch (320–332): keeps the negative `omega_h_sq` but sets
`omega_h = |Ω_h²|^0.5` (sqrt of magnitude) and proceeds — silent sign handling.
Callers: `fit_head_gn`, `fit_head_ppm_from_samples` (both in-file). No direct external callers.

### `fit_head_gn(vc0, wcoul0_static, wcoul0_imfreq, omega_p_ry)` — 348–360
GN wrapper: `fit_head_ppm(..., probe_omega=1j·ω_p)`.
**Zero callers** in src/tests/tools/scripts (grepped `fit_head_gn\b`; only the docstring
mention in `compute_ppm_head_sigma_kij` line 644 references it). DEAD SUSPECT.

### `fit_head_ppm_from_samples(head_static, head_probe, *, probe_omega)` — 363–375
HeadSample-unwrapping wrapper. Caller: `ppm_pipeline._fit_head_correction` (line 102, GN path).

### `fit_head_hl_analytic(vc0, wcoul0_static, omega_p_sq_ry)` — 378–427
BGW-style analytic HL head pole (mirrors `Sigma/wpeff.f90` q=g=g'=0 case):

    I_ε_head = (v − W(0))/v = −W^c(0)/v
    Ω_h² = ω_p² / I_ε_head ,  B_h = −W^c(0)·Ω_h² ,  R_h = B_h/(2 Ω_h)

Guard: `I_eps_head <= 0 → I_eps_head = 1.0` ("graceful fallback; prevents sqrt of negative",
line 411–412). Degenerate w1/vc0 guard returns placeholder Ω_h=1.0 params.
Caller: only via `fit_head_hl_analytic_from_sample`.

### `fit_head_hl_analytic_from_sample(head_static, *, omega_p_sq_ry)` — 430–440
Wrapper. Caller: `ppm_pipeline._fit_head_correction:93` (HL_PPM mode; ω_p² = 16π·N_e/V_cell
computed by the caller).

### `fit_head_with_fixed_omega(vc0, wcoul0_static, omega_h_ry)` — 443–473
User-supplied Ω_h (cross-validation against BGW's analytic head pole); B_h/R_h still from
LORRAX's static head. Caller: only via `_from_sample` wrapper.

### `fit_head_with_fixed_omega_from_sample(head_static, *, omega_h_ry)` — 476–486
Wrapper. Caller: `ppm_pipeline._fit_head_correction:82`, gated on
`config.ppm.head_omega_h_ry` (flag consumed by the caller, not this file).

### `fit_head_gn_from_samples(head_static, head_imag, *, omega_p_ry)` — 489–494
GN wrapper on `fit_head_ppm_from_samples`. **Zero callers** (grep as above). DEAD SUSPECT.

### `compute_static_head_terms(*, vc0, wcoul0_static, occ, cell_volume, nk_tot, source)` — 501–540
Exact static COHSEX head shifts, band-diagonal, with prefactor `pref = 1/(V_cell·N_k)`:

    Σ^X_n      = −v_h · pref · f_n
    Σ^SX_n     = −W_h · pref · f_n
    Σ^{SX−X}_n = −(W_h − v_h) · pref · f_n
    Σ^COH_n    = +½ (W_h − v_h) · pref   (all bands)

Arrays: `occ (nb,)` {0,1} mask → jnp complex128 device arrays (nb,).
Callers: `compute_static_head_terms_from_sample` (in-file), `tests/test_head_correction.py:11,40`.

### `compute_static_head_terms_from_sample(head, *, occ, cell_volume, nk_tot)` — 543–549
Wrapper from HeadSample. Caller: `gw_jax.py:85`.

### `format_static_head_diagnostics(head)` — 552–573
Pretty-printer ("STATIC HEAD TERMS"). Caller: `gw_jax.py:87`.

### `_expand_band_diagonal_to_kij_jit(diag, *, nk_tot, nb)` — 576–581
`@jax.jit(static_argnames=('nk_tot','nb'))`; `eye(nb)·diag → broadcast (nk_tot, nb, nb)`.
Internal.

### `expand_band_diagonal_to_kij(diag, nk_tot)` — 584–593
Thin Python wrapper pulling `nb` from shape; exists to collapse ~6 eager-pjit cache
misses per call into 1 (documented in `src/gw/PERFORMANCE.md:110`, "~27 misses → 4").
Callers: `static_head_terms_to_kij` (in-file) only — no external callers.

### `static_head_terms_to_kij(head, *, nk_tot, do_screened)` — 596–625
Expands (Σ^SX or Σ^X, Σ^COH) diagonals to dense `(nk, nb, nb)` complex128 device
matrices for direct addition to COHSEX Σ matrices. `do_screened=True` → SX head, else X head.
Callers: `gw/cohsex_sigma.py:136` (SX+COH), `:230` and `:309` (bare-X head only, discards
COH), `tests/test_head_correction.py:49,50`.

### `compute_ppm_head_sigma_kij(head, *, omega_grid_ry, enk_ry, efermi_ry, n_occ, cell_volume, nk_tot, eta=1e-6)` — 628–703
Analytic q→0 head contribution to PPM Σ^c(ω) (pure numpy, host):

    Σ^c_n^head(ω) = + R_h/(V_cell·N_k) · [ f_n/(ω − ε_n + Ω_h − iη)
                                         + (1−f_n)/(ω − ε_n − Ω_h + iη) ]

(at q=0, M_nm(k, q→0, G=0)=δ_nm ⇒ head is band-diagonal). Static limit ω→ε_n reproduces
∓W^c(0)/(2 V N_k), consistent with the COHSEX head block. Occupations: `f[:min(n_occ,nb)]=1`.
Output: np.complex128 `(n_omega, nk, nb, nb)`, off-diagonals zero. Early-out zero array if
`|R_h|<1e-30` or `|Ω_h|<1e-30`.
Caller: `ppm_pipeline.py:139` (`_inject_analytic_head`).

### `format_head_diagnostics(head, cell_volume)` — 706–728
Pretty-printer ("HEAD CORRECTION (scalar GN...)"). Caller: `ppm_pipeline.py:105`.

### `_head_rank1_scalars(vhead, whead, cell_volume, omega_index, dtype)` — 748–757
Internal: `(v_h/V_cell, w_h/V_cell)` scalars; `whead` may be scalar or `(n_omega,)`
indexed by `omega_index`.

### `apply_q0_head_rank1(V_qmunu, W_qmunu, G0_mu_nu, vhead, whead, cell_volume, *, omega_index=0)` — 760–793
Rank-1 head injection in the centroid basis at q=0:

    ΔV_{q=0,μν} = (v_h / V_cell) · conj(ζ(0,μ,G=0)) · ζ(0,ν,G=0)

einsum VERBATIM: `jnp.einsum('m,n->mn', jnp.conj(G0_mu_nu), G0_mu_nu)` (line 783).
Conjugation on μ because V_{qμν} = Σ_GG' ζ*(q,μ,G) v(G,G') ζ(q,ν,G'). The 1/V_cell
matches the LORRAX V_qmunu storage convention (module comment cites
`scripts/checks/sigma_direct_check.py` as canonical reference).
Shapes: `V_qmunu (..., nkx, nky, nkz, n_μ, n_ν)`; q=0 assumed at index [..., 0,0,0, :, :]
on the k axes. In-place-style `.at[].add`.
Caller: `src/bse/bse_io.py:816,826` (BSE side). Doc reference: `file_io/tagged_arrays.py:137`,
`gw_jax.py:296`. `gw/v_q_tile.py:1578` explicitly notes the head code does NOT consume
the tiled layout there.

### `apply_q0_head_rank1_sharded(V_q0, W_q, g0_X, g0_Y, vhead, whead, cell_volume, *, omega_index=0)` — 796–832
Sharded variant for BSE `P("x","y")`-on-(μ,ν) tensors; `g0_X (n_μ,) P("x")` and
`g0_Y (n_ν,) P("y")` are duplicated copies of ζ(0,·,G=0) so
`conj(g0_X)[:,None]·g0_Y[None,:]` is local on every proc. `V_q0 (n_μ,n_ν)`;
`W_q (n_μ, n_ν, nkx, nky, nkz)` — note k axes LAST here (vs. k-before-μν in the dense
variant), q=0 slice `[:, :, 0, 0, 0]`.
Caller: `src/bse/bse_io.py:502–503`.

## Entry-point map (grep evidence: src, tests, tools, scripts)

- `HeadResolver` <- gw/gw_jax.py:160; gw/ppm_pipeline.py:35,61,294
- `compute_static_head_terms_from_sample` <- gw/gw_jax.py:85
- `format_head_sample_diagnostics` <- gw/gw_jax.py:83
- `format_static_head_diagnostics` <- gw/gw_jax.py:87
- `static_head_terms_to_kij` <- gw/cohsex_sigma.py:136,230,309; tests/test_head_correction.py
- `fit_head_ppm_from_samples` <- gw/ppm_pipeline.py:102
- `fit_head_hl_analytic_from_sample` <- gw/ppm_pipeline.py:93
- `fit_head_with_fixed_omega_from_sample` <- gw/ppm_pipeline.py:82
- `format_head_diagnostics` <- gw/ppm_pipeline.py:105
- `compute_ppm_head_sigma_kij` <- gw/ppm_pipeline.py:139
- `apply_q0_head_rank1` <- bse/bse_io.py:826
- `apply_q0_head_rank1_sharded` <- bse/bse_io.py:503
- `compute_static_head_terms` <- tests/test_head_correction.py (+ in-file wrapper)
- `resolve_head_override` <- tests/test_head_correction.py (+ resolve_head_sample in-file)
- `resolve_head_sample` <- HeadResolver.at (in-file only)
- `fit_head_gn`, `fit_head_gn_from_samples` <- NONE (dead)
- `fit_head_ppm`, `fit_head_hl_analytic`, `fit_head_with_fixed_omega`,
  `expand_band_diagonal_to_kij` <- in-file wrappers/callers only

## I/O

- Reads `<input_dir>/eps0mat.h5` (BGW epsmat HDF5) via `file_io.epsreader.EPSReader`,
  uses `.epshead` (ε⁻¹ head at q0).
- Reads `<input_dir>/dipole.h5` via `common.chi_from_dipole.read_dipole_h5` →
  `(dipole_cart, deltaE)` for the S(ω) head-screening tensor.
- Writes nothing.

## Flags consumed (HeadConfig / cohsex.in keys)

`wcoul0_source` ("s_tensor" | "epshead"), `wcoul0_eta`, `vhead`, `whead_0freq`,
`whead_imfreq`. (`config.ppm.head_omega_h_ry` and HL/GN mode selection are consumed by
`ppm_pipeline._fit_head_correction`, which then calls into this file.)

## Suspects

### Dead
- `fit_head_gn` (348–360): grep `fit_head_gn\b` across src/tests/tools/scripts → only
  definition + docstring mention at line 644; superseded by `fit_head_ppm_from_samples`
  in ppm_pipeline.
- `fit_head_gn_from_samples` (489–494): same grep, zero callers anywhere.

### Redundancy
- Wrapper-pair proliferation: `fit_head_ppm`/`fit_head_ppm_from_samples`,
  `fit_head_gn`/`fit_head_gn_from_samples`, `fit_head_hl_analytic`/`_from_sample`,
  `fit_head_with_fixed_omega`/`_from_sample`, `compute_static_head_terms`/`_from_sample` —
  five raw/HeadSample pairs; the raw halves have no external callers except
  `compute_static_head_terms` (tests only). Classic "fetch_X next to fetch_X_dyn" pattern.
- `apply_q0_head_rank1` vs `apply_q0_head_rank1_sharded`: parallel dense/sharded variants
  with *different* axis conventions (k-axes before μν in dense V, after μν in sharded W);
  both live (bse_io) but a refactor could unify.
- Three separate pretty-printers (`format_head_sample_diagnostics`,
  `format_static_head_diagnostics`, `format_head_diagnostics`) with overlapping content.

### Weird
- fit_head_ppm 306–315 & fit_head_hl_analytic 403–407: degenerate fallback returns magic
  placeholder pole `Ω_h²=1.0 Ry², Ω_h=1.0` (with B_h=0 so contribution is zero, but the
  fake pole is printed in diagnostics as if fitted).
- fit_head_ppm 320–332: negative Ω_h² branch keeps negative `omega_h_sq` but computes
  `omega_h = |Ω_h²|^0.5` and R_h from it — silent complex-pole → real-magnitude coercion.
- fit_head_hl_analytic 411–412: `I_eps_head <= 0 → 1.0` "graceful fallback" silently
  changes the physics (Ω_h² becomes bare ω_p²) rather than raising.
- resolve_head_sample 191–195: silently falls through to the non-requested head source
  when the configured one fails/is missing — user asks epshead, may silently get s_tensor.
- from_epshead 147: bare `except Exception` swallowing all errors (diagnostic print only).
- from_s_tensor 162–163: `occ[:, :min(nelec, nb)] = 1.0` fills the first nelec bands as
  occupied regardless of spin degeneracy — correct only if `compute_S_omega`'s
  nspin/nspinor arguments compensate; worth verifying against non-bispinor nspin=1.
- HeadResolver cache staleness across eqp self-consistency iterations — flagged as a known
  TODO in `common/chi_from_dipole.py:68–78`, no invalidation hook exists here.
- apply_q0_head_rank1: q=0 hardcoded at index `[..., 0, 0, 0, :, :]` (comment says "q=0 is
  index 0 on each k axis") — implicit ordering contract with the V_q builders.
- `HeadGNParams.omega_p` semantics drift: "historically the imaginary-axis magnitude; for
  HL it's the real frequency itself" (comment 298–299); `fit_head_with_fixed_omega` stores
  Ω_h into `omega_p` (line 472).
- compute_ppm_head_sigma_kij: magic `eta=1e-6` Ry default regularization; many
  `1e-30`/`1e-14` epsilon thresholds throughout the module.
