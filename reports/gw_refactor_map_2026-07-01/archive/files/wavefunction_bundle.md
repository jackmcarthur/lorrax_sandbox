# src/gw/wavefunction_bundle.py — deep-read notes (396 LOC)

Refactor-map catalog, 2026-07-01. Repo: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D.

## Purpose

Canonical device-distributed wavefunction storage for the ISDF-basis GW pipeline. Owns
(1) `BandSlices` — the five canonical band edges b0..b4 and derived local slices,
(2) the `Wavefunctions` dataclass holding **four** sharded copies of ψ_nk(r_μ) — one per
{mesh axis x/y} × {layout bands-fast / centroids-fast} — so every downstream contraction
gets a copy whose μ axis is on the right mesh axis and whose contiguous-memory axis is
the one being summed, (3) the module-level canonical `PartitionSpec` constants for every
intermediate tensor flowing between chi0 / W / Σ / COHSEX kernels, (4) construction
(`build_wavefunctions`), QSGW band-basis rotation (`rotate_wavefunctions`), and the
band-basis projection contraction `project` / `project_ri`.

All four copies store **un-conjugated** ψ; consumers apply `jnp.conj` themselves.

Category: **data layout / sharding hub for the GW stage** (state container + sharding
contract, thin physics).

## Layout table (module docstring, lines 1-15)

| copy | shape | spec | consumer role |
|------|-------|------|---------------|
| psi_xn | (nk, s, μ_X, n) | `P(None,None,'x',None)` | G/χ₀ LHS (conj) |
| psi_xr | (nk, n, s, μ_X) | `P(None,None,None,'x')` | Σ projection LHS (conj) |
| psi_yr | (nk, n, s, μ_Y) | `P(None,None,None,'y')` | G/χ₀ RHS |
| psi_yn | (nk, s, μ_Y, n) | `P(None,None,'y',None)` | Σ projection RHS |

Spinor index s always adjacent to μ so (s,μ) sums sweep contiguous memory.

## Module-level sharding-spec constants (lines 91-125)

These are the *canonical* cross-module layout contract ("kept here so every module sees
the same canonical layout and a reshard-mismatch is caught at import time"):

| const | line | spec (verbatim) | meaning / consumers |
|-------|------|-----------------|---------------------|
| `PSI_XN_SPEC` | 91 | `P(None, None, 'x', None)` | (nk,s,μ_X,n); internal + w_isdf.py:86 |
| `PSI_XR_SPEC` | 92 | `P(None, None, None, 'x')` | (nk,n,s,μ_X); internal only |
| `PSI_YR_SPEC` | 93 | `P(None, None, None, 'y')` | (nk,n,s,μ_Y); internal + w_isdf.py:87 |
| `PSI_YN_SPEC` | 94 | `P(None, None, 'y', None)` | (nk,s,μ_Y,n); internal only |
| `G_FFT7D_SPEC` | 106 | `P(None, None, None, None, 'x', None, 'y')` | G(k) 7-D FFT-box (nkx,nky,nkz,s,μ_X,spinor,μ_Y); cohsex_sigma.py:28,77-78; ppm_sigma.py:518; w_isdf.py:82 |
| `V_FFT5D_SPEC` | 111 | `P(None, None, None, 'x', 'y')` | V_q/W_q 5-D k-space (nkx,nky,nkz,μ_X,μ_Y); cohsex_sigma.py:28,79; ppm_sigma.py:518; experimental/head_wing_schur.py (copies convention) |
| `CHI_Q_SPEC` | 116 | `P(None, None, None, 'x', 'y')` | χ(q)/σ^τ 5-D flat-q; w_isdf.py:84 |
| `G_FLATK_SPEC` | 121 | `P(None, None, 'x', None, 'y')` | G(k) 5-D flat-k (nk_flat,s,μ_X,spinor,μ_Y), output of make_flat_k_fftn; w_isdf.py:83 |
| `CHI_R_SPEC` | 125 | `P(None, 'x', 'y')` | χ/W R-space flat-k (nk_flat,μ_X,μ_Y); w_isdf.py:85 |

Note `CHI_Q_SPEC` and `V_FFT5D_SPEC` are identical values with different names (semantic
distinction, not a bug).

## Function-by-function

### `BandSlices` (dataclass, frozen) — lines 31-85
- Fields: b0 (lowest band), b1 (start of mixed val/cond sigma region), b2 (LUMO), b3
  (end of QP window), b4 (highest band); derived LOCAL slices `val=[0,b2-b0)`,
  `cond=[b2-b0,b4-b0)`, `sigma=[0,b3-b0)`, `full=[0,b4-b0)`, `occ=[0,b2-b0)`.
- `from_band_edges(b0..b4)` (55-67): validates b0<=b1<=b2<=b3<=b4, builds slices.
- Properties: `nb_full` (69-71), `nb_sigma` (73-75), `sigma_range`→(b0,b3) (77-80),
  `full_range`→(b0,b4) (82-85).
- NOTE: `occ` slice is **identical** to `val` slice (both `slice(0, b2-b0)`); no slice
  uses b1 — b1 is only consumed externally (gw_init.py:1160 passes
  `(band_slices.b1, band_slices.b3)` to `get_enk_bandrange`; sc_iteration uses `.sigma`).
- Callers: `gw_jax.py:154` (`BandSlices.from_band_edges(*meta.band_edges)`),
  `sc_iteration.py:75,112`, archived test `tests/archive/test_freqint_stage23.py:11`
  (via stale module path `gw_isdf.wavefunction_bundle`).

### `Wavefunctions` (dataclass) — lines 132-171
- Fields: psi_xn/xr/yr/yn (jax.Array, sharded per specs above), enk (nk,nb_full,
  replicated), occ (nk,nb_full, replicated), slices (BandSlices, static).
- Accessors `xn/xr/yr/yn(bands: slice)` (148-162): each `@functools.partial(jax.jit,
  static_argnames=('bands',))` — the slice is hashable (py3.12+) so jit caches per unique
  slice; comment: without these, eager-pjit gathers cause ~17 cache misses/run on Si
  4×4×4 (see also src/gw/PERFORMANCE.md:112).
- Registered as JAX pytree (167-171): data_fields = the 6 arrays, meta_fields=['slices'].
- Accessor callers (grep `wfns.xn(|wfns.xr(|wfns.yr(|wfns.yn(`):
  - cohsex_sigma.py:96-97,104-105,112 (`s.sigma`, `s.full`)
  - ppm_sigma.py:627-630, 1417-1420, 1501 (`s.full` for coh side, `s.sigma` for proj side)
  - w_isdf.py:610-611, 649-650 (`s.val`, `s.cond` for chi0)
- Whole-bundle consumers: sc_iteration.py:100 (`wfns_dft: Wavefunctions`),
  sigma_dispatch.py:118 (docstring), sigma_x_bispinor.py:63-101
  (`_wfns_with_lorentz_vertices` — `dataclasses.replace(wfns, psi_xn=…, psi_yr=…)`
  folding γ̃ vertices into the xn/yr copies; xr/yn pass through),
  tests/test_sigma_x_bispinor.py:74.

### `_build_occ(enk_full, slices, efermi)` — lines 178-198
- Role: occupation array (nk, nb_full) float64 ∈ {0,1}. If `efermi is None`:
  band-counting insulator fill (`occ[:, slices.occ] = 1.0`); else Fermi-level threshold
  `(enk <= efermi)`.
- Deliberately **numpy on host** with a loud "NOTE TO FUTURE EDITORS … DO NOT 'fix' back
  to jnp" comment: the all-jnp version cost 1.79 s → 0.18 s in wavefunction_setup
  (commit 7781b80, 2026-04-18) due to cross-device scatter.
- Callers: `build_wavefunctions` (238), `rotate_wavefunctions` (361). Internal only.

### `build_wavefunctions(psi_rmu_Y, psi_rmuT_X, *, enk_full, slices, mesh_xy, efermi=None)` — lines 201-246
- Role: assemble the 4-copy bundle from the two centroid-sampled arrays produced by
  `load_centroids_band_chunked` (see common/isdf_fitting.py:1917, file_io/tagged_arrays.py:195).
- Inputs (boundary arrays):
  - `psi_rmu_Y` (nk, nb, ns, n_rmu), sharded `P(None,None,None,'y')`, un-conjugated ψ, device.
  - `psi_rmuT_X` (nk, n_rmu, nb, ns), sharded `P(None,'x',None,None)`, **conjugated ψ***
    (layout chosen by the ISDF pair-density kernel); one `jnp.conj` (line 230) undoes it.
  - `enk_full` (nk, nb_full), replicated.
- Key property: **no cross-device reshards** — y-copies derived from psi_rmu_Y
  (psi_yr = constraint as-is; psi_yn = `.transpose(0,2,3,1)`), x-copies from
  conj(psi_rmuT_X) (psi_xn = `.transpose(0,3,1,2)`; psi_xr = `.transpose(0,2,3,1)`);
  each transpose preserves the μ axis's mesh placement.
- Caller: only `gw_init.build_wavefunction_bundle` (gw_init.py:1155,1163), which is
  called from gw_init.py:1275, 1285 (bispinor transverse bundle), 1314. Note the
  gw_init wrapper never passes `efermi` — the insulator band-count path is what
  production uses at construction time.
- No LorraxConfig flags consumed directly (band edges come in via `slices`).

### QSGW rotation kernels — lines 253-272
Four tiny jitted einsums applying per-k unitary U[k,m,n] = ⟨DFT_m|QP_n⟩:
- `_rotate_psi_xn` (253-256): `jnp.einsum('ksum,kmn->ksun', psi_xn, U, optimize=True)`
  — ψ'_xn[k,s,μ,n] = Σ_m ψ_xn[k,s,μ,m]·U[k,m,n].
- `_rotate_psi_xr` (259-262): `jnp.einsum('kmn,kmsu->knsu', U, psi_xr, optimize=True)`.
- `_rotate_psi_yr` (265-267): `jnp.einsum('kmn,kmsu->knsu', U, psi_yr, optimize=True)`
  — **byte-identical body to _rotate_psi_xr**.
- `_rotate_psi_yn` (270-272): `jnp.einsum('ksum,kmn->ksun', psi_yn, U, optimize=True)`
  — **byte-identical body to _rotate_psi_xn**.
Internal only (called by rotate_wavefunctions).

### `rotate_wavefunctions(wfns_dft, U_dft_to_qp_active, *, enk_active_new, efermi, mesh_xy, active_slice=None)` — lines 275-366
- Role: self-consistent QSGW — return a new bundle with the **active subspace**
  (default `slices.sigma`) rotated into the QP basis; bands outside stay DFT
  (their corrections come from downstream scissor extrapolation).
- Validation: errors if active_slice leaks outside the σ-window (lines 326-330);
  errors on U shape mismatch (332-335).
- Mechanics: pull active blocks via the cached jit accessors, rotate with the 4 kernels,
  `jax.lax.dynamic_update_slice_in_dim` back into copies of the full ψ (axis=-1 for
  xn/yn, axis=1 for xr/yr); enk `.at[:, active_slice].set(...)`; occ rebuilt via
  `_build_occ(enk_full, slices, efermi)` — i.e. Fermi-threshold occupations each
  iteration (efermi computed by caller `_diagonalize_and_get_efermi`).
- Caller: `sc_iteration.py:272` only (imported at sc_iteration.py:74-75).

### `project(psi_xr, psi_yn, sigma_k)` — lines 377-381
- Physics: Σ_mn(k) = Σ_{s,μ,s',μ'} ψ*_m(s,μ) Σ(s,μ,s',μ') ψ_n(s',μ') — band-basis
  projection of the ISDF-space self-energy.
- Einsums (verbatim): `'kmsx,ksxty->kmty'` (conj(psi_xr) with sigma_k), then
  `'kmty,ktyn->kmn'` (with psi_yn).
- Lives here "because the only state these contractions need is the (xr, yn) pair …
  consumers (cohsex_sigma, the AOT memory model) operate at the bundle's seam" (369-375).
- Callers: `cohsex_sigma.py:27` (`from .wavefunction_bundle import project as _project`),
  used at cohsex_sigma.py:97,105.

### `project_ri(psi_xr, psi_yn, sigma_k)` — lines 384-396
- Role: same projection but stacks [Re, Im] channels of sigma_k first
  (`sigma_ri = jnp.stack((real, imag), axis=0)`), for the windowed-PPM Σ^c(ω) τ-loop
  where the crossing window keeps only Im[coeff·σ^τ].
- Einsums (verbatim): `'kmsx,cksxty->ckmty'` then `'ckmty,ktyn->ckmn'`, result
  `.astype(jnp.complex128)`.
- Docstring: "A sharded reduce-scatter variant lives in ``ppm_sigma`` for the
  multi-device path."
- **Zero call sites found.** Grep `project_ri` across src/ tests/ tools/ scripts/:
  only the definition (line 384) plus docstring mentions in ppm_sigma.py:428
  ("Drop-in replacement for wavefunction_bundle.project_ri"), 938, bse_simple.py:27,
  and aot_memory_model/kernels/sigma_kij.py:24. Production PPM path uses
  `ppm_sigma._make_project_ri_reduce_scatter` (ppm_sigma.py:425,528,555;
  sigma_kij.py:115-136) unconditionally.

## Entry points (public, with callers)

- `BandSlices` <- gw_jax.py:154 (from_band_edges), sc_iteration.py:75,112, tests/archive/test_freqint_stage23.py (stale path)
- `Wavefunctions` <- sc_iteration.py:75,100; sigma_x_bispinor.py (via dataclasses.replace); pytree'd through cohsex_sigma / ppm_sigma / w_isdf kernels
- `build_wavefunctions` <- gw_init.build_wavefunction_bundle (gw_init.py:1155,1163) only
- `rotate_wavefunctions` <- sc_iteration.py:272 only
- `project` <- cohsex_sigma.py:27 (as _project)
- `project_ri` <- NONE (see dead suspects)
- Spec constants <- w_isdf.py:81-87 (G_FFT7D, G_FLATK, CHI_Q, CHI_R, PSI_XN, PSI_YR);
  cohsex_sigma.py:28 (G_FFT7D, V_FFT5D); ppm_sigma.py:518 (G_FFT7D, V_FFT5D)
- Accessors xn/xr/yr/yn <- cohsex_sigma.py:96-112, ppm_sigma.py:627-630/1417-1420/1501, w_isdf.py:610-611/649-650

## I/O

None. Pure in-memory device-array module; no files read or written. Upstream data comes
from `load_centroids_band_chunked` (common/isdf_fitting.py) / tagged_arrays staging;
eigenvalues via `common.load_wfns.get_enk_bandrange` in the gw_init wrapper.

## Flags consumed

None directly (no LorraxConfig / cohsex.in keys). Band edges arrive pre-parsed as
`meta.band_edges` → `BandSlices.from_band_edges` in gw_jax.py:154; `efermi` is passed
by callers.

## Suspects

### Dead
- `project_ri` (lines 384-396): zero call sites. Grepped `project_ri` in src, tests,
  tools, scripts — only definition + docstring cross-references; multi-device
  `_make_project_ri_reduce_scatter` in ppm_sigma is used unconditionally (even
  single-device, since mesh always exists). Classic superseded-single-device-path cruft.
- `tests/archive/test_freqint_stage23.py` imports `gw_isdf.wavefunction_bundle` and a
  `WavefunctionBundle` class — neither module path nor class exists anymore (archived,
  stale).

### Redundancy
- `project_ri` vs `ppm_sigma._make_project_ri_reduce_scatter` — documented parallel
  old/new pair; the unsharded one appears vestigial (matches the known "fetch_X_dyn
  next to fetch_X" pattern).
- `_rotate_psi_yr` == `_rotate_psi_xr` and `_rotate_psi_yn` == `_rotate_psi_xn`
  (byte-identical einsum bodies, separate @jax.jit functions) — 4 functions where 2
  suffice; separate names only aid layout readability.
- `BandSlices.occ` is definitionally identical to `BandSlices.val` (`slice(0, b2-b0)`).
- `CHI_Q_SPEC` value-identical to `V_FFT5D_SPEC` (intentional semantic alias, but a
  refactor could unify).

### Weird
- lines 181-190: all-caps "NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS
  INTENTIONAL … DO NOT 'fix' back to jnp" in `_build_occ`; documents a 1.79s→0.18s
  perf regression trap (commit 7781b80). Keep; a refactor must preserve host-numpy here.
- `BandSlices.b1` participates in validation and edge storage but no derived slice; its
  only consumer is gw_init.py:1160 passing `(b1, b3)` to `get_enk_bandrange` — the
  "mixed valence/conduction sigma region" concept is otherwise unused in this module.
- `_build_occ` efermi=None branch assumes an insulator with exactly (b2-b0) filled
  bands per k and per spinor bundle — fine for current systems, silently wrong for
  metals if construction-time occ is ever consumed without a later efermi rebuild
  (gw_init's wrapper never passes efermi).
- `build_wavefunctions` line 230: single `jnp.conj(psi_rmuT_X)` to undo the ISDF
  pair-density kernel's ψ* convention — a cross-module sign/conjugation contract that
  is only enforced by docstring, easy to break in a refactor.
- Accessor jit-with-static-slice trick (lines 145-162) depends on `slice` hashability
  (Python 3.12+) — an implicit interpreter-version floor.
