# src/common/coulomb_sphere.py — deep-read notes (2026-07-01)

LOC: 253. Pure host-side numpy, no JAX, no I/O. No classes.

## Purpose

Single source of truth for the bare-Coulomb G-sphere radius condition
`|q + G|² ≤ bare_coulomb_cutoff` on the flat FFT box, exposed as two
surfaces:

1. **Shared (consumer-side) sphere** — one q=0-centered sphere enlarged by
   `|q_max|_cart` that is a strict superset of every per-q ball, so the V_q
   kernel's gather is contiguous and the `sqrt_v` mask exactly zeroes the
   slack G's. Consumed by `gw.compute_vcoul.make_v_munu_chunked_kernel`.
2. **Per-q (writer-side) spheres** — WFN.h5-style padded per-IBZ-q index +
   Miller-components tables so the G-flat ζ writer (`common.isdf_fitting`)
   can store `zeta_q_G(n_q, n_rmu, ngkmax)` in the WFN.h5
   `gspace/components` layout.

The module docstring proves the superset property: for G outside the shared
sphere, `|G| > √cutoff + |q_max|` so `|q+G| ≥ |G| − |q_max| > √cutoff` at
every q → the reader's scatter of per-q on-disk coeffs into the consumer's
shared sphere is lossless.

Category: **geometry/index machinery for the Coulomb-kernel G-sphere**
(supports both the V_q physics stage and the ζ-writer I/O stage).

## Function table

### `_fft_miller_axes(nx, ny, nz)` — lines 39–45 (private)
Per-axis integer Miller indices in numpy fftfreq order:
`np.rint(np.fft.fftfreq(N) * N).astype(np.int32)` per axis.
**DEAD**: grep for `_fft_miller_axes` across src/, tests/, tools/, scripts/
finds only the definition (line 39). Both public functions inline their own
fftfreq calls instead. Leftover helper — delete or route both callers
through it.

### `_q_max_cart(bvec_f, kgrid)` — lines 48–69 (private)
Returns `max_q |q|_cart` over the *actual BGW-wrapped* q-list on `kgrid`:
enumerates all integer q-triples in `[0,K)³`, wraps `qi > K/2 → qi − K`,
maps `(qi_wrapped / kg) @ bvec_f`, takes max norm. Docstring records a
**fixed bug**: the earlier Wigner-Seitz-corner bound `±0.5/kgrid` was wrong
for even kgrids (BGW leaves `q_int = K/2` unwrapped, so `q_frac = 1/2`, not
`1/(2K)`), under-including edge G's. Caller: `compute_bare_coulomb_sphere_idx`
only (grep confirmed).

### `compute_bare_coulomb_sphere_idx(fft_grid, bvec, kgrid, vcoul_cutoff_ry, *, sys_dim=3)` — lines 72–117 (public)
Returns `(sphere_idx, n_G_sph)` where `sphere_idx` is sorted int32
`(n_G_sph,)` flat-FFT indices into `[0, nx·ny·nz)` (fftfreq order) with
`|G_cart|² ≤ (√vcoul_cutoff_ry + |q_max|_cart)²` (line 98:
`sphere_r2 = (sqrt(cutoff) + q_max_cart)**2`). Returns `(None, n_rtot)`
when `vcoul_cutoff_ry is None` or `sys_dim == 0` (no analytic sphere
reduction for the 0-D box truncation). Raises RuntimeError if flat index 0
(G=0) is excluded — impossible by construction; downstream head-correction
math relies on `sphere_idx[0] == 0` (`g0 = ζ̃[..., sphere_idx[0]]`,
comment lines 107–109).

Arrays: all host numpy; caller `compute_vcoul.py:266` converts to
`jnp.int32` device array (`sphere_idx = jnp.asarray(_sphere_np, ...)`).

Callers (grep of coulomb_sphere across src/tests/tools/scripts):
- `src/gw/compute_vcoul.py:258-266` inside
  `make_v_munu_chunked_kernel` (function-local import). `vcoul_cutoff_ry`
  arg there documents "Use to match BGW's bare_coulomb_cutoff"; threaded
  from `compute_all_V_q(... bare_coulomb_cutoff ...)` at
  compute_vcoul.py:859/982.
- `tests/test_per_q_sphere.py:112` (superset check vs per-q spheres).

### `compute_per_q_bare_coulomb_components(fft_grid, bvec, q_irr_frac, vcoul_cutoff_ry, *, sys_dim=3)` — lines 120–247 (public)
For each IBZ q builds `{G : |q+G|² ≤ vcoul_cutoff_ry}` on the full FFT
grid, pads each list to `ngkmax = max_q ngk[q]` with a sentinel, returns
dict:

| key | shape/type | meaning |
|---|---|---|
| `sphere_idx_padded` | `(n_q_ibz, ngkmax)` int32 | flat-FFT indices (fftfreq order), sorted ascending per q; pad = sentinel_flat |
| `gvec_components_padded` | `(n_q_ibz, 3, ngkmax)` int32 | Miller triples, mirrors WFN.h5 `mf_header/gspace/components` (3, ng) + leading q axis |
| `ngk_per_q` | `(n_q_ibz,)` int32 | logical sphere size; pad slots start at `ngk[q]` |
| `ngkmax` | int | max over q |
| `vcoul_cutoff_ry` | float | echoed for the on-disk header |

Physics/math: `qG_frac = q_irr_frac[:,None,:] + G_frac[None,:,:]`;
`qG_cart = qG_frac @ bvec_f`; `mask = Σ qG_cart² ≤ cutoff` (lines 192–195,
fully broadcast — allocates `(n_q, n_rtot, 3)` float64 host temporaries).
G=0 (flat index 0) asserted inside every per-q sphere (lines 199–204).
Sentinel: flat index `sentinel_flat = (nx//2)*ny*nz + (ny//2)*nz + nz//2`
(line 215), a valid in-bounds index whose Miller triple (from the fftfreq
table) is `(-nx/2, -ny/2, -nz/2)` for even axes. Sort guarantees
`sphere_idx_padded[q, 0] == 0` (comment lines 231–234). Sentinel-slot ζ
coefficients are zeroed by the **writer**, not here.

Callers:
- `src/common/isdf_fitting.py:2256-2266` in the G-flat `fit_zeta_to_h5`
  path (function-local import). Called with `vcoul_cutoff_ry=zeta_cutoff_ry`
  — a *distinct* cutoff from V_q's `bare_coulomb_cutoff`; `gw_init.py:694-699`
  resolves `cfg.head.zeta_cutoff` (default ecutrho, `gw_config.py:276`) and
  validates `zeta_cutoff ≥ bare_coulomb_cutoff` so V_q has every G it needs.
  Outputs threaded through `accumulate_rchunk_to_gflat` and stashed into
  `isdf_header` (`gvec_components`, `ngk_per_q`, `zeta_cutoff_ry` datasets
  in zeta_q.h5).
- `tests/test_per_q_sphere.py:28`, `tests/test_compute_all_V_q_g_flat.py:33`,
  `tests/test_compute_V_q_bispinor_g_flat.py:25`.

## Flags consumed (indirectly, via callers)
- `bare_coulomb_cutoff` (cohsex.in) → `vcoul_cutoff_ry` for the shared
  sphere (compute_vcoul path). Note the sandbox-wide gotcha: LORRAX default
  is 4·ecutwfc vs BGW's ecutwfc.
- `zeta_cutoff` (cohsex.in, `gw_config.py:276/540/924`, default None →
  ecutrho via `gw_init._resolve_cutoff`) → `vcoul_cutoff_ry` for the per-q
  writer sphere.
- `cell_averages` / sys_dim reach it only as the `sys_dim` kwarg (0 disables
  narrowing).

## I/O
None directly. Downstream, `gvec_components_padded` / `ngk_per_q` /
`vcoul_cutoff_ry` land in `zeta_q.h5`'s `isdf_header` group (written by
`file_io.isdf_header.write_isdf_header`, round-tripped in
`tests/test_per_q_sphere.py`).

## Suspects

### Dead
- `_fft_miller_axes` (lines 39–45): zero call sites anywhere in
  src/tests/tools/scripts (grep `_fft_miller_axes`). Both public functions
  recompute the same fftfreq axes inline (lines 100–102 and 183–185).
- `import itertools` (line 34): never used in the file (grep `itertools`
  hits only the import line).

### Redundancy
- fftfreq Miller-axis construction is written three times: `_fft_miller_axes`
  (unused), lines 100–102 (float, shared sphere), lines 183–188 (float +
  int, per-q). Trivial but exactly the kind of duplication the refactor
  should collapse.
- Conceptual near-duplication with `compute_vcoul.fft_integer_axes` /
  `exp_ikr_fftbox` (JAX-side integer G axes on the same box) — two parallel
  "G axes of the FFT box" constructions, one numpy one JAX.
- The `|q+G|² ≤ cutoff` masking logic also exists independently inside
  `compute_vcoul`'s per-q `sqrt_v` computation (`v_scaled = where(denom >
  vcoul_cutoff_ry, 0, ...)`, lines 352–353, 391–392, 840–841) — by design
  ("mask zeroes the slack G's exactly"), but it means the radius condition
  lives in 2 modules and must stay consistent.

### Weird / notes
- Docstring sentinel mismatch: module/`compute_per_q` docstrings say
  sentinel Miller `(nx/2, ny/2, nz/2)` (lines 15, 133) but the actual
  Miller triple in fftfreq order at flat position N//2 is **negative**
  `(-nx/2, -ny/2, -nz/2)` for even axes — which is what
  isdf_fitting.py:2231 and test_per_q_sphere.py's docstring say. Code is
  self-consistent (components taken from the fftfreq table, lines 216–221);
  the coulomb_sphere docstrings are the misleading ones.
- `_q_max_cart` docstring memorializes the fixed even-kgrid BGW-wrap bug —
  keep this comment through the refactor; it's the correctness argument.
- `compute_per_q` builds `(n_q, n_rtot, 3)` float64 broadcast temporaries
  (line 192-193) — fine for typical IBZ sizes, O(n_q · n_rtot · 24 B) host
  RAM; could bite at very large FFT boxes × many q.
- Both call sites use function-local imports (compute_vcoul.py:258,
  isdf_fitting.py:2256) rather than top-of-module — presumably
  import-cycle avoidance; verify during refactor.
- Units convention: `bvec` rows are expected in Bohr⁻¹ with `blat` already
  applied (isdf_fitting passes `wfn.blat * wfn.bvec`; compute_vcoul passes
  its `bvec`) — the cutoff comparison is in Ry via `|k|²` atomic-unit
  convention; nothing in this module checks that scaling, so a caller
  passing unscaled `bvec` fails silently.
