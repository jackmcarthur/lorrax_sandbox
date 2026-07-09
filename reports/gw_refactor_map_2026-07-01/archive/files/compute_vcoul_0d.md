# src/gw/compute_vcoul_0d.py — deep-read notes (2026-07-01)

LOC: 257. Pure numpy, no JAX, no LorraxConfig / cohsex.in keys consumed.

## Purpose

Cell-box (Wigner-Seitz) truncation of the Coulomb potential for 0D / molecular
systems, ported from BerkeleyGW `Common/trunc_cell_box.f90`. No analytic
reciprocal-space formula exists for box truncation, so v(G) is built by
tabulating V(r) = scale / r_min on a dense real-space grid (minimum-image over
±NCELL lattice replicas, half-grid shift to dodge r=0), FFT-ing to G-space,
and sampling with a phase correction that undoes the shift. The G=0 head is
finite (integral of 1/r over the WS cell) so no miniBZ average is needed.
Also carries a `main()` CLI that compares against a BGW `vcoul` text file.

Category: physics — Coulomb kernel / truncation (with a diagnostic CLI tail).

## Module constants (lines 34–37)

| name | value | meaning |
|---|---|---|
| `N_IN_BOX` | 2 | dense-grid density factor (BGW `n_in_box`) |
| `NCELL` | 3 | ±replica range for WS minimum image |
| `TRUNC_SHIFT` | 0.5 | half-grid shift avoiding 1/r singularity |

All mirror BGW `Common/nrtype.f90`.

## Function table

### `_round_up_fft_size(n, max_prime=5)` — lines 40–54
Round n up to a "smooth" FFT size. **BUG-SHAPED**: `primes = [2,3,5,7,11,13][:max_prime]`
uses `max_prime` as a *count*, so the default `max_prime=5` allows factors
{2,3,5,7,11}, while the docstring claims "only prime factors <= max_prime"
and "mirrors BGW Nfac=3 (factors 2,3,5)". E.g. n=98=2·7² is accepted here
but BGW would round to 100 → dense-grid size (hence v(G) values) can differ
from BGW for FFT grids whose 2× has a 7 or 11 factor.
Callers: `compute_vcoul_box` (same file), `gw/compute_vcoul.py:106`
(`compute_sqrt_vcoul_0d` imports it directly).

### `compute_vcoul_box(bdot, fft_grid, gvecs) -> vcoul (nG,) float64` — lines 57–184
Core routine. Steps:
1. `Nfft = round_up(fft_grid)` (computed line 88 but **never used** — dead local);
   `dNfft = round_up(N_IN_BOX * fft_grid)`.
2. Real-space metric: `adot = 4π² · bdot⁻¹`, then `adot[i,j] /= dNfft_i·dNfft_j`
   so grid-index deltas give physical |r|²: `|r|² = ttᵀ·adot·tt`.
3. `scale = 2·sqrt(det(adot))` — normalizes so v(G) matches Ry convention
   v(G)=8π/|G|² untruncated.
4. WS minimum image: loops offsets `l·dNfft`, l ∈ [-NCELL+1, NCELL]³ (343
   replicas, note asymmetric range matching BGW), grid coords
   `i + TRUNC_SHIFT`, quadratic form written out per-element (lines 135–140):
   `d_sq = adot00·tt1² + adot11·tt2² + adot22·tt3² + 2(adot01·tt1·tt2 + adot02·tt1·tt3 + adot12·tt2·tt3)`.
   `r_len_sq = min over replicas`. Memory: two (dNfft) float64 grids per
   replica iteration + a complex128 fftbox — host numpy, no sharding.
5. FFT: multiplies fftbox by `sqrt(N_dense)` then `np.fft.fftn(..., norm="ortho")`
   — net effect identical to a plain unnormalized `fftn`; the two-step dance is
   pointless (the twin in compute_vcoul.py just calls plain `fftn`).
6. Per-G extraction (python loop over nG): dense index `di = j if j>=0 else dNfft+j`;
   phase `2π Σ_i j_i·TRUNC_SHIFT/dNfft_i`; `vcoul[ig] = Re[fftbox_G[di]·e^{-i·phase}]`.

Equation: `v_box(G) = Re{ FFT[ 2·sqrt(det adot) / r_min(r) ](G) · e^{-2πi G·s/dNfft} }`,
s = (½,½,½) grid units; v in Rydberg; `v(G)→8π/|G|²` far from truncation effects.

Callers (grep `compute_vcoul_box` across src/tests/tools/scripts):
- `src/gw/coulomb/box_0d.py:28` (`Box0D.v_qG`) and `:50` (`Box0D.q0_average`) — the
  production path, dispatched from `coulomb/base.py:get_kernel(meta.sys_dim)` /
  `gw/vcoul.py:191`.
- `scripts/checks/w_from_eps0_0d_check.py:57,114` (`_build_v_0d_from_repo`) — diagnostic.
- `main()` in this file.
No tests import it (grep of tests/ found nothing).

Boundary arrays: inputs `bdot (3,3) f64`, `fft_grid (3,) int`, `gvecs (nG,3) int`
(host numpy); output `vcoul (nG,) f64` host numpy. Consumers divide by
`cell_volume` and cast to jnp complex128 (box_0d.py).

### `main()` — lines 187–257
CLI: `--wfn WFN.h5 --vcoul vcoul`. Loads WFN via `file_io.WfnLoader`
(imported with a `sys.path.insert` hack aliased as `WFNReader`, line 191–192),
computes v(G) at q=0 for k-point-0 G-vectors, then optionally loads a BGW
`vcoul` text file (`np.loadtxt`; cols 3:6 = G ints, col 6 = v), matches
G-vectors via dict, prints max abs/rel error. Note: `main()` does **not**
divide by cell_volume (raw Ry convention, matching BGW's vcoul file), unlike
the production Box0D path. No `python -m gw.compute_vcoul_0d` invocations
found in runs/, skills/, or scripts (grep for the module name found only
AGENTS.md, docs/architecture/codebase.md, docs/theory/physics.md mentions).

## I/O

- `main()` reads: `WFN.h5` (BGW HDF5 wavefunction, via `file_io.WfnLoader`:
  bdot, fft_grid, cell_volume, get_gvec_nk(0)); BGW `vcoul` ASCII table
  (loadtxt: columns [.., .., .., Gx, Gy, Gz, v]).
- `compute_vcoul_box` itself does no I/O.
- Nothing is written.

## Cross-module deps

- Imported by: `gw/coulomb/box_0d.py` (production), `gw/compute_vcoul.py`
  (imports `_round_up_fft_size` + the three constants for its own duplicate),
  `scripts/checks/w_from_eps0_0d_check.py`.
- Imports (main only): `src/file_io` WfnLoader via sys.path hack.

## Suspects

### Dead
- `main()` / CLI tail (lines 187–257): no `python -m` or script invocation found
  anywhere in the repo (grep `compute_vcoul_0d` across repo: only doc mentions +
  the two importers above, none call `main`). One-off validation harness.
- Local `Nfft` (line 88): computed, never used.
- Line 109 `fftbox = np.zeros(...)`: dead allocation, unconditionally
  overwritten at line 144.

### Redundancy
- **`gw/compute_vcoul.py:compute_sqrt_vcoul_0d` (lines 74–182 there) is a
  near-verbatim copy-paste** of `compute_vcoul_box`: same adot build, same
  replica loop, same quadratic form, same phase-corrected extraction — differing
  only in (a) plain `fftn` instead of the sqrt(N)+ortho dance, (b) extraction
  onto the full WFN FFT grid via a triple python loop instead of a G-list, and
  (c) returning `sqrt(v/vol)` as jnp complex128. Classic "fetch_X_dyn next to
  fetch_X" cruft; the FFT-and-sample core should be one function with two thin
  samplers.
- `main()`'s vcoul-file G-matching loop duplicates comparison logic that the
  sandbox `skills/compare` parsers are supposed to own.

### Weird
- `_round_up_fft_size` max_prime semantics (see above): count-vs-value bug;
  default silently admits 7- and 11-smooth sizes, contradicting its own
  docstring and BGW's Nfac=3 grid choice. Latent BGW-mismatch for unlucky
  FFT grids.
- sqrt(N)·ortho-FFT construction (lines 151–153): mathematically a no-op
  wrapper around unnormalized fftn; confusing given the twin uses plain fftn.
- Dense-index mapping comment (lines 160–163) justifies `dNfft + j` for j<0 by
  "dNfft is a multiple of Nfft" — after `_round_up_fft_size` dNfft need not be
  a multiple of Nfft; the mapping is still correct (frequency j mod dNfft) but
  the stated reason is wrong. No bounds check that |j| ≤ dNfft/2 (safe in
  practice since dNfft ≈ 2·fft_grid ≥ 2·|j|).
- `sys.path.insert` + `from file_io import WfnLoader as WFNReader` alias hack
  in `main()` (lines 191–192).
- Asymmetric replica range `range(-NCELL+1, NCELL+1)` — intentional (matches
  BGW loop `-ncell+1 .. ncell`) but reads like an off-by-one to the unwary.
