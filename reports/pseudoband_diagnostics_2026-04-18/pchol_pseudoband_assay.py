"""Pivoted-Cholesky pruning assay with gw_jax-ISDF pair windows and
optional pseudoband normalization.

Extends ``pchol_ncond_assay.py`` with:

  --pair-mode {val_cond, isdf_asym}
      "val_cond" (legacy): Gram from left=(0, n_val), right=(n_val, n_val+n_cond).
      "isdf_asym"        : Gram from left=(0, n_val+n_cond), right=(b1, n_val+n_cond)
                           where b1 = n_val - n_val_sigma. Matches the
                           window pair used by ``gw_init.fit_zeta``
                           (``(b0,b3)`` × ``(b1,b4)``) with b4 clamped
                           to the current sweep's max conduction.
  --n-val-sigma N
      Number of valence bands inside the sigma window (b2 - b1).
      Defaults to --n-val (all valence in sigma).
  --use-band-norms / --no-band-norms
      Divide ψ by max(wfn.band_norms, 1.0) on both sides before the
      pair-density einsum. Same clamp recipe as
      ``isdf_fitting.fit_zeta_chunked_to_h5``. Needed for BGW pseudoband
      WFNs where pseudobands carry amplification factors (norms > 1)
      that would otherwise dominate the Gram diagonal.
  --wfn-label NAME
      Short tag written into the JSON for plot legend labels
      (e.g. "parabands_4200", "pseudo_50sl_116").

The safety rule from the LORRAX codebase: if max(left_end, right_end)
exceeds ``0.5 · ngk_max · nspinor`` (half of the per-k plane-wave
basis), the pair-product space starts to blur with the plane-wave
space itself — centroid pruning becomes ill-posed, and the caller
should prune on the full real-space grid directly. The wrapper
``prune_candidates_by_pivoted_cholesky`` raises in this regime; this
script catches the exception per sweep point so a single over-shot
does not kill the whole run.

Usage:

    lxrun python3 pchol_pseudoband_assay.py \\
        --wfn /abs/path/WFN.h5 --wfn-label parabands_4200 \\
        --out pchol_assay_parabands.json \\
        --pair-mode isdf_asym \\
        --n-val 8 --n-val-sigma 8 \\
        --n-cond 8 16 32 64 128 256 400 \\
        --M 2400 --n-keep 1300 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from file_io import WFNReader
from common import symmetry_maps
from centroid.kmeans_isdf import (
    weighted_kmeans_jax,
    snap_centroids_to_grid,
    ensure_unique_centroids,
    BOHR_TO_ANG,
)
from centroid.charge_density import get_charge_density
from centroid.pivoted_cholesky import prune_candidates_by_pivoted_cholesky


def _compute_density(wfn, sym, mode):
    """Build the real-space weight field for k-means.

    ``valence``         — default: QE valence charge density (auto-detected
                          from qe/scf/*.save or fallback to wfn_ibz sum over
                          the first ``wfn.nelec`` bands).
    ``uniform``         — constant ρ(r) ≡ 1 on the FFT grid; k-means then
                          gives a geometrically uniform point set, untouched
                          by bond-charge concentration.
    ``pseudo_combined`` — ρ_valence + Σ_{n: pseudoband} |ψ_n(r)|² / ||ψ_n||²,
                          i.e. valence density plus unit-normalized
                          pseudoband density. Biases k-means toward regions
                          where pseudobands carry amplitude while preserving
                          bond-charge coverage.
    """
    if mode == 'uniform':
        fft_grid = tuple(int(x) for x in wfn.fft_grid)
        return np.ones(fft_grid, dtype=np.float64)
    if mode == 'valence':
        return np.asarray(get_charge_density(wfn=wfn, sym=sym, source='auto'),
                          dtype=np.float64)
    if mode == 'pseudo_combined':
        rho_val = np.asarray(get_charge_density(wfn=wfn, sym=sym, source='auto'),
                             dtype=np.float64)
        # Identify pseudobands: any band with ||ψ||² > 1.01² = pseudo.
        band_norms = np.asarray(wfn.band_norms, dtype=np.float64)
        pseudo_ids = np.where(band_norms > 1.01)[0]
        if pseudo_ids.size == 0:
            # File has no pseudobands — fall through to pure valence.
            return rho_val
        # Build ρ_pseudo via a direct IBZ sum over the pseudoband slice,
        # each unit-normalized so high-n_eff bands don't dominate.
        from centroid.charge_density import rho_from_wfn_ibz
        # Sum contiguous pseudoband ranges (they're usually contiguous past
        # the real KS bands). Use the smallest contiguous block containing
        # all pseudo_ids for simplicity.
        b_lo, b_hi = int(pseudo_ids.min()), int(pseudo_ids.max()) + 1
        rho_pseudo_raw = rho_from_wfn_ibz(wfn, sym, n_val=b_hi)  # bands 0..b_hi
        rho_pseudo_low = rho_from_wfn_ibz(wfn, sym, n_val=b_lo)  # bands 0..b_lo
        rho_pseudo = rho_pseudo_raw - rho_pseudo_low  # bands b_lo..b_hi
        # Rescale pseudoband contribution by an average 1/n_eff so each
        # pseudoband carries unit weight in ρ (instead of n_eff weight).
        n_eff_mean = float(np.mean(band_norms[b_lo:b_hi] ** 2))
        rho_pseudo /= max(n_eff_mean, 1.0)
        # Combine: valence density (bond charge) + pseudoband support.
        return rho_val + rho_pseudo
    raise ValueError(f"Unknown kmeans_density mode: {mode!r}")


def _build_candidates(wfn, sym, M, seed, mesh, kmeans_axis, density_mode):
    """Run k-means for M candidates and return (M, 3) grid indices."""
    rho = _compute_density(wfn, sym, density_mode)
    avec_ang = np.asarray(wfn.avec) * float(wfn.alat) * BOHR_TO_ANG
    rho_jax = jnp.asarray(rho, dtype=jnp.float64)
    avec_jax = jnp.asarray(avec_ang, dtype=jnp.float64)
    _, centroids_jax, _, _ = weighted_kmeans_jax(
        avec_jax, rho_jax, N_c=M, seed=seed, mesh=mesh,
        mesh_axis=kmeans_axis, force_shard=True,
    )
    centroids_frac = np.asarray(centroids_jax)
    centroid_indices, centroids_snapped, n_dups = snap_centroids_to_grid(
        centroids_frac, wfn.fft_grid, deduplicate=True,
    )
    if n_dups > 0:
        centroids_snapped = ensure_unique_centroids(
            centroids_frac, wfn.fft_grid, rho=rho,
        )
        fft_grid = np.asarray(wfn.fft_grid)
        centroid_indices = (
            np.round(centroids_snapped * fft_grid).astype(np.int64) % fft_grid
        )
    return centroid_indices


def _resolve_windows(mode, n_val, n_val_sigma, n_cond):
    """Return (band_range_left, band_range_right) for the given pair mode.

    "val_cond" returns (None, None), indicating legacy (n_val, n_cond)
    handling inside the wrapper. "isdf_asym" returns explicit ranges
    derived from Meta's b0..b4:

        b0 = 0, b1 = n_val - n_val_sigma, b2 = n_val,
        b3 = n_val + n_cond, b4 = b3  (nband clamped to current sweep)
        left  = (b0, b3)  — "all val + sigma cond"
        right = (b1, b4)  — "sigma val + all cond in this sweep"
    """
    if mode == 'val_cond':
        return None, None
    elif mode == 'isdf_asym':
        b0 = 0
        b1 = max(0, int(n_val) - int(n_val_sigma))
        b2 = int(n_val)
        b3 = int(n_val) + int(n_cond)
        b4 = b3
        return (b0, b3), (b1, b4)
    else:
        raise ValueError(f"Unknown pair-mode: {mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--wfn', required=True, help='Path to WFN.h5')
    ap.add_argument('--wfn-label', default=None,
                    help='Short label (used in plots). Default: basename parent dir.')
    ap.add_argument('--out', default='pchol_assay.json')
    ap.add_argument('--n-val', type=int, default=8)
    ap.add_argument('--n-val-sigma', type=int, default=None,
                    help='Valence bands in sigma window (default: same as --n-val).')
    ap.add_argument('--n-cond', type=int, nargs='+', required=True)
    ap.add_argument('--M', type=int, default=2400)
    ap.add_argument('--n-keep', type=int, default=1300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tol-rel', type=float, default=1e-12)
    ap.add_argument('--pair-mode', choices=['val_cond', 'isdf_asym'],
                    default='isdf_asym')
    ap.add_argument('--kmeans-density',
                    choices=['valence', 'uniform', 'pseudo_combined'],
                    default='valence',
                    help='Weight field used by k-means to pick candidates. '
                         '"valence" (default) = QE SCF density. '
                         '"uniform" = flat ρ=1 (geometrically uniform coverage). '
                         '"pseudo_combined" = valence + unit-normalized pseudoband density.')
    ap.add_argument('--use-band-norms', dest='use_band_norms',
                    action='store_true', default=True,
                    help='Apply pseudoband normalization via wfn.band_norms.')
    ap.add_argument('--no-band-norms', dest='use_band_norms',
                    action='store_false')
    args = ap.parse_args()

    if args.n_val_sigma is None:
        args.n_val_sigma = args.n_val

    # Resolve to the ultimate target so we never symlink WFN.h5 → WFN.h5.
    # Only rank 0 touches the filesystem to avoid the multi-process race
    # where two ranks simultaneously remove+recreate the symlink.
    wfn_abs = os.path.realpath(args.wfn)
    cwd_wfn = os.path.abspath('WFN.h5')
    if jax.process_index() == 0 and cwd_wfn != wfn_abs:
        if not os.path.lexists('WFN.h5') or os.path.realpath('WFN.h5') != wfn_abs:
            if os.path.lexists('WFN.h5'):
                os.remove('WFN.h5')
            os.symlink(wfn_abs, 'WFN.h5')
    # Sync: every rank waits until rank 0's symlink is in place before
    # opening the WFN file.
    from jax.experimental.multihost_utils import sync_global_devices
    sync_global_devices('wfn_symlink_ready')

    wfn = WFNReader('WFN.h5')
    sym = symmetry_maps.SymMaps(wfn)

    label = args.wfn_label
    if label is None:
        label = os.path.basename(os.path.dirname(wfn_abs)) or 'WFN'

    ngk_max = int(np.max(wfn.ngk))
    nspinor_file = int(wfn.nspinor)
    npw_basis_half = 0.5 * ngk_max * nspinor_file

    max_band_requested = max(args.n_cond) + args.n_val
    if max_band_requested > wfn.nbands:
        raise ValueError(
            f"max(n_cond)+n_val = {max_band_requested} > wfn.nbands={wfn.nbands}"
        )

    # 2-D mesh matching production.
    from jax.sharding import Mesh
    devs = jax.devices()
    nx = max(k for k in range(1, int(math.isqrt(len(devs))) + 1)
             if len(devs) % k == 0)
    ny = len(devs) // nx
    mesh = Mesh(np.asarray(devs).reshape(nx, ny), ('x', 'y'))
    if jax.process_index() == 0:
        print(f"[assay] mesh = ('x'={nx}, 'y'={ny})")

    # k-means candidate pool.
    if jax.process_index() == 0:
        print(f"[assay] Building M={args.M} candidates via k-means "
              f"(density={args.kmeans_density})…")
    t0 = time.time()
    cand_idx = _build_candidates(wfn, sym, args.M, args.seed, mesh, ('x', 'y'),
                                 density_mode=args.kmeans_density)
    M_cand = cand_idx.shape[0]
    if jax.process_index() == 0:
        print(f"[assay] got M={M_cand} candidates in {time.time()-t0:.1f}s")

    # Pseudoband norms — only used if --use-band-norms.
    band_norms = None
    if args.use_band_norms:
        band_norms = np.asarray(wfn.band_norms, dtype=np.float64)
        n_pseudo = int(np.sum(band_norms > 1.01))
        if jax.process_index() == 0:
            print(f"[assay] band_norms: {n_pseudo}/{band_norms.size} "
                  f"entries > 1.01 (pseudobands), "
                  f"max={band_norms.max():.4f}")

    kgrid_str = 'x'.join(str(int(k)) for k in wfn.kgrid)
    out = {
        'system': f"Si {kgrid_str}",
        'kgrid': [int(k) for k in wfn.kgrid],
        'wfn_path': wfn_abs,
        'wfn_label': label,
        'pair_mode': args.pair_mode,
        'kmeans_density': args.kmeans_density,
        'use_band_norms': bool(args.use_band_norms),
        'n_val_sigma': int(args.n_val_sigma),
        'nbnd_file': int(wfn.nbands),
        'nspinor': nspinor_file,
        'ngk_max': ngk_max,
        'npw_basis_half': npw_basis_half,
        'fft_grid': [int(x) for x in wfn.fft_grid],
        'nk_irr': int(wfn.nkpts),
        'nk_tot': int(sym.nk_tot),
        'n_val': int(args.n_val),
        'M_cand': int(M_cand),
        'n_keep': int(args.n_keep),
        'seed': int(args.seed),
        'runs': {},
    }

    for n_cond in args.n_cond:
        if jax.process_index() == 0:
            print(f"\n[assay] ========== n_cond = {n_cond} ==========")
        left, right = _resolve_windows(
            args.pair_mode, args.n_val, args.n_val_sigma, n_cond,
        )
        max_band = max(left[1], right[1]) if left is not None else args.n_val + n_cond
        if max_band > npw_basis_half:
            msg = (f"[assay] SKIPPING n_cond={n_cond}: max_band={max_band} "
                   f"> 0.5 * ngk_max * nspinor = {npw_basis_half:.0f}")
            if jax.process_index() == 0:
                print(msg)
            out['runs'][str(int(n_cond))] = {
                'n_cond': int(n_cond),
                'skipped': True,
                'skip_reason': msg,
            }
            continue

        t0 = time.time()
        try:
            if args.pair_mode == 'isdf_asym':
                keep_idx, rank, _G, d_final, d_taken, trR_over_trG = (
                    prune_candidates_by_pivoted_cholesky(
                        wfn, sym, cand_idx,
                        n_keep=args.n_keep,
                        band_range_left=left,
                        band_range_right=right,
                        band_norms=band_norms,
                        tol_rel=args.tol_rel,
                        verbose=True,
                        mesh=mesh,
                    )
                )
            else:
                keep_idx, rank, _G, d_final, d_taken, trR_over_trG = (
                    prune_candidates_by_pivoted_cholesky(
                        wfn, sym, cand_idx,
                        n_keep=args.n_keep,
                        n_val=args.n_val, n_cond=n_cond,
                        band_norms=band_norms,
                        tol_rel=args.tol_rel,
                        verbose=True,
                        mesh=mesh,
                    )
                )
        except ValueError as exc:
            msg = f"[assay] SKIPPING n_cond={n_cond}: {exc}"
            if jax.process_index() == 0:
                print(msg)
            out['runs'][str(int(n_cond))] = {
                'n_cond': int(n_cond),
                'skipped': True,
                'skip_reason': str(exc),
            }
            continue

        elapsed = time.time() - t0
        d_taken = np.asarray(d_taken)
        d_final = np.asarray(d_final)
        trR_over_trG = np.asarray(trR_over_trG)
        if jax.process_index() == 0:
            print(f"[assay] n_cond={n_cond}: rank={rank}, wall={elapsed:.1f}s")
        out['runs'][str(int(n_cond))] = {
            'n_cond': int(n_cond),
            'rank': int(rank),
            'band_range_left': list(left) if left is not None else None,
            'band_range_right': list(right) if right is not None else None,
            'd_taken': d_taken.tolist(),
            'trR_over_trG': trR_over_trG.tolist(),
            'd_final_max': float(np.max(d_final)),
            'd_final_mean_inactive': float(
                np.mean(d_final[d_final > 0]) if np.any(d_final > 0) else 0.0,
            ),
            'd0max': float(d_taken[0]),
            'wall_s': elapsed,
        }

    if jax.process_index() == 0:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\n[assay] wrote {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
