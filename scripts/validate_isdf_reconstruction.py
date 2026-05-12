#!/usr/bin/env python3
"""Validate LORRAX's ISDF pair-product reconstruction against the true
pair density, on the full real-space FFT grid, for randomly-sampled
(band, band, k-point, q-vector) tuples.

Physics
-------
LORRAX's ISDF fit produces, per q, a matrix ``ζ_q(μ, r)`` stored in
``zeta_q.h5`` (one row per centroid μ, one column per real-space FFT
grid point r). The ISDF approximation states that for any pair of Bloch
states (ψ_{n,k}, ψ_{m,k-q}),

    ψ*_{m,k-q}(r) · ψ_{n,k}(r)
        ≈ Σ_μ [ψ*_{m,k-q}(r_μ) · ψ_{n,k}(r_μ)] · ζ_q(μ, r)                (1)

where r_μ are the centroid points selected by the pruning stage.
Because ζ_q is fit once and used for every (n, m) pair, the key
practical question is: **how tightly does (1) actually hold when you
sample random (n, m, k, q)?**

This test answers that directly. For each sample:

1. Read coefficients ``c_{n,k}(G)`` and ``c_{m,k−q}(G)`` from the WFN.h5
   file (with the full-BZ unfolding that LORRAX uses internally).
2. Compute the true pair density on the full FFT grid:
     P_true(r) = ψ*_{m,k-q}(r) · ψ_{n,k}(r)
3. Evaluate both wavefunctions at the centroid indices r_μ.
4. Contract with ``zeta_q`` at the matching (qx, qy, qz) to produce
     P_ISDF(r) = Σ_μ [ψ*_{m,k-q}(r_μ) · ψ_{n,k}(r_μ)] · ζ_q(μ, r).
5. Compute the relative L2 error
     ε = ‖P_true − P_ISDF‖₂ / ‖P_true‖₂
   and the per-point max.

Band normalisation
------------------
If the WFN file is a BerkeleyGW pseudoband output, the stored
``c_{n,k}(G)`` coefficients carry non-unit norms (‖ψ_n‖² = n_eff). The
ISDF fit divides both left and right ψ by ``max(‖ψ‖, 1.0)`` before
computing pair densities, so the ζ_q stored on disk is the ζ that
reconstructs *normalized* pair products. This test reproduces that
convention: on both sides of the comparison we use ψ / max(‖ψ‖, 1.0).
The division cancels between numerator and denominator in ε but
affects the absolute scale of P_true and P_ISDF.

(k − q) Umklapp
---------------
For Bloch functions on a uniform k-grid, ψ_{m, k−q+G₀}(r) = ψ_{m, k−q}(r)
exactly — the gauge change e^{−iG₀·r} in u_{m,k-q}(r) cancels the phase
e^{iG₀·r} in the plane-wave envelope. This test therefore uses the
full-BZ k-index of k−q (modulo integer umklapp) without any explicit
phase factor.

Usage
-----
Simplest form::

    $ python3 validate_isdf_reconstruction.py --run-dir /path/to/lorrax_cohsex/

Given a LORRAX COHSEX run directory that contains ``cohsex.in``,
``WFN.h5``, ``centroids_frac.txt``, and ``tmp/zeta_q.h5``, this draws
100 random (n, m, k, q) samples and reports aggregate + per-sample
errors to ``--out`` (default: ``./isdf_validation/``).

Full flag list::

    --run-dir       LORRAX run dir (provides cohsex.in, WFN.h5, centroids,
                    tmp/zeta_q.h5 by default)
    --zeta-h5       explicit ζ-file path (else ``<run-dir>/tmp/zeta_q.h5``)
    --wfn           explicit WFN.h5 path (else ``<run-dir>/WFN.h5``)
    --centroids     explicit centroids file (else ``<run-dir>/centroids_frac.txt``)
    --n-samples     number of (n, m, k, q) tuples to evaluate (default 100)
    --seed          RNG seed for sampling (default 42)
    --band-pool     "val_only" / "cond_only" / "mixed" (default mixed)
                    "mixed" draws (valence, valence), (valence, conduction),
                    and (conduction, conduction) roughly in equal parts.
    --out           output directory for JSON + plots (default ./isdf_validation/)

The script is also importable::

    from validate_isdf_reconstruction import (
        load_run,
        sample_pair_products,
        reconstruct_one,
        aggregate,
    )

See the docstrings of each function for their contracts.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import h5py


# ═══════════════════════════════════════════════════════════════════════
# Run-directory bookkeeping
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RunInputs:
    """Everything the test needs from a LORRAX run directory."""
    wfn_path: Path
    zeta_path: Path
    centroids_path: Path
    n_val: int
    n_cond: int
    nband: int
    fft_grid: tuple[int, int, int]
    kgrid: tuple[int, int, int]
    isdf_pair_mode: str


def parse_cohsex_ini(path: Path) -> dict:
    """Parse the [cohsex] section of a cohsex.in INI file into a dict.

    Values are returned as strings; the caller converts types.
    """
    cp = configparser.ConfigParser()
    cp.read(path)
    if 'cohsex' not in cp.sections():
        raise ValueError(f"No [cohsex] section in {path}")
    return dict(cp['cohsex'])


def load_run(run_dir: Path | str, *,
             wfn: Path | str | None = None,
             zeta_h5: Path | str | None = None,
             centroids: Path | str | None = None) -> RunInputs:
    """Pull everything the test needs from a LORRAX COHSEX run directory.

    Args:
        run_dir: directory produced by ``gw_jax`` COHSEX. Expected to
            contain ``cohsex.in``; other files are relative to it
            unless overridden.
        wfn, zeta_h5, centroids: optional explicit overrides.

    Returns:
        ``RunInputs`` bundle with all resolved paths + band metadata.
    """
    run_dir = Path(run_dir)
    ini = parse_cohsex_ini(run_dir / 'cohsex.in')
    wfn = Path(wfn) if wfn is not None else run_dir / ini.get('wfn_file', 'WFN.h5')
    if not wfn.is_absolute():
        wfn = (run_dir / wfn).resolve()
    zeta = (Path(zeta_h5) if zeta_h5 is not None
            else run_dir / 'tmp' / 'zeta_q.h5')
    cand = (Path(centroids) if centroids is not None
            else run_dir / ini.get('centroids_file', 'centroids_frac.txt'))
    if not cand.is_absolute():
        cand = (run_dir / cand).resolve()

    # Band + grid metadata from zeta_q.h5 attrs (authoritative).
    with h5py.File(zeta, 'r') as f:
        fft_grid = tuple(int(x) for x in f.attrs['fft_grid'])
        kgrid = tuple(int(x) for x in f.attrs['kgrid'])
        isdf_pair_mode = str(f.attrs.get('isdf_pair_mode', 'spin_traced'))

    n_val = int(ini['nval'])
    n_cond = int(ini['ncond'])
    nband = int(ini.get('nband', n_val + n_cond))

    return RunInputs(
        wfn_path=wfn, zeta_path=zeta, centroids_path=cand,
        n_val=n_val, n_cond=n_cond, nband=nband,
        fft_grid=fft_grid, kgrid=kgrid, isdf_pair_mode=isdf_pair_mode,
    )


# ═══════════════════════════════════════════════════════════════════════
# Single-band wavefunction evaluation
# ═══════════════════════════════════════════════════════════════════════


class WFNView:
    """Lightweight reader for c_{n,k}(G) + band_norms from BerkeleyGW WFN.h5.

    Kept separate from ``file_io.WFNReader`` so this test is importable
    outside a shifter container. Only exposes what the test needs.
    """

    def __init__(self, path: Path | str):
        self._f = h5py.File(path, 'r')
        h = self._f
        self.fft_grid = tuple(int(x) for x in h['mf_header/gspace/FFTgrid'][...])
        self.kgrid = tuple(int(x) for x in h['mf_header/kpoints/kgrid'][...])
        self.nkpts = int(h['mf_header/kpoints/nrk'][()])
        self.nbands = int(h['mf_header/kpoints/mnband'][()])
        self.nspinor = int(h['mf_header/kpoints/nspinor'][()])
        self.ngk = np.asarray(h['mf_header/kpoints/ngk'][:], dtype=np.int64)
        self.kpt_starts = np.concatenate(([0], np.cumsum(self.ngk[:-1])))
        # IBZ fractional k-coords.
        self.k_irr_frac = np.asarray(h['mf_header/kpoints/rk'][:],
                                     dtype=np.float64)
        # All G-vectors (flat list, indexed by kpt_starts[ik]..+ngk[ik])
        self._gvecs_flat = h['wfns/gvecs']   # (ng_total, 3) int32
        self._coeffs = h['wfns/coeffs']      # (nb, ns, ng_total, 2) float64

        # Compute band norms at k=0 exactly like wfnreader.py:104-112.
        start0 = int(self.kpt_starts[0])
        end0 = start0 + int(self.ngk[0])
        c0 = self._coeffs[:, :, start0:end0, :]
        self.band_norms = np.sqrt(
            np.sum(c0[:, :, :, 0] ** 2 + c0[:, :, :, 1] ** 2, axis=(1, 2))
        )
        self.band_norms = np.where(self.band_norms > 0.5, self.band_norms, 1.0)

    def close(self):
        self._f.close()

    def coeffs_at_k(self, k_irr_idx: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (c_{n,k_irr}(G), G_miller) for a single band at one IBZ k.

        Output:
            c : (nspinor, ngk_k) complex128
            G : (ngk_k, 3) int32 Miller indices
        """
        start = int(self.kpt_starts[k_irr_idx])
        end = start + int(self.ngk[k_irr_idx])
        raw = self._coeffs[n, :, start:end, :]     # (ns, ngk, 2)
        c = (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex128)
        g = np.asarray(self._gvecs_flat[start:end, :], dtype=np.int32)
        return c, g


def psi_on_grid(c: np.ndarray, gvecs: np.ndarray,
                fft_grid: tuple[int, int, int],
                k_frac: np.ndarray) -> np.ndarray:
    """Evaluate ψ_{n,k}(r) on the full real-space FFT grid from (c, G).

    Args:
        c: (nspinor, ngk) complex — plane-wave coefficients.
        gvecs: (ngk, 3) int Miller indices paired with ``c``.
        fft_grid: (nx, ny, nz) real-space grid.
        k_frac: (3,) fractional k-coordinates (for the Bloch phase).

    Returns:
        psi_r: (nspinor, nx, ny, nz) complex128. Normalized so that
        integrating |psi|² over the cell gives the stored norm (= 1
        for deterministic bands, √n_eff for pseudobands). We follow
        ``load_wfns`` convention: multiply by √n_rtot after iFFT, and
        apply a Bloch phase exp(2πi k·r) at the end.
    """
    ns = c.shape[0]
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    psi_G = np.zeros((ns, nx, ny, nz), dtype=np.complex128)
    # Scatter coefficients into the G-box (wrap negative indices).
    gx = gvecs[:, 0] % nx
    gy = gvecs[:, 1] % ny
    gz = gvecs[:, 2] % nz
    for s in range(ns):
        psi_G[s, gx, gy, gz] = c[s]
    psi_r = np.fft.ifftn(psi_G, axes=(1, 2, 3)) * n_rtot  # LORRAX's √n convention
    # Bloch phase exp(2πi k · r_frac) at each grid point. r_frac = idx / N.
    ix = np.arange(nx) / nx
    iy = np.arange(ny) / ny
    iz = np.arange(nz) / nz
    phase = np.exp(
        2j * np.pi * (
            k_frac[0] * ix[:, None, None]
            + k_frac[1] * iy[None, :, None]
            + k_frac[2] * iz[None, None, :]
        )
    )
    return psi_r * phase[None, :, :, :]


# ═══════════════════════════════════════════════════════════════════════
# Full-BZ unfolding
# ═══════════════════════════════════════════════════════════════════════


def full_bz_kpoints(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Enumerate the Γ-centered full-BZ k-points in lexicographic order.

    Matches LORRAX's ``qp_wfn_rotations.h5/kpoints_crys`` convention
    (that file's first axis runs over kx, then ky, then kz).

    Returns:
        (nk_tot, 3) array of fractional coordinates in [0, 1).
    """
    nx, ny, nz = kgrid
    out = np.zeros((nx * ny * nz, 3), dtype=np.float64)
    for i, (x, y, z) in enumerate(((x, y, z)
                                   for x in range(nx)
                                   for y in range(ny)
                                   for z in range(nz))):
        out[i] = (x / nx, y / ny, z / nz)
    return out


def match_full_to_irr(k_full: np.ndarray, k_irr: np.ndarray,
                      tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """For each full-BZ k, find the IBZ k it maps to via cubic Oh point
    group + integer umklapp. Returns (k_irr_idx, sym_mat) per full k.

    We only need the IBZ index + sym matrix to fetch ``c_{n,k_full}``
    from the WFN. The sym matrix is implicitly applied via the same
    wavefunction-symmetrisation that ``symmetry_maps`` uses; for this
    cheap test we just look up the matching IBZ k and rotate G-vectors
    by the inverse symmetry, then evaluate at the full-BZ k. This
    reproduces the correct physical ψ_{n,k_full}(r) because
    (S · ψ_{n, k_irr}) is the eigenstate at S · k_irr + G₀.
    """
    from itertools import permutations, product
    signs = list(product([1, -1], repeat=3))
    perms = list(permutations(range(3)))
    sym_mats = []
    for p in perms:
        for s in signs:
            M = np.zeros((3, 3), dtype=np.int32)
            for i, pi in enumerate(p):
                M[i, pi] = s[i]
            sym_mats.append(M)

    nk_full = k_full.shape[0]
    out_irr = np.full(nk_full, -1, dtype=np.int64)
    out_S = np.zeros((nk_full, 3, 3), dtype=np.int32)
    for i, k in enumerate(k_full):
        for j, kp in enumerate(k_irr):
            for S in sym_mats:
                diff = k - S @ kp
                diff_int = np.rint(diff)
                if np.all(np.abs(diff - diff_int) < tol):
                    out_irr[i] = j
                    out_S[i] = S
                    break
            if out_irr[i] >= 0:
                break
        if out_irr[i] < 0:
            raise ValueError(
                f"No IBZ match for k_full[{i}]={k}. "
                "This test currently handles cubic Oh only; extend "
                "match_full_to_irr for lower-symmetry crystals.")
    return out_irr, out_S


def psi_at_full_kpt(wfn: WFNView, n: int,
                    k_full_idx: int,
                    k_full_coords: np.ndarray,
                    irr_lookup: np.ndarray, sym_lookup: np.ndarray,
                    ) -> np.ndarray:
    """ψ_{n, k_full}(r) on the full grid, unfolded from IBZ.

    Uses the fact that, for a crystal symmetry S,

        ψ_{n, S·k_irr + G₀}(r) = ψ_{n, k_irr}(S⁻¹ · (r − τ))

    For our simple cubic Si cell (τ = 0), S⁻¹·r means we rotate real
    space by S⁻¹ — equivalently, rotate G-vectors by S before building
    ψ_G, then inverse-FFT, then apply the Bloch phase at k_full. The G₀
    umklapp just shifts the stored coefficients into the first BZ; the
    physical state is the same.
    """
    k_irr_idx = int(irr_lookup[k_full_idx])
    S = sym_lookup[k_full_idx]  # k_full = S · k_irr + G₀
    c, g = wfn.coeffs_at_k(k_irr_idx, n)
    # Rotate G vectors: new_G = S · old_G (reciprocal space transforms
    # by S for ψ_{S·k}(r) = ψ_{k}(S⁻¹·r)).
    g_rot = (g @ S.T).astype(np.int32)
    return psi_on_grid(c, g_rot, wfn.fft_grid, k_full_coords[k_full_idx])


# ═══════════════════════════════════════════════════════════════════════
# Centroids + ISDF contraction
# ═══════════════════════════════════════════════════════════════════════


def load_centroid_indices(path: Path, fft_grid: tuple[int, int, int]
                          ) -> np.ndarray:
    """Load centroid *integer grid indices* (n_mu, 3) from a fractional file."""
    data = np.loadtxt(path)
    fft = np.asarray(fft_grid, dtype=np.int64)
    idx = (np.round(data * fft).astype(np.int64)) % fft
    return idx


def r_flat_from_indices(idx: np.ndarray, fft_grid: tuple[int, int, int]
                        ) -> np.ndarray:
    """Convert (n_mu, 3) grid indices → (n_mu,) flat row-major indices.

    Matches ``zeta_q``'s second-to-last axis ordering (C/row-major over
    (x, y, z) with x slowest).
    """
    nx, ny, nz = fft_grid
    return (idx[:, 0] * ny * nz + idx[:, 1] * nz + idx[:, 2]).astype(np.int64)


def q_index_from_k_diff(k_idx_out: int, k_idx_in: int,
                        k_full: np.ndarray,
                        kgrid: tuple[int, int, int]) -> tuple[int, int, int]:
    """Given k and k' = k − q as full-BZ indices, return (qx, qy, qz) for ζ."""
    q_frac = (k_full[k_idx_out] - k_full[k_idx_in]) % 1.0
    nx, ny, nz = kgrid
    qxi = int(np.round(q_frac[0] * nx)) % nx
    qyi = int(np.round(q_frac[1] * ny)) % ny
    qzi = int(np.round(q_frac[2] * nz)) % nz
    return qxi, qyi, qzi


# ═══════════════════════════════════════════════════════════════════════
# The actual validation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Sample:
    """One (n, m, k, q) evaluation result."""
    n: int
    m: int
    k_full_idx: int
    k_full_minus_q_idx: int
    q_idx: tuple[int, int, int]
    band_class: str                 # 'vv', 'vc', 'cv', 'cc'
    abs_err_L2: float
    rel_err_L2: float
    max_abs_err: float
    P_true_L2: float
    P_ISDF_L2: float


def classify_bands(n: int, m: int, n_val: int) -> str:
    """Return 'vv', 'vc', 'cv', 'cc' based on whether each band index is
    in [0, n_val) (valence) or [n_val, ...) (conduction)."""
    a = 'v' if n < n_val else 'c'
    b = 'v' if m < n_val else 'c'
    return a + b


def draw_samples(rng: np.random.Generator,
                 n_samples: int,
                 nb_total: int,
                 nk_tot: int,
                 n_val: int,
                 band_pool: str = 'mixed') -> Iterator[tuple[int, int, int, int]]:
    """Yield (n, m, k_idx, kp_idx) tuples; q = k − k' is implicit.

    band_pool choices:
        'mixed'      — both n and m drawn uniformly over [0, nb_total)
                       (naturally heavily cc-weighted when n_val ≪ nb_total).
        'val_only'   — both n and m in [0, n_val) (all vv).
        'cond_only'  — both n and m in [n_val, nb_total) (all cc).
        'cross'      — n in [0, n_val), m in [n_val, nb_total) (all vc).
                       Use to force coverage of the valence×conduction
                       quadrant, which is the physically-relevant channel
                       for Σ via ψ*_c ψ_v pair products.
    """
    def _draw_val() -> int:
        return int(rng.integers(0, n_val))

    def _draw_cond() -> int:
        return int(rng.integers(n_val, nb_total))

    def _draw_mixed() -> int:
        return int(rng.integers(0, nb_total))

    draw_n, draw_m = _draw_mixed, _draw_mixed
    if band_pool == 'val_only':
        draw_n = draw_m = _draw_val
    elif band_pool == 'cond_only':
        draw_n = draw_m = _draw_cond
    elif band_pool == 'cross':
        draw_n, draw_m = _draw_val, _draw_cond
    elif band_pool != 'mixed':
        raise ValueError(f"Unknown band_pool: {band_pool!r}")

    for _ in range(n_samples):
        n = draw_n()
        m = draw_m()
        k = int(rng.integers(0, nk_tot))
        kp = int(rng.integers(0, nk_tot))
        yield n, m, k, kp


def reconstruct_one(wfn: WFNView,
                    n: int, m: int,
                    k_full_idx: int, kp_full_idx: int,
                    k_full_coords: np.ndarray,
                    irr_lookup: np.ndarray, sym_lookup: np.ndarray,
                    centroid_r_flat: np.ndarray,
                    zeta_qz: h5py.Dataset,
                    kgrid: tuple[int, int, int]) -> Sample:
    """Compute true + ISDF pair product for one sample and return the diagnostic."""
    # 1. ψ_{n, k} and ψ_{m, k'}=ψ_{m, k−q} on the full FFT grid, flattened over spin.
    psi_n = psi_at_full_kpt(wfn, n, k_full_idx, k_full_coords,
                            irr_lookup, sym_lookup)       # (ns, nx, ny, nz)
    psi_m = psi_at_full_kpt(wfn, m, kp_full_idx, k_full_coords,
                            irr_lookup, sym_lookup)
    norm_n = max(wfn.band_norms[n], 1.0)
    norm_m = max(wfn.band_norms[m], 1.0)
    psi_n_scaled = psi_n / norm_n
    psi_m_scaled = psi_m / norm_m

    # 2. Pair product, spin-traced (matches isdf_pair_mode='spin_traced').
    # P(r) = Σ_s ψ*_m,s(r) · ψ_n,s(r)
    P_true = np.sum(np.conj(psi_m_scaled) * psi_n_scaled, axis=0)  # (nx,ny,nz)
    P_true_flat = P_true.reshape(-1)

    # 3. Pair product at centroids (same spin trace).
    psi_n_flat = psi_n_scaled.reshape(psi_n_scaled.shape[0], -1)   # (ns, n_rtot)
    psi_m_flat = psi_m_scaled.reshape(psi_m_scaled.shape[0], -1)
    psi_n_mu = psi_n_flat[:, centroid_r_flat]                      # (ns, n_mu)
    psi_m_mu = psi_m_flat[:, centroid_r_flat]
    P_cent = np.sum(np.conj(psi_m_mu) * psi_n_mu, axis=0)          # (n_mu,)

    # 4. Contract with ζ_q at the matching (qx, qy, qz).
    qxi, qyi, qzi = q_index_from_k_diff(k_full_idx, kp_full_idx,
                                         k_full_coords, kgrid)
    zeta_slice = np.asarray(zeta_qz[qxi, qyi, qzi, :, :])          # (n_mu, n_rtot)
    P_ISDF_flat = P_cent @ zeta_slice                              # (n_rtot,)

    # 5. Metrics.
    diff = P_ISDF_flat - P_true_flat
    rel_err_L2 = float(np.linalg.norm(diff) / max(np.linalg.norm(P_true_flat),
                                                   1e-30))
    return Sample(
        n=int(n), m=int(m),
        k_full_idx=int(k_full_idx),
        k_full_minus_q_idx=int(kp_full_idx),
        q_idx=(int(qxi), int(qyi), int(qzi)),
        band_class=classify_bands(n, m, wfn.band_norms.size),   # overridden below
        abs_err_L2=float(np.linalg.norm(diff)),
        rel_err_L2=rel_err_L2,
        max_abs_err=float(np.max(np.abs(diff))),
        P_true_L2=float(np.linalg.norm(P_true_flat)),
        P_ISDF_L2=float(np.linalg.norm(P_ISDF_flat)),
    )


def sample_pair_products(run: RunInputs,
                         n_samples: int = 100,
                         seed: int = 42,
                         band_pool: str = 'mixed',
                         verbose: bool = True,
                         device: str = 'cpu') -> list[Sample]:
    """Draw ``n_samples`` random (n, m, k, q) tuples and validate each.

    Loads the full ζ_q tensor into memory once (saves ~0.5-2 s of HDF5
    slicing per sample) and optionally pushes it onto a JAX device so
    the per-sample contraction runs on GPU.

    Args:
        run: :class:`RunInputs` bundle (from :func:`load_run`).
        n_samples: number of random (n, m, k, q) samples to draw.
        seed: RNG seed.
        band_pool: 'mixed' / 'val_only' / 'cond_only' / 'cross'
            (see :func:`draw_samples`).
        verbose: print progress every 20 samples.
        device: 'cpu' (pure numpy) or 'gpu' (JAX). For GPU the ζ array
            must fit in device memory — for Si 2×2×2 with N_μ ≤ 3200
            this is ≤ 20 GB, fine on a 40 GB A100. At N_μ = 6000 it
            hits 36 GB; loads but leaves almost no headroom, so JAX
            fallback to CPU if the put fails is applied automatically.

    Returns:
        list of :class:`Sample`, one per drawn tuple.
    """
    wfn = WFNView(run.wfn_path)
    k_full = full_bz_kpoints(run.kgrid)
    irr_lookup, sym_lookup = match_full_to_irr(k_full, wfn.k_irr_frac)
    cent_idx = load_centroid_indices(run.centroids_path, run.fft_grid)
    cent_r_flat = r_flat_from_indices(cent_idx, run.fft_grid)

    # Load the entire ζ_q tensor into memory once. On a 2×2×2 k-grid
    # and N_μ = 6000, this is ~36 GB in host RAM — fine for a compute
    # node but a lot for a login node. We print shape + size so the
    # caller isn't surprised.
    import time as _time
    t0 = _time.time()
    with h5py.File(run.zeta_path, 'r') as zf:
        zeta_all_np = np.asarray(zf['zeta_q'][...])
    mem_gb = zeta_all_np.nbytes / 1e9
    if verbose:
        print(f"[validate_isdf] loaded ζ: shape={zeta_all_np.shape} "
              f"dtype={zeta_all_np.dtype} size={mem_gb:.1f} GB "
              f"in {_time.time() - t0:.1f}s")

    zeta_dev = None
    if device == 'gpu':
        try:
            import jax
            import jax.numpy as jnp
            zeta_dev = jnp.asarray(zeta_all_np)
            if verbose:
                dv = jax.devices()[0]
                print(f"[validate_isdf] zeta pushed to {dv.platform}:{dv.id}")
        except Exception as e:
            print(f"[validate_isdf] GPU put failed ({e!s}); falling back to CPU")
            zeta_dev = None

    rng = np.random.default_rng(seed)
    samples: list[Sample] = []
    for i, (n, m, k, kp) in enumerate(draw_samples(
            rng, n_samples, run.nband, k_full.shape[0],
            run.n_val, band_pool=band_pool)):
        s = _reconstruct_one_inmem(
            wfn, n, m, k, kp, k_full, irr_lookup, sym_lookup,
            cent_r_flat, zeta_all_np, zeta_dev, run.kgrid)
        s.band_class = classify_bands(n, m, run.n_val)
        samples.append(s)
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_samples}] n={n} m={m} "
                  f"class={s.band_class} rel_err={s.rel_err_L2:.3e}")
    wfn.close()
    return samples


def _reconstruct_one_inmem(wfn, n, m, k_full_idx, kp_full_idx,
                           k_full_coords, irr_lookup, sym_lookup,
                           centroid_r_flat, zeta_all_np, zeta_dev,
                           kgrid):
    """In-memory-ζ version of :func:`reconstruct_one`.

    If ``zeta_dev`` is not None (JAX device array) uses JAX for the
    contraction step; otherwise falls back to numpy.
    """
    psi_n = psi_at_full_kpt(wfn, n, k_full_idx, k_full_coords,
                            irr_lookup, sym_lookup)
    psi_m = psi_at_full_kpt(wfn, m, kp_full_idx, k_full_coords,
                            irr_lookup, sym_lookup)
    norm_n = max(wfn.band_norms[n], 1.0)
    norm_m = max(wfn.band_norms[m], 1.0)
    psi_n_scaled = psi_n / norm_n
    psi_m_scaled = psi_m / norm_m
    P_true = np.sum(np.conj(psi_m_scaled) * psi_n_scaled, axis=0)
    P_true_flat = P_true.reshape(-1)
    psi_n_flat = psi_n_scaled.reshape(psi_n_scaled.shape[0], -1)
    psi_m_flat = psi_m_scaled.reshape(psi_m_scaled.shape[0], -1)
    psi_n_mu = psi_n_flat[:, centroid_r_flat]
    psi_m_mu = psi_m_flat[:, centroid_r_flat]
    P_cent = np.sum(np.conj(psi_m_mu) * psi_n_mu, axis=0)
    qxi, qyi, qzi = q_index_from_k_diff(k_full_idx, kp_full_idx,
                                         k_full_coords, kgrid)

    if zeta_dev is not None:
        import jax
        import jax.numpy as jnp
        P_cent_j = jnp.asarray(P_cent)
        zeta_slice = zeta_dev[qxi, qyi, qzi, :, :]
        P_ISDF_flat = np.asarray(jnp.matmul(P_cent_j, zeta_slice))
    else:
        P_ISDF_flat = P_cent @ zeta_all_np[qxi, qyi, qzi, :, :]

    diff = P_ISDF_flat - P_true_flat
    rel_err_L2 = float(np.linalg.norm(diff) /
                       max(np.linalg.norm(P_true_flat), 1e-30))
    return Sample(
        n=int(n), m=int(m),
        k_full_idx=int(k_full_idx),
        k_full_minus_q_idx=int(kp_full_idx),
        q_idx=(int(qxi), int(qyi), int(qzi)),
        band_class='',  # set by caller
        abs_err_L2=float(np.linalg.norm(diff)),
        rel_err_L2=rel_err_L2,
        max_abs_err=float(np.max(np.abs(diff))),
        P_true_L2=float(np.linalg.norm(P_true_flat)),
        P_ISDF_L2=float(np.linalg.norm(P_ISDF_flat)),
    )


# ═══════════════════════════════════════════════════════════════════════
# Aggregation + reporting
# ═══════════════════════════════════════════════════════════════════════


def aggregate(samples: list[Sample]) -> dict:
    """Aggregate statistics per band class and overall."""
    from collections import defaultdict
    by_class = defaultdict(list)
    for s in samples:
        by_class[s.band_class].append(s.rel_err_L2)
    overall = [s.rel_err_L2 for s in samples]
    return {
        'overall': {
            'mean': float(np.mean(overall)),
            'median': float(np.median(overall)),
            'max': float(np.max(overall)),
            'p90': float(np.percentile(overall, 90)),
            'count': len(overall),
        },
        'by_class': {
            cls: {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'max': float(np.max(vals)),
                'count': len(vals),
            } for cls, vals in by_class.items()
        },
    }


def write_outputs(samples: list[Sample], agg: dict, out_dir: Path):
    """Dump per-sample JSON, aggregate JSON, Markdown summary, 3 plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / 'samples.json').write_text(
        json.dumps([{**s.__dict__, 'q_idx': list(s.q_idx)}
                    for s in samples], indent=2))
    (out_dir / 'aggregate.json').write_text(json.dumps(agg, indent=2))

    lines = [
        '# ISDF pair-product reconstruction validation',
        '',
        f"Samples: {agg['overall']['count']}",
        f"Median rel L2: {agg['overall']['median']:.3e}",
        f"Mean   rel L2: {agg['overall']['mean']:.3e}",
        f"P90    rel L2: {agg['overall']['p90']:.3e}",
        f"Max    rel L2: {agg['overall']['max']:.3e}",
        '',
        '## Per-band class',
        '',
        "| class | count | median | mean | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for cls, st in agg['by_class'].items():
        lines.append(f"| {cls} | {st['count']} | {st['median']:.3e} | "
                     f"{st['mean']:.3e} | {st['max']:.3e} |")
    (out_dir / 'summary.md').write_text('\n'.join(lines) + '\n')

    # Plots
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Histogram of log10(rel_err).
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = np.log10([s.rel_err_L2 for s in samples])
    ax.hist(vals, bins=30, color='tab:blue', alpha=0.8, edgecolor='k')
    ax.set_xlabel(r'$\log_{10}\,\varepsilon_{\mathrm{rel}}$')
    ax.set_ylabel('count')
    ax.set_title(f'ISDF reconstruction rel. L2 error distribution '
                 f'({len(samples)} samples)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'error_histogram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Box plot per class.
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = sorted(agg['by_class'].keys())
    data = [[s.rel_err_L2 for s in samples if s.band_class == c]
            for c in classes]
    ax.boxplot(data, tick_labels=classes, vert=True)
    ax.set_yscale('log')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{rel}}$')
    ax.set_title('ISDF reconstruction rel. L2 error by band class (v=valence, c=conduction)')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'error_by_class.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Scatter: true_L2 vs rel_err (any amplitude dependence?).
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [s.P_true_L2 for s in samples]
    ys = [s.rel_err_L2 for s in samples]
    cols = {'vv': 'tab:blue', 'vc': 'tab:orange',
            'cv': 'tab:green', 'cc': 'tab:red'}
    for c in classes:
        mask = [s.band_class == c for s in samples]
        ax.scatter([x for x, m in zip(xs, mask) if m],
                   [y for y, m in zip(ys, mask) if m],
                   c=cols.get(c, 'k'), s=18, alpha=0.7, label=c)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\|P_{\mathrm{true}}\|_2$')
    ax.set_ylabel(r'$\varepsilon_{\mathrm{rel}}$')
    ax.set_title('Relative error vs true-pair amplitude')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out_dir / 'error_vs_amplitude.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[validate_isdf] wrote {out_dir}/samples.json, aggregate.json, "
          f"summary.md, and 3 .png plots.")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True, type=Path)
    ap.add_argument('--zeta-h5', default=None, type=Path)
    ap.add_argument('--wfn', default=None, type=Path)
    ap.add_argument('--centroids', default=None, type=Path)
    ap.add_argument('--n-samples', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--band-pool',
                    choices=['val_only', 'cond_only', 'mixed', 'cross'],
                    default='mixed',
                    help='"cross" = n in valence, m in conduction — forces '
                         'coverage of the vc quadrant that is usually thin '
                         'under uniform "mixed" draws at high n_cond.')
    ap.add_argument('--out', type=Path, default=Path('./isdf_validation'))
    ap.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                    help='"gpu" uses JAX for the ζ·P_cent contraction '
                         '(only meaningful under lxrun or with jax[cuda] '
                         'installed natively).')
    args = ap.parse_args()

    run = load_run(args.run_dir, wfn=args.wfn,
                   zeta_h5=args.zeta_h5, centroids=args.centroids)
    print(f"[validate_isdf] run_dir={args.run_dir}")
    print(f"  WFN: {run.wfn_path}")
    print(f"  ζ:   {run.zeta_path}")
    print(f"  centroids: {run.centroids_path}")
    print(f"  n_val={run.n_val} n_cond={run.n_cond} nband={run.nband}  "
          f"kgrid={run.kgrid}  fft={run.fft_grid}")

    samples = sample_pair_products(run, args.n_samples, args.seed,
                                   band_pool=args.band_pool,
                                   device=args.device)
    agg = aggregate(samples)
    write_outputs(samples, agg, args.out)

    print("\n=== summary ===")
    print(f"  median rel L2: {agg['overall']['median']:.3e}")
    print(f"  mean   rel L2: {agg['overall']['mean']:.3e}")
    print(f"  p90    rel L2: {agg['overall']['p90']:.3e}")
    print(f"  max    rel L2: {agg['overall']['max']:.3e}")
    for cls in sorted(agg['by_class']):
        st = agg['by_class'][cls]
        print(f"  class {cls}: n={st['count']} median={st['median']:.3e} "
              f"max={st['max']:.3e}")


if __name__ == '__main__':
    main()
