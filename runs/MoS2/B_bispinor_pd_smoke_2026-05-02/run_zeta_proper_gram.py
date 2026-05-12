"""Proper-Gram channel-aware ζ-fit (agent B, 2026-05-02).

Uses the literal Gram K_q from the ISDF normal equations:

  K_q(μ, λ) = Σ_{n_l, n_r, k_l} ρ*_{n_l, n_r, k_l, q}(μ) · ρ_{n_l, n_r, k_l, q}(λ)
  Z_q(μ, r) = Σ_{n_l, n_r, k_l} ρ*_{n_l, n_r, k_l, q}(μ) · ρ_{n_l, n_r, k_l, q}(r)

with ρ(r; n_l, n_r, k_l, q) = Σ_{ab} ψ_l*(r; n_l, k_l, a) γ̃_{ab} ψ_r(r; n_r, k_l+q, b)
a single complex scalar per band-pair-and-position.

K_q is a Gram in the (n_l, n_r, k_l) index → PSD by construction, for any
γ̃.  Cholesky + small ridge works for all four channels — no eigh-pinv,
no truncation.  Cost: explicit O(N_l · N_r · N_k · n_rμ²) per q in the
Gram build (vs the Schur-product CCT's O(N_k · n_rμ²)).  For the MoS2
3×3 smoke that's 9 × 9 × 8 × 16 × 640² ≈ 4 GFLOPs total — milliseconds.

Per-channel centroid sets are unchanged from run_zeta_channel_aware.py:
  μ_L = 0   → scalar centroids (charge density)
  μ_L = i   → current centroids (Gordon-decomposed Pauli current)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")

from runtime import set_default_env
set_default_env()

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
from jax.sharding import Mesh

from common import Meta, symmetry_maps
from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
from common.load_wfns import load_centroids_band_chunked
from centroid.centroid_io import read_centroids
from file_io import WFNReader


LORRAX_RUN = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex"
)
WFN_PATH = LORRAX_RUN / "WFN.h5"
SCALAR_CENTROIDS = LORRAX_RUN / "centroids_frac_640.txt"
CURRENT_CENTROIDS_GLOB = "centroids_frac_*_current.txt"

BAND_RANGE  = (0, 24)
LEFT_RANGE  = (0, 8)
RIGHT_RANGE = (8, 24)
N_TEST_POINTS = 320


def _print(s):
    print(s, flush=True)


def _make_mesh():
    devs = jax.devices()
    return Mesh(np.asarray(devs[:1]).reshape(1, 1), axis_names=("x", "y"))


def _load_centroids(path: Path, fft_grid,
                     expected_density: str | None = None) -> jnp.ndarray:
    """Read a centroid file, snap to FFT grid, dedupe, and (optionally)
    verify the density label in the header matches what we expect.

    ``expected_density`` of None disables the check (legacy files).
    """
    cf = read_centroids(path)
    if expected_density is not None and cf.density != expected_density:
        raise ValueError(
            f"Centroid file {path.name} has density={cf.density!r}, "
            f"expected {expected_density!r}.  Regenerate with "
            f"`centroid.kmeans_cli --density-mode {expected_density}`."
        )
    grid = np.array(fft_grid)
    idx = np.rint(cf.coords * grid).astype(np.int64) % grid
    seen, uniq = set(), []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key); uniq.append(row)
    return jnp.asarray(np.asarray(uniq, dtype=np.int64))


def _sample_test_indices(fft_grid, exclude, n_test, seed=0):
    nx, ny, nz = fft_grid
    rng = np.random.default_rng(seed)
    keys = set()
    for arr in exclude:
        for row in np.asarray(arr):
            keys.add(tuple(int(c) for c in row))
    picks = []
    while len(picks) < n_test:
        c = rng.integers((0, 0, 0), (nx, ny, nz))
        k = tuple(int(x) for x in c)
        if k in keys:
            continue
        keys.add(k); picks.append(c)
    return jnp.asarray(np.asarray(picks, dtype=np.int64))


def _load_psi(*, indices, mesh):
    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    nb = BAND_RANGE[1] - BAND_RANGE[0]
    meta = Meta.from_system(
        wfn, sym, nval=BAND_RANGE[1], ncond=0, nband=nb,
        n_rmu=int(indices.shape[0]), bispinor=True,
    )
    psi_rmu, _ = load_centroids_band_chunked(
        wfn, sym, meta, indices,
        bispinor=True, mesh_xy=mesh,
        band_range=BAND_RANGE, band_chunk_size=min(64, nb),
    )
    return psi_rmu, meta


def _shift_per_q(psi_kgrid, kgrid):
    """Build psi_shifted[q, k, ...] = psi[(k + q) mod kgrid, ...].

    psi_kgrid: (Nx, Ny, Nz, *rest); kgrid = (Nx, Ny, Nz)
    Returns: (n_q, n_k, *rest) flat-q × flat-k.
    """
    Nx, Ny, Nz = kgrid
    n_q = Nx * Ny * Nz
    n_k = n_q
    out = []
    for q_idx in range(n_q):
        qx, qy, qz = q_idx // (Ny * Nz), (q_idx // Nz) % Ny, q_idx % Nz
        # ψ_shifted[k] = ψ[k+q]: shift array in negative direction
        rolled = jnp.roll(psi_kgrid, shift=(-qx, -qy, -qz), axis=(0, 1, 2))
        out.append(rolled.reshape(n_k, *psi_kgrid.shape[3:]))
    return jnp.stack(out, axis=0)


def _build_proper_K_Z(psi_l_train, psi_r_train_shifted,
                      psi_l_test, psi_r_test_shifted, vertex):
    """
    psi_l_train:  (n_k, n_l, ns, n_train)
    psi_r_train_shifted: (n_q, n_k, n_r, ns, n_train)   ← shifted per q
    psi_l_test:   (n_k, n_l, ns, n_test)
    psi_r_test_shifted:  (n_q, n_k, n_r, ns, n_test)
    vertex:       (ns, ns)

    Returns
    -------
    K_q: (n_q, n_train, n_train) — Gram on training centroids
    Z_q: (n_q, n_train, n_test)  — RHS at test points
    """
    # ρ_train(q, k, n_l, n_r, μ) = Σ_{ab} conj(ψ_l[k, n_l, a, μ]) γ̃_{ab} ψ_r[q, k, n_r, b, μ]
    rho_tr = jnp.einsum('klaμ,ab,qkrbμ->qklrμ',
                        jnp.conj(psi_l_train), vertex, psi_r_train_shifted,
                        optimize=True)
    rho_te = jnp.einsum('klaν,ab,qkrbν->qklrν',
                        jnp.conj(psi_l_test), vertex, psi_r_test_shifted,
                        optimize=True)
    # K(μ, λ) = Σ_{k, n_l, n_r} ρ_tr*(μ) · ρ_tr(λ)
    K = jnp.einsum('qklrμ,qklrλ->qμλ', jnp.conj(rho_tr), rho_tr, optimize=True)
    Z = jnp.einsum('qklrμ,qklrν->qμν', jnp.conj(rho_tr), rho_te, optimize=True)
    return K, Z


def _build_rho_band_pair(psi_l, psi_r_q, vertex):
    """Return ρ(q, k, n_l, n_r, μ) = Σ_{ab} conj(ψ_l[k,n_l,a,μ]) γ̃_{ab} ψ_r[q,k,n_r,b,μ]."""
    return jnp.einsum('klaμ,ab,qkrbμ->qklrμ',
                      jnp.conj(psi_l), vertex, psi_r_q, optimize=True)


def _aggregate_recon(rho_centroid, rho_test, zeta_q):
    """Sum-squared reconstruction error over (q, k, n_l, n_r, r_test).

    Returns (rel², total_err², total_ref²).
    rho_centroid: (q, k, n_l, n_r, n_centroid)  — exact ρ at centroids
    rho_test:     (q, k, n_l, n_r, n_test)       — exact ρ at test points
    zeta_q:       (q, n_centroid, n_test)        — solved interpolation weights
    """
    # ρ_recon(q, k, n_l, n_r, t) = Σ_μ ρ_centroid(q, k, n_l, n_r, μ) · ζ(q, μ, t)
    rho_recon = jnp.einsum('qklrμ,qμt->qklrt',
                            rho_centroid, zeta_q, optimize=True)
    diff = rho_recon - rho_test
    err2 = float(jnp.sum(jnp.abs(diff) ** 2))
    ref2 = float(jnp.sum(jnp.abs(rho_test) ** 2))
    rel2 = err2 / max(ref2, 1e-300)
    return rel2, err2, ref2


def _solve_zeta_cholesky(K_q, Z_q):
    """Cholesky+ridge solve, no fallback.  K_q is PSD by construction.
    Returns (zeta_q, residual)."""
    nq, n_rmu, _ = K_q.shape
    trace = jnp.trace(K_q, axis1=-2, axis2=-1)
    ridge = 1e-14 * jnp.abs(trace)[:, None, None] * jnp.eye(n_rmu)[None, :, :]
    L = jnp.linalg.cholesky(K_q + ridge)
    finite = bool(jnp.all(jnp.isfinite(L)))
    if not finite:
        # If this fires for a "PSD-by-construction" matrix, something's wrong upstream.
        raise RuntimeError("Cholesky returned NaN — K_q is not PSD?  Bug somewhere.")
    y    = jax.vmap(lambda Li, Zi: jsla.solve_triangular(Li, Zi, lower=True))(L, Z_q)
    zeta = jax.vmap(lambda Li, yi: jsla.solve_triangular(Li.conj().T, yi, lower=False))(L, y)
    resid = jnp.einsum('qij,qjk->qik', K_q, zeta) - Z_q
    return zeta, resid


def main():
    _print(f"jax devices: {jax.devices()}")
    mesh = _make_mesh()

    cur_files = list(LORRAX_RUN.glob(CURRENT_CENTROIDS_GLOB))
    if not cur_files:
        raise FileNotFoundError("No current-density centroid file. "
                                "Run kmeans_cli --density-mode current first.")
    cur_path = cur_files[0]
    _print(f"current centroid file: {cur_path.name}")

    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    kgrid = tuple(int(x) for x in wfn.kgrid)
    n_q = int(np.prod(kgrid))
    _print(f"  fft_grid = {fft_grid}, kgrid = {kgrid}, n_q = {n_q}")

    # The legacy scalar centroids file (centroids_frac_640.txt) predates
    # the density-header convention so we don't verify it; the current
    # file does have the new header so we cross-check.
    cents_S = _load_centroids(SCALAR_CENTROIDS, fft_grid,
                               expected_density=None)
    cents_C = _load_centroids(cur_path,         fft_grid,
                               expected_density="current")
    _print(f"  scalar centroids:  {int(cents_S.shape[0])}")
    _print(f"  current centroids: {int(cents_C.shape[0])}")

    test_idx = _sample_test_indices(fft_grid, [cents_S, cents_C],
                                     N_TEST_POINTS, seed=42)
    _print(f"  test points: {int(test_idx.shape[0])}")

    _print("\n=== loading ψ at three index sets ===")
    t = time.perf_counter()
    psi_S, _ = _load_psi(indices=cents_S,  mesh=mesh)  # (k, nb, 4, n_S)
    _print(f"  scalar:  {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    psi_C, _ = _load_psi(indices=cents_C,  mesh=mesh)
    _print(f"  current: {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    psi_T, _ = _load_psi(indices=test_idx, mesh=mesh)
    _print(f"  test:    {time.perf_counter()-t:.2f}s")

    # Slice band ranges
    def _slice(psi):
        return (psi[:, LEFT_RANGE[0]:LEFT_RANGE[1], :, :],   # (k, n_l, ns, μ)
                psi[:, RIGHT_RANGE[0]:RIGHT_RANGE[1], :, :])  # (k, n_r, ns, μ)

    # Pre-build shifted-by-q ψ_r tensors (small data — shift cost is trivial)
    def _shift(psi_r_flat_k):
        # psi_r_flat_k: (n_k, n_r, ns, n_rmu) — flat k → reshape to 3D for roll
        n_k = psi_r_flat_k.shape[0]
        rest = psi_r_flat_k.shape[1:]
        psi_r_kgrid = psi_r_flat_k.reshape(*kgrid, *rest)
        return _shift_per_q(psi_r_kgrid, kgrid)   # (n_q, n_k, n_r, ns, n_rmu)

    _print("\n=== building shifted ψ_r tensors ===")
    psi_S_l, psi_S_r = _slice(psi_S)
    psi_C_l, psi_C_r = _slice(psi_C)
    psi_T_l, psi_T_r = _slice(psi_T)
    t = time.perf_counter()
    psi_S_r_q = _shift(psi_S_r)
    psi_C_r_q = _shift(psi_C_r)
    psi_T_r_q = _shift(psi_T_r)
    _print(f"  shifted-tensor build: {time.perf_counter()-t:.2f}s")

    gammas = [gamma0, gamma1, gamma2, gamma3]

    _print("\n=== per-channel proper-Gram fit — A/B comparison ===")
    _print("  (a) channel-aware: scalar centroids for μ_L=0, current for μ_L=i")
    _print("  (b) scalar-only:   scalar centroids for ALL four channels\n")
    _print(f"  Aggregate reconstruction error over EVERY band pair in "
           f"({LEFT_RANGE[1]-LEFT_RANGE[0]} × {RIGHT_RANGE[1]-RIGHT_RANGE[0]} = "
           f"{(LEFT_RANGE[1]-LEFT_RANGE[0])*(RIGHT_RANGE[1]-RIGHT_RANGE[0])}), "
           f"every k ({int(np.prod(kgrid))}), every q ({int(np.prod(kgrid))}), "
           f"every test point ({N_TEST_POINTS}).")
    _print(f"  Total band-pair-q-r samples per channel: ~"
           f"{(LEFT_RANGE[1]-LEFT_RANGE[0])*(RIGHT_RANGE[1]-RIGHT_RANGE[0])*int(np.prod(kgrid))**2*N_TEST_POINTS:,d}\n")

    zetas = {}
    aggregate_results = {}
    for label, mode in (("channel-aware", "channel"),
                         ("scalar-only  ", "scalar")):
        _print(f"  ({mode}):")
        agg_per_channel = {}
        for mu_L in (0, 1, 2, 3):
            if mode == "channel" and mu_L != 0:
                psi_l_train, psi_r_train_q = psi_C_l, psi_C_r_q
                cset = "current"
            else:
                psi_l_train, psi_r_train_q = psi_S_l, psi_S_r_q
                cset = "scalar"

            vertex = gammas[mu_L].astype(jnp.complex128)

            # Build K, Z, and the explicit ρ tensors for aggregate-error scoring
            rho_train = _build_rho_band_pair(psi_l_train, psi_r_train_q, vertex)
            rho_test  = _build_rho_band_pair(psi_T_l,    psi_T_r_q,     vertex)
            K_q = jnp.einsum('qklrμ,qklrλ->qμλ',
                             jnp.conj(rho_train), rho_train, optimize=True)
            Z_q = jnp.einsum('qklrμ,qklrt->qμt',
                             jnp.conj(rho_train), rho_test,  optimize=True)
            evs = jnp.linalg.eigvalsh(
                0.5 * (K_q + jnp.conj(jnp.swapaxes(K_q, -1, -2)))
            )
            ev_min, ev_max = float(jnp.min(evs)), float(jnp.max(evs))

            zeta_q, lse_resid = _solve_zeta_cholesky(K_q, Z_q)
            rel2, err2, ref2 = _aggregate_recon(rho_train, rho_test, zeta_q)

            n_train = K_q.shape[1]
            _print(
                f"    μ_L={mu_L} ({cset}, {n_train} cents): "
                f"eig∈[{ev_min:.3e}, {ev_max:.3e}]  "
                f"|LSE_res|≤{float(jnp.max(jnp.abs(lse_resid))):.3e}  "
                f"|ζ|≤{float(jnp.max(jnp.abs(zeta_q))):.3e}\n"
                f"             aggregate ‖ρ_recon − ρ‖² / ‖ρ‖² = {rel2:.3e}  "
                f"(rel = {rel2 ** 0.5:.3e})"
            )
            if mode == "channel":
                zetas[mu_L] = zeta_q
            agg_per_channel[mu_L] = rel2
        aggregate_results[mode] = agg_per_channel
        _print("")

    _print("  ── aggregate-rel summary (sqrt of summed squared error / summed |ρ|²) ──")
    _print(f"  {'μ_L':<5}{'(a) channel-aware':>22}{'(b) scalar-only':>22}{'ratio (a)/(b)':>18}")
    for mu_L in (0, 1, 2, 3):
        a = aggregate_results['channel'][mu_L] ** 0.5
        b = aggregate_results['scalar'][mu_L] ** 0.5
        ratio = a / b if b > 0 else float('nan')
        _print(f"  {mu_L:<5}{a:>22.3e}{b:>22.3e}{ratio:>18.3f}")

    _print("\nDone.")


if __name__ == "__main__":
    main()
