"""Channel-aware ζ-fit smoke (agent B, 2026-05-02).

Same flow as run_zeta_fit.py but with **per-channel centroid sets**:
  μ_L = 0 → centroids picked from scalar charge density ρ(r)
  μ_L = 1, 2, 3 → centroids picked from band-resolved current density
                   Σ_{n,k,i} (ψ_n,k^† α^i ψ_n,k(r))²

Produces both centroid files via centroid.kmeans_cli (run separately
beforehand):

  cd .../00_lorrax_cohsex
  lxrun python3 -m centroid.kmeans_cli 640                                # scalar  → centroids_frac_640.txt
  lxrun python3 -m centroid.kmeans_cli --density-mode current --out-suffix _current 640
                                                                         # current → centroids_frac_<N>_current.txt
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
from common.isdf_fitting import (
    compute_pair_density_lorentz,
    compute_CCT_from_left_right,
)
from common.load_wfns import load_centroids_band_chunked
from file_io import WFNReader


LORRAX_RUN = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex"
)
WFN_PATH = LORRAX_RUN / "WFN.h5"
SCALAR_CENTROIDS = LORRAX_RUN / "centroids_frac_640.txt"
# Glob for the current-density file — the orbit-closure inflates the count
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


def _load_centroids(path: Path, meta: Meta) -> jnp.ndarray:
    coords = np.loadtxt(path, dtype=np.float64)
    grid = np.array(meta.fft_grid)
    idx = np.rint(coords * grid).astype(np.int64) % grid
    seen, uniq = set(), []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key); uniq.append(row)
    return jnp.asarray(np.asarray(uniq, dtype=np.int64))


def _sample_test_indices(meta: Meta, exclude: list[jnp.ndarray],
                          n_test: int, seed: int = 0) -> jnp.ndarray:
    nx, ny, nz = meta.fft_grid
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


def _load_psi(*, indices: jnp.ndarray, mesh: Mesh):
    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    nb = BAND_RANGE[1] - BAND_RANGE[0]
    meta = Meta.from_system(
        wfn, sym, nval=BAND_RANGE[1], ncond=0, nband=nb,
        n_rmu=int(indices.shape[0]), bispinor=True,
    )
    psi_rmu, psi_rmuT = load_centroids_band_chunked(
        wfn, sym, meta, indices,
        bispinor=True, mesh_xy=mesh,
        band_range=BAND_RANGE, band_chunk_size=min(64, nb),
    )
    return psi_rmu, psi_rmuT, meta


def _slice(psi_rmu, psi_rmuT, lo, hi):
    return psi_rmu[:, lo:hi, :, :], psi_rmuT[:, :, lo:hi, :]


def _build_cct_zct(psi_c_rmu, psi_c_rmuT, psi_t_rmu, mu_L, kgrid, mesh):
    pl_c, pl_cT = _slice(psi_c_rmu, psi_c_rmuT, *LEFT_RANGE)
    pr_c, pr_cT = _slice(psi_c_rmu, psi_c_rmuT, *RIGHT_RANGE)
    pl_t, _     = _slice(psi_t_rmu, psi_c_rmuT, *LEFT_RANGE)
    pr_t, _     = _slice(psi_t_rmu, psi_c_rmuT, *RIGHT_RANGE)
    P_l_train = compute_pair_density_lorentz(pl_cT, pl_c, mu_L, mesh)
    P_r_train = compute_pair_density_lorentz(pr_cT, pr_c, mu_L, mesh)
    P_l_test  = compute_pair_density_lorentz(pl_cT, pl_t, mu_L, mesh)
    P_r_test  = compute_pair_density_lorentz(pr_cT, pr_t, mu_L, mesh)
    CCT = compute_CCT_from_left_right(P_l_train, P_r_train, kgrid, mesh)
    ZCT = compute_CCT_from_left_right(P_l_test,  P_r_test,  kgrid, mesh)
    return CCT, ZCT


def _solve_zeta(CCT, ZCT, mu_L, eig_rtol=1e-10):
    """Per-channel: try Cholesky+ridge first; if CCT is indefinite, fall
    back to truncated-eigendecomp pseudoinverse."""
    nq, n_rmu, _ = CCT.shape
    H = 0.5 * (CCT + jnp.conj(jnp.swapaxes(CCT, -1, -2)))
    evs = jnp.linalg.eigvalsh(H)
    ev_min = float(jnp.min(evs))
    ev_max = float(jnp.max(evs))
    is_psd = ev_min > -1e-10 * max(abs(ev_max), 1.0)
    method = ""
    if is_psd:
        trace = jnp.trace(CCT, axis1=-2, axis2=-1)
        ridge = 1e-14 * jnp.abs(trace)[:, None, None] * jnp.eye(n_rmu)[None, :, :]
        L = jnp.linalg.cholesky(CCT + ridge)
        if bool(jnp.all(jnp.isfinite(L))):
            y = jax.vmap(lambda Li, Zi: jsla.solve_triangular(Li, Zi, lower=True))(L, ZCT)
            zeta = jax.vmap(lambda Li, yi: jsla.solve_triangular(
                Li.conj().T, yi, lower=False))(L, y)
            method = "Cholesky+ridge"
            kept = jnp.full((nq,), n_rmu)
        else:
            is_psd = False
    if not is_psd:
        evs2, U = jnp.linalg.eigh(H)
        thr = eig_rtol * jnp.max(jnp.abs(evs2), axis=-1, keepdims=True)
        keep = jnp.abs(evs2) > thr
        inv_e = jnp.where(keep, 1.0 / evs2, 0.0)
        UH_Z = jnp.einsum('qji,qjk->qik', jnp.conj(U), ZCT)
        zeta = jnp.einsum('qij,qjk->qik', U, inv_e[:, :, None] * UH_Z)
        method = "trunc-eigh-pinv"
        kept = jnp.sum(keep, axis=-1)
    resid = jnp.einsum('qij,qjk->qik', CCT, zeta) - ZCT
    return zeta, resid, method, ev_min, ev_max, kept


def main():
    _print(f"jax devices: {jax.devices()}")
    mesh = _make_mesh()

    # Locate the current-density centroid file (orbit-closure varies n_unique)
    current_files = list(LORRAX_RUN.glob(CURRENT_CENTROIDS_GLOB))
    if not current_files:
        raise FileNotFoundError(
            f"No current-density centroid file found in {LORRAX_RUN}. "
            f"Run 'lxrun python3 -m centroid.kmeans_cli --density-mode current "
            f"--out-suffix _current 640' from that directory first."
        )
    current_centroids_path = current_files[0]
    _print(f"Using current centroids: {current_centroids_path.name}")

    # Probe meta for grid info
    wfn_probe = WFNReader(str(WFN_PATH))
    sym_probe = symmetry_maps.SymMaps(wfn_probe)
    meta_probe = Meta.from_system(
        wfn_probe, sym_probe, nval=BAND_RANGE[1], ncond=0,
        nband=BAND_RANGE[1] - BAND_RANGE[0], n_rmu=1, bispinor=True,
    )
    kgrid = tuple(int(x) for x in meta_probe.kgrid)

    cents_scalar  = _load_centroids(SCALAR_CENTROIDS, meta_probe)
    cents_current = _load_centroids(current_centroids_path, meta_probe)
    _print(f"  scalar centroids:  {int(cents_scalar.shape[0])}")
    _print(f"  current centroids: {int(cents_current.shape[0])}")
    overlap = set(map(tuple, np.asarray(cents_scalar).tolist())) \
            & set(map(tuple, np.asarray(cents_current).tolist()))
    _print(f"  overlap (scalar ∩ current): {len(overlap)} points")

    test_idx = _sample_test_indices(
        meta_probe, [cents_scalar, cents_current], N_TEST_POINTS, seed=42)
    _print(f"  test points: {int(test_idx.shape[0])}")

    _print("\n=== loading ψ at three index sets (bispinor) ===")
    t = time.perf_counter()
    psi_S_rmu, psi_S_rmuT, _ = _load_psi(indices=cents_scalar,  mesh=mesh)
    _print(f"  scalar:  {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    psi_C_rmu, psi_C_rmuT, _ = _load_psi(indices=cents_current, mesh=mesh)
    _print(f"  current: {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    psi_T_rmu, _,           _ = _load_psi(indices=test_idx,     mesh=mesh)
    _print(f"  test:    {time.perf_counter()-t:.2f}s")

    # Per-channel fit
    _print("\n=== per-channel fit with channel-aware centroids ===")
    zetas, evs_lo, evs_hi = {}, {}, {}
    for mu_L in (0, 1, 2, 3):
        if mu_L == 0:
            psi_c_rmu, psi_c_rmuT = psi_S_rmu, psi_S_rmuT
            cset = "scalar"
        else:
            psi_c_rmu, psi_c_rmuT = psi_C_rmu, psi_C_rmuT
            cset = "current"
        CCT, ZCT = _build_cct_zct(psi_c_rmu, psi_c_rmuT, psi_T_rmu,
                                    mu_L, kgrid, mesh)
        zeta, resid, method, ev_lo, ev_hi, kept = _solve_zeta(CCT, ZCT, mu_L)
        zetas[mu_L] = zeta
        evs_lo[mu_L], evs_hi[mu_L] = ev_lo, ev_hi
        _print(
            f"  μ_L={mu_L} ({cset}, {int(psi_c_rmu.shape[3])} centroids): "
            f"|CCT|≤{float(jnp.max(jnp.abs(CCT))):.3e}  "
            f"eig∈[{ev_lo:.3e}, {ev_hi:.3e}]  "
            f"|ζ|≤{float(jnp.max(jnp.abs(zeta))):.3e}  "
            f"|res|≤{float(jnp.max(jnp.abs(resid))):.3e}  "
            f"method={method}  rank-kept∈[{int(jnp.min(kept))},{int(jnp.max(kept))}]"
        )

    # Reconstruction probe at test points (k=0, n_l=0, n_r=8)
    _print("\n=== reconstruction at test points (band pair (0, 8), k=Γ) ===")
    K_PROBE, NL, NR = 0, 0, 8
    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
    gammas = [gamma0, gamma1, gamma2, gamma3]
    psi_l_test = psi_T_rmu[K_PROBE, NL, :, :]      # (4, n_test)
    psi_r_test = psi_T_rmu[K_PROBE, NR, :, :]      # (4, n_test)
    for mu_L in (0, 1, 2, 3):
        if mu_L == 0:
            psi_c = psi_S_rmu
        else:
            psi_c = psi_C_rmu
        psi_l_c = psi_c[K_PROBE, NL, :, :]
        psi_r_c = psi_c[K_PROBE, NR, :, :]
        gtilde = gammas[mu_L]
        P_exact = jnp.einsum('ar,ab,br->r',
                              jnp.conj(psi_l_test), gtilde, psi_r_test)
        P_cent  = jnp.einsum('am,ab,bm->m',
                              jnp.conj(psi_l_c), gtilde, psi_r_c)
        zeta_q0 = zetas[mu_L][0]                   # (n_centroid, n_test)
        P_recon = jnp.einsum('m,mr->r', P_cent, zeta_q0)
        ref = jnp.linalg.norm(P_exact)
        err = jnp.linalg.norm(P_recon - P_exact)
        _print(
            f"  μ_L={mu_L}: ‖P_exact‖={float(ref):.3e}  "
            f"rel(ζ_q=Γ)={float(err / max(ref, 1e-300)):.3e}"
        )

    _print("\nDone.")


if __name__ == "__main__":
    main()
