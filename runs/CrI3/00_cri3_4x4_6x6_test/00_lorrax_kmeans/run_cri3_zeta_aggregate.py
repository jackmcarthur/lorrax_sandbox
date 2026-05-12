"""CrI3 ζ-fit aggregate A/B: channel-aware vs scalar-only centroids
under the proper-Gram path (Cholesky for all four channels).

Same logic as the MoS2 smoke (run_zeta_proper_gram.py); paths point at
this CrI3 run dir.  CrI3 has heavy iodine (Z=53), strong SOC, and is
the system where the channel-aware centroid claim should pay off if
it's going to.

Memory note: ρ tensor is (n_q=36, n_k=36, n_l=8, n_r=16, n_rmu=1800)
≈ 4.8 GB per channel — fits on 1 A100 (36 GB) with headroom but
nothing to spare.  Reduce BAND_RANGE if OOM.
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


RUN_DIR = Path(__file__).resolve().parent
WFN_PATH = RUN_DIR / "WFN.h5"
SCALAR_CENTROIDS = RUN_DIR / "centroids_frac_1800.txt"
CURRENT_CENTROIDS_GLOB = "centroids_frac_*_current.txt"

# Band window for the smoke.  CrI3 has nelec=70 and 180 bands; we use
# a small L⊕R split for parity with the MoS2 smoke and to keep ρ in GPU.
# Could be widened once memory is verified.
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
    cf = read_centroids(path)
    if expected_density is not None and cf.density != expected_density:
        raise ValueError(
            f"{path.name}: density={cf.density!r}, expected {expected_density!r}"
        )
    grid = np.array(fft_grid)
    idx = np.rint(cf.coords * grid).astype(np.int64) % grid
    seen, uniq = set(), []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key); uniq.append(row)
    return jnp.asarray(np.asarray(uniq, dtype=np.int64))


def _sample_test_indices(fft_grid, exclude, n_test, seed=42):
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
    Nx, Ny, Nz = kgrid
    n_q = Nx * Ny * Nz
    n_k = n_q
    out = []
    for q_idx in range(n_q):
        qx, qy, qz = q_idx // (Ny * Nz), (q_idx // Nz) % Ny, q_idx % Nz
        rolled = jnp.roll(psi_kgrid, shift=(-qx, -qy, -qz), axis=(0, 1, 2))
        out.append(rolled.reshape(n_k, *psi_kgrid.shape[3:]))
    return jnp.stack(out, axis=0)


def _build_rho_band_pair(psi_l, psi_r_q, vertex):
    return jnp.einsum('klaμ,ab,qkrbμ->qklrμ',
                      jnp.conj(psi_l), vertex, psi_r_q, optimize=True)


def _aggregate_recon(rho_centroid, rho_test, zeta_q):
    rho_recon = jnp.einsum('qklrμ,qμt->qklrt',
                            rho_centroid, zeta_q, optimize=True)
    diff = rho_recon - rho_test
    err2 = float(jnp.sum(jnp.abs(diff) ** 2))
    ref2 = float(jnp.sum(jnp.abs(rho_test) ** 2))
    rel2 = err2 / max(ref2, 1e-300)
    return rel2, err2, ref2


def _solve_zeta_cholesky(K_q, Z_q):
    nq, n_rmu, _ = K_q.shape
    trace = jnp.trace(K_q, axis1=-2, axis2=-1)
    ridge = 1e-14 * jnp.abs(trace)[:, None, None] * jnp.eye(n_rmu)[None, :, :]
    L = jnp.linalg.cholesky(K_q + ridge)
    if not bool(jnp.all(jnp.isfinite(L))):
        raise RuntimeError("Cholesky NaN — K_q not PSD?")
    y    = jax.vmap(lambda Li, Zi: jsla.solve_triangular(Li, Zi, lower=True))(L, Z_q)
    zeta = jax.vmap(lambda Li, yi: jsla.solve_triangular(
        Li.conj().T, yi, lower=False))(L, y)
    resid = jnp.einsum('qij,qjk->qik', K_q, zeta) - Z_q
    return zeta, resid


def main():
    _print(f"jax devices: {jax.devices()}")
    mesh = _make_mesh()

    cur_files = sorted(RUN_DIR.glob(CURRENT_CENTROIDS_GLOB))
    if not cur_files:
        raise FileNotFoundError(
            f"No current-density centroid file in {RUN_DIR}.  Run "
            f"'lxrun python3 -m centroid.kmeans_cli --density-mode current 1800' "
            f"from that directory first."
        )
    cur_path = cur_files[-1]
    _print(f"current centroid file: {cur_path.name}")

    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    kgrid = tuple(int(x) for x in wfn.kgrid)
    n_q = int(np.prod(kgrid))
    _print(f"  fft_grid = {fft_grid}, kgrid = {kgrid}, n_q = {n_q}, "
           f"alat = {float(wfn.alat):.3f} Bohr")
    _print(f"  bands {BAND_RANGE}: left {LEFT_RANGE} ⊕ right {RIGHT_RANGE}")

    cents_S = _load_centroids(SCALAR_CENTROIDS, fft_grid, expected_density=None)
    cents_C = _load_centroids(cur_path,         fft_grid, expected_density="current")
    _print(f"  scalar centroids:  {int(cents_S.shape[0])}")
    _print(f"  current centroids: {int(cents_C.shape[0])}")

    test_idx = _sample_test_indices(fft_grid, [cents_S, cents_C], N_TEST_POINTS, seed=42)
    _print(f"  test points: {int(test_idx.shape[0])}")

    _print("\n=== loading ψ at three index sets ===")
    t = time.perf_counter(); psi_S, _ = _load_psi(indices=cents_S,  mesh=mesh)
    _print(f"  scalar:  {time.perf_counter()-t:.2f}s")
    t = time.perf_counter(); psi_C, _ = _load_psi(indices=cents_C,  mesh=mesh)
    _print(f"  current: {time.perf_counter()-t:.2f}s")
    t = time.perf_counter(); psi_T, _ = _load_psi(indices=test_idx, mesh=mesh)
    _print(f"  test:    {time.perf_counter()-t:.2f}s")

    def _slice(psi):
        return (psi[:, LEFT_RANGE[0]:LEFT_RANGE[1], :, :],
                psi[:, RIGHT_RANGE[0]:RIGHT_RANGE[1], :, :])

    def _shift(psi_r_flat_k):
        rest = psi_r_flat_k.shape[1:]
        return _shift_per_q(psi_r_flat_k.reshape(*kgrid, *rest), kgrid)

    psi_S_l, psi_S_r = _slice(psi_S)
    psi_C_l, psi_C_r = _slice(psi_C)
    psi_T_l, psi_T_r = _slice(psi_T)
    t = time.perf_counter()
    psi_S_r_q = _shift(psi_S_r)
    psi_C_r_q = _shift(psi_C_r)
    psi_T_r_q = _shift(psi_T_r)
    _print(f"  shifted-tensor build: {time.perf_counter()-t:.2f}s")

    gammas = [gamma0, gamma1, gamma2, gamma3]

    _print(f"\n=== A/B: aggregate over "
           f"{(LEFT_RANGE[1]-LEFT_RANGE[0])*(RIGHT_RANGE[1]-RIGHT_RANGE[0])} band-pairs "
           f"× {n_q} k × {n_q} q × {N_TEST_POINTS} test ===\n")

    aggregate = {}
    for label, mode in (("channel-aware", "channel"), ("scalar-only ", "scalar")):
        _print(f"  ({mode}):")
        per_channel = {}
        for mu_L in (0, 1, 2, 3):
            if mode == "channel" and mu_L != 0:
                psi_l_train, psi_r_train_q = psi_C_l, psi_C_r_q
                cset = "current"
            else:
                psi_l_train, psi_r_train_q = psi_S_l, psi_S_r_q
                cset = "scalar"

            vertex = gammas[mu_L].astype(jnp.complex128)
            t = time.perf_counter()
            rho_train = _build_rho_band_pair(psi_l_train, psi_r_train_q, vertex)
            rho_test  = _build_rho_band_pair(psi_T_l, psi_T_r_q, vertex)
            K_q = jnp.einsum('qklrμ,qklrλ->qμλ',
                             jnp.conj(rho_train), rho_train, optimize=True)
            Z_q = jnp.einsum('qklrμ,qklrt->qμt',
                             jnp.conj(rho_train), rho_test, optimize=True)
            evs = jnp.linalg.eigvalsh(
                0.5 * (K_q + jnp.conj(jnp.swapaxes(K_q, -1, -2)))
            )
            ev_min, ev_max = float(jnp.min(evs)), float(jnp.max(evs))
            zeta_q, lse_resid = _solve_zeta_cholesky(K_q, Z_q)
            rel2, err2, ref2 = _aggregate_recon(rho_train, rho_test, zeta_q)
            t_total = time.perf_counter() - t

            n_train = K_q.shape[1]
            _print(
                f"    μ_L={mu_L} ({cset}, {n_train} cents): "
                f"eig∈[{ev_min:.3e}, {ev_max:.3e}]  "
                f"|LSE_res|≤{float(jnp.max(jnp.abs(lse_resid))):.3e}\n"
                f"             agg ‖Δρ‖²/‖ρ‖² = {rel2:.3e}  "
                f"(rel = {rel2 ** 0.5:.3e})  build+solve={t_total:.1f}s"
            )
            per_channel[mu_L] = rel2
            del rho_train, rho_test, K_q, Z_q, zeta_q, lse_resid
        aggregate[mode] = per_channel
        _print("")

    _print("  ── aggregate-rel summary ──")
    _print(f"  {'μ_L':<5}{'(a) channel-aware':>22}{'(b) scalar-only':>22}{'ratio (a)/(b)':>18}")
    for mu_L in (0, 1, 2, 3):
        a = aggregate['channel'][mu_L] ** 0.5
        b = aggregate['scalar'][mu_L] ** 0.5
        ratio = a / b if b > 0 else float('nan')
        _print(f"  {mu_L:<5}{a:>22.3e}{b:>22.3e}{ratio:>18.3f}")
    _print("")
    _print("Done.")


if __name__ == "__main__":
    main()
