"""Bispinor ISDF: per-channel ζ-fit + 16-fold V tensor → H5.

Pipeline:
  1. Load both centroid files (scalar + Gordon-current).
  2. Load ψ at both centroid sets and at a "test" r-set = union of both
     centroid sets — this is the r-axis on which ζ and the V integrand
     are sampled.
  3. Per channel μ_L ∈ {0, 1, 2, 3}:
        - K_q via proper-Gram band-pair enumeration on the appropriate
          centroid set (Cholesky-able).
        - ZCT_q on r-axis = union centroids (also via band-pair enumeration).
        - Solve ζ^{μ_L}_q (n_rmu × n_runion) via Cholesky+ridge.
  4. Save ζ^{μ_L}_q to H5 in groups `/zeta_lorentz_<μ_L>`.
  5. Compute V_q^{μ_L,ν_L}(μ_c, ν_c):
        - FFT ζ^{μ_L}_q over the r-axis (approximate, since r-axis is sparse)
        - Apply D^{μ_L,ν_L}(q+G) per Coulomb-gauge formula
        - Contract → V_q^{μ_L,ν_L}(μ_c, ν_c)
  6. Save full (n_q, 4, 4, n_rmu_left, n_rmu_right) V tensor to H5
     (μ_L=0 uses scalar centroids, μ_L=i uses current centroids;
     off-block (μ_L=0, ν_L=i) and (i, 0) entries are zero by Coulomb gauge).

Caveat: the FFT step (5) treats the union-centroid r-axis as a Monte-Carlo
sample of the unit cell.  For accurate V_q, the r-axis would need to be
dense (full FFT grid via z-chunked accumulation).  Phase-1.5 work; this
script delivers the architecture end-to-end with approximate magnitudes.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import h5py
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
OUT_H5 = RUN_DIR / "bispinor_isdf.h5"

BAND_RANGE = (0, 24)
LEFT_RANGE = (0, 8)
RIGHT_RANGE = (8, 24)


def _print(s):
    print(s, flush=True)


def _make_mesh():
    devs = jax.devices()
    return Mesh(np.asarray(devs[:1]).reshape(1, 1), axis_names=("x", "y"))


def _load_centroids_idx(path: Path, fft_grid):
    cf = read_centroids(path)
    grid = np.array(fft_grid)
    idx = np.rint(cf.coords * grid).astype(np.int64) % grid
    seen, uniq = set(), []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    return jnp.asarray(np.asarray(uniq, dtype=np.int64))


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


def _build_rho(psi_l, psi_r_q, vertex):
    """ρ(q, k, n_l, n_r, μ) = Σ_{ab} conj(ψ_l[k,n_l,a,μ]) γ̃_{ab} ψ_r[q,k,n_r,b,μ]."""
    return jnp.einsum('klaμ,ab,qkrbμ->qklrμ',
                      jnp.conj(psi_l), vertex, psi_r_q, optimize=True)


def _solve_zeta_cholesky(K_q, Z_q):
    """Cholesky+ridge solve.  Returns ζ_q and residual norm."""
    nq, n_rmu, _ = K_q.shape
    trace = jnp.trace(K_q, axis1=-2, axis2=-1)
    ridge = 1e-14 * jnp.abs(trace)[:, None, None] * jnp.eye(n_rmu)[None, :, :]
    L = jnp.linalg.cholesky(K_q + ridge)
    if not bool(jnp.all(jnp.isfinite(L))):
        raise RuntimeError("Cholesky NaN — K_q not PSD?")
    y = jax.vmap(lambda Li, Zi: jsla.solve_triangular(Li, Zi, lower=True))(L, Z_q)
    zeta = jax.vmap(lambda Li, yi: jsla.solve_triangular(Li.conj().T, yi, lower=False))(L, y)
    resid = jnp.einsum('qij,qjk->qik', K_q, zeta) - Z_q
    return zeta, float(jnp.max(jnp.abs(resid)))


def main():
    t0 = time.perf_counter()
    _print(f"jax devices: {jax.devices()}")
    mesh = _make_mesh()

    cur_files = sorted(RUN_DIR.glob(CURRENT_CENTROIDS_GLOB))
    if not cur_files:
        raise FileNotFoundError("Run kmeans_cli --density-mode current first.")
    cur_path = cur_files[-1]
    _print(f"current centroid file: {cur_path.name}")

    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    kgrid = tuple(int(x) for x in wfn.kgrid)
    n_q = int(np.prod(kgrid))
    alat = float(wfn.alat)
    bvec_dimless = np.asarray(wfn.bvec, dtype=np.float64)
    bvec_cart = bvec_dimless * (2.0 * math.pi / alat)   # Bohr⁻¹
    nx, ny, nz = fft_grid

    cents_S = _load_centroids_idx(SCALAR_CENTROIDS, fft_grid)
    cents_C = _load_centroids_idx(cur_path, fft_grid)
    n_S = int(cents_S.shape[0])
    n_C = int(cents_C.shape[0])
    _print(f"  fft_grid={fft_grid}, kgrid={kgrid}, alat={alat:.3f} Bohr")
    _print(f"  scalar centroids: {n_S}, current centroids: {n_C}")

    # r-axis for ζ evaluation: union of both centroid sets (deduped)
    seen = set()
    union = []
    for arr in (np.asarray(cents_S), np.asarray(cents_C)):
        for row in arr:
            k = tuple(int(c) for c in row)
            if k not in seen:
                seen.add(k)
                union.append(row)
    cents_U = jnp.asarray(np.asarray(union, dtype=np.int64))
    n_U = int(cents_U.shape[0])
    _print(f"  union (r-axis for ζ): {n_U}")

    _print("\n=== loading ψ at three index sets (scalar, current, union) ===")
    t = time.perf_counter()
    psi_S, _ = _load_psi(indices=cents_S, mesh=mesh)
    _print(f"  scalar:  {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    psi_C, _ = _load_psi(indices=cents_C, mesh=mesh)
    _print(f"  current: {time.perf_counter()-t:.2f}s")
    t = time.perf_counter()
    psi_U, _ = _load_psi(indices=cents_U, mesh=mesh)
    _print(f"  union:   {time.perf_counter()-t:.2f}s")

    def _slice(psi):
        return (psi[:, LEFT_RANGE[0]:LEFT_RANGE[1], :, :],
                psi[:, RIGHT_RANGE[0]:RIGHT_RANGE[1], :, :])

    def _shift(psi_r_flat_k):
        rest = psi_r_flat_k.shape[1:]
        return _shift_per_q(psi_r_flat_k.reshape(*kgrid, *rest), kgrid)

    psi_S_l, psi_S_r = _slice(psi_S)
    psi_C_l, psi_C_r = _slice(psi_C)
    psi_U_l, psi_U_r = _slice(psi_U)

    _print("\n=== building shifted ψ_r tensors ===")
    t = time.perf_counter()
    psi_S_r_q = _shift(psi_S_r)
    psi_C_r_q = _shift(psi_C_r)
    psi_U_r_q = _shift(psi_U_r)
    _print(f"  shift build: {time.perf_counter()-t:.2f}s")

    gammas = [gamma0, gamma1, gamma2, gamma3]

    # ─── per-channel ζ-fit ────────────────────────────────────────
    _print("\n=== per-channel proper-Gram ζ-fit ===")
    zetas = {}      # μ_L → (n_q, n_rmu_channel, n_U)
    Ls = {}         # μ_L → Cholesky factor (n_q, n_rmu_channel, n_rmu_channel)
    centroid_set = {0: "scalar", 1: "current", 2: "current", 3: "current"}
    n_rmu_set = {0: n_S, 1: n_C, 2: n_C, 3: n_C}

    @jax.jit
    def _per_q_K(rho_l_q_one):
        # rho_l_q_one: (k, n_l, n_r, μ)
        return jnp.einsum('klrμ,klrλ->μλ', jnp.conj(rho_l_q_one), rho_l_q_one,
                          optimize=True)

    @jax.jit
    def _per_q_Z(rho_l_q_one, rho_u_q_one):
        return jnp.einsum('klrμ,klrt->μt', jnp.conj(rho_l_q_one), rho_u_q_one,
                          optimize=True)

    for mu_L in (0, 1, 2, 3):
        if mu_L == 0:
            psi_l_train, psi_r_train_q = psi_S_l, psi_S_r_q
        else:
            psi_l_train, psi_r_train_q = psi_C_l, psi_C_r_q

        vertex = gammas[mu_L].astype(jnp.complex128)

        t = time.perf_counter()
        n_train = int(psi_l_train.shape[3])
        # Build K_q and Z_q per-q to keep memory bounded.  Each per-q ρ tensor
        # is (n_k, n_l, n_r, n_rmu_axis) ≈ 36×8×16×3500 ≈ 16M complex = 260 MB
        # — plenty of headroom even with both channels in flight.
        K_q = np.zeros((n_q, n_train, n_train), dtype=np.complex128)
        Z_q = np.zeros((n_q, n_train, n_U), dtype=np.complex128)
        for q_idx in range(n_q):
            # ρ_train(k, n_l, n_r, μ_train) for this q
            rho_train_q = jnp.einsum(
                'klaμ,ab,krbμ->klrμ',
                jnp.conj(psi_l_train), vertex, psi_r_train_q[q_idx], optimize=True,
            )
            rho_union_q = jnp.einsum(
                'klaν,ab,krbν->klrν',
                jnp.conj(psi_U_l), vertex, psi_U_r_q[q_idx], optimize=True,
            )
            K_q[q_idx] = np.asarray(_per_q_K(rho_train_q))
            Z_q[q_idx] = np.asarray(_per_q_Z(rho_train_q, rho_union_q))
            del rho_train_q, rho_union_q

        K_q_j = jnp.asarray(K_q)
        Z_q_j = jnp.asarray(Z_q)
        del K_q, Z_q
        zeta_q, lse_resid = _solve_zeta_cholesky(K_q_j, Z_q_j)
        evs = jnp.linalg.eigvalsh(0.5 * (K_q_j + jnp.conj(jnp.swapaxes(K_q_j, -1, -2))))
        ev_min, ev_max = float(jnp.min(evs)), float(jnp.max(evs))
        zeta_max = float(jnp.max(jnp.abs(zeta_q)))
        # Move ζ to host RAM to free GPU memory before next channel.
        zeta_np = np.asarray(zeta_q)
        zetas[mu_L] = zeta_np
        del K_q_j, Z_q_j, zeta_q, evs
        _print(
            f"  μ_L={mu_L} [{centroid_set[mu_L]}, {n_train} cents]: "
            f"eig∈[{ev_min:.3e},{ev_max:.3e}]  |LSE_res|≤{lse_resid:.3e}  "
            f"|ζ|≤{zeta_max:.3e}  build+solve={time.perf_counter()-t:.2f}s"
        )

    # ─── V_q^{μ_L,ν_L}(μ_c, ν_c) via FFT-and-contract on union r-axis ──
    # We approximate the G-space contraction by treating the union centroids
    # as a Monte-Carlo r-set; the FFT magnitudes are proportional to the
    # true G-space ζ but with broader spread (sparse-r aliasing).  Defined
    # but approximate; refine in phase-1.5 with a dense r-axis.
    #
    # D^{μ_L,ν_L}(K) in Coulomb gauge:
    #   D^{0,0}(K) = 4π/|K|²
    #   D^{i,j}(K) = (4π/|K|²) (δ_{ij} − K_i K_j / |K|²)
    #   D^{0,i} = D^{i,0} = 0
    _print("\n=== V_q^{μ_L,ν_L} via FFT-and-contract on union r-axis ===")
    # Build (q+G) Cartesian momenta for each q, in Bohr⁻¹.
    # We'll FFT ζ on the union r-axis treated as samples in the unit cell.
    # The "G-axis" is the dual of the union-r set; for Monte-Carlo r the
    # natural FFT is non-uniform.  We use uniform-grid FFT bins matched to
    # the FFT box size as an approximation.
    union_rcart_frac = jnp.asarray(np.asarray(cents_U) / np.asarray(fft_grid))
    # Cartesian r in Bohr (rows × avec_cart):
    avec_cart = jnp.asarray(np.asarray(wfn.avec) * alat, dtype=jnp.float64)

    # FFT-grid integer momenta → Bohr⁻¹ Cartesian K
    gx_int = jnp.fft.fftfreq(nx, d=1.0/nx).astype(jnp.float64)
    gy_int = jnp.fft.fftfreq(ny, d=1.0/ny).astype(jnp.float64)
    gz_int = jnp.fft.fftfreq(nz, d=1.0/nz).astype(jnp.float64)
    bvec_cart_j = jnp.asarray(bvec_cart, dtype=jnp.float64)

    # For each q, build (q+G) over the FFT grid in Bohr⁻¹.  q index → fractional q vector.
    def _q_cart(q_idx):
        qx = q_idx // (kgrid[1] * kgrid[2])
        qy = (q_idx // kgrid[2]) % kgrid[1]
        qz = q_idx % kgrid[2]
        q_frac = jnp.array([qx / kgrid[0], qy / kgrid[1], qz / kgrid[2]],
                           dtype=jnp.float64)
        return jnp.einsum('a,ab->b', q_frac, bvec_cart_j)

    # We compute V_q^{μ_L,ν_L}(μ_c, ν_c) = Σ_G ζ^{μ_L,*}(q,μ_c,G) D^{μ_L,ν_L}(q+G) ζ^{ν_L}(q,ν_c,G)
    # ζ in G-space approximated via NDFT of ζ on union r-axis.
    # NDFT: ζ_G ≈ Σ_r ζ(r) e^{-iG·r_cart} for sparse r — biased but useful.

    cents_U_int = jnp.asarray(cents_U, dtype=jnp.int64)
    # r_cart = (cents_U_int / fft_grid) @ avec_cart    — Bohr
    r_cart = (cents_U_int.astype(jnp.float64) / jnp.asarray(fft_grid)) @ avec_cart   # (n_U, 3)

    # Coulomb prefactor (Rydberg units, matching LORRAX vcoul: 8π / |K|²)
    # We use 4π for atomic units; the user can rescale.
    PREFAC = 4.0 * jnp.pi

    # Build V_q: (n_q, 4, 4, max_n_rmu, max_n_rmu) — we keep separate per-channel
    # n_rmu so block sizes vary; store as a dict keyed by (μ_L, ν_L).
    V_blocks = {}

    # G-chunked V computation to keep peak memory bounded.
    # Phase tensor (nG_chunk, n_U) at 50K × n_U complex doubles ≈ 2-3 GB.
    nG_total = nx * ny * nz
    g_chunk = 50_000
    n_g_chunks = (nG_total + g_chunk - 1) // g_chunk

    # Pre-build flat FFT-grid integer indices once
    gx_flat = jnp.broadcast_to(gx_int[:, None, None], (nx, ny, nz)).reshape(-1)
    gy_flat = jnp.broadcast_to(gy_int[None, :, None], (nx, ny, nz)).reshape(-1)
    gz_flat = jnp.broadcast_to(gz_int[None, None, :], (nx, ny, nz)).reshape(-1)

    _print(f"  Computing V over {n_q} q-points × 16 (μ_L,ν_L) blocks "
           f"× {n_g_chunks} G-chunks of {g_chunk}...")
    t_v_start = time.perf_counter()
    for q_idx in range(n_q):
        q_cart_local = _q_cart(q_idx)           # (3,) in Bohr⁻¹

        # Initialize V_q^{μ_L,ν_L} accumulators — only nonzero blocks
        V_q_blocks = {}
        for mu_L in (0, 1, 2, 3):
            for nu_L in (0, 1, 2, 3):
                if (mu_L == 0) ^ (nu_L == 0):
                    continue   # zero by Coulomb gauge
                shape_l = n_rmu_set[mu_L]
                shape_r = n_rmu_set[nu_L]
                V_q_blocks[(mu_L, nu_L)] = jnp.zeros((shape_l, shape_r), dtype=jnp.complex128)

        for g_start in range(0, nG_total, g_chunk):
            g_end = min(g_start + g_chunk, nG_total)
            # K = q + G_chunk in Bohr⁻¹
            gx_c = gx_flat[g_start:g_end]
            gy_c = gy_flat[g_start:g_end]
            gz_c = gz_flat[g_start:g_end]
            Kx = q_cart_local[0] + gx_c * bvec_cart_j[0, 0] + gy_c * bvec_cart_j[1, 0] + gz_c * bvec_cart_j[2, 0]
            Ky = q_cart_local[1] + gx_c * bvec_cart_j[0, 1] + gy_c * bvec_cart_j[1, 1] + gz_c * bvec_cart_j[2, 1]
            Kz = q_cart_local[2] + gx_c * bvec_cart_j[0, 2] + gy_c * bvec_cart_j[1, 2] + gz_c * bvec_cart_j[2, 2]
            K2 = Kx**2 + Ky**2 + Kz**2
            K2_safe = jnp.where(K2 > 1e-12, K2, 1.0)
            v_K = jnp.where(K2 > 1e-12, PREFAC / K2_safe, 0.0)   # (n_g_chunk_size,)

            # transverse projector P^T[i,j](K) = δ_ij − K_iK_j/|K|²
            # Build only when needed; per (i, j) it's a 1-D vector
            # K_outer[g, i, j] = K_i(g) K_j(g) / |K|²
            K_stack = jnp.stack([Kx, Ky, Kz], axis=1)   # (nGc, 3)
            K_outer = jnp.where(
                K2[:, None, None] > 1e-12,
                K_stack[:, :, None] * K_stack[:, None, :] / K2_safe[:, None, None],
                0.0,
            )
            P_T = jnp.eye(3)[None, :, :] - K_outer   # (nGc, 3, 3)

            # Phase: e^{-iK·r_cart}, shape (nGc, n_U)
            phase = jnp.exp(-1j * (
                Kx[:, None] * r_cart[None, :, 0]
                + Ky[:, None] * r_cart[None, :, 1]
                + Kz[:, None] * r_cart[None, :, 2]
            ))   # (nGc, n_U)

            # ζ in this G-chunk: (n_rmu_channel, nGc)
            # zetas[μ_L] is on host (numpy); upload per-q slice to device.
            zeta_G_chunk = {}
            for mu_L in (0, 1, 2, 3):
                zeta_at_q = jnp.asarray(zetas[mu_L][q_idx])   # (n_rmu_channel, n_U)
                zeta_G_chunk[mu_L] = jnp.einsum('μu,Gu->μG', zeta_at_q, phase)

            # Accumulate V blocks
            # V^{0,0}(μ,ν) += Σ_G_chunk ζ^{0,*}(μ,G) v(K) ζ^{0}(ν,G)
            V_q_blocks[(0, 0)] = V_q_blocks[(0, 0)] + jnp.einsum(
                'μG,G,νG->μν',
                jnp.conj(zeta_G_chunk[0]), v_K, zeta_G_chunk[0],
                optimize=True,
            )
            for mu_L in (1, 2, 3):
                for nu_L in (1, 2, 3):
                    PT_ij = P_T[:, mu_L - 1, nu_L - 1]   # (nGc,)
                    V_q_blocks[(mu_L, nu_L)] = V_q_blocks[(mu_L, nu_L)] + jnp.einsum(
                        'μG,G,νG->μν',
                        jnp.conj(zeta_G_chunk[mu_L]),
                        v_K * PT_ij,
                        zeta_G_chunk[nu_L],
                        optimize=True,
                    )
            for k_blk in V_q_blocks.values():
                k_blk.block_until_ready()
            del zeta_G_chunk


        # Stash this q's blocks (move to host RAM so GPU is freed for next q)
        for key, V_block in V_q_blocks.items():
            if key not in V_blocks:
                V_blocks[key] = []
            V_blocks[key].append(np.asarray(V_block))
        del V_q_blocks

        if (q_idx + 1) % 4 == 0:
            _print(f"    q {q_idx+1}/{n_q}  ({time.perf_counter()-t_v_start:.1f}s)")

    # Stack per (μ_L, ν_L) block into (n_q, n_rmu, n_rmu) on host
    V_stacked = {key: np.stack(V_list, axis=0) for key, V_list in V_blocks.items()}
    _print(f"  V build total: {time.perf_counter()-t_v_start:.2f}s")

    # ─── Save to H5 ────────────────────────────────────────────────
    _print(f"\n=== writing to {OUT_H5.name} ===")
    with h5py.File(OUT_H5, "w") as f:
        # Metadata
        meta_grp = f.create_group("meta")
        meta_grp.attrs["bispinor"] = True
        meta_grp.attrs["fft_grid"] = np.asarray(fft_grid)
        meta_grp.attrs["kgrid"] = np.asarray(kgrid)
        meta_grp.attrs["alat_bohr"] = alat
        meta_grp.attrs["band_range"] = np.asarray(BAND_RANGE)
        meta_grp.attrs["left_range"] = np.asarray(LEFT_RANGE)
        meta_grp.attrs["right_range"] = np.asarray(RIGHT_RANGE)
        meta_grp.attrs["n_scalar_centroids"] = n_S
        meta_grp.attrs["n_current_centroids"] = n_C
        meta_grp.attrs["n_union_r"] = n_U

        # Centroid index sets
        f.create_dataset("scalar_centroid_indices", data=np.asarray(cents_S))
        f.create_dataset("current_centroid_indices", data=np.asarray(cents_C))
        f.create_dataset("union_r_indices", data=np.asarray(cents_U))

        # ζ tensors: one group per Lorentz channel
        zeta_grp = f.create_group("zeta_lorentz")
        for mu_L in (0, 1, 2, 3):
            ds = zeta_grp.create_dataset(
                f"zeta_{mu_L}", data=np.asarray(zetas[mu_L]),
                compression="gzip", compression_opts=4,
            )
            ds.attrs["centroid_set"] = centroid_set[mu_L]
            ds.attrs["shape"] = "(n_q, n_rmu_channel, n_union_r)"

        # V tensors
        v_grp = f.create_group("V_lorentz")
        for (mu_L, nu_L), V_t in V_stacked.items():
            v_grp.create_dataset(
                f"V_{mu_L}_{nu_L}", data=np.asarray(V_t),
                compression="gzip", compression_opts=4,
            )

    sz_mb = OUT_H5.stat().st_size / 1e6
    _print(f"  H5 size: {sz_mb:.1f} MB")

    _print(f"\n=== summary ===")
    _print(f"  total wall time: {time.perf_counter()-t0:.1f}s")
    _print(f"  ζ shapes: μ_L=0 → {tuple(zetas[0].shape)}  "
           f"μ_L=1 → {tuple(zetas[1].shape)}")
    _print("  V_lorentz blocks saved:")
    for (mu_L, nu_L), V_t in V_stacked.items():
        _print(f"    V^{mu_L}{nu_L}: shape={tuple(V_t.shape)}  "
               f"max|.|={float(np.max(np.abs(V_t))):.3e}")
    _print("\nDone.")


if __name__ == "__main__":
    main()
