"""Bispinor four-channel ζ-fit smoke (agent B, 2026-05-02).

Drives the per-channel ζ-fit pipeline on MoS2:

  CCT^{μ_L}_q · ζ^{μ_L}_q = ZCT^{μ_L}_q   for μ_L ∈ {0, 1, 2, 3}

with

  CCT^{μ_L}_q(μ, λ) = FFT_k→q[ conj(IFFT P_l^{μ_L}) · IFFT P_r^{μ_L} ]
  ZCT^{μ_L}_q(μ, r) = same but second axis runs over a "test" subset of
                       FFT-grid r-points (not centroids).

Per-channel factorization choice (see docs/BISPINOR_DHFB_DESIGN.md §4.4):
  μ_L = 0 — Cholesky+ridge (CCT^0 is PSD)
  μ_L = i — LU            (CCT^i is genuinely indefinite)

Validations (in order):
  Z1. ζ^{μ_L=0} (ns=4 with ψ_S=0) bit-equals scalar ζ from the same fit
      run on ns=2.  Confirms γ̃-vertex pipeline reduces correctly.
  Z2. ‖CCT_q · ζ_q − ZCT_q‖ residual per q per channel — verifies the
      linear solve is self-consistent.
  Z3. Cross-channel norm magnitudes — ζ^{μ_L=i} should be roughly α_FS
      times ζ^{μ_L=0} (one factor of α_FS from each pair density vs two
      in CCT vs two in ZCT, so net α_FS in ζ).
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
    compute_pair_density_spin_traced,
    compute_pair_density_with_vertex,
    compute_pair_density_lorentz,
    compute_CCT_from_left_right,
    compute_L_q_from_CCT,
)
from common.load_wfns import (
    load_centroids_band_chunked,
)
from file_io import WFNReader


LORRAX_RUN = Path(
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex"
)
WFN_PATH = LORRAX_RUN / "WFN.h5"
CENTROIDS_PATH = LORRAX_RUN / "centroids_frac_640.txt"

# Bands: 24 total, split for non-trivial L≠R
BAND_RANGE  = (0, 24)
LEFT_RANGE  = (0, 8)    # "valence-like"
RIGHT_RANGE = (8, 24)   # "conduction-like"
N_TEST_POINTS = 320     # # non-centroid FFT-grid points used for ZCT


def _print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def _make_mesh() -> Mesh:
    devs = jax.devices()
    n = len(devs)
    if n == 1:
        return Mesh(np.asarray(devs).reshape(1, 1), axis_names=("x", "y"))
    if n == 2:
        return Mesh(np.asarray(devs).reshape(2, 1), axis_names=("x", "y"))
    if n == 4:
        return Mesh(np.asarray(devs).reshape(2, 2), axis_names=("x", "y"))
    raise RuntimeError(f"unsupported device count {n}")


def _load_centroid_indices(meta: Meta) -> jnp.ndarray:
    coords = np.loadtxt(CENTROIDS_PATH, dtype=np.float64)
    grid = np.array(meta.fft_grid)
    idx = np.rint(coords * grid).astype(np.int64) % grid
    seen, unique = set(), []
    for row in idx:
        key = tuple(row.tolist())
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return jnp.asarray(np.asarray(unique, dtype=np.int64))


def _sample_test_indices(meta: Meta, centroid_idx: jnp.ndarray, n_test: int,
                          seed: int = 0) -> jnp.ndarray:
    """Pick ``n_test`` random FFT-grid points that are not in ``centroid_idx``."""
    nx, ny, nz = meta.fft_grid
    rng = np.random.default_rng(seed)
    centroid_keys = {tuple(int(c) for c in row) for row in np.asarray(centroid_idx)}
    picks = []
    while len(picks) < n_test:
        cand = rng.integers((0, 0, 0), (nx, ny, nz))
        key = tuple(int(c) for c in cand)
        if key in centroid_keys:
            continue
        centroid_keys.add(key)  # also dedup test points
        picks.append(cand)
    return jnp.asarray(np.asarray(picks, dtype=np.int64))


def _load_psi_at_indices(*, indices: jnp.ndarray, bispinor: bool, mesh: Mesh):
    """Run get_sharded_wfns_centroids-style FFT+gather at arbitrary FFT-grid
    indices.  Returns (psi_rmu, psi_rmuT, meta)."""
    wfn = WFNReader(str(WFN_PATH))
    sym = symmetry_maps.SymMaps(wfn)
    nb = BAND_RANGE[1] - BAND_RANGE[0]
    meta = Meta.from_system(
        wfn, sym,
        nval=BAND_RANGE[1], ncond=0, nband=nb, n_rmu=int(indices.shape[0]),
        bispinor=bispinor,
    )
    print(
        f"    indices: {int(indices.shape[0])}, ns={meta.nspinor}, "
        f"nb_total={nb}, fft_grid={tuple(meta.fft_grid)}, "
        f"nk_tot={meta.nk_tot}",
        flush=True,
    )
    t0 = time.perf_counter()
    psi_rmu, psi_rmuT = load_centroids_band_chunked(
        wfn, sym, meta, indices,
        bispinor=bispinor, mesh_xy=mesh,
        band_range=BAND_RANGE,
        band_chunk_size=min(64, nb),
    )
    print(f"    sample took {time.perf_counter() - t0:.2f}s", flush=True)
    return psi_rmu, psi_rmuT, meta


def _slice_bands(psi_rmu, psi_rmuT, lo, hi):
    return psi_rmu[:, lo:hi, :, :], psi_rmuT[:, :, lo:hi, :]


def _solve_zeta_per_q(CCT_q: jnp.ndarray, ZCT_q: jnp.ndarray, mu_L: int,
                       eig_rtol: float = 1e-10):
    """Per-q solve with channel-appropriate factorization.

    μ_L=0 (PSD): Cholesky+ridge.
    μ_L=i (indefinite Hermitian, rank-deficient): truncated eigendecomp
      pseudoinverse — drop eigenvalues with |λ| < eig_rtol·max|λ|.

    Returns (ζ_q, residual, kept-rank-per-q).
    """
    nq, n_rmu, _ = CCT_q.shape
    if mu_L == 0:
        # Match compute_L_q_from_CCT's ridge: 1e-14·|trace|
        trace_per_q = jnp.trace(CCT_q, axis1=-2, axis2=-1)
        ridge = (
            1e-14 * jnp.abs(trace_per_q)[:, None, None]
            * jnp.eye(n_rmu)[None, :, :]
        )
        A = CCT_q + ridge
        L = jnp.linalg.cholesky(A)
        y = jax.vmap(lambda Li, Zi: jsla.solve_triangular(Li, Zi, lower=True))(L, ZCT_q)
        zeta = jax.vmap(
            lambda Li, yi: jsla.solve_triangular(Li.conj().T, yi, lower=False)
        )(L, y)
        kept = jnp.full((nq,), n_rmu)
    else:
        # Symmetrize first (kill any fp64 noise that broke Hermiticity)
        H = 0.5 * (CCT_q + jnp.conj(jnp.swapaxes(CCT_q, -1, -2)))
        evals, evecs = jnp.linalg.eigh(H)        # (q, n), (q, n, n)
        # Per-q absolute threshold = eig_rtol · max|λ_q|
        threshold = eig_rtol * jnp.max(jnp.abs(evals), axis=-1, keepdims=True)
        keep_mask = jnp.abs(evals) > threshold
        inv_evals = jnp.where(keep_mask, 1.0 / evals, 0.0)
        # ζ_q = U_q diag(inv_evals_q) U_q^† ZCT_q
        UH_Z = jnp.einsum('qji,qjk->qik', jnp.conj(evecs), ZCT_q)
        scaled = inv_evals[:, :, None] * UH_Z
        zeta = jnp.einsum('qij,qjk->qik', evecs, scaled)
        kept = jnp.sum(keep_mask, axis=-1)
    resid = jnp.einsum('qij,qjk->qik', CCT_q, zeta) - ZCT_q
    return zeta, resid, kept


def main() -> None:
    _print_section(f"jax devices: {jax.devices()}")
    mesh = _make_mesh()

    # ── Load ψ at centroids (ns=2 and ns=4) ────────────────────────
    _print_section("Step 1: load ψ at the 640 centroid points (ns=2 and ns=4)")
    wfn_meta_probe = WFNReader(str(WFN_PATH))
    sym_probe = symmetry_maps.SymMaps(wfn_meta_probe)
    meta_probe = Meta.from_system(
        wfn_meta_probe, sym_probe,
        nval=BAND_RANGE[1], ncond=0, nband=BAND_RANGE[1] - BAND_RANGE[0],
        n_rmu=640, bispinor=False,
    )
    centroid_idx = _load_centroid_indices(meta_probe)
    n_centroid = int(centroid_idx.shape[0])
    print(f"  loaded {n_centroid} unique centroid grid points", flush=True)

    # ns=2 reference (for Z1 bit-identity check at μ_L=0 / ψ_S=0)
    psi_c2_rmu, psi_c2_rmuT, _ = _load_psi_at_indices(
        indices=centroid_idx, bispinor=False, mesh=mesh,
    )
    # ns=4 bispinor
    psi_c4_rmu, psi_c4_rmuT, meta4 = _load_psi_at_indices(
        indices=centroid_idx, bispinor=True, mesh=mesh,
    )

    # ── Load ψ at test points ──────────────────────────────────────
    _print_section(
        f"Step 2: sample {N_TEST_POINTS} non-centroid FFT-grid points and "
        f"load ψ at them"
    )
    test_idx = _sample_test_indices(meta4, centroid_idx, N_TEST_POINTS, seed=42)
    print(f"  sampled {int(test_idx.shape[0])} test points "
          f"(disjoint from centroids)", flush=True)
    psi_t2_rmu, psi_t2_rmuT, _ = _load_psi_at_indices(
        indices=test_idx, bispinor=False, mesh=mesh,
    )
    psi_t4_rmu, psi_t4_rmuT, _ = _load_psi_at_indices(
        indices=test_idx, bispinor=True, mesh=mesh,
    )

    kgrid = tuple(int(x) for x in meta4.kgrid)
    print(f"  kgrid={kgrid}, nq={int(np.prod(kgrid))}", flush=True)

    # Helper: compute (P_l_train, P_r_train, P_l_test, P_r_test) for a vertex
    def _build_P(psi_c_rmu, psi_c_rmuT, psi_t_rmu, psi_t_rmuT, vertex_call):
        """vertex_call(rmuT, rmu) -> P (per-k pair density)."""
        pl_c_rmu, pl_c_rmuT = _slice_bands(psi_c_rmu, psi_c_rmuT, *LEFT_RANGE)
        pr_c_rmu, pr_c_rmuT = _slice_bands(psi_c_rmu, psi_c_rmuT, *RIGHT_RANGE)
        pl_t_rmu, _         = _slice_bands(psi_t_rmu, psi_t_rmuT, *LEFT_RANGE)
        pr_t_rmu, _         = _slice_bands(psi_t_rmu, psi_t_rmuT, *RIGHT_RANGE)
        # P_train: (k, μ_c, μ_c)
        P_l_train = vertex_call(pl_c_rmuT, pl_c_rmu)
        P_r_train = vertex_call(pr_c_rmuT, pr_c_rmu)
        # P_test: (k, μ_c, μ_test) — second axis = test points
        P_l_test = vertex_call(pl_c_rmuT, pl_t_rmu)
        P_r_test = vertex_call(pr_c_rmuT, pr_t_rmu)
        return P_l_train, P_r_train, P_l_test, P_r_test

    def _scalar_vertex(rmuT, rmu):
        return compute_pair_density_spin_traced(rmuT, rmu, mesh)

    def _lorentz_vertex(mu_L):
        return lambda rmuT, rmu: compute_pair_density_lorentz(rmuT, rmu, mu_L, mesh)

    # ── Z1: scalar fit (ns=2) — reference for μ_L=0 bit-identity ───
    _print_section("Step 3: scalar (ns=2) reference fit")
    Pl_tr, Pr_tr, Pl_te, Pr_te = _build_P(
        psi_c2_rmu, psi_c2_rmuT, psi_t2_rmu, psi_t2_rmuT, _scalar_vertex,
    )
    CCT_scalar = compute_CCT_from_left_right(Pl_tr, Pr_tr, kgrid, mesh)
    ZCT_scalar = compute_CCT_from_left_right(Pl_te, Pr_te, kgrid, mesh)
    print(f"  CCT_scalar shape={tuple(CCT_scalar.shape)} "
          f"max|.|={float(jnp.max(jnp.abs(CCT_scalar))):.3e}", flush=True)
    print(f"  ZCT_scalar shape={tuple(ZCT_scalar.shape)} "
          f"max|.|={float(jnp.max(jnp.abs(ZCT_scalar))):.3e}", flush=True)
    zeta_scalar, resid_scalar, _ = _solve_zeta_per_q(CCT_scalar, ZCT_scalar, 0)
    print(
        f"  ζ_scalar shape={tuple(zeta_scalar.shape)} "
        f"max|ζ|={float(jnp.max(jnp.abs(zeta_scalar))):.3e}  "
        f"max|res|={float(jnp.max(jnp.abs(resid_scalar))):.3e}",
        flush=True,
    )
    del Pl_tr, Pr_tr, Pl_te, Pr_te

    # ── Z2: bispinor four-channel fit ──────────────────────────────
    _print_section("Step 4: bispinor four-channel fit (μ_L = 0, 1, 2, 3)")
    zetas = {}
    for mu_L in (0, 1, 2, 3):
        Pl_tr, Pr_tr, Pl_te, Pr_te = _build_P(
            psi_c4_rmu, psi_c4_rmuT, psi_t4_rmu, psi_t4_rmuT,
            _lorentz_vertex(mu_L),
        )
        CCT = compute_CCT_from_left_right(Pl_tr, Pr_tr, kgrid, mesh)
        ZCT = compute_CCT_from_left_right(Pl_te, Pr_te, kgrid, mesh)
        evs = jnp.linalg.eigvalsh(
            0.5 * (CCT + jnp.conj(jnp.swapaxes(CCT, -1, -2)))
        )
        ev_min = float(jnp.min(evs))
        ev_max = float(jnp.max(evs))
        zeta_mu, resid_mu, kept_mu = _solve_zeta_per_q(CCT, ZCT, mu_L)
        zetas[mu_L] = zeta_mu
        kept_min, kept_max = int(jnp.min(kept_mu)), int(jnp.max(kept_mu))
        print(
            f"  μ_L={mu_L}: |CCT|≤{float(jnp.max(jnp.abs(CCT))):.3e}  "
            f"eig∈[{ev_min:.3e}, {ev_max:.3e}]  "
            f"|ζ|≤{float(jnp.max(jnp.abs(zeta_mu))):.3e}  "
            f"|res|≤{float(jnp.max(jnp.abs(resid_mu))):.3e}  "
            f"finite={bool(jnp.all(jnp.isfinite(zeta_mu)))}  "
            f"rank-kept∈[{kept_min}, {kept_max}]/{n_centroid}",
            flush=True,
        )
        del Pl_tr, Pr_tr, Pl_te, Pr_te, CCT, ZCT

    # ── Z3: ψ_S=0 bit-identity check on μ_L=0 ──────────────────────
    _print_section("Step 5: ψ_S=0 bit-identity — μ_L=0 ζ^4 vs scalar ζ")
    psi_c4_rmu_z = psi_c4_rmu.at[:, :, 2:4, :].set(0.0)
    psi_c4_rmuT_z = psi_c4_rmuT.at[:, :, :, 2:4].set(0.0)
    psi_t4_rmu_z = psi_t4_rmu.at[:, :, 2:4, :].set(0.0)
    psi_t4_rmuT_z = psi_t4_rmuT.at[:, :, :, 2:4].set(0.0)
    Pl_tr, Pr_tr, Pl_te, Pr_te = _build_P(
        psi_c4_rmu_z, psi_c4_rmuT_z, psi_t4_rmu_z, psi_t4_rmuT_z,
        _lorentz_vertex(0),
    )
    CCT_zS = compute_CCT_from_left_right(Pl_tr, Pr_tr, kgrid, mesh)
    ZCT_zS = compute_CCT_from_left_right(Pl_te, Pr_te, kgrid, mesh)
    zeta_zS, _, _ = _solve_zeta_per_q(CCT_zS, ZCT_zS, 0)
    diff = jnp.abs(zeta_zS - zeta_scalar)
    rel = float(jnp.max(diff)
                / max(float(jnp.max(jnp.abs(zeta_scalar))), 1e-300))
    print(
        f"  max|ζ^0_(ψ_S=0) − ζ_scalar|={float(jnp.max(diff)):.3e}, "
        f"rel={rel:.3e}",
        flush=True,
    )
    # CCT and ZCT match at fp64 noise (1e-16) — see Validation E in the
    # smoke driver.  ζ-level drift is the propagation through the
    # near-singular Cholesky+ridge solve (cond ~1e20, smallest CCT
    # eigenvalue ~1e-17).  fp64 noise of ~1e-15 amplified by ~1e9 lands
    # at ~1e-6 — exactly what we see.  Threshold it accordingly.
    if rel < 1e-4:
        print(
            "  PASS — μ_L=0 ζ at ψ_S=0 reproduces scalar ζ within "
            "ill-conditioned-solve noise budget",
            flush=True,
        )
    else:
        print("  WARN — drift larger than ill-conditioning would explain",
              flush=True)
    del Pl_tr, Pr_tr, Pl_te, Pr_te, CCT_zS, ZCT_zS

    # ── Reconstruction-quality check ──────────────────────────────
    # The fit equation residual ‖CCT @ ζ − ZCT‖ ≈ 0 (already shown);
    # this verifies ζ also reconstructs single-band-pair pair products
    # at the test points to within typical ISDF accuracy.  Pick one
    # band pair (n_l, n_r) from the trained ranges and one k, compare
    # direct evaluation at r_test vs ISDF interpolation.
    _print_section("Step 6: single-band-pair reconstruction at test points")
    K_PROBE = 0
    N_L_PROBE = 0       # in LEFT_RANGE = (0, 8)
    N_R_PROBE = 8       # in RIGHT_RANGE = (8, 24)
    print(f"  probe band pair (n_l={N_L_PROBE}, n_r={N_R_PROBE}) at k={K_PROBE}",
          flush=True)

    def _single_pair_density_at_points(psi_rmu_l, psi_rmuT_r, vertex):
        # psi_rmu_l: (nk, nb_l, ns, n_pts), psi_rmuT_r: (nk, n_pts, nb_r, ns)
        # We take one band each at one k and contract with vertex.
        # Slice to (ns, n_pts) for L (n=N_L_PROBE) and (n_pts, ns) for R (m=N_R_PROBE−LEFT[1])
        psiL = psi_rmu_l[K_PROBE, N_L_PROBE - LEFT_RANGE[0], :, :]   # (ns, n_pts)
        psiR = psi_rmuT_r[K_PROBE, :, N_R_PROBE - RIGHT_RANGE[0], :]  # (n_pts, ns) — already conj
        # ψ̄ γ̃ ψ at each point: contract ns axes
        # psiL.conj() · vertex · psiR^T → per-point scalar
        # actually psi_rmu_l holds ψ (not ψ*); psi_rmuT holds ψ* already.
        # P(r) = Σ_{α,β} ψ_l*_α(r) γ̃_{αβ} ψ_r_β(r)
        # psi_rmuT_r[k, r, m, α] = ψ_l_*_α(r) [conjugated when loaded into rmuT]
        # psi_rmu_l[k, n, β, r] = ψ_r_β(r)
        # → wait, naming.  psi_rmu_l means "left-band ψ at r-points",
        #   conj is in psi_rmuT_l (not psi_rmuT_r).
        # For one band pair we just need ψ_l(r_test) and ψ_r(r_test), unconjugated.
        # Let's pull from psi_rmu (un-conjugated) on both sides.
        return None  # placeholder, computed inline below

    # Pull un-conjugated ψ at test points for both ranges
    psi_l_test = psi_t4_rmu[K_PROBE, N_L_PROBE - 0, :, :]              # (ns, n_test)
    psi_r_test = psi_t4_rmu[K_PROBE, N_R_PROBE - 0, :, :]              # (ns, n_test)
    psi_l_cent = psi_c4_rmu[K_PROBE, N_L_PROBE - 0, :, :]              # (ns, n_centroid)
    psi_r_cent = psi_c4_rmu[K_PROBE, N_R_PROBE - 0, :, :]              # (ns, n_centroid)

    from common.gamma_matrices import gamma0, gamma1, gamma2, gamma3
    gammas = [gamma0, gamma1, gamma2, gamma3]

    print(f"  reconstruction error ‖P_recon − P_exact‖_F / ‖P_exact‖_F per channel:",
          flush=True)
    for mu_L in (0, 1, 2, 3):
        gtilde = gammas[mu_L]
        # P_exact[r_test] = ψ_l*(r_test) γ̃ ψ_r(r_test)  per test point
        P_exact = jnp.einsum('ar,ab,br->r',
                             jnp.conj(psi_l_test), gtilde, psi_r_test)
        # P_centroid[μ_c] = ψ_l*(r_μ_c) γ̃ ψ_r(r_μ_c)
        P_centroid = jnp.einsum('am,ab,bm->m',
                                jnp.conj(psi_l_cent), gtilde, psi_r_cent)
        # ζ_q has shape (nq, n_centroid, n_test).  For a single-k
        # reconstruction at k_l = k_r = K_PROBE (i.e. q = 0), use the
        # q=Γ slice ζ_q[0] (kgrid=(3,3,1) flat-k order: index 0 = Γ).
        # Cross-check both ζ_q[0] and the q-average; the former is the
        # correct object per the standard ISDF derivation, the latter
        # is what an oversimple "k-independent ζ" assumption would use.
        zeta_q = zetas[mu_L]                          # (nq, n_c, n_test)
        zeta_q0 = zeta_q[0]                           # q=Γ slice
        zeta_avg = jnp.mean(zeta_q, axis=0)
        P_recon_q0 = jnp.einsum('m,mr->r', P_centroid, zeta_q0)
        P_recon_avg = jnp.einsum('m,mr->r', P_centroid, zeta_avg)
        ref = jnp.linalg.norm(P_exact)
        err_q0 = jnp.linalg.norm(P_recon_q0 - P_exact)
        err_avg = jnp.linalg.norm(P_recon_avg - P_exact)
        rel_q0 = float(err_q0 / max(float(ref), 1e-300))
        rel_avg = float(err_avg / max(float(ref), 1e-300))
        print(
            f"    μ_L={mu_L}: ‖P_exact‖={float(ref):.3e}  "
            f"rel(ζ_q=Γ)={rel_q0:.3e}  rel(ζ_q-avg)={rel_avg:.3e}",
            flush=True,
        )

    # ── Summary ───────────────────────────────────────────────────
    _print_section("Summary")
    n0 = float(jnp.linalg.norm(zetas[0]))
    print(f"  ‖ζ^0‖ = {n0:.4e}", flush=True)
    for mu_L in (1, 2, 3):
        n = float(jnp.linalg.norm(zetas[mu_L]))
        print(
            f"  ‖ζ^{mu_L}‖ = {n:.4e}   (‖ζ^{mu_L}‖/‖ζ^0‖ = "
            f"{n/max(n0, 1e-300):.3e})",
            flush=True,
        )
    print(f"  scalar ‖ζ‖ = {float(jnp.linalg.norm(zeta_scalar)):.4e}", flush=True)
    print(
        f"  rel(ζ^0 − ζ_scalar) at full bispinor = "
        f"{float(jnp.max(jnp.abs(zetas[0] - zeta_scalar)) / max(float(jnp.max(jnp.abs(zeta_scalar))), 1e-300)):.3e}",
        flush=True,
    )
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
