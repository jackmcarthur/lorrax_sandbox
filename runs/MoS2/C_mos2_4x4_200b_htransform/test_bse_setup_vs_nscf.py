"""Sanity check: bse_setup.compute_wfns_fi vs direct lorrax NSCF on 8×8.

Drives the BSE interpolation setup top-level callable end-to-end, then
compares the X/Y-sharded wfn bundle to a direct 8×8 NSCF ψ at the same
centroids. Uses the gauge-invariant cosine-similarity-of-pair-densities
metric since SOC gives 2-fold (Kramers) degenerate pairs.

Run: LORRAX_NGPU=4 lxrun python3 -u test_bse_setup_vs_nscf.py
Requires WFN_8x8.h5 from `psp.run_nscf -i nscf_8x8.in`.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from runtime import init_jax_distributed
init_jax_distributed()

import jax
from jax.experimental.multihost_utils import process_allgather

from gw.gw_init import read_cohsex_input
from file_io import WFNReader
from file_io.centroids import load_centroids
from common import symmetry_maps, Meta
from common.load_wfns import load_centroids_band_chunked
from bandstructure.htransform import (
    streaming_galerkin_solve, _build_mesh_xy, load_wfns_and_enk_for_sigma,
)
from bandstructure.bse_setup import compute_wfns_fi


def banner(s):
    if jax.process_index() == 0:
        print(f"\n{'=' * 60}\n{s}\n{'=' * 60}", flush=True)


def main():
    log = print if jax.process_index() == 0 else (lambda *a, **k: None)
    params = read_cohsex_input("cohsex.in")
    mesh_xy = _build_mesh_xy()
    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    bispinor = bool(params.get("bispinor", False))

    # ── Step 1: 4×4 htransform ──
    banner("Step 1: 4×4 htransform → ctilde, B_at_mu, enk_sigma")
    wfn_co = WFNReader("WFN.h5")
    sym_co = symmetry_maps.SymMaps(wfn_co)
    _, centroid_idx, n_mu = load_centroids(
        params["centroids_file"], tuple(int(x) for x in wfn_co.fft_grid))
    meta_co = Meta.from_system(wfn_co, sym_co, nval, ncond, nband, n_mu, bispinor)
    band_range = (int(meta_co.b_id_0), int(meta_co.b_id_4))
    nsigmarange, enk_sigma = load_wfns_and_enk_for_sigma(wfn_co, sym_co, nval, ncond, nband)
    with mesh_xy:
        S, ctilde, B_at_mu = streaming_galerkin_solve(
            wfn_co, sym_co, meta_co, centroid_idx, mesh_xy, band_range,
            rtol=1e-8, log_fn=log, bispinor=bispinor,
        )

    # ── Step 2: bse_setup top-level callable on the bottom 32 bands ──
    banner("Step 2: compute_wfns_fi(kgrid_fi=8×8×1, bands [0, 32))")
    kgrid_co = (int(meta_co.nkx), int(meta_co.nky), int(meta_co.nkz))
    band_window_fi = (0, 32)
    with mesh_xy:
        bundle = compute_wfns_fi(
            ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
            kgrid_co=kgrid_co, kgrid_fi="8 8 1",
            band_window_fi=band_window_fi,
            mesh_xy=mesh_xy, log_fn=log,
        )
    log(f"  bundle: psi_rmu_Y={bundle.psi_rmu_Y.shape} "
        f"P{bundle.psi_rmu_Y.sharding.spec}, "
        f"psi_rmuT_X={bundle.psi_rmuT_X.shape} "
        f"P{bundle.psi_rmuT_X.sharding.spec}, "
        f"enk_full={bundle.enk_full.shape}")
    nb_fi = band_window_fi[1] - band_window_fi[0]

    # ── Step 3: load direct 8×8 NSCF ψ at the same centroids, same band window ──
    banner("Step 3: direct lorrax-8×8 NSCF ψ at the same centroids")
    wfn_fi = WFNReader("WFN_8x8.h5")
    sym_fi = symmetry_maps.SymMaps(wfn_fi)
    meta_fi = Meta.from_system(wfn_fi, sym_fi, nval, ncond, nband, n_mu, bispinor)
    # Slice the 8×8 NSCF to bands [b_id_0 + b_min, b_id_0 + b_max).
    band_range_true = (int(meta_fi.b_id_0) + band_window_fi[0],
                       int(meta_fi.b_id_0) + band_window_fi[1])
    with mesh_xy:
        psi_true_Y, _ = load_centroids_band_chunked(
            wfn_fi, sym_fi, meta_fi, centroid_idx, bispinor, mesh_xy,
            band_range=band_range_true, band_chunk_size=64)
    log(f"  ground-truth ψ: {psi_true_Y.shape}")

    psi_pred_h = np.asarray(process_allgather(bundle.psi_rmu_Y))
    psi_true_h = np.asarray(process_allgather(psi_true_Y))
    nk_fi = psi_pred_h.shape[0]

    # Identify NEW vs ORIG k via 4×4 ⊂ 8×8 indexing.
    k_co_np = np.asarray(sym_co.unfolded_kpts)
    k_fi_np = np.asarray(sym_fi.unfolded_kpts)
    def find_index(k, klist, tol=1e-6):
        for i, kk in enumerate(klist):
            if np.max(np.abs(((kk - k + 0.5) % 1.0) - 0.5)) < tol:
                return i
        return -1
    is_orig = np.zeros(nk_fi, dtype=bool)
    for ico in range(len(k_co_np)):
        ifi = find_index(k_co_np[ico], k_fi_np)
        if 0 <= ifi < nk_fi:
            is_orig[ifi] = True
    new_mask = ~is_orig

    # ── Cosine similarity of pair-summed Σ_s |ψ|² along centroid axis ──
    banner("Pair-density cosine similarity vs direct 8×8 NSCF")
    def pair_density(psi):
        d = (np.abs(psi) ** 2).sum(axis=2)                     # (nk, nb, n_μ)
        nb_pair = (nb_fi // 2) * 2
        return d[:, :nb_pair].reshape(d.shape[0], nb_pair // 2, 2, -1).sum(axis=2)
    rho_pred = pair_density(psi_pred_h)
    rho_true = pair_density(psi_true_h)
    num = (rho_pred * rho_true).sum(axis=-1)
    den = (np.linalg.norm(rho_pred, axis=-1) *
           np.linalg.norm(rho_true, axis=-1))
    cos_sim = num / np.maximum(den, 1e-30)
    log(f"  ALL  k:  min {cos_sim.min():.4f}, median {np.median(cos_sim):.4f}")
    log(f"  ORIG k:  min {cos_sim[is_orig].min():.4f}, "
        f"median {np.median(cos_sim[is_orig]):.4f}  (expect ≈1)")
    log(f"  NEW  k:  min {cos_sim[new_mask].min():.4f}, "
        f"median {np.median(cos_sim[new_mask]):.4f}, "
        f"max {cos_sim[new_mask].max():.4f}")

    # ── Verify X copy matches Y copy after transpose (sharding-roundtrip sanity) ──
    psi_X_h = np.asarray(process_allgather(bundle.psi_rmuT_X))    # (nk, n_μ, nb, ns)
    psi_Y_as_X = np.transpose(psi_pred_h, (0, 3, 1, 2))            # (nk, n_μ, nb, ns)
    diff = np.max(np.abs(psi_X_h - psi_Y_as_X))
    log(f"  X vs Y-transposed bundle: max abs diff = {diff:.3e}  (should be 0)")

    # ── Compare recovered enk_full to direct 8×8 NSCF energies (window only) ──
    e_pred = np.asarray(process_allgather(bundle.enk_full))        # (nk_fi, nb_fi), Ry
    e_true = np.asarray(wfn_fi.energies)[0, :,
                                          band_window_fi[0]:band_window_fi[1]]
    de_meV = (np.sort(e_pred, axis=1) - np.sort(e_true, axis=1)) * 13605.7
    log(f"  enk_full vs 8×8-NSCF (meV): "
        f"max abs {float(np.abs(de_meV).max()):.2f}, "
        f"median abs {float(np.median(np.abs(de_meV))):.2f}")
    log(f"    ORIG k: max abs {float(np.abs(de_meV[is_orig]).max()):.2f} meV")
    log(f"    NEW  k: max abs {float(np.abs(de_meV[new_mask]).max()):.2f} meV, "
        f"median abs {float(np.median(np.abs(de_meV[new_mask]))):.2f} meV")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
