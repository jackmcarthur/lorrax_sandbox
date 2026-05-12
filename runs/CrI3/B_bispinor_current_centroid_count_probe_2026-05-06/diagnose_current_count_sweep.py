#!/usr/bin/env python3
"""Conditioning sweep for smaller CrI3 current-centroid subsets."""

from __future__ import annotations

import gc
import math
import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from common import Meta
from common.isdf_fitting import (
    _gamma_tilde_matrix,
    compute_CCT_from_left_right,
    compute_pair_density_spin_traced,
)
from common.load_wfns import load_centroids_band_chunked
from common.symmetry_maps import SymMaps
from common.wfnreader import WFNReader
from file_io.centroids import load_centroids


ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox")
BASE = ROOT / "runs/CrI3/B_bispinor_scalar_metric_80gb_2026-05-06"
PROBE = ROOT / "runs/CrI3/B_bispinor_current_centroid_count_probe_2026-05-06"
WFN_PATH = BASE / "WFN.h5"

CENTROIDS = [
    ("current100_subset", PROBE / "centroids_frac_100_current_subset.txt"),
    ("current200_subset", PROBE / "centroids_frac_200_current_subset.txt"),
    ("current300_subset", PROBE / "centroids_frac_300_current_subset.txt"),
    ("current400_subset", PROBE / "centroids_frac_400_current_subset.txt"),
    ("current600_full", ROOT / "runs/CrI3/B_nonbisp_baseline/centroids_frac_600_current.txt"),
]


def _mesh_from_devices() -> Mesh:
    devices = np.asarray(jax.devices())
    nx = max(k for k in range(1, int(math.isqrt(len(devices))) + 1) if len(devices) % k == 0)
    ny = len(devices) // nx
    return Mesh(devices.reshape((nx, ny)), ("x", "y"))


def _point_current_ratios(psi_y: jax.Array, nelec: int) -> dict[str, float]:
    psi = np.asarray(psi_y[:, :nelec, :, :])
    rho = np.einsum("knsm,knsm->m", np.conj(psi), psi, optimize=True).real
    out: dict[str, float] = {
        "rho_median": float(np.median(rho)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
    }
    denom = np.maximum(rho, 1.0e-30)
    for mu in (1, 2, 3):
        gamma = np.asarray(_gamma_tilde_matrix(mu))
        j_mu = np.einsum("knam,ab,knbm->m", np.conj(psi), gamma, psi, optimize=True)
        ratio = np.abs(j_mu) / denom
        out[f"gamma{mu}_ratio_median"] = float(np.median(ratio))
        out[f"gamma{mu}_ratio_p95"] = float(np.percentile(ratio, 95))
    return out


def _metric_stats(label: str, cfile: Path, wfn: WFNReader, sym: SymMaps, mesh: Mesh) -> None:
    _, centroid_indices, nmu = load_centroids(str(cfile), wfn.fft_grid)
    meta = Meta.from_system(wfn, sym, nval=70, ncond=80, nband=150, n_rmu=nmu, bispinor=True)
    meta.memory_per_device_gb = 28.0

    print(f"\n=== {label} ===", flush=True)
    print(f"centroid_file={cfile}", flush=True)
    print(f"nmu={nmu} band_range=(0,150) nk={meta.nk_tot} mesh={mesh.shape}", flush=True)

    psi_y, psi_xt = load_centroids_band_chunked(
        wfn,
        sym,
        meta,
        jnp.asarray(centroid_indices, dtype=jnp.int32),
        True,
        mesh,
        band_range=(0, 150),
        band_chunk_size=4,
        k_chunk_size=None,
        use_phdf5=False,
    )
    psi_y.block_until_ready()
    psi_xt.block_until_ready()

    ratios = _point_current_ratios(psi_y, nelec=70)
    print("same_point_occupied_density_and_gamma_ratios:", flush=True)
    for key in sorted(ratios):
        print(f"  {key}: {ratios[key]:.6e}", flush=True)

    p_k = compute_pair_density_spin_traced(psi_xt, psi_y, mesh)
    p_k.block_until_ready()
    p_k_right = p_k + jnp.zeros_like(p_k)
    c_q = compute_CCT_from_left_right(p_k, p_k_right, meta.kgrid, mesh)
    c_q.block_until_ready()

    c0 = np.asarray(c_q.reshape((meta.nk_tot, nmu, nmu))[0])
    c0 = 0.5 * (c0 + c0.conj().T)
    evals = np.linalg.eigvalsh(c0).real
    trace = float(np.trace(c0).real)
    scale = trace / nmu
    max_eval = float(evals[-1])
    positive = evals[evals > max(scale * 1.0e-14, 0.0)]
    min_pos = float(positive[0]) if positive.size else float("nan")
    cond = max_eval / min_pos if min_pos > 0 else float("inf")

    print("q0_scalar_metric_eigen_stats:", flush=True)
    print(f"  trace: {trace:.6e}", flush=True)
    print(f"  scale_trace_over_n: {scale:.6e}", flush=True)
    print(f"  min_eval: {float(evals[0]):.6e}", flush=True)
    print(f"  min_pos_eval_rel1e-14: {min_pos:.6e}", flush=True)
    print(f"  max_eval: {max_eval:.6e}", flush=True)
    print(f"  condition_rel1e-14: {cond:.6e}", flush=True)
    for rel in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6):
        print(f"  count_eval_lt_{rel:.0e}_scale: {int(np.sum(evals < rel * scale))}", flush=True)

    del psi_y, psi_xt, p_k, p_k_right, c_q
    gc.collect()
    jax.clear_caches()


def main() -> None:
    print(f"pid={os.getpid()} devices={jax.devices()}", flush=True)
    wfn = WFNReader(str(WFN_PATH))
    sym = SymMaps(wfn)
    mesh = _mesh_from_devices()
    centroid_list = CENTROIDS
    if len(sys.argv) > 1:
        centroid_list = [
            (Path(arg).stem, Path(arg).resolve())
            for arg in sys.argv[1:]
        ]
    for label, cfile in centroid_list:
        _metric_stats(label, cfile, wfn, sym, mesh)


if __name__ == "__main__":
    main()
