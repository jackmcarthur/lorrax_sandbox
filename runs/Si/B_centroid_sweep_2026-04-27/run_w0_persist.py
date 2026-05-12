"""Compute W0 = (1 - V chi0)^(-1) V from existing restart tensors and persist
the unflattened W0_qmunu back to tmp/isdf_tensors_<N>.h5.  Mirrors the
chi0+W stage of gw_jax.main but stops there and writes W instead of running
sigma.

Run this once per centroid-count run dir; afterwards bse_jax can use the
screened W from the same restart file.
"""
from __future__ import annotations

# Mirror gw_jax.py's runtime bootstrap order — env BEFORE jax import.
from runtime import set_default_env
set_default_env()

import argparse
import os
import sys
import time

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from runtime import init_jax_distributed
init_jax_distributed()


def _build_mesh():
    total = jax.process_count() * jax.local_device_count()
    gx = int(np.sqrt(total))
    while gx > 1 and total % gx != 0:
        gx -= 1
    return Mesh(np.array(jax.devices()).reshape(gx, total // gx), ['x', 'y'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="cohsex.in for this run dir")
    args = ap.parse_args()

    from gw.gw_config import LorraxConfig
    from gw.w_isdf import (
        flatten_V_qmunu, compute_chi0, precompile_chi0,
        precompile_solve_w, solve_w, build_static_quadrature,
    )
    from gw.gw_init import prepare_isdf_and_wavefunctions
    from gw.wavefunction_bundle import BandSlices
    from common import symmetry_maps, Meta
    from runtime import nccl_warmup
    from file_io import WFNReader, load_centroids

    import dataclasses
    cfg = LorraxConfig.from_input_file(args.input, print_fn=print)
    cfg = dataclasses.replace(cfg, restart=True)   # force load of V_qmunu
    input_dir = cfg.input_dir
    mesh_xy = _build_mesh()
    nccl_warmup(mesh_xy)

    wfn = WFNReader(cfg.wfn_file)
    sym = symmetry_maps.SymMaps(wfn)
    _, centroid_indices, n_rmu = load_centroids(cfg.centroids_file, wfn.fft_grid)
    tmp_dir = os.path.join(input_dir, "tmp")
    tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{n_rmu}.h5")

    meta = Meta.from_system(wfn, sym, cfg.nval, cfg.ncond, cfg.nband, n_rmu, cfg.bispinor)
    meta.rank = jax.process_index()
    meta.n_proc = jax.process_count()
    meta.sys_dim = cfg.sys_dim
    meta.bispinor = cfg.bispinor
    band_slices = BandSlices.from_band_edges(*meta.band_edges)

    if jax.process_index() == 0:
        print(f"  [w0_persist] tensors file: {tensors_filename}", flush=True)

    isdf = prepare_isdf_and_wavefunctions(
        cfg=cfg, wfn=wfn, sym=sym, meta=meta,
        centroid_indices=centroid_indices, band_slices=band_slices,
        mesh_xy=mesh_xy, tmp_dir=tmp_dir, tensors_filename=tensors_filename,
        print0=print, bgw_v_grid_fn=None,
    )
    V_qmunu = isdf.V_qmunu
    wfns = isdf.wf_bundle

    V_q = flatten_V_qmunu(V_qmunu)
    if jax.process_index() == 0:
        print(f"  [w0_persist] V_qmunu shape={V_qmunu.shape}", flush=True)

    t0 = time.perf_counter()
    quad, e_ref = build_static_quadrature(wfns, cfg.minimax_config, print_fn=print)
    precompile_chi0(wfns, quad, meta, mesh_xy, energy_reference=e_ref)
    chi0_q = compute_chi0(wfns, quad, meta, mesh_xy, energy_reference=e_ref)
    chi0_q.block_until_ready()
    t_chi0 = time.perf_counter() - t0

    t0 = time.perf_counter()
    precompile_solve_w(V_q, chi0_q, meta, mesh_xy, memory_mode=cfg.isdf_memory_mode)
    W_q_flat = solve_w(V_q, chi0_q, meta, mesh_xy, memory_mode=cfg.isdf_memory_mode)
    del chi0_q
    W_q_flat.block_until_ready()
    t_w = time.perf_counter() - t0

    if jax.process_index() == 0:
        print(f"  [w0_persist] chi0 = {t_chi0:.2f}s, W solve = {t_w:.2f}s", flush=True)
        print(f"  [w0_persist] W_q_flat shape = {W_q_flat.shape}", flush=True)

    # Reshape flat-q W back to V_qmunu layout: (1, npol=1, npol=1, nkx, nky, nkz, μ, μ).
    nkx, nky, nkz = V_qmunu.shape[3:6]
    npol = V_qmunu.shape[1]
    n_rmu = W_q_flat.shape[-1]
    if npol != 1:
        raise NotImplementedError("only npol=1 supported here")

    # All-gather sharded W to every rank's host so process 0 can write it.
    from jax.experimental import multihost_utils
    W_global = multihost_utils.process_allgather(W_q_flat, tiled=False)
    # process_allgather with tiled=False returns a stacked array if the
    # sharding has multiple shards; we want a single (nq, μ, μ) ndarray.
    W_host = np.asarray(W_global)
    if W_host.ndim == 4 and W_host.shape[0] == jax.process_count():
        # First axis is process index; concatenation depends on sharding axis.
        # On a single-process run W_host should already be (nq, μ, μ).
        # For multi-process we'd need the original sharding axis to merge.
        # Single-process is the sandbox case; assert and bail otherwise.
        if jax.process_count() != 1:
            raise NotImplementedError(
                "multi-process gather not implemented; single-process sandbox only"
            )
        W_host = W_host[0]
    W_kgrid = W_host.reshape(nkx, nky, nkz, n_rmu, n_rmu)
    W_full = W_kgrid[None, None, None, :, :, :, :, :].astype(V_qmunu.dtype)

    if jax.process_index() == 0:
        print(f"  [w0_persist] writing W0_qmunu shape={W_full.shape} dtype={W_full.dtype}", flush=True)
        with h5py.File(tensors_filename, "a") as f:
            if "W0_qmunu" in f:
                del f["W0_qmunu"]
            f.create_dataset("W0_qmunu", data=W_full)
            f["W0_qmunu"].attrs["W0_ready"] = True
        print(f"  [w0_persist] DONE: W0_ready=True in {tensors_filename}", flush=True)


if __name__ == "__main__":
    main()
