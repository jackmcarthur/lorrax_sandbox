"""Recompute Σ^B[k, m, n] for the CrI3 6x6 80 Ry bispinor IBZ gate.

For the 80Ry IBZ end-to-end gate (2026-05-16): runs X (full-BZ) and Y
(IBZ cascade) share the same WFN and centroids (so wfns + wfns_transverse
are identical), but each produced its own tmp/v_q_bispinor.h5.  The
contracted Σ^B[k, m, n] is the gate observable.

Driver mirrors the 30 Ry analogue at runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/
reconstruct_sigma_b.py — but expects to be launched from run X's working dir.
Writes <out-prefix>_X.npz and <out-prefix>_Y.npz.
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from runtime import set_default_env  # noqa: F401  (must come before jax inits)
from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend  # noqa: F401
from gw.gw_config import LorraxConfig
from gw.gw_init import prepare_isdf_and_wavefunctions, get_effective_chunk_size
from gw.gw_driver_helpers import setup_runtime, build_bgw_v_grid_fn
from gw.gw_jax import _build_mesh, flatten_V_qmunu
from common import Meta, RYD_TO_EV
from common import symmetry_maps
from file_io import WFNReader
from file_io.centroids import load_centroids
from gw.head_correction import HeadResolver
from gw.wavefunction_bundle import BandSlices
from gw.cohsex_sigma import _make_cohsex_kernels, build_Gij
from gw.sigma_x_bispinor import compute_sigma_x_bispinor


def _eval_one(label, v_q_path, wfns, wfns_transverse, V_q, Gij,
              sigma_sx_k, meta, mesh_xy, config, rep):
    if jax.process_index() == 0:
        print(f"\n=== {label}: V_q from {v_q_path} ===", flush=True)
    with mesh_xy:
        sig_x_scalar = sigma_sx_k(wfns, Gij, V_q)
    sig_x_scalar = jax.lax.with_sharding_constraint(sig_x_scalar, rep)
    sig_x_scalar.block_until_ready()

    with mesh_xy:
        sig_x_b = compute_sigma_x_bispinor(
            wfns_transverse=wfns_transverse,
            Gij=Gij,
            bispinor_v_q_path=v_q_path,
            meta=meta, mesh_xy=mesh_xy,
            backend=config.backend.slab_io,
            verbose=True,
        )
    sig_x_b.block_until_ready()

    sig_x_total = sig_x_scalar + sig_x_b
    if jax.process_index() == 0:
        print(f"  sig_x_scalar tr: {float(jnp.einsum('kmm->', sig_x_scalar).real)*RYD_TO_EV:+.6f} eV")
        print(f"  sig_x_b      tr: {float(jnp.einsum('kmm->', sig_x_b).real)*RYD_TO_EV:+.6f} eV")
        print(f"  sig_x_total  tr: {float(jnp.einsum('kmm->', sig_x_total).real)*RYD_TO_EV:+.6f} eV")
    return sig_x_scalar, sig_x_b, sig_x_total


def main(argv=None):
    argp = argparse.ArgumentParser()
    argp.add_argument("-i", "--input", required=True)
    argp.add_argument("--v-q-path-x", required=True)
    argp.add_argument("--v-q-path-y", required=True)
    argp.add_argument("--out-prefix", required=True)
    args = argp.parse_args(argv)

    config = LorraxConfig.from_input_file(args.input)
    input_dir = config.input_dir

    mesh_xy = _build_mesh()
    setup_runtime(config, mesh_xy)

    wfn = WFNReader(config.paths.wfn_file, mesh=mesh_xy)
    sym = symmetry_maps.SymMaps(wfn)
    _, centroid_indices, _n_rmu = load_centroids(
        config.paths.centroids_file, wfn.fft_grid)
    tmp_dir = os.path.join(input_dir, "tmp")
    tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{_n_rmu}.h5")

    meta = Meta.from_system(wfn, sym, config.nval, config.ncond,
                            config.nband, _n_rmu, config.bispinor)
    meta.rank = jax.process_index()
    meta.n_proc = jax.process_count()
    meta.sys_dim = config.sys_dim
    meta.bispinor = config.bispinor
    meta.chunk_size = get_effective_chunk_size(config.memory.chunk_size)
    band_slices = BandSlices.from_band_edges(*meta.band_edges)

    head_resolver = HeadResolver(config, input_dir, wfn, sym, meta,
                                 print_fn=(lambda *a, **k: None))
    bgw_v_grid_fn = build_bgw_v_grid_fn(
        config, wfn=wfn, sym=sym, input_dir=input_dir,
        print_fn=(lambda *a, **k: None))

    isdf = prepare_isdf_and_wavefunctions(
        cfg=config, wfn=wfn, sym=sym, meta=meta,
        centroid_indices=centroid_indices,
        band_slices=band_slices, mesh_xy=mesh_xy,
        tmp_dir=tmp_dir, tensors_filename=tensors_filename,
        print0=(lambda *a, **k: None) if jax.process_index() != 0 else print,
        bgw_v_grid_fn=bgw_v_grid_fn,
    )
    wfns = isdf.wf_bundle
    wfns_transverse = getattr(isdf, "wf_bundle_transverse", None)
    if wfns_transverse is None:
        raise RuntimeError("wfns_transverse is None — bispinor mode not active?")
    V_qmunu = isdf.V_qmunu
    V_q = flatten_V_qmunu(V_qmunu)

    Gij = build_Gij(meta, mesh_xy)
    sigma_sx_k, _, _ = _make_cohsex_kernels(
        mesh_xy, meta.kgrid, int(meta.nk_tot))
    rep = NamedSharding(mesh_xy, P(None, None, None))

    sx_x_scalar, sx_x_b, sx_x_tot = _eval_one(
        "X", args.v_q_path_x, wfns, wfns_transverse, V_q, Gij,
        sigma_sx_k, meta, mesh_xy, config, rep)
    sx_y_scalar, sx_y_b, sx_y_tot = _eval_one(
        "Y", args.v_q_path_y, wfns, wfns_transverse, V_q, Gij,
        sigma_sx_k, meta, mesh_xy, config, rep)

    if jax.process_index() == 0:
        out_x = args.out_prefix + "_X.npz"
        out_y = args.out_prefix + "_Y.npz"
        np.savez(out_x,
                 sig_x_scalar=np.asarray(sx_x_scalar),
                 sig_x_b=np.asarray(sx_x_b),
                 sig_x_total=np.asarray(sx_x_tot),
                 ryd_to_ev=RYD_TO_EV)
        np.savez(out_y,
                 sig_x_scalar=np.asarray(sx_y_scalar),
                 sig_x_b=np.asarray(sx_y_b),
                 sig_x_total=np.asarray(sx_y_tot),
                 ryd_to_ev=RYD_TO_EV)
        print(f"\nWrote {out_x}\nWrote {out_y}")


if __name__ == "__main__":
    sys.exit(main())
