"""Hermitian-transpose check on bispinor V_q tiles.

Re-run the bispinor V_q on existing ζ files, capture the dict, and
compute  max(|V^{j,i} - conj(V^{i,j}.T)|)  for all 3 hermitian pairs.

Run via: lxrun python3 -u check_hermitian.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path

from runtime import set_default_env, init_jax_distributed
set_default_env()
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
init_jax_distributed()

INPUT_FILE = Path(__file__).resolve().parent / "cohsex.in"


def main():
    here = Path(__file__).resolve().parent
    tmp = here / "tmp_test_bispinor"
    print(f"jax devices: {jax.devices()}")

    from gw.gw_config import LorraxConfig
    from common import symmetry_maps
    from file_io import WFNReader, load_centroids
    from common import Meta
    from gw.wavefunction_bundle import BandSlices
    from gw.gw_init import compute_optimal_chunks, compute_V_q
    from common.load_wfns import load_centroids_band_chunked

    config = LorraxConfig.from_input_file(str(INPUT_FILE))
    n_dev = len(jax.devices())
    gx = int(n_dev**0.5)
    while gx > 1 and n_dev % gx != 0:
        gx -= 1
    mesh_xy = Mesh(np.array(jax.devices()).reshape(gx, n_dev // gx), ['x', 'y'])

    wfn = WFNReader(config.wfn_file)
    sym = symmetry_maps.SymMaps(wfn)
    _, centroid_indices, n_rmu = load_centroids(
        config.centroids_file, wfn.fft_grid)
    meta = Meta.from_system(wfn, sym, config.nval, config.ncond,
                            config.nband, n_rmu, config.bispinor)
    meta.rank = jax.process_index()
    meta.n_proc = jax.process_count()
    meta.sys_dim = config.sys_dim
    meta.bispinor = config.bispinor

    band_slices = BandSlices.from_band_edges(*meta.band_edges)
    zeta_path = str(tmp / "zeta_q.h5")
    mem_est = {"available_vcoul_gb": config.memory_per_device_gb}
    with mesh_xy:
        V_blocks, G0 = compute_V_q(
            zeta_path, wfn, meta, mesh_xy, config,
            mem_est=mem_est, print_fn=print, bgw_v_grid_fn=None)

    print("\n=== Hermitian-transpose check ===")
    pairs = [((2, 1), (1, 2)), ((3, 1), (1, 3)), ((3, 2), (2, 3))]
    for (j, i), (i2, j2) in pairs:
        A = V_blocks[(j, i)]
        B = V_blocks[(i2, j2)]
        # Want A == conj(B.T) along last two axes (per q)
        diff = A - jnp.conj(jnp.transpose(B, (0, 2, 1)))
        # Need to gather to a single host
        diff_max = float(jnp.max(jnp.abs(diff)))
        norm = float(jnp.max(jnp.abs(B)))
        print(f"  V^({j},{i}) vs conj(V^({i2},{j2}).T):"
              f"  max|Δ|={diff_max:.3e}   max|V|={norm:.3e}   "
              f"rel={diff_max/max(1e-30, norm):.3e}")

    print("\n=== q=0 trace summary ===")
    # G0 already gathered at end of compute_V_q
    for k in sorted(V_blocks.keys()):
        bl = V_blocks[k]
        tr = float(jnp.trace(bl[0]).real)
        absmax = float(jnp.max(jnp.abs(bl[0])))
        print(f"  ({k[0]},{k[1]})  q=0 trace = {tr:+.3e}   "
              f"max|V|={absmax:.3e}")
    print("Done.")


if __name__ == "__main__":
    main()
