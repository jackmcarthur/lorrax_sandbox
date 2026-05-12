"""Minimal driver that exercises gw_init.fit_zeta with bispinor=True.

Skips kin_ion / sigma / vcoul; just checks that fit_zeta produces
all four zeta_q*.h5 files (μ_L=0,1,2,3) when bispinor=True.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Match gw_jax: enable x64 + multi-process init.  set_default_env MUST run
# before `import jax`; init_jax_distributed must run before any device-ops.
from runtime import set_default_env, init_jax_distributed
set_default_env()
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
init_jax_distributed()

INPUT_FILE = Path(__file__).resolve().parent / "cohsex.in"


def main():
    print(f"jax devices: {jax.devices()}")
    print(f"input file: {INPUT_FILE}")

    from gw.gw_config import LorraxConfig
    from common import symmetry_maps
    from file_io import WFNReader, load_centroids
    from common import Meta
    from gw.wavefunction_bundle import BandSlices
    from gw.gw_init import compute_optimal_chunks, fit_zeta
    from common.load_wfns import load_centroids_band_chunked

    config = LorraxConfig.from_input_file(str(INPUT_FILE))
    input_dir = config.input_dir
    print(f"  bispinor={config.bispinor}, "
          f"centroids_file={config.centroids_file}, "
          f"centroids_file_current={config.centroids_file_current}")

    # Most-square 2D mesh, matching gw_jax._build_mesh.
    n_dev = len(jax.devices())
    gx = int(n_dev ** 0.5)
    while gx > 1 and n_dev % gx != 0:
        gx -= 1
    mesh_xy = Mesh(np.array(jax.devices()).reshape(gx, n_dev // gx), ['x', 'y'])

    wfn = WFNReader(config.wfn_file)
    sym = symmetry_maps.SymMaps(wfn)

    _, centroid_indices, n_rmu = load_centroids(
        config.centroids_file, wfn.fft_grid)
    print(f"  scalar centroids loaded: n_rmu={n_rmu}")

    meta = Meta.from_system(
        wfn, sym, config.nval, config.ncond, config.nband,
        n_rmu, config.bispinor)
    meta.rank = jax.process_index()
    meta.n_proc = jax.process_count()
    meta.sys_dim = config.sys_dim
    meta.bispinor = config.bispinor

    band_slices = BandSlices.from_band_edges(*meta.band_edges)

    tmp_dir = os.path.join(input_dir, "tmp_test_bispinor")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"  tmp_dir: {tmp_dir}")

    with mesh_xy:
        chunks = compute_optimal_chunks(
            meta, mesh_xy,
            memory_budget_gb=config.memory_per_device_gb,
            target_utilization=config.chunk_target_utilization,
            n_b_left=band_slices.b3 - band_slices.b0,
            n_b_right=band_slices.b4 - band_slices.b1,
            r_chunk_override=(config.r_chunk_override
                              if config.r_chunk_override > 0 else None),
            zct_stage_cap_gb=config.zct_stage_cap_gb,
        )

        psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
            wfn, sym, meta,
            jnp.asarray(centroid_indices, dtype=jnp.int32),
            config.bispinor, mesh_xy,
            band_range=band_slices.full_range,
            band_chunk_size=chunks['band_chunk'],
        )

        t0 = time.perf_counter()
        zeta_path, mem_est = fit_zeta(
            wfn, sym, meta,
            jnp.asarray(centroid_indices, dtype=jnp.int32),
            mesh_xy, config, band_slices, tmp_dir,
            psi_rmu_Y, psi_rmuT_X, chunks)
        dt = time.perf_counter() - t0

    print(f"\nfit_zeta total wall: {dt:.1f}s")
    print(f"scalar zeta path: {zeta_path}")

    # Verify all 4 zeta files were produced
    print("\n=== zeta files in tmp ===")
    expected = ['zeta_q.h5', 'zeta_q_mu1.h5', 'zeta_q_mu2.h5', 'zeta_q_mu3.h5']
    all_ok = True
    for fname in expected:
        p = os.path.join(tmp_dir, fname)
        if os.path.exists(p):
            sz = os.path.getsize(p) / 1e9
            print(f"  {fname}: EXISTS ({sz:.3f} GB)")
        else:
            print(f"  {fname}: MISSING")
            all_ok = False

    # Quick magnitude sanity check
    if all_ok:
        import h5py
        print("\n=== zeta magnitudes ===")
        for fname in expected:
            p = os.path.join(tmp_dir, fname)
            with h5py.File(p, "r") as f:
                ds = f['zeta_q']
                # peek at first q
                z0 = ds[0]
                print(f"  {fname}: shape={ds.shape}  "
                      f"|z[0]|max={float(np.abs(z0).max()):.3e}  "
                      f"|z[0]|mean={float(np.abs(z0).mean()):.3e}")

    print("\nDone." if all_ok else "\nFAIL: missing zeta files")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
