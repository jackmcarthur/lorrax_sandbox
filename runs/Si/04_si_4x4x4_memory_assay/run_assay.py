#!/usr/bin/env python3
"""Memory assay: systematically vary chunk sizes and measure per-device
memory consumption and timing at each stage of the ISDF fitting pipeline.

Run via Shifter with 4 GPUs:
    srun --jobid=$JOBID --gres=gpu:4 -N 1 -n 1 $SHIFTER python3 -u run_assay.py

Output: assay_results.csv with columns:
    band_chunk, r_chunk, stage, used_gb, peak_gb, time_s, n_rchunks
"""

import sys, os, gc, time, csv, json
sys.argv = ['gw_jax', '-i', os.path.join(os.path.dirname(__file__) or '.', 'cohsex.in')]

import jax
import jax.numpy as jnp
import numpy as np

def mem():
    """Return (used_gb, peak_gb) for GPU 0."""
    gc.collect()
    s = jax.local_devices()[0].memory_stats()
    return s['bytes_in_use'] / 1e9, s['peak_bytes_in_use'] / 1e9

print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
print(f"GPU memory limit: {jax.local_devices()[0].memory_stats()['bytes_limit'] / 1e9:.1f} GB")

# Import after JAX init
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.meta import Meta
from common import timing
from common.isdf_fitting import (
    fit_zeta_chunked_to_h5,
    load_gspace_for_bands,
    compute_pair_density_spin_traced,
    compute_CCT_from_left_right,
    compute_L_q_from_CCT,
    solve_zeta_from_L_q,
)
from common.load_wfns import (
    read_Gvecs_to_devices,
    get_sharded_wfns_rchunk_slice,
    get_sharded_wfns_centroids,
    load_centroids_band_chunked,
)
from common.fft_helpers import make_sharded_ifftn_3d
import configparser

# Load system
basedir = os.path.dirname(os.path.abspath(__file__))
wfn_path = os.path.join(basedir, 'WFN.h5')
cohsex_path = os.path.join(basedir, 'cohsex.in')
wfn = WFNReader(wfn_path)
sym = SymMaps(wfn)

cfg = configparser.ConfigParser()
cfg.read(cohsex_path)
params = dict(cfg['cohsex'])

nk = sym.nk_tot
nb = int(params['nband'])
nval = int(params['nval'])
ncond = int(params['ncond'])
nspinor = 2
n_rmu = int(params.get('n_centroids', 240))
fft_grid = tuple(int(x) for x in wfn.fft_grid)
n_rtot = fft_grid[0] * fft_grid[1] * fft_grid[2]
n_devices = jax.device_count()
mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(2, 2), ('x', 'y'))

centroids_file = os.path.join(basedir, params['centroids_file'])
# Centroids file has fractional coords; convert to grid indices
centroid_frac = np.loadtxt(centroids_file, dtype=np.float64)
centroid_indices = np.round(centroid_frac * np.array(fft_grid)[None, :]).astype(np.int32)
centroid_indices = centroid_indices % np.array(fft_grid)[None, :]  # wrap

print(f"\nSystem: nk={nk}, nb={nb}, nval={nval}, ncond={ncond}, nspinor={nspinor}")
print(f"FFT grid: {fft_grid}, n_rtot={n_rtot}, n_rmu={len(centroid_indices)}")
print(f"Mesh: {n_devices} devices ({mesh.shape})")

# Parameters to vary
band_chunks_to_test = [9, 18, 35]  # 9=nb/4, 18=nb/2, 35=all
r_chunks_to_test = [1152, 3456, 6912, 13824]  # fractions of n_rtot

results = []

for bc_size in band_chunks_to_test:
    for rc_size in r_chunks_to_test:
        n_bc = (nb + bc_size - 1) // bc_size
        n_rc = (n_rtot + rc_size - 1) // rc_size
        bc_padded = ((bc_size + n_devices - 1) // n_devices) * n_devices

        # Predicted sizes
        shard_gb = nk * (bc_padded // n_devices) * nspinor * n_rtot * 16 / 1e9
        fft_peak_pred = 4 * shard_gb
        reshard_gb = nk * (bc_padded // 2) * nspinor * rc_size * 16 / 1e9  # p_y=2 for 2x2 mesh

        label = f"bc={bc_size}(pad={bc_padded}) rc={rc_size} ({n_bc}×{n_rc} chunks)"
        print(f"\n{'='*70}")
        print(f"TEST: {label}")
        print(f"  Predicted: FFT peak={fft_peak_pred:.2f} GB/dev, reshard={reshard_gb:.2f} GB/dev")
        print(f"{'='*70}")

        # Run each test in isolation to get clean peaks
        # (Can't reset peak within a process, so we measure deltas)

        # 1. Centroid extraction (FFT + gather + reshard)
        gc.collect()
        u0, _ = mem()
        t0 = time.perf_counter()

        try:
            psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
                wfn, sym,
                type('Meta', (), {
                    'nk_tot': nk, 'nspinor': nspinor, 'nspinor_wfnfile': 2,
                    'fft_grid': fft_grid, 'n_rtot': n_rtot,
                    'kgrid': np.array(wfn.kgrid),
                    'memory_per_device_gb': 28,
                })(),
                jnp.asarray(centroid_indices),
                False,  # bispinor
                mesh,
                (0, nb),
                band_chunk_size=bc_size,
            )
            psi_rmu_Y.block_until_ready()
            t_centroid = time.perf_counter() - t0
            u1, p1 = mem()
            print(f"  Centroid: {t_centroid:.2f}s, used={u1:.2f} GB, peak={p1:.2f} GB")

            results.append({
                'band_chunk': bc_size, 'r_chunk': rc_size,
                'stage': 'centroid_extract',
                'used_gb': round(u1, 3), 'peak_gb': round(p1, 3),
                'time_s': round(t_centroid, 3),
                'n_band_chunks': n_bc, 'n_r_chunks': n_rc,
                'pred_fft_peak': round(fft_peak_pred, 3),
                'pred_reshard': round(reshard_gb, 3),
            })
        except Exception as e:
            print(f"  Centroid FAILED: {e}")
            results.append({
                'band_chunk': bc_size, 'r_chunk': rc_size,
                'stage': 'centroid_extract', 'error': str(e)[:200],
            })
            continue

        # 2. G-space cache load
        gc.collect()
        t0 = time.perf_counter()
        meta_fake = type('Meta', (), {
            'nk_tot': nk, 'nspinor': nspinor, 'nspinor_wfnfile': 2,
            'fft_grid': fft_grid, 'n_rtot': n_rtot,
            'kgrid': np.array(wfn.kgrid),
        })()

        try:
            cached_gspace = load_gspace_for_bands(
                wfn, sym, meta_fake, mesh, (0, nb), False,
                band_chunk_size=bc_size,
            )
            t_gcache = time.perf_counter() - t0
            u2, p2 = mem()
            print(f"  G-cache: {t_gcache:.2f}s, used={u2:.2f} GB, peak={p2:.2f} GB")

            results.append({
                'band_chunk': bc_size, 'r_chunk': rc_size,
                'stage': 'g_cache_load',
                'used_gb': round(u2, 3), 'peak_gb': round(p2, 3),
                'time_s': round(t_gcache, 3),
            })
        except Exception as e:
            print(f"  G-cache FAILED: {e}")
            results.append({
                'band_chunk': bc_size, 'r_chunk': rc_size,
                'stage': 'g_cache_load', 'error': str(e)[:200],
            })
            del psi_rmu_Y, psi_rmuT_X; gc.collect()
            continue

        # 3. R-chunk extraction (FFT + r-slice + reshard)
        gc.collect()
        kgrid = np.array(wfn.kgrid)
        kvecs_frac = sym.kvecs_asints / kgrid[None, :]

        t_rchunk_total = 0
        peak_rchunk = 0
        for rc_idx in range(n_rc):
            r_start = rc_idx * rc_size
            r_end = min(r_start + rc_size, n_rtot)
            actual_rc = r_end - r_start

            gc.collect()
            u_pre, _ = mem()
            t0 = time.perf_counter()

            try:
                for bc_idx, (psi_Gtot, bc_range) in enumerate(cached_gspace):
                    psi_rchunk = get_sharded_wfns_rchunk_slice(
                        psi_Gtot, meta_fake, r_start, r_end,
                        kvecs_frac, mesh, bc_range)
                    psi_rchunk.block_until_ready()
                    del psi_rchunk

                dt = time.perf_counter() - t0
                t_rchunk_total += dt
                u_post, p_post = mem()
                peak_rchunk = max(peak_rchunk, p_post)

                if rc_idx == 0:
                    print(f"  R-chunk[0] (r={r_start}:{r_end}): {dt:.2f}s, "
                          f"used={u_post:.2f} GB, peak={p_post:.2f} GB")
            except Exception as e:
                print(f"  R-chunk[{rc_idx}] FAILED: {e}")
                break

        print(f"  R-chunks total: {t_rchunk_total:.2f}s ({n_rc} chunks), "
              f"peak={peak_rchunk:.2f} GB")

        results.append({
            'band_chunk': bc_size, 'r_chunk': rc_size,
            'stage': 'r_chunk_extract',
            'peak_gb': round(peak_rchunk, 3),
            'time_s': round(t_rchunk_total, 3),
            'n_band_chunks': n_bc, 'n_r_chunks': n_rc,
            'pred_reshard': round(reshard_gb, 3),
        })

        # Clean up for next iteration
        del cached_gspace, psi_rmu_Y, psi_rmuT_X
        gc.collect()

# Write results
outpath = os.path.join(os.path.dirname(__file__) or '.', 'assay_results.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults written to {outpath}")

# Summary table
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"{'bc':>4s} {'rc':>6s} {'stage':>20s} {'time':>7s} {'peak':>7s} {'pred_fft':>8s} {'pred_rsh':>8s}")
for r in results:
    if 'error' in r:
        print(f"{r.get('band_chunk','?'):>4} {r.get('r_chunk','?'):>6} "
              f"{r.get('stage','?'):>20s} {'ERROR':>7s}")
    else:
        print(f"{r.get('band_chunk',''):>4} {r.get('r_chunk',''):>6} "
              f"{r.get('stage',''):>20s} {r.get('time_s',0):>7.2f} "
              f"{r.get('peak_gb',0):>7.2f} "
              f"{r.get('pred_fft_peak',''):>8} {r.get('pred_reshard',''):>8}")
