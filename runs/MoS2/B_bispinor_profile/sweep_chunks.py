"""Quick MoS2 fit_zeta sweep — captures (predicted, AOT, measured) for
several (r_chunk, band_chunk) configs in one process.

Output: sweep_results.json next to this script.

Calls compute_optimal_chunks + fit_zeta repeatedly, varying the
r_chunk_override and band_chunk_size args.  Reads the heuristic peak
from the chunks dict, the AOT peak from gw_init.fit_zeta's stdout
parsing (it logs `AOT fit_one_rchunk peak`), and the measured peak
from the value returned by fit_zeta_chunked_to_h5 (which is what
nvidia-smi sampled).
"""
from __future__ import annotations
import os, sys, json, time

from runtime import set_default_env, init_jax_distributed
set_default_env()

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
init_jax_distributed()

from gw.gw_config import LorraxConfig
from common import symmetry_maps, Meta
from file_io import WFNReader, load_centroids
from gw.wavefunction_bundle import BandSlices
from gw.gw_init import compute_optimal_chunks
from common.load_wfns import load_centroids_band_chunked
from common.isdf_fitting import fit_zeta_chunked_to_h5

INPUT_FILE = os.path.join(os.path.dirname(__file__), "cohsex.in")
RUN_DIR = os.path.dirname(__file__)
TMP = os.path.join(RUN_DIR, "tmp_sweep")
os.makedirs(TMP, exist_ok=True)


def aot_peak(meta, mesh_xy, chunks, band_slices):
    try:
        from gw.aot_memory_model import predict_kernel_peak, SysDims, MeshSpec, Knobs
        nb_L = band_slices.b3 - band_slices.b0
        nb_R = band_slices.b4 - band_slices.b1
        nb_full = (max(band_slices.b3, band_slices.b4)
                   - min(band_slices.b0, band_slices.b1))
        sys_dim = SysDims(
            kgrid=tuple(meta.kgrid),
            fft_grid=tuple(meta.fft_grid),
            n_rmu=int(meta.n_rmu),
            n_s=int(meta.nspinor),
            n_b=int(nb_full),
            n_b_sum=int(nb_L + nb_R),
            n_r=int(meta.n_rtot),
        )
        p_x, p_y = mesh_xy.devices.shape
        ms = MeshSpec(p_x=int(p_x), p_y=int(p_y))
        peak = predict_kernel_peak(
            "fit_one_rchunk", sys_dim,
            Knobs.of(chunk_r=int(chunks['chunk_r']),
                     band_chunk=int(chunks['band_chunk'])),
            ms, tag="current",
        )
        return float(peak) / 1e9
    except Exception as e:
        print(f"  AOT predict failed: {e!r}")
        return None


def main():
    cfg = LorraxConfig.from_input_file(INPUT_FILE)
    n_dev = len(jax.devices())
    gx = int(n_dev ** 0.5)
    while gx > 1 and n_dev % gx != 0:
        gx -= 1
    mesh_xy = Mesh(np.array(jax.devices()).reshape(gx, n_dev // gx),
                   ['x', 'y'])
    print(f"mesh: {gx} x {n_dev//gx}")

    wfn = WFNReader(cfg.wfn_file)
    sym = symmetry_maps.SymMaps(wfn)
    _, centroid_indices, n_rmu = load_centroids(cfg.centroids_file, wfn.fft_grid)
    meta = Meta.from_system(wfn, sym, cfg.nval, cfg.ncond, cfg.nband,
                            n_rmu, cfg.bispinor)
    meta.rank = jax.process_index()
    meta.n_proc = jax.process_count()
    meta.sys_dim = cfg.sys_dim
    meta.bispinor = cfg.bispinor

    band_slices = BandSlices.from_band_edges(*meta.band_edges)
    band_range_left = (band_slices.b0, band_slices.b3)
    band_range_right = (band_slices.b1, band_slices.b4)

    # n_rtot = 24*24*80 = 46080 ; sweep across reasonable r_chunks.
    sweeps = [
        # (r_chunk_override, band_chunk_override)
        (1000, None),
        (2000, None),
        (4000, None),
        (8000, None),
        (16000, None),
        (None, None),  # full nr=46080, let chooser pick band_chunk
    ]

    results = []
    for cr_ov, bc_ov in sweeps:
        with mesh_xy:
            chunks = compute_optimal_chunks(
                meta, mesh_xy,
                memory_budget_gb=cfg.memory_per_device_gb,
                target_utilization=cfg.chunk_target_utilization,
                n_b_left=band_slices.b3 - band_slices.b0,
                n_b_right=band_slices.b4 - band_slices.b1,
                r_chunk_override=cr_ov,
                zct_stage_cap_gb=cfg.zct_stage_cap_gb,
                verbose=False,
            )
            if bc_ov is not None:
                chunks['band_chunk'] = int(bc_ov)

            mem_est = chunks['memory_estimate']
            heur_peak = mem_est['peak_estimate_gb']
            stages = mem_est.get('limit_info', {})
            cr = chunks['chunk_r']
            bc = chunks['band_chunk']
            print(f"\n=== sweep cr={cr_ov} -> chunk_r={cr}, band_chunk={bc} ===")
            print(f"  heuristic peak: {heur_peak:.3f} GB  bottleneck={mem_est['bottleneck']}")
            print(f"  per-stage: " + "  ".join(f"{k}={v:.3f}" for k, v in stages.items()))

            aot_gb = aot_peak(meta, mesh_xy, chunks, band_slices)
            if aot_gb is not None:
                print(f"  AOT peak: {aot_gb:.3f} GB")

            psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
                wfn, sym, meta,
                jnp.asarray(centroid_indices, dtype=jnp.int32),
                cfg.bispinor, mesh_xy,
                band_range=band_slices.full_range,
                band_chunk_size=chunks['band_chunk'],
            )

            output = os.path.join(TMP, f"zeta_q_cr{cr}_bc{bc}.h5")
            try:
                t0 = time.perf_counter()
                peak_bytes = fit_zeta_chunked_to_h5(
                    wfn=wfn, sym=sym, meta=meta,
                    centroid_indices=jnp.asarray(centroid_indices, dtype=jnp.int32),
                    mesh_xy=mesh_xy,
                    chunk_r=chunks['chunk_r'],
                    output_file=output,
                    psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
                    band_chunk_size=chunks['band_chunk'],
                    q_chunk_size=chunks['q_chunk'],
                    q_gather_size=chunks.get('q_gather', 0),
                    bispinor=cfg.bispinor,
                    band_range_left=band_range_left,
                    band_range_right=band_range_right,
                    k_chunk_size=chunks.get('k_chunk', 0),
                    band_norms=getattr(wfn, 'band_norms', None),
                    use_ffi_io=cfg.use_ffi_io,
                    gspace_mode=cfg.gspace_mode,
                    max_r_chunks=int(getattr(cfg, 'max_r_chunks', -1) or -1),
                )
                dt = time.perf_counter() - t0
                meas = float(peak_bytes) / 1e9 if peak_bytes else 0.0
                print(f"  measured peak: {meas:.3f} GB ({dt:.1f}s)")
            except Exception as e:
                print(f"  FAILED: {e!r}")
                meas = None
                dt = None

            ratio = (meas / heur_peak) if (meas and heur_peak > 0) else None
            r2 = (meas / aot_gb) if (meas and aot_gb and aot_gb > 0) else None
            results.append({
                'cr_override': cr_ov,
                'chunk_r': int(cr),
                'band_chunk': int(bc),
                'heuristic_peak_gb': float(heur_peak),
                'heuristic_stages': {k: float(v) for k, v in stages.items()},
                'heuristic_bottleneck': str(mem_est['bottleneck']),
                'aot_peak_gb': float(aot_gb) if aot_gb else None,
                'measured_peak_gb': meas,
                'ratio_meas_heur': ratio,
                'ratio_meas_aot': r2,
                'wall_s': dt,
            })

            # Clear traced caches between sweeps
            from common import isdf_fitting as _isdf, load_wfns as _lw
            _isdf._fit_one_rchunk_cache.clear()
            _isdf._compute_pair_density_cache.clear()
            _isdf._accum_pair_density_cache.clear()
            _isdf._compute_pair_density_vertex_cache.clear()
            _isdf._accum_pair_density_vertex_cache.clear()
            _lw._rchunk_slice_cache.clear()
            jax.clear_caches()
            import gc; gc.collect()

    if jax.process_index() == 0:
        out = os.path.join(RUN_DIR, "sweep_results.json")
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out}")
        print("\n=== SUMMARY (cr | bc | heur | aot | meas | meas/heur | meas/aot) ===")
        for r in results:
            print(f"  cr={r['chunk_r']:5d} bc={r['band_chunk']:3d}  "
                  f"heur={r['heuristic_peak_gb']:.2f}  "
                  f"aot={(r['aot_peak_gb'] or 0):.2f}  "
                  f"meas={(r['measured_peak_gb'] or 0):.2f}  "
                  f"r_h={(r['ratio_meas_heur'] or 0):.2f}  "
                  f"r_a={(r['ratio_meas_aot'] or 0):.2f}  "
                  f"bn={r['heuristic_bottleneck']}")


if __name__ == "__main__":
    main()
