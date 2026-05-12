"""Minimal driver that exercises gw_init.fit_zeta with bispinor=True
and (when ``LORRAX_RUN_VQ=1``) the bispinor V_q computation through
``gw_init.compute_V_q``.

Skips kin_ion / sigma / vcoul; just measures fit_zeta + bispinor V_q
wall / GPU peak.

Env knobs:
    PROFILE_OUT     — directory for memory_timeline / xprof / hlo dumps.
                      When set, the run exits at the end of profiling
                      (no zeta-file verification).
    LORRAX_SKIP_FIT — "1" to skip fit_zeta and reuse existing zeta files
                      in tmp_test_bispinor/ (only valid if all 4 are
                      already present).  Use this to isolate V_q timing
                      without re-running fit_zeta.
    LORRAX_RUN_VQ   — "1" (default) to run V_q after fit_zeta.
                      Set to "0" to only measure fit_zeta (legacy mode).
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path

# Match gw_jax: enable x64 + multi-process init.  set_default_env MUST run
# before `import jax`; init_jax_distributed must run before any device-ops.
from runtime import set_default_env, init_jax_distributed, tee_stdout_to_file
set_default_env()

# Profiling setup — pf.setup_env wires HLO dump + JAX_LOG_COMPILES so we get
# memory_timeline + hlo_summary without changing target code.  Skip when not
# wanted by leaving PROFILE_OUT unset.
_PROFILE_OUT = os.environ.get("PROFILE_OUT", "")
if _PROFILE_OUT:
    sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
    import pf
    pf.setup_env(_PROFILE_OUT, hlo=True, log_compiles=True)
    pf.attach_compile_log(f"{_PROFILE_OUT}/compile.log")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
init_jax_distributed()  # auto-gates builtins.print to rank 0 on multi-process runs

# Capture run output to run_logs/<timestamp>.log in this run_dir; rank 0 only.
import datetime as _dt
_run_dir = Path(__file__).resolve().parent
_log_path = (_run_dir / "run_logs"
             / f"test_fit_zeta_{_dt.datetime.now():%Y%m%d_%H%M%S}.log")
tee_stdout_to_file(_log_path)
print(f"[runtime] tee'd stdout/stderr to {_log_path}")

if _PROFILE_OUT:
    pf.start_memory_sampler(interval_s=0.25)

INPUT_FILE = Path(__file__).resolve().parent / "cohsex.in"


def main():
    print(f"jax devices: {jax.devices()}")
    print(f"input file: {INPUT_FILE}")

    from gw.gw_config import LorraxConfig
    from common import symmetry_maps
    from file_io import WFNReader, load_centroids
    from common import Meta
    from gw.wavefunction_bundle import BandSlices
    from gw.gw_init import compute_optimal_chunks, fit_zeta, compute_V_q
    from common.load_wfns import load_centroids_band_chunked

    config = LorraxConfig.from_input_file(str(INPUT_FILE))
    input_dir = config.input_dir
    print(f"  bispinor={config.bispinor}, "
          f"centroids_file={config.centroids_file}, "
          f"centroids_file_current={getattr(config, 'centroids_file_current', '(unset)')}")

    # Most-square 2D mesh, matching gw_jax._build_mesh, unless overridden.
    n_dev = len(jax.devices())
    _force_gx = int(os.environ.get("LORRAX_MESH_GX", "0") or 0)
    if _force_gx > 0 and n_dev % _force_gx == 0:
        gx = _force_gx
    else:
        gx = int(n_dev ** 0.5)
        while gx > 1 and n_dev % gx != 0:
            gx -= 1
    mesh_xy = Mesh(np.array(jax.devices()).reshape(gx, n_dev // gx), ['x', 'y'])
    print(f"  mesh: {gx} x {n_dev // gx}")

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

        skip_fit = os.environ.get("LORRAX_SKIP_FIT", "0") == "1"
        if skip_fit:
            zeta_path = os.path.join(tmp_dir, "zeta_q.h5")
            mem_est = {"available_vcoul_gb": config.memory_per_device_gb}
            if not os.path.exists(zeta_path):
                raise FileNotFoundError(
                    f"LORRAX_SKIP_FIT=1 but no zeta_q.h5 in {tmp_dir}")
            print(f"\n[SKIP-FIT] reusing zeta files in {tmp_dir}")
            dt = 0.0
        else:
            t0 = time.perf_counter()
            zeta_path, mem_est = fit_zeta(
                wfn, sym, meta,
                jnp.asarray(centroid_indices, dtype=jnp.int32),
                mesh_xy, config, band_slices, tmp_dir,
                psi_rmu_Y, psi_rmuT_X, chunks)
            dt = time.perf_counter() - t0

        print(f"\nfit_zeta total wall: {dt:.1f}s")
        print(f"scalar zeta path: {zeta_path}")

        # V_q computation.  Defaults to the bispinor 7+3 path when
        # ``cfg.bispinor`` is True; ``LORRAX_FORCE_SCALAR_VQ=1`` forces
        # the scalar V^{0,0} path instead (used for the baseline timing
        # comparison).
        run_vq = os.environ.get("LORRAX_RUN_VQ", "1") == "1"
        force_scalar = os.environ.get("LORRAX_FORCE_SCALAR_VQ", "0") == "1"
        v_q_dt = None
        if force_scalar:
            print("\n[LORRAX_FORCE_SCALAR_VQ=1] disabling bispinor V_q path "
                  "for baseline timing")
            # config is a frozen dataclass; bypass the setattr guard
            object.__setattr__(config, 'bispinor', False)
        if run_vq:
            print("\n=== V_q computation ({}) ==="
                  .format("bispinor" if config.bispinor else "scalar"))
            t1 = time.perf_counter()
            try:
                V_blocks, G0 = compute_V_q(
                    zeta_path, wfn, meta, mesh_xy, config,
                    mem_est=mem_est, print_fn=print, bgw_v_grid_fn=None)
                v_q_dt = time.perf_counter() - t1
                print(f"\nbispinor V_q total wall: {v_q_dt:.1f}s")
                if isinstance(V_blocks, dict):
                    print(f"  returned dict with {len(V_blocks)} non-zero "
                          f"(μ_L, ν_L) tiles")
            except Exception as exc:
                v_q_dt = time.perf_counter() - t1
                print(f"\nbispinor V_q FAILED after {v_q_dt:.1f}s: "
                      f"{type(exc).__name__}: {exc}")
                if _PROFILE_OUT:
                    # Drain the sampler so the partial run is captured.
                    pf.stop_memory_sampler()
                    pf.write_memory_timeline(f"{_PROFILE_OUT}/memory_timeline.txt")
                raise

    if _PROFILE_OUT:
        pf.stop_memory_sampler()
        pf.write_memory_timeline(f"{_PROFILE_OUT}/memory_timeline.txt")
        pf.snapshot_memory(f"{_PROFILE_OUT}/memprof/end.prof")
        sys.exit(0)

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
