#!/usr/bin/env python3
"""Run one fresh-process memory-cap test with solver-selected chunk sizes.

This is a lightweight assay helper: it computes auto chunks from
compute_optimal_chunks() for a requested memory budget, then runs
fit_zeta_chunked_to_h5() with those chunk sizes (including q_gather).
"""

from __future__ import annotations

import argparse
import configparser
import gc
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.isdf_fitting import fit_zeta_chunked_to_h5
from gw.gw_init import compute_optimal_chunks


_DISTRIBUTED_SENTINEL = "JAX_DISTRIBUTED_INITIALIZED"


def _init_jax_distributed() -> None:
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return
    proc_count = int(
        os.environ.get(
            "JAX_PROCESS_COUNT",
            os.environ.get(
                "JAX_NUM_PROCESSES",
                os.environ.get("SLURM_NTASKS", "1"),
            ),
        )
    )
    if proc_count > 1:
        try:
            jax.distributed.initialize()
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            return
        except Exception:
            pass
        coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
        if coord is None:
            host = os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME") or "localhost"
            coord = f"{host}:12355"
        proc_id = int(
            os.environ.get(
                "JAX_PROCESS_INDEX",
                os.environ.get("SLURM_PROCID", "0"),
            )
        )
        jax.distributed.initialize(
            coordinator_address=coord,
            num_processes=proc_count,
            process_id=proc_id,
        )
    os.environ[_DISTRIBUTED_SENTINEL] = "1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    parser.add_argument("--util", type=float, default=0.85)
    args = parser.parse_args()

    _init_jax_distributed()

    basedir = Path(args.basedir).resolve()
    cohsex_path = basedir / "cohsex.in"
    wfn_path = basedir / "WFN.h5"
    if not wfn_path.exists():
        raise FileNotFoundError(f"Missing WFN.h5: {wfn_path}")

    cfg = configparser.ConfigParser()
    cfg.read(cohsex_path)
    params = dict(cfg["cohsex"])

    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = nval + ncond

    wfn = WFNReader(str(wfn_path))
    sym = SymMaps(wfn)

    centroid_frac = np.loadtxt(basedir / params["centroids_file"], dtype=np.float64)
    centroid_indices = np.round(centroid_frac * np.array(wfn.fft_grid)[None, :]).astype(np.int32)
    centroid_indices = centroid_indices % np.array(wfn.fft_grid)[None, :]

    devices = np.array(jax.devices())
    n_dev = len(devices)
    p_x = int(np.sqrt(n_dev))
    while n_dev % p_x != 0:
        p_x -= 1
    p_y = n_dev // p_x
    mesh = jax.sharding.Mesh(devices.reshape(p_x, p_y), ("x", "y"))

    nqx, nqy, nqz = (int(x) for x in wfn.kgrid)
    n_q = nqx * nqy * nqz

    chunks = compute_optimal_chunks(
        n_k=sym.nk_tot,
        n_b=nband,
        n_s=2,
        n_rmu=len(centroid_frac),
        n_r=int(np.prod(wfn.fft_grid)),
        n_q=n_q,
        fft_grid=tuple(int(x) for x in wfn.fft_grid),
        n_devices=n_dev,
        memory_budget_gb=float(args.memory_gb),
        target_utilization=float(args.util),
        p_x=p_x,
        p_y=p_y,
        n_b_left=nband,
        n_b_right=nband,
        pair_density_channels=1,
        verbose=False,
    )

    if jax.process_index() == 0:
        print(
            "CHUNKS "
            f"memory_gb={args.memory_gb:.3f} "
            f"util={args.util:.3f} "
            f"band_chunk={chunks['band_chunk']} "
            f"r_chunk={chunks['chunk_r']} "
            f"q_chunk={chunks['q_chunk']} "
            f"q_gather={chunks['q_gather']}"
        )
        stage_peaks = chunks.get("memory_estimate", {}).get("stage_peaks_gb", {})
        print(f"PRED_STAGE_PEAKS {stage_peaks}")
        print(
            "PRED_SUMMARY "
            f"peak_gb={chunks.get('memory_estimate', {}).get('peak_estimate_gb', float('nan')):.6f} "
            f"bottleneck={chunks.get('memory_estimate', {}).get('bottleneck', 'unknown')}"
        )

    output_h5 = basedir / "tmp" / f"zeta_memcap_{args.memory_gb:.2f}gb.h5"
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    if output_h5.exists():
        output_h5.unlink()

    meta = type(
        "Meta",
        (),
        {
            "nk_tot": sym.nk_tot,
            "nspinor": 2,
            "nspinor_wfnfile": 2,
            "fft_grid": tuple(int(x) for x in wfn.fft_grid),
            "n_rtot": int(np.prod(wfn.fft_grid)),
            "n_rmu": len(centroid_frac),
            "kgrid": np.array(wfn.kgrid),
            "memory_per_device_gb": float(args.memory_gb),
            "b_id_0": 0,
            "b_id_3": nband,
            "b_id_4": nband,
        },
    )()

    gc.collect()
    baseline_peak = jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0)

    with mesh:
        psi_l, psi_l_t, psi_r, psi_r_t = fit_zeta_chunked_to_h5(
            wfn=wfn,
            sym=sym,
            meta=meta,
            centroid_indices=jnp.asarray(centroid_indices),
            bispinor=False,
            mesh_xy=mesh,
            band_range_left=(0, nband),
            band_range_right=(0, nband),
            chunk_r=int(chunks["chunk_r"]),
            output_file=str(output_h5),
            band_chunk_size=int(chunks["band_chunk"]),
            q_chunk_size=int(chunks["q_chunk"]),
            q_gather_size=int(chunks["q_gather"]),
            use_gspace_cache=True,
            isdf_pair_mode="spin_traced",
        )
        psi_l.block_until_ready()
        psi_l_t.block_until_ready()
        psi_r.block_until_ready()
        psi_r_t.block_until_ready()

    peak_bytes = jax.local_devices()[0].memory_stats().get("peak_bytes_in_use", 0)
    peak_gb = peak_bytes / 1e9
    baseline_gb = baseline_peak / 1e9

    if jax.process_index() == 0:
        print(
            "OBS_SUMMARY "
            f"peak_gb={peak_gb:.6f} "
            f"baseline_peak_gb={baseline_gb:.6f} "
            f"delta_peak_gb={peak_gb - baseline_gb:.6f}"
        )

    jax.experimental.multihost_utils.sync_global_devices(f"memcap_done_{args.memory_gb}")

    try:
        output_h5.unlink()
    except OSError:
        pass


if __name__ == "__main__":
    main()

