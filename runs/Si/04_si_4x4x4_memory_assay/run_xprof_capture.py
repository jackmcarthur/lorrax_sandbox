#!/usr/bin/env python3
"""Capture XProf traces for one LORRAX ISDF pipeline config.

Wraps the full zeta fitting pipeline in jax.profiler.trace() to capture
HLO-level memory attribution. The trace directory will contain .xplane.pb
files that can be analyzed with tools/analyze_xprof_memory.py or opened
in TensorBoard's memory_viewer.

Usage:
    srun --jobid=$JOBID --gres=gpu:4 -N 1 -n 4 $SHIFTER \
        --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
        --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python3 -u run_xprof_capture.py

Output: profiles/xprof/<timestamp>/ with .xplane.pb files
"""

import gc
import os
import sys
import time

os.environ["LORRAX_MEM_PROFILE"] = "1"

# JAX distributed init (same as run_memory_sweep.py)
import jax

_DISTRIBUTED_SENTINEL = "JAX_DISTRIBUTED_INITIALIZED"
def _init_jax_distributed():
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                     os.environ.get("JAX_NUM_PROCESSES",
                     os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            return
        except Exception:
            pass
        import subprocess
        coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
        if coord is None:
            nodelist = os.environ.get("SLURM_NODELIST")
            if nodelist:
                try:
                    result = subprocess.run(
                        ["scontrol", "show", "hostnames", nodelist],
                        capture_output=True, text=True, check=True)
                    first_host = result.stdout.strip().split("\n")[0]
                    coord = f"{first_host}:12355"
                except Exception:
                    pass
            if coord is None:
                host = (os.environ.get("SLURMD_NODENAME")
                        or os.environ.get("HOSTNAME") or "localhost")
                coord = f"{host}:12355"
        proc_id = int(os.environ.get("JAX_PROCESS_INDEX",
                      os.environ.get("SLURM_PROCID", "0")))
        jax.distributed.initialize(coordinator_address=coord,
                                   num_processes=proc_count,
                                   process_id=proc_id)
    os.environ[_DISTRIBUTED_SENTINEL] = "1"

_init_jax_distributed()

import jax.numpy as jnp
import numpy as np
import configparser

from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.isdf_fitting import fit_zeta_chunked_to_h5

proc_id = jax.process_index()
n_devs = jax.device_count()

if proc_id == 0:
    print(f"Processes: {jax.process_count()}, Devices: {n_devs}")
    print(f"Device: {jax.devices()[0].device_kind}")

# Setup
basedir = os.path.dirname(os.path.abspath(__file__))
wfn = WFNReader(os.path.join(basedir, "WFN.h5"))
sym = SymMaps(wfn)

cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, "cohsex.in"))
params = dict(cfg["cohsex"])

nk_tot = sym.nk_tot
nband = int(params["nband"])
nval = int(params["nval"])
nspinor = 2
fft_grid = tuple(int(x) for x in wfn.fft_grid)
n_rtot = fft_grid[0] * fft_grid[1] * fft_grid[2]
kgrid = np.array(wfn.kgrid)

centroids_file = os.path.join(basedir, params["centroids_file"])
centroid_frac = np.loadtxt(centroids_file, dtype=np.float64)
n_rmu = len(centroid_frac)
centroid_indices = np.round(centroid_frac * np.array(fft_grid)[None, :]).astype(np.int32)
centroid_indices = centroid_indices % np.array(fft_grid)[None, :]

devices = np.array(jax.devices())
p_x = int(np.sqrt(n_devs))
while n_devs % p_x != 0:
    p_x -= 1
p_y = n_devs // p_x
mesh = jax.sharding.Mesh(devices.reshape(p_x, p_y), ("x", "y"))

meta = type("Meta", (), {
    "nk_tot": nk_tot, "nspinor": nspinor, "nspinor_wfnfile": 2,
    "fft_grid": fft_grid, "n_rtot": n_rtot, "n_rmu": n_rmu,
    "kgrid": kgrid, "memory_per_device_gb": 28,
    "b_id_0": 0, "b_id_3": nband, "b_id_4": nband,
})()

# Config: use a medium r-chunk to see realistic peaks
band_chunk = 35
r_chunk = 3456
q_chunk = 1

if proc_id == 0:
    print(f"\nSystem: nk={nk_tot}, nb={nband}, ns={nspinor}, n_rmu={n_rmu}")
    print(f"Config: bc={band_chunk}, rc={r_chunk}, qc={q_chunk}")
    print(f"Mesh: {p_x}x{p_y}")

output_h5 = os.path.join(basedir, "tmp", "zeta_xprof.h5")
os.makedirs(os.path.dirname(output_h5), exist_ok=True)

# Warmup run (JIT compilation, no trace — compilation artifacts pollute the trace)
if proc_id == 0:
    print("\n--- Warmup run (JIT compilation) ---")
psi_l_Y, psi_l_X, psi_r_Y, psi_r_X = fit_zeta_chunked_to_h5(
    wfn=wfn, sym=sym, meta=meta,
    centroid_indices=jnp.asarray(centroid_indices),
    bispinor=False, mesh_xy=mesh,
    band_range_left=(0, nband), band_range_right=(0, nband),
    chunk_r=r_chunk, output_file=output_h5,
    band_chunk_size=band_chunk, q_chunk_size=q_chunk,
    use_gspace_cache=True, isdf_pair_mode="spin_traced",
)
psi_l_Y.block_until_ready()
del psi_l_Y, psi_l_X, psi_r_Y, psi_r_X
gc.collect()

# Traced run (all JITs are cached, trace captures execution only)
trace_dir = os.path.join(basedir, "profiles", "xprof",
                         f"bc{band_chunk}_rc{r_chunk}_qc{q_chunk}")
os.makedirs(trace_dir, exist_ok=True)

if proc_id == 0:
    print(f"\n--- Traced run (XProf capture) ---")
    print(f"Trace dir: {trace_dir}")

with jax.profiler.trace(trace_dir):
    psi_l_Y, psi_l_X, psi_r_Y, psi_r_X = fit_zeta_chunked_to_h5(
        wfn=wfn, sym=sym, meta=meta,
        centroid_indices=jnp.asarray(centroid_indices),
        bispinor=False, mesh_xy=mesh,
        band_range_left=(0, nband), band_range_right=(0, nband),
        chunk_r=r_chunk, output_file=output_h5,
        band_chunk_size=band_chunk, q_chunk_size=q_chunk,
        use_gspace_cache=True, isdf_pair_mode="spin_traced",
    )
    psi_l_Y.block_until_ready()

if proc_id == 0:
    print(f"\nTrace written to {trace_dir}")
    print("Analyze with:")
    print(f"  uv run python tools/analyze_xprof_memory.py \\")
    print(f"    {trace_dir}/plugins/profile/*/GPUtop.xplane.pb")

# Cleanup
del psi_l_Y, psi_l_X, psi_r_Y, psi_r_X
if os.path.exists(output_h5):
    try:
        os.remove(output_h5)
    except OSError:
        pass
