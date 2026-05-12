#!/usr/bin/env python3
"""Dump HLO protos for the full ISDF pipeline's JIT modules.

The gpu_after_optimizations.hlo.pb files contain XLA's final memory
assignment, including buffer sizes and live ranges. The memory_viewer
tool can parse these to show every buffer at peak heap.

Filters the dump to only the critical JIT modules to avoid thousands
of small files from utility JITs.
"""
import os
import sys
import glob

DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "profiles", "hlo_pipeline")
os.makedirs(DUMP_DIR, exist_ok=True)

# Only dump the modules we care about (the big memory consumers)
# This regex matches the critical JIT function names from the trace
MODULE_RE = (
    "fft_and_rslice|fft_gather_reshard|reshard_rchunk|"
    "compute_P_traced|compute_CCT_LR|"
    "left_ifft_conj|right_ifft_mul_fft|right_ifft_contract_fft|"
    "solve_batch_and_update|solve_all_at_once|"
    "batched_chol|"
    "process_allgather"
)

xla_flags = (
    f"--xla_dump_to={DUMP_DIR}"
    f" --xla_dump_hlo_as_proto=true"
    f" --xla_dump_hlo_module_re={MODULE_RE}"
)
existing = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = f"{existing} {xla_flags}".strip()
os.environ["LORRAX_MEM_PROFILE"] = "1"

import jax
pc = int(os.environ.get("SLURM_NTASKS", "1"))
if pc > 1:
    jax.distributed.initialize()

import jax.numpy as jnp
import numpy as np
import configparser
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.isdf_fitting import fit_zeta_chunked_to_h5

proc_id = jax.process_index()
if proc_id == 0:
    print(f"Processes: {jax.process_count()}, Devices: {jax.device_count()}")
    print(f"HLO dump: {DUMP_DIR}")
    print(f"Module filter: {MODULE_RE}")

# Setup (same as sweep)
basedir = os.path.dirname(os.path.abspath(__file__))
wfn = WFNReader(os.path.join(basedir, "WFN.h5"))
sym = SymMaps(wfn)
cfg = configparser.ConfigParser()
cfg.read(os.path.join(basedir, "cohsex.in"))
params = dict(cfg["cohsex"])
fft_grid = tuple(int(x) for x in wfn.fft_grid)
kgrid = np.array(wfn.kgrid)
centroids = np.loadtxt(os.path.join(basedir, params["centroids_file"]),
                       dtype=np.float64)
n_rmu = len(centroids)
ci = (np.round(centroids * np.array(fft_grid)[None, :]).astype(np.int32)
      % np.array(fft_grid)[None, :])
devices = np.array(jax.devices())
nd = len(devices)
px = int(np.sqrt(nd))
while nd % px:
    px -= 1
py = nd // px
mesh = jax.sharding.Mesh(devices.reshape(px, py), ("x", "y"))
meta = type("Meta", (), {
    "nk_tot": sym.nk_tot, "nspinor": 2, "nspinor_wfnfile": 2,
    "fft_grid": fft_grid, "n_rtot": fft_grid[0]*fft_grid[1]*fft_grid[2],
    "n_rmu": n_rmu, "kgrid": kgrid, "memory_per_device_gb": 28,
    "b_id_0": 0, "b_id_3": 35, "b_id_4": 35,
})()

if proc_id == 0:
    print(f"\nRunning pipeline (bc=35, rc=3456, qc=1)...")

out = fit_zeta_chunked_to_h5(
    wfn=wfn, sym=sym, meta=meta,
    centroid_indices=jnp.asarray(ci),
    bispinor=False, mesh_xy=mesh,
    band_range_left=(0, 35), band_range_right=(0, 35),
    chunk_r=3456, output_file=os.path.join(basedir, "tmp", "zeta_hlo.h5"),
    band_chunk_size=35, q_chunk_size=1,
    use_gspace_cache=True, isdf_pair_mode="spin_traced",
)
out[0].block_until_ready()

if proc_id == 0:
    all_files = glob.glob(os.path.join(DUMP_DIR, "*"))
    pb_files = [f for f in all_files if "after_optimizations" in f and f.endswith(".pb")]
    print(f"\nDump complete: {len(all_files)} total files, {len(pb_files)} optimized HLO protos")
    for f in sorted(pb_files):
        sz = os.path.getsize(f)
        name = os.path.basename(f)
        # Extract module name
        parts = name.split(".")
        module_name = parts[1] if len(parts) > 1 else name
        print(f"  {module_name:<60s} {sz:>10d} bytes")
