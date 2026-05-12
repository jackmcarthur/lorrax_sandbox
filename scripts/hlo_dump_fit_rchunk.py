"""Dump the HLO buffer-assignment report for fit_one_rchunk on a
4-CPU-device mesh.  This is enough to see the STRUCTURE of
allocations — which ops produce temps, which inputs survive to peak,
etc.  Byte magnitudes differ from the real GPU run but the allocation
count and the op-per-allocation is identical.

No allocation needed — runs on the login node / inside the container.
"""
import os
import sys
from pathlib import Path

OUT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/reports/aot_memory_model_poc_2026-04-20/hlo_dumps")
OUT.mkdir(parents=True, exist_ok=True)

# Force 4 CPU "devices" so 2x2 mesh is possible.  Set BEFORE jax import.
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count=4 "
    f"--xla_dump_to={OUT} "
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_hlo_as_proto=false"
)
# Force CPU platform
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import Mesh
from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs, get_kernel, aot_measure,
)

devs = jax.devices()
print(f"Devices: {len(devs)} ({[d.platform for d in devs]})")

mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))

# Use MoS2 3×3 config but SMALL enough that compile is fast
sys_ = SysDims(
    kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
    n_rmu=320, n_s=1, n_b=40, n_b_sum=80,
    n_r=80 * 72 * 8,
)
knobs = Knobs.of(chunk_r=10000, band_chunk=20)

kernel = get_kernel("fit_one_rchunk")

print(f"\nCompiling fit_one_rchunk at μ={sys_.n_rmu}, n_b={sys_.n_b}, "
      f"cr={knobs.get('chunk_r')}, bc={knobs.get('band_chunk')}")
print("(CPU backend — mesh 2×2; byte counts reflect CPU placement)")

meas = aot_measure(kernel, sys_, knobs, mesh)
print(f"\nmemory_analysis() totals:")
for k in ("argument", "output", "temp", "alias", "total"):
    print(f"  {k:10s} = {meas.get(k, 0)/1e6:>8.2f} MB")

# List dumped files
print(f"\nDumped files under {OUT}:")
for p in sorted(OUT.glob("*")):
    size_kb = p.stat().st_size / 1024
    print(f"  {size_kb:>7.0f} KB  {p.name}")
