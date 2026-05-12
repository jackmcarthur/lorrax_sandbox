"""Probe: AOT-reported FFT memory vs actual runtime peak for the CrI3
fit_zeta in-loop wfn FFT.

Hypothesis: query_fft_peak_bytes (which reads compiled.memory_analysis())
under-counts the cuFFT plan workspace, because cuFFT allocates the plan
scratch at FIRST CALL via JAX's custom allocator and that doesn't show
up in the static analysis.

Run with:  LORRAX_NNODES=2 LORRAX_NGPU=4 lxrun python3 -u /tmp/probe_fft_memory.py

Reports: AOT-predicted peak, actual peak measured via jax.memory_stats()
before / after a single FFT call.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_B/src")

from runtime import set_default_env, init_jax_distributed
set_default_env()
init_jax_distributed()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

n_dev = len(jax.devices())
gx = int(n_dev ** 0.5)
while gx > 1 and n_dev % gx != 0:
    gx -= 1
mesh = Mesh(np.array(jax.devices()).reshape(gx, n_dev // gx), ['x', 'y'])

if jax.process_index() == 0:
    print(f"jax devices: {len(jax.devices())}, mesh: {gx} x {n_dev // gx}")

from common.fft_helpers import (
    query_fft_peak_bytes, make_sharded_ifftn_3d,
)

# CrI3 60Ry on 8 GPU: nk=36, ns=4, bispinor.  fft_grid: read from a WFN
from file_io import WFNReader
wfn = WFNReader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_nonbisp_baseline/WFN.h5")
fft_grid = tuple(int(d) for d in wfn.fft_grid)
n_rtot = fft_grid[0] * fft_grid[1] * fft_grid[2]
nk = 36  # CrI3 3x3 with TR-symmetry full BZ; matches chunker's nk_tot
if jax.process_index() == 0:
    print(f"CrI3 fft_grid={fft_grid}  n_rtot={n_rtot}  nk_full={nk}")

ns = 4  # bispinor lifted
sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None, None, None))

print()
print("=== AOT-predicted FFT peak (query_fft_peak_bytes) ===")
for bc in (4, 8, 16, 32):
    aot_bytes = query_fft_peak_bytes(
        input_shape=(nk, bc, ns, *fft_grid),
        fft_axes=(-3, -2, -1),
        sharding=sharding,
        dtype=jnp.complex128,
    )
    if jax.process_index() == 0:
        print(f"  bc={bc:3d}: AOT peak = {aot_bytes / 1e9:.2f} GB/rank")

print()
print("=== Actual peak via jax.memory_stats() before/after ONE FFT call ===")

import subprocess
def _stats():
    """Read GPU0 memory.used via nvidia-smi (works even when jax.memory_stats()
    returns None under cuda_malloc_async)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            text=True, timeout=5)
        return {'memory_used_bytes': int(out.strip()) * 1024 * 1024}
    except Exception as e:
        return {'memory_used_bytes': 0, 'error': str(e)}

bc = 8  # the chunker's chosen band_chunk for CrI3 8 GPU
local_ifftn = make_sharded_ifftn_3d(
    mesh, P(None, ('x','y'), None, None, None, None),
    P(None, ('x','y'), None, None, None, None),
)
jit_ifftn = jax.jit(local_ifftn, out_shardings=sharding)

# Allocate input as zeros on device.  Force it on device via device_put.
shape = (nk, bc, ns, *fft_grid)
if jax.process_index() == 0:
    print(f"  Test input shape = {shape}, dtype=complex128 ({np.prod(shape)*16/1e9:.2f} GB unsharded)")

jax.clear_caches()

s_before_alloc = _stats()
# Create sharded zeros directly (bypass un-sharded materialization on device 0)
from functools import partial
@partial(jax.jit, out_shardings=sharding)
def _make_sharded_zeros():
    return jnp.zeros(shape, dtype=jnp.complex128)
arr = _make_sharded_zeros()
arr.block_until_ready()
s_after_alloc = _stats()

# Trigger compile + first call (cuFFT plan workspace allocated here)
out = jit_ifftn(arr)
out.block_until_ready()
s_after_first_call = _stats()

# Second call (plan reused)
out2 = jit_ifftn(arr)
out2.block_until_ready()
s_after_second_call = _stats()

if jax.process_index() == 0:
    def _diff(label, a, b):
        k = 'memory_used_bytes'
        if k in a and k in b:
            print(f"  {label}: {a[k]/1e9:.2f} → {b[k]/1e9:.2f}  (Δ={(b[k]-a[k])/1e9:+.2f} GB)")
    print(f"\n  After input device_put:")
    _diff("alloc", s_before_alloc, s_after_alloc)
    print(f"\n  After first FFT call (cuFFT plan workspace allocated here):")
    _diff("first-call", s_after_alloc, s_after_first_call)
    print(f"\n  After second FFT call (plan reused, no new alloc expected):")
    _diff("second-call", s_after_first_call, s_after_second_call)

    # Compare to AOT prediction at bc=8
    aot8 = query_fft_peak_bytes(
        input_shape=(nk, 8, ns, *fft_grid),
        fft_axes=(-3, -2, -1),
        sharding=sharding, dtype=jnp.complex128,
    )
    runtime_peak = s_after_second_call.get('peak_bytes_in_use', 0)
    print(f"\n  AOT prediction (bc=8): {aot8/1e9:.2f} GB")
    print(f"  Runtime peak:           {runtime_peak/1e9:.2f} GB")
    print(f"  Ratio:                  {runtime_peak / max(aot8, 1):.2f}x")
