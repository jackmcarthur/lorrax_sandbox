"""Compile fit_one_rchunk AOT for Si 10^3 and dump:
 1. The full HLO text (post-SPMD partitioning)
 2. The buffer-assignment / memory-usage report
 3. Greps for replicated buffers with any (q, μ, μ)-shape footprint

We specifically want to find every buffer whose HBM-resident size is
~n_k·μ²·16 bytes per device (3.69 GB for Si 10³ at nk=1000, μ=480) and
identify its sharding, its origin op, and why it isn't sharded.
"""
from __future__ import annotations

import os, sys, re
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.gw_jax import _maybe_init_jax_distributed
_maybe_init_jax_distributed()

import numpy as np
import jax
from jax.sharding import Mesh

from gw.aot_memory_model import SysDims, Knobs, get_kernel

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))

# Si 10^3 at the cr/bc the chooser picked at 8 GB: cr=192, bc=8
sd = SysDims(
    kgrid=(10, 10, 10), fft_grid=(24, 24, 24),
    n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24**3,
)
knobs = Knobs.of(chunk_r=192, band_chunk=8)

kernel = get_kernel("fit_one_rchunk")
fn = kernel.build_callable(sd, knobs, mesh)
specs = kernel.build_specs(sd, knobs, mesh)
lowered = fn.lower(*specs)
compiled = lowered.compile(
    compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})

hlo_text = compiled.as_text()
# write to disk
out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/fit_hlo_si10"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "fit_one_rchunk_si10.hlo"), "w") as f:
    f.write(hlo_text)

m = compiled.memory_analysis()
if jax.process_index() == 0:
    print(f"AOT bytes: temp={m.temp_size_in_bytes/1e9:.2f} GB  "
          f"arg={m.argument_size_in_bytes/1e9:.2f} GB  "
          f"out={m.output_size_in_bytes/1e9:.2f} GB  "
          f"alias={m.alias_size_in_bytes/1e9:.2f} GB  "
          f"total={(m.temp_size_in_bytes+m.argument_size_in_bytes+m.output_size_in_bytes-m.alias_size_in_bytes)/1e9:.2f} GB")
    print(f"HLO text written to {out_dir}/fit_one_rchunk_si10.hlo ({len(hlo_text)} bytes)")

    # Grep for any buffers of shape compatible with (nk, μ, μ)-size.
    # Si 10^3: nk=1000, μ=480 → 230,400,000 complex elements.
    # Per-device (sharded 1/4): 57,600,000 elements = ~922 MB complex128.
    # Fully replicated: 230,400,000 = ~3.69 GB complex128.
    # We search for numeric shape patterns matching: (1000,480,480),
    # (1000,240,480), (1000,480,240), (1000,240,240) — any μ split but
    # nk dim intact.
    patterns = [
        r'f64\[1000,480,480\]',
        r'c128\[1000,480,480\]',
        r'c128\[1000,240,480\]',
        r'c128\[1000,480,240\]',
        r'c128\[1000,240,240\]',
    ]
    for pat in patterns:
        hits = re.findall(pat, hlo_text)
        if hits:
            print(f"  Shape match '{pat}': {len(hits)} hits")

    # Find the N longest preallocated temp lines (buffer assignment)
    # — only available if we request the full buffer report.
    try:
        ba = compiled.as_text()  # HLO includes buffer assignment
        # Extract lines that look like allocations with GB sizes
        alloc_lines = [L for L in ba.split('\n')
                       if 'bytes' in L.lower() and 'allocated' in L.lower()]
        if alloc_lines:
            print("  Buffer-assignment lines (first 20):")
            for L in alloc_lines[:20]:
                print(f"    {L.strip()}")
    except Exception as e:
        print(f"  Could not inspect buffer assignment: {e}")
