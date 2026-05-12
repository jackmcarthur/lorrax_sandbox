"""T6: dump Si 10³ fit_one_rchunk HLO at the 8 GB chooser pick (cr=356,
bc=20), compare each of my model's terms to the actual allocations in
the memory-usage-report.  Goal: find the 3-5 GB gap between prediction
7.75 GB and AOT 11.64 GB — figure out if it's bc-loop unrolling, a
persistent buffer I missed, or something else.
"""
from __future__ import annotations

import os, sys
dump_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/t6_si10_hlo"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={dump_dir} --xla_dump_hlo_as_text"
)
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

# The Si 10³ @ 8 GB chooser pick (current model):
sd = SysDims(
    kgrid=(10, 10, 10), fft_grid=(24, 24, 24),
    n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24**3,
)
knobs = Knobs.of(chunk_r=356, band_chunk=20)

kernel = get_kernel("fit_one_rchunk")
fn = kernel.build_callable(sd, knobs, mesh)
specs = kernel.build_specs(sd, knobs, mesh)
lowered = fn.lower(*specs)
compiled = lowered.compile()
m = compiled.memory_analysis()
if jax.process_index() == 0:
    print(f"Si 10³, cr=356, bc=20 — CURRENT chooser pick @ 8 GB")
    print(f"  AOT bytes (slop_factor DEFAULT = runtime peak):")
    print(f"    arg  = {m.argument_size_in_bytes/1e9:.2f} GB")
    print(f"    temp = {m.temp_size_in_bytes/1e9:.2f} GB")
    print(f"    out  = {m.output_size_in_bytes/1e9:.2f} GB")
    total = m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes
    print(f"    total= {total/1e9:.2f} GB")
    print()
    print(f"  XLA dumps in {dump_dir}")
    print("  Look at *-memory-usage-report.txt for the per-slot breakdown.")
