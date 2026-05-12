"""Compile fit_one_rchunk AOT for Si 10^3 and dump XLA's per-buffer
memory-usage report.  XLA writes this if we set
XLA_FLAGS='--xla_dump_to=<dir> --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*'.
"""
from __future__ import annotations

import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
dump_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/fit_si10_xladump"
os.makedirs(dump_dir, exist_ok=True)
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={dump_dir} "
    "--xla_dump_hlo_as_text "
    "--xla_gpu_memory_limit_slop_factor=10000"
)
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.gw_jax import _maybe_init_jax_distributed
_maybe_init_jax_distributed()

import numpy as np
import jax
from jax.sharding import Mesh

from gw.aot_memory_model import SysDims, Knobs, get_kernel

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))

sd = SysDims(
    kgrid=(10, 10, 10), fft_grid=(24, 24, 24),
    n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24**3,
)
knobs = Knobs.of(chunk_r=192, band_chunk=8)

kernel = get_kernel("fit_one_rchunk")
fn = kernel.build_callable(sd, knobs, mesh)
specs = kernel.build_specs(sd, knobs, mesh)
lowered = fn.lower(*specs)
compiled = lowered.compile()

m = compiled.memory_analysis()
if jax.process_index() == 0:
    print(f"temp={m.temp_size_in_bytes/1e9:.2f} GB  "
          f"arg={m.argument_size_in_bytes/1e9:.2f} GB  "
          f"out={m.output_size_in_bytes/1e9:.2f} GB  "
          f"alias={m.alias_size_in_bytes/1e9:.2f} GB")
    print(f"XLA dumps in {dump_dir}")
