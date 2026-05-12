"""T6b: Si 10³ @ 4 GB chooser pick (cr=64, bc=8).  Real runtime peak
from HLO memory-usage-report."""
from __future__ import annotations
import os, sys
dump_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/t6b_si10_4gb_hlo"
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

sd = SysDims(kgrid=(10, 10, 10), fft_grid=(24, 24, 24),
             n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24**3)
knobs = Knobs.of(chunk_r=64, band_chunk=8)

kernel = get_kernel("fit_one_rchunk")
fn = kernel.build_callable(sd, knobs, mesh)
specs = kernel.build_specs(sd, knobs, mesh)
compiled = fn.lower(*specs).compile()
m = compiled.memory_analysis()
if jax.process_index() == 0:
    print(f"Si 10³  cr=64  bc=8  (chooser pick @ 4 GB)")
    print(f"  memory_analysis():  arg={m.argument_size_in_bytes/1e9:.2f} "
          f"temp={m.temp_size_in_bytes/1e9:.2f} "
          f"out={m.output_size_in_bytes/1e9:.2f}  "
          f"sum={(m.argument_size_in_bytes+m.temp_size_in_bytes+m.output_size_in_bytes)/1e9:.2f} GB")
    print(f"  HLO report in {dump_dir}")
