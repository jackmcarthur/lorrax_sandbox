"""Drill into the Si 10^3 / 4 GB-budget fit_one_rchunk AOT peak — dump
the top buffers so we can see what dominates the 13.95 GiB peak vs
the heuristic's 3.88 GiB prediction.
"""
from __future__ import annotations

import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.gw_jax import _maybe_init_jax_distributed
_maybe_init_jax_distributed()

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from common.load_wfns import Meta
from gw.gw_init import compute_optimal_chunks
from gw.aot_memory_model import SysDims, MeshSpec, Knobs, get_kernel

devs = jax.devices()
assert len(devs) >= 4
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))

# Si 10x10x10
nk, mu, nb, ns = 1000, 480, 60, 1
fft_grid = (24, 24, 24)
kgrid = (10, 10, 10)

meta = Meta(
    rank=0, n_proc=4, b_id_0=0, b_id_1=0, b_id_2=0, b_id_3=nb, b_id_4=nb,
    fft_grid=fft_grid, cell_volume=1.0,
    n_rtot=fft_grid[0] * fft_grid[1] * fft_grid[2],
    n_rmu=mu,
    npol=1, nfreq=1,
    nspin=ns, nspinor=ns, nspinor_wfnfile=ns,
    nkx=kgrid[0], nky=kgrid[1], nkz=kgrid[2], nk_tot=nk,
    nbnd_jax=nb, n_rtot_jax=fft_grid[0] * fft_grid[1] * fft_grid[2], n_rmu_jax=mu,
)

for budget_gb in (4.0, 8.0, 16.0, 35.0):
    print(f"\n==================  Si 10³  {budget_gb} GB  ==================")
    chunks = compute_optimal_chunks(
        meta, mesh, memory_budget_gb=budget_gb, verbose=False)
    cr = int(chunks['chunk_r'])
    bc = int(chunks['band_chunk'])
    print(f"chooser picks: cr={cr}  band_chunk={bc}  "
          f"pred_peak_gb={chunks['memory_estimate']['peak_estimate_gb']:.2f}")

    fit_kernel = get_kernel("fit_one_rchunk")
    sys_dims = SysDims(
        kgrid=kgrid, fft_grid=fft_grid,
        n_rmu=mu, n_s=ns, n_b=nb, n_b_sum=2 * nb,
        n_r=fft_grid[0] * fft_grid[1] * fft_grid[2],
    )
    knobs = Knobs.of(chunk_r=cr, band_chunk=bc)
    # Build & AOT-compile.
    fn = fit_kernel.build_callable(sys_dims, knobs, mesh)
    specs = fit_kernel.build_specs(sys_dims, knobs, mesh)
    lowered = fn.lower(*specs)
    compiled = lowered.compile(
        compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    m = compiled.memory_analysis()
    c = compiled.cost_analysis()
    total = (m.temp_size_in_bytes + m.argument_size_in_bytes
             + m.output_size_in_bytes - m.alias_size_in_bytes)
    if jax.process_index() == 0:
        print(f"  AOT bytes: temp={m.temp_size_in_bytes/1e9:.2f} GB  "
              f"arg={m.argument_size_in_bytes/1e9:.2f} GB  "
              f"out={m.output_size_in_bytes/1e9:.2f} GB  "
              f"alias={m.alias_size_in_bytes/1e9:.2f} GB  "
              f"total={total/1e9:.2f} GB")
