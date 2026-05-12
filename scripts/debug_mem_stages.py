"""Print each stage cost + AOT total + argument/temp/output breakdown to
see where the heuristic mispredicts the fit_one_rchunk kernel peak."""
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
from gw.aot_memory_model import SysDims, Knobs, get_kernel

devs = jax.devices()
assert len(devs) >= 4
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))


def run(name, nk, mu, nb, ns, fft_grid, kgrid, budget_gb):
    nx, ny, nz = fft_grid
    meta = Meta(
        rank=0, n_proc=4, b_id_0=0, b_id_1=0, b_id_2=0, b_id_3=nb, b_id_4=nb,
        fft_grid=fft_grid, cell_volume=1.0,
        n_rtot=nx * ny * nz, n_rmu=mu,
        npol=1, nfreq=1,
        nspin=ns, nspinor=ns, nspinor_wfnfile=ns,
        nkx=kgrid[0], nky=kgrid[1], nkz=kgrid[2], nk_tot=nk,
        nbnd_jax=nb, n_rtot_jax=nx * ny * nz, n_rmu_jax=mu,
    )
    chunks = compute_optimal_chunks(meta, mesh, memory_budget_gb=budget_gb, verbose=False)
    if jax.process_index() != 0:
        return
    print(f"\n=== {name} @ {budget_gb} GB ===")
    print(f"  cr={chunks['chunk_r']} bc={chunks['band_chunk']} "
          f"q_chunk={chunks['q_chunk']} q_gather={chunks['q_gather']}")
    print(f"  bottleneck: {chunks['memory_estimate']['bottleneck']}")
    print(f"  overall_peak_gb: {chunks['memory_estimate']['peak_estimate_gb']:.2f}")
    print(f"  limit_info (per-stage GB):")
    for k, v in chunks['memory_estimate']['limit_info'].items():
        print(f"    {k:8s}: {v:.2f}")

    # AOT fit_one_rchunk breakdown.
    fit = get_kernel("fit_one_rchunk")
    sd = SysDims(kgrid=tuple(kgrid), fft_grid=tuple(fft_grid),
                 n_rmu=int(mu), n_s=int(ns), n_b=int(nb), n_b_sum=int(2 * nb),
                 n_r=nx * ny * nz)
    knobs = Knobs.of(chunk_r=int(chunks['chunk_r']),
                     band_chunk=int(chunks['band_chunk']))
    fn = fit.build_callable(sd, knobs, mesh)
    specs = fit.build_specs(sd, knobs, mesh)
    lowered = fn.lower(*specs)
    compiled = lowered.compile(
        compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    m = compiled.memory_analysis()
    print(f"  AOT fit_one_rchunk bytes:")
    print(f"    argument: {m.argument_size_in_bytes/1e9:.2f}")
    print(f"    temp    : {m.temp_size_in_bytes/1e9:.2f}")
    print(f"    output  : {m.output_size_in_bytes/1e9:.2f}")
    print(f"    alias   : {m.alias_size_in_bytes/1e9:.2f}")
    total = (m.temp_size_in_bytes + m.argument_size_in_bytes
             + m.output_size_in_bytes - m.alias_size_in_bytes)
    print(f"    TOTAL   : {total/1e9:.2f}")


# Danger case: Si 10³ @ 8 GB under-predicts
run("Si_10x10x10", 1000, 480, 60, 1, (24, 24, 24), (10, 10, 10), 8.0)
# Over-predict case: MoS2 @ 16 GB
run("MoS2_3x3", 9, 640, 80, 2, (24, 24, 80), (3, 3, 1), 16.0)
