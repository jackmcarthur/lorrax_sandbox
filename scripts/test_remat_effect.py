"""Sanity: does disable_remat=False actually reduce the AOT total?"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.gw_jax import _maybe_init_jax_distributed
_maybe_init_jax_distributed()

import numpy as np
import jax
from jax.sharding import Mesh

from gw.aot_memory_model import SysDims, Knobs, get_kernel, aot_measure

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))

sd = SysDims(kgrid=(10, 10, 10), fft_grid=(24, 24, 24),
             n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24**3)

for cr, bc in [(64, 8), (356, 20)]:
    kernel = get_kernel("fit_one_rchunk")
    knobs = Knobs.of(chunk_r=cr, band_chunk=bc)
    m_with_remat = aot_measure(kernel, sd, knobs, mesh, disable_remat=False)
    m_no_remat   = aot_measure(kernel, sd, knobs, mesh, disable_remat=True)
    if jax.process_index() == 0:
        print(f"\nSi 10³  cr={cr}  bc={bc}")
        print(f"  with remat     : total={m_with_remat['total']/1e9:.2f} GB "
              f"(temp {m_with_remat['temp']/1e9:.2f})")
        print(f"  without remat  : total={m_no_remat['total']/1e9:.2f} GB "
              f"(temp {m_no_remat['temp']/1e9:.2f})")
