"""What does the standalone FFT query return for the ZCT shape?
Compare ZCT-query workspace vs what the real fit_one_rchunk kernel
allocates for ZCT (the 'temp' in the AOT analysis minus other known
temps)."""
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import (
    query_fft_workspace_bytes, query_fft_peak_bytes,
    _query_fft_memory_analysis,
)

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))


def q(label, shape, axes, spec_form):
    sh = NamedSharding(mesh, P(*spec_form))
    m = _query_fft_memory_analysis(shape, axes, sh, jnp.complex128)
    if jax.process_index() == 0:
        total = m['temp'] + m['argument'] + m['output'] - m['alias']
        print(f"{label}:")
        print(f"  shape={shape} axes={axes} spec={spec_form}")
        print(f"  arg={m['argument']/1e9:.2f} temp={m['temp']/1e9:.2f} "
              f"out={m['output']/1e9:.2f} alias={m['alias']/1e9:.2f} "
              f"total={total/1e9:.2f}")


# MoS2 3x3 ZCT: kgrid=(3,3,1), mu=640, cr=46080
q("MoS2_ZCT cr=46080", (3, 3, 1, 640, 46080), (0, 1, 2),
  (None, None, None, 'x', 'y'))
q("MoS2_ZCT cr=20000", (3, 3, 1, 640, 20000), (0, 1, 2),
  (None, None, None, 'x', 'y'))

# Si 10^3 ZCT: kgrid=(10,10,10), mu=480, cr=112
q("Si10_ZCT cr=112", (10, 10, 10, 480, 112), (0, 1, 2),
  (None, None, None, 'x', 'y'))
q("Si10_ZCT cr=400", (10, 10, 10, 480, 400), (0, 1, 2),
  (None, None, None, 'x', 'y'))

# Wfn FFT queries:
# MoS2 wfn: nk=9, nb=80, ns=2, fft=(24,24,80) sharded on bands
q("MoS2_wfn bc=80", (9, 80, 2, 24, 24, 80), (-3, -2, -1),
  (None, ('x', 'y'), None, None, None, None))
# Si 10^3 wfn: nk=1000, nb=60, ns=1, fft=(24,24,24) sharded on bands
q("Si10_wfn bc=4", (1000, 4, 1, 24, 24, 24), (-3, -2, -1),
  (None, ('x', 'y'), None, None, None, None))
q("Si10_wfn bc=60", (1000, 60, 1, 24, 24, 24), (-3, -2, -1),
  (None, ('x', 'y'), None, None, None, None))
