"""Same sweep as validate_memory_model.py, but using the AOT-calibrated
chooser (``choose_chunks_heuristic``) instead of the linear stage-cost
heuristic in ``compute_optimal_chunks``.  Confirms fit-based predictions
match AOT-measured peaks within ~3%."""
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

from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs, get_kernel, aot_measure,
)
from gw.aot_memory_model.chooser import (
    choose_chunks_heuristic, choose_chunks_analytic,
)


devs = jax.devices()
assert len(devs) >= 4
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))
p_x = p_y = 2


SYSTEMS = {
    "MoS2_3x3":    (9,    640, 80, 2, (24, 24, 80), (3,  3,  1)),
    "Si_10x10x10": (1000, 480, 60, 1, (24, 24, 24), (10, 10, 10)),
}
BUDGETS_GB = [4, 8, 16, 28, 35]


def run(name, spec, bgt):
    nk, mu, nb, ns, fft_grid, kgrid = spec
    sd = SysDims(
        kgrid=tuple(kgrid), fft_grid=tuple(fft_grid),
        n_rmu=int(mu), n_s=int(ns), n_b=int(nb), n_b_sum=int(2 * nb),
        n_r=fft_grid[0] * fft_grid[1] * fft_grid[2],
    )
    ms = MeshSpec(p_x=p_x, p_y=p_y)
    try:
        choice = choose_chunks_heuristic(sd, ms, budget_bytes=bgt * 1e9)
    except Exception as e:
        return dict(name=name, bgt=bgt, error=repr(e))

    fit_kernel = get_kernel("fit_one_rchunk")
    knobs = Knobs.of(chunk_r=int(choice.chunk_r),
                     band_chunk=int(choice.band_chunk))
    meas = aot_measure(fit_kernel, sd, knobs, mesh)
    pred_gb = choice.peak_bytes / 1e9
    aot_gb = meas['total'] / 1e9
    ratio = aot_gb / pred_gb if pred_gb > 0 else float('nan')
    return dict(name=name, bgt=bgt,
                cr=int(choice.chunk_r), bc=int(choice.band_chunk),
                pred=pred_gb, aot=aot_gb, ratio=ratio)


rows = []
for name, spec in SYSTEMS.items():
    for b in BUDGETS_GB:
        rows.append(run(name, spec, b))

if jax.process_index() == 0:
    print(f"\n{'system':<14}{'bgt':>5}{'cr':>7}{'bc':>4}"
          f"{'pred':>10}{'aot':>10}{'ratio':>7}")
    print("-" * 57)
    for r in rows:
        if 'error' in r:
            print(f"{r['name']:<14}{r['bgt']:>5} {r['error']}")
        else:
            print(f"{r['name']:<14}{r['bgt']:>5}{r['cr']:>7}{r['bc']:>4}"
                  f"{r['pred']:>10.2f}{r['aot']:>10.2f}{r['ratio']:>7.2f}")
