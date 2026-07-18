"""Reproduce the bs=4 TDA block-Lanczos ghost (solver P1) on the gnppm fixture,
and probe whether the final-slot-overwrite is the lever.  Read-only vs the tree."""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)

RESTART, INPUT = sys.argv[1], sys.argv[2]
from bse.bse_io import load_bse_data_from_restart_sharded
from bse.bse_lanczos import solve_bse_sharded

mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(RESTART, n_val=2, n_cond=2, mesh_xy=mesh,
                                          input_file=INPUT)
# dense TDA reference (from PHASE2_LOG B1 table lowest-4): 0.008189 0.008317 0.010682 0.010763
print("dense TDA ref lowest4: [0.008189 0.008317 0.010682 0.010763]")
for bs in (1, 4):
    ev, evec, nit = solve_bse_sharded(
        data, mesh, n_eig=4, max_iter=48, include_W=True, block_size=bs,
        n_reorth=48, rtol=0.0)
    ev = np.sort(np.asarray(jax.device_get(ev)).real)
    print(f"bs={bs} n_eig=4 lowest4: {ev}")
