"""Phase-2 multi-device differential (two-step): leg '1x1' (single process,
1 GPU) writes eigenvalues; leg 'full' (4 ranks, 2x2 mesh) recomputes and
compares. Mesh-vs-mesh at EACH bs isolates the stack matvec's device-count
invariance from the block-Lanczos solver's own quality (bs=4 known-suspect:
transposed-beta / final-slot, solver-program P1)."""
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src")
tag = sys.argv[2]
if tag == "full":
    from runtime import init_jax_distributed
    init_jax_distributed()

import jax
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh

from bse import bse_io
from bse.bse_ring_comm import create_mesh_2d
from bse.bse_lanczos import solve_bse_sharded

inp = sys.argv[1]
WORK = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp_phase2"
restart = bse_io._find_restart_file(inp)

mesh = (Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
        if tag == "1x1" else create_mesh_2d())
data = bse_io.load_bse_data_from_restart_sharded(
    restart, n_val=2, n_cond=2, mesh_xy=mesh, input_file=inp)

for bs in (1, 4):
    evals, _, _ = solve_bse_sharded(
        data, mesh, n_eig=4, max_iter=20, block_size=bs, include_W=True)
    ev = np.sort(np.asarray(evals)[:4])
    if jax.process_index() == 0:
        print(f"[multidev] mesh={tag} ({mesh.devices.shape}) bs={bs}: {ev}")
        if tag == "1x1":
            np.savetxt(f"{WORK}/evals_1x1_bs{bs}.txt", ev)
        else:
            ref = np.loadtxt(f"{WORK}/evals_1x1_bs{bs}.txt")
            d = float(np.max(np.abs(ev - ref)))
            print(f"[multidev] bs={bs}: max|1x1 - full| = {d:.3e}")
            assert d < 5e-6, f"mesh-dependent at bs={bs}: {d}"

if jax.process_index() == 0 and tag == "full":
    print("[multidev] PASS: stack matvec device-count invariant at bs=1 and bs=4")
