"""Verify the sharded RPA-screening resolvent recipe reproduces disk W0-V."""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)
from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_ring_comm import (build_bse_ring_matvec_full, make_bse_shardings,
    build_realspace_random_transition_generator, build_density_snapshot_operator)
from bse.bse_feast import gmres_solve_sharded_jit, build_preconditioner_diagonal_sharded, _apply_shifted_matvec

inp = sys.argv[1]; restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=999, n_cond=999, mesh_xy=mesh, input_file=inp, inject_head=False)
data["W_R"] = data["W_q"]
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"]); nk = nkx*nky*nkz
n_rmu = int(data["V_q0"].shape[0]); nlog = int(data["n_rmu"])
sh = make_bse_shardings(mesh)
W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
V0 = np.asarray(jax.device_get(data["V_q0"])); T = W0 - V0
matvec = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=False, screening=True)
diag_h = build_preconditioner_diagonal_sharded(data, mesh, include_W=False, use_tda=False)
gen = build_realspace_random_transition_generator(mesh, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
snap = build_density_snapshot_operator(mesh, nkx, nky, nkz)
z = 0.0 + 0.0j
for nu0 in [int(np.argmax(np.linalg.norm(T[:, :nlog], axis=0))), 5, 200]:
    tcol = T[:, nu0]
    g = jnp.zeros((n_rmu,), jnp.float64).at[nu0].set(1.0)
    r = jax.device_put(jnp.broadcast_to(g[None, :, None], (1, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
    rhs = jax.lax.with_sharding_constraint(jnp.stack([f, -f], axis=0).astype(jnp.complex128), sh.X_full)
    x, kit = gmres_solve_sharded_jit(matvec, diag_h, z, rhs, data, max_iter=160, tol=1e-11)
    gres = float(jnp.linalg.norm(rhs - _apply_shifted_matvec(matvec, x, z, data)) / jnp.linalg.norm(rhs))
    s = jax.lax.with_sharding_constraint(x[0] + x[1], sh.X)
    wc = np.asarray(jax.device_get(snap(s, data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])[0]))
    rel = np.linalg.norm(wc[:nlog] - tcol[:nlog]) / np.linalg.norm(tcol[:nlog])
    mx = np.max(np.abs(wc[:nlog] - tcol[:nlog]))
    print(f"nu={nu0:3d} iters={int(kit)} gmres_resid={gres:.2e} relerr={rel:.4e} max|d|={mx:.3e} ||T||={np.linalg.norm(tcol[:nlog]):.3e}")
