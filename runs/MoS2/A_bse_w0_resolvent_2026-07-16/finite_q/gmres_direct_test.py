import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from bse import bse_io
from bse.bse_feast import (ensure_W_R, build_preconditioner_diagonal_sharded,
                           _apply_shifted_matvec, gmres_solve_sharded_jit)
from bse.bse_ring_comm import (build_bse_ring_matvec_full, make_bse_shardings,
                               build_realspace_random_transition_generator)

RESTART, INPUT = sys.argv[1], sys.argv[2]
Q = tuple(int(x) for x in sys.argv[3].split(","))


def roll_k(a, shift, nkx, nky, nkz):
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz)+tail)
    return jnp.roll(a, shift=tuple(int(s) for s in shift), axis=(0, 1, 2)).reshape((nkx*nky*nkz,)+tail)


def main():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        RESTART, n_val=10**9, n_cond=10**9, mesh_xy=mesh, input_file=INPUT,
        inject_head=False, load_v_full=True)
    sh = make_bse_shardings(mesh)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx*nky*nkz; n_rmu = int(data["V_q0"].shape[0])
    nc = int(data["n_cond_pad"]); nv = int(data["n_val_pad"])
    dq = dict(data)
    dq["V_q0"] = data["V_q_full"][:, :, Q[0], Q[1], Q[2]]
    pcx = roll_k(data["psi_c_X"], Q, nkx, nky, nkz)
    dq["psi_c_X"] = jax.lax.with_sharding_constraint(pcx, sh.psi_x)
    dq["psi_c_Y"] = jax.lax.with_sharding_constraint(pcx, sh.psi_y)
    dq["eps_c"] = jax.lax.with_sharding_constraint(roll_k(data["eps_c"], Q, nkx, nky, nkz), sh.eps)
    ensure_W_R(dq, include_W=False)
    matvec = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=False, screening=True)
    diag_full = build_preconditioner_diagonal_sharded(dq, mesh, include_W=False, use_tda=False)
    gen = build_realspace_random_transition_generator(mesh, nkx, nky, nkz, nc, nv)
    nu = 63
    G = jnp.zeros((1, n_rmu)).at[0, nu].set(1.0)
    r = jax.device_put(jnp.broadcast_to(G[:, :, None], (1, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(gen(r, dq["psi_c_X"], dq["psi_v_X"], dq["V_q0"]), sh.X)
    rhs = jax.lax.with_sharding_constraint(jnp.stack([f, -f], 0).astype(jnp.complex128), sh.X_full)
    rhs_i = rhs[:, 0][:, None]  # (2,1,c,v,k): one probe column, matvec batch axis
    z = 0.0 + 0.0j
    x, k_final = gmres_solve_sharded_jit(matvec, diag_full, z, rhs_i, dq, max_iter=200, tol=1e-10)
    r_true = rhs_i - _apply_shifted_matvec(matvec, x, z, dq)
    resid = float(jnp.linalg.norm(r_true) / jnp.linalg.norm(rhs_i))
    print(f"q={Q} nu={nu}: k_final={int(k_final)}  true_resid={resid:.3e}")


if __name__ == "__main__":
    main()
