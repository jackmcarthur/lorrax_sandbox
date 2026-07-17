"""Is the finite-q divergence the SVD-lstsq inside lax.scan?  Compare the
resolvent solve done via (a) lax.scan [apply_screening_resolvent_block] vs
(b) a plain Python loop over the SAME jitted gmres_solve_sharded_jit.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from bse import bse_io
from bse.bse_feast import gmres_solve_sharded_jit, _apply_shifted_matvec
from bse.bse_w_exact import (_build_rpa_resolvent, _select_compare_cols,
                             apply_screening_resolvent_block)
from bse.bse_ring_comm import make_bse_shardings

RESTART, INPUT = sys.argv[1], sys.argv[2]
Q = (0, 1, 0)


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
    nk = nkx*nky*nkz; nlog = int(data["n_rmu"]); n_rmu = int(data["V_q0"].shape[0])
    dq = dict(data)
    dq["V_q0"] = data["V_q_full"][:, :, Q[0], Q[1], Q[2]]
    pcx = roll_k(data["psi_c_X"], Q, nkx, nky, nkz)
    dq["psi_c_X"] = jax.lax.with_sharding_constraint(pcx, sh.psi_x)
    dq["psi_c_Y"] = jax.lax.with_sharding_constraint(pcx, sh.psi_y)
    dq["eps_c"] = jax.lax.with_sharding_constraint(roll_k(data["eps_c"], Q, nkx, nky, nkz), sh.eps)
    matvec, diag_h, gen, snapshot, shs = _build_rpa_resolvent(mesh, dq)

    W0 = np.asarray(jax.device_get(data["W_q"][:, :, Q[0], Q[1], Q[2]]))
    Vf = np.asarray(jax.device_get(data["V_q_full"][:, :, Q[0], Q[1], Q[2]]))
    T = W0 - Vf
    cols, _ = _select_compare_cols(T, nlog, 4, seed=0)
    n_pad = int(np.ceil(len(cols)/1)*1)

    # (a) scan engine
    Gm = np.zeros((n_pad, n_rmu)); [Gm.__setitem__((i, int(c)), 1.0) for i, c in enumerate(cols)]
    W_tile, resids = apply_screening_resolvent_block(
        Gm, 0.0+0j, dq, matvec, diag_h, gen, snapshot, shs, max_iter=200, tol=1e-10)
    rr = np.asarray(jax.device_get(resids)); print(f"(a) scan   max gmres_resid={rr[:len(cols)].max():.3e}")

    # (b) python loop over the same jitted gmres
    z = 0.0+0j
    resb = []
    for c in cols:
        G1 = jnp.zeros((1, n_rmu)).at[0, int(c)].set(1.0)
        r = jax.device_put(jnp.broadcast_to(G1[:, :, None], (1, n_rmu, nk)), shs.S)
        f = jax.lax.with_sharding_constraint(gen(r, dq["psi_c_X"], dq["psi_v_X"], dq["V_q0"]), shs.X)
        rhs = jax.lax.with_sharding_constraint(jnp.stack([f, -f], 0).astype(jnp.complex128), shs.X_full)
        rhs_i = rhs[:, 0][:, None]
        x, kf = gmres_solve_sharded_jit(matvec, diag_h, z, rhs_i, dq, max_iter=200, tol=1e-10)
        rt = float(jnp.linalg.norm(rhs_i - _apply_shifted_matvec(matvec, x, z, dq)) / jnp.linalg.norm(rhs_i))
        resb.append(rt)
    print(f"(b) loop   max gmres_resid={max(resb):.3e}   per-col={['%.1e'%v for v in resb]}")


if __name__ == "__main__":
    main()
