"""Does jnp.linalg.lstsq on the PADDED Hessenberg (fixed jit shape) recover the
active least-squares?  Build the real finite-q Arnoldi H, then compare:
  (a) np active lstsq         [the known-good reference]
  (b) jnp padded lstsq rcond=None
  (c) jnp padded lstsq rcond=1e-12
  (d) jnp padded lstsq rcond=1e-10
via TRUE residual b-(z-H)x.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from bse import bse_io
from bse.bse_feast import ensure_W_R, build_preconditioner_diagonal_sharded, _apply_shifted_matvec
from bse.bse_ring_comm import (build_bse_ring_matvec_full, make_bse_shardings,
                               build_realspace_random_transition_generator)

RESTART, INPUT = sys.argv[1], sys.argv[2]
Q = (0, 1, 0)
MAXK = 200


def roll_k(a, shift, nkx, nky, nkz):
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz) + tail)
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
    z = 0.0 + 0.0j
    m_inv = np.asarray(jax.device_get(1.0/(z-diag_full)))

    def applyH(x):
        return np.asarray(jax.device_get(_apply_shifted_matvec(matvec, jax.device_put(jnp.asarray(x), sh.X_full), z, dq)))

    b = np.asarray(jax.device_get(rhs))
    x0 = m_inv*b; r0 = b-applyH(x0); beta = np.linalg.norm(r0)
    g = np.zeros(MAXK+1, complex); g[0] = beta

    def tres(y):
        x = x0 + sum(y[j]*Z[j] for j in range(len(Z)))
        return np.linalg.norm(b-applyH(x))/np.linalg.norm(b)

    REORTH = True
    V = [r0/beta]; Hpad = np.zeros((MAXK+1, MAXK), complex); Z = []
    checkpts = [10, 20, 30, 50, 80, 120, 199]
    for k in range(MAXK):
        zk = m_inv*V[k]; Z.append(zk); w = applyH(zk)
        for i in range(k+1):
            Hpad[i, k] = np.vdot(V[i], w); w = w - Hpad[i, k]*V[i]
        if REORTH:  # second classical Gram-Schmidt pass
            for i in range(k+1):
                corr = np.vdot(V[i], w); Hpad[i, k] += corr; w = w - corr*V[i]
        Hpad[k+1, k] = np.linalg.norm(w); V.append(w/Hpad[k+1, k])
        if k in checkpts:
            # orthogonality loss of V[0..k+1]
            Vm = np.stack(V, 0).reshape(len(V), -1)
            ortho = np.linalg.norm(Vm.conj() @ Vm.T - np.eye(len(V)))
            # padded jnp lstsq on CURRENT partial H (active cols 0..k)
            Hj = jnp.asarray(Hpad); gj = jnp.asarray(g)
            yj = np.asarray(jax.device_get(jnp.linalg.lstsq(Hj, gj, rcond=None)[0]))
            proj = np.linalg.norm(g - Hpad @ yj)/beta
            print(f"k={k:4d}: ||V^H V - I||={ortho:.2e}  padded-lstsq proj_res={proj:.3e}  true_res={tres(yj):.3e}")


if __name__ == "__main__":
    main()
