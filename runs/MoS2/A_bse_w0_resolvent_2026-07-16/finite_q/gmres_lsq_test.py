"""Is the custom GMRES normal-equations LSQ the finite-q failure?
Run a matrix-free GMRES on the finite-q shifted operator, one probe column,
comparing (a) normal equations solve(H^H H) [what bse_feast uses] vs
(b) numpy lstsq (QR) for the least-squares step.  Report true residuals.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse import bse_io
from bse.bse_feast import ensure_W_R, build_preconditioner_diagonal_sharded, _apply_shifted_matvec
from bse.bse_ring_comm import (build_bse_ring_matvec_full, make_bse_shardings,
                               build_realspace_random_transition_generator)

RESTART, INPUT = sys.argv[1], sys.argv[2]
Q = tuple(int(x) for x in sys.argv[3].split(","))


def roll_k(a, shift, nkx, nky, nkz):
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz) + tail)
    a = jnp.roll(a, shift=tuple(int(s) for s in shift), axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def main():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        RESTART, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=INPUT, inject_head=False, load_v_full=True)
    sh = make_bse_shardings(mesh)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_rmu = int(data["V_q0"].shape[0])
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

    # build RHS for one probe column nu=63
    nu = 63
    G = jnp.zeros((1, n_rmu)).at[0, nu].set(1.0)
    r = jax.device_put(jnp.broadcast_to(G[:, :, None], (1, n_rmu, nk)), sh.S)
    f = jax.lax.with_sharding_constraint(gen(r, dq["psi_c_X"], dq["psi_v_X"], dq["V_q0"]), sh.X)
    rhs = jax.lax.with_sharding_constraint(jnp.stack([f, -f], 0).astype(jnp.complex128), sh.X_full)

    z = 0.0 + 0.0j
    diag = np.asarray(jax.device_get(diag_full))
    m_inv = 1.0 / (z - diag)

    def applyH(x):  # (2,1,c,v,k) numpy -> numpy, computes (z-H)x
        xj = jax.device_put(jnp.asarray(x), sh.X_full)
        return np.asarray(jax.device_get(_apply_shifted_matvec(matvec, xj, z, dq)))

    b = np.asarray(jax.device_get(rhs))
    x0 = m_inv * b
    r0 = b - applyH(x0)
    beta = np.linalg.norm(r0)
    maxk = 120
    V = [r0 / beta]
    H = np.zeros((maxk + 1, maxk), dtype=complex)
    Z = []
    hist_ne, hist_qr = [], []
    for k in range(maxk):
        zk = m_inv * V[k]
        Z.append(zk)
        w = applyH(zk)
        for i in range(k + 1):
            H[i, k] = np.vdot(V[i], w)
            w = w - H[i, k] * V[i]
        H[k + 1, k] = np.linalg.norm(w)
        V.append(w / H[k + 1, k] if H[k + 1, k] != 0 else w)
        # both LSQ variants on H[:k+2,:k+1]
        Hk = H[:k + 2, :k + 1]
        g = np.zeros(k + 2, dtype=complex); g[0] = beta
        # normal equations (bse_feast style)
        lhs = Hk.conj().T @ Hk
        lhs = lhs + 1e-14 * np.trace(lhs).real / max(1, lhs.shape[0]) * np.eye(lhs.shape[0])
        y_ne = np.linalg.solve(lhs, Hk.conj().T @ g)
        # QR lstsq
        y_qr, *_ = np.linalg.lstsq(Hk, g, rcond=None)
        # true residuals
        def true_res(y):
            x = x0 + sum(y[j] * Z[j] for j in range(k + 1))
            return np.linalg.norm(b - applyH(x)) / np.linalg.norm(b)
        if k % 20 == 19 or k == maxk - 1:
            hist_ne.append((k + 1, true_res(y_ne)))
            hist_qr.append((k + 1, true_res(y_qr)))
    print(f"q={Q} nu={nu}  cond(H^H H at end)={np.linalg.cond(H[:maxk+1,:maxk].conj().T@H[:maxk+1,:maxk]):.2e}")
    print("  iter   true_res(normal-eq)   true_res(QR-lstsq)")
    for (k1, rne), (k2, rqr) in zip(hist_ne, hist_qr):
        print(f"  {k1:4d}   {rne:.3e}            {rqr:.3e}")


if __name__ == "__main__":
    main()
