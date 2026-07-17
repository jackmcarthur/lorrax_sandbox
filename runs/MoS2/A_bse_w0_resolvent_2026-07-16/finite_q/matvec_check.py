"""Does the sharded screening matvec compute the correct H_RPA^q at finite q?
Compare sharded build_bse_ring_matvec_full(screening) against a dense
H_RPA^q = [[D+V, V],[-V,-D-V]] built from the same rolled psi_c / eps_c / V_q.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from bse import bse_io
from bse.bse_feast import ensure_W_R
from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings

RESTART, INPUT = sys.argv[1], sys.argv[2]


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
    nlog = int(data["n_rmu"])
    n_pad = int(data["n_rmu_pad"])
    nc = int(data["n_cond_pad"]); nv = int(data["n_val_pad"])

    for q in [(0, 1, 0), (1, 1, 0)]:
        dq = dict(data)
        dq["V_q0"] = data["V_q_full"][:, :, q[0], q[1], q[2]]
        pcx = roll_k(data["psi_c_X"], q, nkx, nky, nkz)
        dq["psi_c_X"] = jax.lax.with_sharding_constraint(pcx, sh.psi_x)
        dq["psi_c_Y"] = jax.lax.with_sharding_constraint(pcx, sh.psi_y)
        dq["eps_c"] = roll_k(data["eps_c"], q, nkx, nky, nkz)
        ensure_W_R(dq, include_W=False)
        matvec = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=False, screening=True)

        key = jax.random.PRNGKey(3)
        X = (jax.random.normal(key, (1, nc, nv, nk)) + 1j * jax.random.normal(key, (1, nc, nv, nk)))
        Y = (jax.random.normal(jax.random.PRNGKey(4), (1, nc, nv, nk))
             + 1j * jax.random.normal(jax.random.PRNGKey(4), (1, nc, nv, nk)))
        Xf = jax.lax.with_sharding_constraint(jnp.stack([X, Y], 0), sh.X_full)
        HX = matvec(Xf, dq["psi_c_X"], dq["psi_c_Y"], dq["psi_v_X"], dq["psi_v_Y"],
                    dq["eps_c"], dq["eps_v"], dq["W_R"], dq["V_q0"])
        HX = np.asarray(jax.device_get(HX))

        # dense reference
        pc = np.asarray(jax.device_get(dq["psi_c_X"]))
        pv = np.asarray(jax.device_get(data["psi_v_X"]))
        ec = np.asarray(jax.device_get(dq["eps_c"]))
        ev = np.asarray(jax.device_get(data["eps_v"]))
        Vq = np.asarray(jax.device_get(dq["V_q0"]))
        Xn = np.asarray(jax.device_get(X))[0]  # (nc,nv,nk)
        Yn = np.asarray(jax.device_get(Y))[0]
        M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), pv)  # (k,c,v,mu)
        D = (ec[:, :, None] - ev[:, None, :]).transpose(1, 2, 0)  # (c,v,k)

        def Aop(Z):  # (c,v,k)
            DZ = D * Z
            S = np.einsum("kcvN,cvk->N", M, Z) / np.sqrt(nk)      # (mu,)
            U = Vq @ S                                            # (mu,)
            VX = np.einsum("kcvM,M->cvk", np.conj(M), U) / np.sqrt(nk)
            return DZ + VX
        def Vop(Z):
            S = np.einsum("kcvN,cvk->N", M, Z) / np.sqrt(nk)
            U = Vq @ S
            return np.einsum("kcvM,M->cvk", np.conj(M), U) / np.sqrt(nk)
        AX = Aop(Xn); VY = Vop(Yn)
        AY = Aop(Yn); VX = Vop(Xn)
        Xout = AX + VY
        Yout = -VX - AY
        ref = np.stack([Xout, Yout], 0)[:, None]  # (2,1,c,v,k)
        rel = np.linalg.norm(HX - ref) / np.linalg.norm(ref)
        print(f"q={q}: sharded-vs-dense matvec rel_err = {rel:.3e}")


if __name__ == "__main__":
    main()
