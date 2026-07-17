"""Dense finite-q chi0 -> W_q, bypassing GMRES, to lock the convention.

W_q = (I - V_q chi0_q)^{-1} V_q ; chi0_q = -2/Nk Σ_cvk M^q conj(M^q)/(e_c(k+q)-e_v(k)).
M^q(mu) = Σ_s conj(psi_c[k+shift](s,mu)) psi_v[k](s,mu).  Compare (W_q - V_q) vs
the on-disk (W0_qmunu - V_qmunu)[q_flat] tile.  Sweeps roll sign x phase.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from bse import bse_io
from common.symmetry_maps import kgrid_shift_map

RESTART, INPUT, CENTROIDS = sys.argv[1], sys.argv[2], sys.argv[3]


def load_cfrac(path, n):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                p = ln.split()
                rows.append([float(p[0]), float(p[1]), float(p[2])])
    s = np.asarray(rows, dtype=np.float64)
    out = np.zeros((n, 3)); out[:s.shape[0]] = s
    return out


def roll_k(a, shift, nkx, nky, nkz):
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz) + tail)
    a = np.roll(a, shift=(int(shift[0]), int(shift[1]), int(shift[2])), axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def main():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        RESTART, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=INPUT, inject_head=False, load_v_full=True)
    nlog = int(data["n_rmu"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    n_pad = int(data["n_rmu_pad"])
    cfrac = load_cfrac(CENTROIDS, n_pad)

    psi_c = np.asarray(jax.device_get(data["psi_c_X"]))   # (nk,nc,ns,mu_pad)
    psi_v = np.asarray(jax.device_get(data["psi_v_X"]))   # (nk,nv,ns,mu_pad)
    eps_c = np.asarray(jax.device_get(data["eps_c"]))     # (nk,nc)
    eps_v = np.asarray(jax.device_get(data["eps_v"]))     # (nk,nv)
    Vfull = np.asarray(jax.device_get(data["V_q_full"]))  # (mu,nu,nkx,nky,nkz)
    Wfull = np.asarray(jax.device_get(data["W_q"]))

    def chi0_dense(q, shift, phase):
        qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
        pc = roll_k(psi_c, shift, nkx, nky, nkz)
        ec = roll_k(eps_c, shift, nkx, nky, nkz)
        if phase:
            q_off = (-int(shift[0]), -int(shift[1]), -int(shift[2]))
            _, G = kgrid_shift_map(nkx, nky, nkz, q_off)
            ph = np.exp(-2j * np.pi * (G.astype(float) @ cfrac.T))   # (nk,n_pad)
            pc = pc * ph[:, None, None, :]
        # M^q[k,c,v,mu] = Σ_s conj(pc) pv
        M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), psi_v)          # (nk,nc,nv,mu)
        D = ec[:, :, None] - eps_v[:, None, :]                        # (nk,nc,nv)
        Mw = M / D[..., None]
        chi = -2.0 / nk * np.einsum("kcvm,kcvn->mn", Mw, np.conj(M))  # (mu,mu)
        return chi[:nlog, :nlog]

    def compare(q, shift, phase):
        qx, qy, qz = int(q[0]), int(q[1]), int(q[2])
        Vq = Vfull[:nlog, :nlog, qx, qy, qz]
        Tdisk = (Wfull[:nlog, :nlog, qx, qy, qz] - Vq)
        chi = chi0_dense(q, shift, phase)
        W = np.linalg.solve(np.eye(nlog) - Vq @ chi, Vq)
        T = W - Vq
        rel = np.linalg.norm(T - Tdisk) / np.linalg.norm(Tdisk)
        return rel

    # q=0 control
    print(f"q=(0,0,0) shift=0 phase=off dense rel_err = {compare((0,0,0),(0,0,0),False):.3e}")
    for q in [(0,1,0),(1,0,0),(1,1,0),(1,2,0)]:
        print(f"\n-- q={q} --")
        for phase in (False, True):
            for sign in (+1, -1):
                rs = (sign*q[0], sign*q[1], sign*q[2])
                try:
                    r = compare(q, rs, phase)
                    print(f"  shift={rs} phase={'on ' if phase else 'off'} dense rel_err = {r:.3e}")
                except Exception as e:
                    print(f"  shift={rs} phase={'on ' if phase else 'off'} ERROR {e}")


if __name__ == "__main__":
    main()
