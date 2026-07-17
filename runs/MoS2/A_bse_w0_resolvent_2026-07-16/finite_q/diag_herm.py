import sys
import numpy as np
import jax
from jax.sharding import Mesh
from bse import bse_io

RESTART, INPUT = sys.argv[1], sys.argv[2]


def roll_k(a, shift, nkx, nky, nkz):
    tail = a.shape[1:]
    a = a.reshape((nkx, nky, nkz) + tail)
    a = np.roll(a, shift=tuple(int(s) for s in shift), axis=(0, 1, 2))
    return a.reshape((nkx * nky * nkz,) + tail)


def main():
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    data = bse_io.load_bse_data_from_restart_sharded(
        RESTART, n_val=10**9, n_cond=10**9, mesh_xy=mesh,
        input_file=INPUT, inject_head=False, load_v_full=True)
    nlog = int(data["n_rmu"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    Vfull = np.asarray(jax.device_get(data["V_q_full"]))
    Wfull = np.asarray(jax.device_get(data["W_q"]))
    psi_c = np.asarray(jax.device_get(data["psi_c_X"]))
    psi_v = np.asarray(jax.device_get(data["psi_v_X"]))
    eps_c = np.asarray(jax.device_get(data["eps_c"]))
    eps_v = np.asarray(jax.device_get(data["eps_v"]))

    for q in [(0, 0, 0), (0, 1, 0), (1, 1, 0)]:
        Vq = Vfull[:nlog, :nlog, q[0], q[1], q[2]]
        herm = np.linalg.norm(Vq - Vq.conj().T) / np.linalg.norm(Vq)
        Wq = Wfull[:nlog, :nlog, q[0], q[1], q[2]]
        wherm = np.linalg.norm(Wq - Wq.conj().T) / np.linalg.norm(Wq)
        # folded solve (Hermitian PD) per a few columns via P-space
        pc = roll_k(psi_c, q, nkx, nky, nkz)
        M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), psi_v)      # (k,c,v,mu)
        D = (eps_c[:, :, None] - eps_v[:, None, :])              # (k,c,v)
        Mf = M.reshape(-1, M.shape[-1])[:, :nlog]                # (P, mu)
        Df = D.reshape(-1)                                       # (P,)
        # K^A (P,P) = (1/nk) conj(M) V M^T  -> too big; use folded in mu-space:
        # W-V = -2/nk VM (D + 2K^A)^-1 M^T V ; equivalently chi0-Dyson (mu space).
        chi = -2.0 / nk * (Mf.conj().T / Df[None, :]) @ Mf.conj()  # wrong order guard
        # correct: chi[m,n] = -2/nk sum_P M[P,m]/D[P] conj(M[P,n])
        chi = -2.0 / nk * ((Mf / Df[:, None]).T @ np.conj(Mf))
        Wc = np.linalg.solve(np.eye(nlog) - Vq @ chi, Vq)
        T = Wc - Vq
        Tdisk = Wq - Vq
        rel = np.linalg.norm(T - Tdisk) / np.linalg.norm(Tdisk)
        Dmin = Df.min(); Dmax = Df.max()
        print(f"q={q}: ||V-V†||/||V||={herm:.2e}  ||W-W†||/||W||={wherm:.2e}  "
              f"D_q range=[{Dmin:.3f},{Dmax:.3f}]  chi0-Dyson rel_err={rel:.3e}")


if __name__ == "__main__":
    main()
