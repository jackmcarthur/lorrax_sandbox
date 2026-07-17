"""A/B: re-based ladder rankcut-1e-4 physical error, WRAPPED vs UNWRAPPED lab
continuation in the interpolant (the vq_loo rk trap). q0=2 (wrap-affected)."""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from proto1_prep import Fixture, relF, truncR_weights

fx = Fixture("MoS2_3x3")
C_q = fx.build_Cq()
kg = fx.kgrid
Rall = np.array([[i, j, 0] for i in range(kg[0]) for j in range(kg[1])])
Rw = ((Rall + kg // 2) % kg) - (kg // 2)
Rsort = Rw[np.argsort(np.sqrt(np.einsum("ri,ij,rj->r", Rw, fx.adot, Rw)))]

def solve_rc(C, Z, lam):
    Ch = 0.5 * (C + C.conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    sinv = np.where(s > lam * s[0], 1.0 / np.where(s > lam * s[0], s, 1), 0.0)
    return (Vh.conj().T * sinv) @ (U.conj().T @ Z)

q0 = 2
train = [q for q in range(fx.nq) if q != q0]
x = fx.gap_window_pairs(q0, 3, 3)
v, nn = fx.vq(q0)
def B(Mg):
    A = Mg[:, :nn] * np.sqrt(v[:nn])[None, :]
    return np.conj(A) @ A.T
B_true = B(x @ fx.ZG[q0])

for conv in ("wrapped", "unwrapped"):
    t0 = time.time()
    qv = fx.qfr if conv == "wrapped" else fx.qfr_raw
    w = truncR_weights(qv[train], qv[q0], Rsort[:7])
    C0 = np.tensordot(w, C_q[train], axes=(0, 0))
    Z0 = np.zeros((fx.n_mu, fx.n_rtot), dtype=np.complex128)
    for wi, q in zip(w, train):
        box = np.zeros((fx.n_mu, fx.n_rtot), dtype=np.complex128)
        fi = fx.flat_idx(fx.gvec[q]); n = int(fx.ngk[q])
        box[:, fi[:n]] = fx.ZG[q][:, :n]
        R = np.fft.ifftn(box.reshape(fx.n_mu, fx.nx, fx.ny, fx.nz),
                         axes=(1, 2, 3), norm="backward").reshape(fx.n_mu, fx.n_rtot)
        zr = R * np.exp(2j * np.pi * (fx.rfrac @ qv[q]))[None, :]
        Z0 += wi * (C_q[q] @ zr)
    z0 = solve_rc(C0, Z0, 1e-4)
    ph = np.exp(-2j * np.pi * (fx.rfrac @ qv[q0]))
    box = np.fft.fftn((z0 * ph[None, :]).reshape(fx.n_mu, fx.nx, fx.ny, fx.nz),
                      axes=(1, 2, 3), norm="backward").reshape(fx.n_mu, fx.n_rtot)
    fi = fx.flat_idx(fx.gvec[q0]); n = int(fx.ngk[q0])
    zt0 = np.zeros((fx.n_mu, fx.ngkmax), dtype=np.complex128)
    zt0[:, :n] = box[:, fi[:n]]
    print(f"{conv:>9s} q0={q0}: ladder rankcut 1e-4 B relF = "
          f"{relF(B(x @ zt0), B_true):.3e}  ({time.time()-t0:.0f}s)", flush=True)
