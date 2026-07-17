import sys
import numpy as np
sys.path.insert(0, ".")
from proto1_prep import Fixture, relF
fx = Fixture("MoS2_3x3")
q0 = 1
x = fx.gap_window_pairs(q0, 3, 3)
Mg_fit = x @ fx.ZG[q0]
rows = np.empty((fx.nk, 3, 3, fx.n_rtot), dtype=np.complex128)
for k in range(fx.nk):
    kq, _ = fx.kq_index(k, q0)
    ukq = fx.u_grid(kq, nbmax=fx.nb); uk = fx.u_grid(k, nbmax=fx.nb)
    rows[k] = np.einsum("csr,vsr->cvr", np.conj(ukq[fx.nv:fx.nv+3]), uk[fx.nv-3:fx.nv])
rows = rows.reshape(-1, fx.n_rtot)
# scale via centroid norm ratio and via lstsq
sc_norm = np.linalg.norm(x) / np.linalg.norm(rows[:, fx.rmu_flat])
num = np.vdot(rows[:, fx.rmu_flat], x); sc_fit = num / np.vdot(rows[:, fx.rmu_flat], rows[:, fx.rmu_flat])
print("sc_norm=", sc_norm, " sc_fit=", sc_fit, " |sc_fit|=", abs(sc_fit))
print("rows@centroids vs x resid (sc_fit):",
      relF(sc_fit * rows[:, fx.rmu_flat], x))
Mg_ex = fx.to_sphere(rows, q0) * sc_fit
v, n = fx.vq(q0)
def B(Mg):
    A = Mg[:, :n] * np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T
Bf, Be = B(Mg_fit), B(Mg_ex)
print("||B_fit||=", np.linalg.norm(Bf), " ||B_exact||=", np.linalg.norm(Be))
print("relF(B_fit, B_exact) =", relF(Bf, Be))
s = np.vdot(Be.ravel(), Bf.ravel()) / np.vdot(Be.ravel(), Be.ravel())
print("after scalar align:", relF(Bf, s * Be), " scale=", s)
print("Mg rel err (fit vs exact, F):", relF(Mg_fit[:, :n], Mg_ex[:, :n]))
print("diag(B) ratios (first 6):", (np.diag(Bf)[:6] / np.diag(Be)[:6]).real)
