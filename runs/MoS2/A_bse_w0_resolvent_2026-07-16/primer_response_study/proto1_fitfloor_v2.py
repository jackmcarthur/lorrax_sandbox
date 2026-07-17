"""Fit floor under the physical metric, with WFN u's block-rotated into the
psi_full_y gauge (psi is the QP-rotated set; C/zeta are invariant under the
within-window rotation, so the fit's faithful non-ISDF reference is the
rotated-orbital pair rows)."""
import sys
import numpy as np
sys.path.insert(0, ".")
from proto1_prep import Fixture, relF, polar
fx = Fixture("MoS2_3x3")

# per-k alignment: u_rot[N] = sum_n ug[n] T[n,N], T = polar(<ug_n|psi_N>_mu)
T_k, sc_k, resid_k = [], [], []
ug_all = []
for k in range(fx.nk):
    ug = fx.u_grid(k, nbmax=fx.nb)
    # least-squares unitary estimate: psi ~= sc * ug W on the centroid set;
    # the centroid Gram M = ug^H ug is NOT ~I, so polar(O) is biased —
    # solve W = M^-1 O then unitarize.
    ugc = ug[:, :, fx.rmu_flat].reshape(fx.nb, -1)
    psc = fx.psi[k].reshape(fx.nb, -1)
    M = np.conj(ugc) @ ugc.T
    O = np.conj(ugc) @ psc.T
    T = polar(np.linalg.solve(M, O))[0]
    ur = np.einsum("nN,nsr->Nsr", T, ug)
    urc = ur[:, :, fx.rmu_flat]
    sc = np.vdot(urc, fx.psi[k]) / np.vdot(urc, urc)
    resid = np.linalg.norm(sc * urc - fx.psi[k]) / np.linalg.norm(fx.psi[k])
    T_k.append(T); sc_k.append(sc); resid_k.append(resid)
    ug_all.append(ur * sc)
print("post-alignment per-k resid:", " ".join(f"{r:.2e}" for r in resid_k))

NV, NC = 3, 3
res = []
for q0 in range(fx.nq):
    x = fx.gap_window_pairs(q0, NV, NC)
    Mg_fit = x @ fx.ZG[q0]
    rows = np.empty((fx.nk, NC, NV, fx.n_rtot), dtype=np.complex128)
    for k in range(fx.nk):
        kq, _ = fx.kq_index(k, q0)
        rows[k] = np.einsum("csr,vsr->cvr",
                            np.conj(ug_all[kq][fx.nv:fx.nv+NC]),
                            ug_all[k][fx.nv-NV:fx.nv])
    Mg_ex = fx.to_sphere(rows.reshape(-1, fx.n_rtot), q0)
    v, n = fx.vq(q0)
    def B(Mg):
        A = Mg[:, :n] * np.sqrt(v[:n])[None, :]
        return np.conj(A) @ A.T
    Bf, Be = B(Mg_fit), B(Mg_ex)
    res.append((relF(Bf, Be), relF(Mg_fit[:, :n], Mg_ex[:, :n])))
    print(f"q0={q0}: fit-floor B relF={res[-1][0]:.3e}  Mg relF={res[-1][1]:.3e}")
print(f"MEDIAN fit-floor: B {np.median([r[0] for r in res]):.3e}  "
      f"Mg {np.median([r[1] for r in res]):.3e}")
