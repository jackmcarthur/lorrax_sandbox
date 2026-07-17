"""Circular sanity for the v2 sphere zcheck plumbing: replace the WFN pair
rows by their ISDF-fitted reconstruction A_fit = X zeta_recon (rows exactly
in the fit ansatz); the same sphere-projected solve must then return the
stored zeta~ to ~solver precision. Closes => plumbing OK and the v2
discrepancy is production-RHS vs naive full-grid pair rows; fails =>
my v2 machinery is buggy."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                   "A_bse_w0_resolvent_2026-07-16/primer_response_study")
from proto1_prep import Fixture, relF

fx = Fixture("MoS2_3x3")
C_q = fx.build_Cq()
q = 1
nsl = int(fx.ngk[q])
fi_q = fx.flat_idx(fx.gvec[q])[:nsl]
zeta_r = fx.recon(q)                      # stored fit, real space (mu, n_rtot)

Zs = np.zeros((fx.n_mu, nsl), dtype=np.complex128)
Cacc = np.zeros((fx.n_mu, fx.n_mu), dtype=np.complex128)
for k in range(fx.nk):
    kq, _ = fx.kq_index(k, q)
    X_k = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[kq]),
                    fx.psi[k]).reshape(-1, fx.n_mu)
    A_k = X_k @ zeta_r                    # fitted rows on the full grid
    Ag = np.fft.fftn(A_k.reshape(-1, fx.nx, fx.ny, fx.nz),
                     axes=(1, 2, 3), norm="backward").reshape(-1, fx.n_rtot)[:, fi_q]
    Zs += np.conj(X_k.T) @ Ag
    Cacc += np.conj(X_k.T) @ X_k
print("Cacc vs C_q relF:", relF(Cacc, C_q[q]))
lam, R = np.linalg.eigh(0.5 * (Cacc + Cacc.conj().T))
keep = lam > lam[-1] * 1e-14
zsol = (R[:, keep] / lam[keep][None, :]) @ (np.conj(R[:, keep].T) @ Zs)
print("circular sphere-solve vs stored zeta~: raw relF =",
      f"{relF(zsol, fx.ZG[q][:, :nsl]):.3e}")
SS = np.sqrt(lam[keep])[::-1]
RR = R[:, keep][:, ::-1]
print("S-weighted relF =",
      f"{relF(SS[:, None] * (np.conj(RR.T) @ zsol), SS[:, None] * (np.conj(RR.T) @ fx.ZG[q][:, :nsl])):.3e}")

# and the FFT-of-zeta consistency alone (no solve): fftn(zeta_r body)|sphere
zg = np.fft.fftn((zeta_r * np.exp(-2j * np.pi * (fx.rfrac @ fx.qfr[q]))[None, :]
                  ).reshape(fx.n_mu, fx.nx, fx.ny, fx.nz),
                 axes=(1, 2, 3), norm="backward").reshape(fx.n_mu, fx.n_rtot)[:, fi_q]
print("fftn(zeta_r body)|sphere vs stored:", f"{relF(zg, fx.ZG[q][:, :nsl]):.3e}")

# per-mu centroid-phase hypothesis: stored zeta corresponds to the LAB-frame
# fit; the u-frame fit differs by zeta^u_mu = e^{+i q.r_mu} zeta^lab_mu.
for sgn in (+1, -1):
    ph = np.exp(sgn * 2j * np.pi * (fx.rmu_frac @ fx.qfr[q]))
    print(f"phase {sgn:+d}: raw relF(zsol, ph*stored) =",
          f"{relF(zsol, ph[:, None] * fx.ZG[q][:, :nsl]):.3e}")
