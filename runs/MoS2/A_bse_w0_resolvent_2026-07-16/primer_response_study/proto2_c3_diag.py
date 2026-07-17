"""proto2 C3 diagnostic — locate the V-TRS anomaly (SELF-CHECK 4: 3.82) and pin
the disk V_qmunu convention.  Hypothesis: the near-null-space content of the
stored per-q zeta solves is solver noise (not TRS-paired), carries most of the
full-rank slab-V TILE norm, but almost none of the physical B-block (the TRUE
fixedr rows already measured the latter).  READ-ONLY numpy."""
import sys, h5py
import numpy as np

restart, zeta = sys.argv[1], sys.argv[2]
with h5py.File(restart, "r") as f:
    psi = f["psi_full_y"][()]; kg = f["kgrid"][()].astype(int)
    Vdisk = f["V_qmunu"][()]
with h5py.File(zeta, "r") as f:
    ZG = f["zeta_q_G"][()]; gvec = f["isdf_header/gvec_components"][()]
    ngk = f["isdf_header/ngk"][()]; qfr = f["mf_header/kpoints/rk"][()]
    bdot = f["mf_header/crystal/bdot"][()]
    celvol = float(np.real(f["mf_header/crystal/celvol"][()]))
nq, n_mu, ngkmax = ZG.shape
b3len = float(np.sqrt(bdot[2, 2])); zc = np.pi / b3len
def relF(a, b): return np.linalg.norm(a - b) / np.linalg.norm(b)

def vcoul_slab(q, zero_g0=False):
    gv = gvec[q].astype(np.float64)
    kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    kxy = np.sqrt(np.einsum('ig,ij,jg->g', kf[:2], bdot[:2, :2], kf[:2]))
    f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kf[2] * b3len * zc)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (8 * np.pi / k2) * f2d / celvol
    v[k2 < 1e-12] = 0.0
    if zero_g0:
        v[np.all(gvec[q] == 0, axis=0)] = 0.0
    return v, int(ngk[q])

def make_V(zt, q, zero_g0=False):
    v, n = vcoul_slab(q, zero_g0)
    A = zt[:, :n] * np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T

ktrip = np.rint(qfr * kg[None, :]).astype(int) % kg[None, :]
trip2idx = {tuple(t): i for i, t in enumerate(ktrip)}
mq = np.array([trip2idx[tuple((-ktrip[q]) % kg)] for q in range(nq)])

Vfull = np.stack([make_V(ZG[q], q) for q in range(nq)])
Vbody = np.stack([make_V(ZG[q], q, zero_g0=True) for q in range(nq)])

print("=== disk V_qmunu convention: which of my slab tiles matches at q!=0? ===")
for q in [1, 4]:
    print(f"  q={q}: relF(disk, slab-full)={relF(Vdisk[q], Vfull[q]):.3e}  "
          f"relF(disk, slab-bodyonly)={relF(Vdisk[q], Vbody[q]):.3e}")

print("=== TRS conj(V_-q) vs V_q per q (full-rank direct tiles) ===")
for tag, V in [("slab FULL", Vfull), ("slab BODY-ONLY", Vbody), ("disk V_qmunu", Vdisk)]:
    errs = [relF(np.conj(V[mq[q]]), V[q]) for q in range(nq)]
    print(f"  {tag:>14}: " + " ".join(f"{e:.1e}" for e in errs))

print("=== TRS on RANK-TRUNCATED true-solve V (junk removed): kappa=1e4 ===")
# rebuild C_q (order-robust, verbatim) and solve with fixed rank on TRUE ingredients
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True)
Rall = np.array([[rx, ry, rz] for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rw = ((Rall + kg // 2) % kg) - (kg // 2)
EqR = np.exp(2j * np.pi * (qfr @ Rw.T))
P_R = (EqR.T @ P.reshape(nq, -1)).reshape(len(Rw), 2, n_mu, n_mu, 2)
C_R = np.einsum('ravmb,ravmb->rvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(((np.exp(-2j * np.pi * (qfr @ Rw.T)) / nq) @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
Vtrunc = np.zeros_like(Vfull)
for q in range(nq):
    Ch = 0.5 * (C_q[q] + C_q[q].conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    r = int(np.sum(s >= s[0] / 1e4))
    sinv = np.zeros_like(s); sinv[:r] = 1.0 / s[:r]
    zt = (Vh.conj().T * sinv) @ (U.conj().T @ (C_q[q] @ ZG[q]))   # C^+_r (C zeta) on sphere rep
    Vtrunc[q] = make_V(zt, q)
errs = [relF(np.conj(Vtrunc[mq[q]]), Vtrunc[q]) for q in range(nq)]
print(f"  rank-1e4 TRUE: " + " ".join(f"{e:.1e}" for e in errs))
print("=== tile norm share of the kappa>1e4 tail (full vs truncated, per q) ===")
print("  " + " ".join(f"{relF(Vtrunc[q], Vfull[q]):.2f}" for q in range(nq)))
