"""Definitive gauge-vs-physical test: contract V_q with vectors GUARANTEED in
range(C_q) (d = C_q_direct @ e, e random).  For such d, s = d* V_q d = e* C Vζ...
is exactly null-space-insensitive, so if the rank-cut interp recovers s while the
tile ‖ΔV‖_F blows up, the tile error is a pure gauge artifact.  (READ-ONLY.)
Also reports contraction with the ζ-fit's OWN centroid pair densities.
"""
import sys, h5py, numpy as np
restart, zeta, label = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(restart, "r") as f:
    psi = f["psi_full_y"][()]; kgrid = f["kgrid"][()].astype(int)
with h5py.File(zeta, "r") as f:
    ZG = f["zeta_q_G"][()]; gvec = f["isdf_header/gvec_components"][()]
    ngk = f["isdf_header/ngk"][()]; fg = f["mf_header/gspace/FFTgrid"][()]
    qfr = f["mf_header/kpoints/rk"][()]; bdot = f["mf_header/crystal/bdot"][()]
nk, nb, ns, n_mu = psi.shape
nx, ny, nz = [int(x) for x in fg]; n_rtot = nx*ny*nz
nq = ZG.shape[0]; ngkmax = ZG.shape[2]; kg = kgrid.astype(int)
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True)
Rall = np.array([[rx, ry, rz] for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rw = ((Rall + kg//2) % kg) - (kg//2)
P_R = (np.exp(2j*np.pi*(qfr @ Rw.T)).T @ P.reshape(nq, -1)).reshape(len(Rw), ns, n_mu, n_mu, ns)
C_R = np.einsum('ravmb,ravmb->rvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(((np.exp(-2j*np.pi*(qfr @ Rw.T))/nq) @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R
def flat_idx(gv): return ((gv[0] % nx)*ny + (gv[1] % ny))*nz + (gv[2] % nz)
RX, RY, RZ = np.meshgrid(np.arange(nx)/nx, np.arange(ny)/ny, np.arange(nz)/nz, indexing='ij')
rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
def recon(q):
    box = np.zeros((n_mu, n_rtot), np.complex128); fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = ZG[q][:, :n]
    R = np.fft.ifftn(box.reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    return R*np.exp(2j*np.pi*(rfrac @ qfr[q]))[None, :]
def to_sphere(zr, q):
    box = np.fft.fftn((zr*np.exp(-2j*np.pi*(rfrac @ qfr[q]))[None, :]).reshape(n_mu, nx, ny, nz),
                      axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    fi = flat_idx(gvec[q]); n = int(ngk[q]); out = np.zeros((n_mu, ngkmax), np.complex128)
    out[:, :n] = box[:, fi[:n]]; return out
def vcoul(q):
    gv = gvec[q].astype(float); kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    with np.errstate(divide='ignore'): v = 8*np.pi/k2
    v[k2 < 1e-8] = 0.0; return v, int(ngk[q])
def make_Vq(zt, q):
    v, n = vcoul(q); A = zt[:, :n]*np.sqrt(v[:n])[None, :]; return np.conj(A) @ A.T
zeta_r = np.zeros((nq, n_mu, n_rtot), np.complex128); Z_r = np.zeros((nq, n_mu, n_rtot), np.complex128)
for q in range(nq):
    zr = recon(q); zeta_r[q] = zr; Z_r[q] = C_q[q] @ zr
Cq_flat = C_q.reshape(nq, -1); Zr_flat = Z_r.reshape(nq, -1)
Vq_direct = np.stack([make_Vq(ZG[q], q) for q in range(nq)])
rng = np.random.default_rng(1)
# range-guaranteed d: d = C_q0_direct @ e  (per q; in range(C_q0) by construction)
E = rng.standard_normal((6, n_mu)) + 1j*rng.standard_normal((6, n_mu))
def phys(zt, q, D):
    v, n = vcoul(q); A = zt[:, :n]*np.sqrt(v[:n])[None, :]
    proj = np.conj(D) @ A; return np.sum(np.abs(proj)**2, 1)
def solve_zeta(C, Z, mode, lam):
    Ch = 0.5*(C + C.conj().T); U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    sinv = 1.0/s if mode == "raw" else np.where(s > lam*s[0], 1.0/np.where(s > lam*s[0], s, 1), 0.0)
    return (Vh.conj().T*sinv) @ (U.conj().T @ Z)
adot = np.linalg.inv(bdot)*(2*np.pi)**2
Rdist = np.sqrt(np.abs(np.einsum('ri,ij,rj->r', Rw, adot, Rw))); Rsort = Rw[np.argsort(Rdist)]
def relF(a, b): return np.linalg.norm(a-b)/np.linalg.norm(b)
nR = min(nq-2, max(7, nq//2))
print(f"[{label}] range-guaranteed d=C·e contraction, leave-one-out nR={nR}:")
print(f"  {'solve':>14} {'tile med':>10} {'phys(C·e) med':>13} {'phys(C·e) max':>13}")
for mode, lam in [("raw", 0.0), ("rankcut", 1e-6), ("rankcut", 1e-4), ("rankcut", 1e-2), ("rankcut", 1e-1)]:
    tl = []; ph = []
    for q0 in range(nq):
        tr = [q for q in range(nq) if q != q0]; Rset = Rsort[:nR]
        F = np.exp(-2j*np.pi*(qfr[tr] @ Rset.T)); w = np.exp(-2j*np.pi*(qfr[q0] @ Rset.T)) @ np.linalg.pinv(F)
        C0 = (w @ Cq_flat[tr]).reshape(n_mu, n_mu); Z0 = (w @ Zr_flat[tr]).reshape(n_mu, n_rtot)
        zt = to_sphere(solve_zeta(C0, Z0, mode, lam), q0)
        D = (C_q[q0] @ E.T).T; D = D/np.linalg.norm(D, axis=1, keepdims=True)
        sd = phys(ZG[q0], q0, D); si = phys(zt, q0, D)
        tl.append(relF(make_Vq(zt, q0), Vq_direct[q0])); ph.append(np.median(np.abs(si-sd)/np.abs(sd)))
    tag = f"{mode} {lam:.0e}" if mode != "raw" else "raw"
    print(f"  {tag:>14} {np.median(tl):>10.3e} {np.median(ph):>13.3e} {np.max(ph):>13.3e}")
print(f"[{label}] done.")
