"""Denser-grid + off-grid-midpoint V_q interpolation test (READ-ONLY numpy).

Order-robust: drives everything by the actual fractional q (mf_header/kpoints/rk)
and integer lattice vectors R — no C-order FFT-reshape assumption.  Interpolate
the ingredients C_q (centroid basis) and Z_q=C·ζ (real-space r basis) via
truncated-R Fourier, solve C ζ = Z at the target q, form V_q, compare to the
direct-fit V_q from the stored ζ̃.

Two experiments:
  (A) leave-one-out on the full grid (native density).
  (B) OFF-GRID midpoints: interpolate from a coarse SUBLATTICE (2x2, 3x3) to the
      fine-only q's, validate against the fine direct-fit truth (the arbitrary-Q
      use case, with ground truth).
"""
import sys, h5py, numpy as np

restart, zeta, label, outnpz = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with h5py.File(restart, "r") as f:
    psi = f["psi_full_y"][()]
    kgrid = f["kgrid"][()].astype(int) if "kgrid" in f else None
with h5py.File(zeta, "r") as f:
    ZG   = f["zeta_q_G"][()]
    gvec = f["isdf_header/gvec_components"][()]
    ngk  = f["isdf_header/ngk"][()]
    fg   = f["mf_header/gspace/FFTgrid"][()]
    qfr  = f["mf_header/kpoints/rk"][()]
    bdot = f["mf_header/crystal/bdot"][()]
nk, nb, ns, n_mu = psi.shape
nx, ny, nz = [int(x) for x in fg]; n_rtot = nx*ny*nz
nq = ZG.shape[0]; ngkmax = ZG.shape[2]
if kgrid is None:
    kgrid = np.array([int(round(1.0/min(abs(qfr[qfr[:,d]>1e-6, d]).min(), 1))) if np.any(qfr[:,d]>1e-6) else 1 for d in range(3)])
kg = np.array([int(x) for x in kgrid])
print(f"[{label}] nq={nq} n_mu={n_mu} nb={nb} kgrid={kg.tolist()} FFT={nx}x{ny}x{nz} ngkmax={ngkmax}")
assert nk == nq, f"psi nk={nk} != zeta nq={nq}"

# C_q rebuild (all bands, charge) — index order = psi order (== zeta order, verified below)
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True)
# k-convolution done as direct DFT over q (order-robust: uses qfr, not a reshape)
# C_q[q] = Σ_R e^{-2πi q·R} Σ_ab conj(P_R)[a,ν,μ,b] P_R[a,ν,μ,b];  P_R = Σ_k e^{+2πi k·R} P_k
Rint = np.array([[rx, ry, rz]
                 for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rw = ((Rint + kg//2) % kg) - (kg//2)                        # wrapped lattice coords
# forward: P_R (norm='forward' inverse == no scale)
EqR = np.exp(2j*np.pi*(qfr @ Rw.T))                          # (nq, nR)  e^{+2πi q·R}
P_flat = P.reshape(nq, -1)
P_R = (EqR.T @ P_flat).reshape(len(Rw), ns, n_mu, n_mu, ns)  # Σ_k e^{+iqR} P_k
C_R = np.einsum('ravmb,ravmb->rvm', np.conj(P_R), P_R, optimize=True)   # (nR, ν, μ)
EqR_inv = np.exp(-2j*np.pi*(qfr @ Rw.T)) / nq               # fftn forward norm = 1/N
C_q = np.transpose((EqR_inv @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R, P_flat
print(f"[{label}] C_q rebuilt (order-robust DFT).")

# recon / forward-to-sphere (Bloch phase from qfr — order-robust)
def flat_idx(gv):
    return ((gv[0] % nx)*ny + (gv[1] % ny))*nz + (gv[2] % nz)
rx = np.arange(nx)/nx; ry = np.arange(ny)/ny; rz = np.arange(nz)/nz
RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing='ij')
rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
def recon(q):
    box = np.zeros((n_mu, n_rtot), dtype=np.complex128)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = ZG[q][:, :n]
    R = np.fft.ifftn(box.reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    return R * np.exp(2j*np.pi*(rfrac @ qfr[q]))[None, :]
def to_sphere(zr, q):
    ph = np.exp(-2j*np.pi*(rfrac @ qfr[q]))
    box = np.fft.fftn((zr*ph[None, :]).reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    out = np.zeros((n_mu, ngkmax), dtype=np.complex128); out[:, :n] = box[:, fi[:n]]
    return out
def vcoul(q):
    gv = gvec[q].astype(np.float64); kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    with np.errstate(divide='ignore'):
        v = 8*np.pi/k2
    v[k2 < 1e-8] = 0.0
    return v, int(ngk[q])
def make_Vq(zt, q):
    v, n = vcoul(q); A = zt[:, :n]*np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T
def relF(a, b): return np.linalg.norm(a-b)/np.linalg.norm(b)

# build ζ_q(μ,r), Z_q = C_q·ζ_q for all q
zeta_r = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
Z_r    = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
for q in range(nq):
    zr = recon(q); zeta_r[q] = zr; Z_r[q] = C_q[q] @ zr
Cq_flat = C_q.reshape(nq, -1); Zr_flat = Z_r.reshape(nq, -1)

# verify psi order == zeta order via grid-point solve
z0 = np.linalg.solve(C_q[1], Z_r[1]); zt0 = to_sphere(z0, 1); n1 = int(ngk[1])
print(f"[{label}] grid-point ζ̃ recovery (order check): "
      f"{relF(zt0[:,:n1], ZG[1,:,:n1]):.2e}")
Vq_direct = np.stack([make_Vq(ZG[q], q) for q in range(nq)])

def solve_zeta(Cmat, Zmat, mode="raw", lam=0.0):
    Ch = 0.5*(Cmat + Cmat.conj().T); U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    if mode == "raw": sinv = 1.0/s
    elif mode == "tikhonov": sinv = 1.0/(s + lam*(s.sum()/len(s)))
    else: sinv = np.where(s > lam*s[0], 1.0/np.where(s > lam*s[0], s, 1), 0.0)
    return (Vh.conj().T*sinv) @ (U.conj().T @ Zmat)

# R-vectors sorted by |R| for truncated-R interpolation
adot = np.linalg.inv(bdot)*(2*np.pi)**2
Rall = np.array([[rx, ry, rz] for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rall = ((Rall + kg//2) % kg) - (kg//2)
Rdist = np.sqrt(np.abs(np.einsum('ri,ij,rj->r', Rall, adot, Rall)))
Rsort = Rall[np.argsort(Rdist)]

def predict(train, target_q, Rset, split, mode, lam):
    F = np.exp(-2j*np.pi*(qfr[train] @ Rset.T)); Fi = np.linalg.pinv(F)
    f0 = np.exp(-2j*np.pi*(qfr[target_q] @ Rset.T)); w = f0 @ Fi
    C0 = (w @ Cq_flat[train]).reshape(n_mu, n_mu) if split in ("both", "C") else C_q[target_q]
    Z0 = (w @ Zr_flat[train]).reshape(n_mu, n_rtot) if split in ("both", "Z") else Z_r[target_q]
    zt = to_sphere(solve_zeta(C0, Z0, mode, lam), target_q); n0 = int(ngk[target_q])
    ze = relF(zt[:, :n0], ZG[target_q, :, :n0]); ve = relF(make_Vq(zt, target_q), Vq_direct[target_q])
    g0 = np.where((gvec[target_q][0] == 0) & (gvec[target_q][1] == 0) & (gvec[target_q][2] == 0))[0]
    he = relF(zt[:, g0[0]], ZG[target_q, :, g0[0]]) if g0.size else np.nan
    return ze, ve, he

# ============ (A) leave-one-out, native density ============
# Regularised (rank-cut λ=1e-6) solve — raw blows up (cond(C)~1e9 amplifies the
# ingredient-interp residual; see the solver sweep below).
REG_MODE, REG_LAM = "rankcut", 1e-6
print(f"\n[{label}] (A) LEAVE-ONE-OUT V_q (both-interp, {REG_MODE} λ={REG_LAM:.0e}) vs #R:")
print(f"  {'nR':>4} {'ζ̃ med':>10} {'V med':>10} {'V max':>10} {'head med':>10}")
nR_list = sorted(set(r for r in [4, 7, 13, 19, 25, nq-3] if 1 <= r <= nq-2))
for nR in nR_list:
    res = np.array([predict([q for q in range(nq) if q != q0], q0, Rsort[:nR], "both", REG_MODE, REG_LAM)
                    for q0 in range(nq)])
    print(f"  {nR:>4d} {np.median(res[:,0]):>10.3e} {np.median(res[:,1]):>10.3e} "
          f"{np.max(res[:,1]):>10.3e} {np.median(res[:,2]):>10.3e}")

nR_best = min(nq-2, 19)
print(f"\n[{label}] (A) split + solver at nR={nR_best}:")
for split in ["C", "Z", "both"]:
    res = np.array([predict([q for q in range(nq) if q != q0], q0, Rsort[:nR_best], split, "raw", 0.0)
                    for q0 in range(nq)])
    print(f"  {split:>5}: ζ̃ med={np.median(res[:,0]):.3e} V med={np.median(res[:,1]):.3e} V max={np.max(res[:,1]):.3e}")
for mode, lam in [("raw", 0.0), ("tikhonov", 1e-6), ("tikhonov", 1e-4), ("rankcut", 1e-6)]:
    res = np.array([predict([q for q in range(nq) if q != q0], q0, Rsort[:nR_best], "both", mode, lam)
                    for q0 in range(nq)])
    print(f"  {mode:>9} λ={lam:.0e}: ζ̃ med={np.median(res[:,0]):.3e} V med={np.median(res[:,1]):.3e} V max={np.max(res[:,1]):.3e}")

# ============ (B) OFF-GRID midpoints from a coarse sublattice ============
def on_sub(q, f):
    x = qfr[q]*f
    return np.all(np.abs(x - np.round(x)) < 1e-4)
print(f"\n[{label}] (B) OFF-GRID midpoint V_q from coarse sublattice (fine truth):")
print(f"  {'coarse':>8} {'ncoarse':>7} {'nmid':>5} {'nR':>4} {'ζ̃ med':>10} {'V med':>10} {'V max':>10} {'head med':>10}")
for f in [2, 3]:
    if np.any(kg % f != 0):
        continue
    coarse = [q for q in range(nq) if on_sub(q, f)]
    fine_only = [q for q in range(nq) if q not in coarse]
    if len(coarse) < 2 or not fine_only:
        continue
    ncoarse_pts = [max(1, kg[d]//f) if kg[d] > 1 else 1 for d in range(3)]
    Rc = np.array([[rx, ry, rz] for rx in range(ncoarse_pts[0])
                   for ry in range(ncoarse_pts[1]) for rz in range(ncoarse_pts[2])])
    Rc = ((Rc + kg//2) % kg) - (kg//2)
    nRc = len(Rc)
    res = np.array([predict(coarse, q0, Rc, "both", REG_MODE, REG_LAM) for q0 in fine_only])
    print(f"  {f}x{f}     {len(coarse):>7d} {len(fine_only):>5d} {nRc:>4d} "
          f"{np.median(res[:,0]):>10.3e} {np.median(res[:,1]):>10.3e} "
          f"{np.max(res[:,1]):>10.3e} {np.median(res[:,2]):>10.3e}")

np.savez(outnpz, nq=nq, n_mu=n_mu, kgrid=kg)
print(f"\n[{label}] done.")
