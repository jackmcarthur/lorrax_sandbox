"""ζ-DIRECT interpolation: skip the ill-conditioned C^{-1}Z solve entirely.
The production fit ζ_q (dataset zeta_q_G) shares the fixed centroid gauge across
all q, so ζ_R = DFT_q ζ_q is well-defined.  Test (a) ζ_R falloff, (b) leave-one-out
reconstruction of ζ_q0 from the other q via ζ_R, rebuilt into V_q and physical
d*V_q d, compared to the direct fit.  READ-ONLY.  argv: restart zeta label
"""
import sys, h5py, numpy as np
restart, zeta, label = sys.argv[1], sys.argv[2], sys.argv[3]
with h5py.File(restart, "r") as f:
    psi = f["psi_full_y"][()]; kgrid = f["kgrid"][()].astype(int)
with h5py.File(zeta, "r") as f:
    ZE = f["zeta_q_G"][()]            # ζ_q (the fit result), sphere order per q
    gvec = f["isdf_header/gvec_components"][()]; ngk = f["isdf_header/ngk"][()]
    fg = f["mf_header/gspace/FFTgrid"][()]; qfr = f["mf_header/kpoints/rk"][()]
    bdot = f["mf_header/crystal/bdot"][()]
nk, nb, ns, n_mu = psi.shape
nx, ny, nz = [int(x) for x in fg]; n_rtot = nx*ny*nz
nq = ZE.shape[0]; ngkmax = ZE.shape[2]; kg = kgrid.astype(int)
# centroid pair densities for physical contraction (single-k, all pairs of a few bands)
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True)  # only used for C_q (phys d via C)
Rall = np.array([[rx, ry, rz] for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rw = ((Rall + kg//2) % kg) - (kg//2)
P_R = (np.exp(2j*np.pi*(qfr @ Rw.T)).T @ P.reshape(nq, -1)).reshape(len(Rw), ns, n_mu, n_mu, ns)
C_R = np.einsum('ravmb,ravmb->rvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(((np.exp(-2j*np.pi*(qfr @ Rw.T))/nq) @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R
def flat_idx(gv): return ((gv[0] % nx)*ny + (gv[1] % ny))*nz + (gv[2] % nz)
RX, RY, RZ = np.meshgrid(np.arange(nx)/nx, np.arange(ny)/ny, np.arange(nz)/nz, indexing='ij')
rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
def recon(q, DAT):  # real-space ζ_μ(r) at q (Bloch-periodic part), from sphere data
    box = np.zeros((n_mu, n_rtot), np.complex128); fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = DAT[q][:, :n]
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
def phys(zt, q, D):
    v, n = vcoul(q); A = zt[:, :n]*np.sqrt(v[:n])[None, :]
    proj = np.conj(D) @ A; return np.sum(np.abs(proj)**2, 1)
# --- ζ in real space, per q, and its R-transform (falloff) ---
zeta_r = np.stack([recon(q, ZE) for q in range(nq)])           # (nq, n_mu, n_rtot)
zr_flat = zeta_r.reshape(nq, -1)
zeta_R = (np.exp(2j*np.pi*(qfr @ Rw.T)).T @ zr_flat) / nq       # (nR, n_mu*n_rtot)
adot = np.linalg.inv(bdot)*(2*np.pi)**2
Rdist = np.sqrt(np.abs(np.einsum('ri,ij,rj->r', Rw, adot, Rw)))
order = np.argsort(Rdist)
amp = np.linalg.norm(zeta_R, axis=1); amp = amp/amp.max()
print(f"[{label}] nq={nq} n_mu={n_mu} kgrid={list(kg)}")
print(f"[{label}] ζ_R falloff (|ζ_R|/max, sorted by |R| Bohr):")
for i in order:
    print(f"    |R|={Rdist[i]:6.2f}  |ζ_R|={amp[i]:.3e}")
# direct V_q + physical d (d = C_q @ e, guaranteed in range)
Vq_direct = np.stack([make_Vq(ZE[q], q) for q in range(nq)])
rng = np.random.default_rng(0); E = rng.standard_normal((6, n_mu)) + 1j*rng.standard_normal((6, n_mu))
def relF(a, b): return np.linalg.norm(a-b)/np.linalg.norm(b)
Rsort = Rw[order]
print(f"[{label}] LEAVE-ONE-OUT ζ-direct interp (no solve): V_q tile + physical d*V_q d vs #R:")
print(f"  {'nR':>4} {'ζ med':>10} {'V tile med':>10} {'V tile max':>10} {'phys med':>10} {'phys max':>10}")
for nR in sorted(set([4, 7, min(nq-1, 8), nq-1] if nq <= 9 else [4, 7, 13, 19, 25, nq-2])):
    if nR > nq-1: continue
    zt, vt, ph = [], [], []
    for q0 in range(nq):
        tr = [q for q in range(nq) if q != q0]; Rset = Rsort[:nR]
        F = np.exp(-2j*np.pi*(qfr[tr] @ Rset.T)); w = np.exp(-2j*np.pi*(qfr[q0] @ Rset.T)) @ np.linalg.pinv(F)
        zr0 = (w @ zr_flat[tr]).reshape(n_mu, n_rtot)
        zt0 = to_sphere(zr0, q0)
        zt.append(relF(zt0[:, :int(ngk[q0])], ZE[q0][:, :int(ngk[q0])]))
        vt.append(relF(make_Vq(zt0, q0), Vq_direct[q0]))
        D = (C_q[q0] @ E.T).T; D = D/np.linalg.norm(D, axis=1, keepdims=True)
        sd = phys(ZE[q0], q0, D); si = phys(zt0, q0, D); ph.append(np.median(np.abs(si-sd)/np.abs(sd)))
    print(f"  {nR:>4} {np.median(zt):>10.3e} {np.median(vt):>10.3e} {np.max(vt):>10.3e} {np.median(ph):>10.3e} {np.max(ph):>10.3e}")
print(f"[{label}] done.")
