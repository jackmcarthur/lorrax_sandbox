"""Physical-observable test: is the V_q TILE blowup under ingredient-interp+solve
a null-space GAUGE artifact, or a real error? (READ-ONLY numpy)

ζ̃ = C⁻¹Z is undetermined in C's near-null space (redundant centroids), so the
V_q(μ,ν) TILE and ‖V_q‖_F blow up when interpolated ingredients push weight into
that space.  But physical observables contract V_q with PAIR DENSITIES d that
live in range(C): s = d* V_q d = Σ_G v |Σ_μ d*_μ ζ̃_μ(G)|² = (C⁻¹d)*... is
null-space-INSENSITIVE for d ∈ range(C).  (Owner's own note: "V_q tile
magnitude/covariance are gauge artifacts — measure the physical quantity.")

Here d^(t)(μ) = Σ_s ψ*_{n,k}(μ) ψ_{n',k}(μ) are genuine ISDF pair densities (in
range(C) by construction).  We compare the physical scalar s_t^interp vs
s_t^direct alongside the tile ‖ΔV‖_F, at each held-out q, for raw & rank-cut.
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
print(f"[{label}] nq={nq} n_mu={n_mu} kgrid={kg.tolist()}")

# C_q (order-robust DFT), ζ_r, Z_r
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True)
Rall = np.array([[rx, ry, rz] for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rw = ((Rall + kg//2) % kg) - (kg//2)
EqR = np.exp(2j*np.pi*(qfr @ Rw.T))
P_R = (EqR.T @ P.reshape(nq, -1)).reshape(len(Rw), ns, n_mu, n_mu, ns)
C_R = np.einsum('ravmb,ravmb->rvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(((np.exp(-2j*np.pi*(qfr @ Rw.T))/nq) @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R

def flat_idx(gv): return ((gv[0] % nx)*ny + (gv[1] % ny))*nz + (gv[2] % nz)
rxx = np.arange(nx)/nx; ryy = np.arange(ny)/ny; rzz = np.arange(nz)/nz
RX, RY, RZ = np.meshgrid(rxx, ryy, rzz, indexing='ij')
rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
def recon(q):
    box = np.zeros((n_mu, n_rtot), dtype=np.complex128); fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = ZG[q][:, :n]
    R = np.fft.ifftn(box.reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    return R*np.exp(2j*np.pi*(rfrac @ qfr[q]))[None, :]
def to_sphere(zr, q):
    box = np.fft.fftn((zr*np.exp(-2j*np.pi*(rfrac @ qfr[q]))[None, :]).reshape(n_mu, nx, ny, nz),
                      axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    fi = flat_idx(gvec[q]); n = int(ngk[q]); out = np.zeros((n_mu, ngkmax), dtype=np.complex128)
    out[:, :n] = box[:, fi[:n]]; return out
def vcoul(q):
    gv = gvec[q].astype(np.float64); kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    with np.errstate(divide='ignore'): v = 8*np.pi/k2
    v[k2 < 1e-8] = 0.0; return v, int(ngk[q])
def make_Vq(zt, q):
    v, n = vcoul(q); A = zt[:, :n]*np.sqrt(v[:n])[None, :]; return np.conj(A) @ A.T

zeta_r = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
Z_r = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
for q in range(nq):
    zr = recon(q); zeta_r[q] = zr; Z_r[q] = C_q[q] @ zr
Cq_flat = C_q.reshape(nq, -1); Zr_flat = Z_r.reshape(nq, -1)
Vq_direct = np.stack([make_Vq(ZG[q], q) for q in range(nq)])

# physical pair densities d^(t)(μ) = Σ_s ψ*_{n,k}(μ) ψ_{n',k}(μ)  (range(C))
rng = np.random.default_rng(0)
D = []
for _ in range(8):
    k = rng.integers(nk); n1 = rng.integers(nb); n2 = rng.integers(nb)
    d = np.einsum('sm,sm->m', np.conj(psi[k, n1]), psi[k, n2])   # (n_mu,)
    D.append(d / np.linalg.norm(d))
D = np.array(D)                                                  # (nd, n_mu)

def phys_scalars(zt, q):
    v, n = vcoul(q); A = zt[:, :n]*np.sqrt(v[:n])[None, :]        # (μ, G)
    proj = np.conj(D) @ A                                        # (nd, G)  Σ_μ d*_μ ζ̃√v
    return np.sum(np.abs(proj)**2, axis=1)                       # (nd,) = d* V_q d, real
S_direct = np.array([phys_scalars(ZG[q], q) for q in range(nq)])  # (nq, nd)

def solve_zeta(Cmat, Zmat, mode, lam):
    Ch = 0.5*(Cmat + Cmat.conj().T); U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    if mode == "raw": sinv = 1.0/s
    else: sinv = np.where(s > lam*s[0], 1.0/np.where(s > lam*s[0], s, 1), 0.0)
    return (Vh.conj().T*sinv) @ (U.conj().T @ Zmat)

adot = np.linalg.inv(bdot)*(2*np.pi)**2
Rdist = np.sqrt(np.abs(np.einsum('ri,ij,rj->r', Rw, adot, Rw))); Rsort = Rw[np.argsort(Rdist)]
def relF(a, b): return np.linalg.norm(a-b)/np.linalg.norm(b)

def loo(nR, mode, lam):
    tile = np.zeros(nq); phys = np.zeros(nq)
    for q0 in range(nq):
        tr = [q for q in range(nq) if q != q0]; Rset = Rsort[:nR]
        F = np.exp(-2j*np.pi*(qfr[tr] @ Rset.T)); w = np.exp(-2j*np.pi*(qfr[q0] @ Rset.T)) @ np.linalg.pinv(F)
        C0 = (w @ Cq_flat[tr]).reshape(n_mu, n_mu); Z0 = (w @ Zr_flat[tr]).reshape(n_mu, n_rtot)
        zt = to_sphere(solve_zeta(C0, Z0, mode, lam), q0)
        tile[q0] = relF(make_Vq(zt, q0), Vq_direct[q0])
        si = phys_scalars(zt, q0)
        phys[q0] = np.median(np.abs(si - S_direct[q0]) / np.abs(S_direct[q0]))
    return tile, phys

nR = min(nq-2, max(7, nq//2))
print(f"[{label}] leave-one-out at nR={nR}: TILE ‖ΔV‖_F/‖V‖ vs PHYSICAL d*V_q d rel-error")
print(f"  {'solve':>14} {'tile med':>10} {'tile max':>10} {'phys med':>10} {'phys max':>10}")
for mode, lam in [("raw", 0.0), ("rankcut", 1e-8), ("rankcut", 1e-6), ("rankcut", 1e-4), ("rankcut", 1e-2)]:
    tile, phys = loo(nR, mode, lam)
    tag = f"{mode} {lam:.0e}" if mode != "raw" else "raw"
    print(f"  {tag:>14} {np.median(tile):>10.3e} {np.max(tile):>10.3e} "
          f"{np.median(phys):>10.3e} {np.max(phys):>10.3e}")
print(f"[{label}] done.")
