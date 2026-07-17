"""Stage 3 — the decisive leave-one-out V_q test (READ-ONLY numpy).

For every on-grid q0: interpolate the SMOOTH INGREDIENTS C_q (centroid basis)
and Z_q (real-space r basis) from the OTHER q's via truncated-R Fourier
interpolation, then SOLVE  C_q0 ζ_q0 = Z_q0  at the held-out q0 (never
interpolate ζ = C^{-1}Z itself).  FFT ζ_q0 back to the sphere, form V_q0, and
compare against the direct-fit V_q0 built from the stored ζ̃.

Ingredients (code-exact, from the stored fixture):
  C_q(μ,ν)     rebuilt from psi_full_y (== isdf.core.c_q_from_psi_sm, charge).
  ζ_q(μ,r)     = recon(stored ζ̃_q)  [scatter sphere -> ifftn -> undo Bloch].
  Z_q(μ,r)     = C_q @ ζ_q(μ,r)      [the code's ZCT, exact since ζ=C^{-1}Z].

Splits reported:  both-interp, C-only (true Z), Z-only (true C); raw vs
Tikhonov/rank-cut solve (to test whether cond(C) amplifies interp noise).
V weighting v = 8π/|q+G|² (head dropped); it enters interp and direct-fit
IDENTICALLY so the relative error isolates interpolation, not the Coulomb kernel.
"""
import sys, h5py, numpy as np

restart = sys.argv[1]
zeta    = sys.argv[2]
label   = sys.argv[3]
outnpz  = sys.argv[4]

# ---------------------------------------------------------------------------
with h5py.File(restart, "r") as f:
    psi   = f["psi_full_y"][()]                # (nk, nb, ns, n_mu)
    kgrid = f["kgrid"][()].astype(int)
    Vqmunu = f["V_qmunu"][()]                  # (nq, n_mu, n_mu) direct-fit truth
with h5py.File(zeta, "r") as f:
    ZG   = f["zeta_q_G"][()]                    # (nq, n_mu, ngkmax) c128 (physical q+G)
    gvec = f["isdf_header/gvec_components"][()] # (nq, 3, ngkmax) int32
    ngk  = f["isdf_header/ngk"][()]             # (nq,)
    fg   = f["mf_header/gspace/FFTgrid"][()]
    qfr  = f["mf_header/kpoints/rk"][()]        # (nq,3) fractional q
    bdot = f["mf_header/crystal/bdot"][()]      # reciprocal metric (Ry units)

nk, nb, ns, n_mu = psi.shape
nkx, nky, nkz = [int(x) for x in kgrid]; nq = nkx*nky*nkz
nx, ny, nz = [int(x) for x in fg]; n_rtot = nx*ny*nz
ngkmax = ZG.shape[2]
print(f"[{label}] nq={nq} n_mu={n_mu} nb={nb} FFT={nx}x{ny}x{nz}={n_rtot} ngkmax={ngkmax}")

# ---------------------------------------------------------------------------
# C_q(μ,ν) rebuild (charge, all bands) — same as falloff_cq.py
# ---------------------------------------------------------------------------
psiY = psi
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psiY, optimize=True).reshape(
    nkx, nky, nkz, ns, n_mu, n_mu, ns)
P_R = np.fft.ifftn(P, axes=(0, 1, 2), norm='forward')
C_R = np.einsum('xyzavmb,xyzavmb->xyzvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(np.fft.fftn(C_R, axes=(0, 1, 2), norm='forward').reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R
print(f"[{label}] C_q rebuilt.")

# ---------------------------------------------------------------------------
# recon ζ_q(μ,r) and forward FFT to sphere (validated convention)
# ---------------------------------------------------------------------------
def flat_idx(gv):
    gx = gv[0] % nx; gy = gv[1] % ny; gz = gv[2] % nz
    return ((gx*ny)+gy)*nz + gz
rx = np.arange(nx)/nx; ry = np.arange(ny)/ny; rz = np.arange(nz)/nz
RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing='ij')
rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)      # (n_rtot,3) C-order

def recon(q):
    """ζ_q(μ,r) on the full FFT grid, band-limited to sphere(q)."""
    box = np.zeros((n_mu, n_rtot), dtype=np.complex128)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = ZG[q][:, :n]
    R = np.fft.ifftn(box.reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    return R * np.exp(2j*np.pi*(rfrac @ qfr[q]))[None, :]       # ζ = e^{+iq.r} R

def to_sphere(zeta_r, q):
    """forward: ζ(μ,r) -> ζ̃(μ,G) on sphere(q)."""
    ph = np.exp(-2j*np.pi*(rfrac @ qfr[q]))
    box = np.fft.fftn((zeta_r*ph[None, :]).reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    out = np.zeros((n_mu, ngkmax), dtype=np.complex128)
    out[:, :n] = box[:, fi[:n]]
    return out

# validate recon<->to_sphere round trip on the sphere
_zt = to_sphere(recon(0), 0)
_n0 = int(ngk[0])
print(f"[{label}] recon/forward round-trip on sphere(Γ): "
      f"{np.linalg.norm(_zt[:,:_n0]-ZG[0,:,:_n0])/np.linalg.norm(ZG[0,:,:_n0]):.2e}")

# Coulomb weight v(q+G) = 8π/|q+G|², head dropped
def vcoul(q):
    gv = gvec[q].astype(np.float64)
    kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    with np.errstate(divide='ignore'):
        v = 8*np.pi/k2
    v[k2 < 1e-8] = 0.0
    return v, int(ngk[q])

def make_Vq(zt, q):
    """V_q(μ,ν)=Σ_G ζ̃*_μ v ζ̃_ν on sphere(q)."""
    v, n = vcoul(q)
    A = zt[:, :n] * np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T

# build ζ_q(μ,r) and Z_q(μ,r) = C_q @ ζ_q for all q
zeta_r = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
Z_r    = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
for q in range(nq):
    zr = recon(q)
    zeta_r[q] = zr
    Z_r[q] = C_q[q] @ zr
print(f"[{label}] ζ_q(μ,r), Z_q(μ,r) built ({zeta_r.nbytes/1e9:.1f} GB each).")

# sanity: grid-point trivial recovery (true C, true Z -> stored ζ̃)
q_chk = 1
z0 = np.linalg.solve(C_q[q_chk] + 0*np.eye(n_mu), Z_r[q_chk])
zt0 = to_sphere(z0, q_chk); nchk = int(ngk[q_chk])
print(f"[{label}] grid-point solve recovers stored ζ̃ (q={q_chk}): "
      f"{np.linalg.norm(zt0[:,:nchk]-ZG[q_chk,:,:nchk])/np.linalg.norm(ZG[q_chk,:,:nchk]):.2e}")

# direct-fit V_q from stored ζ̃ (my v) and its match to the on-disk V_qmunu
Vq_direct = np.stack([make_Vq(ZG[q], q) for q in range(nq)])
# scale-align V_qmunu (different Coulomb prefactor/truncation) for a shape check
def relF(a, b): return np.linalg.norm(a-b)/np.linalg.norm(b)
sc = (np.vdot(Vqmunu.ravel(), Vq_direct.ravel())/np.vdot(Vq_direct.ravel(), Vq_direct.ravel())).real
print(f"[{label}] direct-fit V_q(my v) vs on-disk V_qmunu after scalar align: "
      f"relF={relF(sc*Vq_direct, Vqmunu):.3f} (scale={sc:.3f}; different v — shape check only)")

# ---------------------------------------------------------------------------
# leave-one-out interpolation of ingredients
# ---------------------------------------------------------------------------
def wrap(n, N): return n - N*((2*n) > N)
Rvecs = np.array([[wrap(ix, nkx), wrap(iy, nky), wrap(iz, nkz)]
                  for ix in range(nkx) for iy in range(nky) for iz in range(nkz)])
# metric on R for sorting (fractional is fine — only ordering matters)
adot = np.linalg.inv(bdot) * (2*np.pi)**2   # approx real metric; ordering only
Rdist = np.sqrt(np.abs(np.einsum('ri,ij,rj->r', Rvecs, adot, Rvecs)))
Rsort = Rvecs[np.argsort(Rdist)]
# q fractional in the SAME C-order flat as C_q/Z_r (rebuilt from FFT grid)
qflat = np.array([[ix/nkx, iy/nky, iz/nkz]
                  for ix in range(nkx) for iy in range(nky) for iz in range(nkz)])

def interp_ingredient(vals_flat, q0, nR):
    """Fourier-interp vals[q] (shape (nq, M)) to held-out q0 from others.
    Uses explicit pinv(F) @ vals (BLAS matmul) — lstsq with a huge RHS is
    LAPACK-gelsd-pathological (single-threaded, GB work arrays)."""
    train = [q for q in range(nq) if q != q0]
    Rset = Rsort[:nR]
    F = np.exp(-2j*np.pi*(qflat[train] @ Rset.T))          # (n_train, nR)
    f0 = np.exp(-2j*np.pi*(qflat[q0] @ Rset.T))            # (nR,)
    w = f0 @ np.linalg.pinv(F)                              # (n_train,) predictor row
    return w @ vals_flat[train]                             # (M,)

def solve_zeta(Cmat, Zmat, mode, lam=0.0):
    """Solve C ζ = Z via one Hermitian SVD + BLAS matmuls (multithreaded)."""
    n = Cmat.shape[0]
    Ch = 0.5*(Cmat + Cmat.conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    if mode == "raw":
        sinv = 1.0/s                                        # keep all SVs (noise-amplifying)
    elif mode == "tikhonov":
        alpha = lam*(s.sum()/n)                             # (C+αI)^{-1}, α=λ·tr/n
        sinv = 1.0/(s + alpha)
    elif mode == "rankcut":
        sinv = np.where(s > lam*s[0], 1.0/np.where(s > lam*s[0], s, 1), 0.0)
    else:
        raise ValueError(mode)
    return (Vh.conj().T * sinv) @ (U.conj().T @ Zmat)

Cq_flat = C_q.reshape(nq, -1)
Zr_flat = Z_r.reshape(nq, -1)

def run_split(nR, split, mode, lam):
    """Return per-q (zeta-tilde relF, V relF, head relF)."""
    zt_err = np.zeros(nq); v_err = np.zeros(nq); h_err = np.zeros(nq)
    for q0 in range(nq):
        C0 = (interp_ingredient(Cq_flat, q0, nR).reshape(n_mu, n_mu)
              if split in ("both", "C") else C_q[q0])
        Z0 = (interp_ingredient(Zr_flat, q0, nR).reshape(n_mu, n_rtot)
              if split in ("both", "Z") else Z_r[q0])
        z0 = solve_zeta(C0, Z0, mode, lam)
        zt = to_sphere(z0, q0)
        n0 = int(ngk[q0])
        zt_err[q0] = np.linalg.norm(zt[:, :n0]-ZG[q0, :, :n0])/np.linalg.norm(ZG[q0, :, :n0])
        Vi = make_Vq(zt, q0); Vd = Vq_direct[q0]
        v_err[q0] = relF(Vi, Vd)
        # head vector = g0 column ζ̃(μ,G=0)
        g0 = np.where((gvec[q0][0] == 0) & (gvec[q0][1] == 0) & (gvec[q0][2] == 0))[0]
        if g0.size:
            hi = zt[:, g0[0]]; hd = ZG[q0, :, g0[0]]
            h_err[q0] = np.linalg.norm(hi-hd)/np.linalg.norm(hd)
    return zt_err, v_err, h_err

print(f"\n[{label}] LEAVE-ONE-OUT V_q — both-interp, raw solve, vs #R:")
print(f"  {'nR':>4} {'ζ̃ medRelF':>11} {'V medRelF':>11} {'V maxRelF':>11} {'head medRelF':>12}")
results = {}
for nR in [1, 4, 7]:
    zt, ve, he = run_split(nR, "both", "raw", 0.0)
    results[('both','raw',nR)] = (zt, ve, he)
    print(f"  {nR:>4d} {np.median(zt):>11.4e} {np.median(ve):>11.4e} {np.max(ve):>11.4e} {np.median(he):>12.4e}")

print(f"\n[{label}] SPLIT at nR=7 (raw): which ingredient dominates interp error?")
for split in ["C", "Z", "both"]:
    zt, ve, he = run_split(7, split, "raw", 0.0)
    print(f"  {split:>5}-interp:  ζ̃ med={np.median(zt):.4e}  V med={np.median(ve):.4e}  V max={np.max(ve):.4e}")

print(f"\n[{label}] SOLVER regularisation at nR=7 (both-interp): does cond(C) amplify interp noise?")
for mode, lam in [("raw", 0.0), ("tikhonov", 1e-8), ("tikhonov", 1e-6), ("tikhonov", 1e-4),
                  ("rankcut", 1e-8), ("rankcut", 1e-6)]:
    zt, ve, he = run_split(7, "both", mode, lam)
    print(f"  {mode:>9} λ={lam:.0e}:  ζ̃ med={np.median(zt):.4e}  V med={np.median(ve):.4e}  V max={np.max(ve):.4e}")

np.savez(outnpz,
         loo_nR=np.array([1,4,7]),
         loo_zt=np.array([np.median(results[('both','raw',n)][0]) for n in [1,4,7]]),
         loo_v =np.array([np.median(results[('both','raw',n)][1]) for n in [1,4,7]]),
         nq=nq, n_mu=n_mu)
print(f"\n[{label}] Stage 3 done.")
