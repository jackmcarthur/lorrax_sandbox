"""ARBITRARY-Q ISDF zeta-structure test (read-only, 1 GPU / numpy).

Question (owner-invited): is LORRAX's zeta_q(mu,G) the |q+G|-sphere-windowed
Fourier transform of a q-INDEPENDENT real-space object zeta_mu(r) -- i.e. is
the q-dependence only sphere-window + e^{iqr} bookkeeping, or a genuinely
different fit per q?

G-flat convention (verified from src): zeta_q_G[q,mu,g] = sum_r zeta_q(mu,r)
* exp(-2pi i (q+G_g).r).  So on-disk coeffs are sampled at physical reciprocal
vector k_phys = q + G_g.  Reconstruct real-space zeta_q(mu,r), compare across
neighbouring q up to a per-mu GLOBAL phase (gauge).  Then test the design
payoff: predict zeta~_q and V_q at every on-grid q from ONE master real-space
object (reconstructed at Gamma) + analytic v + analytic centroid phase.
"""
import sys, h5py, numpy as np

P = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5"
with h5py.File(P, "r") as f:
    ZG   = f["zeta_q_G"][()]                    # (nq, n_mu, ngkmax) c128
    gvec = f["isdf_header/gvec_components"][()]  # (nq, 3, ngkmax) int32
    ngk  = f["isdf_header/ngk"][()]              # (nq,)
    fg   = f["mf_header/gspace/FFTgrid"][()]     # (nx,ny,nz)
    qfr  = f["mf_header/kpoints/rk"][()]         # (nq,3) fractional q
    bdot = f["mf_header/crystal/bdot"][()]       # reciprocal metric
    blat = float(f["mf_header/crystal/blat"][()])
    rmu  = f["isdf_header/centroids/r_mu_crystal"][()]  # (n_mu,3)
    g0   = f["g0_mu"][()]                         # (nq, n_mu)

nq, n_mu, ngkmax = ZG.shape
nx, ny, nz = [int(x) for x in fg]; n_rtot = nx*ny*nz
print(f"nq={nq} n_mu={n_mu} ngkmax={ngkmax} FFT={nx}x{ny}x{nz}={n_rtot}")

def flat_idx(gv):  # (3,N) Miller -> C-order flat index into (nx,ny,nz)
    gx = gv[0] % nx; gy = gv[1] % ny; gz = gv[2] % nz
    return ((gx*ny)+gy)*nz + gz

# ---- validate index mapping via g0 = zeta at G=(0,0,0) ---------------------
q0 = int(np.argmin(np.linalg.norm(qfr, axis=1)))   # Gamma
gv0 = gvec[q0]; fi0 = flat_idx(gv0)
zero_slot = np.where((gv0[0]==0)&(gv0[1]==0)&(gv0[2]==0))[0]
print("g0 check: zero-Miller slot(s)", zero_slot[:3],
      "max|zeta[q0,:,slot]-g0[q0]| =",
      float(np.max(np.abs(ZG[q0,:,zero_slot[0]] - g0[q0]))) if zero_slot.size else "NO ZERO G")

# ---- real-space reconstruction --------------------------------------------
# r fractional coords for C-order flat r
rz = np.arange(nz)/nz; ry = np.arange(ny)/ny; rx = np.arange(nx)/nx
RX,RY,RZ = np.meshgrid(rx,ry,rz, indexing='ij')
rfrac = np.stack([RX.ravel(),RY.ravel(),RZ.ravel()],1)   # (n_rtot,3) C-order

def recon(q, mus):
    """R_q(mu,r) = ifftn(scatter(zeta_q_G))  (= exp(-2pi i q.r) zeta_q(mu,r)).
       zeta_q(mu,r) = exp(+2pi i q.r) * R_q."""
    m = len(mus); box = np.zeros((m, n_rtot), dtype=np.complex128)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = ZG[q][mus][:, :n]
    box = box.reshape(m, nx, ny, nz)
    R = np.fft.ifftn(box, axes=(1,2,3), norm='backward').reshape(m, n_rtot)
    ph = np.exp(2j*np.pi*(rfrac @ qfr[q]))          # exp(+2pi i q.r)
    zeta = R * ph[None, :]
    return R, zeta

# neighbouring-q pairs on the 3x3 grid (qx,qy in {0,1/3,2/3})
def qkey(q): return (round(qfr[q,0]*3)%3, round(qfr[q,1]*3)%3)
idx = {qkey(q): q for q in range(nq)}
pairs = []
for (a,b) in idx:
    for d in [((a+1)%3,b), (a,(b+1)%3)]:
        pairs.append((idx[(a,b)], idx[d]))
pairs = list({tuple(sorted(p)) for p in pairs})

MUS = np.linspace(0, n_mu-1, 64).astype(int)   # 64-centroid subset
print(f"\n--- PART A: neighbour-q shape/phase test ({len(MUS)} centroids, {len(pairs)} pairs) ---")
print("p_zeta = min-over-global-phase ||z_a - e^{iphi} z_b||/||z_a||  (H1/H3: small)")
print("p_R    = same for cell-periodic R                              (H2: small)")
print("m_mag  = |||z_a|-|z_b||| / |||z_a|||   (gauge-invariant magnitude diff)")
allpz=[]; allpr=[]; allm=[]; phase_corr=[]
for (a,b) in pairs:
    Ra,Za = recon(a,MUS); Rb,Zb = recon(b,MUS)
    def paligned(Xa,Xb):
        ov = np.sum(np.conj(Xa)*Xb, axis=1)          # (m,)
        phi = ov/np.abs(ov)
        num = np.linalg.norm(Xa - phi[:,None]*Xb, axis=1)
        return num/np.linalg.norm(Xa,axis=1), np.angle(ov)
    pz,phi_z = paligned(Za,Zb)
    pr,_     = paligned(Ra,Rb)
    mm = np.linalg.norm(np.abs(Za)-np.abs(Zb),axis=1)/np.linalg.norm(np.abs(Za),axis=1)
    allpz.append(pz); allpr.append(pr); allm.append(mm)
    # H3 test: does fitted global phase phi_z ~ 2pi (q_b-q_a).r_mu  ?
    dq = qfr[b]-qfr[a]
    predphi = (2*np.pi*(rmu[MUS] @ dq))
    # compare wrapped
    dd = np.angle(np.exp(1j*(phi_z - predphi)))
    phase_corr.append(np.abs(dd))
    print(f"q{qkey(a)}->q{qkey(b)}: p_zeta med/max={np.median(pz):.3f}/{np.max(pz):.3f}"
          f"  p_R med/max={np.median(pr):.3f}/{np.max(pr):.3f}"
          f"  m_mag med={np.median(mm):.3f}"
          f"  |phi - 2pi dq.r_mu| med={np.median(np.abs(dd)):.3f} rad")
allpz=np.concatenate(allpz); allpr=np.concatenate(allpr); allm=np.concatenate(allm)
print(f"\nAGG  p_zeta: med={np.median(allpz):.3f} max={np.max(allpz):.3f}")
print(f"AGG  p_R   : med={np.median(allpr):.3f} max={np.max(allpr):.3f}")
print(f"AGG  m_mag : med={np.median(allm):.3f} max={np.max(allm):.3f}")
print(f"AGG  |phi-2pi dq.r_mu|: med={np.median(np.concatenate(phase_corr)):.3f} rad "
      f"(0 => centroid-phase law H3 holds)")

# ---- PART B: band-limit-CONTROLLED magnitude test (common-G) --------------
# Reconstruct every q using ONLY the G-set common to all spheres, so the
# band-limit is identical across q and any |zeta_q| difference is genuine
# q-dependence (not different truncation).
common = None
for q in range(nq):
    s = set(map(tuple, gvec[q][:, :int(ngk[q])].T.tolist()))
    common = s if common is None else (common & s)
common_arr = np.array(sorted(common)).T                # (3, Ncommon)
fi_c = flat_idx(common_arr)
# map, per q, sphere-slot index of each common G (dict per q)
def recon_common(q, mus):
    n = int(ngk[q]); gv = gvec[q][:, :n]
    lut = {tuple(gv[:,j].tolist()): j for j in range(n)}
    slots = np.array([lut[tuple(g)] for g in common_arr.T])
    m=len(mus); box=np.zeros((m,n_rtot),dtype=np.complex128)
    box[:, fi_c] = ZG[q][mus][:, slots]
    box=box.reshape(m,nx,ny,nz)
    R=np.fft.ifftn(box,axes=(1,2,3),norm='backward').reshape(m,n_rtot)
    return R*np.exp(2j*np.pi*(rfrac@qfr[q]))[None,:]
print(f"\n--- PART B: common-G ({common_arr.shape[1]} G's) magnitude test (band-limit fixed) ---")
mc=[]
for (a,b) in pairs:
    if abs(qkey(a)[0]-qkey(b)[0])+abs((qkey(a)[1]-qkey(b)[1])%3) == 0: continue
    Za=recon_common(a,MUS); Zb=recon_common(b,MUS)
    mm=np.linalg.norm(np.abs(Za)-np.abs(Zb),axis=1)/np.linalg.norm(np.abs(Za),axis=1)
    mc.append(mm)
mc=np.concatenate(mc)
print(f"common-G magnitude diff |||z_a|-|z_b|||/|||z_a|||: med={np.median(mc):.3f} "
      f"max={np.max(mc):.3f}  (band-limit now identical => residual is genuine q-dep)")

# ---- PART C: master-zeta V_q prediction (design payoff) --------------------
print("\n--- PART C: predict zeta~_q & V_q at every on-grid q from Gamma master ---")
# master real-space object = reconstruction at Gamma (all mu), all-mu
ALL = np.arange(n_mu)
_,Zgamma = recon(q0, ALL)      # zeta_Gamma(mu,r), all mu  (at Gamma R==zeta)
def predict_zeta_tilde(q):
    ph = np.exp(-2j*np.pi*(rfrac @ qfr[q]))         # exp(-2pi i q.r)
    box = (Zgamma*ph[None,:]).reshape(n_mu,nx,ny,nz)
    G = np.fft.fftn(box, axes=(1,2,3), norm='backward').reshape(n_mu, n_rtot)
    fi = flat_idx(gvec[q]); n=int(ngk[q])
    out = np.zeros((n_mu, ngkmax), dtype=np.complex128)
    out[:, :n] = G[:, fi[:n]]
    return out, n

def kcart2(q):  # |q+G|^2 in (2pi/blat... ) consistent Ry-ish units via bdot
    gv = gvec[q].astype(np.float64)                 # (3,ngkmax)
    kf = gv + qfr[q][:,None]                          # (3,ngkmax) fractional
    # |k|^2 = k . bdot . k  (bdot already the reciprocal metric in Ry units)
    return np.einsum('ig,ij,jg->g', kf, bdot, kf)

for q in range(nq):
    if q==q0: continue
    Zt, n = predict_zeta_tilde(q)
    Za = ZG[q]                                       # actual
    # per-mu global-phase-aligned zeta~ residual (subset for speed)
    sub = ALL[::10]
    ov = np.sum(np.conj(Za[sub,:n])*Zt[sub,:n],1); ph=ov/np.abs(ov)
    rz_ = np.linalg.norm(Za[sub,:n]-ph[:,None]*Zt[sub,:n],axis=1)/np.linalg.norm(Za[sub,:n],axis=1)
    # V_q from actual and predicted, using v=8pi/|q+G|^2 (G!=head)
    k2 = kcart2(q); k2n = k2[:n].copy()
    with np.errstate(divide='ignore'): v = 8*np.pi/k2n
    v[k2n < 1e-8] = 0.0                              # drop head (analytic-separate)
    MU2 = ALL[::16]                                   # 40-centroid V tile
    Za_s = Za[MU2,:n]; Zt_s = Zt[MU2,:n]
    Va = np.conj(Za_s*np.sqrt(v)[None,:]) @ (Za_s*np.sqrt(v)[None,:]).T
    Vt = np.conj(Zt_s*np.sqrt(v)[None,:]) @ (Zt_s*np.sqrt(v)[None,:]).T
    # analytic centroid phase e^{2pi i q.(r_nu - r_mu)}
    cp = np.exp(2j*np.pi*(rmu[MU2] @ qfr[q]))
    Vt_corr = Vt * (np.conj(cp)[:,None]*cp[None,:])   # e^{2pi i q (r_nu-r_mu)}
    def rel(A,B): return float(np.linalg.norm(A-B)/np.linalg.norm(A))
    print(f"q{qkey(q)}: zeta~ p(mu-phase) med={np.median(rz_):.3f}"
          f" | V_q raw rel={rel(Va,Vt):.3f}  V_q +centroid-phase rel={rel(Va,Vt_corr):.3f}"
          f" | diag rel={float(np.linalg.norm(np.diag(Va)-np.diag(Vt))/np.linalg.norm(np.diag(Va))):.3f}")

# ---- sphere band-limit floor ----------------------------------------------
common = None
for q in range(nq):
    s = set(map(tuple, gvec[q][:, :int(ngk[q])].T.tolist()))
    common = s if common is None else (common & s)
print(f"\nsphere sizes ngk={list(ngk)}; common-G across all q = {len(common)} "
      f"(min ngk={int(ngk.min())}); band-limit differs by "
      f"{100*(1-len(common)/int(ngk.min())):.1f}% of smallest sphere")
