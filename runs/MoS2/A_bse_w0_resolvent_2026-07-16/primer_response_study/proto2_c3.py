"""proto2 C3 — control + owner-mandated re-exam of the §3.5 bar under PHYSICAL metrics.

Pipeline = EXACTLY the measured §3.5 leave-one-out (interpolate C_q AND Z_q to a
held-out on-grid q0 with truncated-R Fourier weights, nR=7, then solve C0 ζ = Z0)
but the solve ladder adds the overtly-regularized FIXED-RANK reduced inverse
(one global r; κ_eff = λ_1/λ_r ∈ {1e2, 1e4, 1e6}), and the VERDICT metrics are
the owner-mandated physical contractions:

  PRIMARY  (a) gap-window B-block  B[t,t'] = Σ_{μν} conj(M_t(μ)) V_q0[μ,ν] M_t'(ν)
               with M_cvk(μ) = Σ_s conj(ψ_{c,k-q0,s}(μ)) ψ_{v,k,s}(μ),
               v ∈ top-3 valence, c ∈ bottom-3 conduction, all k  (81 rows at 3x3)
           (b) TDA exciton shift: H(q0) = D - K^d + K^x assembled from stored
               W0_qmunu + enk_full (production convention bse_serial.py:23 —
               K^x = (1/Nk) conj(M) V M^T adds, K^d subtracts, no singlet 2 with
               spinors); swap ONLY the V tile inside K^x; Δλ of lowest 4 in meV.
  SECONDARY tile ‖ΔV‖_F/‖V‖ and the §3.5 random-all-band-pair d*V d
           (verbatim physical_contract.py seed-0 pairs) for continuity.

Verbatim-reused routines (provenance):
  - order-robust C_q rebuild:            interp_study/physical_contract.py:31-39
  - flat_idx/recon/to_sphere/vcoul/make_Vq: interp_study/vq_loo.py:61-105
  - truncated-R Fourier weights (pinv):  interp_study/vq_loo.py:145-154
  - SVD solve ladder raw/tikhonov/rankcut: interp_study/vq_loo.py:156-170
  - random pair set (seed 0, 8 pairs):   interp_study/physical_contract.py:71-77
One deviation, mathematically exact: the solve and the r->sphere projection act
on different axes (μ vs r) and commute, so we solve on the SPHERE representation
Zt = to_sphere(Z_r) instead of full-r Z (per-mode cost 640×~2000 instead of
640×46080).  Verified against the literal vq_loo order below (SELF-CHECK 5).

Fixed-rank reduced inverse (the task formula, implemented via the same Hermitian
SVD as the rankcut rows so the two differ ONLY in globalizing r):
  eigh(C0) = R diag(λ) R^H,  λ_i desc;  ζ̂ = Σ_{i<=r} λ_i^{-1} R_i R_i^H Z0.
  [task's "Rh[:,1:r] Sh_r^-2 Rh[:,1:r]^H" read as top-r eigenpairs, 1-based;
   λ = Sh², so κ_eff = (Sh_1/Sh_r)² = λ_1/λ_r.]
ONE global r per κ target: r_i(q0) = #{λ >= λ_1/κ} on the INTERPOLATED spectra,
r_glob = round(median_q0 r_i); attained κ_eff range reported.  The SAME r_glob is
used for the true-ingredient control rows so truncation-only loss is isolated.

Production mapping note: every eigh/SVD here is a 640×640 single-device call;
in production these become the N_mu² P('x','y')-sharded + cusolvermp/slate FFI
forms (deferred until results warrant, per owner ruling).

READ-ONLY on fixtures and all LORRAX source.  numpy+h5py only.
"""
import sys, time, h5py
import numpy as np

t_start = time.time()
restart = sys.argv[1]
zeta    = sys.argv[2]
label   = sys.argv[3]
outnpz  = sys.argv[4]

RY2MEV = 13605.693122994
def log(msg): print(f"[{label}] {msg}", flush=True)
def relF(a, b): return np.linalg.norm(a - b) / np.linalg.norm(b)

# ---------------------------------------------------------------------------
# load fixtures
# ---------------------------------------------------------------------------
with h5py.File(restart, "r") as f:
    psi    = f["psi_full_y"][()]              # (nk, nb, ns, n_mu)
    kgrid  = f["kgrid"][()].astype(int)
    Vdisk  = f["V_qmunu"][()]                 # (nq, mu, nu) production bare tile (G=0 zeroed)
    W0     = f["W0_qmunu"][()]                # (nq, mu, nu) production static screened tile
    enk    = f["enk_full"][()]                # (nk, nb) Ry
    vhead  = float(np.real(f["vhead"][()]))
with h5py.File(zeta, "r") as f:
    ZG   = f["zeta_q_G"][()]                  # (nq, n_mu, ngkmax)
    gvec = f["isdf_header/gvec_components"][()]
    ngk  = f["isdf_header/ngk"][()]
    fg   = f["mf_header/gspace/FFTgrid"][()]
    qfr  = f["mf_header/kpoints/rk"][()]      # (nq,3) fractional q (BGW wrap)
    bdot = f["mf_header/crystal/bdot"][()]
    ifmax  = int(np.max(f["mf_header/kpoints/ifmax"][()]))
    celvol = float(np.real(f["mf_header/crystal/celvol"][()]))
assert bdot[0, 2] == 0 and bdot[1, 2] == 0, "slab kernel assumes b3 orthogonal to b1,b2"
b3len = float(np.sqrt(bdot[2, 2]))
zc = np.pi / b3len                            # = Lz/2 (production compute_vcoul.py:143)

nk, nb, ns, n_mu = psi.shape
kg = kgrid
nq = int(kg[0] * kg[1] * kg[2])
nx, ny, nz = [int(x) for x in fg]; n_rtot = nx * ny * nz
ngkmax = ZG.shape[2]
assert nk == nq == ZG.shape[0]
log(f"nq={nq} n_mu={n_mu} nb={nb} ns={ns} FFT={nx}x{ny}x{nz}={n_rtot} ngkmax={ngkmax} vhead={vhead:.1f}")

# ---------------------------------------------------------------------------
# q-grid index maps (order-robust: never assume qfr is C-order)
# ---------------------------------------------------------------------------
ktrip = np.rint(qfr * kg[None, :]).astype(int) % kg[None, :]     # (nq,3) integer triples
_d = qfr - ktrip / kg[None, :]
_d -= np.rint(_d)                                                # wrapped distance to grid point
assert np.max(np.abs(_d)) < 1e-9, "qfr not on the kgrid"
trip2idx = {tuple(t): i for i, t in enumerate(ktrip)}
assert len(trip2idx) == nq, "duplicate k-points"
iG = trip2idx[(0, 0, 0)]
log(f"q-grid map OK; Gamma at flat index {iG}")
kmq  = np.array([[trip2idx[tuple((ktrip[k] - ktrip[q]) % kg)] for q in range(nq)] for k in range(nq)])  # kmq[k,q] = idx(k-q)
qkk  = np.array([[trip2idx[tuple((ktrip[k] - ktrip[kp]) % kg)] for kp in range(nq)] for k in range(nq)])  # qkk[k,k'] = idx(k-k')

# ---------------------------------------------------------------------------
# C_q rebuild — VERBATIM physical_contract.py:31-39 (order-robust DFT over qfr)
# ---------------------------------------------------------------------------
psiX = np.conj(psi).transpose(0, 3, 1, 2)
P = np.einsum('kmna,knbr->karmb', psiX, psi, optimize=True)
Rall = np.array([[rx, ry, rz] for rx in range(kg[0]) for ry in range(kg[1]) for rz in range(kg[2])])
Rw = ((Rall + kg // 2) % kg) - (kg // 2)
EqR = np.exp(2j * np.pi * (qfr @ Rw.T))
P_R = (EqR.T @ P.reshape(nq, -1)).reshape(len(Rw), ns, n_mu, n_mu, ns)
C_R = np.einsum('ravmb,ravmb->rvm', np.conj(P_R), P_R, optimize=True)
C_q = np.transpose(((np.exp(-2j * np.pi * (qfr @ Rw.T)) / nq) @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
del P, P_R, C_R
log(f"C_q rebuilt ({time.time()-t_start:.0f}s)")

# ---------------------------------------------------------------------------
# recon / to_sphere / vcoul / make_Vq — VERBATIM vq_loo.py:61-105
# ---------------------------------------------------------------------------
def flat_idx(gv):
    gx = gv[0] % nx; gy = gv[1] % ny; gz = gv[2] % nz
    return ((gx * ny) + gy) * nz + gz
rx = np.arange(nx) / nx; ry = np.arange(ny) / ny; rz = np.arange(nz) / nz
RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing='ij')
rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)

def recon(q):
    box = np.zeros((n_mu, n_rtot), dtype=np.complex128)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    box[:, fi[:n]] = ZG[q][:, :n]
    R = np.fft.ifftn(box.reshape(n_mu, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(n_mu, n_rtot)
    return R * np.exp(2j * np.pi * (rfrac @ qfr[q]))[None, :]

def to_sphere(zeta_r, q):
    ph = np.exp(-2j * np.pi * (rfrac @ qfr[q]))
    box = np.fft.fftn((zeta_r * ph[None, :]).reshape(-1, nx, ny, nz), axes=(1, 2, 3), norm='backward').reshape(-1, n_rtot)
    fi = flat_idx(gvec[q]); n = int(ngk[q])
    out = np.zeros((zeta_r.shape[0], ngkmax), dtype=np.complex128)
    out[:, :n] = box[:, fi[:n]]
    return out

def vcoul(q):
    """3D bare v = 8π/|q+G|² — VERBATIM vq_loo.py:92-99 (§3.5 continuity kernel)."""
    gv = gvec[q].astype(np.float64)
    kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    with np.errstate(divide='ignore'):
        v = 8 * np.pi / k2
    v[k2 < 1e-8] = 0.0
    return v, int(ngk[q])

def vcoul_slab(q):
    """Production 2D slab-truncated v (compute_vcoul.py:178-183):
    v = (8π/|q+G|²)·[1 − e^{−zc·k_xy}·cos(k_z·zc)]/V_cell, zc = π/|b3| = Lz/2.
    PRIMARY-metric kernel — matches the kernel that built the stored V/W tiles."""
    gv = gvec[q].astype(np.float64)
    kf = gv + qfr[q][:, None]
    k2 = np.einsum('ig,ij,jg->g', kf, bdot, kf)
    kxy = np.sqrt(np.einsum('ig,ij,jg->g', kf[:2], bdot[:2, :2], kf[:2]))
    kz = kf[2] * b3len
    f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = (8 * np.pi / k2) * f2d / celvol
    v[k2 < 1e-12] = 0.0
    return v, int(ngk[q])

def make_Vq(zt, q, kern=vcoul):
    v, n = kern(q)
    A = zt[:, :n] * np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T

# SELF-CHECK 1: recon/forward round trip
_zt = to_sphere(recon(iG), iG); _n0 = int(ngk[iG])
rt = np.linalg.norm(_zt[:, :_n0] - ZG[iG, :, :_n0]) / np.linalg.norm(ZG[iG, :, :_n0])
log(f"SELF-CHECK 1 recon/forward round-trip on sphere(Gamma): {rt:.2e}")
assert rt < 1e-12

# Z_q(mu,r) = C_q @ zeta_q(mu,r)  (code's ZCT identity), + sphere rep of true Z
Z_r = np.zeros((nq, n_mu, n_rtot), dtype=np.complex128)
Zt_true = np.zeros((nq, n_mu, ngkmax), dtype=np.complex128)
for q in range(nq):
    zr = recon(q)
    Z_r[q] = C_q[q] @ zr
    Zt_true[q] = to_sphere(Z_r[q], q)
    del zr
log(f"Z_q built ({Z_r.nbytes/1e9:.1f} GB) ({time.time()-t_start:.0f}s)")

# SELF-CHECK 2 (null test, machine precision): true C, true Z -> stored zeta / direct V
Vq_direct = np.stack([make_Vq(ZG[q], q) for q in range(nq)])            # 3D bare (§3.5 continuity)
Vq_dslab  = np.stack([make_Vq(ZG[q], q, vcoul_slab) for q in range(nq)])  # slab (PRIMARY metrics)
q_chk = 1
z0 = np.linalg.solve(C_q[q_chk], Zt_true[q_chk])
nchk = int(ngk[q_chk])
nul = np.linalg.norm(z0[:, :nchk] - ZG[q_chk, :, :nchk]) / np.linalg.norm(ZG[q_chk, :, :nchk])
nulV = relF(make_Vq(z0, q_chk), Vq_direct[q_chk])
log(f"SELF-CHECK 2 grid-point solve (true C, true Z, q={q_chk}): zeta relF={nul:.2e}  V relF={nulV:.2e}")

# SELF-CHECK 3: Gamma-tile alignment vs on-disk production V_qmunu.  With the
# production slab kernel this should be scale~1 and small relF — non-circular
# validation of recon/to_sphere/make_Vq vs the production V pipeline.
sc3 = float((np.vdot(Vdisk[iG].ravel(), Vq_direct[iG].ravel())
             / np.vdot(Vq_direct[iG].ravel(), Vq_direct[iG].ravel())).real)
scS = float((np.vdot(Vdisk[iG].ravel(), Vq_dslab[iG].ravel())
             / np.vdot(Vq_dslab[iG].ravel(), Vq_dslab[iG].ravel())).real)
log(f"SELF-CHECK 3 Gamma tile vs disk V_qmunu: 3D-bare sc={sc3:.4e} relF={relF(sc3*Vq_direct[iG], Vdisk[iG]):.2e}"
    f"  |  slab sc={scS:.6f} relF={relF(scS*Vq_dslab[iG], Vdisk[iG]):.2e}")
sc = 1.0   # slab kernel already carries 1/V_cell — production scale directly

# ---------------------------------------------------------------------------
# gap window + physical pair rows
# ---------------------------------------------------------------------------
n_occ = ifmax                                # mf_header/kpoints/ifmax (BGW: highest occupied band)
Egap = np.min(enk[:, n_occ]) - np.max(enk[:, n_occ - 1])
log(f"band window: n_occ={n_occ} (from ifmax; 0-indexed VBM={n_occ-1}) indirect gap={Egap:.4f} Ry = {Egap*RY2MEV/1e3:.3f} eV")
assert 0.03 < Egap < 0.4, "gap suspicious — check enk units / ifmax"
vbands = [n_occ - 3, n_occ - 2, n_occ - 1]
cbands = [n_occ, n_occ + 1, n_occ + 2]
# degeneracy hygiene at the metric-window edges (BGW TOL_Degeneracy = 1e-6 Ry)
edge_v = np.min(enk[:, n_occ - 3] - enk[:, n_occ - 4])
edge_c = np.min(enk[:, n_occ + 3] - enk[:, n_occ + 2])
log(f"window-edge degeneracy: min_k e[v_top3_edge]-e[below]={edge_v:.3e} Ry, "
    f"min_k e[above]-e[c_bot3_edge]={edge_c:.3e} Ry "
    f"{'(SPLIT MULTIPLET at an edge — metric-window hygiene flag)' if min(edge_v, edge_c) < 1e-6 else '(edges clean)'}")

nvw, ncw = len(vbands), len(cbands)
ntr = ncw * nvw * nk        # 81 at 3x3
psi_c_all = psi[:, cbands, :, :]     # (nk, 3, ns, mu)
psi_v_all = psi[:, vbands, :, :]

def build_Mx(q0):
    """M_t(mu), t=(c,v,k): sum_s conj(psi_{c,k-q0,s}(mu)) psi_{v,k,s}(mu)."""
    pc = psi_c_all[kmq[:, q0]]                                  # (nk, 3, ns, mu) at k-q0
    M = np.einsum('kcsm,kvsm->kcvm', np.conj(pc), psi_v_all)    # (nk, c, v, mu)
    return M.transpose(1, 2, 0, 3).reshape(ntr, n_mu)           # t=(c,v,k) C-order

# secondary continuity metric — VERBATIM physical_contract.py:71-77 random pairs
rng = np.random.default_rng(0)
Drand = []
for _ in range(8):
    k = rng.integers(nk); n1 = rng.integers(nb); n2 = rng.integers(nb)
    d = np.einsum('sm,sm->m', np.conj(psi[k, n1]), psi[k, n2])
    Drand.append(d / np.linalg.norm(d))
Drand = np.array(Drand)

def phys_scalars(zt, q):
    v, n = vcoul(q)
    A = zt[:, :n] * np.sqrt(v[:n])[None, :]
    proj = np.conj(Drand) @ A
    return np.sum(np.abs(proj) ** 2, axis=1)
S_direct = np.array([phys_scalars(ZG[q], q) for q in range(nq)])

# ---------------------------------------------------------------------------
# TDA exciton Hamiltonian pieces at exciton momentum q0 (transitions v,k -> c,k-q0)
# H = diag(D) + K^x - K^d, production convention bse_serial.py:23-70:
#   K^x[t,t'] = (1/Nk) conj(M_t(mu)) V[mu,nu] M_t'(nu)
#   K^d[t,t'] = (1/Nk) [sum_s conj(psi_{c,k-q0,s}(M)) psi_{c',k'-q0,s}(M)] W_{k-k'}[M,N]
#                      [sum_s psi_{v,k,s}(N) conj(psi_{v',k',s}(N))]
# W tiles / D energies are IDENTICAL between direct-V and interp-V variants: only
# the V tile inside K^x is swapped (the owner-mandated control variable).
# Head channels: W-head and (at Gamma) V-head rank-1 terms are omitted in BOTH
# variants (production stores them separately; identical in both -> no effect on
# the swap metric; absolute binding slightly shifted, noted in report).
# ---------------------------------------------------------------------------
# SELF-CHECK 4: stored-tile symmetry properties the K^d Hermiticity relies on:
# per-tile Hermiticity W_q = W_q^H and TRS conj(W_{-q}) = W_q (same for V).
mq = np.array([trip2idx[tuple((-ktrip[q]) % kg)] for q in range(nq)])
w_h = max(relF(W0[q].conj().T, W0[q]) for q in range(nq))
w_t = max(relF(np.conj(W0[mq[q]]), W0[q]) for q in range(nq))
v_t = max(relF(np.conj(Vq_dslab[mq[q]]), Vq_dslab[q]) for q in range(nq))
log(f"SELF-CHECK 4 tile symmetry: max relF(W_q^H,W_q)={w_h:.2e}  "
    f"max relF(conj(W_-q),W_q)={w_t:.2e}  max relF(conj(V_-q),V_q)={v_t:.2e}")

Bv = np.einsum('kvsN,lwsN->kvlwN', psi_v_all, np.conj(psi_v_all))   # (k,v,k',v',N) valence leg

def build_H_pieces(q0):
    pc = psi_c_all[kmq[:, q0]]                                      # (nk,3,ns,mu) conduction at k-q0
    Ac = np.einsum('kcsM,ldsM->kcldM', np.conj(pc), pc)             # (k,c,k',c',M)
    Kd = np.zeros((ncw, nvw, nk, ncw, nvw, nk), dtype=np.complex128)
    for k in range(nk):
        for kp in range(nk):
            Wq = W0[qkk[k, kp]]
            blk = np.einsum('xyM,MN,vwN->xvyw', Ac[k, :, kp, :, :], Wq, Bv[k, :, kp, :, :], optimize=True)
            Kd[:, :, k, :, :, kp] = blk
    Kd = Kd.reshape(ntr, ntr) / nk
    D = np.array([enk[kmq[k, q0], c] - enk[k, v]
                  for c in cbands for v in vbands for k in range(nk)])
    return D, Kd

def exciton_eigs(D, Kd, Mx, Vtile, nlow=4):
    Kx = (np.conj(Mx) @ (sc * Vtile) @ Mx.T) / nk
    H = np.diag(D) + Kx - Kd
    herm = np.linalg.norm(H - H.conj().T) / np.linalg.norm(H)
    ev = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    return ev[:nlow], herm

# ---------------------------------------------------------------------------
# leave-one-out machinery — weights VERBATIM vq_loo.py:134-154
# ---------------------------------------------------------------------------
def wrap(n, N): return n - N * ((2 * n) > N)
Rvecs = np.array([[wrap(ix, kg[0]), wrap(iy, kg[1]), wrap(iz, kg[2])]
                  for ix in range(kg[0]) for iy in range(kg[1]) for iz in range(kg[2])])
adot = np.linalg.inv(bdot) * (2 * np.pi) ** 2
Rdist = np.sqrt(np.abs(np.einsum('ri,ij,rj->r', Rvecs, adot, Rvecs)))
Rsort = Rvecs[np.argsort(Rdist)]
NR = 7

def interp_weights(q0, nR):
    train = [q for q in range(nq) if q != q0]
    Rset = Rsort[:nR]
    F = np.exp(-2j * np.pi * (qfr[train] @ Rset.T))
    f0 = np.exp(-2j * np.pi * (qfr[q0] @ Rset.T))
    return train, f0 @ np.linalg.pinv(F)

Cq_flat = C_q.reshape(nq, -1)
Zr_flat = Z_r.reshape(nq, -1)

# solve ladder — raw/tikhonov/rankcut VERBATIM vq_loo.py:156-170; fixedr added
def solve_factors(Cmat):
    Ch = 0.5 * (Cmat + Cmat.conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    return U, s, Vh

def apply_solve(U, s, Vh, Zmat, mode, lam=0.0, r=None):
    n = len(s)
    if mode == "raw":
        sinv = 1.0 / s
    elif mode == "tikhonov":
        alpha = lam * (s.sum() / n)
        sinv = 1.0 / (s + alpha)
    elif mode == "rankcut":
        sinv = np.where(s > lam * s[0], 1.0 / np.where(s > lam * s[0], s, 1), 0.0)
    elif mode == "fixedr":
        sinv = np.zeros_like(s); sinv[:r] = 1.0 / s[:r]
    else:
        raise ValueError(mode)
    return (Vh.conj().T * sinv) @ (U.conj().T @ Zmat)

# interpolated ingredients per q0 (computed once, all rows reuse)
C0_all = np.zeros((nq, n_mu, n_mu), dtype=np.complex128)
Zt0_all = np.zeros((nq, n_mu, ngkmax), dtype=np.complex128)
for q0 in range(nq):
    train, w = interp_weights(q0, NR)
    C0_all[q0] = (w @ Cq_flat[train]).reshape(n_mu, n_mu)
    Zt0_all[q0] = to_sphere((w @ Zr_flat[train]).reshape(n_mu, n_rtot), q0)
log(f"interpolated ingredients built ({time.time()-t_start:.0f}s)")

# SELF-CHECK 5: sphere-side solve == literal vq_loo full-r order (raw, q0=1)
train, w = interp_weights(1, NR)
Z0_r = (w @ Zr_flat[train]).reshape(n_mu, n_rtot)
U1, s1, Vh1 = solve_factors(C0_all[1])
zt_lit = to_sphere(apply_solve(U1, s1, Vh1, Z0_r, "raw"), 1)
zt_opt = apply_solve(U1, s1, Vh1, Zt0_all[1], "raw")
comm = np.linalg.norm(zt_lit - zt_opt) / np.linalg.norm(zt_lit)
log(f"SELF-CHECK 5 solve/to_sphere commutation (raw, q0=1): relF={comm:.2e}")
assert comm < 1e-10

# cache Hermitian-SVD factors (reused by all ladder rows)
fac_interp = [solve_factors(C0_all[q0]) for q0 in range(nq)]
fac_true = [solve_factors(C_q[q0]) for q0 in range(nq)]

# global fixed r per kappa target, from the INTERPOLATED spectra (task spec)
spec_i = np.stack([fac_interp[q0][1] for q0 in range(nq)])
spec_t = np.stack([fac_true[q0][1] for q0 in range(nq)])
KAPPAS = [1e2, 1e4, 1e6]
r_glob = {}
for kap in KAPPAS:
    r_i = np.array([int(np.sum(spec_i[q0] >= spec_i[q0][0] / kap)) for q0 in range(nq)])
    r_glob[kap] = int(round(np.median(r_i)))
    att_i = spec_i[:, 0] / spec_i[np.arange(nq), r_glob[kap] - 1]
    att_t = spec_t[:, 0] / spec_t[np.arange(nq), r_glob[kap] - 1]
    log(f"kappa target {kap:.0e}: per-q0 r in [{r_i.min()},{r_i.max()}], GLOBAL r={r_glob[kap]}/{n_mu}; "
        f"attained kappa_eff interp [{att_i.min():.1e},{att_i.max():.1e}] true [{att_t.min():.1e},{att_t.max():.1e}]")
cond_t = spec_t[:, 0] / spec_t[:, -1]
log(f"cond(C_q) true: min {cond_t.min():.1e} med {np.median(cond_t):.1e} max {cond_t.max():.1e}")

# ---------------------------------------------------------------------------
# JUNK-CLEANED training ingredients (mechanism ablation, added after the
# diagnostic finding): the stored per-q zeta solves carry INDEPENDENT
# near-null-space solver noise (non-covariant: TRS violated at O(1) in tile
# norm; disk V_qmunu is IBZ+sym-unfolded and clean).  Z = C zeta_stored bakes
# that junk into the training data, and junk is NOT a smooth function of q, so
# the interpolant chases noise.  Ablation: project each TRAINING Z_q onto the
# top-r eigenspace of its own TRUE C_q (Z_clean = U_r U_r^H Z — exactly the
# "TRUE fixedr" object validated physically at 3e-3), THEN interpolate, THEN
# fixed-r solve.  If this reaches few-percent physical error the C3 failure was
# curable data hygiene; if it still fails, the physical subspace itself rotates
# too fast for a 7-point stencil (the C1/C2 frame-transport claim).
# ---------------------------------------------------------------------------
Zt0_clean = {}
Zc_flat = np.zeros_like(Zr_flat)
for kap in KAPPAS:
    for q in range(nq):
        U, s, Vh = fac_true[q]
        rq = int(np.sum(s >= s[0] / kap))
        Ur = U[:, :rq]
        Zc_flat[q] = (Ur @ (Ur.conj().T @ Z_r[q])).reshape(-1)
    Ztc = np.zeros((nq, n_mu, ngkmax), dtype=np.complex128)
    for q0 in range(nq):
        train, w = interp_weights(q0, NR)
        Ztc[q0] = to_sphere((w @ Zc_flat[train]).reshape(n_mu, n_rtot), q0)
    Zt0_clean[kap] = Ztc
    log(f"junk-cleaned training Z (kappa {kap:.0e}) interpolated ({time.time()-t_start:.0f}s)")
del Z_r, Zr_flat, Zc_flat

# ---------------------------------------------------------------------------
# assemble per-q0 direct references (metrics baselines)
# ---------------------------------------------------------------------------
Mx_all, B_direct, H_pieces, ev_direct, herms = [], [], [], [], []
for q0 in range(nq):
    Mx = build_Mx(q0)
    Mx_all.append(Mx)
    B_direct.append(np.conj(Mx) @ Vq_dslab[q0] @ Mx.T)     # PRIMARY: slab kernel
    D, Kd = build_H_pieces(q0)
    H_pieces.append((D, Kd))
    ev, herm = exciton_eigs(D, Kd, Mx, Vq_dslab[q0])
    ev_direct.append(ev); herms.append(herm)
    log(f"H_direct(q0={q0}) herm-asym={herm:.2e} lowest4(eV)=" +
        " ".join(f"{e*RY2MEV/1e3:.4f}" for e in ev))
ev_direct = np.array(ev_direct)
log(f"H Hermiticity over all q0: max asym={max(herms):.2e} (validates K^d/K^x index+conjugation conventions)")
Dmin = min(H_pieces[q0][0].min() for q0 in range(nq))
log(f"exciton sanity: min transition D={Dmin*RY2MEV/1e3:.3f} eV; "
    f"lowest exciton over q0={ev_direct[:,0].min()*RY2MEV/1e3:.3f} eV "
    f"(binding vs D_min: {(Dmin-ev_direct[:,0].min())*RY2MEV:.0f} meV)")

# top-decile mask per q0 (on |B_direct|)
td_masks = []
for q0 in range(nq):
    a = np.abs(B_direct[q0])
    td_masks.append(a >= np.quantile(a, 0.9))

# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
ROWS = [
    ("TRUE raw (null test)",      "true", "raw",     0.0,  None),
    ("TRUE fixedr k1e6",          "true", "fixedr",  0.0,  1e6),
    ("TRUE fixedr k1e4",          "true", "fixedr",  0.0,  1e4),
    ("TRUE fixedr k1e2",          "true", "fixedr",  0.0,  1e2),
    ("INTERP raw",                "interp", "raw",     0.0,  None),
    ("INTERP tikhonov 1e-6",      "interp", "tikhonov", 1e-6, None),
    ("INTERP rankcut 1e-8",       "interp", "rankcut",  1e-8, None),
    ("INTERP rankcut 1e-6",       "interp", "rankcut",  1e-6, None),
    ("INTERP rankcut 1e-4",       "interp", "rankcut",  1e-4, None),
    ("INTERP rankcut 1e-2",       "interp", "rankcut",  1e-2, None),
    ("INTERP fixedr k1e6",        "interp", "fixedr",  0.0,  1e6),
    ("INTERP fixedr k1e4",        "interp", "fixedr",  0.0,  1e4),
    ("INTERP fixedr k1e2",        "interp", "fixedr",  0.0,  1e2),
    ("CLEAN-INTERP fixedr k1e6",  "clean",  "fixedr",  0.0,  1e6),
    ("CLEAN-INTERP fixedr k1e4",  "clean",  "fixedr",  0.0,  1e4),
    ("CLEAN-INTERP fixedr k1e2",  "clean",  "fixedr",  0.0,  1e2),
]

results = {}
for name, src, mode, lam, kap in ROWS:
    tile = np.zeros(nq); rand = np.zeros(nq)
    brel = np.zeros(nq); btd = np.zeros(nq); btd_max = np.zeros(nq)
    dl_max4 = np.zeros(nq); dl_1 = np.zeros(nq)
    for q0 in range(nq):
        if src == "true":
            Zt = Zt_true[q0]; U, s, Vh = fac_true[q0]
        elif src == "interp":
            Zt = Zt0_all[q0]; U, s, Vh = fac_interp[q0]
        else:                                   # "clean": cleaned-Z interp, interp C
            Zt = Zt0_clean[kap][q0]; U, s, Vh = fac_interp[q0]
        r = r_glob[kap] if mode == "fixedr" else None
        zt = apply_solve(U, s, Vh, Zt, mode, lam, r)
        Vi = make_Vq(zt, q0)                                # 3D bare — §3.5 continuity
        Vis = make_Vq(zt, q0, vcoul_slab)                   # slab — PRIMARY metrics
        tile[q0] = relF(Vi, Vq_direct[q0])
        si = phys_scalars(zt, q0)
        rand[q0] = np.median(np.abs(si - S_direct[q0]) / np.abs(S_direct[q0]))
        Mx = Mx_all[q0]
        Bi = np.conj(Mx) @ Vis @ Mx.T
        brel[q0] = relF(Bi, B_direct[q0])
        m = td_masks[q0]
        ee = np.abs(Bi[m] - B_direct[q0][m]) / np.abs(B_direct[q0][m])
        btd[q0] = np.median(ee); btd_max[q0] = np.max(ee)
        D, Kd = H_pieces[q0]
        ev, _ = exciton_eigs(D, Kd, Mx, Vis)
        dl = np.abs(ev - ev_direct[q0]) * RY2MEV
        dl_max4[q0] = dl.max(); dl_1[q0] = dl[0]
    results[name] = dict(tile=tile, rand=rand, brel=brel, btd=btd, btd_max=btd_max,
                         dl_max4=dl_max4, dl_1=dl_1)
    log(f"row done: {name} ({time.time()-t_start:.0f}s)")

# ---------------------------------------------------------------------------
# report tables
# ---------------------------------------------------------------------------
def med(x): return np.median(x)
print(f"\n[{label}] ===== C3 LADDER — MoS2 3x3 LOO nR={NR} "
      f"(med over {nq} held-out q0; max in parens) =====")
print("  tile+randpair: 3D-bare v (verbatim §3.5); B-block+exciton: production slab v")
hdr = (f"{'row':>24} | {'tile relF':>10} | {'randpair d*Vd':>13} | "
       f"{'B relF':>10} {'B max':>9} | {'B topdec med':>12} | {'dLambda_1..4 meV':>18}")
print(hdr); print("-" * len(hdr))
for name, *_ in ROWS:
    R = results[name]
    print(f"{name:>24} | {med(R['tile']):>10.3e} | {med(R['rand']):>13.3e} | "
          f"{med(R['brel']):>10.3e} {np.max(R['brel']):>9.2e} | {med(R['btd']):>12.3e} | "
          f"{med(R['dl_max4']):>8.2f} (max {np.max(R['dl_max4']):>8.2f})")
print(f"\n[{label}] 3.5-bar continuity check (must reproduce the logged study):")
print(f"  expected tile med: raw 3.7e6 | rc1e-6 1.2e4 | rc1e-4 1.1e1 | rc1e-2 1.00")
print(f"  measured tile med: raw {med(results['INTERP raw']['tile']):.1e} | "
      f"rc1e-6 {med(results['INTERP rankcut 1e-6']['tile']):.1e} | "
      f"rc1e-4 {med(results['INTERP rankcut 1e-4']['tile']):.1e} | "
      f"rc1e-2 {med(results['INTERP rankcut 1e-2']['tile']):.2f}")
print(f"  expected randpair med: raw 7.4e4 | rc1e-6 1.5e3 | rc1e-4 2.1e1 | rc1e-2 0.89")
print(f"  measured randpair med: raw {med(results['INTERP raw']['rand']):.1e} | "
      f"rc1e-6 {med(results['INTERP rankcut 1e-6']['rand']):.1e} | "
      f"rc1e-4 {med(results['INTERP rankcut 1e-4']['rand']):.1e} | "
      f"rc1e-2 {med(results['INTERP rankcut 1e-2']['rand']):.2f}")

np.savez(outnpz,
         rows=np.array([r[0] for r in ROWS]),
         **{f"{k}_{r[0].replace(' ', '_')}": results[r[0]][k]
            for r in ROWS for k in ("tile", "rand", "brel", "btd", "btd_max", "dl_max4", "dl_1")},
         ev_direct=ev_direct, sc=sc, sc3=sc3, scS=scS,
         r_glob=np.array([r_glob[k] for k in KAPPAS]),
         kappas=np.array(KAPPAS), cond_true=cond_t, Egap=Egap, n_occ=n_occ)
log(f"saved {outnpz}; total {time.time()-t_start:.0f}s")
