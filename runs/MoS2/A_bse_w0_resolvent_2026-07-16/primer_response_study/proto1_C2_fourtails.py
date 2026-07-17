"""proto1_C2_fourtails — C2 construction, stage 1: global BZ-periodic frame +
the response's own falsification diagnostic (sec 10C): four R-space tails.

Pipeline:
  1. load fixture, order-robust C_q rebuild, self-check battery (gate).
  2. non-circular fit-convention check: Z from WFN u's (torus vs phys rows),
     solve C zeta = Z on an r-subset, compare to stored zeta (recon).
  3. frames: eigh(C_q) -> R,S at FIXED global rank; spectra dump.
  4. links: T_j(q) = polar(S^-1 R^H H R' S'^-1) for +b1/+b2 edges on the
     wrapped torus; band transport t = polar(<u|u'>_G) (WFN primary) with
     centroid-quadrature fallback delta; principal cosines per edge (diag B).
  5. plaquette holonomies ||W_box - I|| (diag D).
  6. global gauge: row pass (axis 1, q2=0) with seam Wilson W_row log-
     distributed; column passes (axis 2, per q1) with branch continuity +
     det-winding (topological obstruction detector).
  7. gauge-randomization covariance check (links invariant under random
     within-multiplet band unitaries + phases).
  8. TRS check: V_{-q} == conj(V_q) (stored zeta); VcSR spectra at +-q.
  9. FOUR TAILS per-shell ||.||_F table (lab frame primary, cell-periodic
     secondary): raw zeta_R | untransported Phi_R | transported Phi~_R |
     transported VcSR_R (alpha ladder, w/ and w/o taper); C_R reference.

Verdict rule (from the task): if transported Phi~_R stays flat after the
seam/self-checks close, the response's central claim is dead at these
densities by its own criterion; if it decays toward C_R's profile, C2
proceeds to the interpolation stage (proto1_C2_loo.py).

Production mapping note: eigh/SVD/polar on (n_mu,n_mu) -> N_mu^2 P('x','y')
sharded cusolvermp/slate FFI; per-k H accumulation -> sharded zgemm scan.

Run: JID=<jid> ./lxrun_free.sh <thisdir> python3 proto1_C2_fourtails.py MoS2_3x3
"""
import sys, time
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                   "A_bse_w0_resolvent_2026-07-16/primer_response_study")
from proto1_prep import (Fixture, relF, polar, frames_from_C, matlog_u,
                         run_selfchecks)

t00 = time.time()
name = sys.argv[1] if len(sys.argv) > 1 else "MoS2_3x3"
DO_ZCHECK = "--no-zcheck" not in sys.argv
fx = Fixture(name)
print(f"[{name}] nq={fx.nq} nb={fx.nb} ns={fx.ns} n_mu={fx.n_mu} "
      f"FFT={fx.nx}x{fx.ny}x{fx.nz} nv={fx.nv} zeta_cut={fx.zeta_cutoff} Ry",
      flush=True)
kg = fx.kgrid
N1, N2, N3 = int(kg[0]), int(kg[1]), int(kg[2])
assert N3 == 1, "construction below is 2D (N3=1) — MoS2 fixtures"

# --------------------------------------------------------------------------
# 1. C_q + self-checks
# --------------------------------------------------------------------------
C_q = fx.build_Cq()
print(f"[{name}] C_q rebuilt ({time.time()-t00:.0f}s). Self-checks:", flush=True)
rep = run_selfchecks(fx, C_q)

el = fx.load_wfn()["el"]
print(f"  [check] enk_full_vs_WFN_el (max|d|, Ry)          "
      f"{np.max(np.abs(fx.enk - el[:, :fx.nb])):.3e}")

# degeneracy-hygiene report at the window top (primer closed-shell rule)
TOLD = 1e-6
gap_top = np.array([el[k, fx.nb] - el[k, fx.nb - 1] for k in range(fx.nk)])
print(f"  [check] window-top gap el[nb]-el[nb-1]: min {gap_top.min():.3e} Ry "
      f"({'CLOSED SHELL' if gap_top.min() > TOLD else '** SPLIT MULTIPLET **'})",
      flush=True)

# --------------------------------------------------------------------------
# 2. Non-circular convention check (torus vs phys rows) at q=1
# --------------------------------------------------------------------------
if DO_ZCHECK:
    # v2: compare ON THE SPHERE. The stored zeta is band-limited to sphere(q);
    # the least-squares solve is diagonal in the r/G index (C acts on mu only)
    # so projection commutes with the solve: compare C^-1 Z~|_sphere against
    # the stored ZG directly. Z~(mu,G) = sum_k X_k^H fftn(A_k)|_sphere with
    # A_k the torus u-rows (cell-periodic body; NO Bloch strip — stored
    # zeta~ = fftn of the u-frame body).
    q = 1
    nsl = int(fx.ngk[q])
    fi_q = fx.flat_idx(fx.gvec[q])[:nsl]
    for conv in ("torus", "phys"):
        Zs = np.zeros((fx.n_mu, nsl), dtype=np.complex128)
        Cacc = np.zeros((fx.n_mu, fx.n_mu), dtype=np.complex128)
        for k in range(fx.nk):
            kq, G0k = fx.kq_index(k, q)
            ukq = fx.u_grid(kq, nbmax=fx.nb)
            uk = fx.u_grid(k, nbmax=fx.nb)
            A_k = np.einsum("nsr,Msr->nMr", np.conj(ukq), uk).reshape(-1, fx.n_rtot)
            X_k = A_k[:, fx.rmu_flat].copy()
            if conv == "phys" and np.any(G0k):
                A_k *= np.exp(-2j * np.pi * (fx.rfrac @ G0k.astype(float)))[None, :]
                X_k = A_k[:, fx.rmu_flat].copy()
            Ag = np.fft.fftn(A_k.reshape(-1, fx.nx, fx.ny, fx.nz),
                             axes=(1, 2, 3), norm="backward").reshape(-1, fx.n_rtot)[:, fi_q]
            Zs += np.conj(X_k.T) @ Ag
            Cacc += np.conj(X_k.T) @ X_k
        lam, R = np.linalg.eigh(0.5 * (Cacc + Cacc.conj().T))
        keep = lam > lam[-1] * 1e-14
        zsol = (R[:, keep] / lam[keep][None, :]) @ (np.conj(R[:, keep].T) @ Zs)
        err = relF(zsol, fx.ZG[q][:, :nsl])
        SS = np.sqrt(lam[keep])[::-1]
        RR = R[:, keep][:, ::-1]
        errp = relF(SS[:, None] * (np.conj(RR.T) @ zsol),
                    SS[:, None] * (np.conj(RR.T) @ fx.ZG[q][:, :nsl]))
        print(f"  [zcheck q={q}] {conv:>5s}-convention C^-1 Z|sphere vs stored "
              f"zeta~: raw relF = {err:.3e}  S-weighted(Phi) relF = {errp:.3e}",
              flush=True)

# --------------------------------------------------------------------------
# 3. frames at fixed global rank
# --------------------------------------------------------------------------
spec = np.zeros((fx.nq, fx.n_mu))
for q in range(fx.nq):
    spec[q] = np.linalg.eigvalsh(0.5 * (C_q[q] + C_q[q].conj().T))[::-1]
print(f"\n[{name}] C_q spectra: lam_max range "
      f"[{spec[:,0].min():.3e},{spec[:,0].max():.3e}]")
for frac in (1e-6, 1e-8, 1e-10, 1e-12):
    cnt = [int(np.sum(spec[q] > frac * spec[q, 0])) for q in range(fx.nq)]
    print(f"    lam/lam1 > {frac:.0e}: n per q = min {min(cnt)} max {max(cnt)}")
neg = int(np.sum(spec <= 0))
print(f"    nonpositive eigenvalues across all q: {neg}")
RANK = min(int(np.sum(spec[q] > 0)) for q in range(fx.nq))
print(f"    FIXED GLOBAL RANK r = {RANK} (full resolved rank; response sec 9)",
      flush=True)

Rq, Sq = [], []
for q in range(fx.nq):
    R, S, _ = frames_from_C(C_q[q], RANK)
    Rq.append(R)
    Sq.append(S)
print(f"    sqrt-cond per q: min {min(S[0]/S[-1] for S in Sq):.2e} "
      f"max {max(S[0]/S[-1] for S in Sq):.2e}")

# K-identity hard assert. MATH CORRECTION (v2, supersedes the spec's formula
# AND the first ^T attempt): with Phi = S R^H zeta and the conj-on-left
# contraction K[i,j] = sum_G conj(Phi_i) v Phi_j, the exact identity is
#     K == S R^T V R^* S          (V = make_Vq, Hermitian)
# because K = S R^T V conj(R) S elementwise; the spec's K == S R^H V R S
# holds only in the transposed-V convention (the response's implicit
# V' = zeta v zeta^H = V^T). Both are the same physics; asserting the
# self-consistent form. B = conj(a) K a^T with a = x R S^-1 G then reduces
# to conj(x) V x^T exactly (conj(R) R^T = I), verified by the LOO null test.
q = 1
PhiG_test = Sq[q][:, None] * (np.conj(Rq[q].T) @ fx.ZG[q])
K_test = fx.contractK(PhiG_test, q)
V_test = fx.make_Vq(fx.ZG[q], q)
K_ref = Sq[q][:, None] * ((Rq[q].T @ V_test @ np.conj(Rq[q]))) * Sq[q][None, :]
K_ref_spec = Sq[q][:, None] * (np.conj(Rq[q].T) @ V_test @ Rq[q]) * Sq[q][None, :]
errK = relF(K_test, K_ref)
print(f"  [check] K-identity K == S R^T V R^* S             {errK:.3e}"
      f"  (spec form S R^H V R S: {relF(K_test, K_ref_spec):.3e}; "
      f"its ^T: {relF(K_test, K_ref_spec.T):.3e})  "
      f"{'OK' if errK < 1e-12 else '** FAIL **'}", flush=True)

# --------------------------------------------------------------------------
# 4. links
# --------------------------------------------------------------------------
def gidx(i, j):
    return fx.k_lookup[(i % N1, j % N2, 0)]

t_cache, t_diag = {}, []

def band_t(ka, kb, psi=None, wfn_rot=None):
    """t_{ka <- kb} = polar(<u_ka|u_kb>_G). psi/wfn_rot: overrides for the
    gauge-randomization check (rotated centroid psi + band rotations)."""
    key = (ka, kb)
    if psi is None and key in t_cache:
        return t_cache[key]
    O = fx.band_overlap_G(ka, kb, nbmax=fx.nb)
    if wfn_rot is not None:
        O = np.conj(wfn_rot[ka].T) @ O @ wfn_rot[kb]
    t, s = polar(O)
    if psi is None:
        Oc = fx.band_overlap_centroid(ka, kb)
        tc, _ = polar(Oc)
        t_cache[key] = t
        t_diag.append((ka, kb, s.min(), s.max(),
                       np.linalg.norm(t - tc) / np.sqrt(2 * fx.nb)))
    return t

def link(qa_idx, qb_idx, psi=None, wfn_rot=None):
    """T_{qa <- qb} = polar(M), M = S^-1 R^H H R' S'^-1;
    H = X_qa^H B X_qb accumulated per k with the left-band-index rotation
    (pair-space B never materialized). Returns (T, principal_cosines).

    MATH CORRECTION #2 (found by the gauge-randomization gate): the response
    writes B = t^* (x) I, but with t = polar(<u_a|u_b>) the gauge-covariant
    pair contraction is
        H[mu,nu] = sum_k sum_{n n' m} u^a_n(mu) conj(u_m(mu))
                                      t_k[n,n'] conj(u^b_n'(nu)) u_m(nu),
    i.e. the connector enters UNconjugated between the unconjugated a-orbital
    and the conjugated b-orbital (per-element algebra: the V_b factors close
    to V_b V_b^H = I only in this orientation; with conj(t) they close to
    V_b V_b^T != I and covariance fails, which is exactly what the gate
    measured: ||T_rand - T|| ~ ||random unitary||)."""
    P = fx.psi if psi is None else psi
    H = np.zeros((fx.n_mu, fx.n_mu), dtype=np.complex128)
    for k in range(fx.nk):
        ka, _ = fx.kq_index(k, qa_idx)
        kb, _ = fx.kq_index(k, qb_idx)
        t = band_t(ka, kb, psi=psi, wfn_rot=wfn_rot)
        Xa = np.einsum("nsm,Msm->nMm", np.conj(P[ka]), P[k]).reshape(-1, fx.n_mu)
        Xb = np.einsum("nsm,Msm->nMm", np.conj(P[kb]), P[k])
        rot = np.einsum("nN,NMm->nMm", t, Xb).reshape(-1, fx.n_mu)
        H += np.conj(Xa.T) @ rot
    M = (np.conj(Rq[qa_idx].T) @ H @ Rq[qb_idx]) / Sq[qa_idx][:, None] / Sq[qb_idx][None, :]
    T, s = polar(M)
    return T, s

t_links = time.time()
T1 = {}   # T1[(i,j)] = T_{(i,j) <- (i+1,j)}
T2 = {}   # T2[(i,j)] = T_{(i,j) <- (i,j+1)}
cos_edges = {}
for i in range(N1):
    for j in range(N2):
        T1[(i, j)], s1 = link(gidx(i, j), gidx(i + 1, j))
        T2[(i, j)], s2 = link(gidx(i, j), gidx(i, j + 1))
        cos_edges[("b1", i, j)] = s1
        cos_edges[("b2", i, j)] = s2
print(f"\n[{name}] links built ({time.time()-t_links:.0f}s). "
      f"Principal-cosine spectra (diag B):")
allc = np.array([cos_edges[k] for k in sorted(cos_edges)])
print(f"    1 - s_min per edge: med {np.median(1-allc.min(1)):.3e} "
      f"max {np.max(1-allc.min(1)):.3e}")
print(f"    per-edge cosine percentiles (all {allc.shape[0]} edges pooled): "
      f"p0={allc.min():.4f} p10={np.percentile(allc,10):.4f} "
      f"p50={np.percentile(allc,50):.4f} p100={allc.max():.4f}")
nsmall = int(np.sum(allc < 0.5)) / allc.shape[0]
print(f"    mean # cosines < 0.5 per edge: {nsmall:.1f} of r={RANK} "
      f"(response 10B: many small => subspace underresolved)")
print(f"    band-transport diag: min svals(O) over edges "
      f"{min(d[2] for d in t_diag):.4f}; "
      f"G-space vs centroid-quadrature polar delta ||t-tc||/sqrt(2nb): "
      f"med {np.median([d[4] for d in t_diag]):.3e} "
      f"max {max(d[4] for d in t_diag):.3e}", flush=True)

# --------------------------------------------------------------------------
# 5. plaquette holonomies (diag D)
# --------------------------------------------------------------------------
hol = np.zeros((N1, N2))
for i in range(N1):
    for j in range(N2):
        # loop (i,j) <- (i+1,j) <- (i+1,j+1) <- (i,j+1) <- (i,j)
        W = T1[(i, j)] @ T2[((i + 1) % N1, j)] \
            @ np.conj(T1[(i, (j + 1) % N2)].T) @ np.conj(T2[(i, j)].T)
        hol[i, j] = np.linalg.norm(W - np.eye(RANK))
print(f"\n[{name}] plaquette holonomy ||W_box - I||_F (diag D): "
      f"med {np.median(hol):.3e} max {hol.max():.3e} "
      f"(vs sqrt(2r)={np.sqrt(2*RANK):.1f} for a random unitary)", flush=True)

# --------------------------------------------------------------------------
# 6. global gauge (row pass axis 1 at j=0, then column passes axis 2)
# --------------------------------------------------------------------------
G = {qi: np.eye(RANK, dtype=np.complex128) for qi in range(fx.nq)}

# --- row pass along axis 1 at q2=0: G(i+1,0) = T1(i,0)^H G(i,0)
for i in range(N1 - 1):
    G[gidx(i + 1, 0)] = np.conj(T1[(i, 0)].T) @ G[gidx(i, 0)]
G_trans = np.conj(T1[(N1 - 1, 0)].T) @ G[gidx(N1 - 1, 0)]   # arrive at (0,0)+G0
W_row = np.conj(G[gidx(0, 0)].T) @ G_trans                   # sewn^H transported


def frac_pow_u(W, s):
    """W^s for unitary W, principal branch: U diag(e^{i s theta}) U^{-1}."""
    ev, U = np.linalg.eig(W)
    th = np.angle(ev)
    return (U * np.exp(1j * s * th)[None, :]) @ np.linalg.inv(U), th


_, th_row = frac_pow_u(W_row, 0.0)
print(f"\n[{name}] Wilson row (axis1, j=0): ||W-I||={np.linalg.norm(W_row-np.eye(RANK)):.3e} "
      f"max|theta|={np.max(np.abs(th_row)):.3f} rad "
      f"(branch margin pi-|th|max = {np.pi-np.max(np.abs(th_row)):.3f})")
for i in range(N1):
    G[gidx(i, 0)] = G[gidx(i, 0)] @ frac_pow_u(W_row, -(i / N1))[0]

# --- column passes along axis 2 for each i: G(i,j+1) = T2(i,j)^H G(i,j)
W_cols, L_cols, th_cols = [], [], []
for i in range(N1):
    for j in range(N2 - 1):
        G[gidx(i, j + 1)] = np.conj(T2[(i, j)].T) @ G[gidx(i, j)]
    G_tr = np.conj(T2[(i, N2 - 1)].T) @ G[gidx(i, N2 - 1)]
    Wc = np.conj(G[gidx(i, 0)].T) @ G_tr
    Lc, thc = matlog_u(Wc)
    W_cols.append(Wc)
    L_cols.append(Lc)
    th_cols.append(thc)
    for j in range(N2):
        G[gidx(i, j)] = G[gidx(i, j)] @ frac_pow_u(Wc, -(j / N2))[0]

# branch continuity + det winding across the closed i-cycle
detph = np.array([np.angle(np.linalg.det(W)) for W in W_cols])
dw = np.diff(np.concatenate([detph, detph[:1]]))
dw = (dw + np.pi) % (2 * np.pi) - np.pi
winding = int(np.round(np.sum(dw) / (2 * np.pi)))
Ldiff = [np.linalg.norm(L_cols[i] - L_cols[(i - 1) % N1]) for i in range(N1)]
print(f"[{name}] Wilson columns: ||W-I|| = "
      + " ".join(f"{np.linalg.norm(W - np.eye(RANK)):.3e}" for W in W_cols))
print(f"    max|theta| per col = " + " ".join(f"{np.max(np.abs(t)):.3f}" for t in th_cols)
      + f"  (branch margin min = {min(np.pi-np.max(np.abs(t)) for t in th_cols):.3f})")
print(f"    log-branch continuity ||L_i - L_(i-1)||: "
      + " ".join(f"{d:.3e}" for d in Ldiff))
print(f"    det-W winding across q1 cycle = {winding} "
      f"({'NO topological obstruction' if winding == 0 else '** OBSTRUCTION — fall back to C1 local patches **'})",
      flush=True)

# residual uniformity of the gauged links (self-check of the construction)
res = []
for i in range(N1):
    for j in range(N2):
        r1 = np.conj(G[gidx(i, j)].T) @ T1[(i, j)] @ G[gidx(i + 1, j)]
        r2 = np.conj(G[gidx(i, j)].T) @ T2[(i, j)] @ G[gidx(i, j + 1)]
        res.append((np.linalg.norm(r1 - np.eye(RANK)),
                    np.linalg.norm(r2 - np.eye(RANK))))
res = np.array(res)
print(f"    gauged-link residual ||G^H T G' - I||: axis1 med {np.median(res[:,0]):.3e} "
      f"max {res[:,0].max():.3e}; axis2 med {np.median(res[:,1]):.3e} "
      f"max {res[:,1].max():.3e}", flush=True)

# --------------------------------------------------------------------------
# 7. gauge-randomization covariance check (self-check iii, link level)
# --------------------------------------------------------------------------
rng = np.random.default_rng(7)
rots = []
psi_r = np.empty_like(fx.psi)
for k in range(fx.nk):
    U = np.zeros((fx.nb, fx.nb), dtype=np.complex128)
    b = 0
    while b < fx.nb:
        e = b + 1
        while e < fx.nb and abs(fx.enk[k, e] - fx.enk[k, e - 1]) < TOLD:
            e += 1
        blk = e - b
        A = rng.normal(size=(blk, blk)) + 1j * rng.normal(size=(blk, blk))
        Qu, _ = np.linalg.qr(A)
        U[b:e, b:e] = Qu
        b = e
    rots.append(U)
    psi_r[k] = np.einsum("nsm,nN->Nsm", fx.psi[k], np.conj(U))
    # psi'_N = sum_n conj(U[n,N]) psi_n  <=>  |n'> = sum_n |n> U*_{n,n'} — a
    # unitary relabeling of the band basis (any convention works if consistent:
    # the WFN overlap is rotated with the SAME matrices below).
wfn_rot = {k: np.conj(rots[k]) for k in range(fx.nk)}
dTmax = 0.0
for (i, j) in [(0, 0), (1, 0), (0, 1), (2, 2)]:
    Tr, _ = link(gidx(i, j), gidx(i + 1, j), psi=psi_r, wfn_rot=wfn_rot)
    dTmax = max(dTmax, np.linalg.norm(Tr - T1[(i, j)]))
print(f"\n[{name}] gauge-randomization: max ||T_rand - T|| over 4 probe edges = "
      f"{dTmax:.3e}  {'OK' if dTmax < 1e-10 else '** FAIL — transport not covariant **'}",
      flush=True)

# --------------------------------------------------------------------------
# 8. TRS checks (self-check iv)
# --------------------------------------------------------------------------
# V_{-q} == conj(V_q) on the stored zeta (fixture-level TRS) + attribution:
# C-level TRS (my rebuild) and disk-tile TRS separate the fixture's own
# zeta-solve TRS breaking (null-space content enters V quadratically) from
# any prep bug.
trs_err, trs_C, trs_disk = [], [], []
for q in range(fx.nq):
    mq = fx.k_lookup[tuple((-fx.k_int[q]) % kg)]
    Vp = fx.make_Vq(fx.ZG[q], q)
    Vm = fx.make_Vq(fx.ZG[mq], mq)
    trs_err.append(relF(Vm, np.conj(Vp)))
    trs_C.append(relF(C_q[mq], np.conj(C_q[q])))
    trs_disk.append(relF(fx.Vqmunu[mq], np.conj(fx.Vqmunu[q])))
print(f"[{name}] TRS V(-q)==conj(V(q)) [stored zeta]: med {np.median(trs_err):.3e} "
      f"max {np.max(trs_err):.3e}")
print(f"    attribution: C-level TRS med {np.median(trs_C):.3e} max {np.max(trs_C):.3e}; "
      f"DISK V_qmunu TRS med {np.median(trs_disk):.3e} max {np.max(trs_disk):.3e}")

# --------------------------------------------------------------------------
# 9. FOUR TAILS
# --------------------------------------------------------------------------
# lattice R set + shells (wrapped, Bohr metric)
Rall = np.array([[i, j, 0] for i in range(N1) for j in range(N2)])
Rw = ((Rall + kg // 2) % kg) - (kg // 2)
Rdist = np.sqrt(np.einsum("ri,ij,rj->r", Rw, fx.adot, Rw))
shells = {}
for idx, d in enumerate(Rdist):
    shells.setdefault(round(d, 2), []).append(idx)
shell_keys = sorted(shells)
print(f"\n[{name}] R shells (Bohr): "
      + " ".join(f"{k}({len(shells[k])})" for k in shell_keys), flush=True)

# reference: C_R shells from the order-robust lattice image (already built)
C_R = fx.C_R_full.reshape(len(Rw), -1)   # physical_contract Rw order == Rw here
cref = np.array([np.linalg.norm(C_R[i]) for i in range(len(Rw))])

def shell_table(vals):
    """vals: per-R-index norm; -> per-shell max, normalised to the R=0 shell."""
    out = []
    v0 = max(max(vals[i] for i in shells[shell_keys[0]]), 1e-300)
    for key in shell_keys:
        out.append(max(vals[i] for i in shells[key]) / v0)
    return out

# build Phi_q(r) (lab) for all q + gauged version; also raw zeta_r norms
t_ph = time.time()
Phi = np.empty((fx.nq, RANK, fx.n_rtot), dtype=np.complex128)
EqR = np.exp(-2j * np.pi * (fx.qfr @ Rw.T))                  # (nq, nR)
zeta_hold = np.empty((fx.nq, fx.n_mu, fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    zr = fx.recon(q)
    zeta_hold[q] = zr
    Phi[q] = (Sq[q][:, None] * np.conj(Rq[q].T)) @ zr
print(f"[{name}] Phi_q built ({time.time()-t_ph:.0f}s; "
      f"{Phi.nbytes/1e9:.1f} GB).", flush=True)

def dft_tail_norms(stack):
    """stack: (nq, ...) -> per-R ||(1/nq) sum_q e^-2pi i q.R stack_q||_F.
    zgemv form (no (nq, M) temporary)."""
    flat = stack.reshape(fx.nq, -1)
    out = np.zeros(len(Rw))
    for iR in range(len(Rw)):
        acc = (EqR[:, iR] @ flat) / fx.nq
        out[iR] = np.linalg.norm(acc)
    return out

tails = {}
tails["C_R (reference)"] = shell_table(cref)
tails["raw zeta_R"] = shell_table(dft_tail_norms(zeta_hold))
tails["Phi_R (no transport)"] = shell_table(dft_tail_norms(Phi))

PhiT = np.empty_like(Phi)
for q in range(fx.nq):
    PhiT[q] = np.conj(G[q].T) @ Phi[q]
tails["Phi~_R (transported)"] = shell_table(dft_tail_norms(PhiT))

# cell-periodic variants (strip e^{i q.r}) — secondary diagnostic
bloch = np.exp(-2j * np.pi * (fx.rfrac @ fx.qfr.T)).T        # (nq, n_rtot)
tails["zeta_R cell-periodic"] = shell_table(dft_tail_norms(zeta_hold * bloch[:, None, :]))
tails["Phi~_R cell-periodic"] = shell_table(dft_tail_norms(PhiT * bloch[:, None, :]))
del zeta_hold

# Parseval check for the transported tail (power must be conserved by G)
pw_phi = np.sum(np.abs(Phi) ** 2)
pw_phiT = np.sum(np.abs(PhiT) ** 2)
print(f"    Parseval ||Phi||^2 vs ||Phi~||^2: rel diff "
      f"{abs(pw_phi-pw_phiT)/pw_phi:.2e}")

# transported SR kernel tails (r x r), alpha ladder + taper variant + total
LR_REACH = 10.0    # Bohr, measured C_R reach (task spec)
alphas = [0.5 * 2 * np.pi / LR_REACH, 1.0 * 2 * np.pi / LR_REACH,
          2.0 * 2 * np.pi / LR_REACH, 4.0 * 2 * np.pi / LR_REACH]
PhiG_T = [fx.to_sphere(PhiT[q], q) for q in range(fx.nq)]
VcSR_spectra_pm = []
for al in alphas:
    Vc = np.stack([fx.contractK(PhiG_T[q], q, kind="slab_sr", alpha=al)
                   for q in range(fx.nq)])
    tails[f"VcSR~_R (a={al:.2f})"] = shell_table(dft_tail_norms(Vc))
    if al == alphas[1]:
        Vc_t = np.stack([fx.contractK(PhiG_T[q], q, kind="slab_sr", alpha=al,
                                      taper=(20.0, 30.0)) for q in range(fx.nq)])
        tails[f"VcSR~_R (a={al:.2f}, taper 20-30Ry)"] = shell_table(dft_tail_norms(Vc_t))
        # TRS on the frame-basis SR kernel: eigenvalue match at +-q
        for q in range(1, fx.nq):
            mq = fx.k_lookup[tuple((-fx.k_int[q]) % kg)]
            if mq <= q:
                continue
            ep = np.linalg.eigvalsh(0.5 * (Vc[q] + np.conj(Vc[q].T)))
            em = np.linalg.eigvalsh(0.5 * (Vc[mq] + np.conj(Vc[mq].T)))
            VcSR_spectra_pm.append(np.max(np.abs(ep - em)) / max(np.max(np.abs(ep)), 1e-300))
Vc_tot = np.stack([fx.contractK(PhiG_T[q], q) for q in range(fx.nq)])
tails["Vc~_R (total, transported)"] = shell_table(dft_tail_norms(Vc_tot))
Vc_un = np.stack([fx.contractK(fx.to_sphere(Phi[q], q), q) for q in range(fx.nq)])
tails["Vc_R (total, no transport)"] = shell_table(dft_tail_norms(Vc_un))
if VcSR_spectra_pm:
    print(f"    TRS VcSR~ +-q eigenvalue match: max rel {max(VcSR_spectra_pm):.3e}")

# --------------------------------------------------------------------------
# table
# --------------------------------------------------------------------------
print(f"\n[{name}] ================ FOUR-TAILS SHELL TABLE (per-shell max "
      f"||.||_F / shell-0) ================")
hdr = "  ".join(f"|R|={k:>6.2f}" for k in shell_keys)
print(f"    {'':<38s} {hdr}")
for lbl, row in tails.items():
    print(f"    {lbl:<38s} " + "  ".join(f"{v:>10.3e}" for v in row))

np.savez(f"/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         f"A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         f"proto1_C2_fourtails_{name}.npz",
         shell_keys=np.array(shell_keys),
         **{f"tail_{i}": np.array(v) for i, v in enumerate(tails.values())},
         tail_labels=np.array(list(tails.keys())),
         holonomy=hol, winding=winding, rank=RANK,
         cos_edges=allc, spec=spec)

# verdict line
pt = tails["Phi~_R (transported)"]
cr = tails["C_R (reference)"]
zt_ = tails["raw zeta_R"]
print(f"\n[{name}] VERDICT INPUT: shell-1 ratios — C_R {cr[1]:.2e} | "
      f"raw zeta {zt_[1]:.2e} | Phi~ transported {pt[1]:.2e}")
if pt[1] > 0.3:
    print(f"[{name}] Phi~_R FLAT at shell 1 => response's central claim FAILS "
          f"its own sec-10C criterion at this density (C2 terminates).")
else:
    print(f"[{name}] Phi~_R decays => proceed to proto1_C2_loo.py.")
print(f"[{name}] done in {time.time()-t00:.0f}s.")
