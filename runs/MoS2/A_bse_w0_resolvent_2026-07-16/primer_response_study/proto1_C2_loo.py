"""proto1_C2_loo — C2 construction, stage 2: leave-one-out interpolation under
the PRIMARY physical metrics (owner pushback: gap-window exchange B-block +
TDA exciton shift), plus the rank ladder and the re-scored #3.5 ladder.

Per held-out q0 (all on-grid q):
  TRUTH      B_true = gap-window rows (centroid, through stored full-rank fit)
             contracted with exact slab v(q0+G), G=0 excluded.
  CONTEXT    B_exact from non-ISDF pair rows (WFN u's, FFT to sphere): the
             ISDF fit floor under the physical metric (what "few percent"
             must be compared against).
  NULL/RANK  B_r from the C2 chain (frames+gauge, NO interpolation) at rank
             r in RANKS: r=full must be machine-zero (null test); lower r =
             the rank-cut-under-physical-metric curve the owner asked for.
  C2 INTERP  transported-frame Fourier interpolation of Phi~ (Strategy A):
             Phi~_pred(q0) = truncated-R fit from the 8 training q's; target
             gauge G(q0) by construction AND by neighbor alignment (spread
             reported); B_pred, SR/LR/total splits, tile relF (secondary),
             TDA exciton shift for lowest 4 states (meV).
  LADDER     #3.5 ingredient-interp + C^-1 solve ladder (raw / tikhonov 1e-6
             / rankcut 1e-4 / 1e-2) re-scored under the SAME B-block metric.

Conventions: proto1_prep docstring. K transforms as K' = G^T K G^* under
Phi~ = G^H Phi with the conj-on-left contraction; a0 = x R S^-1 G restores
the physical block (derivation in the report; null test enforces it).

Run: JID=<jid> ./proto1_run.sh python3 -u proto1_C2_loo.py MoS2_3x3
"""
import sys, time
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                   "A_bse_w0_resolvent_2026-07-16/primer_response_study")
from proto1_prep import (Fixture, relF, polar, frames_from_C, matlog_u,
                         truncR_weights)

t00 = time.time()
name = sys.argv[1] if len(sys.argv) > 1 else "MoS2_3x3"
fx = Fixture(name)
kg = fx.kgrid
N1, N2 = int(kg[0]), int(kg[1])
NVW = NCW = 3
RANKS = [640, 480, 320, 160, 80]
NR_INTERP = [4, 7, 9]
ALPHA = 2.0 * np.pi / 10.0          # 1.0 * 2pi/L_R, L_R = 10 Bohr
RY2MEV = 13605.693

print(f"[{name}] LOO stage: nq={fx.nq} nb={fx.nb} n_mu={fx.n_mu} nv={fx.nv}")
C_q = fx.build_Cq()

# frames (full resolved rank)
spec0 = np.linalg.eigvalsh(0.5 * (C_q[0] + C_q[0].conj().T))
RANK = fx.n_mu if spec0[0] > 0 else int(np.sum(spec0 > 0))
RANK = min(RANK, min(int(np.sum(np.linalg.eigvalsh(
    0.5 * (C_q[q] + C_q[q].conj().T)) > 0)) for q in range(fx.nq)))
Rq, Sq, lam_full = [], [], []
for q in range(fx.nq):
    R, S, lam = frames_from_C(C_q[q], RANK)
    Rq.append(R)
    Sq.append(S)
    lam_full.append(lam)
print(f"[{name}] fixed global rank r={RANK}")

# --------------------------------------------------------------------------
# links + global gauge (same construction as proto1_C2_fourtails; duplicated
# for a self-contained stage-2 script — prototype-only duplication)
# --------------------------------------------------------------------------
def gidx(i, j):
    return fx.k_lookup[(i % N1, j % N2, 0)]

t_cache = {}
def band_t(ka, kb):
    key = (ka, kb)
    if key not in t_cache:
        t_cache[key] = polar(fx.band_overlap_G(ka, kb, nbmax=fx.nb))[0]
    return t_cache[key]

def link(qa_idx, qb_idx):
    # gauge-covariant orientation: connector t UNconjugated (see fourtails
    # MATH CORRECTION #2; the response's t^* form is not covariant here).
    H = np.zeros((fx.n_mu, fx.n_mu), dtype=np.complex128)
    for k in range(fx.nk):
        ka, _ = fx.kq_index(k, qa_idx)
        kb, _ = fx.kq_index(k, qb_idx)
        t = band_t(ka, kb)
        Xa = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[ka]), fx.psi[k]).reshape(-1, fx.n_mu)
        Xb = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[kb]), fx.psi[k])
        rot = np.einsum("nN,NMm->nMm", t, Xb).reshape(-1, fx.n_mu)
        H += np.conj(Xa.T) @ rot
    M = (np.conj(Rq[qa_idx].T) @ H @ Rq[qb_idx]) / Sq[qa_idx][:, None] / Sq[qb_idx][None, :]
    return polar(M)[0]

def frac_pow_u(W, s):
    ev, U = np.linalg.eig(W)
    return (U * np.exp(1j * s * np.angle(ev))[None, :]) @ np.linalg.inv(U)

t_l = time.time()
T1 = {(i, j): link(gidx(i, j), gidx(i + 1, j)) for i in range(N1) for j in range(N2)}
T2 = {(i, j): link(gidx(i, j), gidx(i, j + 1)) for i in range(N1) for j in range(N2)}
G = {qi: np.eye(RANK, dtype=np.complex128) for qi in range(fx.nq)}
for i in range(N1 - 1):
    G[gidx(i + 1, 0)] = np.conj(T1[(i, 0)].T) @ G[gidx(i, 0)]
W_row = np.conj(G[gidx(0, 0)].T) @ (np.conj(T1[(N1 - 1, 0)].T) @ G[gidx(N1 - 1, 0)])
for i in range(N1):
    G[gidx(i, 0)] = G[gidx(i, 0)] @ frac_pow_u(W_row, -(i / N1))
for i in range(N1):
    for j in range(N2 - 1):
        G[gidx(i, j + 1)] = np.conj(T2[(i, j)].T) @ G[gidx(i, j)]
    Wc = np.conj(G[gidx(i, 0)].T) @ (np.conj(T2[(i, N2 - 1)].T) @ G[gidx(i, N2 - 1)])
    for j in range(N2):
        G[gidx(i, j)] = G[gidx(i, j)] @ frac_pow_u(Wc, -(j / N2))
print(f"[{name}] links+gauge built ({time.time()-t_l:.0f}s)")

# transported sections Phi~_q(r)  (r x n_rtot each)
Phi = np.empty((fx.nq, RANK, fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    Phi[q] = np.conj(G[q].T) @ ((Sq[q][:, None] * np.conj(Rq[q].T)) @ fx.recon(q))

# R-stencil machinery (vq_loo conventions)
Rall = np.array([[i, j, 0] for i in range(N1) for j in range(N2)])
Rw = ((Rall + kg // 2) % kg) - (kg // 2)
Rdist = np.sqrt(np.einsum("ri,ij,rj->r", Rw, fx.adot, Rw))
Rsort = Rw[np.argsort(Rdist)]

# --------------------------------------------------------------------------
# physical-metric machinery
# --------------------------------------------------------------------------
def gap_rows_centroid(q0):
    return fx.gap_window_pairs(q0, NVW, NCW)          # (81, n_mu)

def B_from_MG(Mg, q0, **kw):
    v, n = fx.vq(q0, **kw)
    A = Mg[:, :n] * np.sqrt(v[:n])[None, :]
    return np.conj(A) @ A.T

def B_truth_and_exact(q0):
    """B through the stored full-rank fit; and the non-ISDF exact block."""
    x = gap_rows_centroid(q0)
    Mg_fit = x @ fx.ZG[q0]                             # (81, ngkmax) ISDF rep
    B_fit = B_from_MG(Mg_fit, q0)
    # exact rows: u on grid -> spin-traced pair rows -> sphere
    rows = np.empty((fx.nk, NCW, NVW, fx.n_rtot), dtype=np.complex128)
    for k in range(fx.nk):
        kq, _ = fx.kq_index(k, q0)
        ukq = fx.u_grid(kq, nbmax=fx.nb)
        uk = fx.u_grid(k, nbmax=fx.nb)
        cs = slice(fx.nv, fx.nv + NCW)
        vs = slice(fx.nv - NVW, fx.nv)
        rows[k] = np.einsum("csr,vsr->cvr", np.conj(ukq[cs]), uk[vs])
    Mg_ex = fx.to_sphere(rows.reshape(-1, fx.n_rtot), q0)
    # WFN u's differ from psi_full_y by the constant wfn scale; normalize out
    sc = np.linalg.norm(x) / max(np.linalg.norm(rows.reshape(-1, fx.n_rtot)[:, fx.rmu_flat]), 1e-300)
    B_ex = B_from_MG(Mg_ex * sc, q0)
    return B_fit, B_ex, x

def top_decile_rel(Bp, Bt):
    at = np.abs(Bt).ravel()
    thr = np.quantile(at, 0.9)
    m = at >= thr
    return np.median(np.abs((Bp - Bt).ravel()[m]) / at[m])

# TDA exciton at Q=q0: H = D - W_dir + K_x  (81-dim); swap only the exchange V.
def exciton_shift(q0, B_pred, B_true):
    nvw, ncw = NVW, NCW
    npair = fx.nk * ncw * nvw
    cs = list(range(fx.nv, fx.nv + ncw))
    vs = list(range(fx.nv - nvw, fx.nv))
    D = np.zeros(npair)
    idx = 0
    for k in range(fx.nk):
        kq, _ = fx.kq_index(k, q0)
        for c in cs:
            for v in vs:
                D[idx] = fx.enk[kq, c] - fx.enk[k, v]
                idx += 1
    H_dir = np.zeros((npair, npair), dtype=np.complex128)
    for k in range(fx.nk):
        kq, _ = fx.kq_index(k, q0)
        for kp in range(fx.nk):
            kpq, _ = fx.kq_index(kp, q0)
            qkk = fx.k_lookup[tuple((fx.k_int[k] - fx.k_int[kp]) % kg)]
            W = fx.W0[qkk]
            Tc = np.einsum("nsm,Nsm->nNm", np.conj(fx.psi[kq][cs]), fx.psi[kpq][cs])
            Tv = np.einsum("nsm,Nsm->nNm", fx.psi[k][vs], np.conj(fx.psi[kp][vs]))
            blk = np.einsum("cCm,mn,vVn->cvCV", Tc, W, Tv, optimize=True)
            H_dir[np.ix_(range(k * ncw * nvw, (k + 1) * ncw * nvw),
                         range(kp * ncw * nvw, (kp + 1) * ncw * nvw))] = \
                blk.reshape(ncw * nvw, ncw * nvw)
    H_dir /= fx.nk
    ev_out = []
    for B in (B_true, B_pred):
        H = np.diag(D).astype(np.complex128) - H_dir + B / fx.nk
        H = 0.5 * (H + np.conj(H.T))
        ev = np.linalg.eigvalsh(H)
        ev_out.append(ev[:4])
    return (ev_out[1] - ev_out[0]) * RY2MEV, ev_out[0] * RY2MEV

# --------------------------------------------------------------------------
# ladder machinery (#3.5 re-score): interp C and Z, solve, B-block
# --------------------------------------------------------------------------
t_z = time.time()
Z_r = np.empty((fx.nq, fx.n_mu, fx.n_rtot), dtype=np.complex128)
zeta_r_all = np.empty((fx.nq, fx.n_mu, fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    zr = fx.recon(q)
    zeta_r_all[q] = zr
    Z_r[q] = C_q[q] @ zr
Cq_flat = C_q.reshape(fx.nq, -1)
Zr_flat = Z_r.reshape(fx.nq, -1)
print(f"[{name}] Z_r built ({time.time()-t_z:.0f}s)")

def solve_zeta(Cmat, Zmat, mode, lam):
    Ch = 0.5 * (Cmat + Cmat.conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    if mode == "raw":
        sinv = 1.0 / s
    elif mode == "tikhonov":
        sinv = 1.0 / (s + lam * (s.sum() / len(s)))
    else:
        sinv = np.where(s > lam * s[0], 1.0 / np.where(s > lam * s[0], s, 1), 0.0)
    return (Vh.conj().T * sinv) @ (U.conj().T @ Zmat)

# --------------------------------------------------------------------------
# main LOO loop
# --------------------------------------------------------------------------
res = {}   # res[label][q0] = dict of metrics
def put(lbl, q0, **kw):
    res.setdefault(lbl, {})[q0] = kw

fit_floor = []
for q0 in range(fx.nq):
    tq = time.time()
    B_true, B_exact, x = B_truth_and_exact(q0)
    nB = np.linalg.norm(B_true)
    fit_floor.append(relF(B_true, B_exact))
    V_true = fx.make_Vq(fx.ZG[q0], q0)

    # ---- rank ladder (no interpolation; C2 chain null + rank-cut curve) ----
    # r == RANK goes through the FULL transported chain (gauge G included; it
    # must cancel => machine-zero null test). r < RANK uses untransported
    # frame slices (the gauge was built at full rank and its leading block is
    # not a valid lower-rank gauge; the gauge cancels in B anyway).
    PhiG_q0 = fx.to_sphere(Phi[q0], q0)               # transported, full RANK
    for r in RANKS:
        if r > RANK:
            continue
        if r < RANK:
            PhiG_r = Sq[q0][:r, None] * (np.conj(Rq[q0][:, :r].T) @ fx.ZG[q0])
            a0 = x @ (Rq[q0][:, :r] / Sq[q0][None, :r])
        else:
            PhiG_r = PhiG_q0
            a0 = (x @ (Rq[q0] / Sq[q0][None, :])) @ G[q0]
        K_r = fx.contractK(PhiG_r, q0)
        B_r = np.conj(a0) @ K_r @ a0.T
        put(f"rank_{r}", q0, B=relF(B_r, B_true), Bdec=top_decile_rel(B_r, B_true))

    # ---- C2 transported-Phi interpolation ----------------------------------
    train = [q for q in range(fx.nq) if q != q0]
    # align-protocol target gauge (nR-independent): nearest training neighbor
    # on the torus; G(q0) = T_{q0<-qn} G(qn) (path-consistent form)
    dmin, nbr = 1e9, train[0]
    for t in train:
        d = fx.qfr[t] - fx.qfr[q0]
        d = d - np.round(d)
        dd = float(np.sum((fx.bvec.T @ d) ** 2))
        if dd < dmin:
            dmin, nbr = dd, t
    G_align = link(q0, nbr) @ G[nbr]
    print(f"    q0={q0}: gauge-protocol spread ||G_align - G_constr||_F = "
          f"{np.linalg.norm(G_align - G[q0]):.3e} (vs sqrt(2r)="
          f"{np.sqrt(2*RANK):.1f} for unrelated unitaries)", flush=True)
    for nR in NR_INTERP:
        w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rsort[:nR])
        Phi_pred = np.tensordot(w, Phi[train], axes=(0, 0))
        for proto in ("construction", "align"):
            Gq0 = G[q0] if proto == "construction" else G_align
            a0 = (x @ (Rq[q0] / Sq[q0][None, :])) @ Gq0
            # Phi_pred is already in the transported/global frame; the target
            # gauge enters ONLY through a0 (K built from Phi_pred as-is).
            PhiG_pred = fx.to_sphere(Phi_pred, q0)
            K_pred = fx.contractK(PhiG_pred, q0)
            B_pred = np.conj(a0) @ K_pred @ a0.T
            lbl = f"C2_nR{nR}_{proto}"
            met = {"B": relF(B_pred, B_true), "Bdec": top_decile_rel(B_pred, B_true)}
            if proto == "construction" and nR == 7:
                # SR/LR split (both exact target v; C2 Strategy A)
                for kind, tag in (("slab_sr", "BSR"), ("slab_lr", "BLR")):
                    Kk = fx.contractK(PhiG_pred, q0, kind=kind, alpha=ALPHA)
                    Bk = np.conj(a0) @ Kk @ a0.T
                    Bt = B_from_MG(x @ fx.ZG[q0], q0, kind=kind, alpha=ALPHA)
                    met[tag] = relF(Bk, Bt)
                # tile (secondary): V_pred back in centroid basis.
                # K_pred lives in the gauged frame (Phi~ = G^H Phi with the
                # conj-on-left contract => K~ = G^T K_frame G^*), so
                # K_frame = conj(G) K~ G^T and V = conj(RS^-1) K_frame RS^-T
                # (exact at full rank: conj(R) R^T = I).
                RS = Rq[q0] * (1.0 / Sq[q0])[None, :]
                K_frame = np.conj(Gq0) @ K_pred @ Gq0.T
                Vp = np.conj(RS) @ K_frame @ RS.T
                met["tile"] = relF(Vp, V_true)
                # exciton shift
                dmeV, e0 = exciton_shift(q0, B_pred, B_true)
                met["exc_meV"] = float(np.max(np.abs(dmeV)))
                met["e0_meV"] = float(e0[0])
            put(lbl, q0, **met)

    # ---- #3.5 ladder under the physical metric -----------------------------
    for mode, lam in [("raw", 0.0), ("tikhonov", 1e-6), ("rankcut", 1e-4),
                      ("rankcut", 1e-2)]:
        w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rsort[:7])
        C0 = (w @ Cq_flat[train]).reshape(fx.n_mu, fx.n_mu)
        Z0 = (w @ Zr_flat[train]).reshape(fx.n_mu, fx.n_rtot)
        zt0 = fx.to_sphere(solve_zeta(C0, Z0, mode, lam), q0)
        Mg = x @ zt0
        B_l = B_from_MG(Mg, q0)
        V_l = fx.make_Vq(zt0, q0)
        lbl = f"ladder_{mode}{'' if mode=='raw' else f'_{lam:.0e}'}"
        met = {"B": relF(B_l, B_true), "Bdec": top_decile_rel(B_l, B_true),
               "tile": relF(V_l, V_true)}
        if mode == "rankcut" and lam == 1e-2:
            dmeV, _ = exciton_shift(q0, B_l, B_true)
            met["exc_meV"] = float(np.max(np.abs(dmeV)))
        put(lbl, q0, **met)

    # ---- zeta-direct interpolation (skip the solve) — ladder context -------
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rsort[:7])
    zdir = np.tensordot(w, zeta_r_all[train], axes=(0, 0))
    zt_d = fx.to_sphere(zdir, q0)
    B_d = B_from_MG(x @ zt_d, q0)
    put("zeta_direct_nR7", q0, B=relF(B_d, B_true), Bdec=top_decile_rel(B_d, B_true),
        tile=relF(fx.make_Vq(zt_d, q0), V_true))

    print(f"  q0={q0} done ({time.time()-tq:.0f}s)", flush=True)

# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
print(f"\n[{name}] ISDF fit floor under physical metric relF(B_fit, B_exact): "
      f"med {np.median(fit_floor):.3e} max {np.max(fit_floor):.3e}")
print(f"\n[{name}] ============ LOO results (median / max over q0) ============")
print(f"    {'label':<28s} {'B med':>10s} {'B max':>10s} {'Bdec med':>10s} "
      f"{'tile med':>10s} {'exc meV med':>12s}")
for lbl in sorted(res):
    rows = res[lbl]
    Bm = [rows[q]["B"] for q in rows]
    Bd = [rows[q]["Bdec"] for q in rows]
    tm = [rows[q].get("tile") for q in rows if "tile" in rows[q]]
    em = [rows[q].get("exc_meV") for q in rows if "exc_meV" in rows[q]]
    print(f"    {lbl:<28s} {np.median(Bm):>10.3e} {np.max(Bm):>10.3e} "
          f"{np.median(Bd):>10.3e} "
          f"{(f'{np.median(tm):>10.3e}' if tm else '        --')} "
          f"{(f'{np.median(em):>12.3e}' if em else '          --')}")

np.savez(f"/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         f"A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         f"proto1_C2_loo_{name}.npz",
         labels=np.array(sorted(res)),
         fit_floor=np.array(fit_floor),
         **{f"B_{lbl}": np.array([res[lbl][q]["B"] for q in sorted(res[lbl])])
            for lbl in res})
print(f"[{name}] LOO stage done in {time.time()-t00:.0f}s.")
