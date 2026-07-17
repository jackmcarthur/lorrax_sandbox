"""proto0 script A — MoS2 3x3: gates + baseline_rebase + C1 LOO ladder.

C1 = target-frame transported V^SR interpolation (response Method A + Strategy
B, exact-target dressing).  FLAVOR NOTE (correction C-5): with the resolved
C = X^H X over x-rows (x = conj(psi_c) psi_v), the response's bounded objects
are Phi_R = S R^H zeta~ = L^H A~ (rows of L = X R S^-1 have norm <= 1) and the
natural kernel  B_nat = l Vc_R l^H  with l = M R S^-1;  the CODE-side physical
block is its conjugate, B = conj(B_nat).  The transport T (built from
H = X^H B X') aligns THESE frames.  The conjugate flavor
(Phi_C = S R^H conj(zeta~), a = conj(M) R S^-1) satisfies the same algebraic
identities on-grid but pairs the transport with the wrong bundle — kept as the
wrong-flavor ablation.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

import prep
from prep import (load_fixture, load_wfn, grid_geometry, recon, to_sphere,
                  relF, rebuild_C_falloff, gram_C_from_rows, cross_gram_H,
                  vdim_at, v_split_at, sphere_K, alpha_ladder, make_Vq_generic,
                  eigh_frame, phi_from_zeta, vc_tile, build_regauged_fields,
                  band_overlaps_fields, band_overlaps_centroid, polar_unitary,
                  edge_links_T, gap_window_rows, B_from_tile, B_metrics,
                  assemble_H, lowest_eigs, w_fourier, w_nn_avg, r_sorted,
                  wfn_u_on_grid, RY_TO_MEV)

np.set_printoptions(precision=3, suppress=False)
OUT = {}

fx = load_fixture("mos2_3x3")
grid_geometry(fx)
wfn = load_wfn(fx, prep.FIX["mos2_3x3"]["wfn"])
print(f"[3x3] nk={fx.nk} nb={fx.nb} ns={fx.ns} nmu={fx.nmu} nocc={fx.nocc} "
      f"FFT={fx.fg} ngkmax={fx.ngkmax}")

# =====================================================================
# GATE 0 — plumbing
# =====================================================================
zt0 = to_sphere(fx, recon(fx, 0), 0)
n0 = int(fx.ngk[0])
rt = relF(zt0[:, :n0], fx.ZG[0][:, :n0])
print(f"[gate0] recon/forward sphere round-trip: {rt:.2e}")
assert rt < 1e-14

g0chk = max(relF(fx.g0_mu[q], fx.ZG[q, :, 0]) for q in range(fx.nq))
print(f"[gate0] g0_mu == ZG[:,:,G=0 slot]: {g0chk:.2e}")

vq_match = []
Vfull_true = []
for q in range(fx.nq):
    Kf, n = sphere_K(fx, q)
    v, _ = vdim_at(fx, Kf)
    Vt = make_Vq_generic(fx.ZG[q], v, n)
    Vfull_true.append(Vt)
    vq_match.append(relF(Vt, fx.Vdisk[q]))
Vfull_true = np.array(Vfull_true)
print(f"[gate0] production-v rebuild vs disk V_qmunu, relF per q: "
      f"med={np.median(vq_match):.2e} max={np.max(vq_match):.2e}")
assert np.max(vq_match) < 1e-12
OUT["vq_match"] = np.array(vq_match)

el = fx.el[0]
enk_dev = np.max(np.abs(el[:, :fx.nb] - fx.enk))
gap_top = np.min(el[:, fx.nb] - el[:, fx.nb - 1])
print(f"[gate0] el vs enk max dev: {enk_dev:.2e} Ry; window-top gap "
      f"min_k (e_81 - e_80) = {gap_top:.3e} Ry "
      f"({'CLEAN' if gap_top > 1e-6 else 'SPLIT MULTIPLET — covariance poisoned'})")
OUT["hygiene_gap_top"] = gap_top

assert np.max(np.abs(wfn["rk"] - fx.qfr)) < 1e-12, "WFN rk order mismatch"
k_chk = 0
u = wfn_u_on_grid(fx, wfn, k_chk, list(range(fx.nb)))
dev = relF(u[:, :, fx.rmu_flat], fx.psi[k_chk])
print(f"[gate0] WFN u(r_mu) vs psi_full_y (k=0): relF={dev:.2e}")
assert dev < 1e-10, "WFN normalization/gauge mismatch with fixture psi"

# =====================================================================
# GATE 1 — conventions (rows, flavors, boundedness)
# =====================================================================
C_fall = rebuild_C_falloff(fx)
qt = 1
Cx = gram_C_from_rows(fx, qt, conj_left=True)
Cy = gram_C_from_rows(fx, qt, conj_left=False)
ex, ey = relF(Cx, C_fall[qt]), relF(Cy, C_fall[qt])
print(f"[gate1] X^H X vs falloff C (q={qt}): conj_left=True {ex:.2e} | "
      f"conj_left=False {ey:.2e}")
CONJ_LEFT = bool(ex < ey)
assert min(ex, ey) < 1e-12
print(f"[gate1] resolved row convention: conj_left={CONJ_LEFT}")

R_by_q, S_by_q, lam_by_q = {}, {}, {}
for q in range(fx.nq):
    R, lam = eigh_frame(C_fall[q])
    R_by_q[q] = R
    S_by_q[q] = np.sqrt(np.maximum(lam, 1e-300))
    lam_by_q[q] = lam
conds = [lam_by_q[q][0] / lam_by_q[q][-1] for q in range(fx.nq)]
print(f"[gate1] cond(C_q): med={np.median(conds):.2e} max={np.max(conds):.2e}")
OUT["condC"] = np.array(conds)

# Flavor identities at q=qt (both must close; response flavor is PRIMARY):
#   R-flavor: Phi_R = S R^H zt        ; Vc_R == S R^H conj(V_make) R S
#   C-flavor: Phi_C = S R^H conj(zt)  ; Vc_C == S R^H V_make R S
q = qt
R, S = R_by_q[q], S_by_q[q]
Kf, n = sphere_K(fx, q)
v_full, _ = vdim_at(fx, Kf)
Phi_R, _ = phi_from_zeta(fx, q, R, S, phi_conj=False)
Phi_C, _ = phi_from_zeta(fx, q, R, S, phi_conj=True)
refC = (S[:, None] * (R.conj().T @ Vfull_true[q] @ R)) * S[None, :]
refR = (S[:, None] * (R.conj().T @ np.conj(Vfull_true[q]) @ R)) * S[None, :]
errR = relF(vc_tile(Phi_R, v_full), refR)
errC = relF(vc_tile(Phi_C, v_full), refC)
print(f"[gate1] Vc identities | R-flavor {errR:.2e} | C-flavor {errC:.2e}")
assert errR < 1e-12 and errC < 1e-12

# dressing rows: PRIMARY l = M R S^-1 (rows of L, norm <= 1);
# B = conj(l Vc_R l^H) must equal the tile block. Ablation a = conj(M) R S^-1.
Mrows = gap_window_rows(fx, q)
B_ref = B_from_tile(Mrows, Vfull_true[q])
lM = (Mrows @ R) / S[None, :]
aC = (np.conj(Mrows) @ R) / S[None, :]
Vc_R_full = vc_tile(Phi_R, v_full)
Vc_C_full = vc_tile(Phi_C, v_full)
eR = relF(np.conj(lM @ Vc_R_full @ lM.conj().T), B_ref)
eC = relF(aC @ Vc_C_full @ aC.conj().T, B_ref)
rmR = float(np.max(np.linalg.norm(lM, axis=1)))
rmC = float(np.max(np.linalg.norm(aC, axis=1)))
print(f"[gate1] B routes | R: relF={eR:.2e} rowmax(l)={rmR:.3f} (<=1: leverage "
      f"bound holds) | C-ablation: relF={eC:.2e} rowmax(a)={rmC:.3f}")
assert eR < 1e-10 and eC < 1e-10
assert rmR <= 1.0 + 1e-8, "leverage bound violated — flavor pairing wrong"
OUT["rowmax"] = (rmR, rmC)
print(f"[gate1] ||Phi_R||_op = {np.linalg.norm(Phi_R, 2):.3e} "
      f"(sigma_max = {S[0]:.3e})")

def l_rows(Mr, R, S, r):
    return (Mr @ R[:, :r]) / S[:r][None, :]

def a_rows_C(Mr, R, S, r):
    return (np.conj(Mr) @ R[:, :r]) / S[:r][None, :]

# =====================================================================
# GATE 2 — seam
# =====================================================================
q_seam = int(np.where((fx.iq3 == [2, 0, 0]).all(1))[0][0])
G0 = np.array([1, 0, 0])
zr = recon(fx, q_seam)
qp_frac = fx.qwrap[q_seam] + G0
gv_shift = fx.gvec[q_seam].copy()
gv_shift[:, :] = gv_shift - G0[:, None]
zt_shift = to_sphere(fx, zr, q_seam, qfrac=qp_frac, gvec=gv_shift)
n = int(fx.ngk[q_seam])
seam_ii = relF(zt_shift[:, :n], fx.ZG[q_seam][:, :n])
print(f"[gate2] (ii) zbar_(q+G0)(G-G0) == zbar_q(G): {seam_ii:.2e}")
assert seam_ii < 1e-12

d_seam = np.exp(-2j * np.pi * (fx.rmu_frac @ G0))
Dph = d_seam if CONJ_LEFT else np.conj(d_seam)
C_seam = (np.conj(Dph)[:, None] * C_fall[q_seam]) * Dph[None, :]
zt_seam = np.conj(d_seam)[:, None] * zt_shift[:, :n]
Kf_seam = qp_frac[:, None] + gv_shift[:, :n].astype(np.float64)
v_seam, _ = vdim_at(fx, Kf_seam)
V_seam = make_Vq_generic(np.pad(zt_seam, ((0, 0), (0, fx.ngkmax - n))), v_seam, n)
M_seam = gap_window_rows(fx, q_seam) * d_seam[None, :]
B_seam = B_from_tile(M_seam, V_seam)
B_orig = B_from_tile(gap_window_rows(fx, q_seam), Vfull_true[q_seam])
seam_i = relF(B_seam, B_orig)
print(f"[gate2] (i) physical block across seam (X D_G0 + relabel): {seam_i:.2e}")
assert seam_i < 1e-12

# =====================================================================
# GATE 3 — TRS
# =====================================================================
trs = []
for q in range(fx.nq):
    mq3 = (-fx.iq3[q]) % np.array(fx.kgrid)
    mq = int((mq3[0] * fx.kgrid[1] + mq3[1]) * fx.kgrid[2] + mq3[2])
    trs.append(relF(Vfull_true[mq], np.conj(Vfull_true[q])))
print(f"[gate3] TRS V(-q) vs conj(V(q)): med={np.median(trs):.2e} "
      f"max={np.max(trs):.2e}")

# =====================================================================
# precompute per q: v-splits, Phi (both flavors), Vc tiles, M rows, LR-exact
# =====================================================================
alphas = alpha_ladder(fx)
C_ALPHAS = list(alphas.keys())
RANKS = [640, 480, 320, 160]
vsr_by_qa, vlr_by_qa = {}, {}
VcsrR_by_qa, VcsrC_by_qa = {}, {}
vfull_by_q, Mrows_by_q = {}, {}
for q in range(fx.nq):
    Kf, n = sphere_K(fx, q)
    v_full, _ = vdim_at(fx, Kf)
    vfull_by_q[q] = v_full
    PhiR, _ = phi_from_zeta(fx, q, R_by_q[q], S_by_q[q], phi_conj=False)
    PhiC, _ = phi_from_zeta(fx, q, R_by_q[q], S_by_q[q], phi_conj=True)
    Mrows_by_q[q] = gap_window_rows(fx, q)
    for ca, a_int in alphas.items():
        vs, vl = v_split_at(fx, Kf, a_int)
        vsr_by_qa[(q, ca)] = vs
        vlr_by_qa[(q, ca)] = vl
        VcsrR_by_qa[(q, ca)] = vc_tile(PhiR, vs)
        VcsrC_by_qa[(q, ca)] = vc_tile(PhiC, vs)

# on-grid alpha-invariance identity (R flavor; C checked at gate1)
inv_err = []
for q in range(fx.nq):
    R, S = R_by_q[q], S_by_q[q]
    refR = (S[:, None] * (R.conj().T @ np.conj(Vfull_true[q]) @ R)) * S[None, :]
    PhiR, _ = phi_from_zeta(fx, q, R, S, phi_conj=False)
    for ca in C_ALPHAS:
        VcL = vc_tile(PhiR, vlr_by_qa[(q, ca)])
        inv_err.append(relF(VcsrR_by_qa[(q, ca)] + VcL, refR))
print(f"[gate4] on-grid alpha-invariance Vc_sr+Vc_lr == S R^H conj(V) R S: "
      f"max={np.max(inv_err):.2e}")
assert np.max(inv_err) < 1e-12

print("[prep] regauging WFN fields to the fixture band gauge ...")
UF, rg_res = build_regauged_fields(fx, wfn)
print(f"[prep] regauge residual at centroids: med={np.median(rg_res):.2e} "
      f"max={np.max(rg_res):.2e}  (code loader re-gauges multiplets at k!=0)")
assert np.max(rg_res) < 1e-5
OUT["regauge_resid"] = rg_res

print("[prep] building exact pair form factors F_p from regauged fields ...")
occ = fx.nocc
vb = list(range(occ - 3, occ))
cb = list(range(occ, occ + 3))
u_cache = {}
for k in range(fx.nk):
    u_cache[k] = UF[k][vb + cb]
B_lr_exact, B_lr_fit, B_full_exact = {}, {}, {}
fitq_rows = []
for q0 in range(fx.nq):
    n = int(fx.ngk[q0])
    fi = prep.flat_idx(fx, fx.gvec[q0])[:n]
    ph = np.exp(-2j * np.pi * (fx.rfrac @ fx.qwrap[q0]))
    F = np.zeros((fx.nk * 9, n), dtype=np.complex128)
    r = 0
    for k in range(fx.nk):
        uc = u_cache[fx.kmq_idx[q0, k]][3:]
        uv = u_cache[k][:3]
        rho = np.einsum("csm,vsm->cvm", np.conj(uc), uv).reshape(9, fx.n_rtot)
        # stored zeta~ is the UNNORMALIZED FFT (validated: to_sphere == plain
        # fftn reproduces ZG at 2.3e-16); F must match: NO 1/N_r here.
        box = np.fft.fftn((rho * ph[None, :]).reshape(9, *fx.fg),
                          axes=(1, 2, 3), norm="backward")
        F[r:r + 9] = box.reshape(9, fx.n_rtot)[:, fi]
        r += 9
    v_full = vfull_by_q[q0]
    Bfe = np.conj(F * v_full[None, :n]) @ F.T
    B_full_exact[q0] = Bfe
    fitq_rows.append(relF(B_from_tile(Mrows_by_q[q0], Vfull_true[q0]), Bfe))
    for ca in C_ALPHAS:
        vl = vlr_by_qa[(q0, ca)]
        B_lr_exact[(q0, ca)] = np.conj(F * vl[None, :n]) @ F.T
        B_lr_fit[(q0, ca)] = B_from_tile(
            Mrows_by_q[q0], make_Vq_generic(fx.ZG[q0], vl, n))
print(f"[prep] ISDF fit quality on the gap-window B (fit vs exact, full v): "
      f"med={np.median(fitq_rows):.3e} max={np.max(fitq_rows):.3e}")
OUT["fit_quality_B"] = np.array(fitq_rows)
lr_cons = {ca: np.median([relF(B_lr_fit[(q, ca)], B_lr_exact[(q, ca)])
                          for q in range(fx.nq)]) for ca in C_ALPHAS}
print("[prep] LR-channel fit-vs-exact consistency med relF per c_alpha: "
      + "  ".join(f"{ca}:{v:.2e}" for ca, v in lr_cons.items()))
OUT["lr_consistency"] = {str(k): float(v) for k, v in lr_cons.items()}

B_true = {q: B_from_tile(Mrows_by_q[q], Vfull_true[q]) for q in range(fx.nq)}
B_sr_true = {(q, ca): B_from_tile(
    Mrows_by_q[q], make_Vq_generic(fx.ZG[q], vsr_by_qa[(q, ca)], int(fx.ngk[q])))
    for q in range(fx.nq) for ca in C_ALPHAS}

print("[prep] exciton H_true per q0 ...")
H_base = {}
eig_true = {}
for q0 in range(fx.nq):
    Hb = assemble_H(fx, q0, np.zeros_like(B_true[q0]))
    H_base[q0] = Hb
    eig_true[q0] = lowest_eigs(Hb + B_true[q0] / fx.nq)
OUT["eig_true"] = np.array([eig_true[q] for q in range(fx.nq)])

def exciton_shift(q0, B_pred):
    e = lowest_eigs(H_base[q0] + B_pred / fx.nq)
    return np.abs(e - eig_true[q0]) * RY_TO_MEV

# =====================================================================
# BASELINE REBASE
# =====================================================================
print("\n================ BASELINE REBASE (Sec 3.5 ladder, primary metrics) ====")
zeta_r = np.stack([recon(fx, q) for q in range(fx.nq)])
Z_r = np.stack([C_fall[q] @ zeta_r[q] for q in range(fx.nq)])
Cq_flat = C_fall.reshape(fx.nq, -1)
Zr_flat = Z_r.reshape(fx.nq, -1)
zr_flat = zeta_r.reshape(fx.nq, -1)

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

BL_METHODS = [("masterzeta", None, None), ("interpCZ", "raw", 0.0),
              ("interpCZ", "tikhonov", 1e-6), ("interpCZ", "rankcut", 1e-4),
              ("interpCZ", "rankcut", 1e-2), ("zetadirect", None, None)]
NR = 7
baseline = {m: {"B": [], "Bp90": [], "tile": [], "exc": []}
            for m in range(len(BL_METHODS))}
for q0 in range(fx.nq):
    tr = [q for q in range(fx.nq) if q != q0]
    w = w_fourier(fx, fx.qfr[q0], fx.qfr[tr], NR)
    n = int(fx.ngk[q0])
    for mi, (meth, mode, lam) in enumerate(BL_METHODS):
        if meth == "masterzeta":
            if q0 == 0:
                continue
            zt = to_sphere(fx, zeta_r[0], q0)
        elif meth == "zetadirect":
            zt = to_sphere(fx, (w @ zr_flat[tr]).reshape(fx.nmu, fx.n_rtot), q0)
        else:
            C0 = (w @ Cq_flat[tr]).reshape(fx.nmu, fx.nmu)
            Z0 = (w @ Zr_flat[tr]).reshape(fx.nmu, fx.n_rtot)
            zt = to_sphere(fx, solve_zeta(C0, Z0, mode, lam), q0)
        Vp = make_Vq_generic(zt, vfull_by_q[q0], n)
        Bp = B_from_tile(Mrows_by_q[q0], Vp)
        rB, rmed, rp90 = B_metrics(Bp, B_true[q0])
        baseline[mi]["B"].append(rB)
        baseline[mi]["Bp90"].append(rp90)
        baseline[mi]["tile"].append(relF(Vp, Vfull_true[q0]))
        baseline[mi]["exc"].append(np.max(exciton_shift(q0, Bp)))
print(f"{'method':>22} {'B relF med':>11} {'B relF max':>11} {'elem p90':>10} "
      f"{'tile med':>10} {'exc med(meV)':>12} {'exc max(meV)':>12}")
for mi, (meth, mode, lam) in enumerate(BL_METHODS):
    tag = meth if mode is None else f"{meth}-{mode}-{lam:g}"
    d = baseline[mi]
    print(f"{tag:>22} {np.median(d['B']):>11.3e} {np.max(d['B']):>11.3e} "
          f"{np.median(d['Bp90']):>10.3e} {np.median(d['tile']):>10.3e} "
          f"{np.median(d['exc']):>12.3e} {np.max(d['exc']):>12.3e}")
OUT["baseline"] = baseline

# =====================================================================
# C1 — transported target-frame LOO
# =====================================================================
print("\n================ C1: target-frame transported V^SR ====================")
pairs = [(a, b) for a in range(fx.nk) for b in range(fx.nk) if a != b]
O_wfn = band_overlaps_fields(fx, UF, pairs)   # exact, fixture gauge
O_cen = band_overlaps_centroid(fx, pairs)
d_ov = [np.max(np.abs(O_wfn[p] - O_cen[p])) for p in pairs]
smin_ov = [np.min(np.linalg.svd(O_wfn[p], compute_uv=False)) for p in pairs]
print(f"[transport] band-overlap WFN vs centroid-quadrature: med|d|="
      f"{np.median(d_ov):.2e} max={np.max(d_ov):.2e}")
print(f"[transport] window-subspace overlap min-sv: med={np.median(smin_ov):.3f} "
      f"min={np.min(smin_ov):.3f}")
OUT["overlap_dev"] = (float(np.median(d_ov)), float(np.max(d_ov)))
OUT["overlap_smin"] = (float(np.median(smin_ov)), float(np.min(smin_ov)))

cosA = edge_links_T(fx, 1, 2, O_wfn, R_by_q, S_by_q, 640, CONJ_LEFT, True,
                    H_cache=None)[1]
cosB = edge_links_T(fx, 1, 2, O_wfn, R_by_q, S_by_q, 640, CONJ_LEFT, False,
                    H_cache=None)[1]
print(f"[transport] edge(1<-2) cosines | t_conj=True: min={cosA.min():.4f} "
      f"med={np.median(cosA):.4f} | t_conj=False: min={cosB.min():.4f} "
      f"med={np.median(cosB):.4f}")
T_CONJ = bool(np.median(cosA) > np.median(cosB))
print(f"[transport] resolved t_conj={T_CONJ}")

prep.save_flags({"CONJ_LEFT": bool(CONJ_LEFT), "PHI_CONJ": False,
                 "A_CONJ": True, "T_CONJ": T_CONJ, "FLAVOR": "response"})

print("[C1] building all 72 cross-Grams H ...")
H_cache = {}
T_by = {}
cos_by = {}
for q0 in range(fx.nq):
    for qi in range(fx.nq):
        if qi == q0:
            continue
        for r in RANKS:
            T, cos = edge_links_T(fx, q0, qi, O_wfn, R_by_q, S_by_q, r,
                                  CONJ_LEFT, T_CONJ, H_cache=H_cache)
            T_by[(q0, qi, r)] = T
            if r == 640:
                cos_by[(q0, qi)] = cos
cos_all = np.array([cos_by[e] for e in cos_by])
print(f"[diagB] principal cosines over 72 edges (r=640): "
      f"min={cos_all.min():.4f} p1={np.percentile(cos_all,1):.4f} "
      f"med={np.median(cos_all):.4f} "
      f"frac<0.9={np.mean(cos_all < 0.9):.3f} frac<0.99={np.mean(cos_all < 0.99):.3f}")
OUT["cos_all_summary"] = [float(cos_all.min()), float(np.median(cos_all)),
                          float(np.mean(cos_all < 0.9))]

hol = []
kg = fx.kgrid
for ix in range(kg[0]):
    for iy in range(kg[1]):
        def qat(ax, ay):
            return int((((ix + ax) % kg[0]) * kg[1] + ((iy + ay) % kg[1])) * kg[2])
        loop = [qat(0, 0), qat(1, 0), qat(1, 1), qat(0, 1)]
        W = np.eye(640, dtype=np.complex128)
        for a, b in zip(loop, loop[1:] + loop[:1]):
            key = (b, a, 640)
            if key not in T_by:
                T, _ = edge_links_T(fx, b, a, O_wfn, R_by_q, S_by_q, 640,
                                    CONJ_LEFT, T_CONJ, H_cache=H_cache)
                T_by[key] = T
            W = T_by[key] @ W
        hol.append(np.linalg.norm(W - np.eye(640)) / np.sqrt(640))
print(f"[diagD] plaquette holonomy ||W-I||_F/sqrt(r): med={np.median(hol):.3e} "
      f"max={np.max(hol):.3e}")
OUT["holonomy"] = np.array(hol)

def run_loo(q0, wts, rank, ca, flavor="R"):
    r = rank
    Vc = np.zeros((r, r), dtype=np.complex128)
    tiles = VcsrR_by_qa if flavor == "R" else VcsrC_by_qa
    for qi, w in wts.items():
        T = T_by[(q0, qi, r)]
        Vc = Vc + w * (T @ tiles[(qi, ca)][:r, :r] @ T.conj().T)
    Vc = 0.5 * (Vc + Vc.conj().T)
    if flavor == "R":
        lM = l_rows(Mrows_by_q[q0], R_by_q[q0], S_by_q[q0], r)
        B_sr = np.conj(lM @ Vc @ lM.conj().T)
    else:
        aM = a_rows_C(Mrows_by_q[q0], R_by_q[q0], S_by_q[q0], r)
        B_sr = aM @ Vc @ aM.conj().T
    return B_sr, B_sr + B_lr_exact[(q0, ca)], Vc

# null test (R flavor)
null_err = []
for q0 in range(fx.nq):
    lM = l_rows(Mrows_by_q[q0], R_by_q[q0], S_by_q[q0], 640)
    B_sr = np.conj(lM @ VcsrR_by_qa[(q0, 1.0)] @ lM.conj().T)
    null_err.append(relF(B_sr + B_lr_fit[(q0, 1.0)], B_true[q0]))
print(f"[null] on-grid no-interp reconstruction (R flavor, fit-LR): "
      f"max relF = {np.max(null_err):.2e}  (must be machine)")
assert np.max(null_err) < 1e-10

results = {}
print(f"\n[C1 LOO 3x3] PRIMARY gap-window B metrics (R flavor)")
print(f"{'w':>4} {'r':>4} {'c_a':>4} | {'SR relF med':>11} {'SR max':>9} | "
      f"{'TOT relF med':>12} {'TOT max':>9} {'p90elem':>9} | "
      f"{'exc med':>9} {'exc max':>9} | {'tile med':>9}")
for wname in ["nn4", "f7"]:
    for rank in RANKS:
        for ca in C_ALPHAS:
            srs, tots, p90s, excs, tiles_e = [], [], [], [], []
            for q0 in range(fx.nq):
                if wname == "nn4":
                    wts = w_nn_avg(fx, q0)
                else:
                    tr = [q for q in range(fx.nq) if q != q0]
                    wv = w_fourier(fx, fx.qfr[q0], fx.qfr[tr], NR)
                    wts = {qi: wv[j] for j, qi in enumerate(tr)}
                B_sr, B_tot, Vc_pred = run_loo(q0, wts, rank, ca, "R")
                srs.append(relF(B_sr, B_sr_true[(q0, ca)]))
                rB, rmed, rp90 = B_metrics(B_tot, B_true[q0])
                tots.append(rB)
                p90s.append(rp90)
                excs.append(np.max(exciton_shift(q0, B_tot)))
                # SECONDARY: reconstructed SR tile (diagnostic only).
                r = rank
                R, S = R_by_q[q0], S_by_q[q0]
                W = (R[:, :r] / S[:r][None, :])
                Vtile = np.conj(W @ Vc_pred @ W.conj().T)   # back to make-conv
                Vsrt = make_Vq_generic(fx.ZG[q0], vsr_by_qa[(q0, ca)],
                                       int(fx.ngk[q0]))
                tiles_e.append(relF(Vtile, Vsrt))
            key = (wname, rank, ca)
            results[key] = dict(sr=srs, tot=tots, p90=p90s, exc=excs,
                                tile=tiles_e)
            print(f"{wname:>4} {rank:>4} {ca:>4} | {np.median(srs):>11.3e} "
                  f"{np.max(srs):>9.2e} | {np.median(tots):>12.3e} "
                  f"{np.max(tots):>9.2e} {np.median(p90s):>9.2e} | "
                  f"{np.median(excs):>9.3f} {np.max(excs):>9.3f} | "
                  f"{np.median(tiles_e):>9.2e}")
OUT["c1_results"] = {str(k): v for k, v in results.items()}

# wrong-flavor ablation (nn4, r=640, all alphas)
print("\n[C1 ABLATION] conj-flavor (transport applied to the wrong bundle):")
for ca in C_ALPHAS:
    tots = []
    for q0 in range(fx.nq):
        _, B_tot, _ = run_loo(q0, w_nn_avg(fx, q0), 640, ca, "C")
        tots.append(B_metrics(B_tot, B_true[q0])[0])
    print(f"   nn4 r=640 c_a={ca}: TOT relF med={np.median(tots):.3e} "
          f"max={np.max(tots):.3e}")

print("\n[C1] alpha-flatness of TOTAL (nn4, r=640) per-alpha med TOT relF:")
for ca in C_ALPHAS:
    print(f"   c_alpha={ca}: {np.median(results[('nn4', 640, ca)]['tot']):.3e}")

# =====================================================================
# GATE 5 — gauge randomization (R flavor)
# =====================================================================
print("\n[gate5] gauge-randomization invariance at q0=4 (nn4, r=640, c_a=1):")
q0 = 4
wts = w_nn_avg(fx, q0)
B_ref_pred = run_loo(q0, wts, 640, 1.0, "R")[1]

rng = np.random.default_rng(7)
def rand_u(n):
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    return Q

# (a) band gauge
U_by_k = [rand_u(fx.nb) for _ in range(fx.nk)]
psi_g = np.einsum("kmsr,kmn->knsr", fx.psi, np.stack(U_by_k))
O_g = {}
for (ka, kb) in pairs:
    O_g[(ka, kb)] = np.stack(U_by_k)[ka].conj().T @ O_wfn[(ka, kb)] @ np.stack(U_by_k)[kb]
Hc_g = {}
Vc_g = np.zeros((640, 640), dtype=np.complex128)
for qi, w in wts.items():
    T, _ = edge_links_T(fx, q0, qi, O_g, R_by_q, S_by_q, 640, CONJ_LEFT, T_CONJ,
                        psi=psi_g, H_cache=Hc_g)
    Vc_g += w * (T @ VcsrR_by_qa[(qi, 1.0)] @ T.conj().T)
Vc_g = 0.5 * (Vc_g + Vc_g.conj().T)
lM = l_rows(Mrows_by_q[q0], R_by_q[q0], S_by_q[q0], 640)
B_g = np.conj(lM @ Vc_g @ lM.conj().T) + B_lr_exact[(q0, 1.0)]
print(f"   band-gauge: relF(B_pred) = {relF(B_g, B_ref_pred):.2e}")
OUT["gauge_band"] = relF(B_g, B_ref_pred)

# (b) aux frame gauge
R_g, S_gv = {}, {}
for q in range(fx.nq):
    lam = lam_by_q[q]
    ph = np.exp(2j * np.pi * rng.random(640))
    G = np.diag(ph).astype(np.complex128)
    i = 0
    while i < 640:
        j = i + 1
        while j < 640 and abs(lam[j] - lam[i]) < 1e-10 * lam[0]:
            j += 1
        if j - i > 1:
            G[i:j, i:j] = rand_u(j - i)
        i = j
    R_g[q] = R_by_q[q] @ G
    S_gv[q] = S_by_q[q]
Vc_gg = np.zeros((640, 640), dtype=np.complex128)
for qi, w in wts.items():
    PhiR_i, _ = phi_from_zeta(fx, qi, R_g[qi], S_gv[qi], phi_conj=False)
    Vcsr_i = vc_tile(PhiR_i, vsr_by_qa[(qi, 1.0)])
    T, _ = edge_links_T(fx, q0, qi, O_wfn, R_g, S_gv, 640, CONJ_LEFT, T_CONJ,
                        H_cache=H_cache)
    Vc_gg += w * (T @ Vcsr_i @ T.conj().T)
Vc_gg = 0.5 * (Vc_gg + Vc_gg.conj().T)
lM_g = (Mrows_by_q[q0] @ R_g[q0]) / S_gv[q0][None, :]
B_gg = np.conj(lM_g @ Vc_gg @ lM_g.conj().T) + B_lr_exact[(q0, 1.0)]
print(f"   aux-frame-gauge: relF(B_pred) = {relF(B_gg, B_ref_pred):.2e}")
OUT["gauge_aux"] = relF(B_gg, B_ref_pred)

import json
with open(f"{prep.STUDY}/proto0_a_results.json", "w") as f:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    json.dump(clean(OUT), f)
print("\n[proto0_a] DONE")
