"""stress_si3d — STRESS TARGET 2 (sec-13.3 scope caveat, 3D-bulk variant):
the K-ball LR fit program on the Si 4x4x4 full-BZ fixture (work_old), where
q_z is continuous — K_z = G_z is NOT a discrete channel, so the natural
generalization is ONE 3D polynomial (x Gaussian) basis over the whole
LR ball. On-grid LOO over all 64 q, physical B metric + excitons.

Si is the campaign's hard case (fcc primitive cell, bare-3D Coulomb with
the miniBZ-averaged head; C_R falloff under-resolved at 4x4x4 — ingredient
scheme needs nR13 to reach 2.9-3.1e-3, sec 11.6/offgrid_si). Expected:
tile-level scheme tracks the ingredient scheme with the SR stencil error
dominating; the question quantified here is whether the GLOBAL 3D fit
carries the LR channel at the exact-LR ceiling like the slab case
(fit is 'harder — slower falloff' per the stress brief).

Construction (mirrors tile_prep/lr_prep, Si-adapted; per-element math
identical, only the kernel and the basis differ):
  v(K)   = 8 pi/K^2 / celvol, head at (q != 0, G=0) = miniBZ MC average
           (production convention, gate-verified vs disk); q=0 head zeroed.
  split  v_LR = v exp(-K^2/4 alpha^2), v_SR = v (-expm1(...)) — head
           patched BEFORE the window, so v_SR + v_LR == v identically.
  gset   fixed global Miller set, min over the 3D BZ (9^3 offsets) of
           |q+G|^2 <= 4 alpha^2 ln(1/eps_LR).
  F      phase-factored cleaned channels on gset (Tikhonov gauge
           S_eps = R g_eps(lam) R^H, eps_rel 1e-4 — sec 13.1 mandate).
  fit    weighted LSQ (w = v_LR incl. patched head) of 3D polynomials
           K^a K^b K^c, a+b+c <= d (optionally x even-tempered Gaussians)
           over ALL (q, G) samples; per-q normal blocks -> honest LOO.
  D      Sum_w V_SR_c(q_i) + V_model(q0);  anchors C (exact target LR)
           and F (channel interp). Stencils nR13 (Si-required) and nR7.
alpha ladder: {1.5, 2.0} x Dq, Dq = min|b_i|/4 (the slab study's optimum
sat at 1.5-3 x Dq).

Run: JID=<jid> ./proto1_run.sh python3 -u stress_si3d.py
"""
import time
import numpy as np

from proto1_prep import relF, truncR_weights
from offgrid_prep import (SiOldFixture, fix_sphere_wrap, run_gates,
                          sorted_stencil, top_decile_rel, build_Hdir,
                          exciton_evs, RY2MEV)
from tile_prep import B_tile, EPS_LR, LN_EPS

EPS_TIK = 1e-4
t00 = time.time()
NPZ = {}

fx = SiOldFixture()
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0, 42), wfn_check=False)
fx.vq(1, kind="bare3d")                    # populate the miniBZ head table
HVTAB = fx._vhead_tab
Dq = min(np.linalg.norm(fx.bvec[i]) for i in range(3)) / fx.kgrid[0]
ALPHAS = tuple(round(f * Dq, 3) for f in (1.5, 2.0))
print(f"[si3d] Dq = {Dq:.4f} 1/bohr; alphas {ALPHAS}; "
      f"sphere cutoff {fx.zeta_cutoff} Ry")

# ---------------- Si tile machinery (bare3d + head) ------------------------
print("[prep] eigh(C_q) + Tik gauge + reference tiles")
eig = []
for q in range(fx.nq):
    lam, R = np.linalg.eigh(0.5 * (C_q[q] + C_q[q].conj().T))
    eig.append((lam[::-1].copy(), R[:, ::-1].copy()))
Stik = []
for q in range(fx.nq):
    lam, R = eig[q]
    g = lam ** 2 / (lam ** 2 + (EPS_TIK * lam[0]) ** 2)
    Stik.append((R * g[None, :]) @ R.conj().T)
V_ref = np.stack([fx.make_Vq(fx.ZG[q], q, kind="bare3d")
                  for q in range(fx.nq)])
Vc = np.stack([np.conj(Stik[q]) @ V_ref[q] @ np.conj(Stik[q])
               for q in range(fx.nq)])
print(f"  [prep] cleaned tiles done ({time.time()-t00:.0f}s)")


def gset3d(alpha):
    K2max = 4.0 * alpha ** 2 * LN_EPS
    Kmax = np.sqrt(K2max)
    nmax = [int(np.ceil(Kmax / np.linalg.norm(fx.bvec[i]))) + 1
            for i in range(3)]
    gr = [np.arange(-n, n + 1) for n in nmax]
    GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
    Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0)
    ts = np.linspace(-0.5, 0.5, 9, endpoint=False)
    m = np.full(Gall.shape[1], np.inf)
    for tx in ts:
        for ty in ts:
            for tz in ts:
                qf = np.array([tx, ty, tz])
                K = fx.bvec.T @ (qf[:, None] + Gall.astype(np.float64))
                m = np.minimum(m, np.sum(K * K, axis=0))
    return np.ascontiguousarray(Gall[:, m <= K2max])


def v_set(qfrac, GS, kind, alpha):
    """bare-3D kernel with production head on an explicit Miller set."""
    K = fx.bvec.T @ (np.asarray(qfrac)[:, None] + GS.astype(np.float64))
    K2 = np.sum(K * K, axis=0)
    zero = K2 < 1e-12
    v = np.where(zero, 0.0, 8.0 * np.pi / np.where(zero, 1.0, K2)
                 / fx.celvol)
    qn = np.asarray(qfrac)
    if np.max(np.abs(qn)) > 1e-12:          # on-grid q != 0: patched head
        kg = fx.kgrid
        qi = tuple(int(np.round(qn[c] * kg[c])) % int(kg[c])
                   for c in range(3))
        hv = float(HVTAB[qi])
        g0 = np.all(GS == 0, axis=0)
        if hv != 0.0 and g0.any():
            v = np.where(g0, hv, v)
    if kind == "lr":
        v = v * np.exp(-K2 / (4.0 * alpha ** 2))
    elif kind == "sr":
        v = v * (-np.expm1(-K2 / (4.0 * alpha ** 2)))
    return v, K


def sphere_slot(q, GS):
    n = int(fx.ngk[q])
    lut = {tuple(g): i for i, g in enumerate(fx.gvec[q][:, :n].T)}
    return np.array([lut.get(tuple(g), -1) for g in GS.T])


def F_channels(GS):
    """Tik-cleaned phase-factored channels on GS at every coarse q."""
    out = np.empty((fx.nq, fx.n_mu, GS.shape[1]), dtype=np.complex128)
    nmiss, wmax = 0, 0.0
    for q in range(fx.nq):
        zt = Stik[q] @ fx.ZG[q]
        idx = sphere_slot(q, GS)
        miss = idx < 0
        if miss.any():
            nmiss += int(miss.sum())
            K = fx.bvec.T @ (fx.qfr[q][:, None]
                             + GS[:, miss].astype(np.float64))
            K2 = np.sum(K * K, 0)
            wmax = max(wmax, float(np.exp(-K2.min()
                                          / (4 * max(ALPHAS) ** 2))))
        zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1),
                                              np.complex128)], 1)
        qG = fx.qfr[q][None, :] + GS.T.astype(np.float64)
        ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
        out[q] = ph * zt_ext[:, idx]
    print(f"  [info] F_channels: {nmiss} out-of-sphere (q,G) zero-filled; "
          f"worst Gaussian weight {wmax:.2e}")
    return out


def V_from_F(Fq, qfrac, GS, alpha):
    v, _ = v_set(qfrac, GS, "lr", alpha)
    qG = np.asarray(qfrac)[None, :] + GS.T.astype(np.float64)
    zt = np.exp(-2j * np.pi * (fx.rmu_frac @ qG.T)) * Fq
    A = zt * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


def VLR_exact_c(alpha):
    """Tik-cleaned exact LR tile on the stored sphere per q."""
    out = np.empty_like(V_ref)
    for q in range(fx.nq):
        v, n = fx.vq(q, kind="bare3d")
        _, K2, _ = fx.Kvecs(q)
        vlr = v[:n] * np.exp(-K2 / (4.0 * alpha ** 2))
        A = fx.ZG[q][:, :n] * np.sqrt(vlr)[None, :]
        VLR = np.conj(A) @ A.T
        out[q] = np.conj(Stik[q]) @ VLR @ np.conj(Stik[q])
    return out


# ---------------- 3D-ball weighted fit -------------------------------------
def poly3d(d):
    return [(None, a, b, c) for t in range(d + 1) for a in range(t + 1)
            for b in range(t - a + 1) for c in [t - a - b]]


def gto3d(d, sigmas):
    return [(s, a, b, c) for s in sigmas for (_, a, b, c) in poly3d(d)]


def eval_basis3d(K, spec, alpha):
    s = 1.0 / (2.0 * alpha)
    x, y, z = K[0] * s, K[1] * s, K[2] * s
    K2 = K[0] ** 2 + K[1] ** 2 + K[2] ** 2
    cols = []
    for sig, a, b, c in spec:
        t = (x ** a) * (y ** b) * (z ** c) if (a or b or c) \
            else np.ones_like(x)
        if sig is not None:
            t = t * np.exp(-K2 / (4.0 * sig ** 2))
        cols.append(t)
    return np.stack(cols, 1)


class Si3DFit:
    RIDGE = 1e-11

    def __init__(self, GS, Fch, alpha, spec, tag=""):
        self.GS, self.alpha, self.spec, self.tag = GS, alpha, spec, tag
        nb = len(spec)
        self.AtA = np.empty((fx.nq, nb, nb))
        self.AtY = np.empty((fx.nq, nb, fx.n_mu), dtype=np.complex128)
        self._resid_src = []
        for q in range(fx.nq):
            w, K = v_set(fx.qfr[q], GS, "lr", alpha)
            Phi = eval_basis3d(K, spec, alpha)
            Y = Fch[q].T                       # (nG, nmu)
            Pw = Phi * w[:, None]
            self.AtA[q] = Phi.T @ Pw
            self.AtY[q] = Pw.T @ Y
            self._resid_src.append((Phi, w, Y))

    def n_coeff(self):
        return len(self.spec)

    def coeffs(self, exclude=None):
        sel = [q for q in range(fx.nq) if q != exclude]
        A = self.AtA[sel].sum(0)
        Y = self.AtY[sel].sum(0)
        A = A + self.RIDGE * (np.trace(A) / A.shape[0]) * np.eye(A.shape[0])
        return np.linalg.solve(A, Y)

    def model_F(self, C, qfrac):
        _, K = v_set(qfrac, self.GS, "lr", self.alpha)
        Phi = eval_basis3d(K, self.spec, self.alpha)
        return (Phi @ C).T

    def resid_stats(self, C):
        num = den = fib = 0.0
        for q in range(fx.nq):
            Phi, w, Y = self._resid_src[q]
            R = Y - Phi @ C
            num += float(np.sum(w[:, None] * np.abs(R) ** 2))
            den += float(np.sum(w[:, None] * np.abs(Y) ** 2))
            ws = float(w.sum())
            if ws > 0:
                mq = (w[:, None] * R).sum(0) / ws
                fib += float(ws * np.sum(np.abs(mq) ** 2))
        return np.sqrt(num / den), fib / max(num, 1e-300)


# ---------------- exciton/probe caches -------------------------------------
XW, TRUE_B = {}, {}
for q0 in range(fx.nq):
    XW[q0] = fx.gap_window_pairs(q0, 3, 3)
    TRUE_B[q0] = B_tile(XW[q0], V_ref[q0])
EXCQ = set(range(0, fx.nq, 8))
HD = {}
R4 = sorted_stencil(fx, [[a, b, c] for a in range(-1, 3)
                         for b in range(-1, 3) for c in range(-1, 3)])
STENCILS = (("nR13", R4[:13]), ("nR7", R4[:7]))

# ===========================================================================
for ALPHA in ALPHAS:
    ta = time.time()
    print(f"\n################ Si alpha = {ALPHA} "
          f"({ALPHA/Dq:.2f} Dq) ################")
    GS = gset3d(ALPHA)
    print(f"  [info] gset3d({ALPHA}): {GS.shape[1]} G; ball radius "
          f"{2*ALPHA*np.sqrt(LN_EPS):.3f} 1/bohr; sphere-tail bound "
          f"{np.exp(-fx.zeta_cutoff/(4*ALPHA**2)):.2e}; eps_LR={EPS_LR}")
    Fch = F_channels(GS)
    VLRc = VLR_exact_c(ALPHA)
    SR = Vc - VLRc
    # gate: own rebuild == cleaned exact LR (channels + kernel consistency)
    r = [relF(V_from_F(Fch[q], fx.qfr[q], GS, ALPHA), VLRc[q])
         for q in range(fx.nq)]
    print(f"  [gate] F_own_rebuild_vs_cleaned_VLR max {np.max(r):.3e}"
          + ("  OK" if np.max(r) < 1e-6 else "  ** FAIL **"))
    assert np.max(r) < 1e-6

    SIG = [ALPHA / np.sqrt(2.0), ALPHA, ALPHA * np.sqrt(2.0)]
    RUNGS = {"p3d_d2": poly3d(2), "p3d_d3": poly3d(3), "p3d_d4": poly3d(4),
             "rich_g3d4": gto3d(4, SIG)}
    fits, Cown, fid = {}, {}, {}
    for name, spec in RUNGS.items():
        cf = Si3DFit(GS, Fch, ALPHA, spec, tag=name)
        C = cf.coeffs()
        fits[name], Cown[name] = cf, C
        rr = [relF(V_from_F(cf.model_F(C, fx.qfr[q]), fx.qfr[q], GS, ALPHA),
                   VLRc[q]) for q in range(fx.nq)]
        wres, fibf = cf.resid_stats(C)
        fid[name] = (float(np.median(rr)), float(np.max(rr)))
        print(f"  [fit] {name:<10s} coeffs/mu {cf.n_coeff():>4d}  wres "
              f"{wres:.3f} (per-q coherent frac {fibf:.2f})  fidelity med "
              f"{fid[name][0]:.3e} max {fid[name][1]:.3e}")
        NPZ[f"si_a{ALPHA}_fid_{name}"] = np.array(rr)

    res = {}
    for q0 in range(fx.nq):
        train = [q for q in range(fx.nq) if q != q0]
        Cloo = {n: fits[n].coeffs(exclude=q0) for n in RUNGS}
        for sname, Rset in STENCILS:
            w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rset)
            SRi = np.tensordot(w, SR[train], axes=(0, 0))
            preds = {}
            preds[f"{sname}_C_anchor"] = SRi + VLRc[q0]
            Fi = np.tensordot(w, Fch[train], axes=(0, 0))
            preds[f"{sname}_F_anchor"] = SRi + V_from_F(Fi, fx.qfr[q0],
                                                        GS, ALPHA)
            for n in RUNGS:
                preds[f"{sname}_D_{n}"] = SRi + V_from_F(
                    fits[n].model_F(Cloo[n], fx.qfr[q0]), fx.qfr[q0], GS,
                    ALPHA)
            for lbl, Vp in preds.items():
                Bp = B_tile(XW[q0], Vp)
                met = {"B": relF(Bp, TRUE_B[q0]),
                       "Bdec": top_decile_rel(Bp, TRUE_B[q0])}
                if q0 in EXCQ and sname == "nR13" and (
                        "anchor" in lbl or "d3" in lbl or "d4" in lbl):
                    if q0 not in HD:
                        HD[q0] = build_Hdir(fx, q0)
                        HD[(q0, "ev")] = exciton_evs(fx, *HD[q0],
                                                     TRUE_B[q0])
                    ev_p = exciton_evs(fx, *HD[q0], Bp)
                    met["exc_meV"] = float(
                        np.max(np.abs(ev_p - HD[(q0, "ev")])) * RY2MEV)
                res.setdefault(lbl, {})[q0] = met
        if q0 % 16 == 0:
            print(f"  q0={q0} done ({time.time()-ta:.0f}s)", flush=True)

    print(f"\n  ==== Si 4x4x4 alpha={ALPHA}: median / max over {fx.nq} "
          f"LOO targets ====")
    print(f"    {'label':<22s} {'c/mu':>5s} {'fid med':>9s} {'B med':>10s} "
          f"{'B max':>10s} {'exc med':>8s} {'exc max':>8s}")
    for lbl in sorted(res):
        rows = res[lbl]
        Bm = [rows[q]["B"] for q in rows]
        em = [rows[q]["exc_meV"] for q in rows if "exc_meV" in rows[q]]
        em_s = (f"{np.median(em):>8.3f} {np.max(em):>8.3f}" if em
                else "      --       --")
        key = lbl.split("_D_")[-1] if "_D_" in lbl else None
        f_s = f"{fid[key][0]:>9.2e}" if key in fid else "       --"
        nc = fits[key].n_coeff() if key in fid else (
            GS.shape[1] if "F_anchor" in lbl else 0)
        print(f"    {lbl:<22s} {nc:>5d} {f_s} {np.median(Bm):>10.3e} "
              f"{np.max(Bm):>10.3e} {em_s}")
        qs = sorted(rows)
        NPZ[f"si_a{ALPHA}__{lbl}__q0"] = np.array(qs)
        for k2 in ("B", "Bdec", "exc_meV"):
            NPZ[f"si_a{ALPHA}__{lbl}__{k2}"] = np.array(
                [rows[q].get(k2, np.nan) for q in qs], dtype=float)
    print(f"  [ingredient baseline, offgrid_si.log stage B]: nR13 "
          f"rankcut_1e-3 B med 2.870e-3 max 4.013e-3 (carries Z)")
    print(f"  [alpha={ALPHA}] done in {time.time()-ta:.0f}s")

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "stress_si3d_results.npz", **NPZ)
print(f"\n[stress_si3d] ALL DONE in {time.time()-t00:.0f}s")
