"""lr_basis_ladder — EXPERIMENT 2 of the LR compact-representation study:
basis ladder for the K-ball fit of the phase-factored form factors
M_mu(K), scored in the full F-scheme assembly (sec-12 LOO harness, swap
the LR channel only).

Rungs (all per-Gz in-plane, weighted LSQ, lr_prep machinery):
  unif-poly-d  : same poly degree every Gz channel (fidelity ladder; d=2 is
                 the storage twin of the FAILED literal o2 moments, 6/Gz)
  budget-*     : per-|Gz| degree allocation following the v_LR weight
                 shares — the <= n_mu x (10-30) target rungs
  rich         : gto3 x poly4 (|Gz|<=2) + gto3 x poly2 (rest) — fidelity
                 ceiling of the fit program and the SVD-compression source
  svd-r        : rank-r truncation of the rich fit in the weighted basis
                 metric ("learned multipoles": r coefficients per mu
                 against r shared analytic K-profiles)
Assemblies at LOO target q0 (w = nR7 stencil, train = all q != q0):
  D-style: sum_i w_i V_SR_c(q_i) + V_model[LOO coeffs](q0)   (honest; model
           fidelity enters directly, like the F-scheme's own structure)
  E-style: sum_i w_i [V_c - V_model_LOO](q_i) + V_model_LOO(q0)
           (consistent subtract/re-add; tolerant of model bias)
Anchors recomputed in-run (continuity with sec 12.1): F channel-interp
(must land at its logged 6.23e-3 med) and C exact-LR ceiling (5.22e-3).

Run: JID=<jid> ./proto1_run.sh python3 -u lr_basis_ladder.py MoS2_6x6
"""
import sys
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel, build_Hdir, exciton_evs, RY2MEV)
from tile_prep import TileStudy, B_tile, check_slab_axes
from lr_prep import (LRSamples, ChannelFit, spec_poly, spec_gto,
                     svd_compress, sample_matrix_svals, eval_basis)

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_6x6"
TIK = "--tik" in sys.argv       # Tikhonov gauge (lr_fiber_source verdict:
# the q-fiber is the HARD-CUT edge; S_eps = R g_eps(L) R^H with
# g = l^2/(l^2 + (1e-4 l0)^2) makes M(K) single-valued to ~1%)
ALPHA = 0.30
RC = 1e-4
EPS_TIK = 1e-4
SIG = [ALPHA / np.sqrt(2.0), ALPHA, ALPHA * np.sqrt(2.0)]
SVD_RANKS = [4, 8, 12, 16, 24, 32]
t00 = time.time()
NPZ = {}

print(f"[lr_ladder] fixture {FIXNAME}; alpha={ALPHA}, rc={RC}")
fx = Fixture(FIXNAME)
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12
V_ref = ts.V_ref
if TIK:
    print(f"  [info] TIKHONOV gauge, eps_rel={EPS_TIK} (sec 12.3 smooth "
          f"filter; cleaning op S = conj(S_eps), tile V_c = S V S)")
    Stik = []
    for q in range(fx.nq):
        lam, R = ts.eig[q]
        g = lam ** 2 / (lam ** 2 + (EPS_TIK * lam[0]) ** 2)
        Stik.append((R * g[None, :]) @ R.conj().T)
    Vc = np.stack([np.conj(Stik[q]) @ V_ref[q] @ np.conj(Stik[q])
                   for q in range(fx.nq)])
    VLRc = np.stack([np.conj(Stik[q])
                     @ fx.make_Vq(fx.ZG[q], q, kind="slab_lr", alpha=ALPHA)
                     @ np.conj(Stik[q]) for q in range(fx.nq)])
    lr = LRSamples(ts, None, ALPHA)
    for q in range(fx.nq):
        zt = Stik[q] @ fx.ZG[q]
        idx = ts.sphere_slot(q, lr.GS)
        zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1),
                                              np.complex128)], 1)
        qG = fx.qfr[q][None, :] + lr.GS.T.astype(np.float64)
        ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
        lr.Fch[q] = ph * zt_ext[:, idx]

    def clean_op(q):
        return np.conj(Stik[q])
else:
    lr = LRSamples(ts, RC, ALPHA)
    Vc = ts.Vc(RC)
    VLRc = ts.VLR_exact_c(RC, ALPHA)

    def clean_op(q):
        return np.conj(ts.P(q, RC))
GS = lr.GS
print(f"  [info] gset({ALPHA}): {lr.nG} G, Gz channels "
      f"{sorted(lr.gz_vals)}")
ws = lr.wshare_gz()
wabs = {}
for g, v in ws.items():
    wabs[abs(g)] = wabs.get(abs(g), 0.0) + v
print("  [info] v_LR weight share per |Gz|: "
      + " ".join(f"{g}:{wabs[g]:.4f}" for g in sorted(wabs)))

# gate: F own rebuild == PI V_LR PI (channel machinery continuity)
r = [relF(ts.V_from_F(lr.Fch[q], fx.qfr[q], GS, ALPHA), VLRc[q])
     for q in range(fx.nq)]
print(f"  [gate] F_own_rebuild_vs_PIVLRPI max {np.max(r):.3e}"
      + ("  OK" if np.max(r) < 1e-6 else "  ** FAIL **"))
assert np.max(r) < 1e-6

# ===========================================================================
# Stage T — Gz-truncation ceiling (exact channels, drop |Gz| > gzmax)
# ===========================================================================
print("\n[stageT] Gz-truncation ceiling (exact channels, own rebuild)")
for gzmax in (0, 1, 2, 3, 4):
    keep = np.abs(GS[2]) <= gzmax
    rl, rb = [], []
    for q in range(fx.nq):
        Ftr = lr.Fch[q] * keep[None, :]
        Vtr = ts.V_from_F(Ftr, fx.qfr[q], GS, ALPHA)
        rl.append(relF(Vtr, VLRc[q]))
        x = fx.gap_window_pairs(q, 3, 3)
        rb.append(relF(B_tile(x, Vc[q] - VLRc[q] + Vtr),
                       B_tile(x, V_ref[q])))
    print(f"  [T] |Gz|<={gzmax}: LR-tile relF med {np.median(rl):.3e}  "
          f"B med {np.median(rb):.3e} max {np.max(rb):.3e}")
    NPZ[f"gztrunc_{gzmax}"] = np.array([np.median(rl), np.median(rb),
                                        np.max(rb)])

# ===========================================================================
# Stage R — rung definitions + own-fit fidelity
# ===========================================================================
print("\n[stageR] rung fits + own-fidelity relF(V_model_own, PI V_LR PI)")


def budget_spec(bydeg):
    """bydeg: {|gz|: spec or None(drop)}; others dropped."""
    out = {}
    for g in lr.gz_vals:
        sp = bydeg.get(abs(g))
        if sp is not None:
            out[g] = sp
    return out


RUNGS = {}
for d in (0, 1, 2, 3):
    RUNGS[f"unif_poly{d}"] = {g: spec_poly(d) for g in lr.gz_vals}
RUNGS["b16p"] = budget_spec({0: spec_poly(2), 1: spec_poly(1),
                             2: spec_poly(0), 3: spec_poly(0)})
RUNGS["b26p"] = budget_spec({0: spec_poly(3), 1: spec_poly(2),
                             2: spec_poly(0), 3: spec_poly(0)})
RUNGS["b28g"] = budget_spec({0: spec_gto(2, SIG), 1: spec_gto(0, SIG),
                             2: spec_poly(0), 3: spec_poly(0)})
RUNGS["b45p"] = budget_spec({0: spec_poly(4), 1: spec_poly(3),
                             2: spec_poly(1), 3: spec_poly(0),
                             4: spec_poly(0)})
RUNGS["rich"] = {g: (spec_gto(4, SIG) if abs(g) <= 2 else spec_gto(2, SIG))
                 for g in lr.gz_vals}

fits, Cown, fid = {}, {}, {}
for name, specs in RUNGS.items():
    t0 = time.time()
    cf = ChannelFit(lr, specs, tag=name)
    C = cf.coeffs()
    fits[name], Cown[name] = cf, C
    rr = [relF(ts.V_from_F(cf.model_F(C, fx.qfr[q]), fx.qfr[q], GS, ALPHA),
               VLRc[q]) for q in range(fx.nq)]
    fid[name] = (float(np.median(rr)), float(np.max(rr)))
    st = cf.resid_stats(C)
    print(f"  [R] {name:<12s} coeffs/mu {cf.n_coeff():>3d}  wres "
          f"{st['rel']:.3f}  fidelity med {fid[name][0]:.3e} "
          f"max {fid[name][1]:.3e}  ({time.time()-t0:.0f}s)")
    NPZ[f"fid_{name}"] = np.array(rr)

# SVD rungs from the rich fit (own): spectrum + fidelity per rank
svals, Cr_own = svd_compress(fits["rich"], Cown["rich"], SVD_RANKS)
print("  [R] rich-fit weighted-metric svals (norm): "
      + " ".join(f"{s/svals[0]:.1e}" for s in svals[:32]))
NPZ["svals_fit"] = svals
ssamp = sample_matrix_svals(lr)
print("  [R] sample-matrix svals (norm, model-free effective rank): "
      + " ".join(f"{s/ssamp[0]:.1e}" for s in ssamp[:32]))
NPZ["svals_sample"] = ssamp
for rk in SVD_RANKS:
    rr = [relF(ts.V_from_F(fits["rich"].model_F(Cr_own[rk], fx.qfr[q]),
                           fx.qfr[q], GS, ALPHA), VLRc[q])
          for q in range(fx.nq)]
    fid[f"svd{rk}"] = (float(np.median(rr)), float(np.max(rr)))
    print(f"  [R] svd r={rk:<3d} coeffs/mu {rk:>3d}(+shared)  fidelity "
          f"med {fid[f'svd{rk}'][0]:.3e} max {fid[f'svd{rk}'][1]:.3e}")
    NPZ[f"fid_svd{rk}"] = np.array(rr)

# --- hybrid rungs motivated by the lr_singlevalued S3 finding (65% of the
# rich-fit residual is COHERENT per-(q,Gz) — a real q-fiber beyond K, ~10%
# of the LR signal; no K-only model can carry it, so carry it explicitly):
#   hyb : model + per-(q,Gz) weighted-mean residual vectors, stencil-
#         interpolated in q  (per-q storage n_mu x n_Gz ~ 17)
#   ftop: exact/interp F channels on the n highest-v_LR-weight G columns,
#         model tail on the rest  (per-q storage n_mu x n)
def fiber_corr(cf, C, qset):
    """{gz: (nq, nmu)} weighted-mean residual per (q, gz); rows only for
    q in qset (others zero)."""
    lrx = cf.lr
    out = {g: np.zeros((fx.nq, fx.n_mu), np.complex128) for g in cf.specs}
    for g, spec in cf.specs.items():
        c = lrx.cols[g]
        for q in qset:
            Phi = eval_basis(lrx.K[q][:2][:, c], spec, lrx.alpha)
            w = lrx.W[q][c]
            R = lrx.Fch[q][:, c].T - Phi @ C[g]
            ws = float(w.sum())
            if ws > 0:
                out[g][q] = (w[:, None] * R).sum(0) / ws
    return out


def apply_corr(Fmod, corr_w):
    """Add per-gz constant correction vectors {gz: (nmu,)} to model F."""
    out = Fmod.copy()
    for g, v in corr_w.items():
        out[:, lr.cols[g]] += v[:, None]
    return out


HYB_BASE = "b26p"
wmean = lr.W.mean(0)
TOPNS = (12, 30)
TOPS = {n: np.argsort(-wmean)[:n] for n in TOPNS}
# own fidelity of the hybrid rungs
corr_own = fiber_corr(fits[HYB_BASE], Cown[HYB_BASE], range(fx.nq))
rr_h, rr_t = [], {n: [] for n in TOPNS}
for q in range(fx.nq):
    Fm = fits[HYB_BASE].model_F(Cown[HYB_BASE], fx.qfr[q])
    Fh = apply_corr(Fm, {g: corr_own[g][q] for g in corr_own})
    rr_h.append(relF(ts.V_from_F(Fh, fx.qfr[q], GS, ALPHA), VLRc[q]))
    for n in TOPNS:
        Fx = Fm.copy()
        Fx[:, TOPS[n]] = lr.Fch[q][:, TOPS[n]]
        rr_t[n].append(relF(ts.V_from_F(Fx, fx.qfr[q], GS, ALPHA),
                            VLRc[q]))
ngz_mod = len(fits[HYB_BASE].specs)
fid["hyb26"] = (float(np.median(rr_h)), float(np.max(rr_h)))
print(f"  [R] hyb26 (b26p + per-(q,Gz) corr, {ngz_mod}/q): fidelity med "
      f"{fid['hyb26'][0]:.3e} max {fid['hyb26'][1]:.3e}")
NPZ["fid_hyb26"] = np.array(rr_h)
for n in TOPNS:
    fid[f"ftop{n}"] = (float(np.median(rr_t[n])), float(np.max(rr_t[n])))
    print(f"  [R] ftop{n} (exact top-{n} channels + b26p tail): fidelity "
          f"med {fid[f'ftop{n}'][0]:.3e} max {fid[f'ftop{n}'][1]:.3e}")
    NPZ[f"fid_ftop{n}"] = np.array(rr_t[n])

# LOO-coefficient stability for b26p (robustness proxy: the coefficients
# are GLOBAL — their only q-dependence is which q the LOO withholds)
cst = []
for q0 in range(fx.nq):
    Cl = fits["b26p"].coeffs(exclude=q0)
    cst.append(max(relF(Cl[g], Cown["b26p"][g]) for g in Cl))
print(f"  [R] b26p LOO-coefficient stability: med {np.median(cst):.3e} "
      f"max {np.max(cst):.3e}")
NPZ["coeff_stability_b26p"] = np.array(cst)

# ===========================================================================
# Stage L — LOO ladder (D-style all rungs, E-style top rungs, anchors)
# ===========================================================================
print(f"\n[stageL] LOO ladder ({fx.nq} targets, stencil nR7)")
R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                         for j in range(-2, 4)])[:7]
E_RUNGS = ("b26p", "rich")
EXC_LABELS = {"F_anchor", "C_anchor", "D_b26p", "D_rich", "D_svd16",
              "E_b26p", "E_rich", "D_hyb26", "D_ftop30"}
res = {}
for q0 in range(fx.nq):
    tq = time.time()
    train = [q for q in range(fx.nq) if q != q0]
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
    x = fx.gap_window_pairs(q0, 3, 3)
    B_true = B_tile(x, V_ref[q0])
    D_diag, Hdir = build_Hdir(fx, q0)
    ev_true = exciton_evs(fx, D_diag, Hdir, B_true)
    SR = Vc[train] - VLRc[train]
    SRi = np.tensordot(w, SR, axes=(0, 0))
    preds = {}
    # anchors
    Fi = np.tensordot(w, lr.Fch[train], axes=(0, 0))
    preds["F_anchor"] = SRi + ts.V_from_F(Fi, fx.qfr[q0], GS, ALPHA)
    PI0 = clean_op(q0)
    VLR0 = PI0 @ fx.make_Vq(fx.ZG[q0], q0, kind="slab_lr", alpha=ALPHA) @ PI0
    preds["C_anchor"] = SRi + VLR0
    # D-style rungs (honest LOO coefficients)
    Cloo = {}
    for name in RUNGS:
        Cloo[name] = fits[name].coeffs(exclude=q0)
        preds[f"D_{name}"] = SRi + ts.V_from_F(
            fits[name].model_F(Cloo[name], fx.qfr[q0]), fx.qfr[q0], GS,
            ALPHA)
    # SVD rungs (recompress the LOO rich fit per target)
    _, Cr = svd_compress(fits["rich"], Cloo["rich"], SVD_RANKS)
    for rk in SVD_RANKS:
        preds[f"D_svd{rk}"] = SRi + ts.V_from_F(
            fits["rich"].model_F(Cr[rk], fx.qfr[q0]), fx.qfr[q0], GS, ALPHA)
    # hybrid: model + interpolated per-(q,Gz) correction (honest: corr from
    # TRAIN residuals against the LOO model)
    corr = fiber_corr(fits[HYB_BASE], Cloo[HYB_BASE], train)
    corr_w = {g: np.tensordot(w, corr[g][train], axes=(0, 0))
              for g in corr}
    Fm0 = fits[HYB_BASE].model_F(Cloo[HYB_BASE], fx.qfr[q0])
    preds["D_hyb26"] = SRi + ts.V_from_F(apply_corr(Fm0, corr_w),
                                         fx.qfr[q0], GS, ALPHA)
    # ftop: interpolated exact channels on the top-n weight columns,
    # LOO-model tail elsewhere
    for n in TOPNS:
        Fx = Fm0.copy()
        Fx[:, TOPS[n]] = np.tensordot(w, lr.Fch[train][:, :, TOPS[n]],
                                      axes=(0, 0))
        preds[f"D_ftop{n}"] = SRi + ts.V_from_F(Fx, fx.qfr[q0], GS, ALPHA)
    # E-style for the top rungs (same LOO model both sides)
    for name in E_RUNGS:
        Vm_tr = np.stack([ts.V_from_F(fits[name].model_F(Cloo[name],
                                                         fx.qfr[qi]),
                                      fx.qfr[qi], GS, ALPHA)
                          for qi in train])
        preds[f"E_{name}"] = np.tensordot(w, Vc[train] - Vm_tr,
                                          axes=(0, 0)) \
            + ts.V_from_F(fits[name].model_F(Cloo[name], fx.qfr[q0]),
                          fx.qfr[q0], GS, ALPHA)
    for lbl, Vp in preds.items():
        Bp = B_tile(x, Vp)
        met = {"B": relF(Bp, B_true), "Bdec": top_decile_rel(Bp, B_true),
               "tile": relF(Vp, V_ref[q0])}
        if lbl in EXC_LABELS:
            ev_p = exciton_evs(fx, D_diag, Hdir, Bp)
            met["exc_meV"] = float(np.max(np.abs(ev_p - ev_true)) * RY2MEV)
        res.setdefault(lbl, {})[q0] = met
    print(f"  q0={q0} done ({time.time()-tq:.0f}s)", flush=True)

# ===========================================================================
# report
# ===========================================================================
NC = {f"D_{n}": fits[n].n_coeff() for n in RUNGS}
NC.update({f"E_{n}": fits[n].n_coeff() for n in E_RUNGS})
NC.update({f"D_svd{rk}": rk for rk in SVD_RANKS})
NC.update({"F_anchor": lr.nG, "C_anchor": 0, "D_hyb26": ngz_mod})
NC.update({f"D_ftop{n}": n for n in TOPNS})
# NC convention: global-fit rungs list TOTAL coeffs/mu (q-independent);
# per-q rungs (F_anchor, hyb26, ftop) list the PER-COARSE-q channel count;
# hyb26/ftop additionally carry the b26p global coefficients.
print(f"\n  ========== {FIXNAME} LR-representation ladder: median / max "
      f"over {fx.nq} LOO targets ==========")
print(f"    {'label':<16s} {'c/mu':>5s} {'fid med':>9s} {'B med':>10s} "
      f"{'B max':>10s} {'Bdec md':>9s} {'exc med':>8s} {'exc max':>8s}")
for lbl in sorted(res):
    rows = res[lbl]
    Bm = [rows[q]["B"] for q in rows]
    em = [rows[q]["exc_meV"] for q in rows if "exc_meV" in rows[q]]
    em_s = (f"{np.median(em):>8.3f} {np.max(em):>8.3f}" if em
            else "      --       --")
    key = lbl.replace("E_", "").replace("D_", "")
    f_s = (f"{fid[key][0]:>9.2e}" if key in fid else
           ("        0" if lbl == "C_anchor" else "       --"))
    print(f"    {lbl:<16s} {NC.get(lbl, -1):>5d} {f_s} "
          f"{np.median(Bm):>10.3e} {np.max(Bm):>10.3e} "
          f"{np.median([rows[q]['Bdec'] for q in rows]):>9.2e} {em_s}")
for lbl in sorted(res):
    for q in sorted(res[lbl]):
        m = res[lbl][q]
        extra = f" exc={m['exc_meV']:.3f}meV" if "exc_meV" in m else ""
        print(f"      [row] {lbl} q0={q}: B={m['B']:.3e} "
              f"Bdec={m['Bdec']:.3e} tile={m['tile']:.3e}{extra}")
for lbl in res:
    qs = sorted(res[lbl])
    NPZ[f"loo__{lbl}__q0"] = np.array(qs)
    for key in ("B", "Bdec", "tile", "exc_meV"):
        NPZ[f"loo__{lbl}__{key}"] = np.array(
            [res[lbl][q].get(key, np.nan) for q in qs], dtype=float)

SUF = "_tik" if TIK else ""
np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         f"lr_basis_ladder_{FIXNAME}{SUF}_results.npz", **NPZ)
print(f"\n[lr_basis_ladder {FIXNAME}{SUF}] ALL DONE in {time.time()-t00:.0f}s")
