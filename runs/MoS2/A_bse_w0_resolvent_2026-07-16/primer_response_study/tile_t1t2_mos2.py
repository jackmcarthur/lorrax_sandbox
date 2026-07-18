"""tile_t1t2_mos2 — driver for the OWNER-SPEC-COMPLIANT arbitrary-Q
prototype (constructions T1/T2, tile_prep.py): n_mu^2 tiles + n_mu-x-small
moment vectors ONLY — no Z_q, no r_tot object, no solve at the target.

Stages
  G   gates: inherited battery (wrap fix, makeVq-vs-disk, XHX) + tile-clean
      equivalence + B_tile==B_from_MG + split exactness + exact-stencil
      nulls (tile and mpconsist chains) + the clean-floor continuity anchor
      (must reproduce the campaign's rankcut-on-TRUE-data floor).
  T2  moment smoothness: adjacent-q diffs (cleaned vs raw vs g0 slot
      vector), R-space falloff vs C_R shells, z-Taylor diagnostic (pure-3D
      multipole vs the exact per-Gz channels), model fidelity vs exact LR.
  L   LOO ladder over all coarse q: variants A/B/C/D/E (tile_prep header),
      physical gap-window B + top-decile + TDA exciton swap for headline
      labels.  Predictions under test (parent bounds sketch): raw tile
      ~5-10%; cleaned tile ~ingredient level (~0.4%); cleaned moments
      smooth where raw g0 is not.

Run: JID=<jid> ./proto1_run.sh python3 -u tile_t1t2_mos2.py MoS2_3x3
     JID=<jid> ./proto1_run.sh python3 -u tile_t1t2_mos2.py MoS2_6x6
"""
import sys
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          B_from_MG, top_decile_rel, build_Hdir,
                          exciton_evs, RY2MEV)
from tile_prep import TileStudy, B_tile, check_slab_axes, EPS_LR

FIXNAME = sys.argv[1] if len(sys.argv) > 1 else "MoS2_3x3"
ALPHAS = [0.20, 0.30, 0.45]
ASTAR = 0.30
RCS = [1e-3, 1e-4, 1e-5]
RCSTAR = 1e-4
t00 = time.time()
NPZ = {}

print(f"[tile] fixture {FIXNAME}; alphas {ALPHAS} (a*={ASTAR}), "
      f"rc {RCS} (rc*={RCSTAR}), eps_LR={EPS_LR}")
fx = Fixture(FIXNAME)
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)

# ===========================================================================
# Stage G — gates
# ===========================================================================
ok = True


def gate(k, v, tol=None):
    global ok
    flag = "" if tol is None else ("  OK" if v <= tol else "  ** FAIL **")
    if tol is not None and v > tol:
        ok = False
    print(f"  [gate] {k:<52s} {v:.3e}{flag}")


print("\n[stageG] tile-machinery gates")
gate("avec_vs_adot", ts.avec_err, 1e-10)
gate("slab_axes_offdiag", check_slab_axes(fx), 1e-12)
for q in (0, min(1, fx.nq - 1)):
    Vzeta = fx.make_Vq(ts.P(q, RCSTAR) @ fx.ZG[q], q, kind="slab")
    PI = np.conj(ts.P(q, RCSTAR))
    gate(f"tileclean_PIVPI_vs_makeVq(Pzeta)_q{q}",
         relF(PI @ ts.V_ref[q] @ PI, Vzeta), 1e-11)
x1 = fx.gap_window_pairs(1, 3, 3)
gate("B_tile_vs_B_from_MG_q1",
     relF(B_tile(x1, ts.V_ref[1]), B_from_MG(fx, x1 @ fx.ZG[1], 1)), 1e-11)
for a in ALPHAS:
    Vsr = fx.make_Vq(fx.ZG[1], 1, kind="slab_sr", alpha=a)
    Vlr = fx.make_Vq(fx.ZG[1], 1, kind="slab_lr", alpha=a)
    gate(f"tile_split_SR+LR==V_q1_alpha{a}", relF(Vsr + Vlr, ts.V_ref[1]),
         1e-12)
    print(f"  [info] alpha={a}: sphere-tail bound "
          f"exp(-cutoff/4a^2)={ts.sphere_tail_bound(a):.2e}; "
          f"LR G-superset {ts.gset(a).shape[1]} G")
gz = ts.gz_list(max(ALPHAS))
print(f"  [info] global Gz channels: {len(gz)} "
      f"(|Gz|<={int(np.max(np.abs(gz)))}); FFT box "
      f"({fx.nx},{fx.ny},{fx.nz}); ranks at rc*: "
      f"{[ts.rank(q, RCSTAR) for q in range(min(4, fx.nq))]}...")

# moments (zdiag on the cleaned rc* set; raw set for the smoothness A/B)
t0 = time.time()
mom_c = ts.moments(RCSTAR, max(ALPHAS), zdiag=True)
mom_raw = ts.moments(None, max(ALPHAS))
print(f"  [info] moments built (cleaned rc* + raw) in {time.time()-t0:.0f}s")

# exact-stencil nulls (full-grid trig weights TO a training point)
kg = fx.kgrid
Rfull = sorted_stencil(fx, [[i, j, 0]
                            for i in range(-(kg[0] // 2), kg[0] - kg[0] // 2)
                            for j in range(-(kg[1] // 2), kg[1] - kg[1] // 2)])
q0n = min(2, fx.nq - 1)
wfull = truncR_weights(fx.qfr, fx.qfr[q0n], Rfull)
Vc4 = ts.Vc(RCSTAR)
gate("null_exact_stencil_cleantile",
     relF(np.tensordot(wfull, Vc4, axes=(0, 0)), Vc4[q0n]), 5e-9)
GSs = ts.gset(ASTAR)
VMP_own_null = np.stack([ts.V_MP(ts.mom_at(mom_c, q), fx.qfr[q], GSs,
                                 ASTAR, 2) for q in range(fx.nq)])
Tnull = Vc4 - VMP_own_null
mi = ts.mom_interp(mom_c, wfull, list(range(fx.nq)))
Vnull = np.tensordot(wfull, Tnull, axes=(0, 0)) \
    + ts.V_MP(mi, fx.qfr[q0n], GSs, ASTAR, 2)
gate("null_exact_stencil_mpconsist", relF(Vnull, Vc4[q0n]), 5e-9)

# F-channel machinery gates: own-rebuild == PI V_LR PI up to the eps_LR tail
Fch = {a: ts.F_channels(RCSTAR, a) for a in ALPHAS}
for a in (ASTAR,):
    r = [relF(ts.V_from_F(Fch[a][q], fx.qfr[q], ts.gset(a), a),
              ts.VLR_exact_c(RCSTAR, a)[q]) for q in range(fx.nq)]
    gate(f"F_own_rebuild_vs_PIVLRPI_a{a}_max", float(np.max(r)), 1e-6)

# clean-floor continuity anchor (== campaign rankcut floor on TRUE data)
fl = {rc: [] for rc in RCS}
for q in range(fx.nq):
    x = fx.gap_window_pairs(q, 3, 3)
    Bt = B_tile(x, ts.V_ref[q])
    for rc in RCS:
        fl[rc].append(relF(B_tile(x, ts.Vc(rc)[q]), Bt))
for rc in RCS:
    print(f"  [anchor] clean-floor B rc={rc:.0e}: med "
          f"{np.median(fl[rc]):.3e} max {np.max(fl[rc]):.3e}"
          + ("   <- must match campaign rankcut-1e-4 TRUE-data floor"
             if rc == RCSTAR else ""))
    NPZ[f"cleanfloor_rc{rc:.0e}"] = np.array(fl[rc])
assert ok, "gate battery FAILED — stop (KNOWN_SANDBOX_ERRORS rule)"

# ===========================================================================
# Stage T2 — moment smoothness / falloff / z-Taylor / model fidelity
# ===========================================================================
print("\n[stageT2] moment-vector diagnostics")
pairs = []
for q in range(fx.nq):
    q2 = fx.k_lookup[tuple((fx.k_int[q] + np.array([1, 0, 0])) % kg)]
    pairs.append((q, q2))


def adjdiff(F):
    out = [np.linalg.norm(F[a] - F[b]) / np.linalg.norm(F[a])
           for a, b in pairs]
    return float(np.median(out)), float(np.max(out))


for tag, mom in (("cleaned", mom_c), ("raw", mom_raw)):
    for f in ("m0", "d", "Th"):
        med, mx = adjdiff(mom[f].reshape(fx.nq, -1))
        print(f"  [T2] adjacent-q rel diff {tag:>7s} {f:<3s}: "
              f"med {med:.3f} max {mx:.3f}")
        NPZ[f"adj_{tag}_{f}"] = np.array([med, mx])
g0slot = fx.ZG[:, :, 0]
med, mx = adjdiff(g0slot)
print(f"  [T2] adjacent-q rel diff raw g0 slot vector : med {med:.3f} "
      f"max {mx:.3f}   <- the winding-afflicted object")
NPZ["adj_g0slot"] = np.array([med, mx])
Fraw = ts.F_channels(None, ASTAR)
for tag, F in (("cleaned", Fch[ASTAR]), ("raw", Fraw)):
    med, mx = adjdiff(F.reshape(fx.nq, -1))
    print(f"  [T2] adjacent-q rel diff {tag:>7s} F-channels (a*): "
          f"med {med:.3f} max {mx:.3f}")
    NPZ[f"adj_{tag}_F"] = np.array([med, mx])

# R-space falloff (vs C_R shells)
Rw = fx.Rw
dR = np.sqrt(np.einsum("ri,ij,rj->r", Rw, fx.adot, Rw))
shells = np.unique(np.round(dR, 6))
EqR = np.exp(2j * np.pi * (fx.qfr @ Rw.T)) / fx.nq


def shell_norms(Fq):
    FR = EqR.T @ Fq.reshape(fx.nq, -1)
    n = np.array([np.linalg.norm(FR[np.isclose(dR, s)]) for s in shells])
    return n / n[0]


CRn = np.array([np.linalg.norm(fx.C_R_full[np.isclose(dR, s)])
                for s in shells])
CRn = CRn / CRn[0]
print(f"  [T2] R shells |R| (bohr): "
      + " ".join(f"{s:.2f}" for s in shells))
print(f"  [T2] C_R shell norms     : " + " ".join(f"{v:.2e}" for v in CRn))
for tag, mom in (("cleaned", mom_c), ("raw", mom_raw)):
    for f in ("m0", "d"):
        sn = shell_norms(mom[f])
        print(f"  [T2] {tag:>7s} {f:<3s} R-falloff : "
              + " ".join(f"{v:.2e}" for v in sn))
        NPZ[f"Rfall_{tag}_{f}"] = sn
snF = shell_norms(Fch[ASTAR].reshape(fx.nq, -1))
print(f"  [T2] cleaned F   R-falloff : " + " ".join(f"{v:.2e}" for v in snF))
NPZ["Rfall_cleaned_F"] = snF
NPZ["Rfall_CR"] = CRn
NPZ["Rfall_shells"] = shells

# z-Taylor diagnostic: pure-3D DQ model error per Gz channel + v_LR weight
b3 = fx.bvec[2, 2]
qmid = fx.qfr[min(1, fx.nq - 1)]
GSd = ts.gset(ASTAR)
vw = ts.v_on_set(qmid, GSd, kind="slab_lr", alpha=ASTAR)
wsum = vw.sum()
print(f"  [T2] pure-3D z-Taylor error per Gz channel (med over mu; cleaned)"
      f" + v_LR weight share at q={np.round(qmid, 3)}, alpha={ASTAR}:")
zrows = []
for i, g in enumerate(gz):
    Kz = b3 * float(g)
    m3 = mom_c["m0"][:, :, list(gz).index(0)] \
        - 1j * Kz * mom_c["dz"] - 0.5 * Kz * Kz * mom_c["Tzz"]
    num = np.abs(mom_c["m0"][:, :, i] - m3)
    den = np.abs(mom_c["m0"][:, :, i]) + 1e-300
    e = float(np.median(num / den))
    wsh = float(vw[GSd[2] == g].sum() / wsum)
    zrows.append((int(g), e, wsh))
    if abs(int(g)) <= 6:
        print(f"      Gz={int(g):+d}: 3D-Taylor err {e:.3f}   "
              f"v_LR weight {wsh:.3f}")
NPZ["ztaylor"] = np.array(zrows)

# model fidelity vs the exact (cleaned) LR tile, per order and alpha
print("  [T2] moment-model fidelity relF(V_MP_own, PI V_LR PI):")
for a in ALPHAS:
    VLRc = ts.VLR_exact_c(RCSTAR, a)
    GS = ts.gset(a)
    for order in (0, 1, 2):
        r = [relF(ts.V_MP(ts.mom_at(mom_c, q), fx.qfr[q], GS, a, order),
                  VLRc[q]) for q in range(fx.nq)]
        print(f"      alpha={a} order={order}: med {np.median(r):.3e} "
              f"max {np.max(r):.3e}")
        NPZ[f"fidelity_a{a}_o{order}"] = np.array(r)

# ===========================================================================
# Stage L — LOO ladder over all coarse q
# ===========================================================================
print(f"\n[stageL] LOO ladder ({fx.nq} targets, stencil nR7)")
R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                         for j in range(-2, 4)])[:7]
V_ref = ts.V_ref
ALPHAS_C = ALPHAS + [0.6]
VLRc4 = {a: ts.VLR_exact_c(RCSTAR, a) for a in ALPHAS_C}
VMPown = {}
for a in ALPHAS:
    GS = ts.gset(a)
    for order in ((0, 1, 2) if a == ASTAR else (2,)):
        VMPown[(a, order)] = np.stack(
            [ts.V_MP(ts.mom_at(mom_c, q), fx.qfr[q], GS, a, order)
             for q in range(fx.nq)])
print(f"  [stageL] coarse caches built ({time.time()-t00:.0f}s)")

EXC_LABELS = {"A_rawtile", f"B_cleantile_rc{RCSTAR:.0e}",
              f"C_cleanSR_LRex_rc{RCSTAR:.0e}_a{ASTAR}",
              f"C_cleanSR_LRex_rc{RCSTAR:.0e}_a0.6",
              f"D_cleanSR_LRmp_rc{RCSTAR:.0e}_a{ASTAR}_o2",
              f"E_mpconsist_rc{RCSTAR:.0e}_a{ASTAR}_o2",
              f"F_chaninterp_rc{RCSTAR:.0e}_a{ASTAR}"}
res = {}
for q0 in range(fx.nq):
    tq = time.time()
    train = [q for q in range(fx.nq) if q != q0]
    w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
    x = fx.gap_window_pairs(q0, 3, 3)
    B_true = B_tile(x, V_ref[q0])
    D_diag, Hdir = build_Hdir(fx, q0)
    ev_true = exciton_evs(fx, D_diag, Hdir, B_true)
    preds = {}
    # A raw tile
    preds["A_rawtile"] = np.tensordot(w, V_ref[train], axes=(0, 0))
    # B cleaned tile (rc ladder)
    for rc in RCS:
        preds[f"B_cleantile_rc{rc:.0e}"] = np.tensordot(
            w, ts.Vc(rc)[train], axes=(0, 0))
    # C/D/F: cleaned SR + LR re-add (exact | moment model | F channels)
    mi = ts.mom_interp(mom_c, w, train)
    for a in ALPHAS_C:
        SR = ts.Vc(RCSTAR)[train] - VLRc4[a][train]
        SRi = np.tensordot(w, SR, axes=(0, 0))
        PI0 = np.conj(ts.P(q0, RCSTAR))
        VLR0 = PI0 @ fx.make_Vq(fx.ZG[q0], q0, kind="slab_lr", alpha=a) @ PI0
        preds[f"C_cleanSR_LRex_rc{RCSTAR:.0e}_a{a}"] = SRi + VLR0
        if a == ASTAR:
            preds[f"D_cleanSR_LRmp_rc{RCSTAR:.0e}_a{a}_o2"] = \
                SRi + ts.V_MP(mi, fx.qfr[q0], ts.gset(a), a, 2)
        if a in ALPHAS:
            Fi = np.tensordot(w, Fch[a][train], axes=(0, 0))
            preds[f"F_chaninterp_rc{RCSTAR:.0e}_a{a}"] = \
                SRi + ts.V_from_F(Fi, fx.qfr[q0], ts.gset(a), a)
    # E: consistent moment-model subtract/re-add (model from rc* moments in
    # every rung — E is exact-at-coarse for ANY model, so the rc ladder
    # isolates the tile-cleaning strength)
    VMP_tgt = {}
    for (a, order), VMP in VMPown.items():
        VMP_tgt[(a, order)] = ts.V_MP(mi, fx.qfr[q0], ts.gset(a), a, order)
        T = ts.Vc(RCSTAR)[train] - VMP[train]
        preds[f"E_mpconsist_rc{RCSTAR:.0e}_a{a}_o{order}"] = \
            np.tensordot(w, T, axes=(0, 0)) + VMP_tgt[(a, order)]
    for rc in (1e-3, 1e-5):
        T = ts.Vc(rc)[train] - VMPown[(ASTAR, 2)][train]
        preds[f"E_mpconsist_rc{rc:.0e}_a{ASTAR}_o2"] = \
            np.tensordot(w, T, axes=(0, 0)) + VMP_tgt[(ASTAR, 2)]
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
print(f"\n  ========== {FIXNAME} LOO ladder: median / max over "
      f"{fx.nq} targets ==========")
print(f"    {'label':<44s} {'B med':>10s} {'B max':>10s} {'Bdec md':>9s} "
      f"{'tile md':>9s} {'exc med':>8s} {'exc max':>8s}")
for lbl in sorted(res):
    rows = res[lbl]
    Bm = [rows[q]["B"] for q in rows]
    em = [rows[q]["exc_meV"] for q in rows if "exc_meV" in rows[q]]
    em_s = (f"{np.median(em):>8.3f} {np.max(em):>8.3f}" if em
            else "      --       --")
    print(f"    {lbl:<44s} {np.median(Bm):>10.3e} {np.max(Bm):>10.3e} "
          f"{np.median([rows[q]['Bdec'] for q in rows]):>9.2e} "
          f"{np.median([rows[q]['tile'] for q in rows]):>9.2e} {em_s}")
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

np.savez(f"/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         f"A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         f"tile_t1t2_{FIXNAME}_results.npz", **NPZ)
print(f"\n[tile_t1t2 {FIXNAME}] ALL DONE in {time.time()-t00:.0f}s")
