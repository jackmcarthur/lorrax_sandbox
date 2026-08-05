"""stress_alpha_ladder — STRESS TARGET 1 (sec-13 open item, alpha budget):
rerun the global-fit LR ladder at alpha = 0.20 / 0.30 / 0.45 in the Tikhonov
gauge, MoS2 6x6, LOO over all 36 coarse q. Question: is the alpha=0.3 b26p
result robust across the window width, and does alpha=0.45 need its budget
re-allocated toward more G_z channels (sec 13.5 item 4)?

Rungs per alpha (lr_prep machinery, honest LOO coefficients):
  C_anchor   clean-SR stencil + exact target LR (ceiling; breaks LOO on LR)
  F_anchor   clean-SR + channel-interp LR (n_mu x |gset(alpha)| per q)
  D_b26p     fixed sec-13 budget {|Gz|=0:3, 1:2, 2:0, 3:0}   (26 coeffs)
  D_b16p     fixed {2,1,0,0}                                  (16 coeffs)
  D_b26r     re-allocated same-size budget {0:3, 1:1, 2:1, 3:0, 4:0}
             (26 coeffs; in-plane degree traded for deeper G_z coverage —
             the sec-13.5 re-allocation candidate)
  D_bshare   share-follow allocation from THIS alpha's measured v_LR weight
             shares (s>=0.25 -> d3, >=0.06 -> d2, >=0.012 -> d1,
             >=0.0015 -> d0, else drop) — size floats with alpha
  D_rich     gto3 x poly4 (|Gz|<=2) fit ceiling, sigmas scaled with alpha
  E_b26p     consistent subtract/re-add variant of b26p
Metrics: gap-window B (med/max), top-decile, TDA exciton swap for the
anchor + b26p/b26r/bshare rungs. Continuity anchor: alpha=0.30 rows must
reproduce lr_basis_ladder_6x6_tik.log (C 5.402e-3, F 5.848e-3, D_b26p
5.368e-3 med).

Run: JID=<jid> ./proto1_run.sh python3 -u stress_alpha_ladder.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel, build_Hdir, exciton_evs, RY2MEV)
from tile_prep import TileStudy, B_tile, check_slab_axes
from lr_prep import ChannelFit, spec_poly, spec_gto
from stress_prep import build_tik, tik_objects, gate_F_rebuild

ALPHAS = (0.20, 0.30, 0.45)
EPS_TIK = 1e-4
EXC_LABELS = {"C_anchor", "F_anchor", "D_b26p", "D_b26r", "D_bshare"}
t00 = time.time()
NPZ = {}

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12
Stik = build_tik(fx, ts, EPS_TIK)
R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                         for j in range(-2, 4)])[:7]

# per-target exciton state (alpha-independent)
print("[prep] Hdir/exciton caches")
HD = {}
for q0 in range(fx.nq):
    HD[q0] = build_Hdir(fx, q0)


def budget_spec(lr, bydeg):
    out = {}
    for g in lr.gz_vals:
        sp = bydeg.get(abs(g))
        if sp is not None:
            out[g] = sp
    return out


def share_alloc(wabs):
    """Deterministic share-follow degree rule (printed for provenance)."""
    bydeg = {}
    for g, s in sorted(wabs.items()):
        if s >= 0.25:
            bydeg[g] = spec_poly(3)
        elif s >= 0.06:
            bydeg[g] = spec_poly(2)
        elif s >= 0.012:
            bydeg[g] = spec_poly(1)
        elif s >= 0.0015:
            bydeg[g] = spec_poly(0)
    return bydeg


for ALPHA in ALPHAS:
    ta = time.time()
    print(f"\n################ alpha = {ALPHA} ################")
    Vc, VLRc, lr = tik_objects(fx, ts, Stik, ALPHA)
    gate_F_rebuild(ts, fx, lr, VLRc, ALPHA, tag=f"_a{ALPHA}")
    GS = lr.GS
    ws = lr.wshare_gz()
    wabs = {}
    for g, v in ws.items():
        wabs[abs(g)] = wabs.get(abs(g), 0.0) + v
    print(f"  [info] gset({ALPHA}): {lr.nG} G; v_LR weight share per |Gz|: "
          + " ".join(f"{g}:{wabs[g]:.4f}" for g in sorted(wabs)))
    NPZ[f"a{ALPHA}_wshare"] = np.array([wabs.get(g, 0.0) for g in range(10)])

    SIG = [ALPHA / np.sqrt(2.0), ALPHA, ALPHA * np.sqrt(2.0)]
    RUNGS = {
        "b26p": budget_spec(lr, {0: spec_poly(3), 1: spec_poly(2),
                                 2: spec_poly(0), 3: spec_poly(0)}),
        "b16p": budget_spec(lr, {0: spec_poly(2), 1: spec_poly(1),
                                 2: spec_poly(0), 3: spec_poly(0)}),
        "b26r": budget_spec(lr, {0: spec_poly(3), 1: spec_poly(1),
                                 2: spec_poly(1), 3: spec_poly(0),
                                 4: spec_poly(0)}),
        "bshare": budget_spec(lr, share_alloc(wabs)),
        "rich": {g: (spec_gto(4, SIG) if abs(g) <= 2 else spec_gto(2, SIG))
                 for g in lr.gz_vals},
    }
    print("  [info] bshare degrees: "
          + " ".join(f"|Gz|={g}:{len(sp)}c"
                     for g, sp in sorted(share_alloc(wabs).items())))
    fits, Cown, fid = {}, {}, {}
    for name, specs in RUNGS.items():
        cf = ChannelFit(lr, specs, tag=name)
        C = cf.coeffs()
        fits[name], Cown[name] = cf, C
        rr = [relF(ts.V_from_F(cf.model_F(C, fx.qfr[q]), fx.qfr[q], GS,
                               ALPHA), VLRc[q]) for q in range(fx.nq)]
        fid[name] = (float(np.median(rr)), float(np.max(rr)))
        print(f"  [fit] {name:<7s} coeffs/mu {cf.n_coeff():>3d}  fidelity "
              f"med {fid[name][0]:.3e} max {fid[name][1]:.3e}")

    SR = Vc - VLRc
    res = {}
    for q0 in range(fx.nq):
        train = [q for q in range(fx.nq) if q != q0]
        w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
        x = fx.gap_window_pairs(q0, 3, 3)
        B_true = B_tile(x, ts.V_ref[q0])
        D_diag, Hdir = HD[q0]
        ev_true = exciton_evs(fx, D_diag, Hdir, B_true)
        SRi = np.tensordot(w, SR[train], axes=(0, 0))
        preds = {}
        Fi = np.tensordot(w, lr.Fch[train], axes=(0, 0))
        preds["F_anchor"] = SRi + ts.V_from_F(Fi, fx.qfr[q0], GS, ALPHA)
        VLR0 = np.conj(Stik[q0]) @ fx.make_Vq(
            fx.ZG[q0], q0, kind="slab_lr", alpha=ALPHA) @ np.conj(Stik[q0])
        preds["C_anchor"] = SRi + VLR0
        Cloo = {}
        for name in RUNGS:
            Cloo[name] = fits[name].coeffs(exclude=q0)
            preds[f"D_{name}"] = SRi + ts.V_from_F(
                fits[name].model_F(Cloo[name], fx.qfr[q0]), fx.qfr[q0], GS,
                ALPHA)
        # E-style b26p (consistent subtract/re-add)
        Vm_tr = np.stack([ts.V_from_F(fits["b26p"].model_F(Cloo["b26p"],
                                                           fx.qfr[qi]),
                                      fx.qfr[qi], GS, ALPHA)
                          for qi in train])
        preds["E_b26p"] = np.tensordot(w, Vc[train] - Vm_tr, axes=(0, 0)) \
            + ts.V_from_F(fits["b26p"].model_F(Cloo["b26p"], fx.qfr[q0]),
                          fx.qfr[q0], GS, ALPHA)
        for lbl, Vp in preds.items():
            Bp = B_tile(x, Vp)
            met = {"B": relF(Bp, B_true), "Bdec": top_decile_rel(Bp, B_true)}
            if lbl in EXC_LABELS:
                ev_p = exciton_evs(fx, D_diag, Hdir, Bp)
                met["exc_meV"] = float(np.max(np.abs(ev_p - ev_true))
                                       * RY2MEV)
            res.setdefault(lbl, {})[q0] = met
        if q0 % 12 == 0:
            print(f"  q0={q0} done ({time.time()-ta:.0f}s)", flush=True)

    print(f"\n  ==== alpha={ALPHA}: median / max over {fx.nq} LOO targets "
          f"====")
    print(f"    {'label':<10s} {'c/mu':>5s} {'fid med':>9s} {'B med':>10s} "
          f"{'B max':>10s} {'exc med':>8s} {'exc max':>8s}")
    for lbl in sorted(res):
        rows = res[lbl]
        Bm = [rows[q]["B"] for q in rows]
        em = [rows[q]["exc_meV"] for q in rows if "exc_meV" in rows[q]]
        em_s = (f"{np.median(em):>8.3f} {np.max(em):>8.3f}" if em
                else "      --       --")
        key = lbl.replace("E_", "").replace("D_", "")
        f_s = (f"{fid[key][0]:>9.2e}" if key in fid else
               ("        0" if lbl == "C_anchor" else "       --"))
        nc = (fits[key].n_coeff() if key in fits else
              (lr.nG if lbl == "F_anchor" else 0))
        print(f"    {lbl:<10s} {nc:>5d} {f_s} {np.median(Bm):>10.3e} "
              f"{np.max(Bm):>10.3e} {em_s}")
        qs = sorted(rows)
        NPZ[f"a{ALPHA}__{lbl}__q0"] = np.array(qs)
        for key2 in ("B", "Bdec", "exc_meV"):
            NPZ[f"a{ALPHA}__{lbl}__{key2}"] = np.array(
                [rows[q].get(key2, np.nan) for q in qs], dtype=float)
    for lbl in sorted(res):
        for q in sorted(res[lbl]):
            m = res[lbl][q]
            extra = f" exc={m['exc_meV']:.3f}meV" if "exc_meV" in m else ""
            print(f"      [row] a{ALPHA} {lbl} q0={q}: B={m['B']:.3e} "
                  f"Bdec={m['Bdec']:.3e}{extra}")
    print(f"  [alpha={ALPHA}] done in {time.time()-ta:.0f}s")

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "stress_alpha_ladder_results.npz", **NPZ)
print(f"\n[stress_alpha_ladder] ALL DONE in {time.time()-t00:.0f}s")
