"""stress_robust — STRESS TARGET 4 (recipe stability): is the sec-13 b26p
pipeline finely tuned or robust? MoS2 6x6, alpha=0.30, LOO over all 36 q,
B metric (excitons on the extreme rungs only). Four independent axes:

  A  cleaning-eps sweep: Tikhonov eps_rel in {1e-3, 1e-4, 1e-5, 1e-6} plus
     the hard cut rc=1e-4 (sec-12 reference). eps -> 0 approaches the RAW
     (uncleaned) gauge — the ridge-zeta small-eps question: does the whole
     pipeline survive under-regularized cleaning, or does the q-fiber
     (sec 13.1: cut-edge rotation) come back to bite B?
  B  fit-ridge sweep: ChannelFit.RIDGE in {1e-8, 1e-11, 1e-14} at
     eps_rel=1e-4 (the normal-equation regularizer, distinct from A).
  C  SR-stencil truncation: nR in {4, 7, 10, 13, 19} at fixed b26p LR.
  D  budget perturbations around {3,2,0,0}: {2,2,0,0} {3,1,0,0} {4,2,0,0}
     {3,3,0,0} {3,2,1,0} {4,3,0,0} {3,2,0,-}(drop |Gz|=3) — each one global
     fit + LOO assembly.

Continuity anchor: (eps 1e-4, ridge 1e-11, nR7, {3,2,0,0}) must reproduce
lr_basis_ladder_6x6_tik.log D_b26p B med 5.368e-3 / max 3.960e-2.

Run: JID=<jid> ./proto1_run.sh python3 -u stress_robust.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel, build_Hdir, exciton_evs, RY2MEV)
from tile_prep import TileStudy, B_tile, check_slab_axes
from lr_prep import ChannelFit, spec_poly
from stress_prep import build_tik, tik_objects, gate_F_rebuild

ALPHA = 0.30
t00 = time.time()
NPZ = {}

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12
Rall = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                           for j in range(-2, 4)])
B26 = {0: spec_poly(3), 1: spec_poly(2), 2: spec_poly(0), 3: spec_poly(0)}

TRUE_B, HD, EV_TRUE, XW = {}, {}, {}, {}
for q0 in range(fx.nq):
    XW[q0] = fx.gap_window_pairs(q0, 3, 3)
    TRUE_B[q0] = B_tile(XW[q0], ts.V_ref[q0])


def budget_spec(lr, bydeg):
    out = {}
    for g in lr.gz_vals:
        sp = bydeg.get(abs(g))
        if sp is not None:
            out[g] = sp
    return out


def loo_scan(tag, Vc, VLRc, lr, cf, nR=7, ridge=None, exciton=False):
    """One full LOO pass of the D-composition; returns (Bmed, Bmax)."""
    if ridge is not None:
        cf.RIDGE = ridge
    GS = lr.GS
    SR = Vc - VLRc
    Rset = Rall[:nR]
    out_B, out_dec, out_exc = [], [], []
    for q0 in range(fx.nq):
        train = [q for q in range(fx.nq) if q != q0]
        w = truncR_weights(fx.qfr[train], fx.qfr[q0], Rset)
        SRi = np.tensordot(w, SR[train], axes=(0, 0))
        Cl = cf.coeffs(exclude=q0)
        Vp = SRi + ts.V_from_F(cf.model_F(Cl, fx.qfr[q0]), fx.qfr[q0], GS,
                               ALPHA)
        Bp = B_tile(XW[q0], Vp)
        out_B.append(relF(Bp, TRUE_B[q0]))
        out_dec.append(top_decile_rel(Bp, TRUE_B[q0]))
        if exciton:
            if q0 not in HD:
                HD[q0] = build_Hdir(fx, q0)
                EV_TRUE[q0] = exciton_evs(fx, *HD[q0], TRUE_B[q0])
            ev_p = exciton_evs(fx, *HD[q0], Bp)
            out_exc.append(float(np.max(np.abs(ev_p - EV_TRUE[q0]))
                                 * RY2MEV))
    e_s = (f"  exc {np.median(out_exc):.3f}/{np.max(out_exc):.3f} meV"
           if out_exc else "")
    print(f"  [{tag:<28s}] B med {np.median(out_B):.3e} max "
          f"{np.max(out_B):.3e}  dec {np.median(out_dec):.2e}{e_s}",
          flush=True)
    NPZ[f"{tag}__B"] = np.array(out_B)
    if out_exc:
        NPZ[f"{tag}__exc"] = np.array(out_exc)
    return float(np.median(out_B)), float(np.max(out_B))


# ===========================================================================
# Axis A — cleaning-eps ladder (incl. hard-cut reference)
# ===========================================================================
print("\n[axisA] cleaning sweep (fit+pipeline rebuilt per gauge)")
for eps in (1e-3, 1e-4, 1e-5, 1e-6):
    Stik = build_tik(fx, ts, eps)
    Vc, VLRc, lr = tik_objects(fx, ts, Stik, ALPHA)
    gate_F_rebuild(ts, fx, lr, VLRc, ALPHA, tag=f"_eps{eps:.0e}")
    cf = ChannelFit(lr, budget_spec(lr, B26), tag=f"eps{eps:.0e}")
    loo_scan(f"A_tik_eps{eps:.0e}", Vc, VLRc, lr, cf,
             exciton=(eps in (1e-4, 1e-6)))
# hard-cut rc=1e-4 (sec-12 gauge) for reference
from lr_prep import LRSamples  # noqa: E402
RC = 1e-4
lr_h = LRSamples(ts, RC, ALPHA)
Vc_h = ts.Vc(RC)
VLR_h = ts.VLR_exact_c(RC, ALPHA)
cf_h = ChannelFit(lr_h, budget_spec(lr_h, B26), tag="hard")
loo_scan("A_hard_rc1e-4", Vc_h, VLR_h, lr_h, cf_h, exciton=True)

# ===========================================================================
# Axes B/C/D share the standard gauge (eps 1e-4)
# ===========================================================================
Stik = build_tik(fx, ts, 1e-4)
Vc, VLRc, lr = tik_objects(fx, ts, Stik, ALPHA)
gate_F_rebuild(ts, fx, lr, VLRc, ALPHA, tag="_std")

print("\n[axisB] fit-ridge sweep (gauge eps=1e-4, nR7, b26p)")
for ridge in (1e-8, 1e-11, 1e-14):
    cf = ChannelFit(lr, budget_spec(lr, B26), tag=f"ridge{ridge:.0e}")
    loo_scan(f"B_ridge{ridge:.0e}", Vc, VLRc, lr, cf, ridge=ridge)

print("\n[axisC] SR-stencil truncation (b26p LR fixed)")
cf26 = ChannelFit(lr, budget_spec(lr, B26), tag="b26p")
for nR in (4, 7, 10, 13, 19):
    loo_scan(f"C_nR{nR}", Vc, VLRc, lr, cf26, nR=nR,
             exciton=(nR in (4, 19)))

print("\n[axisD] budget perturbations around {3,2,0,0}")
PERT = {
    "d_2200": {0: 2, 1: 2, 2: 0, 3: 0},
    "d_3100": {0: 3, 1: 1, 2: 0, 3: 0},
    "d_4200": {0: 4, 1: 2, 2: 0, 3: 0},
    "d_3300": {0: 3, 1: 3, 2: 0, 3: 0},
    "d_3210": {0: 3, 1: 2, 2: 1, 3: 0},
    "d_4300": {0: 4, 1: 3, 2: 0, 3: 0},
    "d_320x": {0: 3, 1: 2, 2: 0},
}
for name, degs in PERT.items():
    spec = budget_spec(lr, {g: spec_poly(d) for g, d in degs.items()})
    cf = ChannelFit(lr, spec, tag=name)
    nc = cf.n_coeff()
    loo_scan(f"D_{name}_{nc}c", Vc, VLRc, lr, cf)

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "stress_robust_results.npz", **NPZ)
print(f"\n[stress_robust] ALL DONE in {time.time()-t00:.0f}s")
