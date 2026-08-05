"""stress_negctrl — STRESS TARGET 5 (harness hygiene): one deliberately-
WRONG-CONVENTION run to confirm the harness still catches the 155x-class
wrap trap LOUDLY on the winning pipeline.

Wrong convention: q labels left UNWRAPPED (fx.qfr = raw QE rk in [0,1) —
the exact trap of KNOWN_SANDBOX_ERRORS 2026-07-17 item 1 /
proto1_ladder_wrap_ab, which measured B 4.5e-3 wrapped vs 0.70 unwrapped
= 155x on the ingredient ladder).

Two layers, both must fire:
  1  GATE layer: run_gates on the unwrapped fixture must FAIL (sphere gate
     max|q+G|^2 - cutoff >> 0 and/or makeVq-vs-disk explodes). Caught
     via try/except; a PASS here is a harness regression.
  2  DOWNSTREAM layer (gates deliberately bypassed): the b26p pipeline on
     unwrapped labels, scored against the CORRECT-convention truth, on 6
     wrap-affected LOO targets. Expect B O(0.1-1) vs the 5.4e-3 anchor —
     the loud number for the report.

Run: JID=<jid> ./proto1_run.sh python3 -u stress_negctrl.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel)
from tile_prep import TileStudy, B_tile
from lr_prep import ChannelFit, spec_poly
from stress_prep import build_tik, tik_objects

ALPHA = 0.30
EPS_TIK = 1e-4
t00 = time.time()

# correct-convention fixture: truth side
fxW = Fixture("MoS2_6x6")
changed = fix_sphere_wrap(fxW)
CqW = fxW.build_Cq()
run_gates(fxW, CqW, xhx_q=(0,))
tsW = TileStudy(fxW, CqW)
wrapq = sorted({q for q, *_ in changed}
               | {q for q in range(fxW.nq)
                  if np.max(np.abs(fxW.qfr_raw[q]
                                   - np.round(fxW.qfr_raw[q])
                                   - fxW.qfr[q])) > 1e-12
                  or np.max(np.abs(np.round(fxW.qfr_raw[q]))) > 0})
print(f"[negctrl] wrap-affected q set (raw label != sphere center): "
      f"{wrapq}")

# wrong-convention fixture: raw UNWRAPPED labels
fxU = Fixture("MoS2_6x6")
fxU.qfr = fxU.qfr_raw.copy()
CqU = fxU.build_Cq()          # identical to CqW (phases mod 1) — checked
print(f"[negctrl] C_q identity under relabeling (must be ~0): "
      f"{relF(CqU, CqW):.3e}")

# ---- layer 1: gates must fire ----
print("\n[layer1] run_gates on the UNWRAPPED fixture (expect ** FAIL **):")
try:
    run_gates(fxU, CqU, xhx_q=(0,), wfn_check=False)
    print("  [layer1] *** HARNESS REGRESSION: gates PASSED on unwrapped "
          "labels — record in KNOWN_SANDBOX_ERRORS ***")
    raise SystemExit(1)
except AssertionError as e:
    print(f"  [layer1] OK — gate battery raised: {e}")

# ---- layer 2: bypass gates, run the pipeline, score vs correct truth ----
print("\n[layer2] pipeline on unwrapped labels vs correct truth "
      "(gates bypassed deliberately)")
tsU = TileStudy(fxU, CqU)
targets = [q for q in wrapq][:6] or list(range(6))
res = {}
for tag, fx, ts in (("wrapped", fxW, tsW), ("unwrapped", fxU, tsU)):
    Stik = build_tik(fx, ts, EPS_TIK)
    Vc, VLRc, lr = tik_objects(fx, ts, Stik, ALPHA)
    B26 = {g: spec_poly({0: 3, 1: 2, 2: 0, 3: 0}[abs(g)])
           for g in lr.gz_vals if abs(g) <= 3}
    cf = ChannelFit(lr, B26, tag=tag)
    R7 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                             for j in range(-2, 4)])[:7]
    SR = Vc - VLRc
    out = []
    for q0 in targets:
        train = [q for q in range(fx.nq) if q != q0]
        w = truncR_weights(fx.qfr[train], fx.qfr[q0], R7)
        SRi = np.tensordot(w, SR[train], axes=(0, 0))
        Cl = cf.coeffs(exclude=q0)
        Vp = SRi + ts.V_from_F(cf.model_F(Cl, fx.qfr[q0]), fx.qfr[q0],
                               lr.GS, ALPHA)
        # truth is ALWAYS the correct-convention stored tile + probe
        xT = fxW.gap_window_pairs(q0, 3, 3)
        B_true = B_tile(xT, tsW.V_ref[q0])
        Bp = B_tile(xT, Vp)
        out.append((relF(Bp, B_true), top_decile_rel(Bp, B_true)))
        print(f"  [{tag}] q0={q0}: B={out[-1][0]:.3e} Bdec={out[-1][1]:.3e}")
    res[tag] = np.array(out)

ratio = np.median(res["unwrapped"][:, 0]) / np.median(res["wrapped"][:, 0])
print(f"\n[negctrl verdict] wrap-affected targets {targets}: "
      f"B med wrapped {np.median(res['wrapped'][:, 0]):.3e} vs unwrapped "
      f"{np.median(res['unwrapped'][:, 0]):.3e}  ->  {ratio:.0f}x blowup "
      f"(155x-class trap {'CAUGHT' if ratio > 20 else '** NOT CAUGHT **'})")
np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "stress_negctrl_results.npz",
         wrapped=res["wrapped"], unwrapped=res["unwrapped"],
         targets=np.array(targets))
print(f"[stress_negctrl] ALL DONE in {time.time()-t00:.0f}s")
