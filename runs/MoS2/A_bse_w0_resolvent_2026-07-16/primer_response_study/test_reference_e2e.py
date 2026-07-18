"""test_reference_e2e — end-to-end smoke test of the reference arbitrary-Q
pipeline (REFERENCE_arbitrary_q_vq.py) on the small MoS2 3x3 fixture.

Exercises the full chain in a few minutes on one node:
    load fixture -> gates -> STAGE 1 prepare_coarse (Tik clean + SR/LR
    split + F samples) -> STAGE 2 global b26p fit -> nulls -> STAGE 3
    eval at every held-out q (honest LOO: target's samples excluded from
    the fit, nR7 SR stencil) -> compare B = M^H V M and the TDA exciton
    swap shift against the stored-fit truth.

PASS THRESHOLDS.  Machine-level nulls are hard gates (inside run_gates /
run_nulls).  The accuracy thresholds below sit ~1.5x above the measured
baseline of the first verified run of this script (2026-07-17, logged in
test_reference_e2e.log: B med 1.409e-2 / max 3.553e-2, exciton med
0.642 / max 2.542 meV).  The 3x3 B median tracks the sec-12.1 3x3
F-scheme rows at alpha=0.3 (1.87e-2 hard-gauge; Tik slightly better) —
the 3x3 grid is ~2x coarser than 6x6, so its interp error is ~3x the
6x6 headline and its excitons ~15x (nk=9 exchange weighting); that is
q-spacing, not a regression (arbitrary_q_bse.md sec 11, subgrid-LOO
bridging row).  A regression that breaks the formalism (wrap trap,
split, phase, fit weighting) moves B by 10-100x and blows through
these immediately.

Run (never on a login node):
    JID=<jid> ./proto1_run.sh python3 -u test_reference_e2e.py
"""
import sys
import time

import numpy as np

from REFERENCE_arbitrary_q_vq import (
    build_cq, build_hdir, b_block, eval_vq, exciton_evs, fit_lr_model,
    gap_window_pairs, load_fixture, lr_design_blocks, make_vq,
    prepare_coarse, relF, run_gates, run_nulls, RY2MEV,
)

THRESH = {
    "B_med": 2.2e-2,      # measured 1.409e-2 (2026-07-17 baseline)
    "B_max": 5.5e-2,      # measured 3.553e-2
    "exc_med": 1.00,      # meV; measured 0.642
    "exc_max": 4.00,      # meV; measured 2.542
}

t0 = time.time()
print("[e2e] MoS2_3x3 smoke test of the reference arbitrary-Q pipeline")
fx = load_fixture("MoS2_3x3")
C_q = build_cq(fx)
run_gates(fx, C_q)                 # hard machine-level gates
prep = prepare_coarse(fx, C_q)     # stage 1
des = lr_design_blocks(fx, prep)   # stage 2 design
coeffs = fit_lr_model(des)
run_nulls(fx, prep, des, coeffs)   # hard machine-level nulls

Bs, excs = [], []
for q0 in range(fx["nq"]):         # stage 3 at every held-out q
    train = [q for q in range(fx["nq"]) if q != q0]
    C_loo = fit_lr_model(des, exclude=q0)
    Vp = eval_vq(fx, prep, des, C_loo, fx["qfr"][q0], train=train)
    x = gap_window_pairs(fx, q0)
    B_true = b_block(x, make_vq(fx, fx["ZG"][q0], q0))
    Bp = b_block(x, Vp)
    Bs.append(relF(Bp, B_true))
    D, Hdir = build_hdir(fx, q0)
    dev = np.abs(exciton_evs(fx, D, Hdir, Bp)
                 - exciton_evs(fx, D, Hdir, B_true))
    excs.append(float(np.max(dev) * RY2MEV))
    print(f"  q0={q0}: B={Bs[-1]:.3e}  exc={excs[-1]:.3f} meV", flush=True)

got = {"B_med": float(np.median(Bs)), "B_max": float(np.max(Bs)),
       "exc_med": float(np.median(excs)), "exc_max": float(np.max(excs))}
ok = True
print("\n  ===== smoke thresholds =====")
for k, thr in THRESH.items():
    good = got[k] <= thr
    ok &= good
    print(f"    {k:<8s} got {got[k]:.3e}  <=  {thr:.3e}  "
          f"{'OK' if good else '** FAIL **'}")
print(f"\n[e2e] {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
sys.exit(0 if ok else 1)
