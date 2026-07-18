"""smoke_refit — per-Q ζ-refit ground truth: on-grid null + off-grid sanity.

1. ON-GRID NULL: refit at coarse q1/q4 must reproduce the stored
   V_qmunu tile up to the Galerkin/htransform floor.  Metrics: tile relF
   AND the physical gap-window B (the campaign verdict variable).
2. OFF-GRID: refit at (0,0.25,0)-type points — Hermiticity (construction
   null) + B-distance to the interp prediction (the first true off-grid
   interp-vs-truth number).
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh

import numpy as _np

from bse import vq_interp as vqi

RD = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_exciton_bands_2026-07-17"
FX = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
      "05_lorrax_cohsex_native/tmp")
INP = f"{RD}/exciton_smoke.in"

t0 = time.time()
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
C_q = vqi.build_cq(zx)
with mesh:
    prep = vqi.prepare_coarse(zx, C_q, mesh)
des = vqi.lr_design_blocks(zx, prep)
coeffs = vqi.fit_lr_model(des)

with mesh:
    rst = vqi.refit_prepare(INP, mesh, zx)
    ok = True
    print("\n===== on-grid FORMULATION null (stored m-leg) =====")
    for q0 in (1, 4):
        V_r = vqi.refit_vq(zx, rst, zx["qfr"][q0], mesh, m_leg="stored")
        V_st = zx["Vqmunu"][q0]
        x = vqi.gap_window_pairs(zx, q0)
        rB = vqi.relF(vqi.b_block(x, V_r), vqi.b_block(x, V_st))
        print(f"  q{q0}: tile relF {vqi.relF(V_r, V_st):.3e}  "
              f"B relF {rB:.3e}")
        # flat-bottom systematic of the rank-640 fit (WORKLOG
        # 2026-07-18): stored-vs-refit differ within the ~11%
        # expansion-floor flatness; B systematic 3.7-5.1% here.
        ok &= rB < 8e-2

    print("\n===== on-grid nulls (refit vs stored fit, htransform m-leg) =====")
    for q0 in (1, 4):
        V_r = vqi.refit_vq(zx, rst, zx["qfr"][q0], mesh)
        V_st = zx["Vqmunu"][q0]
        r_tile = vqi.relF(V_r, V_st)
        x = vqi.gap_window_pairs(zx, q0)
        rB = vqi.relF(vqi.b_block(x, V_r), vqi.b_block(x, V_st))
        herm = float(np.linalg.norm(V_r - V_r.conj().T)
                     / np.linalg.norm(V_r))
        print(f"  q{q0} {np.array2string(zx['qfr'][q0], precision=4)}: "
              f"tile relF {r_tile:.3e}  B relF {rB:.3e}  herm {herm:.1e}")
        ok &= rB < 5e-2 and herm < 1e-12

    print("\n===== off-grid: refit truth vs interp prediction =====")
    for qf in (np.array([0.0, -0.25, 0.0]), np.array([-1.0 / 6, -1.0 / 6, 0.0])):
        V_r = vqi.refit_vq(zx, rst, qf, mesh)
        herm = float(np.linalg.norm(V_r - V_r.conj().T) / np.linalg.norm(V_r))
        V_i = vqi.eval_vq_host(zx, prep, des, coeffs, qf)
        # B on the gap window of the NEAREST on-grid q (comparison window
        # convention: both sides share it; the exact window q is a label)
        x = vqi.gap_window_pairs(zx, 0)
        rB = vqi.relF(vqi.b_block(x, V_i), vqi.b_block(x, V_r))
        r_tile = vqi.relF(V_i, V_r)
        print(f"  q={np.array2string(qf, precision=4)}: herm {herm:.1e}  "
              f"interp-vs-refit tile relF {r_tile:.3e}  B relF {rB:.3e}")
        ok &= herm < 1e-12

print(f"\n[smoke_refit] {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
sys.exit(0 if ok else 1)
