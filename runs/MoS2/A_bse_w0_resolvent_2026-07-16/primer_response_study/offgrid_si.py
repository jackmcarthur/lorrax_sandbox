"""offgrid_si — Si 4x4x4 NEGATIVE CONTROL for the surviving candidate
(CAMPAIGN_REPORT.md sec 5a item 2): same rank-cut ingredient-interp scheme,
same physical metric, on the Si full-BZ fixture.

EXPECTED: poor off-grid/LOO physical error — sec 3.5 measured the 3D C_R
falloff under-resolved at 4x4x4 (C_q LOO ~33%, off-grid ~72%). A pass here
at MoS2-like accuracy would be an overfitting alarm on the harness; the
scheme "passing the control" means its error TRACKS the falloff resolution.

Fixture: work_old/tmp (n_mu=960, 64 q, stored zeta at ALL 64 q; bare-3D
Coulomb). NOTE the campaign report points at work_sym/tmp/isdf_tensors_792.h5
— that fixture has IBZ-only zeta (8 spheres), unusable for per-q-truth
scoring; work_old is the only full-BZ-zeta Si restart on disk (recorded in
KNOWN_SANDBOX_ERRORS).

Stages:
  0  gates (sphere-derived wrap incl. half-boundary q's, makeVq-vs-disk at
     all 64 q with the bare-3D kernel, XHX, nulls, trig-exactness).
  A  OFF-GRID: 2x2x2 subgrid (8 pts) -> 56 complement, stencils nR=8
     (exact trig on the mod-2 R lattice) and nR=7; rank-cut ladder;
     exciton swap on an 8-target subset.
  B  on-grid LOO (63-train), stencils nR=7 (campaign-matched) and nR=13
     (R=0 + full nearest fcc shell); exciton swap on every 8th target.

Run: JID=<jid> ./proto1_run.sh python3 -u offgrid_si.py
"""
import time
import numpy as np

import offgrid_prep as op
from offgrid_prep import (relF, truncR_weights, fix_sphere_wrap, run_gates,
                          svd_herm, apply_rung, null_solve_chain,
                          null_trig_exact, SiOldFixture)
from offgrid_prep import (ladder_at_target, report, save_res,
                          sorted_stencil)

t00 = time.time()
NPZ = {}

print("[stage0] Si_4x4x4_old (work_old, full-BZ zeta) gates")
fx = SiOldFixture()
print(f"  [stage0] nq={fx.nq} nb={fx.nb} ns={fx.ns} n_mu={fx.n_mu} "
      f"nv={fx.nv} FFT={fx.nx}x{fx.ny}x{fx.nz} zeta_cut={fx.zeta_cutoff} Ry "
      f"kgrid={list(fx.kgrid)}")
changed = fix_sphere_wrap(fx)
C_q = fx.build_Cq()
# q index (i*4+j)*4+l; q=42 = (2,2,2) -> (1/2,1/2,1/2): fully wrap-affected
run_gates(fx, C_q, xhx_q=(0, 42), wfn_check=False)
for q0 in (5, 42):
    print(f"  [null] solve-chain at q0={q0}: {null_solve_chain(fx, C_q, q0)}")

SG = [(i * 4 + j) * 4 + l for i in (0, 2) for j in (0, 2) for l in (0, 2)]
TG = [q for q in range(fx.nq) if q not in SG]
print(f"  [stage0] subgrid q idx {SG}; {len(TG)} off-grid targets")
R2 = sorted_stencil(fx, [[a, b, c] for a in (0, 1) for b in (0, 1)
                         for c in (0, 1)])
print(f"  [null] trig-exactness (subgrid train, nR=8): "
      f"{null_trig_exact(fx, C_q, SG, R2):.3e}")

t_z = time.time()
Zr = np.empty((fx.nq, fx.n_mu * fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    Zr[q] = (C_q[q] @ fx.recon(q)).ravel()
print(f"  [stage0] Z_r all 64 q built ({time.time()-t_z:.0f}s)")

# ===========================================================================
# Stage A — off-grid 2x2x2 -> 56 complement
# ===========================================================================
tA = time.time()
print("\n[stageA] Si OFF-GRID: 2x2x2 subgrid -> 4x4x4 complement")
excA = set(TG[::7])
resA = {}
for q0 in TG:
    tq = time.time()
    ladder_at_target(fx, C_q, Zr, SG, q0, [("nR8", R2), ("nR7", R2[:7])],
                     resA, "offgrid",
                     exciton_stencils=("nR8",) if q0 in excA else (),
                     Ztrue_r=Zr[q0].reshape(fx.n_mu, fx.n_rtot))
    print(f"  q0={q0} done ({time.time()-tq:.0f}s)", flush=True)
report(resA, "Si STAGE A — OFF-GRID (negative control)")
save_res(resA, "si_stageA", NPZ)
print(f"[stageA] done ({time.time()-tA:.0f}s)")

# ===========================================================================
# Stage B — on-grid LOO
# ===========================================================================
tB = time.time()
print("\n[stageB] Si on-grid LOO (63-train)")
R4 = sorted_stencil(fx, [[a, b, c] for a in range(-1, 3)
                         for b in range(-1, 3) for c in range(-1, 3)])
excB = set(range(0, fx.nq, 8))
resB = {}
for q0 in range(fx.nq):
    tr = [q for q in range(fx.nq) if q != q0]
    ladder_at_target(fx, C_q, Zr, tr, q0, [("nR7", R4[:7]), ("nR13", R4[:13])],
                     resB, "loo4x4x4",
                     exciton_stencils=("nR7",) if q0 in excB else (),
                     Ztrue_r=Zr[q0].reshape(fx.n_mu, fx.n_rtot))
    if q0 % 8 == 0:
        print(f"  loo q0={q0} done", flush=True)
report(resB, "Si STAGE B — on-grid LOO")
save_res(resB, "si_stageB", NPZ)
print(f"[stageB] done ({time.time()-tB:.0f}s)")

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "offgrid_si_results.npz", **NPZ)
print(f"\n[offgrid_si] ALL DONE in {time.time()-t00:.0f}s")
