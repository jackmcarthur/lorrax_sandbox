"""offgrid_mos2 — DECISIVE test for the campaign's surviving candidate
(CAMPAIGN_REPORT.md sec 5a item 1): off-grid-with-truth on MoS2.

Scheme under test: plain rank-cut ingredient interpolation of C_q / Z_q in
the production BGW-wrapped q-labeling (sphere-derived centers, incl. the
half-boundary extension), one rank-truncated solve in the TARGET's own
frame, judged on the PHYSICAL metric: gap-window B = M^H V_Q M (3v x 3c x
all k, spin-traced) + TDA exciton swap shift. Truth = the stored per-q
full-rank fit (psi_full_y band-span trap respected: no WFN-content truth).

Stages:
  C  3x3 harness continuity (LOO ladder must reproduce the campaign's logged
     rows: rankcut 1e-4 B med 4.699e-3 / max 3.235e-2) + the exciton number
     at rankcut 1e-4 that the campaign left unlogged (sec 6).
  0  6x6 gates + wrap fix + solve-chain nulls + solve/to_sphere commutation.
  A  OFF-GRID: interpolate C/Z from the 9-point 3x3 subgrid of the 6x6
     fixture to the 27 complement q (truth exists), rank-cut ladder
     (1e-4 +- one rung), stencils nR=9 (exact trig) and nR=7 (campaign).
  B  6x6 on-grid LOO (35-train, nR=7/13) for the density trend, plus LOO
     within the 9-point subgrid (bridging row vs the 3x3 fixture's 4.7e-3).

Run: JID=<jid> ./proto1_run.sh python3 -u offgrid_mos2.py
"""
import time
import numpy as np

from offgrid_prep import (Fixture, relF, truncR_weights, fix_sphere_wrap,
                          run_gates, svd_herm, apply_rung,
                          null_solve_chain, null_trig_exact,
                          sorted_stencil, ladder_at_target, report, save_res)

t00 = time.time()
NPZ = {}


# ===========================================================================
# Stage C — 3x3 harness continuity + missing exciton at rankcut 1e-4
# ===========================================================================
print("[stageC] MoS2_3x3 harness continuity (campaign ladder reproduction)")
fx3 = Fixture("MoS2_3x3")
ch3 = fix_sphere_wrap(fx3)
assert len(ch3) == 0, "3x3 has no half-boundary q; wrapfix must be a no-op"
C3 = fx3.build_Cq()
run_gates(fx3, C3, xhx_q=(0, 2))
print(f"  [null] solve-chain at q0=2: {null_solve_chain(fx3, C3, 2)}")
Zr3 = np.empty((fx3.nq, fx3.n_mu * fx3.n_rtot), dtype=np.complex128)
for q in range(fx3.nq):
    Zr3[q] = (C3[q] @ fx3.recon(q)).ravel()
R3 = sorted_stencil(fx3, [[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)])
print(f"  [null] trig-exactness (to a training point, nR=9): "
      f"{null_trig_exact(fx3, C3, list(range(fx3.nq)), R3):.3e}")
# solve/to_sphere commutation (C3 gate, inherited): raw solve both orders
q0c = 2
tr = [q for q in range(fx3.nq) if q != q0c]
wc = truncR_weights(fx3.qfr[tr], fx3.qfr[q0c], R3[:7])
C0c = np.tensordot(wc, C3[tr], axes=(0, 0))
Z0c = (wc @ Zr3[tr]).reshape(fx3.n_mu, fx3.n_rtot)
Uc, sc_, Vhc = svd_herm(C0c)
zA = fx3.to_sphere(apply_rung(Uc, sc_, Vhc, Z0c, "rankcut", 1e-4), q0c)
zB = apply_rung(Uc, sc_, Vhc, fx3.to_sphere(Z0c, q0c), "rankcut", 1e-4)
print(f"  [null] solve/to_sphere commutation (rc 1e-4, q0=2): "
      f"{relF(zB, zA):.3e}")

res3 = {}
for q0 in range(fx3.nq):
    tr = [q for q in range(fx3.nq) if q != q0]
    ladder_at_target(fx3, C3, Zr3, tr, q0, [("nR7", R3[:7])], res3,
                     "loo3x3", exciton_stencils=("nR7",),
                     Ztrue_r=Zr3[q0].reshape(fx3.n_mu, fx3.n_rtot))
del Zr3
report(res3, "STAGE C — 3x3 LOO (continuity: rc 1e-4 must be ~4.699e-3 med "
             "/ 3.235e-2 max; campaign out_proto1_C2_loo_3x3.log)")
save_res(res3, "stageC", NPZ)
c_med = np.median([res3["loo3x3_nR7_rankcut_1e-04"][q]["B"] for q in range(9)])
assert abs(c_med - 4.699e-3) / 4.699e-3 < 0.05, \
    f"harness continuity BROKEN: 3x3 rc1e-4 B med {c_med:.3e} != 4.699e-3"
print(f"  [stageC] continuity CONFIRMED: rc1e-4 B med = {c_med:.4e}")
del fx3, C3, res3
print(f"[stageC] done ({time.time()-t00:.0f}s)")

# ===========================================================================
# Stage 0 — 6x6 fixture, gates + nulls
# ===========================================================================
t0 = time.time()
print("\n[stage0] MoS2_6x6 gates")
fx = Fixture("MoS2_6x6")
changed = fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0, 21))     # q=21 = (1/2,1/2): wrap-affected
for q0 in (13, 21):
    print(f"  [null] solve-chain at q0={q0}: {null_solve_chain(fx, C_q, q0)}")

SG = [6 * i + j for i in (0, 2, 4) for j in (0, 2, 4)]      # 3x3 subgrid
TG = [q for q in range(fx.nq) if q not in SG]               # 27 complement
print(f"  [stage0] subgrid q idx {SG}; {len(TG)} off-grid targets")
R3 = sorted_stencil(fx, [[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)])
print(f"  [null] trig-exactness (subgrid train, nR=9): "
      f"{null_trig_exact(fx, C_q, SG, R3):.3e}")

t_z = time.time()
Zr = np.empty((fx.nq, fx.n_mu * fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    Zr[q] = (C_q[q] @ fx.recon(q)).ravel()
print(f"  [stage0] Z_r all 36 q built ({time.time()-t_z:.0f}s); "
      f"gates done ({time.time()-t0:.0f}s)")

# ===========================================================================
# Stage A — THE DECISIVE OFF-GRID TEST: 9 subgrid -> 27 complement
# ===========================================================================
tA = time.time()
print("\n[stageA] OFF-GRID-WITH-TRUTH: 3x3 subgrid -> 6x6 complement")
resA = {}
for q0 in TG:
    tq = time.time()
    ladder_at_target(fx, C_q, Zr, SG, q0,
                     [("nR9", R3), ("nR7", R3[:7])], resA, "offgrid",
                     exciton_stencils=("nR9", "nR7"),
                     Ztrue_r=Zr[q0].reshape(fx.n_mu, fx.n_rtot))
    print(f"  q0={q0} done ({time.time()-tq:.0f}s)", flush=True)
report(resA, "STAGE A — OFF-GRID (decisive)")
save_res(resA, "stageA", NPZ)
print(f"[stageA] done ({time.time()-tA:.0f}s)")

# ===========================================================================
# Stage B — 6x6 on-grid LOO (density trend) + subgrid-LOO bridging row
# ===========================================================================
tB = time.time()
print("\n[stageB] 6x6 on-grid LOO (35-train) + subgrid LOO (8-train)")
R6 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4) for j in range(-2, 4)])
resB = {}
for q0 in range(fx.nq):
    tr = [q for q in range(fx.nq) if q != q0]
    ladder_at_target(fx, C_q, Zr, tr, q0,
                     [("nR7", R6[:7]), ("nR13", R6[:13])], resB, "loo6x6",
                     exciton_stencils=("nR7",),
                     Ztrue_r=Zr[q0].reshape(fx.n_mu, fx.n_rtot))
    print(f"  loo q0={q0} done", flush=True)
resBS = {}
for q0 in SG:
    tr = [q for q in SG if q != q0]
    ladder_at_target(fx, C_q, Zr, tr, q0, [("nR7", R3[:7])], resBS,
                     "loosub", exciton_stencils=(),
                     Ztrue_r=Zr[q0].reshape(fx.n_mu, fx.n_rtot))
report(resB, "STAGE B — 6x6 on-grid LOO (density trend)")
report(resBS, "STAGE B' — subgrid LOO (bridging row vs 3x3 fixture 4.7e-3)")
save_res(resB, "stageB", NPZ)
save_res(resBS, "stageBsub", NPZ)
print(f"[stageB] done ({time.time()-tB:.0f}s)")

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "offgrid_mos2_results.npz", **NPZ)
print(f"\n[offgrid_mos2] ALL DONE in {time.time()-t00:.0f}s")
