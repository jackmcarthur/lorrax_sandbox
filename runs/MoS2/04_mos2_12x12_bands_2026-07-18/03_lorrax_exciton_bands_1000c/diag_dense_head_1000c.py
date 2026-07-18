"""Dense on-grid exciton ground truth, WITH the q=0 W-head rank-1 injection
(the piece diag_loo.py's dense section missed — its head-less W0 sat every
state ~470 meV too high; the driver's loader injects
W0[q=0] += (whead[0]/V_cell) conj(g0) g0^H, bse_io.py:588 /
head_correction.apply_q0_head_rank1_sharded).

Per on-grid path point (exact fractions; the driver's path carries
4-decimal node truncation, |dQ| ~ 3e-5 frac = sub-meV, footnoted):
  dense-stored  = dense eigh, STORED disk exchange tile (ground truth)
  dense-interp  = dense eigh, eval_vq_host exchange tile
  driver row    = the production .dat row (interp tile + htransform caches
                  + block-Lanczos) — joined for the spot-check table.
"""
import time

import h5py
import numpy as np

from bse import vq_interp
from bse.bse_io import _find_restart_file
from bse.bse_w_exact import _create_mesh_xy

T0 = time.time()
RY2EV = 13.6056980659
INPUT = "exciton_bands_nw_1000c.in"
DAT = "exciton_bands_1000c.dat"
ONGRID = [(5, (0, 1 / 6, 0)), (10, (0, 1 / 3, 0)), (15, (0, 0.5, 0)),
          (19, (1 / 6, 5 / 12, 0)), (23, (1 / 3, 1 / 3, 0)),
          (27, (0.25, 0.25, 0)), (31, (1 / 6, 1 / 6, 0)),
          (35, (1 / 12, 1 / 12, 0))]

mesh_xy = _create_mesh_xy(1, 1)
restart = _find_restart_file(INPUT)
zx = vq_interp.load_zeta_coarse(restart, restart.rsplit("/", 1)[0]
                                + "/zeta_q.h5")
with h5py.File(restart, "r") as f:
    g0 = np.asarray(f["G0_mu_nu"][:], dtype=np.complex128)
    whead0 = complex(np.asarray(f["whead"][:]).ravel()[0])
V_cell = zx["celvol"]
iq0 = zx["k_lookup"][(0, 0, 0)]
print(f"[{time.time()-T0:.1f}s] head inject: whead0={whead0:.3f}, "
      f"V_cell={V_cell:.2f}, q0 index {iq0}")
zx["W0"] = zx["W0"].copy()
zx["W0"][iq0] += (whead0 / V_cell) * np.conj(g0)[:, None] * g0[None, :]

C_q = vq_interp.build_cq(zx)
vq_interp.run_gates(zx, C_q)
prep = vq_interp.prepare_coarse(zx, C_q, mesh_xy)
des = vq_interp.lr_design_blocks(zx, prep)
coeffs = vq_interp.fit_lr_model(des)
print(f"[{time.time()-T0:.1f}s] trainer ready")

# driver rows (interp mode) from the production .dat
drv = {}
with open(DAT, encoding="utf8") as fh:
    for ln in fh:
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        if t[5] == "interp":
            drv[int(t[0])] = np.array([float(x) for x in t[6:]])

print("\n=== spot-check join (all energies eV; dE in meV vs dense-stored) ===")
for iQ, Q in ONGRID:
    t1 = time.time()
    qt = -np.asarray(Q, dtype=np.float64)
    qt -= np.round(qt)
    qi = vq_interp.kq_index_of_frac(zx, qt)
    V_true = zx["Vqmunu"][qi]
    V_pred = vq_interp.eval_vq_host(zx, prep, des, coeffs, zx["qfr"][qi])
    D, Hdir = vq_interp.build_hdir(zx, qi, nvw=4, ncw=4)
    M = vq_interp.gap_window_pairs(zx, qi, nvw=4, ncw=4)
    E_true = vq_interp.exciton_evs(zx, D, Hdir,
                                   vq_interp.b_block(M, V_true), nstate=8)
    E_pred = vq_interp.exciton_evs(zx, D, Hdir,
                                   vq_interp.b_block(M, V_pred), nstate=8)
    fmt = lambda E: " ".join(f"{e:9.6f}" for e in E)
    print(f"iQ {iQ:3d} Q=({Q[0]:.4f},{Q[1]:.4f})")
    print(f"  dense-stored : {fmt(E_true * RY2EV)}")
    print(f"  dense-interp : {fmt(E_pred * RY2EV)}   "
          f"max|dE| {np.max(np.abs(E_pred-E_true))*RY2EV*1e3:7.3f} meV")
    if iQ in drv:
        d = drv[iQ] - E_true * RY2EV
        print(f"  driver row   : {fmt(drv[iQ])}   "
              f"max|dE| {np.max(np.abs(d))*1e3:7.3f} meV")
    print(f"  ({time.time()-t1:.1f}s)")
print(f"TOTAL {time.time()-T0:.1f}s")
