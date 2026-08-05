"""Subsampled leave-one-out (LOO) diagnostic of the vq_interp trainer on the
12x12 restart (owner deliverable: fit diagnostics; full 144-q LOO is slow, so
every 6th q = 24 held-out points).

Per held-out coarse q:
  * retrain the b26p LR fit WITHOUT q  (fit_lr_model(des, exclude=q) — the
    per-q normal blocks make this an O(nb^2) re-sum, no re-design)
  * rebuild the SR stencil pinv on the remaining 143 training q
  * predict V(q) with eval_vq_host(train=all-but-q)
  * score against the CLEANED reference tile  Sc V_ref Sc  (the trainer's
    own target object; S from prepare_coarse) — relF on the tile AND on the
    physical gap-window exchange block B = M^H V M (3v x 3c x all-k rows,
    vq_interp.gap_window_pairs — the campaign verdict variable).

Run (never login node):  JID=... ./run_wt.sh <this dir> python3 -u diag_loo.py
"""
import time

import numpy as np

from bse import vq_interp
from bse.bse_io import _find_restart_file
from bse.bse_w_exact import _create_mesh_xy

T0 = time.time()


def tlog(msg):
    print(f"[{time.time()-T0:8.1f}s] {msg}", flush=True)


INPUT = "exciton_bands.in"
mesh_xy = _create_mesh_xy(1, 1)

restart = _find_restart_file(INPUT)
zeta = restart.rsplit("/", 1)[0] + "/zeta_q.h5"
tlog(f"restart: {restart}")
zx = vq_interp.load_zeta_coarse(restart, zeta)
tlog(f"loaded: nq={zx['nq']} nk={zx['nk']} n_mu={zx['n_mu']} "
     f"ngkmax={zx['ngkmax']} kgrid={zx['kgrid']}")
C_q = vq_interp.build_cq(zx)
tlog("build_cq done")
vq_interp.run_gates(zx, C_q)
tlog("gates OK")
prep = vq_interp.prepare_coarse(zx, C_q, mesh_xy)
tlog("prepare_coarse done")
des = vq_interp.lr_design_blocks(zx, prep)
coeffs_full = vq_interp.fit_lr_model(des)
vq_interp.run_nulls(zx, prep, des, coeffs_full)
tlog("nulls OK")

loo_q = list(range(0, zx["nq"], 6))
rows = []
for q in loo_q:
    t1 = time.time()
    train = [j for j in range((zx["nq"])) if j != q]
    coeffs_q = vq_interp.fit_lr_model(des, exclude=q)
    V_pred = vq_interp.eval_vq_host(zx, prep, des, coeffs_q,
                                    zx["qfr"][q], train=train)
    Sc = np.conj(prep["S"][q])
    V_true = Sc @ vq_interp.make_vq(zx, zx["ZG"][q], q) @ Sc
    tile_rel = vq_interp.relF(V_pred, V_true)
    M = vq_interp.gap_window_pairs(zx, q)
    B_pred = vq_interp.b_block(M, V_pred[: zx["n_mu"], : zx["n_mu"]])
    B_true = vq_interp.b_block(M, V_true)
    b_rel = vq_interp.relF(B_pred, B_true)
    rows.append((q, tile_rel, b_rel))
    tlog(f"LOO q={q:3d} qfr=({zx['qfr'][q][0]:+.4f},{zx['qfr'][q][1]:+.4f}) "
         f"tile_rel={tile_rel:.3e}  B_rel={b_rel:.3e}  ({time.time()-t1:.1f}s)")

tiles = np.array([r[1] for r in rows])
bs = np.array([r[2] for r in rows])
print("\n=== LOO summary (every 6th q, {} points) ===".format(len(rows)))
print(f"tile relF : median {np.median(tiles):.3e}  max {tiles.max():.3e} "
      f"(argmax q={rows[int(tiles.argmax())][0]})")
print(f"B relF    : median {np.median(bs):.3e}  max {bs.max():.3e} "
      f"(argmax q={rows[int(bs.argmax())][0]})")

# ---------------------------------------------------------------------------
# Dense on-grid exciton ground truth at the 8 on-grid path points of the
# 15/8/16 Gamma-M-K-Gamma path.  Stored psi + stored W0 + dense eigh
# (n_flat = nk*4*4 = 2304) — no htransform, no Lanczos.  Two exchange
# sources: the STORED disk tile (truth) and the interp tile (isolates the
# V_Q interpolation error at the exciton level).  Driver rows at these iQ
# additionally carry the htransform-cache + Lanczos error (joined in the
# report).  Window nvw=ncw=4 == driver --n-val 4 --n-cond 4.
# ---------------------------------------------------------------------------
RY2EV = 13.6056980659
ONGRID = [(5, (0, 1 / 6, 0)), (10, (0, 1 / 3, 0)), (15, (0, 0.5, 0)),
          (19, (1 / 6, 5 / 12, 0)), (23, (1 / 3, 1 / 3, 0)),
          (27, (0.25, 0.25, 0)), (31, (1 / 6, 1 / 6, 0)),
          (35, (1 / 12, 1 / 12, 0))]
print("\n=== dense on-grid exciton ground truth (path points) ===")
print("iQ   Q                mode    E_1..E_8 (eV)")
for iQ, Q in ONGRID:
    t1 = time.time()
    qt = -np.asarray(Q, dtype=np.float64)
    qt -= np.round(qt)
    qi = vq_interp.kq_index_of_frac(zx, qt)
    V_true = zx["Vqmunu"][qi]
    V_pred = vq_interp.eval_vq_host(zx, prep, des, coeffs_full, zx["qfr"][qi])
    D, Hdir = vq_interp.build_hdir(zx, qi, nvw=4, ncw=4)
    M = vq_interp.gap_window_pairs(zx, qi, nvw=4, ncw=4)
    E_true = vq_interp.exciton_evs(
        zx, D, Hdir, vq_interp.b_block(M, V_true), nstate=8)
    E_pred = vq_interp.exciton_evs(
        zx, D, Hdir, vq_interp.b_block(M, V_pred), nstate=8)
    fmt = lambda E: " ".join(f"{e*RY2EV:.6f}" for e in E)
    print(f"{iQ:3d}  ({Q[0]:.4f},{Q[1]:.4f})  stored  {fmt(E_true)}")
    print(f"{iQ:3d}  ({Q[0]:.4f},{Q[1]:.4f})  interp  {fmt(E_pred)}")
    d = (E_pred - E_true) * RY2EV * 1e3
    print(f"{iQ:3d}  dE(interp-stored) meV: "
          + " ".join(f"{v:8.3f}" for v in d) + f"   ({time.time()-t1:.1f}s)")
print(f"TOTAL {time.time()-T0:.1f}s")
