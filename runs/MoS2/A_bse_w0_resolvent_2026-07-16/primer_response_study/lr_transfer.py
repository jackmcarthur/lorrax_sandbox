"""lr_transfer — EXPERIMENT 3 (system-consistency angle, grid-transfer
form): the MoS2 3x3 and 6x6 fixtures share ALL 640 centroids (verified
r_mu_fft_idx sets identical), so the global K-ball fit coefficients are
directly comparable PER MU across grids.  Test: fit the b26p (and rich)
representation on the 3x3 data alone, deploy it on the 6x6 fixture —
coefficient distance, tile fidelity against the 6x6 exact LR, and the
full 6x6 LOO B-metric with the 3x3-fitted model as the LR channel.

Honesty note: the 3x3 q-points are a sublattice of the 6x6 grid, but the
two fixtures' zeta FITS are independent (different k-sums, independent
LSQ) — this measures transfer of the REPRESENTATION across datasets, not
data leakage.  Tikhonov gauge on both sides (lr_fiber_source verdict).

Run: JID=<jid> ./proto1_run.sh python3 -u lr_transfer.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import (fix_sphere_wrap, run_gates, sorted_stencil,
                          top_decile_rel)
from tile_prep import TileStudy, B_tile, check_slab_axes
from lr_prep import LRSamples, ChannelFit, spec_poly

ALPHA = 0.30
EPS_TIK = 1e-4
t00 = time.time()


def load(fixname):
    fx = Fixture(fixname)
    fix_sphere_wrap(fx)
    C_q = fx.build_Cq()
    run_gates(fx, C_q, xhx_q=(0,))
    ts = TileStudy(fx, C_q)
    assert check_slab_axes(fx) < 1e-12
    Stik = []
    for q in range(fx.nq):
        lam, R = ts.eig[q]
        g = lam ** 2 / (lam ** 2 + (EPS_TIK * lam[0]) ** 2)
        Stik.append((R * g[None, :]) @ R.conj().T)
    Vc = np.stack([np.conj(Stik[q]) @ ts.V_ref[q] @ np.conj(Stik[q])
                   for q in range(fx.nq)])
    VLRc = np.stack([np.conj(Stik[q])
                     @ fx.make_Vq(fx.ZG[q], q, kind="slab_lr", alpha=ALPHA)
                     @ np.conj(Stik[q]) for q in range(fx.nq)])
    lr = LRSamples(ts, None, ALPHA)
    for q in range(fx.nq):
        zt = Stik[q] @ fx.ZG[q]
        idx = ts.sphere_slot(q, lr.GS)
        zt_ext = np.concatenate([zt, np.zeros((fx.n_mu, 1),
                                              np.complex128)], 1)
        qG = fx.qfr[q][None, :] + lr.GS.T.astype(np.float64)
        ph = np.exp(2j * np.pi * (fx.rmu_frac @ qG.T))
        lr.Fch[q] = ph * zt_ext[:, idx]
    return fx, ts, lr, Vc, VLRc


print("[lr_transfer] loading 3x3 + 6x6 (Tikhonov gauge)")
fx3, ts3, lr3, Vc3, VLRc3 = load("MoS2_3x3")
fx6, ts6, lr6, Vc6, VLRc6 = load("MoS2_6x6")
assert (fx3.nx, fx3.ny, fx3.nz) == (fx6.nx, fx6.ny, fx6.nz)
assert np.array_equal(np.sort(fx3.rmu_flat), np.sort(fx6.rmu_flat))
assert np.max(np.abs(fx3.bvec - fx6.bvec)) < 1e-12
# align mu order across fixtures (same centroid set, order may differ)
p3 = np.argsort(fx3.rmu_flat)
p6 = np.argsort(fx6.rmu_flat)
perm = np.empty(fx6.n_mu, dtype=int)      # perm[mu6] = mu3
perm[p6] = p3
print(f"  [info] centroid orders identical: "
      f"{bool(np.array_equal(fx3.rmu_flat, fx6.rmu_flat))}")

BS = {0: spec_poly(3), 1: spec_poly(2), 2: spec_poly(0), 3: spec_poly(0)}


def bspec(lr):
    return {g: BS[abs(g)] for g in lr.gz_vals if abs(g) in BS}


cf3 = ChannelFit(lr3, bspec(lr3), tag="b26p_3x3")
C3 = cf3.coeffs()
cf6 = ChannelFit(lr6, bspec(lr6), tag="b26p_6x6")
C6 = cf6.coeffs()

# (i) coefficient transfer distance per gz (mu-aligned)
print("\n[T1] b26p coefficient distance 3x3-fit vs 6x6-fit (relF, "
      "mu-aligned):")
for g in sorted(set(C3) & set(C6), key=abs):
    d = relF(C3[g][:, perm], C6[g])
    print(f"  gz={g:+d}: {d:.3f}")
m0_3 = C3[0][0, perm]
m0_6 = C6[0][0]
print(f"  fitted monopole m0(Gz=0): relF {relF(m0_3, m0_6):.3f}, "
      f"corr {np.abs(np.vdot(m0_3, m0_6))/np.linalg.norm(m0_3)/np.linalg.norm(m0_6):.4f}")

# (ii) tile fidelity of the 3x3-fitted model on the 6x6 exact LR
C3on6 = {g: C3[g][:, perm] for g in C3}
rr_x = [relF(ts6.V_from_F(cf6.model_F(C3on6, fx6.qfr[q]), fx6.qfr[q],
                          lr6.GS, ALPHA), VLRc6[q]) for q in range(fx6.nq)]
rr_o = [relF(ts6.V_from_F(cf6.model_F(C6, fx6.qfr[q]), fx6.qfr[q],
             lr6.GS, ALPHA), VLRc6[q]) for q in range(fx6.nq)]
print(f"\n[T2] 6x6 LR-tile fidelity: 3x3-fit model med "
      f"{np.median(rr_x):.3e} max {np.max(rr_x):.3e}  |  6x6 own fit med "
      f"{np.median(rr_o):.3e} max {np.max(rr_o):.3e}")

# (iii) 6x6 LOO B-metric with the transferred model as the LR channel
R7 = sorted_stencil(fx6, [[i, j, 0] for i in range(-2, 4)
                          for j in range(-2, 4)])[:7]
Bx, Bo = [], []
for q0 in range(fx6.nq):
    train = [q for q in range(fx6.nq) if q != q0]
    w = truncR_weights(fx6.qfr[train], fx6.qfr[q0], R7)
    x = fx6.gap_window_pairs(q0, 3, 3)
    B_true = B_tile(x, ts6.V_ref[q0])
    SRi = np.tensordot(w, Vc6[train] - VLRc6[train], axes=(0, 0))
    Vx = SRi + ts6.V_from_F(cf6.model_F(C3on6, fx6.qfr[q0]), fx6.qfr[q0],
                            lr6.GS, ALPHA)
    Bx.append(relF(B_tile(x, Vx), B_true))
    C6l = cf6.coeffs(exclude=q0)
    Vo = SRi + ts6.V_from_F(cf6.model_F(C6l, fx6.qfr[q0]), fx6.qfr[q0],
                            lr6.GS, ALPHA)
    Bo.append(relF(B_tile(x, Vo), B_true))
print(f"\n[T3] 6x6 LOO B: transferred-3x3 model med {np.median(Bx):.3e} "
      f"max {np.max(Bx):.3e}  |  6x6 LOO own med {np.median(Bo):.3e} "
      f"max {np.max(Bo):.3e}")
np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "lr_transfer_results.npz",
         Bx=np.array(Bx), Bo=np.array(Bo), rr_x=np.array(rr_x),
         rr_o=np.array(rr_o))
print(f"\n[lr_transfer] ALL DONE in {time.time()-t00:.0f}s")
