"""stress_gamma_edge — STRESS TARGET 3 (edge behavior of the winning
pipeline): Q -> Gamma, zone-boundary crossing, BZ-diagonal anisotropy, and
exact BZ periodicity of the assembly. MoS2 6x6, Tikhonov gauge eps=1e-4,
alpha=0.30, b26p global fit + F channel-interp; tile_path.py conventions
(fixed Gamma probe, full-train stencil — NOT LOO; smoothness metric
d2/range).

Stages:
  G  Gamma approach: Q = t*(1/6,0,0), t log-spaced 1e-6..1 plus linear.
     (i)  G0-EXCLUDED assembled B(t) must approach the stored q=0 truth
          (head-zeroed per sec-12 conventions) smoothly — report
          relF(B(t), B_ref0) and max successive jump;
     (ii) the G0 channel alone: closed-form v_LR(Q)|x.m(Q)|^2 — log-log
          slope p in B_G0 ~ t^p. Physical 2D exchange vanishes ~ t
          (pair-monopole orthogonality); a fitted-monopole residual
          eps_m0 turns that into an ARTIFACT ~ eps_m0^2/t divergence below
          a crossover t*. Measure p(t), t*, and the artifact size at
          t = Dq/10 relative to |B|.
  M  zone-boundary crossing: Q(u) = (u,0,0), u in [0.35,0.65] across
     M = (1/2,0,0); fixed-probe B trajectories + smoothness; PLUS the
     exact-periodicity gate relF(V(u,0,0), V(u-1,0,0)) for u > 1/2
     (stencil weights and model phases are analytic-periodic; the defect
     measures gset tail closure).
  D  BZ-diagonal path Gamma -> K = (1/3,1/3,0), 9 points, nR7 + nR36:
     same observables as tile_path's x-hat run — stencil anisotropy read
     against its logged 3.6e-2 (nR36) / 4.3e-2 (nR7).
  E  edge-class breakdown of the sec-13 LOO rows (loads
     lr_basis_ladder_MoS2_6x6_tik_results.npz): B med/max for D_b26p,
     C_anchor, F_anchor per class Gamma / M / K / edge / interior.

Run: JID=<jid> ./proto1_run.sh python3 -u stress_gamma_edge.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import fix_sphere_wrap, run_gates, sorted_stencil
from tile_prep import TileStudy, B_tile, check_slab_axes
from lr_prep import ChannelFit, spec_poly
from stress_prep import (build_tik, tik_objects, gate_F_rebuild, q_class,
                         smooth_stat)

ALPHA = 0.30
EPS_TIK = 1e-4
t00 = time.time()
NPZ = {}

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
assert check_slab_axes(fx) < 1e-12
Stik = build_tik(fx, ts, EPS_TIK)
Vc, VLRc, lr = tik_objects(fx, ts, Stik, ALPHA)
gate_F_rebuild(ts, fx, lr, VLRc, ALPHA)
GS = lr.GS
B26 = {g: spec_poly({0: 3, 1: 2, 2: 0, 3: 0}[abs(g)]) for g in lr.gz_vals
       if abs(g) <= 3}
cf = ChannelFit(lr, B26, tag="b26p")
C26 = cf.coeffs()          # full-train global fit (path convention)
SR = Vc - VLRc
R36 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                          for j in range(-2, 4)])
train = list(range(fx.nq))
Mp = fx.gap_window_pairs(0, 3, 3)
g0col = int(np.where(np.all(GS == 0, axis=0))[0][0])
print(f"[prep] done ({time.time()-t00:.0f}s); G0 column {g0col}")


def assemble(qt, Rset, model="b26p", drop_g0=False):
    w = truncR_weights(fx.qfr[train], qt, Rset)
    SRi = np.tensordot(w, SR, axes=(0, 0))
    if model == "b26p":
        Fm = cf.model_F(C26, qt)
    else:                       # F channel-interp
        Fm = np.tensordot(w, lr.Fch, axes=(0, 0))
    return SRi + ts.V_from_F(Fm, qt, GS, ALPHA, drop_g0=drop_g0)


def g0_block(qt):
    """Closed-form G0 channel of the b26p LR rebuild at qt (rank-1)."""
    v = ts.v_on_set(qt, GS, kind="slab_lr", alpha=ALPHA)[g0col]
    Fm = cf.model_F(C26, qt)[:, g0col]
    qG = np.asarray(qt)
    zt = np.exp(-2j * np.pi * (fx.rmu_frac @ qG)) * Fm
    return v * np.outer(np.conj(zt), zt), v, zt


# ===========================================================================
# Stage G — Gamma approach
# ===========================================================================
print("\n[stageG] Q -> Gamma along (1/6,0,0)*t")
TS_LOG = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.5,
                   0.75, 1.0])
QDIR = np.array([1.0 / 6.0, 0.0, 0.0])
B_ref0 = B_tile(Mp, ts.V_ref[0])          # stored q=0 truth, head-zeroed
nB0 = np.linalg.norm(B_ref0)
rel_t, g0_t, vs = [], [], []
for t in TS_LOG:
    qt = t * QDIR
    Vng = assemble(qt, R36, "b26p", drop_g0=True)
    Bng = B_tile(Mp, Vng)
    Bg0, v0, zt0 = g0_block(qt)
    xg = Mp @ zt0                          # pair-contracted head amplitude
    bg0 = float(v0 * np.linalg.norm(np.outer(np.conj(xg), xg)))
    rel_t.append(relF(Bng, B_ref0))
    g0_t.append(bg0)
    vs.append(v0)
    print(f"  t={t:<8.1e} relF(B_nog0, B_ref0)={rel_t[-1]:.4e}  "
          f"|B_G0|={bg0:.4e} (/|B0| {bg0/nB0:.2e})  v_LR(G0)={v0:.4e}")
rel_t, g0_t = np.array(rel_t), np.array(g0_t)
NPZ["G__t"], NPZ["G__rel_nog0"], NPZ["G__Bg0"] = TS_LOG, rel_t, g0_t
# t=0 exact-stencil value (weights -> delta at Gamma) as the null anchor
V00 = assemble(np.zeros(3), R36, "b26p", drop_g0=True)
rel00 = relF(B_tile(Mp, V00), B_ref0)
print(f"  t=0 anchor (exact stencil at Gamma): relF={rel00:.4e}")
jumps = np.abs(np.diff(rel_t))
print(f"  [G-i] approach smooth: max successive jump {np.max(jumps):.3e}; "
      f"rel(t<=1e-3) spread {np.ptp(rel_t[TS_LOG <= 1e-3]):.3e}")
# log-log slope of the G0 channel
sl = np.diff(np.log(g0_t)) / np.diff(np.log(TS_LOG))
print("  [G-ii] d ln|B_G0| / d ln t per interval: "
      + " ".join(f"{s:+.2f}" for s in sl))
tstar = None
for i in range(len(sl)):
    if sl[i] > 0:
        tstar = TS_LOG[i]
        break
print(f"  [G-ii] artifact/physical crossover t* ~ {tstar}  "
      f"(below: ~1/t artifact from fitted-monopole residual)")
i10 = int(np.argmin(np.abs(TS_LOG - 0.1)))
print(f"  [G-ii] at t=0.1 (Dq*0.6): |B_G0|/|B0| = {g0_t[i10]/nB0:.3e}")
NPZ["G__rel00"], NPZ["G__slopes"] = np.float64(rel00), sl

# ===========================================================================
# Stage M — zone-boundary crossing + exact periodicity
# ===========================================================================
print("\n[stageM] crossing M = (1/2,0,0): u in [0.35, 0.65]")
US = np.linspace(0.35, 0.65, 13)
ent_ref = None
traj = {"b26p": [], "F": []}
for u in US:
    qt = np.array([u, 0.0, 0.0])
    for mdl in ("b26p", "F"):
        V = assemble(qt, R36, mdl)
        B = B_tile(Mp, V)
        if ent_ref is None:
            iu = np.triu_indices(B.shape[0])
            order_e = np.argsort(np.abs(B[iu]))[::-1][:6]
            ent_ref = (iu[0][order_e], iu[1][order_e])
        traj[mdl].append(np.concatenate(
            [np.linalg.eigvalsh(B)[::-1][:6], B[ent_ref].real]))
for mdl in traj:
    A = np.array(traj[mdl])
    ss = [smooth_stat(A[:, j]) for j in range(A.shape[1])]
    print(f"  [M] {mdl}: d2/range max {max(ss):.3e} med "
          f"{np.median(ss):.3e} over 6 eigs + 6 entries")
    NPZ[f"M__{mdl}__traj"] = A
NPZ["M__u"] = US
# periodicity gate: V(u,0,0) vs V(u-1,0,0), both models
print("  [M] BZ periodicity gate V(Q) vs V(Q-b1):")
for u in (0.55, 0.60, 0.65):
    for mdl in ("b26p", "F"):
        Va = assemble(np.array([u, 0.0, 0.0]), R36, mdl)
        Vb = assemble(np.array([u - 1.0, 0.0, 0.0]), R36, mdl)
        r = relF(Va, Vb)
        rB = relF(B_tile(Mp, Va), B_tile(Mp, Vb))
        print(f"    u={u:.2f} {mdl:<5s}: tile relF {r:.3e}  B relF {rB:.3e}")
        NPZ[f"M__per_{mdl}_u{u:.2f}"] = np.array([r, rB])

# ===========================================================================
# Stage D — BZ-diagonal path Gamma -> K
# ===========================================================================
print("\n[stageD] diagonal path Gamma -> K=(1/3,1/3,0)")
TS9 = np.arange(9) / 8.0
KDIR = np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])
qK = int(np.argmin([np.linalg.norm(fx.qfr[q] - KDIR)
                    for q in range(fx.nq)]))
assert np.allclose(fx.qfr[qK], KDIR), fx.qfr[qK]
truth_K = B_tile(Mp, ts.V_ref[qK])
for sname, Rset in (("nR36", R36), ("nR7", R36[:7])):
    for mdl in ("b26p", "F"):
        rows = []
        for t in TS9:
            V = assemble(t * KDIR, Rset, mdl)
            B = B_tile(Mp, V)
            rows.append(np.concatenate(
                [np.linalg.eigvalsh(B)[::-1][:6], B[ent_ref].real]))
        A = np.array(rows)
        ss_e = [smooth_stat(A[:, j]) for j in range(6)]
        ss_n = [smooth_stat(A[:, j]) for j in range(6, 12)]
        # endpoint anchors: t=0 vs stored q=0; t=1 vs stored K tile
        a1 = relF(B_tile(Mp, assemble(KDIR, Rset, mdl)), truth_K)
        print(f"  [D] {sname} {mdl:<5s}: eigs d2/range max {max(ss_e):.3e} "
              f"med {np.median(ss_e):.3e} | entries max {max(ss_n):.3e} "
              f"med {np.median(ss_n):.3e} | t=1 anchor {a1:.3e}")
        NPZ[f"D__{sname}_{mdl}__traj"] = A
        NPZ[f"D__{sname}_{mdl}__anchor1"] = np.float64(a1)

# ===========================================================================
# Stage E — edge-class breakdown of the sec-13 LOO rows
# ===========================================================================
print("\n[stageE] edge-class breakdown of lr_basis_ladder tik LOO rows")
dat = np.load("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
              "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
              "lr_basis_ladder_MoS2_6x6_tik_results.npz")
classes = [q_class(fx, q) for q in range(fx.nq)]
print("  class census: " + " ".join(
    f"{c}:{classes.count(c)}" for c in
    ("Gamma", "M", "K", "edge", "interior")))
for lbl in ("D_b26p", "C_anchor", "F_anchor"):
    q0s = dat[f"loo__{lbl}__q0"]
    Bs = dat[f"loo__{lbl}__B"]
    out = []
    for c in ("Gamma", "M", "K", "edge", "interior"):
        sel = [i for i, q in enumerate(q0s) if classes[int(q)] == c]
        if sel:
            out.append(f"{c} med {np.median(Bs[sel]):.2e} "
                       f"max {np.max(Bs[sel]):.2e} (n={len(sel)})")
    print(f"  [E] {lbl}: " + " | ".join(out))
NPZ["E__classes"] = np.array([("GMKei".index(c[0])) for c in classes])

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "stress_gamma_edge_results.npz", **NPZ)
print(f"\n[stress_gamma_edge] ALL DONE in {time.time()-t00:.0f}s")
