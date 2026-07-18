"""tile_path — off-grid smoothness of the SPEC-COMPLIANT tile schemes along
Gamma -> (1/6,0,0) x-hat on the MoS2 6x6 data (owner path-design of
arbitrary_q_bse.md sec 11.3, transcribed to the n_mu^2 tile level: NO Z, NO
solve at any t — per-target work is a tile stencil + an analytic LR rebuild).

Compositions swept (all rc*=1e-4, per tile_prep conventions):
  F_a<al>   Sum_i w_i(t) [V_c - V_LR_c](q_i) + V_from_F(F_interp(t), q(t))
  E_o<n>    Sum_i w_i(t) [V_c - V_MP_own](q_i) + V_MP(mom_interp(t), q(t))
  B_clean   Sum_i w_i(t) V_c(q_i)                  (no split — control)
  A_raw     Sum_i w_i(t) V_ref(q_i)                (raw control)

Observables (offgrid_path conventions): fixed Gamma gap-window probe
(81 rows), Btilde(t) = conj(Mp) V(t) Mp^T; top-eig + fixed-entry
trajectories with d2/range smoothness; endpoint anchors vs the stored-fit
tiles.  The target-side LR G=0 channel carries the slab |Q| exchange cusp
toward Gamma: swept with keep_g0 True (physical composition; anchor at t=1
vs the stored tile WITH its finite G=0 term) and False (offgrid_path's
G0-excluded convention; anchor vs the stored tile with the rank-1 G=0 dyad
subtracted).  Brunin criterion: smoothness of the interpolated REMAINDER
R(t) = Sum_i w_i [V_c - V_MP_own] at moment orders 0/1/2 — does the
quadrupole smooth the remainder, or kink it?

Run: JID=<jid> ./proto1_run.sh python3 -u tile_path.py
"""
import time
import numpy as np

from proto1_prep import Fixture, relF, truncR_weights
from offgrid_prep import fix_sphere_wrap, run_gates, sorted_stencil
from tile_prep import TileStudy, B_tile

t00 = time.time()
NPZ = {}
TS = np.arange(9) / 8.0
QDIR = np.array([1.0 / 6.0, 0.0, 0.0])
RCSTAR = 1e-4
ALPHAS = [0.30, 0.45]
ASTAR = 0.30
NEIG = 6
NENT = 6

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))
ts = TileStudy(fx, C_q)
mom_c = ts.moments(RCSTAR, ASTAR)
Fch = {a: ts.F_channels(RCSTAR, a) for a in ALPHAS}
VLRc = {a: ts.VLR_exact_c(RCSTAR, a) for a in ALPHAS}
VMPown = {o: np.stack([ts.V_MP(ts.mom_at(mom_c, q), fx.qfr[q],
                               ts.gset(ASTAR), ASTAR, o)
                       for q in range(fx.nq)]) for o in (0, 1, 2)}
print(f"[tile_path] caches built ({time.time()-t00:.0f}s)")

Mp = fx.gap_window_pairs(0, 3, 3)
R36 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4)
                          for j in range(-2, 4)])
STENCILS = [("nR36", R36), ("nR7", R36[:7])]
train = list(range(fx.nq))

# endpoint truths (stored-fit tiles), both G0 conventions
g0q6 = fx.ZG[6][:, 0]
v6 = fx.vq(6)[0]
Vt1_g0dyad = v6[0] * np.outer(np.conj(g0q6), g0q6)
assert np.all(fx.gvec[6][:, 0] == 0)
truth = {(0.0, True): B_tile(Mp, ts.V_ref[0]),
         (0.0, False): B_tile(Mp, ts.V_ref[0]),
         (1.0, True): B_tile(Mp, ts.V_ref[6]),
         (1.0, False): B_tile(Mp, ts.V_ref[6] - Vt1_g0dyad)}
assert np.allclose(fx.qfr[6], QDIR), fx.qfr[6]

# fixed entry set: largest-|B| entries of the t=1 keep-G0 truth (upper tri)
iu = np.triu_indices(Mp.shape[0])
order_e = np.argsort(np.abs(truth[(1.0, True)][iu]))[::-1][:NENT]
ent_idx = (iu[0][order_e], iu[1][order_e])

traj = {}
for ti, t in enumerate(TS):
    qt = t * QDIR
    for sname, Rset in STENCILS:
        w = truncR_weights(fx.qfr[train], qt, Rset)
        preds = {}
        preds["A_raw"] = np.tensordot(w, ts.V_ref, axes=(0, 0))
        preds["B_clean"] = np.tensordot(w, ts.Vc(RCSTAR), axes=(0, 0))
        mi = ts.mom_interp(mom_c, w, train)
        for a in ALPHAS:
            SRi = np.tensordot(w, ts.Vc(RCSTAR) - VLRc[a], axes=(0, 0))
            Fi = np.tensordot(w, Fch[a], axes=(0, 0))
            for keep in (True, False):
                tag = "" if keep else "_nog0"
                preds[f"F_a{a}{tag}"] = SRi + ts.V_from_F(
                    Fi, qt, ts.gset(a), a, drop_g0=not keep)
        for o in (0, 1, 2):
            Ri = np.tensordot(w, ts.Vc(RCSTAR) - VMPown[o], axes=(0, 0))
            preds[f"E_o{o}"] = Ri + ts.V_MP(mi, qt, ts.gset(ASTAR), ASTAR, o)
            preds[f"Rem_o{o}"] = Ri              # Brunin: remainder alone
        for lbl, Vp in preds.items():
            B = B_tile(Mp, Vp)
            ev = np.linalg.eigvalsh(B)[::-1][:NEIG]
            rec = {"eigs": ev, "entries": B[ent_idx].real}
            keep = not lbl.endswith("_nog0")
            if t in (0.0, 1.0) and not lbl.startswith("Rem"):
                rec["anchor"] = relF(B, truth[(t, keep)])
            traj.setdefault((sname, lbl), {})[ti] = rec
    print(f"  t={t:.3f} done ({time.time()-t00:.0f}s)", flush=True)


def smooth_stat(y):
    y = np.asarray(y, dtype=float)
    d2 = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
    return float(d2.max() / max(y.max() - y.min(), 1e-300))


print("\n========== TILE PATH (Gamma -> (1/6,0,0)), fixed Gamma probe "
      "==========")
for key in sorted(traj):
    rows = traj[key]
    lbl = f"{key[0]}_{key[1]}"
    eig = np.array([rows[ti]["eigs"] for ti in range(len(TS))])
    ent = np.array([rows[ti]["entries"] for ti in range(len(TS))])
    anch = {TS[ti]: rows[ti]["anchor"] for ti in range(len(TS))
            if "anchor" in rows[ti]}
    s_eig = [smooth_stat(eig[:, j]) for j in range(NEIG)]
    s_ent = [smooth_stat(ent[:, j]) for j in range(NENT)]
    a_s = (f"anchors t0 {anch.get(0.0, np.nan):.3e} t1 "
           f"{anch.get(1.0, np.nan):.3e}" if anch else "no anchors (remainder)")
    print(f"\n  ---- {lbl} ----")
    print(f"    {a_s}")
    print(f"    smooth d2/range: eigs max {max(s_eig):.3e} med "
          f"{np.median(s_eig):.3e} | entries max {max(s_ent):.3e} med "
          f"{np.median(s_ent):.3e}")
    for j in range(3):
        print(f"    eig{j}(t): " + " ".join(f"{x: .6e}" for x in eig[:, j]))
    NPZ[f"path__{lbl}__eigs"] = eig
    NPZ[f"path__{lbl}__entries"] = ent
    NPZ[f"path__{lbl}__anchor_t0"] = np.float64(anch.get(0.0, np.nan))
    NPZ[f"path__{lbl}__anchor_t1"] = np.float64(anch.get(1.0, np.nan))
NPZ["path__TS"] = TS

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "tile_path_results.npz", **NPZ)
print(f"\n[tile_path] ALL DONE in {time.time()-t00:.0f}s")
