"""offgrid_path — OWNER-REDESIGNED off-grid test, part (a): smoothness of the
rank-cut ingredient-interp scheme along the path Gamma -> first 6x6 neighbor
along x-hat, on the 6x6 MoS2 dataset alone (no cross-grid-class confound).

q(t) = t * (1/6, 0, 0), t in {0, 1/8, ..., 1} (9 points; endpoints on-grid).
Training = ALL 36 on-grid q (the production use case: fine q between grid
points). Stencils: nR=36 (exact trigonometric interpolant — reproduces the
stored ingredients at the endpoints exactly) and nR=7/13 (truncated-R fits).

Observables (fixed-probe, htransform-free): Btilde(t) = Mp^H V_q(t) Mp with
Mp = the stored-psi gap-window rows AT GAMMA held FIXED along the path, V
contracted on a FIXED G-superset (union of instantaneous spheres along the
path) with the analytic slab v(q(t)+G) — everything smooth in t by
construction except the scheme's own errors. The G=0 slot is EXCLUDED at all
t (the slab head diverges ~1/|q_par| toward Gamma; it is the analytic rank-1
head channel, handled separately in production — its interpolant coefficient
zeta~(t, G=0) is tracked as its own smoothness diagnostic).

Judgment: (i) smoothness — normalized second differences of the top
eigenvalues + largest fixed entries of Btilde(t) along t; (ii) endpoint
anchoring — relF(Btilde_pred, Btilde_true) at t=0,1 vs the stored fits
(same probe, same G0-excluded contraction).

The physical M(t)/swap-H(t) trajectory (htransform wavefunctions at k-q(t))
is offgrid_path_htr.py.

Run: JID=<jid> ./proto1_run.sh python3 -u offgrid_path.py
"""
import time
import numpy as np

import offgrid_prep as op
from offgrid_prep import (Fixture, relF, truncR_weights, fix_sphere_wrap,
                          run_gates, svd_herm, sorted_stencil)

t00 = time.time()
NPZ = {}
TS = np.arange(9) / 8.0
QDIR = np.array([1.0 / 6.0, 0.0, 0.0])
RUNGS = [("rankcut", 1e-3), ("rankcut", 1e-4), ("rankcut", 1e-5), ("raw", 0.0)]
NEIG = 8
NENT = 6

fx = Fixture("MoS2_6x6")
fix_sphere_wrap(fx, verbose=False)
C_q = fx.build_Cq()
run_gates(fx, C_q, xhx_q=(0,))

Zr = np.empty((fx.nq, fx.n_mu * fx.n_rtot), dtype=np.complex128)
for q in range(fx.nq):
    Zr[q] = (C_q[q] @ fx.recon(q)).ravel()
print(f"[path] Z_r built ({time.time()-t00:.0f}s)")

# ---------------------------------------------------------------------------
# fixed G-superset along the path (G=0 excluded), sampled-min criterion
# ---------------------------------------------------------------------------
gr = [np.arange(-(n // 2), n - n // 2) for n in (fx.nx, fx.ny, fx.nz)]
GX, GY, GZ = np.meshgrid(*gr, indexing="ij")
Gall = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 0).astype(np.int64)
tfine = np.linspace(0.0, 1.0, 33)
m = np.full(Gall.shape[1], np.inf)
for t in tfine:
    K = fx.bvec.T @ (t * QDIR[:, None] + Gall.astype(np.float64))
    m = np.minimum(m, np.sum(K * K, axis=0))
keep = (m <= fx.zeta_cutoff + 1e-9) & ~np.all(Gall == 0, axis=0)
GS = Gall[:, keep]                                   # (3, nGS), no G=0
nGS = GS.shape[1]
print(f"[path] union G-superset (G0 excluded): {nGS} G "
      f"(endpoint spheres {int(fx.ngk[0])}, {int(fx.ngk[6])})")


def to_superset(zr_rows, qfrac):
    """rows(r) -> rows(G) on the fixed superset at momentum qfrac."""
    ph = np.exp(-2j * np.pi * (fx.rfrac @ qfrac))
    box = np.fft.fftn((zr_rows * ph[None, :]).reshape(-1, fx.nx, fx.ny, fx.nz),
                      axes=(1, 2, 3), norm="backward").reshape(zr_rows.shape[0],
                                                               fx.n_rtot)
    return box[:, fx.flat_idx(GS)]


def v_slab(qfrac):
    K = fx.bvec.T @ (qfrac[:, None] + GS.astype(np.float64))
    K2 = np.sum(K * K, axis=0)
    zc = np.pi / fx.bvec[2, 2]
    f2d = 1.0 - np.exp(-zc * np.sqrt(K[0]**2 + K[1]**2)) * np.cos(K[2] * zc)
    return 8.0 * np.pi / K2 * f2d / fx.celvol


def g0_coeff(zr_rows, qfrac):
    """zeta~(q, G=0) (the analytic-head channel coefficient), (n_rows,)."""
    ph = np.exp(-2j * np.pi * (fx.rfrac @ qfrac))
    return (zr_rows * ph[None, :]).sum(axis=1)


# fixed probe: stored-psi gap-window rows at Gamma
Mp = fx.gap_window_pairs(0, 3, 3)                     # (81, n_mu)


def Btilde(zt_rows, qfrac):
    v = v_slab(qfrac)
    A = (Mp @ zt_rows) * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


# endpoint truths (stored fits, projected r->superset for identical G set)
truth = {}
for t_end, qi in ((0.0, 0), (1.0, 6)):               # q=6 is (1/6, 0, 0)
    assert np.allclose(fx.qfr[qi], t_end * QDIR), (qi, fx.qfr[qi])
    zt_true = to_superset(fx.recon(qi), t_end * QDIR)
    truth[t_end] = Btilde(zt_true, t_end * QDIR)
print(f"[path] endpoint truths built; probe rows {Mp.shape[0]}")

# fixed entry index set: largest-|B| entries of the t=1 truth (upper tri)
iu = np.triu_indices(Mp.shape[0])
order = np.argsort(np.abs(truth[1.0][iu]))[::-1][:NENT]
ent_idx = (iu[0][order], iu[1][order])

# ---------------------------------------------------------------------------
# the path sweep
# ---------------------------------------------------------------------------
R36 = sorted_stencil(fx, [[i, j, 0] for i in range(-2, 4) for j in range(-2, 4)])
STENCILS = [("nR36", R36), ("nR13", R36[:13]), ("nR7", R36[:7])]
train = list(range(fx.nq))

traj = {}       # traj[(sname, rung)][ti] = dict(eigs, entries, g0n, anchor)
for ti, t in enumerate(TS):
    qt = t * QDIR
    for sname, Rset in STENCILS:
        w = truncR_weights(fx.qfr[train], qt, Rset)
        C0 = np.tensordot(w, C_q, axes=(0, 0))
        Z0 = np.zeros(fx.n_mu * fx.n_rtot, dtype=np.complex128)
        for wi, tr in zip(w, train):
            Z0 += wi * Zr[tr]
        Z0 = Z0.reshape(fx.n_mu, fx.n_rtot)
        Zt0 = to_superset(Z0, qt)
        c0 = g0_coeff(Z0, qt)                        # head-channel RHS coeff
        U, s, Vh = svd_herm(C0)
        UZ = np.conj(U.T) @ Zt0
        Uc0 = np.conj(U.T) @ c0
        for mode, lam in RUNGS:
            sinv = op.rung_sinv(s, mode, lam)
            zt0 = (np.conj(Vh.T) * sinv) @ UZ
            B = Btilde(zt0, qt)
            ev = np.linalg.eigvalsh(B)[::-1][:NEIG]
            zg0 = (np.conj(Vh.T) * sinv) @ Uc0       # zeta~(t, G=0) per mu
            rec = {"eigs": ev, "entries": B[ent_idx].real,
                   "entries_im": B[ent_idx].imag,
                   "g0n": float(np.linalg.norm(zg0)), "zg0": zg0}
            if t in truth:
                rec["anchor"] = relF(B, truth[t])
            traj.setdefault((sname, op.rung_label(mode, lam)), {})[ti] = rec
    print(f"  t={t:.3f} done", flush=True)

# ---------------------------------------------------------------------------
# report: trajectories + smoothness + anchors
# ---------------------------------------------------------------------------
def smooth_stat(y):
    """max |second difference| / (range of y) over the uniform t-grid."""
    y = np.asarray(y, dtype=float)
    d2 = np.abs(y[2:] - 2 * y[1:-1] + y[:-2])
    rng = y.max() - y.min()
    return float(d2.max() / max(rng, 1e-300)), float(rng)


print("\n========== PATH TRAJECTORIES (Gamma -> (1/6,0,0)), fixed probe, "
      "G0-excluded ==========")
for key in sorted(traj):
    rows = traj[key]
    lbl = f"{key[0]}_{key[1]}"
    eig = np.array([rows[ti]["eigs"] for ti in range(len(TS))])   # (9, NEIG)
    ent = np.array([rows[ti]["entries"] for ti in range(len(TS))])
    anch = {TS[ti]: rows[ti].get("anchor") for ti in range(len(TS))
            if "anchor" in rows[ti]}
    # head-channel smoothness: successive-overlap of zg0(t)
    ov = []
    for ti in range(len(TS) - 1):
        a, b = rows[ti]["zg0"], rows[ti + 1]["zg0"]
        ov.append(abs(np.vdot(a, b)) / max(np.linalg.norm(a)
                                           * np.linalg.norm(b), 1e-300))
    s_eig = [smooth_stat(eig[:, j])[0] for j in range(NEIG)]
    s_ent = [smooth_stat(ent[:, j])[0] for j in range(NENT)]
    print(f"\n  ---- {lbl} ----")
    print(f"    anchors relF(B,truth): t=0 {anch.get(0.0, np.nan):.3e}   "
          f"t=1 {anch.get(1.0, np.nan):.3e}")
    print(f"    smooth d2/range: eigs max {max(s_eig):.3e} med "
          f"{np.median(s_eig):.3e} | entries max {max(s_ent):.3e} med "
          f"{np.median(s_ent):.3e}")
    print(f"    head-channel |<zg0(t),zg0(t+1)>| min {min(ov):.6f}")
    for j in range(4):
        print(f"    eig{j}(t): " + " ".join(f"{x: .6e}" for x in eig[:, j]))
    for j in range(3):
        print(f"    ent{j}(t): " + " ".join(f"{x: .6e}" for x in ent[:, j]))
    NPZ[f"path__{lbl}__eigs"] = eig
    NPZ[f"path__{lbl}__entries"] = ent
    NPZ[f"path__{lbl}__anchor_t0"] = np.float64(anch.get(0.0, np.nan))
    NPZ[f"path__{lbl}__anchor_t1"] = np.float64(anch.get(1.0, np.nan))
    NPZ[f"path__{lbl}__g0n"] = np.array([rows[ti]["g0n"]
                                         for ti in range(len(TS))])

NPZ["path__TS"] = TS
NPZ["path__truth_eigs_t0"] = np.linalg.eigvalsh(truth[0.0])[::-1][:NEIG]
NPZ["path__truth_eigs_t1"] = np.linalg.eigvalsh(truth[1.0])[::-1][:NEIG]
print("\n  truth eigs t=0: " + " ".join(
    f"{x: .6e}" for x in NPZ["path__truth_eigs_t0"][:4]))
print("  truth eigs t=1: " + " ".join(
    f"{x: .6e}" for x in NPZ["path__truth_eigs_t1"][:4]))

np.savez("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
         "A_bse_w0_resolvent_2026-07-16/primer_response_study/"
         "offgrid_path_results.npz", **NPZ)
print(f"\n[offgrid_path] ALL DONE in {time.time()-t00:.0f}s")
